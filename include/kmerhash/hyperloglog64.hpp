/*
 * Copyright 2017 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * hyperloglog64, suitable for estimating the number of hash entries for a hash table.
 *  based on publications
 *  	https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40671.pdf,
 *  	http://algo.inria.fr/flajolet/Publications/FlMa85.pdf,
 *
 *  and implementations
 *  	https://github.com/hideo55/cpp-HyperLogLog and
 *  	https://github.com/dialtr/libcount.
 *  	not yet SIMD enabled like https://github.com/mindis/hll
 *
 * Since this is for a hash table, several modifications were made that are appropriate for hash tables only.
 * 	1. incorporates 64 bit hash values, allowing bypass of large cardinality correction factor (based on hyperloglog++)
 * 	2. bypass bias corrected estimation, and just do linear counting as needed, for 2 reasons below.  note that this can be turned on later for accuracy.
 * 		a. for small cardinality, overestimating cardinality is not significantly costly.
 * 		b. the hash table's load factor provides a buffer that allows real cardinality to be measured for the final resizing.
 *
 * this is structured as a class because we want to be able to merge instances
 *
 * the precision is a function of the number of buckets, 1.04/sqrt(m), where m = 2^precision.
 *
 * TODO:
 * [ ] distributed estimation of global count
 * [ ] distributed estimation of local count - average? NOT post distribution estimate per rank - distributed estimate is not needed then.
 * [ ] distributed estimation of local count from global hash bins - scan input, estimate local.
 * [X] exclude some leading bits (for use e.g. after data is distributed.)
 * [ ] exclude some trailing bits....
 * [ ] batch processing (to allow prefetching later.)  - not needed for now since 2^12 fits in cache well.
 *
 *  Created on: Mar 1, 2017
 *      Author: tpan
 */

#ifndef KMERHASH_HYPERLOGLOG64_HPP_
#define KMERHASH_HYPERLOGLOG64_HPP_

#include <vector>
#include <stdint.h>
#include <iostream> // std::cout
#include <cmath>

#include "kmerhash/mem_utils.hpp"

#ifdef USE_MPI
#include <mxx/comm.hpp>
#include <mxx/collective.hpp>
#include <mxx/reduction.hpp>
#endif


// FROM https://github.com/hideo55/cpp-HyperLogLog.
// modified to conform to the leftmost 1 bit convention, and faster implementation. and to make macro safer.
// zero -> return 65.  else return b+1
#if defined(__has_builtin) && (defined(__GNUC__) || defined(__clang__))

inline uint8_t leftmost_set_bit(uint64_t x) {
	return (x == 0) ? 65 : static_cast<uint8_t>(::__builtin_clzl(x) + 1);
}

inline uint8_t leftmost_set_bit(uint32_t x) {
	return (x == 0) ? 33 : static_cast<uint8_t>(::__builtin_clz(x) + 1);
}

#else

#if defined (_MSC_VER)

inline uint8_t leftmost_set_bit(uint64_t x) {
	if (x == 0) return 65;
    uint64_t b = 64;
    ::_BitScanReverse64(&b, x);
    return static_cast<uint8_t>(b);
}

inline uint8_t leftmost_set_bit(uint32_t x) {
	if (x == 0) return 33;
    uint32_t b = 32;
    ::_BitScanReverse(&b, x);
    return static_cast<uint8_t>(b);
}

#else
// from https://en.wikipedia.org/wiki/Find_first_set, extended to 64 bit
static const uint8_t clz_table_4bit[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
inline uint8_t leftmost_set_bit(uint64_t x) {
  if (x == 0) return 65;
  uint8_t n = 0;
  if ((x & 0xFFFFFFFF00000000ULL) == 0) {n += 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ULL) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ULL) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ULL) == 0) {n +=  4; x <<=  4;}
  n += clz_table_4bit[x >> 60];  // 64 - 4
  return n+1;
}

inline uint8_t leftmost_set_bit(uint32_t x) {
  if (x == 0) return 33;
  uint8_t n = 0;
  if ((x & 0xFFFF0000U) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF000000U) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000U) == 0) {n +=  4; x <<=  4;}
  n += clz_table_4bit[x >> 28];  // 32 - 4
  return n+1;
}

#endif /* defined(_MSC_VER) */

#endif /* defined(__GNUC__) */



//========= general
//   for all hash functions (assume good uniform distribution), the msb are used for register indexing,
//   and lsb are for counting leading 0s.  both have to fit within the maximum number of bits used
//   by hash values
//========= specializations for std::hash.
//   std::hash passes through primitive types, e.g. uint8_t, which only has 8 bits occupied.
//   the high bits of the primitive types are used for register indexing while the low bits are
//   used for leading zero counting.
//   we define unused_msb to be 0 for data type larger than 64, else we use 64-sizeof(T)*8.
//========== non-std::hash
//   the unused_msb is 0.
//========== when same hash values are used for data distribution, followed by estimation (MPI)
//   a rank receives values that have the top bits similar or identical, resulting in large number of
//     empty slots.  instead, we treat the top bits used for distribution as "ignored.", and
//     use next smallest bits as register index.
//   64 bit.  0xIIRRVVVVVV  // MSB: ignored bits II;  high bits: register index RR;  low: values VVVVVV
template <typename T, typename Hash, uint8_t precision = 12U>
class hyperloglog64 {
	  static_assert((precision >= 4U) && (precision <= 18U),
			  "ERROR: precision for hyperloglog should be in [4, 18].");

public:
		using REG_T = uint8_t;
		using HVT = decltype(::std::declval<Hash>().operator()(::std::declval<T>()));
		static constexpr uint8_t hvt_size = sizeof(HVT) * 8U;

		static_assert(hvt_size == 32 || hvt_size == 64, "Only allow 4 or 8 byte hash values");

protected:
		static constexpr uint32_t nRegisters = 0x1U << precision;   // e.g. 0x00000100
		static constexpr uint8_t unused_msb =
				(std::is_same<::std::hash<T>, Hash>::value) ?
				((sizeof(T) >= sizeof(HVT)) ? 0U : (sizeof(HVT) - sizeof(T)) * 8U) :   // if std::hash then via input size.
				0U;
				//(64U - (sizeof(decltype(::std::declval<Hash>().operator()(::std::declval<T>()))) << 3));   // if not, check Hash operator return type.
		// 64 bit.  0xIIRRVVVVVV  // MSB: ignored bits II,  high bits: reg, RR.  low: values, VVVVVV
		static constexpr uint8_t no_ignore_value_bits = hvt_size - precision;  // e.g. 0x00FFFFFF
			// assumes that ignored part has be left shifted out, so no_ignore_value_bits is basically all bits except precision.

		mutable ::std::vector<REG_T> registers;  // stores count of leading zeros

	mutable double amm;
	mutable HVT lzc_mask;   // lowest bits set to 1 to prevent counting into that region.

	mutable uint8_t ignored_msb; // MSB to ignore.
	Hash h;


  inline void internal_update(REG_T* regs, HVT const & no_ignore) {
    // no_ignore has bits 0xRRVVVVVV00, not that the II bits had already been shifted away.
    // extract RR:      0x0000000000RR
        HVT i = no_ignore >> no_ignore_value_bits;   // first precision bits are for register id
        // next count:  want to count 0xVVVVVV1111.
        // compute lzcnt +1 from VVVVVV
        REG_T rank = leftmost_set_bit((no_ignore << precision) | lzc_mask);  // then find leading 1 in remaining
        if (rank > regs[i]) {
            regs[i] = rank;
        }
		//::std::cout << "v=" << std::hex << no_ignore << std::endl;
  }

	inline void internal_update(::std::vector<REG_T> & regs, HVT const & no_ignore) {
		internal_update(regs.data(), no_ignore);
	}

  inline void internal_merge(REG_T * target, const REG_T* src) {
    // precisions identical, so don't need to check number of registers either.

    // iterate over both, merge, and update the zero count.
    for (size_t i = 0; i < nRegisters; ++i) {
      target[i] = ::std::max(target[i], src[i]);
    }
  }


  double internal_estimate(const REG_T* regs) const {
        double est = static_cast<double>(0.0);
        double sum = static_cast<double>(0.0);

        // compute the denominator of the harmonic mean
        for (size_t i = 0; i < nRegisters; i++) {
            sum += static_cast<double>(1.0) / static_cast<double>(1ULL << regs[i]);
        }
        est = amm / sum; // E in the original paper
//        std::cout << static_cast<size_t>(hvt_size) << " bit, mask " << lzc_mask << " ignored " << static_cast<size_t>(ignored_msb)
//            << " no ignore " << static_cast<size_t>(no_ignore_value_bits) << " unused " << static_cast<size_t>(unused_msb) 
////            << " estimate " << est
//            << std::endl;

        if (est <= static_cast<double>(5ULL * (nRegisters >> 1ULL))) {  // 5m/2
          size_t zeros = count_zeros(regs);
          if (zeros > 0ULL) {
//              std::cout << "linear_count: zero: " << zeros << " estimate " << est << " linear count " << linear_count(zeros) << std::endl;
        	  return linear_count(zeros);
          } else {
//        	   std::cout << "linear_count: no zero: " << zeros << " estimate " << est << std::endl;
        	   return est;
          }
        } else if ((hvt_size == 32) && (est > (static_cast<double>(1ULL << 31U) / static_cast<double>(15.0)))) {
          // don't need this because of 64bit.
        	est = - ::std::log(static_cast<double>(1.0) - (est / static_cast<double>(1ULL << 32U))) * static_cast<double>(1ULL << 32U);
//            std::cout << "32bit large estimate: " 
//            << est << std::endl;
            return est;
        } else {
//        	   std::cout << "normal estimate: " << est << std::endl;
        	          return est;
        }
  }


  double internal_estimate(::std::vector<REG_T> const & regs) const {
	  return internal_estimate(regs.data());
  }

  inline uint32_t count_zeros(const REG_T* regs) const {
    uint32_t zeros = 0;
    for (size_t i = 0; i < nRegisters; i++) {
      if (regs[i] == 0) ++zeros;
    }
    return zeros;
  }


  inline uint32_t count_zeros(::std::vector<REG_T> const & regs) const {
	  return count_zeros(regs.data());
  }

  inline double linear_count(uint32_t const & zeros) const {
    return static_cast<double>(nRegisters) *
        ::std::log(static_cast<double>(nRegisters)/ static_cast<double>(zeros));  //natural log
  }


public:

  static constexpr double est_error_rate = static_cast<double>(1.04) / static_cast<double>(0x1U << (precision >> 1U));   // avoid std::sqrt in constexpr.

  	/// constructor.  ignore_leading does not count the leading bits.  ignore_trailing does not count the trailing bits.
  	  /// these are for use when the leading and/or trailing bits are identical in an input set.
	hyperloglog64(uint8_t const & ignore_msb = 0) :
		registers(nRegisters),
		lzc_mask( (~(static_cast<HVT>(0))) >> (hvt_size - precision - ignore_msb - unused_msb)),
		ignored_msb(ignore_msb + unused_msb)
		{

        switch (precision) {
            case 4:
                amm = static_cast<double>(0.673);
                break;
            case 5:
                amm = static_cast<double>(0.697);
                break;
            case 6:
                amm = static_cast<double>(0.709);
                break;
            default:
                amm = static_cast<double>(0.7213) / (static_cast<double>(1.0) + (static_cast<double>(1.079) / static_cast<double>(nRegisters)));
                break;
        }
        amm *= static_cast<double>(0x1ULL << (precision << 1U));  // 2^(2*precision)

//        std::cout << " ignore msb = " << static_cast<size_t>(ignore_msb) <<
//        		" unused_msb = " << static_cast<size_t>(unused_msb) << std::endl;
//        std::cout << "unused msb = " << static_cast<size_t>(64U - (sizeof(decltype(::std::declval<Hash>().operator()(::std::declval<T>()))) << 3)) << std::endl;
//        std::cout << "unused msb2 = " << static_cast<size_t>(64U - (sizeof(typename std::result_of<Hash&(const T &)>::type) << 3)) << std::endl;
//        std::cout << "unused msb3 = " << static_cast<size_t>(64U - (sizeof(typename std::result_of<Hash(const T &)>::type) << 3)) << std::endl;
//        std::cout << "unused msb3 = " << static_cast<size_t>(64U - (sizeof(typename std::result_of<decltype(&Hash::operator())(Hash, const T &)>::type) << 3)) << std::endl;

	}

	hyperloglog64(hyperloglog64 const & other) :
		registers(other.registers), amm(other.amm), lzc_mask(other.lzc_mask), ignored_msb(other.ignored_msb) {
	}

	hyperloglog64(hyperloglog64 && other) :
		registers(std::move(other.registers)), amm(other.amm), lzc_mask(other.lzc_mask), ignored_msb(other.ignored_msb)  {
	}

	hyperloglog64& operator=(hyperloglog64 const & other) {
		registers = other.registers;
		amm = other.amm;
		ignored_msb = other.ignored_msb;
		lzc_mask = other.lzc_mask;

		return *this;
	}

	hyperloglog64& operator=(hyperloglog64 && other) {
		registers.swap(other.registers);
		amm = other.amm;
		//registers.swap(other.registers);
		ignored_msb = other.ignored_msb;
		lzc_mask = other.lzc_mask;

		return *this;
	}


	void swap(hyperloglog64 && other) {
		std::swap(registers, other.registers);
		std::swap(amm, other.amm);
		std::swap(ignored_msb, other.ignored_msb);
		std::swap(lzc_mask, other.lzc_mask);
	}

	inline void set_ignored_msb(uint8_t const & ignore_msb) {
		this->ignored_msb = ignore_msb + unused_msb;
	}
	inline uint8_t get_ignored_msb() {
		return this->ignored_msb - unused_msb;
	}
	hyperloglog64 make_empty_copy() {
		return hyperloglog64(this->ignored_msb - unused_msb);
	}
	hyperloglog64 make_copy() {
		return hyperloglog64(*this);
	}


	inline HVT update(T const & val) {
    HVT hval = h(val);
      internal_update(this->registers, hval << ignored_msb);
      return hval;
	}

	inline void update_via_hashval(HVT const & hval) {
	//	::std::cout << ::std::hex << static_cast<uint64_t>(hval) << ", unused msb " << static_cast<size_t>(unused_msb) << " ignored msb  " << static_cast<size_t>(ignored_msb) << " val " <<  (static_cast<uint64_t>(hval) << ignored_msb) << std::endl;
      internal_update(this->registers, hval << ignored_msb);
	}

  //BATCH INTERFACE

  // ========== update from value, compute hash on the fly.

	// if batch mode not present.
  template <typename H = Hash, typename TT>
  inline auto update_impl(TT const * vals, size_t const & count, long)
  -> decltype(::std::declval<H>()(::std::declval<TT>()), void()) {
//	  printf("UPDATE_IMPL:  discard hash val, singleton\n");
    for (size_t i = 0; i < count; ++i) {
      internal_update(this->registers, h(vals[i]) << ignored_msb);
    }
  }

  // int as last parameter preferred.  batch mode avaialble
  template <typename H = Hash, typename TT>
  inline auto update_impl(TT const * vals, size_t const & count, int)
  -> decltype(::std::declval<H>()(::std::declval<TT const *>(), ::std::declval<size_t>(), ::std::declval<HVT*>()), void()) {
//	  printf("UPDATE_IMPL:  discard hash val, vector\n");
    assert(((h.batch_size & (h.batch_size - 1)) == 0) && "batch size should be power of 2.");

    size_t max = count - (count & (h.batch_size - 1) );
    size_t i = 0, j = 0;

    HVT* buf = ::utils::mem::aligned_alloc<HVT>(h.batch_size);  // 64 byte alignment.

    for (; i < max; i += h.batch_size ) {
      h(vals + i, h.batch_size, buf);

      for (j = 0; j < h.batch_size; ++j) {
        internal_update(this->registers, buf[j] << ignored_msb);
      }
    }
    // last part, do linearly.
    if (count > max) {
		h(vals + i, count - max, buf);
		for (j = 0; j < (count - max); ++j) {
		  internal_update(this->registers, buf[j] << ignored_msb);
		}
    }
    free(buf);
  }

  template <typename TT>
  inline void update(TT const * vals, size_t const & count) {
    update_impl(vals, count, 0);   // prefer integer as 0 defaults to int.
  }

  // ======== update from value, saving the hash values for later use also.

	// if batch mode not present.
  template <typename H = Hash, typename TT>
  inline auto update_impl(TT const * vals, size_t const & count, HVT* hvals, long)
  -> decltype(::std::declval<H>()(::std::declval<TT>()), void()) {
//	  printf("UPDATE_IMPL:  return hash val, singleton\n");
    HVT hv;
    for (size_t i = 0; i < count; ++i) {
      hv = h(vals[i]); 
      internal_update(this->registers, hv << ignored_msb);
      hvals[i] = hv;
    }
  }

  // int as last parameter preferred.  batch mode avaialble
  template <typename H = Hash, typename TT>
  inline auto update_impl(TT const * vals, size_t const & count, HVT* hvals, int)
  -> decltype(::std::declval<H>()(::std::declval<TT const *>(), ::std::declval<size_t>(), ::std::declval<HVT*>()), void()) {
//	  printf("UPDATE_IMPL:  return hash val, vector\n");
    assert(((h.batch_size & (h.batch_size - 1)) == 0) && "batch size should be power of 2.");

    size_t max = count - (count & (h.batch_size - 1) );
    size_t i = 0, j = 0, jmax;

    for (; i < max; i += h.batch_size ) {
      h(vals + i, h.batch_size, hvals + i);

      for (j = i, jmax = i + h.batch_size; j < jmax; ++j) {
        internal_update(this->registers, hvals[j] << ignored_msb);
      }
    }
    // last part, do linearly.
    h(vals + i, count - max, hvals + i);
    for (; i < count; ++i) {
      internal_update(this->registers, hvals[i] << ignored_msb);
    }
  }


  template <typename TT>
  inline void update(TT const * vals, size_t const & count, HVT* hvals) {
    update_impl(vals, count, hvals, 0);   // prefer integer as 0 defaults to int.
  }


  // =============== update when we already have hash values.
  inline void update_via_hashval(HVT const * hashes, size_t const & count) {
//  	std::cout << "h0: " << hashes[0] << std::endl;
    for (size_t i = 0; i < count; ++i) {
      internal_update(this->registers, hashes[i] << ignored_msb);
    }
  }



	double estimate() const {
	  return internal_estimate(this->registers);
	}

	void merge(hyperloglog64 const & other) {
	  internal_merge(this->registers.data(), other.registers.data());
	}

	void clear() {
		registers.assign(nRegisters, static_cast<REG_T>(0));
	}


#ifdef USE_MPI
	// distributed merge, for estimating globally
	::std::vector<REG_T> merge_distributed(::mxx::comm const & comm) const {
	  return ::mxx::allreduce(registers, ::mxx::max<REG_T>(), comm);
	}


	// global estimate for the current object
  inline double estimate_global(::mxx::comm const & comm) const {
    return internal_estimate(merge_distributed(comm));
  }

  /// estimate per node, assuming that the hash values and original input values are uniformly randomly distributed (good balance)
  inline double estimate_average_per_rank(::mxx::comm const & comm) const {
    return estimate_global(comm) / static_cast<double>(comm.size());
  }


  // global estimate for new data.
  template <typename SIZE>
  inline double estimate_local_by_hashval(HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                           ::mxx::comm const & comm) {
    // create the registers
    ::std::vector<REG_T> regs(nRegisters, static_cast<REG_T>(0));

    // perform updates on the registers.
    for (; first != last; ++first) {
      internal_update(regs, *first << ignored_msb);
    }

    // now merge distributed and estimate
    return internal_estimate(regs);
  }

  // global estimate for new data.
  template <typename SIZE>
  inline double estimate_global_by_hashval(HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                           ::mxx::comm const & comm) {
    // create the registers
    ::std::vector<REG_T> regs(nRegisters, static_cast<REG_T>(0));

    // perform updates on the registers.
    for (; first != last; ++first) {
      internal_update(regs, *first << ignored_msb);
    }

    // now merge distributed and estimate
    return internal_estimate(::mxx::allreduce(regs, ::mxx::max<REG_T>(), comm));
  }

  /// estimate per node, assuming that the hash values and original input values are
  // uniformly randomly distributed (good balance)
  // NOTE: inexact estimate.  does not require permutation.  PREFERRED.   uses Allreduce, with log(p) iterations (likely).
  template <typename SIZE>
  inline double estimate_average_per_rank_by_hashval(HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                          ::mxx::comm const & comm) {
    return estimate_global_by_hashval(first, last, send_counts, comm) / static_cast<double>(comm.size());
  }




  // distributed estimate.  exact solution, performed bucket by bucket.
  // input: source hash value array, or input array, and bucket send counts.
  // internal:  2^precision buckets
  // final: reduced 2^precision buckets
  // output: estimate.
  // NOTE: exact, but requires permuting the hash values.  also, P iterations of merge, so slower for high P.
  // assumes accumulating has the correct data.
  template <typename SIZE>
  void update_per_rank_by_hashval_internal(REG_T* accumulating, HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                      ::mxx::comm const & comm) {

    size_t input_size = ::std::distance(first, last);

    assert((static_cast<int>(send_counts.size()) == comm.size()) && "send_count size not same as _comm size.");

    // make sure tehre is something to do.
    bool empty = input_size == 0;
    empty = mxx::all_of(empty);

    if (empty) {
      return;
    }

    // if there is comm size is 1.
    if (comm.size() == 1) {
      // perform updates on the registers.
      for (; first != last; ++first) {
        internal_update(accumulating, *first << ignored_msb);
      }
      return;
    }

    // compute the displacements.
    std::vector<size_t> send_displs;
    send_displs.reserve(send_counts.size() + 1);
    send_displs.emplace_back(0ULL);
    for (size_t i = 0; i < send_counts.size(); ++i) {
      send_displs.emplace_back(send_displs.back() + send_counts[i]);
    }

    // setup the temporary storage.  double buffered.
    // allocate recv
    REG_T* buffers = ::utils::mem::aligned_alloc<REG_T>(nRegisters * 4ULL);
    REG_T* updating = buffers;
    REG_T* sending = buffers + nRegisters;
    REG_T* recving = buffers + 2UL * nRegisters;
    REG_T* recved = buffers + 3UL * nRegisters;


    // loop and process each processor's assignment.  use isend and irecv.
    const int ialltoall_reduce_tag = 1973;

    // local (step 0) - skip the send recv, just directly process.
    size_t comm_size = comm.size();
    size_t comm_rank = comm.rank();
    size_t curr_peer, next_peer, prev_peer = comm_rank;

    bool is_pow2 = ( comm_size & (comm_size-1)) == 0;

    //===  for prev_peer:  first compute self estimate.
    auto max = first + send_displs[prev_peer + 1];
    memset(recved, 0, nRegisters * sizeof(REG_T));
    for (auto it = first + send_displs[prev_peer]; it != max; ++it) {
      internal_update(recved, *it << ignored_msb);
    }

    //=== for curr_peer:
    if ( is_pow2 )  {  // power of 2
      curr_peer = comm_rank ^ 1;
    } else {
      curr_peer = (comm_rank + 1) % comm_size;
    }
    // compute the array
    max = first + send_displs[curr_peer + 1];
    memset(sending, 0, nRegisters * sizeof(REG_T));
    for (auto it = first + send_displs[curr_peer]; it != max; ++it) {
      internal_update(sending,  *it << ignored_msb);
    }

    size_t step;

    mxx::datatype dt = mxx::get_datatype<REG_T>();
    std::vector<MPI_Request> reqs(2);
    int completed = false;

    for (step = 2; step < comm_size; ++step) {
      //====  first setup send and recv.

      // send and recv next.  post recv first.
      MPI_Irecv(recving, nRegisters, dt.type(),
                curr_peer, ialltoall_reduce_tag, comm, &reqs[0] );
      MPI_Isend(sending, nRegisters, dt.type(),
                curr_peer, ialltoall_reduce_tag, comm, &reqs[1] );

      // try using test to kick start the send?
      MPI_Test(&reqs[1], &completed, MPI_STATUS_IGNORE);
      //std::cout << "send from " << comm_rank << " to " << curr_peer << " is " << (completed ? "" : " not ") << " complete "<< std::endl;


      //=== compute the local update.
      // target rank
      if ( is_pow2 )  {  // power of 2
        next_peer = comm_rank ^ step;
      } else {
        next_peer = (comm_rank + step) % comm_size;
      }
      memset(updating, 0, nRegisters * sizeof(REG_T));
      max = first + send_displs[next_peer + 1];
      for (auto it = first + send_displs[next_peer]; it != max; ++it) {
        internal_update(updating,  *it << ignored_msb);
      }

      //=== and accumulate
      internal_merge(accumulating, recved);
      //std::cout << "  per rank estimate for step " << (step - 2) << " peer " << prev_peer << ": " << internal_estimate(accumulating) << std::endl;

      // wait for both to complete
      MPI_Waitall(2, reqs.data(), MPI_STATUSES_IGNORE);

      // now swap.
      std::swap(updating, sending);
      std::swap(recving, recved);

      prev_peer = curr_peer;
      curr_peer = next_peer;

    }

    //=== process curr_peer
    // send and recv next.  post recv first.
    MPI_Irecv(recving, nRegisters, dt.type(),
              curr_peer, ialltoall_reduce_tag, comm, &reqs[0] );
    MPI_Isend(sending, nRegisters, dt.type(),
              curr_peer, ialltoall_reduce_tag, comm, &reqs[1] );

    // try using test to kick start the send?
    MPI_Test(&reqs[1], &completed, MPI_STATUS_IGNORE);
    //std::cout << "send from " << comm_rank << " to " << curr_peer << " is " << (completed ? "" : " not ") << " complete "<< std::endl;

    //=== no more new updates

    //=== and accumulate
    internal_merge(accumulating, recved);
    //std::cout << "  per rank estimate for step " << (comm_size - 2) << " peer " << prev_peer << ": " << internal_estimate(accumulating) << std::endl;

    // wait for both to complete
    MPI_Waitall(2, reqs.data(), MPI_STATUSES_IGNORE);

    // now swap.
    std::swap(recving, recved);
    prev_peer = curr_peer;

    //=== last accumulate, for prev_peer
    internal_merge(accumulating, recved);
    //std::cout << "  per rank estimate for step " << (comm_size - 1) << " peer " << prev_peer << ": " << internal_estimate(accumulating) << std::endl;


    free(buffers);

  }
  // NOT using MPI_reduce per bucket: (2^precision)*log(p) for each of p iterations. in time and space.  the above is (2^precision)*p

  template <typename SIZE>
  void update_per_rank_by_hashval(HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                      ::mxx::comm const & comm) {
    update_per_rank_by_hashval_internal(this->registers.data(), first, last, send_counts, comm);
  }


  // distributed estimate. it is not clear that this way is mathematically correct, other than relying on the fact that
  // the input on different nodes should be iid with uniform distribution, so that individual buckets should be iids as well.
  // assuming that bins do not need to contain strictly partitioned data, then the same bin on each rank with the same hash
  // value ranges can be treated as separate and independent bins.
  // then we can shuffle the bins thus construct new registers.
  // in fact, for each bucket, we only need max(1, (2^precision)/p) bins and the rest of the algorithm can be similar to estimate per rank algo.
  // however, recall that the bits for mapping to ranks are the most significant bits
  // and they are excluded, so we'd need some way to properly assign the bins.
  // NOTE: OVERESTIMATES BY NEARLY DOUBLE FOR Fvesca.  DO NOT USE.
  template <typename SIZE>
  ::std::vector<REG_T> update_per_rank_experimental_by_hashval(HVT* first, HVT* last, std::vector<SIZE> const & send_counts,
                                        ::mxx::comm const & comm) {
    // first compute the local estimates
    // create the registers
    ::std::vector<REG_T> regs(nRegisters, static_cast<REG_T>(0));

    // perform updates on the registers.
    for (; first != last; ++first) {
      internal_update(regs, *first << ignored_msb);
    }

    // now compute send counts.  each rank ends up with 2^precision number of entries.
    size_t count = 0;
    size_t last_sc = 0, sc= 0;
    size_t comm_size = comm.size();
    size_t comm_rank = comm.rank();
    size_t rem = 0;
    std::vector<SIZE> scounts;
    scounts.reserve(comm_size);
    for (size_t i = 1; i <= comm_size; ++i) {
      count = i * nRegisters;
      rem = count % comm_size;
      sc = (count + ((comm_rank < rem) ? (comm_size - 1) : 0)) / comm_size;
      scounts.emplace_back(sc - last_sc);
      last_sc = sc;
    }


    // then shuffle.
    return mxx::all2allv(regs, scounts, comm);
  }


#endif


};
template <typename T, typename Hash, uint8_t precision>
constexpr uint32_t hyperloglog64<T, Hash, precision>::nRegisters;
template <typename T, typename Hash, uint8_t precision>
constexpr uint8_t hyperloglog64<T, Hash, precision>::hvt_size;
template <typename T, typename Hash, uint8_t precision>
constexpr uint8_t hyperloglog64<T, Hash, precision>::no_ignore_value_bits;
template <typename T, typename Hash, uint8_t precision>
constexpr uint8_t hyperloglog64<T, Hash, precision>::unused_msb;
template <typename T, typename Hash, uint8_t precision>
constexpr double hyperloglog64<T, Hash, precision>::est_error_rate;





#endif // KMERHASH_HYPERLOGLOG64_HPP_
