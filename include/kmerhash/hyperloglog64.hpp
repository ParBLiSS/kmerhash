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
 * [ ] distributed estimation
 * [ ] exclude some leading bits (for use e.g. after data is distributed.)
 * [ ] exclude some trailing bits....
 * [ ] batch processing (to allow prefetching later.)
 *
 *  Created on: Mar 1, 2017
 *      Author: tpan
 */

#ifndef KMERHASH_HYPERLOGLOG64_HPP_
#define KMERHASH_HYPERLOGLOG64_HPP_

#include <array>
#include <stdint.h>
#include <iostream> // std::cout

// FROM https://github.com/hideo55/cpp-HyperLogLog.
// modified to conform to the leftmost 1 bit convention, and faster implementation. and to make macro safer.
// zero -> return 65.  else return b+1
#if defined(__has_builtin) && (defined(__GNUC__) || defined(__clang__))

inline uint8_t leftmost_set_bit(uint64_t x) {
	return (x == 0) ? 65 : static_cast<uint8_t>(::__builtin_clzl(x) + 1);
}

#else

#if defined (_MSC_VER)
inline uint8_t leftmost_set_bit(uint64_t x) {
	if (x == 0) return 65;
    uint64_t b = 64;
    ::_BitScanReverse64(&b, x);
    return static_cast<uint8_t>(b);
}
#else
// from https://en.wikipedia.org/wiki/Find_first_set, extended to 64 bit
static const uint8_t clz_table_4bit[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
inline uint8_t leftmost_set_bit(uint64_t x) {
  if (x == 0) return 65;
  uint8_t n;
  if ((x & 0xFFFFFFFF00000000ULL) == 0) {n  = 32; x <<= 32;} else {n = 0;}
  if ((x & 0xFFFF000000000000ULL) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ULL) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ULL) == 0) {n +=  4; x <<=  4;}
  n += clz_table_4bit[x >> 60];  // 64 - 4
  return n+1;
}

#endif /* defined(_MSC_VER) */

#endif /* defined(__GNUC__) */




//========= specializations for std::hash. std::hash passes through primitive types, e.g. uint8_t.
//   this creates problems in that the high bits are all 0.  so for std::hash,
//   we use the lower bits for register and upper bits for the bit counts.
//   also, for small datatypes, the number of leading zeros may be artificially large.
//      we can fix this by tracking the minimum leading zero as well.
// make the unused_msb be 0 for data type larger than 64, else 64  - actual data size.
//========== non-std::hash
//   the unused msb is 0.

template <typename T, typename Hash,
		uint8_t precision = 12U,
		uint8_t unused_msb = (std::is_same<::std::hash<T>, Hash>::value) ?
				((sizeof(T) >= 8ULL) ? 0U : (64U - (sizeof(T) * 8ULL))) :
				0U
		>
class hyperloglog64 {
	  static_assert((precision >= 4ULL) && (precision <= 16ULL),
			  "ERROR: precision for hyperloglog should be in [4, 16].");

protected:
	using REG_T = uint8_t;

	// 64 bit.  0xIIRRVVVVVV  // MSB: ignored bits II,  high bits: reg, RR.  low: values, VVVVVV
	static constexpr uint8_t value_bits = 64U - precision;  // e.g. 0x00FFFFFF
	static constexpr uint64_t nRegisters = 0x1ULL << precision;   // e.g. 0x00000100
	mutable double amm;

	mutable uint8_t ignored_msb; // MSB to ignore.
	mutable uint64_t lzc_mask;   // lowest bits set to 1 to prevent counting into that region.

	::std::vector<REG_T> registers;  // stores count of leading zeros

	Hash h;

	inline void internal_update(uint64_t const & no_ignore) {
		// bits we have are 0xRRVVVVVV0000
		// extract RR:      0x0000000000RR
        uint64_t i = no_ignore >> value_bits;   // first precision bits are for register id
        // next count:  want to count 0xVVVVVV111111.
        // compute lzcnt +1 from VVVVVV
        REG_T rank = leftmost_set_bit((no_ignore << precision) | lzc_mask);  // then find leading 1 in remaining
        if (rank > registers[i]) {
            registers[i] = rank;
        }
	}


public:

  static constexpr double est_error_rate = static_cast<double>(1.04) / static_cast<double>(0x1ULL << (precision >> 1ULL));   // avoid std::sqrt in constexpr.

  	/// constructor.  ignore_leading does not count the leading bits.  ignore_trailing does not count the trailing bits.
  	  /// these are for use when the leading and/or trailing bits are identical in an input set.
	hyperloglog64(uint8_t const & ignore_msb = 0) :
		ignored_msb(ignore_msb + unused_msb), lzc_mask( ~(0x0ULL) >> (64 - precision - ignore_msb - unused_msb)),
		registers(nRegisters, static_cast<REG_T>(0)) {

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
        amm *= static_cast<double>(0x1ULL << (precision << 1ULL));  // 2^(2*precision)
	}

	hyperloglog64(hyperloglog64 const & other) :
		amm(other.amm), ignored_msb(other.ignored_msb), lzc_mask(other.lzc_mask), registers(other.registers) {
	}

	hyperloglog64(hyperloglog64 && other) :
		amm(other.amm), ignored_msb(other.ignored_msb), lzc_mask(other.lzc_mask), registers(std::move(other.registers)) {
	}

	hyperloglog64& operator=(hyperloglog64 const & other) {
		amm = other.amm;
		registers = other.registers;
		ignored_msb = other.ignored_msb;
		lzc_mask = other.lzc_mask;

		return *this;
	}

	hyperloglog64& operator=(hyperloglog64 && other) {
		amm = other.amm;
		registers.swap(other.registers);
		ignored_msb = other.ignored_msb;
		lzc_mask = other.lzc_mask;

		return *this;
	}

	void swap(hyperloglog64 & other) {
		std::swap(amm, other.amm);
		std::swap(registers, other.registers);
		std::swap(ignored_msb, other.ignored_msb);
		std::swap(lzc_mask, other.lzc_mask);
	}

	inline void set_ignored_msb(uint8_t const & ignored_msb) {
		this->ignored_msb = ignored_msb;
	}
	inline uint8_t get_ignored_msb() {
		return this->ignored_msb;
	}
	hyperloglog64 make_empty_copy() {
		return hyperloglog64(this->ignored_msb);
	}


	inline void update(T const & val) {
        internal_update(h(val) << ignored_msb);
	}

	inline void update_via_hashval(uint64_t const & hash) {
        internal_update(hash << ignored_msb);
	}


	double estimate() const {
        double estimate = static_cast<double>(0.0);
        double sum = static_cast<double>(0.0);

        // compute the denominator of the harmonic mean
        for (size_t i = 0; i < nRegisters; i++) {
            sum += static_cast<double>(1.0) / static_cast<double>(1ULL << registers[i]);
        }
        estimate = amm / sum; // E in the original paper

        if (estimate <= static_cast<double>(5ULL * (nRegisters >> 1ULL))) {  // 5m/2
        	size_t zeros = count_zeros();
        	if (zeros > 0ULL) {
//        		std::cout << "linear_count: zero: " << zeros << " estimate " << estimate << std::endl;
				return linear_count(zeros);
        	} else
        		return estimate;
//        } else if (estimate > (1.0 / 30.0) * pow_2_32) {
        	// don't need this because of 64bit.
//            estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
        } else
        	return estimate;
	}



	void merge(hyperloglog64 const & other) {
		// procisions identical, so don't need to check number of registers either.

		// iterate over both, merge, and update the zero count.
		for (size_t i = 0; i < nRegisters; ++i) {
			registers[i] = std::max(registers[i], other.registers[i]);
		}
	}

	inline uint64_t count_zeros() const {
		uint64_t zeros = 0;
		for (size_t i = 0; i < nRegisters; i++) {
			if (registers[i] == 0) ++zeros;
		}
		return zeros;
	}

	inline double linear_count(uint64_t const & zeros) const {
		return static_cast<double>(nRegisters) *
				std::log(static_cast<double>(nRegisters)/ static_cast<double>(zeros));  //natural log
	}

	void clear() {
		registers.assign(nRegisters, static_cast<REG_T>(0));
	}

};





////========= specializations for std::hash. std::hash passes through primitive types, e.g. uint8_t.
////   this creates problems in that the high bits are all 0.  so for std::hash,
////   we use the lower bits for register and upper bits for the bit counts.
////   also, for small datatypes, the number of leading zeros may be artificially large.
////      we can fix this by tracking the minimum leading zero as well.
//// make the unused_msb be 0 for data type larger than 64, else 64  - actual data size.
//template <typename T, uint8_t precision>
//class hyperloglog64<T, std::hash<T>, precision, (sizeof(T) >= 8) ? 0 : (64 - (sizeof(T) * 8ULL))> {
//
//	  static_assert((precision < (sizeof(T) * 8ULL)) && (precision >= 4ULL) && (precision <= 16ULL),
//			  "precision must be set to lower than sizeof(T) * 8 and in range [4, 16]");
//
//protected:
//	  static constexpr uint8_t data_bits = ((sizeof(T) * 8ULL) > 64ULL) ? 64ULL : (sizeof(T) * 8ULL);
//	  static constexpr uint8_t value_bits = data_bits - precision - ignored_msb;
//	  static constexpr uint8_t lead_zero_bits = 64ULL - value_bits;  // in case sizeof(T) < 8.  exclude reg bits and ignored leading zeros..
//
//	  // register mask is the upper most precision bits.
//  static constexpr uint64_t nRegisters = 0x1ULL << precision;
//  static constexpr uint64_t shifted_reg_mask = nRegisters - 1ULL;   // need to shift right value_bits first.
//  static constexpr uint64_t val_mask = ~(0x0ULL) >> lead_zero_bits;
//
//  double amm;
//
//	using REG_T = uint8_t;
//	std::array<REG_T, nRegisters> registers;  // stores count of leading zeros
//
//  std::hash<T> h;
//
//public:
//  static constexpr double est_error_rate = static_cast<double>(1.04) / static_cast<double>(0x1ULL << (precision >> 1ULL));   // avoid std::sqrt in constexpr.
//
//
//  hyperloglog64() {
//    registers.fill(static_cast<REG_T>(0));
//
//        switch (precision) {
//            case 4:
//                amm = static_cast<double>(0.673);
//                break;
//            case 5:
//                amm = static_cast<double>(0.697);
//                break;
//            case 6:
//                amm = static_cast<double>(0.709);
//                break;
//            default:
//                amm = static_cast<double>(0.7213) / (static_cast<double>(1.0) + static_cast<double>(1.079) / static_cast<double>(nRegisters));
//                break;
//        }
//        amm *= static_cast<double>(0x1ULL << (precision << 1ULL));  // 2 ^(precision *2)
//  }
//
//  hyperloglog64(hyperloglog64 const & other) : amm(other.amm), registers(other.registers) {}
//
//  hyperloglog64(hyperloglog64 && other) : amm(other.amm),
//      registers(std::move(other.registers)) {}
//
//
//  hyperloglog64& operator=(hyperloglog64 const & other) {
//    amm = other.amm;
//    registers = other.registers;
//  }
//
//  hyperloglog64& operator=(hyperloglog64 && other) {
//    amm = other.amm;
//    registers.swap(std::move(other.registers));
//  }
//
//  void swap(hyperloglog64 & other) {
//    std::swap(amm, other.amm);
//    std::swap(registers, other.registers);
//  }
//
//  inline void update(T const & val) {
//        update_via_hashval(h(val));
//  }
//  inline void update_via_hashval(uint64_t const & hash) {
//        uint64_t i = (hash >> value_bits) & shifted_reg_mask;   // first precision bits are for register id
//        uint8_t rank = leftmost_set_bit(hash & val_mask) - lead_zero_bits;  // then find leading 1 in remaining
//        if (rank > registers[i]) {
//            registers[i] = rank;
//        }
//  }
//
//  double estimate() const {
//        double estimate = static_cast<double>(0.0);
//        double sum = static_cast<double>(0.0);
//
//        // compute the denominator of the harmonic mean
//        for (size_t i = 0; i < nRegisters; i++) {
//            sum += static_cast<double>(1.0) / static_cast<double>(1ULL << registers[i] );
//        }
//        estimate = amm / sum; // E in the original paper
//
//        if (estimate <= static_cast<double>(5ULL * (nRegisters >> 1ULL))) { // 5m/2
//        	size_t zeros = count_zeros();
//        	if (zeros > 0ULL) {
////        		std::cout << "linear_count: zero: " << zeros << " estimate " << estimate << std::endl;
//				return linear_count(zeros);
//        	} else
//        		return estimate;
////        } else if (estimate > (1.0 / 30.0) * pow_2_32) {
//            // don't need below because of 64bit.
////            estimate = neg_pow_2_32 * log(1.0 - (estimate / pow_2_32));
//        } else
//          return estimate;
//  }
//
//
//
//  void merge(hyperloglog64 const & other) {
//    // precisions identical, so don't need to check number of registers either.
//
//    // iterate over both, merge, and update the zero count.
//    for (size_t i = 0; i < nRegisters; ++i) {
//      registers[i] = std::max(registers[i], other.registers[i]);
//    }
//  }
//
//	inline uint64_t count_zeros() const {
//		uint64_t zeros = 0;
//		for (size_t i = 0; i < nRegisters; i++) {
//			if (registers[i] == 0) ++zeros;
//		}
//		return zeros;
//	}
//
//	inline double linear_count(uint64_t const & zeros) const {
//		return static_cast<double>(nRegisters) *
//				std::log(static_cast<double>(nRegisters)/ static_cast<double>(zeros));  //natural log
//	}
//
//	void clear() {
//		registers.fill(static_cast<REG_T>(0));
//	}
//
//};




#endif // KMERHASH_HYPERLOGLOG64_HPP_
