/*
 * Copyri 1:ght 2015 Georgia Institute of Technology
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

/**
 * @file    hash.hpp
 * @ingroup fsc::hash
 * @author  tpan
 * @brief   collections of hash functions defined for kmers.
 * @details support the following:  raw bits directly extracted; std::hash version; murmurhash; and farm hash
 *
 *          assuming the use is in a distributed hash table with total buckets N,  N = p * t * l,
 *          where p is number of processes, t is number of threads, and l is number of local buckets.
 *
 *          then the key is assigned to a bucket via hash(key) % N.
 *            the process assignment is: hash(key) / (t * l)
 *            the thread assignment is:  (hash(key) / l) % t
 *            the local bucket assignment is: hash(key) % l;
 *
 *          there are unlikely to be more than 2^64 local buckets, so we can limit the hash(key) % l to be the lower 64bit of hash(key).
 *          this also means that if the hash key is 64 bits, then no shifting or bit masking is needed, which improves performance for local hashtable lookup.
 *
 *          l is a variable that is decided by the local hash table based on number of entries.
 *          we should instead look at the first ceil(log (p*t)) bits of the hash(key).  let's call this "prefix".
 *
 *          process assignment is then:  prefix / t
 *          thread assignment is then:   prefix % t.
 *
 *          prefix for process assignment can be changed to use ceil(log(p)) bits, let's call this "pre-prefix".
 *
 *          process assignment is then:  pre-prefix % p.
 *
 *          2 functions are sufficient then:  prefix_hash(), and suffix_hash().
 *            we restrict our hash() functions to return 64 bits, and make suffix hash() to be the same as hash().
 *
 *
 *          namespace bliss::hash::kmer has a generic hash function that can work with kmer, kmer xor rev comp, computed or provided.
 *          the generic hash function also allows customization via bliss::hash::kmer::detail::{std,identity,murmur,farm} specializations
 *
 *          as stated above, 2 versions for each specialization: hash() and hash_prefix().  the specialization is especially for identity and murmur hashes,
 *          as murmur hash produces 128 bit value, and identity hash uses the original kmer.
 *
 *NOTE: this is implemented based on murmur3 128 bit hash for x86 hardware, because there are no mullo_epi64 in avx or sse.
 * 
 * 
 *NOTE:
 *      avx2 state transition when upper 128 bit may not be zero: STATE C:  up to 6x slower.
 *      	https://software.intel.com/en-us/articles/intel-avx-state-transitions-migrating-sse-code-to-avx
 *      using vtunes shows that this has not been an issue.
 *
 *      	however, clearing all registers would also clear all stored constants, which would then need to be reloaded.
 *      	this can be done, but will require  some code change.
 *  TODO: [ ] proper AVX state transition, with load.  NOTE: vtunes measurement with benchmark_hashtables does not seem to indicate penalities in canonicalization or hashing.
 *        [ ] tuning to avoid skipped reading - avoid cache set contention.
 *        [ ] try to stick to some small multiples of 64 bytes.
 *        [ ] schedule instructions to maximize usage of ALU and other vector units.
 *        [ ] at 64 bytes per element, we are at limit of 8-way set associative cache (L1)...
 */
#ifndef MURMUR3_64_AVX_HPP_
#define MURMUR3_64_AVX_HPP_

#include <type_traits> // enable_if
#include <cstring>     // memcpy
#include <stdexcept>   // logic error
#include <stdint.h>    // std int strings
#include <iostream>    // cout

#include "utils/filter_utils.hpp"
#include "utils/transform_utils.hpp"
#include "kmerhash/math_utils.hpp"

#ifndef FSC_FORCE_INLINE

#if defined(_MSC_VER)

#define FSC_FORCE_INLINE __forceinline

// Other compilers

#else // defined(_MSC_VER)

#define FSC_FORCE_INLINE inline __attribute__((always_inline))

#endif // !defined(_MSC_VER)

#endif

#include <x86intrin.h>

namespace fsc
{

namespace hash
{
// code below assumes sse and not avx (yet)
namespace sse
{

// TODO: [ ] remove use of set1 in code.
#if defined(__AVX2__)
// for 32 bit buckets
// original: body: 16 inst per iter of 4 bytes; tail: 15 instr. ; finalization:  8 instr.
// about 4 inst per byte + 8, for each hash value.

// base class to park some methods.  note that we have const variables and not static ones - not clear what it means for __m256i to be static.
template <typename T>
class Murmur64AVX
{

protected:
  static const __m256i mix_const1;
  static const __m256i mix_const2;
  static const __m256i c11;
  static const __m256i c12;
  static const __m256i c13;
  static const __m256i c14;
  static const __m256i c41;
  static const __m256i c42;
  static const __m256i c43;
  static const __m256i c44;
  static const __m256i length;


  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void rotl(uint8_t const & r1, uint8_t const & r2,
		  uint8_t const & r3, uint8_t const & r4,
		  __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {

    __m256i tt0, tt1, tt2, tt3;

    if (INT_CNT >= 1) {
      tt1 = _mm256_slli_epi32(t0, r1);   // SLLI: L1 C1 p0
      tt0 = _mm256_srli_epi32(t0, (32 - r1));   // SRLI: L1 C1 p0
    }
    if (INT_CNT >= 2) {
      tt3 = _mm256_slli_epi32(t1, r2);   // SLLI: L1 C1 p0
      tt2 = _mm256_srli_epi32(t1, (32 - r2));   // SRLI: L1 C1 p0
    }
    if (INT_CNT >= 1) t0 = _mm256_or_si256(tt1, tt0);  // OR: L1 C0.33 p015
    if (INT_CNT >= 2) t1 = _mm256_or_si256(tt3, tt2);  // OR: L1 C0.33 p015

    if (INT_CNT >= 3) {
      tt1 = _mm256_slli_epi32(t2, r3);   // SLLI: L1 C1 p0
      tt0 = _mm256_srli_epi32(t2, (32 - r3));   // SRLI: L1 C1 p0
    }
    if (INT_CNT >= 4) {
      tt3 = _mm256_slli_epi32(t3, r4);   // SLLI: L1 C1 p0
      tt2 = _mm256_srli_epi32(t3, (32 - r4));   // SRLI: L1 C1 p0
    }

    if (INT_CNT >= 3) t2 = _mm256_or_si256(tt1, tt0);  // OR: L1 C0.33 p015
    if (INT_CNT >= 4) t3 = _mm256_or_si256(tt3, tt2);  // OR: L1 C0.33 p015
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void mul(__m256i const & m1,
		  __m256i const & m2,
		  __m256i const & m3,
		  __m256i const & m4,
		  __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    if (INT_CNT >= 1) t0 = _mm256_mullo_epi32(t0, m1); // sse   // Lat10, CPI2
    if (INT_CNT >= 2) t1 = _mm256_mullo_epi32(t1, m2); // sse   // Lat10, CPI2
    if (INT_CNT >= 3) t2 = _mm256_mullo_epi32(t2, m3); // sse   // Lat10, CPI2
    if (INT_CNT >= 4) t3 = _mm256_mullo_epi32(t3, m4); // sse   // MULLO: L10 C2 2p0
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void update_part2(__m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    
    rotl<INT_CNT>(15, 16, 17, 18, t0, t1, t2, t3);
    mul<INT_CNT>(this->c12, this->c13, this->c14, this->c11, t0, t1, t2, t3);
    
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void xor4(
		  __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3,
          __m256i const &t0, __m256i const &t1, __m256i const &t2, __m256i const &t3) const
  {
  // unlike 32 bit, we will do not wait for next iteration to finish the addition.
	// first do the add.
	if (INT_CNT >= 1) h0 = _mm256_xor_si256(h0, t0); // sse
	if (INT_CNT >= 2) h1 = _mm256_xor_si256(h1, t1); // sse
	if (INT_CNT >= 3) h2 = _mm256_xor_si256(h2, t2); // sse
	if (INT_CNT >= 4) h3 = _mm256_xor_si256(h3, t3); // sse                                    // XOR: L1 C0.33 p015
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void update_part3(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3,
                                     __m256i const &t0, __m256i const &t1, __m256i const &t2, __m256i const &t3) const
  {
	  // unlike 32 bit, we will do not wait for next iteration to finish the addition.
	__m256i hh1, hh2, hh3;
	hh1 = h1; hh2 = h2; hh3 = h3;
	  // next we need to do h1, h2, h3, h4 in order.
	  // h1 depends on h2 with addition
	  // h2 depends on h3 with addition. ...
	  // h4 depends on completed processing of h1.

	// first do the add.
	xor4<INT_CNT>(h0, h1, h2, h3, t0, t1, t2, t3);

	// next rotate
	rotl<4>(19, 17, 15, 13, h0, h1, h2, h3);

	// then mix the h0, h1, h2, h3...  Here we do the first 3 (h0..h2), and then the last one separately.
	h0 = _mm256_add_epi32(h0, hh1);
	h1 = _mm256_add_epi32(h1, hh2);
	h2 = _mm256_add_epi32(h2, hh3);

	// just need h1 calculated.
	hh3 = _mm256_slli_epi32(h0, 2);
	hh1 = _mm256_slli_epi32(h1, 2);

	h0 = _mm256_add_epi32(h0, hh3); // multiply by 5.  do as 1x + 4x, which is lower latency.
	h1 = _mm256_add_epi32(h1, hh1); // multiply by 5.  do as 1x + 4x, which is lower latency.

	h0 = _mm256_add_epi32(h0, this->c41);
	h1 = _mm256_add_epi32(h1, this->c42);

	// add newly calculated h1 to h3.
	h3 = _mm256_add_epi32(h3, h0);


	hh2 = _mm256_slli_epi32(h2, 2);
	hh3 = _mm256_slli_epi32(h3, 2);

	h2 = _mm256_add_epi32(h2, hh2); // multiply by 5.  do as 1x + 4x, which is lower latency.
	h3 = _mm256_add_epi32(h3, hh3); // multiply by 5.  do as 1x + 4x, which is lower latency.

	h2 = _mm256_add_epi32(h2, this->c43);
	h3 = _mm256_add_epi32(h3, this->c44);
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t INT_CNT>
  FSC_FORCE_INLINE void update_part3_partial(
		  __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3,
          __m256i const &t0, __m256i const &t1, __m256i const &t2, __m256i const &t3) const
  {
  // unlike 32 bit, we will do not wait for next iteration to finish the addition.
	// first do the add.
	xor4<INT_CNT>(h0, h1, h2, h3, t0, t1, t2, t3);
  }


  FSC_FORCE_INLINE void hmix32(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
	  h0 = _mm256_add_epi32(h0, h1);										    // ADD: L1 C0.5 p15
	  __m256i t2 = _mm256_add_epi32(h2, h3);
	  h0 = _mm256_add_epi32(h0, t2);  // h0 = h0 + h1 + h2 + h3

	  h1 = _mm256_add_epi32(h1, h0);  // h1 = 2*h1 + h0 + h2 + h3
	  h2 = _mm256_add_epi32(h2, h0);  // h2 = 2*h2 + h0 + h1 + h3
	  h3 = _mm256_add_epi32(h3, h0);  // h3 = 2*h3 + h0 + h1 + h2
  }

  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void shift_xor(uint8_t const & shift, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    __m256i tt0, tt1;
    if (VEC_CNT >= 1) tt0 = _mm256_srli_epi32(h0, shift);
    if (VEC_CNT >= 2) tt1 = _mm256_srli_epi32(h1, shift);
    if (VEC_CNT >= 1) h0 = _mm256_xor_si256(h0, tt0); // h ^= h >> 16;      sse2
    if (VEC_CNT >= 2) h1 = _mm256_xor_si256(h1, tt1); // h ^= h >> 16;      sse2

    if (VEC_CNT >= 3) tt0 = _mm256_srli_epi32(h2, shift); 
    if (VEC_CNT >= 4) tt1 = _mm256_srli_epi32(h3, shift);                         // SRLI: L1, C1, p0.  
    if (VEC_CNT >= 3) h2 = _mm256_xor_si256(h2, tt0); // h ^= h >> 16;      sse2
    if (VEC_CNT >= 4) h3 = _mm256_xor_si256(h3, tt1); // h ^= h >> 16;      sse2  // XOR: L1, C0.33, p015
    
  }


  /// fmix32 for 16 elements at a time.
  FSC_FORCE_INLINE void fmix32(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // // should have 0 idle latency cyles and 0 cpi cycles here.
    shift_xor<4>(16, h0, h1, h2, h3);
    mul<4>(this->mix_const1, this->mix_const1,this->mix_const1,this->mix_const1, h0, h1, h2, h3);

    // // should have 1 idle latency cyles and 2 cpi cycles here.

    shift_xor<4>(13, h0, h1, h2, h3);
    mul<4>(this->mix_const2, this->mix_const2, this->mix_const2, this->mix_const2, h0, h1, h2, h3);

    // // latencies.
    // // h3  Lat 1, cpi 2
    // // h0  Lat 4, cpi 2

    // // expect Lat 0, cycle 1
    shift_xor<4>(16, h0, h1, h2, h3);

  }



  /// fmix32 for 16 elements at a time.
  FSC_FORCE_INLINE void fmix32_part2(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // should have 1 idle latency cyles and 2 cpi cycles here.

    shift_xor<4>(13, h0, h1, h2, h3);
    mul<4>(this->mix_const2, this->mix_const2,this->mix_const2,this->mix_const2,h0, h1, h2, h3);

    // latencies.
    // h3  Lat 1, cpi 2
    // h0  Lat 4, cpi 2

    // expect Lat 0, cycle 1
    shift_xor<4>(16, h0, h1, h2, h3);

  }

protected:
  // LATENCIES for instruction with latency > 1
  // 1. mullo_epi16 has latency of 5 cycles, CPI of 1 to 0.5 cycles - need unroll 2x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 2. mullo_epi32 has latency of 10 cycles, CPI of 1 to 0.5 cycles - need unroll 3-4x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 3. _mm256_permutevar8x32_epi32, _mm256_permute4x64_epi64, and _mm256_insertf128_si256  have latency of 3, CPI of 1.
  // 4. note that most extract calls have latency of 3 and CPI of 1, except for _mm256_extracti128_si256, which has latency of 1.
  // 5. _mm256_insertf128_si256 has latency of 3. and CPI of 1.

  // NOTE: use static variable for consts and array for per instance values.  WORK AROUND FOR STACK NOT ALIGNED TO 32 BYTES.
  //     when too many ymm registers, or for function returns, ymm copy to stack could fail due to insufficient alignment.
  static const __m256i permute1;
  static const __m256i permute16;
  static const __m256i shuffle0; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  static const __m256i shuffle1; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  static const __m256i shuffle2; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  static const __m256i shuffle3; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  static const __m256i shuffle1_epi8;
  static const __m256i ones;
  static const __m256i zeros;
  static const __m128i zeroi128;

  uint32_t seed_arr[8];

public:
  static constexpr size_t batch_size = (sizeof(T) == 1) ? 32 : ((sizeof(T) == 2) ? 16 : 8);

  explicit Murmur64AVX(uint32_t const & _seed = 43U) 
  {
    for (int i = 0; i < 8; ++i) {
      seed_arr[i] = _seed;
    }
  }

  explicit Murmur64AVX(Murmur64AVX const &other) 
  {
    memcpy(seed_arr, other.seed_arr, 32);
  }

  explicit Murmur64AVX(Murmur64AVX &&other) 
  {
    memcpy(seed_arr, other.seed_arr, 32);
  }

  Murmur64AVX &operator=(Murmur64AVX const &other)
  {

    memcpy(seed_arr, other.seed_arr, 32);
    return *this;
  }

  Murmur64AVX &operator=(Murmur64AVX &&other)
  {
    memcpy(seed_arr, other.seed_arr, 32);
    return *this;
  }

  // useful for computing 4 32bit hashes in 1 pass (for hashing into less than 2^32 buckets)
  // assume 4 streams are available.
  // working with 4 bytes at a time because there are
  // init: 4 instr.
  // body: 13*4 + 12  per iter of 16 bytes
  // tail: about the same
  // finalize: 11 inst. for 4 elements.
  // about 5 inst per byte + 11 inst for 4 elements.
  // for types that are have size larger than 8 or not power of 2.
  // hash up to 32 elements at a time. at a time.  each is 1 byte
  // STREAMING:  default do not stream, as a lot of use cases end up reading the same region.
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE == 1), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint64_t *out) const
  {
    // process 32 streams at a time.  all should be the same length.

    assert((nstreams <= 32) && "maximum number of streams is 32");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m256i h0, h1, h2, h3, h4, h5, h6, h7;

    hash(key, h0, h1, h2, h3, h4, h5, h6, h7);

    // do the full ones first.
    size_t blocks = nstreams >> 2;   // divide by 4 per vector
    uint8_t rem = nstreams & 3; // remainder.  at most 3.
    
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      if (blocks >= 1) _mm256_stream_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_stream_si256((__m256i *)(out + 4), h1);
      if (blocks >= 3) _mm256_stream_si256((__m256i *)(out + 8), h2);
      if (blocks >= 4) _mm256_stream_si256((__m256i *)(out + 12), h3);
      if (blocks >= 5) _mm256_stream_si256((__m256i *)(out + 16), h4);
      if (blocks >= 6) _mm256_stream_si256((__m256i *)(out + 20), h5);
      if (blocks >= 7) _mm256_stream_si256((__m256i *)(out + 24), h6);
      if (blocks >= 8) _mm256_stream_si256((__m256i *)(out + 28), h7);
    } else {
      if (blocks >= 1) _mm256_storeu_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_storeu_si256((__m256i *)(out + 4), h1);
      if (blocks >= 3) _mm256_storeu_si256((__m256i *)(out + 8), h2);
      if (blocks >= 4) _mm256_storeu_si256((__m256i *)(out + 12), h3);
      if (blocks >= 5) _mm256_storeu_si256((__m256i *)(out + 16), h4);
      if (blocks >= 6) _mm256_storeu_si256((__m256i *)(out + 20), h5);
      if (blocks >= 7) _mm256_storeu_si256((__m256i *)(out + 24), h6);
      if (blocks >= 8) _mm256_storeu_si256((__m256i *)(out + 28), h7);
    }
    if (rem > 0)
    {
      // write remainders
      switch (blocks)
      {
        case 7:
        memcpy(out + 28, reinterpret_cast<uint64_t *>(&h7), rem << 3); // copy 8 bytes
        break;
      case 6:
        memcpy(out + 24, reinterpret_cast<uint64_t *>(&h6), rem << 3); // copy bytes
        break;
      case 5:
        memcpy(out + 20, reinterpret_cast<uint64_t *>(&h5), rem << 3); // copy bytes
        break;
      case 4:
        memcpy(out + 16, reinterpret_cast<uint64_t *>(&h4), rem << 3); // copy bytes
        break;
        case 3:
        memcpy(out + 12, reinterpret_cast<uint64_t *>(&h3), rem << 3); // copy bytes
        break;
      case 2:
        memcpy(out + 8, reinterpret_cast<uint64_t *>(&h2), rem << 3); // copy bytes
        break;
      case 1:
        memcpy(out + 4, reinterpret_cast<uint64_t *>(&h1), rem << 3); // copy bytes
        break;
      case 0:
        memcpy(out, reinterpret_cast<uint64_t *>(&h0), rem << 3); // copy bytes
        break;
      default:
        break;
      }
    }
  
  }
  // STREAMING:  default do not stream, as a lot of use cases end up reading the same region.
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE == 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint64_t *out) const
  {
    // process 4 streams at a time.  all should be the same length.
    // process 4 streams at a time.  all should be the same length.

    assert((nstreams <= 16) && "maximum number of streams is 16");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m256i h0, h1, h2, h3;

    hash(key, h0, h1, h2, h3);

    // do the full ones first.
    size_t blocks = nstreams >> 2;   // divide by 4 per vector
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      if (blocks >= 1) _mm256_stream_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_stream_si256((__m256i *)(out + 4), h1);
      if (blocks >= 3) _mm256_stream_si256((__m256i *)(out + 8), h2);
      if (blocks >= 4) _mm256_stream_si256((__m256i *)(out + 12), h3);
    } else {      
      if (blocks >= 1) _mm256_storeu_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_storeu_si256((__m256i *)(out + 4), h1);
      if (blocks >= 3) _mm256_storeu_si256((__m256i *)(out + 8), h2);
      if (blocks >= 4) _mm256_storeu_si256((__m256i *)(out + 12), h3);
    }

    uint8_t rem = nstreams & 3; // remainder.
    if (rem > 0)
    {
      // write remainders
      switch (blocks)
      {
        case 3:
        memcpy(out + 12, reinterpret_cast<uint64_t *>(&h3), rem << 3); // copy bytes
        break;
      case 2:
        memcpy(out + 8, reinterpret_cast<uint64_t *>(&h2), rem << 3); // copy bytes
        break;
      case 1:
        memcpy(out + 4, reinterpret_cast<uint64_t *>(&h1), rem << 3); // copy bytes
        break;
      case 0:
        memcpy(out, reinterpret_cast<uint64_t *>(&h0), rem << 3); // copy bytes
        break;
      default:
        break;
      }
    }
  }
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE > 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint64_t *out) const
  {
    // process 8 streams at a time.  all should be the same length.

    assert((nstreams <= 8) && "maximum number of streams is 8");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m256i h0, h1;

    switch (nstreams) {
      case 8:     hash<8>(key, h0, h1); break;
      case 7:     hash<7>(key, h0, h1); break;
      case 6:     hash<6>(key, h0, h1); break;
      case 5:     hash<5>(key, h0, h1); break;
      case 4:     hash<4>(key, h0, h1); break;
      case 3:     hash<3>(key, h0, h1); break;
      case 2:     hash<2>(key, h0, h1); break;
      case 1:     hash<1>(key, h0, h1); break;
    }

    // do the full ones first.
    size_t blocks = nstreams >> 2;   // divide by 4 per vector
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      if (blocks >= 1) _mm256_stream_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_stream_si256((__m256i *)(out + 4), h1);
    } else {
      if (blocks >= 1) _mm256_storeu_si256((__m256i *)out, h0);
      if (blocks >= 2) _mm256_storeu_si256((__m256i *)(out + 4), h1);
        
    }

    uint8_t rem = nstreams & 3; // remainder.
    if (rem > 0)
    {
      // write remainders
      switch (blocks)
      {
      case 1:
        memcpy(out + 4, reinterpret_cast<uint64_t *>(&h1), rem << 3); // copy bytes
        break;
      case 0:
        memcpy(out, reinterpret_cast<uint64_t *>(&h0), rem << 3); // copy bytes
        break;
      default:
        break;
      }
    }
  }


  // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
  // above 2, the finalize and updates will dominate and better to do those in parallel.
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE > 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint64_t *out) const
  {
    // batch of 8
    __m256i h0, h1;
    hash<8>(key, h0, h1);
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      _mm256_stream_si256((__m256i *)out, h0);
      _mm256_stream_si256((__m256i *)(out + 4), h1);
    } else {
      _mm256_storeu_si256((__m256i *)out, h0);
      _mm256_storeu_si256((__m256i *)(out + 4), h1);  
    }
  }
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE == 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint64_t *out) const
  {
    // batch of 16
    __m256i h0, h1, h2, h3;
    hash(key, h0, h1, h2, h3);
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      _mm256_stream_si256((__m256i *)out, h0);
      _mm256_stream_si256((__m256i *)(out + 4), h1);
      _mm256_stream_si256((__m256i *)(out + 8), h2);
      _mm256_stream_si256((__m256i *)(out + 12), h3);
    } else {      
      _mm256_storeu_si256((__m256i *)out, h0);
      _mm256_storeu_si256((__m256i *)(out + 4), h1);
      _mm256_storeu_si256((__m256i *)(out + 8), h2);
      _mm256_storeu_si256((__m256i *)(out + 12), h3);
    }
  }
  template <bool STREAMING = false, size_t KEY_SIZE = sizeof(T), typename ::std::enable_if<(KEY_SIZE == 1), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint64_t *out) const
  {
    // batch of 32
    __m256i h0, h1, h2, h3, h4, h5, h6, h7;
    hash(key, h0, h1, h2, h3, h4, h5, h6, h7);
    if (STREAMING && ((reinterpret_cast<uint64_t>(out) & 31) == 0)) {
      _mm256_stream_si256((__m256i *)out, h0);
      _mm256_stream_si256((__m256i *)(out + 4), h1);
      _mm256_stream_si256((__m256i *)(out + 8), h2);
      _mm256_stream_si256((__m256i *)(out + 12), h3);
      _mm256_stream_si256((__m256i *)(out + 16), h4);
      _mm256_stream_si256((__m256i *)(out + 20), h5);
      _mm256_stream_si256((__m256i *)(out + 24), h6);
      _mm256_stream_si256((__m256i *)(out + 28), h7);
    } else {
      _mm256_storeu_si256((__m256i *)out, h0);
      _mm256_storeu_si256((__m256i *)(out + 4), h1);
      _mm256_storeu_si256((__m256i *)(out + 8), h2);
      _mm256_storeu_si256((__m256i *)(out + 12), h3);
      _mm256_storeu_si256((__m256i *)(out + 16), h4);
      _mm256_storeu_si256((__m256i *)(out + 20), h5);
      _mm256_storeu_si256((__m256i *)(out + 24), h6);
      _mm256_storeu_si256((__m256i *)(out + 28), h7);
    }
  }

  /// NOTE: multiples of 32.
  // USING load, INSERT plus unpack is FASTER than i32gather.
  // load with an offset from start of key.
  template <size_t CNT = 8>
  FSC_FORCE_INLINE void load_stride16(T const *key, size_t const &off,
                                      __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3 //,
                                      // __m256i & t4, __m256i & t5, __m256i & t6, __m256i & t7
                                      ) const
  {

    // faster than gather.

    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    __m256i k0, k1, k2, k3;
    __m128i j0, j1, j2, j3;

//    // load 8 keys at a time, 16 bytes each time,
//    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + off);     // SSE3  // L3 C0.5 p23
//    j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); // SSE3
//    j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + off); // SSE3
//    j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + off); // SSE3
//
//    // get the 32 byte vector.
//    // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
//    // y m i version, L4, C0.5, p015 p23
//    k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), *(reinterpret_cast<const __m128i *>(key + 2) + off), 1); //aa'AA'cc'CC'
//    k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), *(reinterpret_cast<const __m128i *>(key + 3) + off), 1); //bb'BB'dd'DD'
//    k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), *(reinterpret_cast<const __m128i *>(key + 6) + off), 1); //EG
//    k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), *(reinterpret_cast<const __m128i *>(key + 7) + off), 1); //FH
    // load 8 keys at a time, 16 bytes each time,
    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + off);     // SSE3  // L3 C0.5 p23
    if (CNT > 1) j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); else j1 = zeroi128;  // SSE3
    if (CNT > 4) j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + off); else j2 = zeroi128;  // SSE3
    if (CNT > 5) j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + off); else j3 = zeroi128; // SSE3

    // get the 32 byte vector.
    // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
    // y m i version, L4, C0.5, p015 p23
    if (CNT > 2) k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), *(reinterpret_cast<const __m128i *>(key + 2) + off), 1); //aa'AA'cc'CC'
    else k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), zeroi128, 1);

    if (CNT > 3) k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), *(reinterpret_cast<const __m128i *>(key + 3) + off), 1); //bb'BB'dd'DD'
    else if (CNT > 1) k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), zeroi128, 1);
    else k1 = zeros;

    if (CNT > 6) k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), *(reinterpret_cast<const __m128i *>(key + 6) + off), 1); //EG
    else if (CNT > 4) k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), zeroi128, 1);
    else k2 = zeros;

    if (CNT > 7) k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), *(reinterpret_cast<const __m128i *>(key + 7) + off), 1); //FH
    else if (CNT > 5) k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), zeroi128, 1);
    else k3 = zeros;

    // target is abef cdgh,  start with aa'AA', so want AC BD EG FH above.


    // MERGED shuffling and update part 1.
    // now unpack and update
    t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'cdc'd'                           // L1 C1 p5   
    t2 = _mm256_unpackhi_epi32(k0, k1); 
    t1 = _mm256_unpacklo_epi32(k2, k3); // efe'f'ghg'h'
    
    k0 = _mm256_unpacklo_epi64(t0, t1);   // abefcdgh
    k1 = _mm256_unpackhi_epi64(t0, t1);   // a'b'e'f'c'd'g'h'

    t0 = _mm256_mullo_epi32(k0, this->c11); // avx                                  // L10 C2 p0

    t3 = _mm256_unpackhi_epi32(k2, k3);                                            // L1 C1 p5
    
    t1 = _mm256_mullo_epi32(k1, this->c12); // avx  // Lat10, CPI2

    
    k2 = _mm256_unpacklo_epi64(t2, t3);    //ABEFCDGH
    k3 = _mm256_unpackhi_epi64(t2, t3);

    t2 = _mm256_mullo_epi32(k2, this->c13); // avx  // Lat10, CPI2
    t3 = _mm256_mullo_epi32(k3, this->c14); // avx  // Lat10, CPI2

    // latency:  should be Lat3, C2 for temp

    // loading 8 32-byte keys is slower than loading 8 16-byte keys.
  }

  // USING load, INSERT plus unpack is FASTER than i32gather.
  // also probably going to be faster for non-power of 2 less than 8 (for 9 to 15, this is needed anyways).
  //   because we'd need to shift across lane otherwise.
  // load with an offset from start of key, and load partially.  blocks of 16,
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T), size_t off = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15)>
  FSC_FORCE_INLINE void load_partial16(T const *key,
                                       __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3 //,
                                       // __m256i & t4, __m256i & t5, __m256i & t6, __m256i & t7
                                       ) const
  {

    // a lot faster than gather.

    static_assert(rem > 0, "ERROR: should not call load_partial when remainder if 0");

    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    __m256i k0, k1, k2, k3;
    __m128i j0, j1, j2, j3;
    __m256i mask = _mm256_srli_si256(ones, 16 - rem); // shift right to keep just the remainder part

    // load 8 keys at a time, 16 bytes each time,
    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + off);     // SSE3  // L3 C0.5 p23
    if (CNT > 1) j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); else j1 = zeroi128;  // SSE3
    if (CNT > 4) j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + off); else j2 = zeroi128;  // SSE3
    if (CNT > 5) j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + off); else j3 = zeroi128; // SSE3

    // get the 32 byte vector.
    // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
    // y m i version, L4, C0.5, p015 p23
    if (CNT > 2) k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), *(reinterpret_cast<const __m128i *>(key + 2) + off), 1); //aa'AA'cc'CC'
    else k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), zeroi128, 1);

    if (CNT > 3) k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), *(reinterpret_cast<const __m128i *>(key + 3) + off), 1); //bb'BB'dd'DD'
    else if (CNT > 1) k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), zeroi128, 1);
    else k1 = zeros;

    if (CNT > 6) k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), *(reinterpret_cast<const __m128i *>(key + 6) + off), 1); //EG
    else if (CNT > 4) k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), zeroi128, 1);
    else k2 = zeros;

    if (CNT > 7) k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), *(reinterpret_cast<const __m128i *>(key + 7) + off), 1); //FH
    else if (CNT > 5) k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), zeroi128, 1);
    else k3 = zeros;


    // ZERO the leading bytes to keep just the lower.
    // latency of 3 and CPI of 1, so can do the masking here...
    k0 = _mm256_and_si256(k0, mask);
    if (CNT > 1) k1 = _mm256_and_si256(k1, mask);
    if (CNT > 4) k2 = _mm256_and_si256(k2, mask);
    if (CNT > 5) k3 = _mm256_and_si256(k3, mask);
    
    // MERGED shuffling and update part 1.
    // now unpack and update
    // RELY ON COMPILER OPTIMIZATION HERE TO REMOVE THE CONDITIONAL CHECKS
    t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'cdc'd'                          // L1 C1 p5    
    if (rem > 8) t2 = _mm256_unpackhi_epi32(k0, k1);
    t1 = _mm256_unpacklo_epi32(k2, k3); // efe'f' ghg'h'
    
    k0 = _mm256_unpacklo_epi64(t0, t1);   // abefcdgh
    if (rem > 4) k1 = _mm256_unpackhi_epi64(t0, t1);   // a'b'e'f'c'd'g'h'

    t0 = _mm256_mullo_epi32(k0, this->c11); // avx                                  // L10 C2 p0

    if (rem > 8) t3 = _mm256_unpackhi_epi32(k2, k3);                                            // L1 C1 p5

    if (rem > 4) t1 = _mm256_mullo_epi32(k1, this->c12); // avx  // Lat10, CPI2

    
    if (rem > 8) k2 = _mm256_unpacklo_epi64(t2, t3);
    if (rem > 12) k3 = _mm256_unpackhi_epi64(t2, t3);

    if (rem > 8) t2 = _mm256_mullo_epi32(k2, this->c13); // avx  // Lat10, CPI2
    if (rem > 12) t3 = _mm256_mullo_epi32(k3, this->c14); // avx  // Lat10, CPI2
    // now unpack and update

    // latency:  should be Lat3, C2 for temp

  }

  /// NOTE: non-power of 2 length keys ALWAYS use AVX gather, which may be slower.
  // for hasing non multiple of 16 and non power of 2.
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T), size_t nblocks = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15),
            typename std::enable_if<((KEY_LEN & (KEY_LEN - 1)) > 0) && ((KEY_LEN & 15) > 0), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1) const
  {
    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    static_assert(rem > 0, "ERROR remainder is 0.");

    // load 16 bytes at a time.
    __m256i t00, t01, t02, t03;

    __m256i h2, h3;
    // read input, 8 keys at a time.  need 4 rounds.
    h0 = h1 = h2 = h3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));

    // read input, 2 keys per vector.
    // combined load and update_part1 and update_part2 (data parallel part.)
    size_t i = 0;
    for (; i < nblocks; ++i)
    {
      this->template load_stride16<CNT>(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
      // if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
      // if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
      // if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
    
          // update part 2.
      this->template update_part2<4>(t00, t01, t02, t03);

      // now do part 3.
      this->template update_part3<4>(h0, h1, h2, h3, t00, t01, t02, t03);
      // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
      // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
      // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
    }
    
    // latency: h3: L0, C0.  h0: L1,C2



    // DO LAST ADD FROM UPDATE32
    // do remainder.
    if (rem > 0)
    { // NOT multiple of 16.

      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      this->template load_partial16<CNT>(key, t00, t01, t02, t03); // , t04, t05, t06, t07);
      // if (VEC_CNT >= 2)  this->load_partial16(key + 8, t10, t11, t12, t13); // , t14, t15, t16, t17);
      // if (VEC_CNT >= 3)  this->load_partial16(key + 16, t20, t21, t22, t23); // , t24, t25, t26, t27);
      // if (VEC_CNT >= 4)  this->load_partial16(key + 24, t30, t31, t32, t33); // , t34, t35, t36, t37);

      // update part 2.  note that we compute for parts that have non-zero values, determined in blocks of 4 bytes.
      this->template update_part2<((rem + 3) >> 2)>(t00, t01, t02, t03);

      this->template xor4<((rem + 3) >> 2)>(h0, h1, h2, h3, t00, t01, t02, t03);
    }

    // DO LAST ADD FROM UPDATE32
    this->template xor4<4>(h0, h1, h2, h3, length, length, length, length);
    
    this->hmix32(h0, h1, h2, h3);
    
    // Latency: h3: L1 C2, h0:L1 C2
    this->fmix32(h0, h1, h2, h3);
    
    this->hmix32(h0, h1, h2, h3);
    

    // already as abef cdgh
    // need aa'bb'cc'dd'  ee'ff'gg'hh'
    // unpacklo and high means src are
    //     abef cdgh  and a'b'e'f' c'd'g'h'
    //  so permute needs to change from aceg

    t00 = _mm256_unpacklo_epi32(h0, h1);
    h1 = _mm256_unpackhi_epi32(h0, h1);
    h0 = t00;
  }

  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // multiple of 16 that are greater than 16.
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T), size_t nblocks = (sizeof(T) >> 4),
            typename std::enable_if<((KEY_LEN & 15) == 0) && (KEY_LEN > 16), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1) const
  {
    // we now assume no specific layout, so we need to load 8 at a time.

    // load 16 bytes at a time.
    
        // load 16 bytes at a time.
        __m256i t00, t01, t02, t03;
    
            __m256i h2, h3;
        // read input, 8 keys at a time.  need 4 rounds.
        h0 = h1 = h2 = h3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));
    
        // read input, 2 keys per vector.
        // combined load and update_part1 and update_part2 (data parallel part.)
        size_t i = 0;
        for (; i < nblocks; ++i)
        {
          this->template load_stride16<CNT>(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
          // if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
          // if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
          // if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
        
              // update part 2.
          this->template update_part2<4>(t00, t01, t02, t03);
    
          // now do part 3.
          this->template update_part3<4>(h0, h1, h2, h3, t00, t01, t02, t03);
          // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
          // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
          // this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
        }
        
        // latency: h3: L0, C0.  h0: L1,C2
    
    
        // DO LAST ADD FROM UPDATE32
        this->template xor4<4>(h0, h1, h2, h3, length, length, length, length);
        
        this->hmix32(h0, h1, h2, h3);
        
        // Latency: h3: L1 C2, h0:L1 C2
        this->fmix32(h0, h1, h2, h3);
        
        this->hmix32(h0, h1, h2, h3);
        
    
        // already as abef cdgh
        // need aa'bb'cc'dd'  ee'ff'gg'hh'
        // unpacklo and high means src are
        //     abef cdgh  and a'b'e'f' c'd'g'h'
        //  so permute needs to change from aceg
    
        t00 = _mm256_unpacklo_epi32(h0, h1);
        h1 = _mm256_unpackhi_epi32(h0, h1);
        h0 = t00;
  }


  // load 8 16 byte keys, and shuffle (matrix transpose) into 4 8-int vectors.  Also do first multiply of the update32 op.
  template <size_t CNT = 8>
  FSC_FORCE_INLINE void load16(T const *key, __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    __m256i k0, k1, k2, k3;

    // get the data first - 8 keys at a time.  // start as abcd efgh.  want abef cdgh.  doing it here involves 4 lane crossing., do it later, 2.
    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // SSE3  // L3 C0.5 p23
    if (CNT > 2) k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 2)); // SSE3
    else k1 = zeros;
    if (CNT > 4) k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); // SSE3
    else k2 = zeros;
    if (CNT > 6) k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 6)); // SSE3
    else k3 = zeros;

    // next shuffle to group the 32 bit ints together by their ordre in the keys, i.e. get the k1, k2, k3, k4...
    // unpack to get the right set.  require 8 unpack ops
    // start with    aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
    t0 = _mm256_unpacklo_epi32(k0, k1); // aca'c' bdb'd'      
    t1 = _mm256_unpacklo_epi32(k2, k3); // ege'g' fhf'h'
    t2 = _mm256_unpackhi_epi32(k0, k1);  // ACA'C' BDB'D'

    k0 = _mm256_unpacklo_epi64(t0, t1);   // aceg bdfh
    k1 = _mm256_unpackhi_epi64(t0, t1);   // a'c'e'g' b'd'f'h'

    // update32 PART1
    t0 = _mm256_mullo_epi32(k0, this->c11); // avx  // Lat10, CPI2

    t3 = _mm256_unpackhi_epi32(k2, k3); // EGE'G' FHF'H'

    // update32 PART1
    t1 = _mm256_mullo_epi32(k1, this->c12); // avx  // Lat10, CPI2

    k2 = _mm256_unpacklo_epi64(t2, t3);   // ACEG BDFH
    k3 = _mm256_unpackhi_epi64(t2, t3);   // A'C'E'G' B'D'F'H'

    // update32 PART1
    t2 = _mm256_mullo_epi32(k2, this->c13); // avx  // Lat10, CPI2
    t3 = _mm256_mullo_epi32(k3, this->c14); // avx  // Lat10, CPI2

    // latency:  should be Lat3, C2 for temp
  }


  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 16), int>::type = 1> // 16 bytes exactly.
  FSC_FORCE_INLINE void
  hash(T const *key, __m256i &h0, __m256i &h1) const
  {

    // example layout, with each dash representing 4 bytes
    //     aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
    // k0  -- -- -- --
    // k1               -- -- -- --
    // k2                            -- -- -- --
    // k3                                         -- -- -- --

    //__m256i t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;
	__m256i t0, t1, t2, t3, hh2, hh3;

    // read input, 8 keys at a time.  need 4 rounds.

    // read input, 2 keys per vector.
    // combined load and update_part1 and update_part2 (data parallel part.)
    // loaded as t00 = acegbdfh, t01 = a'c'e'g'b'd'f'h', t02 = ACEGBDFH, t03 = A'C'E'G'B'D'F'H'
	  this->template load16<CNT>(key, t0, t1, t2, t3);

    // at this point, part2 of update should be done.
    
    h0 = h1 = hh2 = hh3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));

    // UPDATE part2.
    this->template update_part2<4>(t0, t1, t2, t3);
    
    // now do part 3.
    this->template update_part3<4>(h0, h1, hh2, hh3, t0, t1, t2, t3);

    // latency: h3: L0, C0.  h0: L1,C2

    // DO LAST ADD FROM UPDATE32
    this->template xor4<4>(h0, h1, hh2, hh3, length, length, length, length);

    this->hmix32(h0, h1, hh2, hh3);

    // Latency: h3: L1 C2, h0:L1 C2
    this->fmix32(h0, h1, hh2, hh3);

    this->hmix32(h0, h1, hh2, hh3);


    // finally, do the permutation for h0 and h1, and make 64bit hash values.
    // from aceg bdfh. to abef cdgh .  need [0 4 2 6 1 5 3 7]
    t0 = _mm256_permutevar8x32_epi32(h0, permute16); // L3 C1 p5
    t1 = _mm256_permutevar8x32_epi32(h1, permute16); // L3 C1 p5

    // need aa'bb'cc'dd'  ee'ff'gg'hh'
    // unpacklo and high means src are
    //     abef cdgh  and a'b'e'f' c'd'g'h'
    //  so permute needs to change from aceg

    h0 = _mm256_unpacklo_epi32(t0, t1);
    h1 = _mm256_unpackhi_epi32(t0, t1);
  }



  // this is using a number of different ports with biggest contention over p0.
  // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T),
  typename std::enable_if<(KEY_LEN == 8), int>::type = 1>
  FSC_FORCE_INLINE void load8(T const *key, __m256i &t00, __m256i &t01) const //, __m256i &t10, __m256i &t11) const
  {
    __m256i k0, k1;

    // aAbBcCdD eEfFgGhH
    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // L3 C0.5 p23
    if (CNT > 4) k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4));
    else k1 = zeros;

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // aAbBcCdD eEfFgGhH -> aeAEcgCG bfBFdhDH  with unpacklo/hi 32
    t00 = _mm256_unpacklo_epi32(k0, k1); 
    t01 = _mm256_unpackhi_epi32(k0, k1); 
    // aeAEcgCG bfBFdhDH -> abefcdgh ABEFCDGH, so 2x 
    k0 = _mm256_unpacklo_epi32(t00, t01); 
    k1 = _mm256_unpackhi_epi32(t00, t01); 
    
    t00 = _mm256_mullo_epi32(k0, this->c11); // avx                       // L10, C2, p0
    t01 = _mm256_mullo_epi32(k1, this->c12); // avx  // L10, C2, p0

  }

  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 8), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1) const
  {

    // example layout, with each dash representing 4 bytes
    //     aAbBcCdD eEfFgGhH
    // k0  --------
    // k1           --------

    __m256i t0, t1, t2, t3, hh2, hh3; //, t10, t11, t12, t13;

    // read input, 4 keys per vector.
    // do not use unpacklo and unpackhi - interleave would be aeAEcgCG
    // instead use shuffle + 2 blend + another shuffle.
    // OR: shift, shift, blend, blend
    this->template load8<CNT>(key, t0, t1); //, t10, t11);

    t2 = t3 = zeros;

    // FINISH FIRST MULLO FROM UPDATE32

    // // update with t1
    // h1 = update32(h1, t0); // transpose 4x2  SSE2
    // // update with t0
    // h1 = update32(h1, t1);

    // rotl32 + second mullo of update32.

    this->template update_part2<2>(t0, t1, t2, t3);
    //this->template update_part2<VEC_CNT>(t01, t11, t21, t31);

    h0 = h1 = hh2 = hh3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));    

    // final step of update, xor the length, and fmix32.
    // finalization
    //this->template update_part3<2>(h0, h1, hh2, hh3, t0, t1, t2, t3);
    this->template xor4<2>(h0, h1, hh2, hh3, t0, t1, t2, t3);

    // latency: h3: L0, C0.  h0: L1,C2

    // DO LAST ADD FROM UPDATE32
    this->template xor4<4>(h0, h1, hh2, hh3, length, length, length, length);
    
    this->hmix32(h0, h1, hh2, hh3);
    
    // Latency: h3: L1 C2, h0:L1 C2
    this->fmix32(h0, h1, hh2, hh3);
    
    this->hmix32(h0, h1, hh2, hh3);
    
    
    // finally, do the permutation for h0 and h1, and make 64bit hash values.
    // need aa'bb'cc'dd'  ee'ff'gg'hh'
    // unpacklo and high means src are
    //     abef cdgh  and a'b'e'f' c'd'g'h'
    t0 = _mm256_unpacklo_epi32(h0, h1);
    h1 = _mm256_unpackhi_epi32(h0, h1);
    h0 = t0;

  }


  // this is using a number of different ports with biggest contention over p0.
  // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T),
  typename std::enable_if<(KEY_LEN == 4), int>::type = 1>
  FSC_FORCE_INLINE void load4(T const *key, __m256i &t00) const //, __m256i &t10, __m256i &t20, __m256i &t30) const
  {
    // 16 keys per vector. can potentially do 2 iters.
    t00 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3  L3 C1 p23
    // if (VEC_CNT >= 2) t10 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); // SSE3  L3 C1 p23
    // if (VEC_CNT >= 3) t20 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3  L3 C1 p23
    // if (VEC_CNT >= 4) t30 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24)); // SSE3  L3 C1 p23

    t00 = _mm256_mullo_epi32(t00, this->c11); // AVX  L10 C2 p0
    // if (VEC_CNT >= 2) t10 = _mm256_mullo_epi32(t10, this->c11); // AVX  L10 C2 p0
    // if (VEC_CNT >= 3) t20 = _mm256_mullo_epi32(t20, this->c11); // AVX  L10 C2 p0
    // if (VEC_CNT >= 4) t30 = _mm256_mullo_epi32(t30, this->c11); // AVX  L10 C2 p0
  }

  // hashing 32 elements worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // for 4 byte, testing with 50M, on i7-4770, shows 0.0356, 0.0360, 0.0407, 0.0384 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
  template <size_t CNT = 8, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 4), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1) const //, __m256i &h2, __m256i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 4 bytes
    //     abcd efgh
    // k0  ---- ----
    __m256i t0, t1, t2, t3, hh2, hh3;
    
        // read input, 8 keys at a time.  need 4 rounds.
    
        // read input, 2 keys per vector.
        // combined load and update_part1 and update_part2 (data parallel part.)
        // loaded as t00 = acegbdfh, t01 = a'c'e'g'b'd'f'h', t02 = ACEGBDFH, t03 = A'C'E'G'B'D'F'H'
        this->template load4<8>(key, t0); //, t1, t2, t3);
    
        // at this point, part2 of update should be done.
        t1 = t2 = t3 = zeros;

        this->template update_part2<1>(t0, t1, t2, t3);
    
        h0 = h1 = hh2 = hh3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));

        // now do part 3.
        this->template xor4<1>(h0, h1, hh2, hh3, t0, t1, t2, t3);
    
        // latency: h3: L0, C0.  h0: L1,C2
    
        // DO LAST ADD FROM UPDATE32
        this->template xor4<4>(h0, h1, hh2, hh3, length, length, length, length);
    
        this->hmix32(h0, h1, hh2, hh3);
    
        // Latency: h3: L1 C2, h0:L1 C2
        this->fmix32(h0, h1, hh2, hh3);
    
        this->hmix32(h0, h1, hh2, hh3);
    
    
        // finally, do the permutation for h0 and h1, and make 64bit hash values.
        // from abcd efgh. to abef cdgh .  need [0 2 1 3] = 0xD8
        t0 = _mm256_permute4x64_epi64(h0, 0xD8); // L3 C1 p5
        t1 = _mm256_permute4x64_epi64(h1, 0xD8); // L3 C1 p5
    
        // need aa'bb'cc'dd'  ee'ff'gg'hh'
        // unpacklo and high means src are
        //     abef cdgh  and a'b'e'f' c'd'g'h'
        //  so permute needs to change from aceg
        h0 = _mm256_unpacklo_epi32(t0, t1);
        h1 = _mm256_unpackhi_epi32(t0, t1);

  }

  // this is using a number of different ports with biggest contention over p0.
  // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  template <size_t KEY_LEN = sizeof(T),
    typename std::enable_if<(KEY_LEN == 2), int>::type = 1>
  FSC_FORCE_INLINE void load2(T const *key,
     __m256i &t00, __m256i &t10) const //, __m256i &t20, __m256i &t30) const
  {
    __m256i k0;


    // 16 keys per vector. can potentially do 2 iters.  abcdefgh ijklmnop
    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3
    // permute across lane, 64bits at a time, with pattern 0 2 1 3 -> 11011000 == 0xD8
    //if (VEC_CNT > 2) k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3

    // abcdefgh ijklmnop -> abefijmn cdghklop.  use permute8x32_epi32, with [0 2 4 6 1 3 5 7] (permute1)
    k0 = _mm256_permutevar8x32_epi32(k0, permute1); // AVX2, latency 3, CPI 1
    //if (VEC_CNT > 2)  k1 = _mm256_permute4x64_epi64(k1, 0xd8);                              // AVX2, latency 3, CPI 1

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // want 0a0b0e0f 0c0d0g0h  and 0i0j0m0n 0k0l0o0p.  use unpack lo and hi, epi16, with zeros.
    t00 = _mm256_unpacklo_epi16(k0, zeros); // AVX2, latency 1, CPI 1
    // transform to i0j0k0l0 m0n0o0p0.  interleave with 0.
    t10 = _mm256_unpackhi_epi16(k0, zeros);  // AVX2, latency 1, CPI 1

    
    // if (VEC_CNT > 2)
    // {
    //   // yz12
    //   t20 = _mm256_unpacklo_epi16(k0, zeros); // AVX2, latency 1, CPI 1
    //   t30 = _mm256_unpackhi_epi16(k0, zeros);  // AVX2, latency 1, CPI 1
  
    //   t20 = _mm256_mullo_epi32(t20, this->c11); // avx  // Lat10, CPI2
    //   t30 = _mm256_mullo_epi32(t30, this->c11); // avx  // Lat10, CPI2
    // }
  }


  // hashing 32 elements worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // NOTE: experiment shows that naive loop unrolling, by calling fmix32 and update32_partial directly, and relying on compiler, showed suboptimal improvements.
  // for 2 byte, testing with 50M, on i7-4770, shows 0.0290, 0.0304, 0.0312, 0.0294 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
  template <size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h00, __m256i &h01, __m256i &h10, __m256i &h11) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     abcdefgh ijklmnop
    // k0  -------- --------
    __m256i t00, t10, t2, t3, h2, h3, seed;
    
        // read input, 8 keys at a time.  need 4 rounds.
    
        // read input, 2 keys per vector.
        // combined load and update_part1 and update_part2 (data parallel part.)
        // loaded as t00 = abefcdgh, t02 = ABEFCDGH
        this->load2(key, t00, t10); //, t1, t2, t3);

        t00 = _mm256_mullo_epi32(t00, this->c11); // avx  // Lat10, CPI2
        t10 = _mm256_mullo_epi32(t10, this->c11); // avx  // Lat10, CPI2    

        // at this point, part2 of update should be done.
        t2 = t3 = zeros;

        // do 2 update_part2
        rotl<2>(15, 15, 0, 0, t00, t10, t2, t3);
        mul<2>(this->c12, this->c12, this->c12, this->c12, t00, t10, t2, t3);
    

        h00 = h01 = h10 = h11 = h2 = h3 = seed = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));

        // now do part 3.  here is just xor.
        this->template xor4<2>(h00, h10, h2, h3, t00, t10, t2, t3);
    
        // latency: h3: L0, C0.  h0: L1,C2
        t2 = t3 = seed;
        
        // DO LAST ADD FROM UPDATE32
        this->template xor4<4>(h00, h01, h2, h3, length, length, length, length);
        this->template xor4<4>(h10, h11, t2, t3, length, length, length, length);
        
        this->hmix32(h00, h01, h2, h3);
        this->hmix32(h10, h11, t2, t3);
        
        // Latency: h3: L1 C2, h0:L1 C2
        this->fmix32(h00, h01, h2, h3);
        this->fmix32(h10, h11, t2, t3);
        
        this->hmix32(h00, h01, h2, h3);
        this->hmix32(h10, h11, t2, t3);

        // already in form abefcdgh ijmn klop, so no permute needed

        // finally, make 64bit hash values.
        // need aa'bb'cc'dd'  ee'ff'gg'hh'
        // unpacklo and high means src are
        //     abef cdgh  and a'b'e'f' c'd'g'h'
        //  so permute needs to change from aceg
        t00 = _mm256_unpacklo_epi32(h00, h01);
        h01 = _mm256_unpackhi_epi32(h00, h01);
        h00 = t00;

        t00 = _mm256_unpacklo_epi32(h10, h11);
        h11 = _mm256_unpackhi_epi32(h10, h11);
        h10 = t00;
  }


  // this is using a number of different ports with biggest contention over p0.
  // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  FSC_FORCE_INLINE void load1(T const *key,
    __m256i &t00, __m256i &t10, __m256i &t20, __m256i &t30
          ) const
 {
   __m256i k0;

   // abcdefgh ijklmnop qrstuvwx yz123456
   k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // L3 C0.5 p23

   // convert to abefijmn cdghklop qruvyz34 stwx1256  [0 1 4 5 8 9 12 13 2 3 6 7 10 11 14 15]
   k0 = _mm256_shuffle_epi8(k0, shuffle1_epi8); 
    // need to permute with permutevar4x64, idx = [0 2 1 3] = 0xD8
    k0 = _mm256_permute4x64_epi64(k0, 0xD8); // AVX2,    // L3 C1
    // now have abef ijmn qruv yz34 | cdgh klop stwx 1256

    // use shuffle
    // transform to a000b000e000f000 c000d000g000h000.  interleave with 0.
    t00 = _mm256_shuffle_epi8(k0, shuffle0); // AVX2, latency 1, CPI 1
    // ijmn
    t10 = _mm256_shuffle_epi8(k0, shuffle1); // AVX2, latency 1, CPI 1
    // qruv
    t20 = _mm256_shuffle_epi8(k0, shuffle2); // AVX2, latency 1, CPI 1
    // yz34
    t30 = _mm256_shuffle_epi8(k0, shuffle3); // AVX2, latency 1, CPI 1


 }


  // hashing 32 bytes worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // NOTE: experiment shows that naive loop unrolling, by calling fmix32 and update32_partial directly, and relying on compiler, showed suboptimal improvements.
  // for 1 byte, testing with 50M, on i7-4770, shows 0.0271, 0.0275, 0.0301, 0.0282 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
  template <size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 1), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, 
     __m256i &h00, __m256i &h01,
     __m256i &h10, __m256i &h11, 
     __m256i &h20, __m256i &h21, 
     __m256i &h30, __m256i &h31) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 1 bytes
    //     abcdefghijklmnop qrstuvwxyz123456
    // k0  ---------------- ----------------

    __m256i t00, t10, t20, t30, t2, t3, h2, h3, seed;
    
        // read input, 8 keys at a time.  need 4 rounds.
    
        // read input, 2 keys per vector.
        // combined load and update_part1 and update_part2 (data parallel part.)
        // loaded as t00 = acegbdfh, t01 = a'c'e'g'b'd'f'h', t02 = ACEGBDFH, t03 = A'C'E'G'B'D'F'H'
        load1(key, t00, t10, t20, t30); //, t1, t2, t3);

            // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
        t00 = _mm256_mullo_epi32(t00, this->c11); // avx  // Lat10, CPI2
        t10 = _mm256_mullo_epi32(t10, this->c11);  // avx  // Lat10, CPI2
        t20 = _mm256_mullo_epi32(t20, this->c11);  // avx  // Lat10, CPI2
        t30 = _mm256_mullo_epi32(t30, this->c11);  // avx  // Lat10, CPI2


        // at this point, part2 of update should be done.
        
        // do 2 update_part2
        rotl<4>(15, 15, 15, 15, t00, t10, t20, t30);
        mul<4>(this->c12, this->c12, this->c12, this->c12, t00, t10, t20, t30);
    
        h00 = h10 = h20 = h30 = seed = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(seed_arr));
        
        // now do part 3.  here is just xor.
        this->template xor4<4>(h00, h10, h20, h30, t00, t10, t20, t30);



        h01 = h2 = h3 = seed;
        // latency: h3: L0, C0.  h0: L1,C2
        h11 = t2 = t3 = seed;
        
        // DO LAST ADD FROM UPDATE32
        this->template xor4<4>(h00, h01, h2, h3, length, length, length, length);
        this->template xor4<4>(h10, h11, t2, t3, length, length, length, length);
        
        this->hmix32(h00, h01, h2, h3);
        this->hmix32(h10, h11, t2, t3);
        
        // Latency: h3: L1 C2, h0:L1 C2
        this->fmix32(h00, h01, h2, h3);
        this->fmix32(h10, h11, t2, t3);
        
        this->hmix32(h00, h01, h2, h3);
        this->hmix32(h10, h11, t2, t3);
                    
        // already in abef order.


        // need aa'bb'cc'dd'  ee'ff'gg'hh'
        // unpacklo and high means src are
        //     abef cdgh  and a'b'e'f' c'd'g'h'
        //  so permute needs to change from aceg
        t00 = _mm256_unpacklo_epi32(h00, h01);
        h01 = _mm256_unpackhi_epi32(h00, h01);
        h00 = t00;

        t00 = _mm256_unpacklo_epi32(h10, h11);
        h11 = _mm256_unpackhi_epi32(h10, h11);
        h10 = t00;

        h21 = h2 = h3 = seed;
        // latency: h3: L0, C0.  h0: L1,C2
        h31 = t2 = t3 = seed;
        
        // DO LAST ADD FROM UPDATE32
        this->template xor4<4>(h20, h21, h2, h3, length, length, length, length);
        this->template xor4<4>(h30, h31, t2, t3, length, length, length, length);
        
        this->hmix32(h20, h21, h2, h3);
        this->hmix32(h30, h31, t2, t3);
        
        // Latency: h3: L1 C2, h0:L1 C2
        this->fmix32(h20, h21, h2, h3);
        this->fmix32(h30, h31, t2, t3);
        
        this->hmix32(h20, h21, h2, h3);
        this->hmix32(h30, h31, t2, t3);
                    
        // already in abef order.

        // need aa'bb'cc'dd'  ee'ff'gg'hh'
        // unpacklo and high means src are
        //     abef cdgh  and a'b'e'f' c'd'g'h'
        //  so permute needs to change from aceg
        t00 = _mm256_unpacklo_epi32(h20, h21);
        h21 = _mm256_unpackhi_epi32(h20, h21);
        h20 = t00;

        t00 = _mm256_unpacklo_epi32(h30, h31);
        h31 = _mm256_unpackhi_epi32(h30, h31);
        h30 = t00;
  }
};
template <typename T> const __m256i Murmur64AVX<T>::mix_const1 = _mm256_set1_epi32(0x85ebca6bU);
template <typename T> const __m256i Murmur64AVX<T>::mix_const2 = _mm256_set1_epi32(0xc2b2ae35U);
template <typename T> const __m256i Murmur64AVX<T>::c11 = _mm256_set1_epi32(0x239b961bU);
template <typename T> const __m256i Murmur64AVX<T>::c12 = _mm256_set1_epi32(0xab0e9789U);
template <typename T> const __m256i Murmur64AVX<T>::c13 = _mm256_set1_epi32(0x38b34ae5U);
template <typename T> const __m256i Murmur64AVX<T>::c14 = _mm256_set1_epi32(0xa1e38b93U);
template <typename T> const __m256i Murmur64AVX<T>::c41 = _mm256_set1_epi32(0x561ccd1bU);
template <typename T> const __m256i Murmur64AVX<T>::c42 = _mm256_set1_epi32(0x0bcaa747U);
template <typename T> const __m256i Murmur64AVX<T>::c43 = _mm256_set1_epi32(0x96cd1c35U);
template <typename T> const __m256i Murmur64AVX<T>::c44 = _mm256_set1_epi32(0x32ac3b17U);
template <typename T> const __m256i Murmur64AVX<T>::length = _mm256_set1_epi32(static_cast<uint32_t>(sizeof(T)));
template <typename T> const __m256i Murmur64AVX<T>::permute1 = _mm256_setr_epi32(0U, 2U, 4U, 6U, 1U, 3U, 5U, 7U);
template <typename T> const __m256i Murmur64AVX<T>::permute16 = _mm256_setr_epi32(0U, 4U, 2U, 6U, 1U, 5U, 3U, 7U);
template <typename T> const __m256i Murmur64AVX<T>::shuffle0 = _mm256_setr_epi32(0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U, 0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U);
template <typename T> const __m256i Murmur64AVX<T>::shuffle1 = _mm256_setr_epi32(0x80808004U, 0x80808005U, 0x80808006U, 0x80808007U, 0x80808004U, 0x80808005U, 0x80808006U, 0x80808007U);
template <typename T> const __m256i Murmur64AVX<T>::shuffle2 = _mm256_setr_epi32(0x80808008U, 0x80808009U, 0x8080800AU, 0x8080800BU, 0x80808008U, 0x80808009U, 0x8080800AU, 0x8080800BU);
template <typename T> const __m256i Murmur64AVX<T>::shuffle3 = _mm256_setr_epi32(0x8080800CU, 0x8080800DU, 0x8080800EU, 0x8080800FU, 0x8080800CU, 0x8080800DU, 0x8080800EU, 0x8080800FU);
template <typename T> const __m256i Murmur64AVX<T>::shuffle1_epi8 = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
template <typename T> const __m256i Murmur64AVX<T>::ones = _mm256_cmpeq_epi32(ones, ones);
template <typename T> const __m256i Murmur64AVX<T>::zeros = _mm256_setzero_si256();
template <typename T> const __m128i Murmur64AVX<T>::zeroi128 = _mm_setzero_si128();
template <typename T> constexpr size_t Murmur64AVX<T>::batch_size;

#endif

} // namespace sse


#if defined(__AVX2__)

/**
     * @brief MurmurHash.  using lower 64 bits.
     * @details computing 8 at a time.  Currently, both AVX and SSE are slower than farmhash, but faster than the 32 bit murmur3.
     *    we are not using farmhash as it interferes with prefetching.
     *
     *    prefetch: NTA vs T0 - no difference.
     *              reduce the number of prefetches based on input data type.  no difference.
     *    tried prefetching here, which provides 10 to 15% improvement. however, it might still interfere with prefetching else where.
     *    therefore we are disabling it.
     *
     */
template <typename T>
class murmur3avx64
{
public:
  static constexpr size_t batch_size = ::fsc::hash::sse::Murmur64AVX<T>::batch_size;

protected:
  ::fsc::hash::sse::Murmur64AVX<T> hasher;
//   mutable uint64_t temp[batch_size] __attribute__ ((aligned (32)));

public:
    using result_type = uint64_t;
  using argument_type = T;

  murmur3avx64(uint64_t const &_seed = 43) : hasher(_seed){};

  inline uint64_t operator()(const T &key) const
  {
    uint64_t h;
    hasher.hash(&key, 1, &h);

    return h;
  }

  template <bool STREAMING = false>
  FSC_FORCE_INLINE void operator()(T const *keys, size_t count, uint64_t *results) const
  {
    hash<STREAMING>(keys, count, results);
  }

  // results always 32 bit.
  template <bool STREAMING = false>
  FSC_FORCE_INLINE void hash(T const *keys, size_t count, uint64_t *results) const
  {

    size_t rem = count & (batch_size - 1);
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      hasher.template hash<STREAMING>(&(keys[i]), results + i);
    }

    if (rem > 0)
      hasher.template hash<STREAMING>(&(keys[i]), rem, results + i);
  }

//  // assume consecutive memory layout.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint64_t modulus) const
//  {
//    size_t rem = count & (batch_size - 1);
//    size_t max = count - rem;
//    size_t i = 0, j = 0;
//    for (; i < max; i += batch_size)
//    {
//      hasher.hash(&(keys[i]), temp);
//
//      for (j = 0; j < batch_size; ++j)
//        results[i + j] = temp[j] % modulus;
//    }
//
//    if (rem > 0)
//    {
//      hasher.hash(&(keys[i]), rem, temp);
//
//      for (j = 0; j < rem; ++j)
//        results[i + j] = temp[j] % modulus;
//    }
//  }
//
//  // assume consecutive memory layout.
//  // note that the paremter is modulus bits.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint64_t modulus) const
//  {
//    assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
//    --modulus;
//
//    size_t rem = count & (batch_size - 1);
//    size_t max = count - rem;
//    size_t i = 0, j = 0;
//    for (; i < max; i += batch_size)
//    {
//      hasher.hash(&(keys[i]), temp);
//      for (j = 0; j < batch_size; ++j)
//        results[i + j] = temp[j] & modulus;
//    }
//
//    // last part.
//    if (rem > 0)
//    {
//      hasher.hash(&(keys[i]), rem, temp);
//      for (j = 0; j < rem; ++j)
//        results[i + j] = temp[j] & modulus;
//    }
//  }

  // TODO: [ ] add a transform_hash_mod.
};
template <typename T>
constexpr size_t murmur3avx64<T>::batch_size;

#endif

} // namespace hash

} // namespace fsc

#endif /* MURMUR3_64_AVX_HPP_ */
