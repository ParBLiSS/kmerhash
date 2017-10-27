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
 *
 *NOTE:
 *      avx2 state transition when upper 128 bit may not be zero: STATE C:  up to 6x slower.
 *      	https://software.intel.com/en-us/articles/intel-avx-state-transitions-migrating-sse-code-to-avx
 *
 *      	however, clearing all registers would also clear all stored constants, which would then need to be reloaded.
 *      	this can be done, but will require  some code change.
 *  TODO: [ ] proper AVX state transition, with load.  NOTE: vtunes measurement with benchmark_hashtables does not seem to indicate penalities in canonicalization or hashing.
 *        [ ] tuning to avoid skipped reading - avoid cache set contention.
 *        [ ] try to stick to some small multiples of 64 bytes.
 *        [ ] schedule instructions to maximize usage of ALU and other vector units.
 *        [ ] at 64 bytes per element, we are at limit of 8-way set associative cache (L1)...
 */
#ifndef HASH_HPP_
#define HASH_HPP_

#include <type_traits> // enable_if
#include <cstring>     // memcpy
#include <stdexcept>   // logic error
#include <stdint.h>    // std int strings
#include <iostream>    // cout

#include "utils/filter_utils.hpp"
#include "utils/transform_utils.hpp"
#include "kmerhash/mem_utils.hpp"
#include "kmerhash/math_utils.hpp"

// includ the murmurhash code.
#ifndef _MURMURHASH3_H_
#include <smhasher/MurmurHash3.cpp>
#endif

// and farm hash
#ifndef FARM_HASH_H_
#include <farmhash/src/farmhash.cc>
#endif

#if defined(__AVX2__)

#ifndef INCLUDE_CLHASH_H_
#include <clhash/src/clhash.c>
#endif
#endif

#if defined(_MSC_VER)

#define FSC_FORCE_INLINE __forceinline

// Other compilers

#else // defined(_MSC_VER)

#define FSC_FORCE_INLINE inline __attribute__((always_inline))

#endif // !defined(_MSC_VER)

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
class Murmur32AVX
{

protected:
  // make static so initialization at beginning of class...
  // static constexpr uint32_t BLISS_ALIGNED_ARRAY(mix_const1, 8, 64) = {
  //   0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,
  //   0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,0x85ebca6bU};
  const __m256i mix_const1;
  const __m256i mix_const2;
  const __m256i c1;
  const __m256i c2;
  const __m256i c3;
  const __m256i c4;
  const __m256i length;

  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void update_part2(__m256i & t0, __m256i & t1, __m256i & t2, __m256i & t3) const {
    switch (VEC_CNT) {
      case 4:
          t3 = _mm256_or_si256(_mm256_slli_epi32(t3, 15), _mm256_srli_epi32(t3, 17));
          t3 = _mm256_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
          case 3:
          t2 = _mm256_or_si256(_mm256_slli_epi32(t2, 15), _mm256_srli_epi32(t2, 17));
          t2 = _mm256_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
          case 2:
          t1 = _mm256_or_si256(_mm256_slli_epi32(t1, 15), _mm256_srli_epi32(t1, 17));
          t1 = _mm256_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
          case 1:    
          t0 = _mm256_or_si256(_mm256_slli_epi32(t0, 15), _mm256_srli_epi32(t0, 17));
          t0 = _mm256_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2      
          }
    }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT, bool add_prev_iter>
  FSC_FORCE_INLINE void update_part3(__m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3,
    __m256i const & t0, __m256i const & t1, __m256i const & t2, __m256i const & t3) const {
      switch (VEC_CNT) {
        case 4:
        if (add_prev_iter) h3 = _mm256_add_epi32(h3, this->c4);
        h3 = _mm256_xor_si256(h3, t3); // avx
        h3 = _mm256_or_si256(_mm256_slli_epi32(h3, 13), _mm256_srli_epi32(h3, 19));
        h3 = _mm256_mullo_epi32(h3, this->c3);
      case 3:
      if (add_prev_iter) h2 = _mm256_add_epi32(h2, this->c4);
      h2 = _mm256_xor_si256(h2, t2); // avx
      h2 = _mm256_or_si256(_mm256_slli_epi32(h2, 13), _mm256_srli_epi32(h2, 19));
      h2 = _mm256_mullo_epi32(h2, this->c3);
      case 2:
      if (add_prev_iter) h1 = _mm256_add_epi32(h1, this->c4);
      h1 = _mm256_xor_si256(h1, t1); // avx
      h1 = _mm256_or_si256(_mm256_slli_epi32(h1, 13), _mm256_srli_epi32(h1, 19));
      h1 = _mm256_mullo_epi32(h1, this->c3);
      case 1:
      if (add_prev_iter) h0 = _mm256_add_epi32(h0, this->c4);      
      h0 = _mm256_xor_si256(h0, t0); // avx
      h0 = _mm256_or_si256(_mm256_slli_epi32(h0, 13), _mm256_srli_epi32(h0, 19));
      h0 = _mm256_mullo_epi32(h0, this->c3);
      }
    }

  /// fmix32 for 32 elements at a time.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32(__m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
  {

      // should have 0 idle latency cyles and 0 cpi cycles here.
      switch (VEC_CNT) {
      case 4:
        h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
        h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
      case 3:
        h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
        h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
      case 2:
        h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
        h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
      case 1:
        h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
        h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
      }

      // should have 1 idle latency cyles and 2 cpi cycles here.

      //h1 = fmix32(h1); // ***** SSE4.1 **********
      switch (VEC_CNT) {
      case 4:  		h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
                  h3 = _mm256_mullo_epi32(h3, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 3:			h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
                h2 = _mm256_mullo_epi32(h2, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 2:			h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
                h1 = _mm256_mullo_epi32(h1, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 1:			h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
              h0 = _mm256_mullo_epi32(h0, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      }

      // latencies.
      // h3  Lat 1, cpi 2
      // h0  Lat 4, cpi 2
    
      // expect Lat 0, cycle 1
      switch (VEC_CNT) {
      case 4: h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      case 3: h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      case 2: h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
      case 1: h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
      }

  }

  /// fmix32 for 32 elements at a time.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32_part2(__m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
  {
      // should have 1 idle latency cyles and 2 cpi cycles here.

      //h1 = fmix32(h1); // ***** SSE4.1 **********
      switch (VEC_CNT) {
      case 4:  		h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
                  h3 = _mm256_mullo_epi32(h3, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 3:			h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
                h2 = _mm256_mullo_epi32(h2, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 2:			h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
                h1 = _mm256_mullo_epi32(h1, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      case 1:			h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
              h0 = _mm256_mullo_epi32(h0, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
      }

      // latencies.
      // h3  Lat 1, cpi 2
      // h0  Lat 4, cpi 2

      // expect Lat 0, cycle 1
      switch (VEC_CNT) {
      case 4: h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      case 3: h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      case 2: h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
      case 1: h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
      }

  }
  
  
// LATENCIES for instruction with latency > 1
// 1. mullo_epi16 has latency of 5 cycles, CPI of 1 to 0.5 cycles - need unroll 2x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
// 2. mullo_epi32 has latency of 10 cycles, CPI of 1 to 0.5 cycles - need unroll 3-4x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
// 3. _mm256_permutevar8x32_epi32, _mm256_permute4x64_epi64, and _mm256_insertf128_si256  have latency of 3, CPI of 1.
// 4. note that most extract calls have latency of 3 and CPI of 1, except for _mm256_extracti128_si256, which has latency of 1.
// 5. _mm256_insertf128_si256 has latency of 3. and CPI of 1.
protected:
  // make static so initialization at beginning of class...
  // static constexpr uint32_t BLISS_ALIGNED_ARRAY(mix_const1, 8, 64) = {
  //   0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,
  //   0x85ebca6bU,0x85ebca6bU,0x85ebca6bU,0x85ebca6bU};
  const __m256i permute1;
  const __m256i permute16;
  const __m256i shuffle0; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle1; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle2; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle3; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i ones;
  mutable __m256i seed;
  

//  // input is 8 unsigned ints.
//  FSC_FORCE_INLINE __m256i rotl32(__m256i x, int8_t r) const
//  {
//    // return (x << r) | (x >> (32 - r));
//    return _mm256_or_si256(              // sse2
//        _mm256_slli_epi32(x, r),         // sse2
//        _mm256_srli_epi32(x, (32 - r))); // sse2
//  }
//
//  // unroll 2x to hide latency of 1st and 2nd, 5 times for 3rd
//  FSC_FORCE_INLINE __m256i update32(__m256i h, __m256i k) const
//  {
//    // preprocess the 4 streams
//    k = _mm256_mullo_epi32(k, c1); // SSE2
//    k = rotl32(k, 15);             // sse2
//    k = _mm256_mullo_epi32(k, c2); // SSE2
//    // merge with existing.
//    h = _mm256_xor_si256(h, k); // SSE
//    // this is done per block of 4 bytes.  the last block (smaller than 4 bytes) does not do this.  do for every byte except last,
//    h = rotl32(h, 13);                                   // sse2
//    h = _mm256_add_epi32(_mm256_mullo_epi32(h, c3), c4); // SSE
//    return h;
//  }
//
//  // count cannot be zero.
//  // unroll 1x to hide latency of first, and 5x to hide latency of second mullo.
//  FSC_FORCE_INLINE __m256i update32_zeroing(__m256i h, __m256i k, uint8_t const &count) const
//  {
//    assert((count > 0) && (count < 4) && "count should be between 1 and 3");
//
//    unsigned int shift = (4U - count) * 8U;
//    // clear the upper bytes
//    k = _mm256_srli_epi32(_mm256_slli_epi32(k, shift), shift); // sse2
//
//    // preprocess the 4 streams
//    k = _mm256_mullo_epi32(k, c1); // SSE2
//    k = rotl32(k, 15);             // sse2
//    k = _mm256_mullo_epi32(k, c2); // SSE2
//    // merge with existing.
//    h = _mm256_xor_si256(h, k); // SSE
//    return h;
//  }
//
//  // count cannot be zero.
//  // unroll 2x to hide latency of first, and 5x to hide latency of second mullo.
//  FSC_FORCE_INLINE __m256i update32_partial(__m256i h, __m256i k, uint8_t const &count) const
//  {
//
//    // preprocess the 4 streams
//    k = _mm256_mullo_epi32(k, c1); // SSE2
//    k = rotl32(k, 15);             // sse2
//    k = _mm256_mullo_epi32(k, c2); // SSE2
//    // merge with existing.
//    h = _mm256_xor_si256(h, k); // SSE
//    return h;
//  }
//
//  // input is 4 unsigned ints.
//  // is ((h ^ f) * c) carryless multiplication with (f = h >> d)?
//  FSC_FORCE_INLINE __m256i fmix32(__m256i h) const
//  {
//    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16)); // h ^= h >> 16;      sse2
//    h = _mm256_mullo_epi32(h, mix_const1);             // h *= 0x85ebca6b;   sse4.1
//    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13)); // h ^= h >> 13;      sse2
//    h = _mm256_mullo_epi32(h, mix_const2);             // h *= 0xc2b2ae35;   sse4.1
//    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16)); // h ^= h >> 16;      sse2
//
//    return h;
//  }

  explicit Murmur32AVX(__m256i _seed) : mix_const1(_mm256_set1_epi32(0x85ebca6bU)),
                                        mix_const2(_mm256_set1_epi32(0xc2b2ae35U)),
                                        c1(_mm256_set1_epi32(0xcc9e2d51U)),
                                        c2(_mm256_set1_epi32(0x1b873593U)),
                                        c3(_mm256_set1_epi32(0x5U)),
                                        c4(_mm256_set1_epi32(0xe6546b64U)),
                                        length(_mm256_set1_epi32(static_cast<uint32_t>(sizeof(T)))),
                                        permute1(_mm256_setr_epi32(0U, 2U, 4U, 6U, 1U, 3U, 5U, 7U)),
                                        permute16(_mm256_setr_epi32(0U, 4U, 1U, 5U, 2U, 6U, 3U, 7U)),
										shuffle0(_mm256_setr_epi32(0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U,
																   0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U)),
										shuffle1(_mm256_setr_epi32(0x80808004U, 0x80808005U, 0x80808006U, 0x80808007U,
																   0x80808004U, 0x80808005U, 0x80808006U, 0x80808007U)),
										shuffle2(_mm256_setr_epi32(0x80808008U, 0x80808009U, 0x8080800AU, 0x8080800BU,
																   0x80808008U, 0x80808009U, 0x8080800AU, 0x8080800BU)),
										shuffle3(_mm256_setr_epi32(0x8080800CU, 0x8080800DU, 0x8080800EU, 0x8080800FU,
																   0x8080800CU, 0x8080800DU, 0x8080800EU, 0x8080800FU)),
										ones(_mm256_cmpeq_epi32(length, length)),
//                                        offs(_mm256_setr_epi32(
//                                            static_cast<uint32_t>(0),
//                                            static_cast<uint32_t>(sizeof(T)),
//                                            static_cast<uint32_t>(sizeof(T) * 2),
//                                            static_cast<uint32_t>(sizeof(T) * 3),
//                                            static_cast<uint32_t>(sizeof(T) * 4),
//                                            static_cast<uint32_t>(sizeof(T) * 5),
//                                            static_cast<uint32_t>(sizeof(T) * 6),
//                                            static_cast<uint32_t>(sizeof(T) * 7))),
                                        seed(_seed)

  {
  }

public:
  static constexpr uint8_t batch_size = 32;

  explicit Murmur32AVX(uint32_t _seed) : Murmur32AVX(_mm256_set1_epi32(_seed))
  {
  }

  explicit Murmur32AVX(Murmur32AVX const &other) : Murmur32AVX(other.seed)
  {
  }

  explicit Murmur32AVX(Murmur32AVX &&other) : Murmur32AVX(other.seed)
  {
  }

  Murmur32AVX &operator=(Murmur32AVX const &other)
  {
    seed = other.seed;

    return *this;
  }

  Murmur32AVX &operator=(Murmur32AVX &&other)
  {
    seed = other.seed;
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
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint32_t *out) const
  {
    // process 4 streams at a time.  all should be the same length.
	    // process 4 streams at a time.  all should be the same length.

	    assert((nstreams <= 32) && "maximum number of streams is 31");
	    assert((nstreams > 0) && "minimum number of streams is 1");

	    __m256i h0, h1, h2, h3;

	    // do the full ones first.
	    uint8_t blocks = (nstreams + 7) >> 3;  // divide by 8.
	    switch (blocks) {
	       case 1:
	    	    hash<1>(key, h0, h1, h2, h3); break;
	       case 2:
	    	    hash<2>(key, h0, h1, h2, h3); break;
	       case 3:
	    	    hash<3>(key, h0, h1, h2, h3); break;
	       case 4:
	    	    hash<4>(key, h0, h1, h2, h3); break;
	       default:
	         break;
	     }

	    blocks = nstreams >> 3;
	    switch (blocks) {
	      case 4:
	        _mm256_storeu_si256((__m256i *)(out + 24), h3);
	      case 3:
	        _mm256_storeu_si256((__m256i *)(out + 16), h2);
	      case 2:
	        _mm256_storeu_si256((__m256i *)(out + 8), h1);
	      case 1:
	        _mm256_storeu_si256((__m256i *)out, h0);
	      default:
	        break;
	    }

	    uint8_t rem = nstreams & 7;  // remainder.
	    if (rem > 0) {
			// write remainders
			switch (blocks) {
			  case 3:
				memcpy(out + 24, reinterpret_cast<uint32_t *>(&h3), rem << 2); // copy bytes
				break;
			  case 2:
				memcpy(out + 16, reinterpret_cast<uint32_t *>(&h2), rem << 2); // copy bytes
				break;
			  case 1:
				memcpy(out + 8, reinterpret_cast<uint32_t *>(&h1), rem << 2); // copy bytes
				break;
			  case 0:
				memcpy(out, reinterpret_cast<uint32_t *>(&h0), rem << 2); // copy bytes
				break;
			  default:
				break;

			}
	  }
  }


  // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
  // above 2, the finalize and updates will dominate and better to do those in parallel.
  FSC_FORCE_INLINE void hash(T const *key, uint32_t *out) const
  {
    __m256i h0, h1, h2, h3;
    hash<4>(key, h0, h1, h2, h3);
    _mm256_storeu_si256((__m256i *)(out + 24), h3);
    _mm256_storeu_si256((__m256i *)(out + 16), h2);
    _mm256_storeu_si256((__m256i *)(out + 8), h1);
    _mm256_storeu_si256((__m256i *)out, h0);
  }



  /// NOTE: multiples of 32.
  // USING load, INSERT plus unpack is FASTER than i32gather.
  // load with an offset from start of key.
	  FSC_FORCE_INLINE void load_stride16(T const *key, size_t const & off,
	    __m256i & t0, __m256i & t1, __m256i & t2, __m256i & t3 //,
		// __m256i & t4, __m256i & t5, __m256i & t6, __m256i & t7
		) const {

	  // faster than gather.

	    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
	    // and can pipeline 4 at a time, about 40 cycles?
	    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
	    // while we still have 8 "update"s, the programming cost is becoming costly.
	    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
	    // still need to shuffle more than 4 times.

	    __m256i k0, k1, k2, k3, tt1, tt3;
	    __m128i j0, j1, j2, j3, j4, j5, j6, j7;

	    // load 8 keys at a time, 16 bytes each time,
	    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)     + off); // SSE3
	    j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); // SSE3
	    j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + off); // SSE3
	    j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + off); // SSE3
	    j4 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + off); // SSE3
	    j5 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + off); // SSE3
	    j6 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 6) + off); // SSE3
	    j7 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 7) + off); // SSE3

	      // get the 32 byte vector.
	      // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
	      k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), (j7), 1);  //DH
	      k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), (j6), 1);  //CG
	      k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), (j5), 1);  //bb'BB'ff'FF'
	      k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), (j4), 1);  //aa'AA'ee'EE'

	    // MERGED shuffling and update part 1.
	      // now unpack and update
	      tt3 = _mm256_unpackhi_epi32(k2, k3);
	      tt1 = _mm256_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
	      t2 = _mm256_unpackhi_epi32(k0, k1);

	      k3 = _mm256_unpackhi_epi64(t2, tt3);
	      t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2

	      k2 = _mm256_unpacklo_epi64(t2, tt3);
	      t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2

	      // now unpack and update
	      t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'efe'f'

	      k1 = _mm256_unpackhi_epi64(t0, tt1); // a'b'c'd'e'f'g'h'
	      t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

	      k0 = _mm256_unpacklo_epi64(t0, tt1); // abcdefgh
	      t0 = _mm256_mullo_epi32(k0, this->c1); // avx  // Lat10, CPI2

	      // latency:  should be Lat3, C2 for temp
	    // update part 2.
	    this->template update_part2<4>(t0, t1, t2, t3);


	    // below is slower.
	//        __m256i k0, k1, k2, k3, k4, k5, k6, k7;
	//        __m256i j0, j1, j2, j3, j4, j5, j6, j7;
	//
	//        // load 8 keys at a time, 16 bytes each time,
	//        k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)     + off); // a0a1a2a3 a4a5a6a7
	//        k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 1) + off); // b0b1b2b3 b4b5b6b7
	//        k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 2) + off); // c0c1c2c3 c4c5c6c7
	//        k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 3) + off); // d0d1d2d3 d4d5d6d7
	//        k4 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4) + off); // e0e1e2e3 e4e5e6e7
	//        k5 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 5) + off); // f0f1f2f3 f4f5f6f7
	//        k6 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 6) + off); // g0g1g2g3 g4g5g6g7
	//        k7 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 7) + off); // h0h1h2h3 h4h5h6h7
	//
	//        // MERGED shuffling and update part 1.
	//        // final level of unpacking.  0x20 selects low for both, and 0x31 for high for both, src2 to final high bits.
	//        j0 = _mm256_permute2x128_si256(k0, k4, 0x20);   // a0a1a2a3 e0e1e2e3
	//        j1 = _mm256_permute2x128_si256(k1, k5, 0x20);   // b0b1b2b3 f0f1f2f3
	//        j2 = _mm256_permute2x128_si256(k2, k6, 0x20);   // c0c1c2c3 g0g1g2g3
	//        j3 = _mm256_permute2x128_si256(k3, k7, 0x20);   // d0d1d2d3 h0h1h2h3
	//        j4 = _mm256_permute2x128_si256(k0, k4, 0x31);   // a4a5a6a7 e4e5e6e7
	//        j5 = _mm256_permute2x128_si256(k1, k5, 0x31);   // b4b5b6b7 f4f5f6f7
	//        j6 = _mm256_permute2x128_si256(k2, k6, 0x31);   // c4c5c6c7 g4g5g6g7
	//        j7 = _mm256_permute2x128_si256(k3, k7, 0x31);   // d4d5d6d7 h4h5h6h7
	//
	//        k0 = _mm256_unpacklo_epi32(j0, j1);   // a0b0a1b1 e0f0e1f1
	//        k1 = _mm256_unpackhi_epi32(j0, j1);   // a2b3a2b3 e2f3e2f3
	//        k2 = _mm256_unpacklo_epi32(j2, j3);   // c0d0c1d1 g0h0g1h1
	//        k3 = _mm256_unpackhi_epi32(j2, j3);   // c2d3c2d3 g2h3g2h3
	//        // another level of unpacking
	//        j0 = _mm256_unpacklo_epi64(k0, k2);   // a0b0c0d0 e0f0g0h0
	//        t0 = _mm256_mullo_epi32(j0, this->c1); // avx  // Lat10, CPI2
	//        j1 = _mm256_unpackhi_epi64(k0, k2);   // a1b1c1d1 e1f1g1h1
	//        t1 = _mm256_mullo_epi32(j1, this->c1); // avx  // Lat10, CPI2
	//        j2 = _mm256_unpacklo_epi64(k1, k3);   // a2b2c2d2 e2f2g2h2
	//        t2 = _mm256_mullo_epi32(j2, this->c1); // avx  // Lat10, CPI2
	//        j3 = _mm256_unpackhi_epi64(k1, k3);   // a3b3c3d3 e3f3g3h3
	//        t3 = _mm256_mullo_epi32(j3, this->c1); // avx  // Lat10, CPI2
	//
	//
	//        k4 = _mm256_unpacklo_epi32(j4, j5);   // a4b4a5b5 e4f4e5f5
	//        k5 = _mm256_unpackhi_epi32(j4, j5);   // a6b7a6b7 e6f7e6f7
	//        k6 = _mm256_unpacklo_epi32(j6, j7);   // c4d4c5d5 g4h4g5h5
	//        k7 = _mm256_unpackhi_epi32(j6, j7);   // c6d7c6d7 g6h7g6h7
	//        // another level of unpacking
	//        j4 = _mm256_unpacklo_epi64(k4, k6);   // a4b4c4d4 e4f4g4h4
	//        t4 = _mm256_mullo_epi32(j4, this->c1); // avx  // Lat10, CPI2
	//        j5 = _mm256_unpackhi_epi64(k4, k6);   // a5b5c5d5 e5f5g5h5
	//        t5 = _mm256_mullo_epi32(j5, this->c1); // avx  // Lat10, CPI2
	//        j6 = _mm256_unpacklo_epi64(k5, k7);   // a6b6c6d6 e6f6g6h6
	//        t6 = _mm256_mullo_epi32(j6, this->c1); // avx  // Lat10, CPI2
	//        j7 = _mm256_unpackhi_epi64(k5, k7);   // a7b7c7d7 e7f7g7h7
	//        t7 = _mm256_mullo_epi32(j7, this->c1); // avx  // Lat10, CPI2
	//
	//          // latency:  should be Lat3, C2 for temp
	//        // update part 2.
	//        this->template update_part2<4>(t0, t1, t2, t3);
	//        this->template update_part2<4>(t4, t5, t6, t7);

	  }


  // USING load, INSERT plus unpack is FASTER than i32gather.
  // also probably going to be faster for non-power of 2 less than 8 (for 9 to 15, this is needed anyways).
  //   because we'd need to shift across lane otherwise.
  // load with an offset from start of key, and load partially.  blocks of 16,
  template <size_t KEY_LEN = sizeof(T), size_t offset = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15)>
	  FSC_FORCE_INLINE void load_partial16(T const *key,
	    __m256i & t0, __m256i & t1, __m256i & t2, __m256i & t3 //,
		// __m256i & t4, __m256i & t5, __m256i & t6, __m256i & t7
		) const {

	  // a lot faster than gather.

	  static_assert(rem > 0, "ERROR: should not call load_partial when remainder if 0");

	    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
	    // and can pipeline 4 at a time, about 40 cycles?
	    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
	    // while we still have 8 "update"s, the programming cost is becoming costly.
	    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
	    // still need to shuffle more than 4 times.

	    __m256i k0, k1, k2, k3, tt1, tt3;
	    __m128i j0, j1, j2, j3, j4, j5, j6, j7;
	    __m256i mask = _mm256_srli_si256(ones, 16 - rem);  // shift right to keep just the remainder part

	    // load 8 keys at a time, 16 bytes each time,
	    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)     + offset); // SSE3
	    j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + offset); // SSE3
	    j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + offset); // SSE3
	    j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + offset); // SSE3
	    j4 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + offset); // SSE3
	    j5 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + offset); // SSE3
	    j6 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 6) + offset); // SSE3
	    j7 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 7) + offset); // SSE3

	      // get the 32 byte vector.
	      // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
	      k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), (j7), 1);  //DH
	      k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), (j6), 1);  //CG
	      k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), (j5), 1);  //bb'BB'ff'FF'
	      k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), (j4), 1);  //aa'AA'ee'EE'

	      // ZERO the leading bytes to keep just the lower.
	      // latency of 3 and CPI of 1, so can do the masking here...
	      k3 = _mm256_and_si256(k3, mask);
	      k2 = _mm256_and_si256(k2, mask);
	      k1 = _mm256_and_si256(k1, mask);
	      k0 = _mm256_and_si256(k0, mask);

	    // MERGED shuffling and update part 1.
	      // now unpack and update
	      // RELY ON COMPILER OPTIMIZATION HERE TO REMOVE THE CONDITIONAL CHECKS
	      tt1 = _mm256_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
	      if (rem > 8) {
	    	  tt3 = _mm256_unpackhi_epi32(k2, k3);
	    	  t2 = _mm256_unpackhi_epi32(k0, k1);
	      }

	      if (rem > 12) {
			  k3 = _mm256_unpackhi_epi64(t2, tt3);
			  t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2
	      }

	      if (rem > 8) {
			  k2 = _mm256_unpacklo_epi64(t2, tt3);
			  t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
	      }

	      t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'efe'f'

	      if (rem > 4) {
	    	  k1 = _mm256_unpackhi_epi64(t0, tt1); // a'b'c'd'e'f'g'h'
	    	  t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2
	      }

	      k0 = _mm256_unpacklo_epi64(t0, tt1); // abcdefgh
	      t0 = _mm256_mullo_epi32(k0, this->c1); // avx  // Lat10, CPI2

	      // now unpack and update

	      // latency:  should be Lat3, C2 for temp
	      // update part 2.  note that we compute for parts that have non-zero values, determined in blocks of 4 bytes.
	      this->template update_part2<((rem + 3) >> 2)>(t0, t1, t2, t3);

	  }


  /// NOTE: non-power of 2 length keys ALWAYS use AVX gather, which may be slower.
  // for hasing non multiple of 16 and non power of 2.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T), size_t nblocks = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15),
            typename std::enable_if<((KEY_LEN & (KEY_LEN - 1)) > 0) && ((KEY_LEN & 15) > 0), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
  {
	    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
	    // and can pipeline 4 at a time, about 40 cycles?
	    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
	    // while we still have 8 "update"s, the programming cost is becoming costly.
	    // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
	    // still need to shuffle more than 4 times.

	  static_assert(rem > 0, "ERROR remainder is 0.");

	    // load 16 bytes at a time.
	    __m256i t00, t01, t02, t03, //t04, t05, t06, t07,
	            t10, t11, t12, t13, //t14, t15, t16, t17,
	            t20, t21, t22, t23, //t24, t25, t26, t27,
	            t30, t31, t32, t33 //, t34, t35, t36, t37
				;

	    // read input, 8 keys at a time.  need 4 rounds.
	    h0 = h1 = h2 = h3 = seed;

	    size_t i = 0;
	    for (; i < nblocks; ++i) {

	      // read input, 2 keys per vector.
	      // combined load and update_part1 and update_part2 (data parallel part.)
	      switch (VEC_CNT) {
	        case 4:
	        this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);

	        case 3:
	        this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);

	        case 2:
	        this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);

	        case 1:
	        this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
	      }


	      // now do part 3.
	      if (i == 0)
	        this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
	      else
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
	    }
	      // latency: h3: L0, C0.  h0: L1,C2

	      // DO LAST ADD FROM UPDATE32
	    // do remainder.
	    if (rem > 0) {  // NOT multiple of 16.

		  // read input, 2 keys per vector.
		  // combined load and update_part1 and update_part2 (data parallel part.)
		  switch (VEC_CNT) {
			case 4:
			this->load_partial16(key + 24, t30, t31, t32, t33); // , t34, t35, t36, t37);

			case 3:
			this->load_partial16(key + 16, t20, t21, t22, t23); // , t24, t25, t26, t27);

			case 2:
			this->load_partial16(key + 8, t10, t11, t12, t13); // , t14, t15, t16, t17);

			case 1:
			this->load_partial16(key, t00, t01, t02, t03); // , t04, t05, t06, t07);
		  }
	    }

		  // For the last b < 4 bytes, we do not do full update.
		  if (rem >= 4) {
			  if (i == 0)
				this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
			  else
				this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
		  }
		  if (rem >= 8)  this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
		  if (rem >= 12)  this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);

		  __m256i t0, t1, t2, t3;
		  if ((rem & 3) > 0) {   // has part of an int.
			  //  LAST PART OF UPDATE, which is an xor, then fmix.
			  switch (rem >> 2) {
			  case 0:
				  t0 = t00; t1 = t10; t2 = t20; t3 = t30; break;
			  case 1:
				  t0 = t01; t1 = t11; t2 = t21; t3 = t31; break;
			  case 2:
				  t0 = t02; t1 = t12; t2 = t22; t3 = t32; break;
			  case 3:
				  t0 = t03; t1 = t13; t2 = t23; t3 = t33; break;
			  }
		  }

		  // should have 0 idle latency cyles and 0 cpi cycles here.
		  switch (VEC_CNT) {
		  case 4:
			if (rem >= 4)   // complete the prev update_part3
				h3 = _mm256_add_epi32(h3, this->c4); // avx
			if ((rem & 3) > 0) // has partial int
				h3 = _mm256_xor_si256(h3, t3); // avx
			h3 = _mm256_xor_si256(h3, this->length); // sse
		  h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
		  h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
		  case 3:
				if (rem >= 4)  // complete the prev update_part3
					h2 = _mm256_add_epi32(h2, this->c4); // avx
				if ((rem & 3) > 0) // has partial int
					h2 = _mm256_xor_si256(h2, t2); // avx
				h2 = _mm256_xor_si256(h2, this->length); // sse
		  h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
		  h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
		  case 2:
				if (rem >= 4)  // complete the prev update_part3
					h1 = _mm256_add_epi32(h1, this->c4); // avx
				if ((rem & 3) > 0) // has partial int
					h1 = _mm256_xor_si256(h1, t1); // avx
				h1 = _mm256_xor_si256(h1, this->length); // sse
		  h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
		  h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
		  case 1:
				if (rem >= 4)  // complete the prev update_part3
					h0 = _mm256_add_epi32(h0, this->c4); // avx
				if ((rem & 3) > 0) // has partial int
					h0 = _mm256_xor_si256(h0, t0); // avx
				h0 = _mm256_xor_si256(h0, this->length); // sse
		  h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
		  h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
		  }
	  //    // should have 0 idle latency cyles and 0 cpi cycles here.
	  //

	      // Latency: h3: L1 C2, h0:L1 C2
	      this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);

	  }


	  // hashing 32 elements worth of keys at a time.
	  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
	  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
	  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
	  // multiple of 16 that are greater than 16.
  	  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
	            typename std::enable_if<((KEY_LEN & 15) == 0) && (KEY_LEN > 16), int>::type = 1>
	  FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
	  {
	    // we now assume no specific layout, so we need to load 8 at a time.

	    // load 16 bytes at a time.
	    const int nblocks = KEY_LEN >> 4;


	    __m256i t00, t01, t02, t03, //t04, t05, t06, t07,
	            t10, t11, t12, t13, //t14, t15, t16, t17,
	            t20, t21, t22, t23, //t24, t25, t26, t27,
	            t30, t31, t32, t33 //, t34, t35, t36, t37
				;

	    // read input, 8 keys at a time.  need 4 rounds.
	    h0 = h1 = h2 = h3 = seed;

	    int i = 0;
	    for (; i < nblocks; ++i) {

	      // read input, 2 keys per vector.
	      // combined load and update_part1 and update_part2 (data parallel part.)
	      switch (VEC_CNT) {
	        case 4:
	        this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);

	        case 3:
	        this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);

	        case 2:
	        this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);

	        case 1:
	        this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
	      }


	      // now do part 3.   ORDER MATTERS. from low to high.
	      if (i == 0)
	        this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
	      else
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
	        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
	//        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t04, t14, t24, t34);
	//		this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t05, t15, t25, t35);
	//		this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t06, t16, t26, t36);
	//		this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t07, t17, t27, t37);
	    }
	      // latency: h3: L0, C0.  h0: L1,C2

	      // DO LAST ADD FROM UPDATE32

//	      // permute here, since there is an excess of instructions here.
//	      // permute from aceg bdfh to a'b'c'd' e'f'g'h'.  crosses lane here.
//	      // [ 0 4 1 5 2 6 3 7 ], which is permute16
//	      switch (VEC_CNT) {
//	      case 4:
//	        h3 = _mm256_add_epi32(h3, this->c4);
//	        h3 = _mm256_xor_si256(h3, this->length); // sse
//	      //  h3 = _mm256_permutevar8x32_epi32(h3, permute16); // latency 3, CPI 1
//	      case 3:
//	        h2 = _mm256_add_epi32(h2, this->c4);
//	        h2 = _mm256_xor_si256(h2, this->length); // sse
//	      //  h2 = _mm256_permutevar8x32_epi32(h2, permute16); // latency 3, CPI 1
//	      case 2:
//	        h1 = _mm256_add_epi32(h1, this->c4);
//	        h1 = _mm256_xor_si256(h1, this->length); // sse
//	      //  h1 = _mm256_permutevar8x32_epi32(h1, permute16); // latency 3, CPI 1
//	      case 1:
//	        h0 = _mm256_add_epi32(h0, this->c4);
//	        h0 = _mm256_xor_si256(h0, this->length); // sse
//	      //  h0 = _mm256_permutevar8x32_epi32(h0, permute16); // latency 3, CPI 1
//	      }
//
//	      // Latency: h3: L1 C2, h0:L1 C2
//	      this->template fmix32<VEC_CNT>(h0, h1, h2, h3);

	      // should have 0 idle latency cyles and 0 cpi cycles here.
	      switch (VEC_CNT) {
	      case 4:
	      h3 = _mm256_add_epi32(h3, this->c4);
	      h3 = _mm256_xor_si256(h3, this->length); // sse
	      h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
	      h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
	      case 3:
	      h2 = _mm256_add_epi32(h2, this->c4);
	      h2 = _mm256_xor_si256(h2, this->length); // sse
	      h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
	      h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
	      case 2:
	      h1 = _mm256_add_epi32(h1, this->c4);
	      h1 = _mm256_xor_si256(h1, this->length); // sse
	      h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
	      h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
	      case 1:
	      h0 = _mm256_add_epi32(h0, this->c4);
	      h0 = _mm256_xor_si256(h0, this->length); // sse
	      h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
	      h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
	      }
	  //    // should have 0 idle latency cyles and 0 cpi cycles here.
	  //
	      this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);

	  }


   FSC_FORCE_INLINE void load16(T const *key, __m256i & t0, __m256i & t1, __m256i & t2, __m256i & t3) const {
     __m256i k0, k1, k2, k3, tt1, tt3;

     k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // SSE3
     k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 2)); // SSE3
     k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); // SSE3
     k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 6)); // SSE3


     // MERGED shuffling and update part 1.
     // unpack to get the right set.  require 8 unpack ops
     // start.    aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
     t2 = _mm256_unpackhi_epi32(k0, k1);   // ACA'C' BDB'D'
     tt3 = _mm256_unpackhi_epi32(k2, k3);   // EGE'G' FHF'H'
     tt1 = _mm256_unpacklo_epi32(k2, k3);   // ege'g' fhf'h'

     // one more time.
     k3 = _mm256_unpackhi_epi64(t2, tt3);   // A'C'E'G' B'D'F'H'
     t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2

     k2 = _mm256_unpacklo_epi64(t2, tt3);   // ACEG BDFH
     t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2

     t0 = _mm256_unpacklo_epi32(k0, k1);   // aca'c' bdb'd'

     k1 = _mm256_unpackhi_epi64(t0, tt1);   // a'c'e'g' b'd'f'h'
     t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

     k0 = _mm256_unpacklo_epi64(t0, tt1);   // aceg bdfh
     t0 = _mm256_mullo_epi32(k0, this->c1); // avx  // Lat10, CPI2


     // latency:  should be Lat3, C2 for temp
     // update part 2.
     this->template update_part2<4>(t0, t1, t2, t3);
   }


   // hashing 32 elements worth of keys at a time.
   // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
   // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
   // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
   template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
             typename std::enable_if<(KEY_LEN == 16), int>::type = 1>  // 16 bytes exactly.
   FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
   {

     // example layout, with each dash representing 4 bytes
     //     aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
     // k0  -- -- -- --
     // k1               -- -- -- --
     // k2                            -- -- -- --
     // k3                                         -- -- -- --


     __m256i t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;

     // read input, 8 keys at a time.  need 4 rounds.


     // read input, 2 keys per vector.
     // combined load and update_part1 and update_part2 (data parallel part.)
     switch (VEC_CNT) {
       case 4:
       this->load16(key + 24, t30, t31, t32, t33);

       case 3:
       this->load16(key + 16, t20, t21, t22, t23);

       case 2:
       this->load16(key + 8, t10, t11, t12, t13);

       case 1:
       this->load16(key, t00, t01, t02, t03);
     }

     h0 = h1 = h2 = h3 = seed;

     // now do part 3.
     this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
     this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
     this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
     this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);

       // latency: h3: L0, C0.  h0: L1,C2

       // DO LAST ADD FROM UPDATE32

       // permute here, since there is an excess of instructions here.
       // permute from aceg bdfh to a'b'c'd' e'f'g'h'.  crosses lane here.
       // [ 0 4 1 5 2 6 3 7 ], which is permute16
       switch (VEC_CNT) {
       case 4:
         h3 = _mm256_add_epi32(h3, this->c4);
         h3 = _mm256_xor_si256(h3, this->length); // sse
         h3 = _mm256_permutevar8x32_epi32(h3, permute16); // latency 3, CPI 1
       case 3:
         h2 = _mm256_add_epi32(h2, this->c4);
         h2 = _mm256_xor_si256(h2, this->length); // sse
         h2 = _mm256_permutevar8x32_epi32(h2, permute16); // latency 3, CPI 1
       case 2:
         h1 = _mm256_add_epi32(h1, this->c4);
         h1 = _mm256_xor_si256(h1, this->length); // sse
         h1 = _mm256_permutevar8x32_epi32(h1, permute16); // latency 3, CPI 1
       case 1:
         h0 = _mm256_add_epi32(h0, this->c4);
         h0 = _mm256_xor_si256(h0, this->length); // sse
         h0 = _mm256_permutevar8x32_epi32(h0, permute16); // latency 3, CPI 1
       }

       // Latency: h3: L1 C2, h0:L1 C2
       this->template fmix32<VEC_CNT>(h0, h1, h2, h3);

   }

   FSC_FORCE_INLINE void load8(T const *key, __m256i & t0, __m256i & t1) const {
     __m256i k0, k1;

     k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // SSE3
     k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); // SSE3

   	// MERGED SHUFFLE AND UPDATE_PARTIAL
     // make aebfcgdh and AEBFCGDH .  Order matters.  do lower first.
     t0 = _mm256_slli_si256(k1, 4);  // Lat1, Cpi1.  0eEf0gGh
     t0 = _mm256_blend_epi32(k0, t0, 0xAA);  // Lat1, cpi 0.3.
     t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2

     t1 = _mm256_srli_si256(k0, 4);  // Lat1, Cpi1.  AbB0CdD0
     // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
     t1 = _mm256_blend_epi32(t1, k1, 0xAA);  // Lat1, cpi 0.3.  // AEBF
     t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2

   }


   // hashing 32 elements worth of keys at a time.
   // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
   // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
   // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
   template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
             typename std::enable_if<(KEY_LEN == 8), int>::type = 1>
   FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
   {

     // example layout, with each dash representing 4 bytes
     //     aAbBcCdD eEfFgGhH
     // k0  --------
     // k1           --------

     __m256i t00, t01, t10, t11, t20, t21, t30, t31;

     // read input, 4 keys per vector.
     // do not use unpacklo and unpackhi - interleave would be aeAEcgCG
     // instead use shuffle + 2 blend + another shuffle.
     // OR: shift, shift, blend, blend
     switch (VEC_CNT) {
       case 4:
       // k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24));     // SSE3
       // k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 28)); // SSE3

       // t30 = _mm256_slli_si256(k1, 4);  // Lat1, Cpi1.  0eEf0gGh
       // t31 = _mm256_srli_si256(k0, 4);  // Lat1, Cpi1.  AbB0CdD0
       // // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
       // t30 = _mm256_blend_epi32(k0, t30, 0xAA);  // Lat1, cpi 0.3.
       // t31 = _mm256_blend_epi32(t31, k1, 0xAA);  // Lat1, cpi 0.3.  // AEBF

       // t30 = _mm256_mullo_epi32(t30, this->c1); // avx  // Lat10, CPI2
       // t31 = _mm256_mullo_epi32(t31, this->c1); // avx  // Lat10, CPI2
       load8(key + 24, t30, t31);
       case 3:
     // k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16));     // SSE3
     // k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 20)); // SSE3

     // t20 = _mm256_slli_si256(k1, 4);  // Lat1, Cpi1.  0eEf0gGh
     // t21 = _mm256_srli_si256(k0, 4);  // Lat1, Cpi1.  AbB0CdD0
     // // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
     // t20 = _mm256_blend_epi32(k0, t20, 0xAA);  // Lat1, cpi 0.3.
     // t21 = _mm256_blend_epi32(t21, k1, 0xAA);  // Lat1, cpi 0.3.  // AEBF

     // t20 = _mm256_mullo_epi32(t20, this->c1); // avx  // Lat10, CPI2
     // t21 = _mm256_mullo_epi32(t21, this->c1); // avx  // Lat10, CPI2
     load8(key + 16, t20, t21);

     case 2:
     // k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8));     // SSE3
     // k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 12)); // SSE3

     // t10 = _mm256_slli_si256(k1, 4);  // Lat1, Cpi1.  0eEf0gGh
     // t11 = _mm256_srli_si256(k0, 4);  // Lat1, Cpi1.  AbB0CdD0
     // // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
     // t10 = _mm256_blend_epi32(k0, t10, 0xAA);  // Lat1, cpi 0.3.
     // t11 = _mm256_blend_epi32(t11, k1, 0xAA);  // Lat1, cpi 0.3.  // AEBF

     // t10 = _mm256_mullo_epi32(t10, this->c1); // avx  // Lat10, CPI2
     // t11 = _mm256_mullo_epi32(t11, this->c1); // avx  // Lat10, CPI2
     load8(key + 8, t10, t11);


     case 1:
     // k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // SSE3
     // k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); // SSE3

   	// // MERGED SHUFFLE AND UPDATE_PARTIAL
     // // make aebfcgdh and AEBFCGDH
     // t00 = _mm256_slli_si256(k1, 4);  // Lat1, Cpi1.  0eEf0gGh
     // t00 = _mm256_blend_epi32(k0, t00, 0xAA);  // Lat1, cpi 0.3.
     // t00 = _mm256_mullo_epi32(t00, this->c1); // avx  // Lat10, CPI2

     // t01 = _mm256_srli_si256(k0, 4);  // Lat1, Cpi1.  AbB0CdD0
     // // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
     // t01 = _mm256_blend_epi32(t01, k1, 0xAA);  // Lat1, cpi 0.3.  // AEBF
     // t01 = _mm256_mullo_epi32(t01, this->c1); // avx  // Lat10, CPI2

     load8(key, t00, t01);
   }

   // FINISH FIRST MULLO FROM UPDATE32

     // // update with t1
     // h1 = update32(h1, t0); // transpose 4x2  SSE2
     // // update with t0
     // h1 = update32(h1, t1);

     // rotl32 + second mullo of update32.

     this->template update_part2<VEC_CNT>(t00, t10, t20, t30);
     this->template update_part2<VEC_CNT>(t01, t11, t21, t31);


     h0 = h1 = h2 = h3 = seed;

     // final step of update, xor the length, and fmix32.
     // finalization
     this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);

     this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);

       // latency: h3: L0, C0.  h0: L1,C2

       // DO LAST ADD FROM UPDATE32

       // permute here, since there is an excess of instructions here.
       // permute from aebfcgdh to a'b'c'd' e'f'g'h'.  crosses lane here.
       // [ 0 2 4 6 1 3 5 7 ], which is permute1
       switch (VEC_CNT) {
       case 4:
         h3 = _mm256_add_epi32(h3, this->c4);
         h3 = _mm256_xor_si256(h3, this->length); // sse
         h3 = _mm256_permutevar8x32_epi32(h3, permute1); // latency 3, CPI 1
       case 3:
         h2 = _mm256_add_epi32(h2, this->c4);
         h2 = _mm256_xor_si256(h2, this->length); // sse
         h2 = _mm256_permutevar8x32_epi32(h2, permute1); // latency 3, CPI 1
       case 2:
         h1 = _mm256_add_epi32(h1, this->c4);
         h1 = _mm256_xor_si256(h1, this->length); // sse
         h1 = _mm256_permutevar8x32_epi32(h1, permute1); // latency 3, CPI 1
       case 1:
         h0 = _mm256_add_epi32(h0, this->c4);
         h0 = _mm256_xor_si256(h0, this->length); // sse
         h0 = _mm256_permutevar8x32_epi32(h0, permute1); // latency 3, CPI 1
       }

       // Latency: h3: L1 C2, h0:L1 C2
       this->template fmix32<VEC_CNT>(h0, h1, h2, h3);

   }

  // hashing 32 elements worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // for 4 byte, testing with 50M, on i7-4770, shows 0.0356, 0.0360, 0.0407, 0.0384 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots. 
   template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
             typename std::enable_if<(KEY_LEN == 4), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

	    // example layout, with each dash representing 4 bytes
	    //     abcd efgh
	    // k0  ---- ----

    __m256i t0, t1, t2, t3;

	// MERGED SHUFFLE AND UPDATE_PARTIAL
	// h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2

	// 16 keys per vector. can potentially do 2 iters.
    switch (VEC_CNT) {
    case 4:  		t3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24)); // SSE3
        	 	 	t3 = _mm256_mullo_epi32(t3, this->c1); // avx  // Lat10, CPI2
    case 3:			t2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3
        			t2 = _mm256_mullo_epi32(t2, this->c1); // avx  // Lat10, CPI2
    case 2:			t1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); // SSE3
        			t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2
    case 1:			t0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3
    				t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2
    }


    // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
    h0 = h1 = h2 = h3 = seed;

    // rotl32
    //t0 = rotl32(t0, 15);
    this->template update_part2<VEC_CNT>(t0, t1, t2, t3);
    // merge with existing.

    // should have 0 idle latency cyles and 0 cpi cycles here.

    // final step of update, xor the length, and fmix32.
    // finalization
    this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t0, t1, t2, t3);

    // should have 0 idle latency cyles and 0 cpi cycles here.
    switch (VEC_CNT) {
    case 4:
    h3 = _mm256_add_epi32(h3, this->c4);
    h3 = _mm256_xor_si256(h3, this->length); // sse
    h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
    case 3:
    h2 = _mm256_add_epi32(h2, this->c4);
    h2 = _mm256_xor_si256(h2, this->length); // sse
    h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
    case 2:
    h1 = _mm256_add_epi32(h1, this->c4);
    h1 = _mm256_xor_si256(h1, this->length); // sse
    h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
    case 1:
    h0 = _mm256_add_epi32(h0, this->c4);
    h0 = _mm256_xor_si256(h0, this->length); // sse
    h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
    }
//    // should have 0 idle latency cyles and 0 cpi cycles here.
//
    this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);
  }

   // hashing 32 elements worth of keys at a time.  uses 10 to 11 registers.
   // if we go to 64 bytes, then we'd be using 20 to 21 registers
   // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
   // NOTE: experiment shows that naive loop unrolling, by calling fmix32 and update32_partial directly, and relying on compiler, showed suboptimal improvements.
   // for 2 byte, testing with 50M, on i7-4770, shows 0.0290, 0.0304, 0.0312, 0.0294 secs for
   //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
   //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
   template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
             typename std::enable_if<(KEY_LEN == 2), int>::type = 1>
   FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
   {
     // process 32 streams at a time, each 1 byte.  all should be the same length.

       // example layout, with each dash representing 2 bytes
       //     abcdefgh ijklmnop
       // k0  -------- --------

     __m256i k0, k1;
     __m256i t0, t1, t2, t3;

     if (VEC_CNT > 2) {
      k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3
      k1 = _mm256_permute4x64_epi64(k1, 0xd8); // AVX2, latency 3, CPI 1
     }

     // 16 keys per vector. can potentially do 2 iters.
     k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3
     // permute across lane, 64bits at a time, with pattern 0 2 1 3 -> 11011000 == 0xD8
     k0 = _mm256_permute4x64_epi64(k0, 0xd8); // AVX2, latency 3, CPI 1

      // result abcd ijkl efgh mnop
     __m256i zero = _mm256_setzero_si256();


     // MERGED SHUFFLE AND UPDATE_PARTIAL
     // transform to a0b0c0d0 e0f0g0h0.  interleave with 0.
     if (VEC_CNT > 2) {
 		// yz12
 		t3 = _mm256_unpackhi_epi16(k1, zero); // AVX2, latency 1, CPI 1
 		t3 = _mm256_mullo_epi32(t3, this->c1); // avx  // Lat10, CPI2

 		t2 = _mm256_unpacklo_epi16(k1, zero); // AVX2, latency 1, CPI 1
 		t2 = _mm256_mullo_epi32(t2, this->c1); // avx  // Lat10, CPI2
     }
     // ijkl
     t1 = _mm256_unpackhi_epi16(k0, zero); // AVX2, latency 1, CPI 1
     t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2

     t0 = _mm256_unpacklo_epi16(k0, zero); // AVX2, latency 1, CPI 1
     // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
     t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2
     // qrst
     // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
     h0 = h1 = h2 = h3 = seed;

     // rotl32
     //t0 = rotl32(t0, 15);
     if (VEC_CNT > 2) {
 		t3 = _mm256_or_si256(_mm256_slli_epi32(t3, 15), _mm256_srli_epi32(t3, 17));
 		t3 = _mm256_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
 		t2 = _mm256_or_si256(_mm256_slli_epi32(t2, 15), _mm256_srli_epi32(t2, 17));
 		t2 = _mm256_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
     }
     t1 = _mm256_or_si256(_mm256_slli_epi32(t1, 15), _mm256_srli_epi32(t1, 17));
     t1 = _mm256_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
     t0 = _mm256_or_si256(_mm256_slli_epi32(t0, 15), _mm256_srli_epi32(t0, 17));
     t0 = _mm256_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2
     // merge with existing.
 //    this->template update_part2<VEC_CNT>(t0, t1, t2, t3);

     // should have 0 idle latency cyles and 0 cpi cycles here.

     // final step of update, xor the length, and fmix32.
     // finalization
     if (VEC_CNT > 2) {
 		h3 = _mm256_xor_si256(h3, t3); // avx
 		h3 = _mm256_xor_si256(h3, this->length); // sse
 		h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
 		h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2

 		h2 = _mm256_xor_si256(h2, t2); // avx
 		h2 = _mm256_xor_si256(h2, this->length); // sse
 		h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
 		h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2
     }
     h1 = _mm256_xor_si256(h1, t1); // avx
     h1 = _mm256_xor_si256(h1, this->length); // sse
     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
     h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2

     h0 = _mm256_xor_si256(h0, t0); // avx
     h0 = _mm256_xor_si256(h0, this->length); // sse
     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
     h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2


     // should have 0 idle latency cyles and 0 cpi cycles here.

     //h1 = fmix32(h1); // ***** SSE4.1 **********
     if (VEC_CNT > 2) {
 		h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
 		h3 = _mm256_mullo_epi32(h3, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

 		h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
 		h2 = _mm256_mullo_epi32(h2, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
     }
     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
     h1 = _mm256_mullo_epi32(h1, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
     h0 = _mm256_mullo_epi32(h0, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

     // latencies.
     // h3  Lat 4, cpi 2

     if (VEC_CNT > 2) {
     	h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
     	h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
     }
     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
 //    this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);

   }

   // hashing 32 bytes worth of keys at a time.  uses 10 to 11 registers.
   // if we go to 64 bytes, then we'd be using 20 to 21 registers
   // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
   // NOTE: experiment shows that naive loop unrolling, by calling fmix32 and update32_partial directly, and relying on compiler, showed suboptimal improvements.
   // for 1 byte, testing with 50M, on i7-4770, shows 0.0271, 0.0275, 0.0301, 0.0282 secs for
   //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
   //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
   template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
             typename std::enable_if<(KEY_LEN == 1), int>::type = 1>
   FSC_FORCE_INLINE void hash(T const *key, __m256i & h0, __m256i & h1, __m256i & h2, __m256i & h3 ) const
   {
     // process 32 streams at a time, each 1 byte.  all should be the same length.

     // example layout, with each dash representing 1 bytes
     //     abcdefghijklmnop qrstuvwxyz123456
     // k0  ---------------- ----------------

     __m256i k0, t0, t1, t2, t3;

     // 32 keys per vector, can potentially do 4 rounds.
     k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3

     // 2 extra calls.
     // need to permute with permutevar8x32, idx = [0 2 4 6 1 3 5 7]
     k0 = _mm256_permutevar8x32_epi32(k0, permute1); // AVX2, latency of 3, CPI 1
     // abcd ijkl qrst yz12 efgh mnop uvwx 3456


     // MERGED SHUFFLE AND UPDATE_PARTIAL
     // USE shuffle_epi8, with mask.
     // yz12
     t3 = _mm256_shuffle_epi8(k0, shuffle3); // AVX2, latency 1, CPI 1
     t3 = _mm256_mullo_epi32(t3, this->c1); // avx  // Lat10, CPI2
     // qrst
     t2 = _mm256_shuffle_epi8(k0, shuffle2); // AVX2, latency 1, CPI 1
     t2 = _mm256_mullo_epi32(t2, this->c1); // avx  // Lat10, CPI2
     // ijkl
     t1 = _mm256_shuffle_epi8(k0, shuffle1); // AVX2, latency 1, CPI 1
     t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2
     // transform to a000b000c000d000 e000f000g000h000.  interleave with 0.
     t0 = _mm256_shuffle_epi8(k0, shuffle0); // AVX2, latency 1, CPI 1
     // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
     t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2

     // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
     h0 = h1 = h2 = h3 = seed;

     // rotl32
     //t0 = rotl32(t0, 15);
     t3 = _mm256_or_si256(_mm256_slli_epi32(t3, 15), _mm256_srli_epi32(t3, 17));
     t3 = _mm256_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
     t2 = _mm256_or_si256(_mm256_slli_epi32(t2, 15), _mm256_srli_epi32(t2, 17));
     t2 = _mm256_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
     t1 = _mm256_or_si256(_mm256_slli_epi32(t1, 15), _mm256_srli_epi32(t1, 17));
     t1 = _mm256_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
     t0 = _mm256_or_si256(_mm256_slli_epi32(t0, 15), _mm256_srli_epi32(t0, 17));
     t0 = _mm256_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2
     // merge with existing.
 //    this->template update_part2<4>(t0, t1, t2, t3);

     // should have 0 idle latency cyles and 0 cpi cycles here.

     // final step of update, xor the length, and fmix32.
     // finalization

     h3 = _mm256_xor_si256(h3, t3); // avx
     h3 = _mm256_xor_si256(h3, this->length); // sse
     h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
     h3 = _mm256_mullo_epi32(h3, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2

     h2 = _mm256_xor_si256(h2, t2); // avx
     h2 = _mm256_xor_si256(h2, this->length); // sse
     h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
     h2 = _mm256_mullo_epi32(h2, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2

     h1 = _mm256_xor_si256(h1, t1); // avx
     h1 = _mm256_xor_si256(h1, this->length); // sse
     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
     h1 = _mm256_mullo_epi32(h1, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2

     h0 = _mm256_xor_si256(h0, t0); // avx
     h0 = _mm256_xor_si256(h0, this->length); // sse
     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
     h0 = _mm256_mullo_epi32(h0, this->mix_const1);             // h *= 0x85ebca6b;   Lat10, CPI2


     // should have 0 idle latency cyles and 0 cpi cycles here.

 //    //h1 = fmix32(h1); // ***** SSE4.1 **********
     h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
     h3 = _mm256_mullo_epi32(h3, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

     h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
     h2 = _mm256_mullo_epi32(h2, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
     h1 = _mm256_mullo_epi32(h1, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2

     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
     h0 = _mm256_mullo_epi32(h0, this->mix_const2);             // h *= 0xc2b2ae35;   Lat10, CPI2
 //
 //    // latencies.
 //    // h3  Lat 4, cpi 2
 //
     h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
     h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
     h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
     h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
 //    this->template fmix32_part2<4>(h0, h1, h2, h3);

   }

};
template <typename T>
constexpr uint8_t Murmur32AVX<T>::batch_size;

#endif




#if defined(__SSE4_1__)
// for 32 bit buckets
// original: body: 16 inst per iter of 4 bytes; tail: 15 instr. ; finalization:  8 instr.
// about 4 inst per byte + 8, for each hash value.
template <typename T, size_t bytes = sizeof(T)>
class Murmur32SSE
{

protected:
  // make static so initialization at beginning of class...
  mutable __m128i seed;
  const __m128i mix_const1;
  const __m128i mix_const2;
  const __m128i c1;
  const __m128i c2;
  const __m128i c3;
  const __m128i c4;
  const __m128i zero;
  const __m128i length;

  // input is 4 unsigned ints.
  FSC_FORCE_INLINE __m128i rotl32(__m128i x, int8_t r) const
  {
    // return (x << r) | (x >> (32 - r));
    return _mm_or_si128(              // sse2
        _mm_slli_epi32(x, r),         // sse2
        _mm_srli_epi32(x, (32 - r))); // sse2
  }

  FSC_FORCE_INLINE __m128i update32(__m128i h, __m128i k) const
  {
    // preprocess the 4 streams
    k = _mm_mullo_epi32(k, c1); // SSE2
    k = rotl32(k, 15);          // sse2
    k = _mm_mullo_epi32(k, c2); // SSE2
    // merge with existing.
    h = _mm_xor_si128(h, k); // SSE
    // this is done per block of 4 bytes.  the last block (smaller than 4 bytes) does not do this.  do for every byte except last,
    h = rotl32(h, 13);                             // sse2
    h = _mm_add_epi32(_mm_mullo_epi32(h, c3), c4); // SSE
    return h;
  }

  // count cannot be zero.
  FSC_FORCE_INLINE __m128i update32_zeroing(__m128i h, __m128i k, uint8_t const &count) const
  {
    assert((count > 0) && (count < 4) && "count should be between 1 and 3");

    unsigned int shift = (4U - count) * 8U;
    // clear the upper bytes
    k = _mm_srli_epi32(_mm_slli_epi32(k, shift), shift); // sse2

    // preprocess the 4 streams
    k = _mm_mullo_epi32(k, c1); // SSE2
    k = rotl32(k, 15);          // sse2
    k = _mm_mullo_epi32(k, c2); // SSE2
    // merge with existing.
    h = _mm_xor_si128(h, k); // SSE
    return h;
  }

  // count cannot be zero.
  FSC_FORCE_INLINE __m128i update32_partial(__m128i h, __m128i k, uint8_t const &count) const
  {

    // preprocess the 4 streams
    k = _mm_mullo_epi32(k, c1); // SSE2
    k = rotl32(k, 15);          // sse2
    k = _mm_mullo_epi32(k, c2); // SSE2
    // merge with existing.
    h = _mm_xor_si128(h, k); // SSE
    return h;
  }

  // input is 4 unsigned ints.
  // is ((h ^ f) * c) carryless multiplication with (f = h >> d)?
  FSC_FORCE_INLINE __m128i fmix32(__m128i h) const
  {
    h = _mm_xor_si128(h, _mm_srli_epi32(h, 16)); // h ^= h >> 16;      sse2
    h = _mm_mullo_epi32(h, mix_const1);          // h *= 0x85ebca6b;   sse4.1
    h = _mm_xor_si128(h, _mm_srli_epi32(h, 13)); // h ^= h >> 13;      sse2
    h = _mm_mullo_epi32(h, mix_const2);          // h *= 0xc2b2ae35;   sse4.1
    h = _mm_xor_si128(h, _mm_srli_epi32(h, 16)); // h ^= h >> 16;      sse2

    return h;
  }

  explicit Murmur32SSE(__m128i _seed) : seed(_seed),
                                        mix_const1(_mm_set1_epi32(0x85ebca6b)),
                                        mix_const2(_mm_set1_epi32(0xc2b2ae35)),
                                        c1(_mm_set1_epi32(0xcc9e2d51)),
                                        c2(_mm_set1_epi32(0x1b873593)),
                                        c3(_mm_set1_epi32(0x5)),
                                        c4(_mm_set1_epi32(0xe6546b64)),
                                        zero(_mm_setzero_si128()), // SSE2
                                        length(_mm_set1_epi32(bytes))
  {
  }

public:
  explicit Murmur32SSE(uint32_t _seed) : Murmur32SSE(_mm_set1_epi32(_seed))
  {
  }

  explicit Murmur32SSE(Murmur32SSE const &other) : Murmur32SSE(other.seed)
  {
  }

  explicit Murmur32SSE(Murmur32SSE &&other) : Murmur32SSE(other.seed)
  {
  }

  Murmur32SSE &operator=(Murmur32SSE const &other)
  {
    seed = other.seed;

    return *this;
  }

  Murmur32SSE &operator=(Murmur32SSE &&other)
  {
    seed = other.seed;

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
  //          template <uint64_t len = bytes,
  //              typename std::enable_if<((len & (len - 1)) > 0) && ((len & 15) > 0), int>::type = 1>
  //          FSC_FORCE_INLINE void hash( T const * key, uint8_t nstreams, uint32_t * out) const {
  //            // process 4 streams at a time.  all should be the same length.
  //
  //            assert((nstreams <= 4) && "maximum number of streams is 4");
  //            assert((nstreams > 0) && "minimum number of streams is 1");
  //
  //            // example layout, with each dash representing 2 bytes
  //            //     AAAAABBBBBCCCCCDDDDDEEEEE
  //            // k0  --------
  //            // k1       --------
  //            // k2            --------
  //            // k3                 --------
  //            __m128i k0, k1, k2, k3, t0, t1;
  //            __m128i h1 = seed;
  //
  //            //----------
  //            // first do blocks of 16 bytes.
  //
  //            const int nblocks = len >> 4;
  //
  //            // init to zero
  //            switch (nstreams) {
  //              case 1: k1 = zero;  // SSE
  //              case 2: k2 = zero;  // SSE
  //              case 3: k3 = zero;  // SSE
  //              default:
  //                break;
  //            }
  //
  //            for (int i = 0; i < nblocks; ++i) {
  //              // read streams
  //              switch (nstreams) {
  //                case 4: k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 3) + i);  // SSE3
  //                case 3: k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2) + i);  // SSE3
  //                case 2: k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 1) + i);  // SSE3
  //                case 1: k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key) + i);  // SSE3
  //                default:
  //                  break;
  //              }
  //
  //              // transpose the streams (4x4 matrix), so that each uint32 in h1 is one hash value.
  //              //  this adds extra 8 instructions in total
  //              t0 = _mm_unpacklo_epi32(k0, k1); // transpose 2x2   SSE2
  //              t1 = _mm_unpacklo_epi32(k2, k3); // transpose 2x2   SSE2
  //
  //              // row 0
  //              h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
  //
  //              // row 1
  //              h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2
  //
  //              // transpose some more.
  //              t0 = _mm_unpackhi_epi32(k0, k1); // transpose 2x2  SSE2
  //              t1 = _mm_unpackhi_epi32(k2, k3); // transpose 2x2  SSE2
  //
  //              // row 2
  //              h1 = update32(h1, _mm_unpacklo_epi64(t0, t1));  // transpose 4x2  SSE2
  //
  //              // row 3
  //              h1 = update32(h1, _mm_unpackhi_epi64(t0, t1));   // transpose 4x2  SSE2
  //            }
  //
  //            // next do the remainder if any.
  //            if ((len & 0xF) > 0) {
  //
  //              // read more stream.  over read, and zero out.
  //              switch (nstreams) {
  //                case 4: k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 3) + nblocks);  // SSE3
  //                case 3: k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2) + nblocks);  // SSE3
  //                case 2: k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 1) + nblocks);  // SSE3
  //                case 1: k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key) + nblocks);  // SSE3
  //                default:
  //                  break;
  //              }
  //
  //              // transpose (8 ops), set missing bytes to 0 (2 extra ops), and do as many words as there are.
  //
  //              // needed by all cases.
  //              t0 = _mm_unpacklo_epi32(k0, k1);         // transpose 2x2   SSE2
  //              t1 = _mm_unpacklo_epi32(k2, k3);         // transpose 2x2   SSE2
  //
  //
  //              // zeroing out unused bytes takes 2 instructions.
  //              constexpr uint8_t words = ((len & 15UL) + 3) >> 2;  // use ceiling word count.  else case 3 needs conditional to check for 12 bytes
  //              constexpr uint8_t rem = len & 0x3;
  //              //            std::cout << " len " << static_cast<size_t>(len) << " words " << static_cast<size_t>(words) <<
  //              //                " rem " << static_cast<size_t>(rem) << std::endl;
  //              switch (words) {
  //                case 4:
  //                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
  //                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2
  //
  //                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
  //                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
  //                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1));
  //                  // last word needs to be padded with 0 always.
  //                  h1 =  update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem);  // remainder  > 0
  //                  break;
  //                case 3:
  //                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
  //                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2
  //
  //                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
  //                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
  //                  // last word needs to be padded with 0.  3 rows only
  //                  h1 = (rem > 0) ?
  //                      update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
  //                      update32(h1, _mm_unpacklo_epi64(t0, t1));
  //                  break;
  //                case 2:
  //                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
  //                  // last word needs to be padded with 0.  2 rows only.  rem >= 0
  //                  h1 = (rem > 0) ?
  //                      update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem) :  // remainder  >= 0
  //                      update32(h1, _mm_unpackhi_epi64(t0, t1));
  //                  break;
  //                case 1:
  //                  // last word needs to be padded with 0.  1 rows only.  remainder must be >= 0
  //                  h1 = (rem > 0) ?
  //                      update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
  //                      update32(h1, _mm_unpacklo_epi64(t0, t1));
  //                  break;
  //                default:
  //                  break;
  //              }
  //
  //            }
  //
  //            //----------
  //            // finalization
  //            // or the length.
  //            h1 = _mm_xor_si128(h1, length);  // sse
  //
  //            h1 = fmix32(h1);  // ***** SSE4.1 **********
  //
  //            // store all 4 out
  //            switch (nstreams) {
  //              case 4: _mm_storeu_si128((__m128i*)out, h1);  // sse
  //              break;
  //              case 3: out[2] = _mm_extract_epi32(h1, 2);   // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
  //              case 2: *(reinterpret_cast<uint64_t *>(out)) = _mm_extract_epi64(h1, 0);
  //              break;
  //              case 1: out[0] = _mm_extract_epi32(h1, 0);
  //              default:
  //                break;;
  //            }
  //          }
  //

  // power of 2, or multiples of 16.
  //          template <uint64_t len = bytes,
  //              typename std::enable_if<((len & (len - 1)) == 0) || ((len & 15) == 0), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint32_t *out) const
  {
    // process 4 streams at a time.  all should be the same length.

    assert((nstreams <= 4) && "maximum number of streams is 4");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m128i h1 = hash4(key);

    // store all 4 out
    switch (nstreams)
    {
    case 4:
      _mm_storeu_si128((__m128i *)out, h1); // sse
      break;
    case 3:
      out[2] = _mm_extract_epi32(h1, 2); // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
    case 2:
      *(reinterpret_cast<uint64_t *>(out)) = _mm_extract_epi64(h1, 0);
      break;
    case 1:
      out[0] = _mm_extract_epi32(h1, 0);
    default:
      break;
      ;
    }
  }

  // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
  // above 2, the finalize and updates will dominate and better to do those in parallel.

  FSC_FORCE_INLINE void hash4(T const *key, uint32_t *out) const
  {
    __m128i res = hash4(key);
    _mm_storeu_si128((__m128i *)out, res);
  }

  // not power of 2, and not multiples of 16.
  template <uint64_t len = bytes,
            typename std::enable_if<((len & (len - 1)) > 0) && ((len & 15) > 0), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    __m128i k0, k1, k2, k3, t0, t1, t2, t3;
    __m128i h1 = seed;

    //----------
    // first do blocks of 16 bytes.
    const int nblocks = len >> 4;

    // init to zero
    for (int i = 0; i < nblocks; ++i)
    {
      k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + i);     // SSE3
      k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + i); // SSE3
      k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + i); // SSE3
      k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + i); // SSE3

      // transpose the streams (4x4 matrix), so that each uint32 in h1 is one hash value.
      //  this adds extra 8 instructions in total
      t0 = _mm_unpacklo_epi32(k0, k1); // transpose 2x2   SSE2
      t2 = _mm_unpackhi_epi32(k0, k1); // transpose 2x2  SSE2
      t1 = _mm_unpacklo_epi32(k2, k3); // transpose 2x2   SSE2
      t3 = _mm_unpackhi_epi32(k2, k3); // transpose 2x2  SSE2

      // row 0
      h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2

      // row 1
      h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

      // row 2
      h1 = update32(h1, _mm_unpacklo_epi64(t2, t3)); // transpose 4x2  SSE2

      // row 3
      h1 = update32(h1, _mm_unpackhi_epi64(t2, t3)); // transpose 4x2  SSE2
    }

    // next do the remainder if any.
    if ((len & 0xF) > 0)
    {

      // read more stream.  over read, and zero out.
      k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + nblocks);     // SSE3
      k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + nblocks); // SSE3
      k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + nblocks); // SSE3
      k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + nblocks); // SSE3

      // transpose (8 ops), set missing bytes to 0 (2 extra ops), and do as many words as there are.

      // needed by all cases.
      t0 = _mm_unpacklo_epi32(k0, k1); // transpose 2x2   SSE2
      t1 = _mm_unpacklo_epi32(k2, k3); // transpose 2x2   SSE2

      // zeroing out unused bytes takes 2 instructions.
      const uint8_t words = ((len & 15UL) + 3) >> 2; // use ceiling word count.  else case 3 needs conditional to check for 12 bytes
      const uint8_t rem = len & 0x3;
      //            std::cout << " len " << static_cast<size_t>(len) << " words " << static_cast<size_t>(words) <<
      //                " rem " << static_cast<size_t>(rem) << std::endl;
      switch (words)
      {
      case 4:
        h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
        h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

        t0 = _mm_unpackhi_epi32(k0, k1); // transpose 2x2   SSE2
        t1 = _mm_unpackhi_epi32(k2, k3); // transpose 2x2   SSE2
        h1 = update32(h1, _mm_unpacklo_epi64(t0, t1));
        // last word needs to be padded with 0.
        //                h1 = (rem > 0) ?
        h1 = update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem); // remainder  > 0
        //                    update32(h1, _mm_unpackhi_epi64(t0, t1));
        break;
      case 3:
        h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
        h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

        t0 = _mm_unpackhi_epi32(k0, k1); // transpose 2x2   SSE2
        t1 = _mm_unpackhi_epi32(k2, k3); // transpose 2x2   SSE2
        // last word needs to be padded with 0.  3 rows only
        h1 = (rem > 0) ? update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) : // remainder  >= 0
                 update32(h1, _mm_unpacklo_epi64(t0, t1));
        break;
      case 2:
        h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
        // last word needs to be padded with 0.  2 rows only.  rem >= 0
        h1 = (rem > 0) ? update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem) : // remainder  >= 0
                 update32(h1, _mm_unpackhi_epi64(t0, t1));
        break;
      case 1:
        // last word needs to be padded with 0.  1 rows only.  remainder must be >= 0
        h1 = (rem > 0) ? update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) : // remainder  >= 0
                 update32(h1, _mm_unpacklo_epi64(t0, t1));
        break;
      default:
        break;
      }
    }

    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }

  // multiples of 16
  template <uint64_t len = bytes,
            typename std::enable_if<((len & 15) == 0), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    __m128i k0, k1, k2, k3, t0, t1;
    __m128i h1 = seed;

    //----------
    // do blocks of 16 bytes.
    const int nblocks = len >> 4;

    // init to zero
    for (int i = 0; i < nblocks; ++i)
    {

      k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + i);     // SSE3
      k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + i); // SSE3
      k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + i); // SSE3
      k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + i); // SSE3

      // transpose the streams (4x4 matrix), so that each uint32 in h1 is one hash value.
      //  this adds extra 8 instructions in total
      t0 = _mm_unpacklo_epi32(k0, k1); // transpose 2x2   SSE2
      t1 = _mm_unpacklo_epi32(k2, k3); // transpose 2x2   SSE2

      // row 0
      h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2

      // row 1
      h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

      // transpose some more.
      t0 = _mm_unpackhi_epi32(k0, k1); // transpose 2x2  SSE2
      t1 = _mm_unpackhi_epi32(k2, k3); // transpose 2x2  SSE2

      // row 2
      h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2

      // row 3
      h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2
    }
    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }

  template <uint64_t len = bytes,
            typename std::enable_if<(len == 8), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     aaAAbbBBccCCddDD
    // k0  --------
    // k1          --------

    __m128i k0, k1, t0, t1;
    __m128i h1 = seed;

    // read input
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key));     // SSE3
    k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2)); // SSE3

    // 4 extra inst.
    // transform to aabbAABB ccddCCDD.  immediate = 3 1 2 0 => 11 01 10 00 == 0xd8
    k0 = _mm_shuffle_epi32(k0, 0xd8); // transpose 2x2   SSE2
    k1 = _mm_shuffle_epi32(k1, 0xd8); // transpose 2x2   SSE2

    t0 = _mm_unpacklo_epi64(k0, k1);
    t1 = _mm_unpackhi_epi64(k0, k1);

    // row 0.  unpacklo puts it to aabbccdd
    h1 = update32(h1, t0); // transpose 4x2  SSE2

    // row 1.  unpackhi puts it to AABBCCDD
    h1 = update32(h1, t1); // transpose 4x2  SSE2

    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }

  template <uint64_t len = bytes,
            typename std::enable_if<(len == 4), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     aabbccdd
    // k0  --------

    __m128i k0;
    __m128i h1 = seed;

    // no extra inst
    // blocks of 4
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3

    // row 0.  unpacklo puts it to aabbccdd
    h1 = update32(h1, k0); // transpose 4x2  SSE2

    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }

  template <uint64_t len = bytes,
            typename std::enable_if<(len == 2), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     abcdefgh
    // k0  --------

    __m128i k0;
    __m128i h1 = seed;

    // blocks of 2
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3

    // 1 extra inst
    // transform to a0b0c0d0.  interleave with 0.
    // row 0.
    h1 = update32_partial(h1, _mm_unpacklo_epi16(k0, zero), 2); // transpose 4x2  SSE2

    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }

  template <uint64_t len = bytes,
            typename std::enable_if<(len == 1), int>::type = 1>
  FSC_FORCE_INLINE __m128i hash4(T const *key) const
  {
    // process 4 streams at a time.  all should be the same length.

    // example layout, with each dash representing 1 bytes
    //     abcdefghijklmop
    // k0  ---------------

    __m128i k0, t0;
    __m128i h1 = seed;

    // blocks of 2
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3

    // 2 extra inst.
    // transform to a0b0c0d0e0f0g0h0.  interleave with 0.
    t0 = _mm_unpacklo_epi8(k0, zero);

    // transform to a000b000c000d000.  interleave with 0.
    // row 0.
    h1 = update32_partial(h1, _mm_unpacklo_epi16(t0, zero), 1); // transpose 4x2  SSE2

    //----------
    // finalization
    // or the length.
    h1 = _mm_xor_si128(h1, length); // sse

    h1 = fmix32(h1); // ***** SSE4.1 **********

    return h1;
  }
};

//      NOTE: no _mm_mullo_epi64 below avx512, so no point in this for Broadwell and earlier.
//          if we were to do x = a 2^32 + b and y = c 2^32 + d, 
//          then x * y = b * d + (a * d + c * b) 2^32, with truncation.  a*c does not matter.
//             we would need 2 shifts, to get a and c., 3 multiplies with Lat5 C1,
//             then 1 addition, 1 shift, and another addition.  some masking may be needed.
//        w = shuffle x (w = d 2^32 + c), w = mullo(x, w), z = x * y; idle_w x8; hadd(w, 0); idle_h x 2; mask; shift; z = z+w.  - slower, and need 4 to offset latency. 24 cycles for 16 elements - faster.
//      or    a = x >> 32; a = a * d; c = y >> 32; c = c * x; z = x * d; idle_a, idle_a, idle_c, idle_c; a = a+c ; a = a << 32; z = z + a;  need 2 to offset latency.  16 cycles for 8 elements.
//          but y is a constant, so c is a constant.

//      // compute a pair of 64bit at a time, equivalent to the lower 64 bit of the 128 bit murmur hash.
//      // this simplifies the computation given the originam murmur3 128bit has dependencies that
//      // makes them hard to map to 128 bits.
//      template <typename T, size_t bytes = sizeof(T)>
//      class Murmur64SSE {
//
//
//        protected:
//          // make static so initialization at beginning of class...
//          const __m128i seed;
//          const __m128i mix_const1;
//          const __m128i mix_const2;
//          const __m128i c1;
//          const __m128i c2;
//          const __m128i c3;
//          const __m128i c4_1;
//          const __m128i c4_2;
//          const __m128i zero;
//          const __m128i length;
//
//          // input is 4 unsigned ints.
//          FSC_FORCE_INLINE __m128i rotl64 ( __m128i x, int8_t r ) const
//          {
//            // return (x << r) | (x >> (32 - r));
//            return _mm_or_si128(                // sse2
//                            _mm_slli_epi64(x, r),           // sse2
//                            _mm_srli_epi64(x, (64 - r)));   // sse2
//          }
//
//
//          FSC_FORCE_INLINE __m128i update64_partial( __m128i h, __m128i k,
//        		  __m128i _c1, int8_t r, __m128i _c2) const {
//
//            //            k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
//            //            k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
//
//            k = _mm_mullo_epi64(k, _c1);
//            k = rotl64(k, r);
//            k = _mm_mullo_epi64(k, _c2);
//
//            return _mm_xor_si128(h, k);
//          }
//
//          FSC_FORCE_INLINE __m128i update64_partial1( __m128i h1, __m128i k1) const {
//        	  return update64_partial(h1, k1, c1, 31, c2);
//          }
//          FSC_FORCE_INLINE __m128i update64_partial2( __m128i h2, __m128i k2) const {
//        	  return update64_partial(h2, k2, c2, 33, c1);
//          }
//
//
//          FSC_FORCE_INLINE __m128i update64( __m128i h, __m128i h2, __m128i k,
//        		  __m128i _c1, int8_t r1, __m128i _c2,
//				  int8_t r2, __m128i _c4) const {
//
//            // k1 and k2 are independent.  but need to save original h2, and concate with current h1.
//            //            k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
//            //            h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
//            //            k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
//            //            h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
//            h = update64_partial(h, k, _c1, r1, _c2);
//
//            h = rotl64(h, r2);
//
//            h = _mm_add_epi64(h, h2);
//
//            return _mm_add_epi64(_mm_mullo_epi64(h, c3), _c4);
//          }
//
//          FSC_FORCE_INLINE __m128i update64_1( __m128i h1, __m128i h2, __m128i k1) const {
//        	  return update64(h1, h2, k1, c1, 31, c2, 27, c4_1);
//          }
//
//          FSC_FORCE_INLINE __m128i update64_2( __m128i h2, __m128i h1, __m128i k2) const {
//        	  return update64(h2, h1, k2, c2, 33, c1, 31, c4_2);
//          }
//
//
//          // count cannot be zero.
//          FSC_FORCE_INLINE __m128i zeroing( __m128i k, uint8_t const & count) const {
//            assert((count > 0) && (count < 8) && "count should be between 1 and 3");
//
//            unsigned int shift = (8U - count) * 8U;
//            // clear the upper bytes
//            return _mm_srli_epi64(_mm_slli_epi64(k, shift), shift);  // sse2
//          }
//
//          // input is the 2 halves of the hash values.
//          // this is called for h1 and h2 at the same time.
//          // is ((h ^ f) * c) carryless multiplication with (f = h >> d)? NO.
//          FSC_FORCE_INLINE __m128i fmix64 ( __m128i h ) const
//          {
//            h = _mm_xor_si128(h, _mm_srli_epi64(h, 33));  // k ^= k >> 33;                            sse2
//            h = _mm_mullo_epi64(h, mix_const1);           // k *= BIG_CONSTANT(0xff51afd7ed558ccd);   sse4.1
//            h = _mm_xor_si128(h, _mm_srli_epi64(h, 33));  // k ^= k >> 33;                            sse2
//            h = _mm_mullo_epi64(h, mix_const2);           // k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);   sse4.1
//            h = _mm_xor_si128(h, _mm_srli_epi64(h, 33));  // k ^= k >> 33;                            sse2
//
//            return h;
//          }
//
//        public:
//          Murmur64SSE(uint64_t _seed) :
//            seed(_mm_set1_epi64x(_seed)),
//            mix_const1(_mm_set1_epi64x(0xff51afd7ed558ccd)),  // same for h1 and h0
//            mix_const2(_mm_set1_epi64x(0xc4ceb9fe1a85ec53)),  // same for h1 and h0
//            c1(_mm_set1_epi64x(0x4cf5ad432745937f)),   // c1 and c2 are concat .  c2, c1 for k2, k1,
//            c2(_mm_set1_epi64x(0x87c37b91114253d5)),   //  then c1 and c2 (flipped0 for the second computation.
//            c3(_mm_set1_epi64x(5)),
//            c4_1(_mm_set1_epi64x(0x38495ab5)),   //   different for h1 and h0
//			      c4_2(_mm_set1_epi64x(0x52dce729)),   //   different for h1 and h0
//            zero(_mm_setzero_si128()),// SSE2
//            length(_mm_set1_epi64x(bytes))
//        {}
//
//          // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
//          // above 2, the finalize and updates will dominate and better to do those in parallel.
//          // TODO: [ ] hash2.  do 2 streams at the same time.
//
//          // multiples of 16 bytes
//          template <uint64_t len = bytes>
//          FSC_FORCE_INLINE __m128i hash2(T const * key) const {
//
//
//            const int nblocks = len >> 4;
//
//            __m128i k1, k2, t1, t2;
//            __m128i h1 = seed, h2 = seed;
//
//            for (size_t i = 0; i < nblocks; ++i) {
//            	// aA and bB
//              k1 = _mm_lddqu_si128(reinterpret_cast<__m128i const *>(key) + i);
//              k2 = _mm_lddqu_si128(reinterpret_cast<__m128i const *>(key + 1) + i);
//
//              t1 = _mm_unpacklo_epi64(k1, k2);   //ab
//              t2 = _mm_unpackhi_epi64(k1, k2);   //AB
//
//              // now update both at the same time.
////              k1 *= c1; k1  = ROTL64(k1,31); k1 *= c2; h1 ^= k1;
////              h1 = ROTL64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
////              k2 *= c2; k2  = ROTL64(k2,33); k2 *= c1; h2 ^= k2;
////              h2 = ROTL64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
//              h1 = update64_1(h1, h2, t1);
//              h2 = update64_2(h2, h1, t2);
//            }
//
//            // remainder
//            if (len & 15) {
//            	k1 = _mm_lddqu_si128(reinterpret_cast<__m128i const *>(key) + nblocks);		 //aA
//            	k2 = _mm_lddqu_si128(reinterpret_cast<__m128i const *>(key + 1) + nblocks);  //bB
//
//            	t1 = _mm_unpacklo_epi64(k1, k2);   //ab
//                t2 = _mm_unpackhi_epi64(k1, k2);   //AB
//            }
//            if ((len & 15) > 8) {
//            	// zero out the extra stuff.
//                t2 = zeroing(t2, (len & 7));
//
//            } else if ((len & 15) == 8) {
//            	t2 = zero;
//
//            } else if ((len & 15) > 0) {
//            	// zero out the extra stuff.
//            	t1 = zeroing(t1, (len & 7));
//
//            	t2 = zero;
//            }
//            if (len & 15) {
//                // now update
//                h1 = update64_partial1(h1, t1);
//                h2 = update64_partial1(h2, t2);
//            }
//
//            // finalize
//            h1 = _mm_xor_si128(h1, length);
//            h2 = _mm_xor_si128(h2, length);
//
//            h1 = _mm_add_epi64(h1, h2);
//            h2 = _mm_add_epi64(h2, h1);
//
//            h1 = fmix64(h1);
//            h2 = fmix64(h2);
//
//            h1 = _mm_add_epi64(h1, h2);
//            h2 = _mm_add_epi64(h2, h1);
//
//            return h1;
//
//            //
//            //          //----------
//            //          // finalization
//            //
//            //          h1 ^= len; h2 ^= len;
//            //
//            //          h1 += h2;
//            //          h2 += h1;
//            //
//            //          h1 = fmix64(h1);
//            //          h2 = fmix64(h2);
//            //
//            //          h1 += h2;
//            //          h2 += h1;
//            //
//            //          ((uint64_t*)out)[0] = h1;
//            //          ((uint64_t*)out)[1] = h2;
//
//          }
//
//          // useful for computing 4 32bit hashes in 1 pass (for hashing into less than 2^32 buckets)
//          // assume 4 streams are available.
//          // working with 4 bytes at a time because there are
//          // init: 4 instr.
//          // body: 13*4 + 12  per iter of 16 bytes
//          // tail: about the same
//          // finalize: 11 inst. for 4 elements.
//          // about 5 inst per byte + 11 inst for 4 elements.
//          // for types that are have size larger than 8 or not power of 2.
//
//          // power of 2, or multiples of 16.
////          template <uint64_t len = bytes,
////              typename std::enable_if<((len & (len - 1)) == 0) || ((len & 15) == 0), int>::type = 1>
//          FSC_FORCE_INLINE void hash(T const * key, uint8_t nstreams, uint64_t * out) const {
//            // process 4 streams at a time.  all should be the same length.
//
//            assert((nstreams <= 2) && "maximum number of streams is 2");
//            assert((nstreams > 0) && "minimum number of streams is 1");
//
//            __m128i h1 = hash2(key);
//
//            // store all 4 out
//            switch (nstreams) {
//              case 2: _mm_storeu_si128((__m128i*)out, h1);  // sse
//                break;
//              case 1: out[0] = _mm_extract_epi64(h1, 0);
//              default:
//                break;;
//            }
//          }
//
//          FSC_FORCE_INLINE void hash2(T const *  key, uint64_t * out) const {
//            __m128i res = hash2(key);
//            _mm_storeu_si128((__m128i*)out, res);
//          }
//
//      };
//

#endif

} // namespace sse

/**
     * @brief  returns the least significant 64 bits directly as identity hash.
     * @note   since the number of buckets is not known ahead of time, can't have nbit be a type
     */
template <typename T>
class identity
{

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint64_t;
  using argument_type = T;

  /// operator to compute hash value
  inline uint64_t operator()(const T &key) const
  {
    if (sizeof(T) >= 8) // more than 64 bits, so use the lower 64 bits.
      return *(reinterpret_cast<uint64_t const *>(&key));
    else if (sizeof(T) == 4)
      return *(reinterpret_cast<uint32_t const *>(&key));
    else if (sizeof(T) == 2)
      return *(reinterpret_cast<uint16_t const *>(&key));
    else if (sizeof(T) == 1)
      return *(reinterpret_cast<uint8_t const *>(&key));
    else
    {
      // copy into 64 bits
      uint64_t out = 0;
      memcpy(&out, &key, sizeof(T));
      return out;
    }
  }
};
template <typename T>
constexpr uint8_t identity<T>::batch_size;

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
class murmur3avx32
{

protected:
  ::fsc::hash::sse::Murmur32AVX<T> hasher;
  mutable uint32_t temp[8];

public:
  static constexpr uint8_t batch_size = ::fsc::hash::sse::Murmur32AVX<T>::batch_size;
  using result_type = uint32_t;
  using argument_type = T;

  murmur3avx32(uint32_t const &_seed = 43) : hasher(_seed){};

  inline uint32_t operator()(const T &key) const
  {
    uint32_t h;
    hasher.hash(&key, 1, &h);

    return h;
  }

  FSC_FORCE_INLINE void operator()(T const *keys, size_t count, uint32_t *results) const
  {
    hash(keys, count, results);
  }

  // results always 32 bit.
  FSC_FORCE_INLINE void hash(T const *keys, size_t count, uint32_t *results) const
  {

    size_t rem = count & (batch_size - 1);
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      hasher.hash(&(keys[i]), results + i);
    }

    if (rem > 0)
      hasher.hash(&(keys[i]), rem, results + i);
  }

  // assume consecutive memory layout.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    size_t rem = count & (batch_size - 1);
    size_t max = count - rem;
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      hasher.hash(&(keys[i]), temp);

      for (j = 0; j < batch_size; ++j)
    	  results[i + j] = temp[j] % modulus;
    }

    if (rem > 0) {
      hasher.hash(&(keys[i]), rem, temp);

      for (j = 0; j < rem; ++j)
    	  results[i + j] = temp[j] % modulus;
    }
  }

  // assume consecutive memory layout.
  // note that the paremter is modulus bits.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
    --modulus;

    size_t rem = count & (batch_size - 1);
    size_t max = count - rem;
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      hasher.hash(&(keys[i]), temp);
      for (j = 0; j < batch_size; ++j)
    	  results[i + j] = temp[j] & modulus;
    }

    // last part.
    if (rem > 0) {
      hasher.hash(&(keys[i]), rem, temp);
      for (j = 0; j < rem; ++j)
    	  results[i + j] = temp[j] & modulus;
    }
  }

  // TODO: [ ] add a transform_hash_mod.
};
template <typename T>
constexpr uint8_t murmur3avx32<T>::batch_size;

#endif

#if defined(__SSE4_1__)
/**
     * @brief MurmurHash.  using lower 64 bits.
     * @details.  prefetching did not help
     */
template <typename T>
class murmur3sse32
{

protected:
  ::fsc::hash::sse::Murmur32SSE<T> hasher;
  mutable uint32_t temp[4];

public:
  static constexpr uint8_t batch_size = 4;
  using result_type = uint32_t;
  using argument_type = T;

  murmur3sse32(uint32_t const &_seed = 43) : hasher(_seed){};

  inline uint32_t operator()(const T &key) const
  {
    uint32_t h;
    hasher.hash(&key, 1, &h);
    return h;
  }

  FSC_FORCE_INLINE void operator()(T const *keys, size_t count, uint32_t *results) const
  {
    hash(keys, count, results);
  }

  // results always 32 bit.
  FSC_FORCE_INLINE void hash(T const *keys, size_t count, uint32_t *results) const
  {
    size_t rem = count & 0x3;
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hasher.hash4(&(keys[i]), results + i);
    }

    if (rem > 0)
      hasher.hash(&(keys[i]), rem, results + i);
  }

  // assume consecutive memory layout.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    size_t rem = count & 0x3;
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hasher.hash4(&(keys[i]), temp);
      results[i] = temp[0] % modulus;
      results[i + 1] = temp[1] % modulus;
      results[i + 2] = temp[2] % modulus;
      results[i + 3] = temp[3] % modulus;
    }

    // last part.
    if (rem > 0)
      hasher.hash(&(keys[i]), rem, temp);
    switch (rem)
    {
    case 3:
      results[i + 2] = temp[2] % modulus;
    case 2:
      results[i + 1] = temp[1] % modulus;
    case 1:
      results[i] = temp[0] % modulus;
    default:
      break;
    }
  }

  // assume consecutive memory layout.
  // note that the paremter is modulus bits.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
    --modulus;

    size_t rem = count & 0x3;
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hasher.hash4(&(keys[i]), temp);
      results[i] = temp[0] & modulus;
      results[i + 1] = temp[1] & modulus;
      results[i + 2] = temp[2] & modulus;
      results[i + 3] = temp[3] & modulus;
    }

    // last part.
    if (rem > 0)
      hasher.hash(&(keys[i]), rem, temp);
    switch (rem)
    {
    case 3:
      results[i + 2] = temp[2] & modulus;
    case 2:
      results[i + 1] = temp[1] & modulus;
    case 1:
      results[i] = temp[0] & modulus;
    default:
      break;
    }
  }

  // TODO: [ ] add a transform_hash_mod.
};
template <typename T>
constexpr uint8_t murmur3sse32<T>::batch_size;

//    /**
//     * @brief MurmurHash.  using lower 64 bits.
//     * @details.  prefetching did not help
//                  NOTE: no _mm_mullo_epi64.  so no point in this for Broadwell and earlier.
//     */
//    template <typename T>
//    class murmur3sse64 {
//
//
//      protected:
//        ::fsc::hash::sse::Murmur64SSE<T> hasher;
//        mutable void const * kptrs[2];
//        mutable uint64_t temp[2];
//
//      public:
//        static constexpr uint8_t batch_size = 2;
//
//        murmur3sse64(uint64_t const & _seed = 43 ) : hasher(_seed) {};
//
//        inline uint64_t operator()(const T & key) const
//        {
//          //          kptrs[0] = &key;
//          uint64_t h;
//          hasher.hash(&key, 1, &h);
//          return h;
//        }
//
//        // results always 32 bit.
//        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint64_t * results) const {
//          size_t rem = count & 0x1;
//          size_t max = count - rem;
//          size_t i = 0;
//          for (; i < max; i += 2) {
//            hasher.hash2(&(keys[i]), results + i);
//          }
//
//          if (rem > 0)
//            hasher.hash(&(keys[i]), rem, results + i);
//        }
//
//        // assume consecutive memory layout.
//        template<typename OT>
//        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint64_t modulus) const {
//          size_t rem = count & 0x1;
//          size_t max = count - rem;
//          size_t i = 0;
//          for (; i < max; i += 2) {
//            hasher.hash2(&(keys[i]), temp);
//            results[i] = temp[0] % modulus;
//            results[i+1] = temp[1] % modulus;
//          }
//
//          if (rem > 0) {
//              hasher.hash(&(keys[i]), rem, temp);
//              results[i] = temp[0] % modulus;
//          }
//        }
//
//        // assume consecutive memory layout.
//        // note that the paremter is modulus bits.
//        template<typename OT>
//        FSC_FORCE_INLINE void hash_and_mod_pow2(T const * keys, size_t count, OT * results, uint64_t modulus) const {
//          assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
//          --modulus;
//
//          size_t rem = count & 0x1;
//          size_t max = count - rem;
//          size_t i = 0;
//          for (; i < max; i += 2) {
//            hasher.hash2(&(keys[i]), temp);
//            results[i]   = temp[0] & modulus;
//            results[i+1] = temp[1] & modulus;
//          }
//
//          if (rem > 0) {
//            hasher.hash(&(keys[i]), rem, temp);
//            results[i]   = temp[0] & modulus;
//          }
//        }
//
//
//        // TODO: [ ] add a transform_hash_mod.
//    };
#endif

/**
     * @brief MurmurHash.  using lower 64 bits.
     *
     */
template <typename T>
class murmur32
{

protected:
  uint32_t seed;

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint32_t;
  using argument_type = T;

  murmur32(uint32_t const &_seed = 43) : seed(_seed){};

  inline uint32_t operator()(const T &key) const
  {
    // produces 128 bit hash.
    uint32_t h;
    // let compiler optimize out all except one of these.
    MurmurHash3_x86_32(&key, sizeof(T), seed, &h);

    // use the upper 64 bits.
    return h;
  }
};
template <typename T>
constexpr uint8_t murmur32<T>::batch_size;

/**
     * @brief MurmurHash.  using lower 64 bits.
     *
     */
template <typename T>
class murmur
{

protected:
  uint32_t seed;

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint64_t;
  using argument_type = T;

  murmur(uint32_t const &_seed = 43) : seed(_seed){};

  inline uint64_t operator()(const T &key) const
  {
    // produces 128 bit hash.
    uint64_t h[2];
    // let compiler optimize out all except one of these.
    if (sizeof(void *) == 8)
      MurmurHash3_x64_128(&key, sizeof(T), seed, h);
    else if (sizeof(void *) == 4)
      MurmurHash3_x86_128(&key, sizeof(T), seed, h);
    else
      throw ::std::logic_error("ERROR: neither 32 bit nor 64 bit system");

    // use the upper 64 bits.
    return h[0];
  }
};
template <typename T>
constexpr uint8_t murmur<T>::batch_size;

#if defined(__SSE4_2__)

/**
     * @brief crc.  32 bit hash..
     * @details  operator should require sizeof(T)/8  + 6 operations + 2 cycle latencies.
     *            require SSE4.2
     *
     *          prefetching did not help
     *   algo at https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_crc&expand=5247,1244
     *   PROBLEM:  using a random seed at beginning of algo does not seem to help.
     *   		inputs at regular intervals (from a partition by a previous crc32c)
     *   			shows the same kind of regular intervals as the seed does not appear to sufficiently mix the bits.
     *   	transforms via seed does not seem to have an effect.
     *   HOWEVER, CRC can be applied when WITH a different family of hash.  given its speed, maybe at critical section.
     *
     *  TAKE AWAY: CRC32C is good for fast hashing but should not be used as a FAMILY of hash functions.
     */
template <typename T>
class crc32c
{

protected:
  uint32_t seed;
  static constexpr size_t blocks = sizeof(T) >> 3; // divide by 8
  static constexpr size_t rem = sizeof(T) & 0x7;   // remainder.
  static constexpr size_t offset = (sizeof(T) >> 3) << 3;
  mutable uint32_t temp[4];

  FSC_FORCE_INLINE uint32_t hash1(const T &key) const
  {

    // block of 8 bytes
    uint64_t crc64 = seed; //_mm_crc32_u32(1, seed);
    if (sizeof(T) >= 8)
    {
      uint64_t const *data64 = reinterpret_cast<uint64_t const *>(&key);
      for (size_t i = 0; i < blocks; ++i)
      {
        crc64 = _mm_crc32_u64(crc64, data64[i]);
      }
    }

    uint32_t crc = static_cast<uint32_t>(crc64);

    unsigned char const *data = reinterpret_cast<unsigned char const *>(&key);

    // rest.  do it cleanly
    size_t off = offset; // * 8
    if (rem & 0x4)
    { // has 4 bytes
      crc = _mm_crc32_u32(crc, *(reinterpret_cast<uint32_t const *>(data + off)));
      off += 4;
    }
    if (rem & 0x2)
    { // has 2 bytes extra
      crc = _mm_crc32_u16(crc, *(reinterpret_cast<uint16_t const *>(data + off)));
      off += 2;
    }
    if (rem & 0x1)
    { // has 1 byte extra
      crc = _mm_crc32_u8(crc, *(reinterpret_cast<uint8_t const *>(data + off)));
    }

    return crc;
  }

  FSC_FORCE_INLINE void hash4(T const *keys, uint32_t *results) const
  {
    // loop over 4 keys at a time
    uint64_t aa, bb, cc, dd;
    aa = bb = cc = dd = seed; // _mm_crc32_u32(1, seed);

    if (sizeof(T) >= 8)
    {
      // block of 8 bytes
      uint64_t const *data64a = reinterpret_cast<uint64_t const *>(&(keys[0]));
      uint64_t const *data64b = reinterpret_cast<uint64_t const *>(&(keys[1]));
      uint64_t const *data64c = reinterpret_cast<uint64_t const *>(&(keys[2]));
      uint64_t const *data64d = reinterpret_cast<uint64_t const *>(&(keys[3]));

      for (size_t i = 0; i < blocks; ++i)
      {
        aa = _mm_crc32_u64(aa, data64a[i]);
        bb = _mm_crc32_u64(bb, data64b[i]);
        cc = _mm_crc32_u64(cc, data64c[i]);
        dd = _mm_crc32_u64(dd, data64d[i]);
      }
    }
    uint32_t a = static_cast<uint32_t>(aa);
    uint32_t b = static_cast<uint32_t>(bb);
    uint32_t c = static_cast<uint32_t>(cc);
    uint32_t d = static_cast<uint32_t>(dd);

    unsigned char const *dataa = reinterpret_cast<unsigned char const *>(&(keys[0]));
    unsigned char const *datab = reinterpret_cast<unsigned char const *>(&(keys[1]));
    unsigned char const *datac = reinterpret_cast<unsigned char const *>(&(keys[2]));
    unsigned char const *datad = reinterpret_cast<unsigned char const *>(&(keys[3]));

    // rest.  do it cleanly
    size_t off = offset; // * 8
    if (rem & 0x4)
    { // has 4 bytes
      a = _mm_crc32_u32(a, *(reinterpret_cast<uint32_t const *>(dataa + off)));
      b = _mm_crc32_u32(b, *(reinterpret_cast<uint32_t const *>(datab + off)));
      c = _mm_crc32_u32(c, *(reinterpret_cast<uint32_t const *>(datac + off)));
      d = _mm_crc32_u32(d, *(reinterpret_cast<uint32_t const *>(datad + off)));
      off += 4;
    }
    if (rem & 0x2)
    { // has 2 bytes extra
      a = _mm_crc32_u16(a, *(reinterpret_cast<uint16_t const *>(dataa + off)));
      b = _mm_crc32_u16(b, *(reinterpret_cast<uint16_t const *>(datab + off)));
      c = _mm_crc32_u16(c, *(reinterpret_cast<uint16_t const *>(datac + off)));
      d = _mm_crc32_u16(d, *(reinterpret_cast<uint16_t const *>(datad + off)));
      off += 2;
    }
    if (rem & 0x1)
    { // has 1 byte extra
      a = _mm_crc32_u8(a, *(reinterpret_cast<uint8_t const *>(dataa + off)));
      b = _mm_crc32_u8(b, *(reinterpret_cast<uint8_t const *>(datab + off)));
      c = _mm_crc32_u8(c, *(reinterpret_cast<uint8_t const *>(datac + off)));
      d = _mm_crc32_u8(d, *(reinterpret_cast<uint8_t const *>(datad + off)));
    }

    results[0] = a;
    results[1] = b;
    results[2] = c;
    results[3] = d;
  }

public:
  static constexpr uint8_t batch_size = 4;

  using result_type = uint32_t;
  using argument_type = T;

  crc32c(uint32_t const &_seed = 37) : seed(_seed){};

  // do 1 element.
  FSC_FORCE_INLINE uint32_t operator()(const T &key) const
  {
    //          std::cout << "CRC32C operator1()" << std::endl;
    return hash1(key);
  }

  FSC_FORCE_INLINE void operator()(T const *keys, size_t count, uint32_t *results) const
  {
    //          std::cout << "CRC32C operatorN() " << count << std::endl;
    hash(keys, count, results);
  }

  // results always 32 bit.
  // do 3 at the same time.  since latency of crc32 is 3 cycles.
  // however, to limit the modulus, do 4 at a time.
  FSC_FORCE_INLINE void hash(T const *keys, size_t count, uint32_t *results) const
  {
    // VERIFY THAT RIGHT CRC32 is being used.  VERIFIED.
    // std::cout << "DEBUG: seed set to " << seed << std::endl;

    // loop over 4 keys at a time
    size_t max = count - (count & 3);
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hash4(keys + i, results + i);
    }

    // handle the remainder
    for (; i < count; ++i)
    {
      results[i] = hash1(keys[i]);
    }
  }

  // do 3 at the same time.  since latency of crc32 is 3 cycles.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    // loop over 3 keys at a time
    size_t max = count - (count & 3);
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hash4(keys + i, temp);

      results[i] = temp[0] % modulus;
      results[i + 1] = temp[1] % modulus;
      results[i + 2] = temp[2] % modulus;
      results[i + 3] = temp[3] % modulus;
    }

    // handle the remainder
    for (; i < count; ++i)
    {
      results[i] = hash1(keys[i]) % modulus;
    }
  }

  // do 3 at the same time.  since latency of crc32 is 3 cycles.
  template <typename OT>
  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint32_t modulus) const
  {
    assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");

    --modulus; // convert to mask.

    // loop over 3 keys at a time
    size_t max = count - (count & 3);
    size_t i = 0;
    for (; i < max; i += 4)
    {
      hash4(keys + i, temp);

      results[i] = temp[0] & modulus;
      results[i + 1] = temp[1] & modulus;
      results[i + 2] = temp[2] & modulus;
      results[i + 3] = temp[3] & modulus;
    }

    // handle the remainder
    for (; i < count; ++i)
    {
      results[i] = hash1(keys[i]) & modulus;
    }
  }
};
template <typename T>
constexpr uint8_t crc32c<T>::batch_size;

#endif

#if defined(__AVX2__)

/**
     * @brief  farm hash
     *
     * MAY NOT WORK CONSISTENTLY between prefetching on and off.
     */
template <typename T>
class clhash
{

protected:
  mutable size_t rand_numbers[RANDOM_64BITWORDS_NEEDED_FOR_CLHASH] __attribute__((aligned(16)));

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint64_t;
  using argument_type = T;

  clhash(uint64_t const &_seed = 43)
  {
    srand(_seed);
    for (int i = 0; i < RANDOM_64BITWORDS_NEEDED_FOR_CLHASH; ++i)
    {
      rand_numbers[i] = (static_cast<unsigned long>(rand()) << 32) | static_cast<unsigned long>(rand());
    }
  };

  /// operator to compute hash.  64 bit again.
  inline uint64_t operator()(const T &key) const
  {
    return ::clhash(reinterpret_cast<const void *>(rand_numbers), reinterpret_cast<const char *>(&key), sizeof(T));
  }
};
template <typename T>
constexpr uint8_t clhash<T>::batch_size;
#endif

template <typename T>
class farm
{

protected:
  uint64_t seed;

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint64_t;
  using argument_type = T;

  farm(uint64_t const &_seed = 43) : seed(_seed){};

  /// operator to compute hash.  64 bit again.
  inline uint64_t operator()(const T &key) const
  {
    return ::util::Hash64WithSeed(reinterpret_cast<const char *>(&key), sizeof(T), seed);
  }
};
template <typename T>
constexpr uint8_t farm<T>::batch_size;

template <typename T>
class farm32
{

protected:
  uint32_t seed;

public:
  static constexpr uint8_t batch_size = 1;
  using result_type = uint32_t;
  using argument_type = T;

  farm32(uint32_t const &_seed = 43) : seed(_seed){};

  /// operator to compute hash.  64 bit again.
  inline uint32_t operator()(const T &key) const
  {
    return ::util::Hash32WithSeed(reinterpret_cast<const char *>(&key), sizeof(T), seed);
  }
};
template <typename T>
constexpr uint8_t farm32<T>::batch_size;

/// custom version of transformed hash that does a few things:
///    1. potentially bypass transform if it is identity.
///    2. provide batch mode operation, if not supported by transform and hash then do those one by one.
///    3. simplify the implementation of distributed hash map.
/// require that batch operation to be defined.
// TODO:
//     [ ] extend to support non-batching hash and transforms.
//
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform = ::bliss::transform::identity,
          template <typename> class PostTransform = ::bliss::transform::identity>
class TransformedHash
{

protected:
  using PRETRANS_T = PreTransform<Key>;

  // determine output type of PreTransform.
  using PRETRANS_VAL_TYPE =
      decltype(::std::declval<PRETRANS_T>().operator()(::std::declval<Key>()));

  using HASH_T = Hash<PRETRANS_VAL_TYPE>;

public:
  // determine output type of hash.  could be 64 bit or 32 bit.
  using HASH_VAL_TYPE =
      decltype(::std::declval<Hash<PRETRANS_VAL_TYPE>>().operator()(::std::declval<PRETRANS_VAL_TYPE>()));

  // determine output type of return value.
  using result_type =
      decltype(::std::declval<PostTransform<HASH_VAL_TYPE>>().operator()(::std::declval<HASH_VAL_TYPE>()));
  using argument_type = Key;

  // lowest common multiple of the three.  default to 64byte/sizeof(HASH_VAL_TYPE) for now (cacheline multiple)
  static constexpr uint8_t batch_size = 64 / sizeof(HASH_VAL_TYPE); //(sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);
                                                                    // HASH_T::batch_size; //(sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);

  static_assert((batch_size & (batch_size - 1)) == 0, "ERROR: batch_size should be a power of 2.");

protected:
  using POSTTRANS_T = PostTransform<HASH_VAL_TYPE>;

  // need some buffers
  // use local static array instead of dynamic ones so when
  // default copy construction/assignment happens,
  // we are not copying pointers that later gets freed by multiple objects.
  mutable Key key_buf[batch_size] __attribute__((aligned(64)));
  mutable PRETRANS_VAL_TYPE trans_buf[batch_size] __attribute__((aligned(64)));
  mutable HASH_VAL_TYPE hash_buf[batch_size] __attribute__((aligned(64)));

public:
  // potentially runs into double free issue when the pointers are copied.
  PRETRANS_T trans;
  HASH_T h;
  POSTTRANS_T posttrans;

  TransformedHash(HASH_T const &_hash = HASH_T(),
                  PRETRANS_T const &pre_trans = PRETRANS_T(),
                  POSTTRANS_T const &post_trans = POSTTRANS_T()) : //batch_size(lcm(lcm(pretrans_batch_size, hash_batch_size), postrans_batch_size)),
                                                                   trans(pre_trans),
                                                                   h(_hash), posttrans(post_trans){};

  ~TransformedHash()
  {
  }

  // conditionally defined, there should be just 1 defined methods after compiler resolves all this.
  // note that the compiler may do the same if it notices no-op....
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value && ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
    return h(k);
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
    return h(trans(k));
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
    return posttrans(h(k));
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
    return posttrans(h(trans(k)));
  }

  template <typename V>
  inline result_type operator()(::std::pair<Key, V> const &x) const
  {
    return this->operator()(x.first);
  }
  template <typename V>
  inline result_type operator()(::std::pair<const Key, V> const &x) const
  {
    return this->operator()(x.first);
  }

  //======= now for batched hash version.
protected:
  // use construct from
  //  https://stackoverflow.com/questions/257288/is-it-possible-to-write-a-template-to-check-for-a-functions-existence
  // namely to use auto and -> to check the presense of the function that we'd be calling.
  // if none of them have batch mode, then fall back to sequential.
  // use 3 dummy vars to indicate that batch mode is prefered if both apis are defined.

  // case when both transforms are identity.  last param is dummy.
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value && ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value, // &&
                                                                                                                                                          //					::std::is_same<size_t,
                                                                                                                                                          //					 decltype(std::declval<HT>().operator()(
                                                                                                                                                          //							 std::declval<PRETRANS_VAL_TYPE const *>(),
                                                                                                                                                          //							 std::declval<size_t>(),
                                                                                                                                                          //							 std::declval<HASH_VAL_TYPE *>()))>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, int) const
      -> decltype(::std::declval<HT>()(k, count, out), size_t())
  {
    h(k, count, out);
    return count;
    // no last part
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value && ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, int) const
      -> decltype(::std::declval<HT>()(::std::declval<Key>()), size_t())
  {
    // no batched part.
    return 0;
  }
  // pretrans is not identity, post is identity.
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      h(trans_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = h(trans_buf[j]);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, int, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        trans_buf[j] = trans(k[i + j]);
      h(trans_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, long, int) const
      -> decltype(::std::declval<PrT>()(::std::declval<Key>()), ::std::declval<HT>()(::std::declval<PRETRANS_VAL_TYPE>()), size_t())
  {
    // no batched part.
    return 0;
  }
  // posttrans is not identity, post is identity.
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, int) const
      -> decltype(::std::declval<HT>()(k, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      h(k + i, batch_size, hash_buf);
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, int) const
      -> decltype(::std::declval<HT>()(*k), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        hash_buf[j] = h(k[i + j]);
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, long) const
      -> decltype(::std::declval<HT>()(k, count, hash_buf), ::std::declval<PoT>()(*hash_buf), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      h(k + i, batch_size, hash_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(hash_buf[j]);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, long) const
      -> decltype(::std::declval<HT>()(::std::declval<Key>()), ::std::declval<PoT>()(::std::declval<HASH_VAL_TYPE>()), size_t())
  {
    // no batched part.
    return 0;
  }
  // ==== none are identity
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      h(trans_buf, batch_size, hash_buf);
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      for (j = 0; j < batch_size; ++j)
        hash_buf[j] = h(trans_buf[j]);
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, int, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        trans_buf[j] = trans(k[i + j]);
      h(trans_buf, batch_size, hash_buf);
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, long, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t())
  {

    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        hash_buf[j] = h(trans(k[i + j]));
      posttrans(hash_buf, batch_size, out + i);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, long) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(*hash_buf), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      h(trans_buf, batch_size, hash_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(hash_buf[j]);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, long) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(*hash_buf), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(h(trans_buf[j]));
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, int, long) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(*hash_buf), size_t())
  {
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        trans_buf[j] = trans(k[i + j]);
      h(trans_buf, batch_size, hash_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(hash_buf[j]);
    }
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, long, long, long) const
      -> decltype(::std::declval<PrT>()(::std::declval<Key>()),
                  ::std::declval<HT>()(::std::declval<PRETRANS_VAL_TYPE>()),
                  ::std::declval<PoT>()(::std::declval<HASH_VAL_TYPE>()), size_t())
  {
    // no batched part.
    return 0;
  }

public:
  inline void operator()(Key const *k, size_t const &count, result_type *out) const
  {
    size_t max = count - (count & (batch_size - 1));
    max = this->batch_op(k, max, out, 0, 0, 0); // 0 has type int....

    for (size_t i = max; i < count; ++i)
    {
      out[i] = this->operator()(k[i]);
    }
  }

  template <typename V>
  inline void operator()(::std::pair<Key, V> const *x, size_t const &count, result_type *out) const
  {
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        key_buf[j] = x[i + j].first;
      this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
    }
    // last part
    for (; i < count; ++i)
    {
      out[i] = this->operator()(x[i].first);
    }
  }

  template <typename V>
  inline void operator()(::std::pair<const Key, V> const *x, size_t const &count, result_type *out) const
  {
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        key_buf[j] = x[i + j].first;
      this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
    }
    // last part
    for (; i < count; ++i)
    {
      out[i] = this->operator()(x[i].first);
    }
  }
};
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform,
          template <typename> class PostTransform>
constexpr uint8_t TransformedHash<Key, Hash, PreTransform, PostTransform>::batch_size;

// TODO:  [ ] batch mode transformed_predicate
//		[ ] batch mode transformed_comparator

} // namespace hash

} // namespace fsc

#endif /* HASH_HPP_ */
