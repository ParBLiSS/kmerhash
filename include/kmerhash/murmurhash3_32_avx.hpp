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
#ifndef MURMUR3_32_AVX_HPP_
#define MURMUR3_32_AVX_HPP_

#include <type_traits> // enable_if
#include <cstring>     // memcpy
#include <stdexcept>   // logic error
#include <stdint.h>    // std int strings
#include <iostream>    // cout

#include "utils/filter_utils.hpp"
#include "utils/transform_utils.hpp"
#include "kmerhash/mem_utils.hpp"
#include "kmerhash/math_utils.hpp"

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
  const __m256i mix_const1;
  const __m256i mix_const2;
  const __m256i c1;
  const __m256i c2;
  const __m256i c4;
  const __m256i length;


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void rotl(uint8_t const & rot, __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {

    __m256i tt0, tt1, tt2, tt3;

    if (VEC_CNT >= 1) {
      tt1 = _mm256_slli_epi32(t0, rot);   // SLLI: L1 C1 p0
      tt0 = _mm256_srli_epi32(t0, (32 - rot));   // SRLI: L1 C1 p0
    }
    if (VEC_CNT >= 2) {
      tt3 = _mm256_slli_epi32(t1, rot);   // SLLI: L1 C1 p0
      tt2 = _mm256_srli_epi32(t1, (32 - rot));   // SRLI: L1 C1 p0
    }
    if (VEC_CNT >= 1) t0 = _mm256_or_si256(tt1, tt0);  // OR: L1 C0.33 p015
    if (VEC_CNT >= 2) t1 = _mm256_or_si256(tt3, tt2);  // OR: L1 C0.33 p015

    if (VEC_CNT >= 3) {
      tt1 = _mm256_slli_epi32(t2, rot);   // SLLI: L1 C1 p0
      tt0 = _mm256_srli_epi32(t2, (32 - rot));   // SRLI: L1 C1 p0
    }
    if (VEC_CNT >= 4) {
      tt3 = _mm256_slli_epi32(t3, rot);   // SLLI: L1 C1 p0
      tt2 = _mm256_srli_epi32(t3, (32 - rot));   // SRLI: L1 C1 p0
    }

    if (VEC_CNT >= 3) t2 = _mm256_or_si256(tt1, tt0);  // OR: L1 C0.33 p015
    if (VEC_CNT >= 4) t3 = _mm256_or_si256(tt3, tt2);  // OR: L1 C0.33 p015
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }

  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void mul(__m256i const & mult, __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    if (VEC_CNT >= 1) t0 = _mm256_mullo_epi32(t0, mult); // sse   // Lat10, CPI2
    if (VEC_CNT >= 2) t1 = _mm256_mullo_epi32(t1, mult); // sse   // Lat10, CPI2
    if (VEC_CNT >= 3) t2 = _mm256_mullo_epi32(t2, mult); // sse   // Lat10, CPI2
    if (VEC_CNT >= 4) t3 = _mm256_mullo_epi32(t3, mult); // sse   // MULLO: L10 C2 2p0
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void update_part2(__m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    
    rotl<VEC_CNT>(15, t0, t1, t2, t3);
    mul<VEC_CNT>(this->c2, t0, t1, t2, t3);
    
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }



  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT, bool add_prev_iter>
  FSC_FORCE_INLINE void add_xor(__m256i const & adder, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3,
                                     __m256i const &t0, __m256i const &t1, __m256i const &t2, __m256i const &t3) const
  {
    // first do the add.
    if (add_prev_iter) {
      if (VEC_CNT >= 1) h0 = _mm256_add_epi32(h0, adder);
      if (VEC_CNT >= 2) h1 = _mm256_add_epi32(h1, adder);
    }
    if (VEC_CNT >= 1) h0 = _mm256_xor_si256(h0, t0); // sse
    if (VEC_CNT >= 2) h1 = _mm256_xor_si256(h1, t1); // sse

    if (add_prev_iter) {
      if (VEC_CNT >= 3) h2 = _mm256_add_epi32(h2, adder);
      if (VEC_CNT >= 4) h3 = _mm256_add_epi32(h3, adder);                                   // ADD: L1 C0.5 p15
    }
    if (VEC_CNT >= 3) h2 = _mm256_xor_si256(h2, t2); // sse
    if (VEC_CNT >= 4) h3 = _mm256_xor_si256(h3, t3); // sse                                    // XOR: L1 C0.33 p015
  }



      // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT, bool add_prev_iter>
  FSC_FORCE_INLINE void update_part3(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3,
                                     __m256i const &t0, __m256i const &t1, __m256i const &t2, __m256i const &t3) const
  {
    add_xor<VEC_CNT, add_prev_iter>(this->c4, h0, h1, h2, h3, t0, t1, t2, t3);


    rotl<VEC_CNT>(13, h0, h1, h2, h3);
//    mul<VEC_CNT>(this->c3, h0, h1, h2, h3);

    __m256i hh0, hh1, hh2, hh3;
    hh0 = _mm256_slli_epi32(h0, 2);
    hh1 = _mm256_slli_epi32(h1, 2);
    hh2 = _mm256_slli_epi32(h2, 2);
    hh3 = _mm256_slli_epi32(h3, 2);

    // do 1x + 4x instead of mul by 5.
    h0 = _mm256_add_epi32(h0, hh0);
    h1 = _mm256_add_epi32(h1, hh1);
    h2 = _mm256_add_epi32(h2, hh2);
    h3 = _mm256_add_epi32(h3, hh3);
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
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // // should have 0 idle latency cyles and 0 cpi cycles here.
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const1, h0, h1, h2, h3);

    // // should have 1 idle latency cyles and 2 cpi cycles here.

    shift_xor<VEC_CNT>(13, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const2, h0, h1, h2, h3);

    // // latencies.
    // // h3  Lat 1, cpi 2
    // // h0  Lat 4, cpi 2

    // // expect Lat 0, cycle 1
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);

  }



  /// fmix32 for 16 elements at a time.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32_part2(__m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // should have 1 idle latency cyles and 2 cpi cycles here.

    shift_xor<VEC_CNT>(13, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const2, h0, h1, h2, h3);

    // latencies.
    // h3  Lat 1, cpi 2
    // h0  Lat 4, cpi 2

    // expect Lat 0, cycle 1
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);

  }

protected:
  // LATENCIES for instruction with latency > 1
  // 1. mullo_epi16 has latency of 5 cycles, CPI of 1 to 0.5 cycles - need unroll 2x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 2. mullo_epi32 has latency of 10 cycles, CPI of 1 to 0.5 cycles - need unroll 3-4x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 3. _mm256_permutevar8x32_epi32, _mm256_permute4x64_epi64, and _mm256_insertf128_si256  have latency of 3, CPI of 1.
  // 4. note that most extract calls have latency of 3 and CPI of 1, except for _mm256_extracti128_si256, which has latency of 1.
  // 5. _mm256_insertf128_si256 has latency of 3. and CPI of 1.
  const __m256i permute1;
  const __m256i permute16;
  const __m256i shuffle0; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle1; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle2; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i shuffle3; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m256i ones;
  const __m256i zeros;
  mutable __m256i seed;


  explicit Murmur32AVX(__m256i _seed) : mix_const1(_mm256_set1_epi32(0x85ebca6bU)),
                                        mix_const2(_mm256_set1_epi32(0xc2b2ae35U)),
                                        c1(_mm256_set1_epi32(0xcc9e2d51U)),
                                        c2(_mm256_set1_epi32(0x1b873593U)),
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
                                        zeros(_mm256_setzero_si256()),
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

    assert((nstreams <= 32) && "maximum number of streams is 32");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m256i h0, h1, h2, h3;

    // do the full ones first.
    uint8_t blocks = (nstreams + 7) >> 3; // divide by 8.
    switch (blocks)
    {
    case 1:
      hash<1>(key, h0, h1, h2, h3);
      break;
    case 2:
      hash<2>(key, h0, h1, h2, h3);
      break;
    case 3:
      hash<3>(key, h0, h1, h2, h3);
      break;
    case 4:
      hash<4>(key, h0, h1, h2, h3);
      break;
    default:
      break;
    }

    blocks = nstreams >> 3;
    if (blocks >= 1) _mm256_storeu_si256((__m256i *)out, h0);
    if (blocks >= 2) _mm256_storeu_si256((__m256i *)(out + 8), h1);
    if (blocks >= 3) _mm256_storeu_si256((__m256i *)(out + 16), h2);
    if (blocks >= 4) _mm256_storeu_si256((__m256i *)(out + 24), h3);
    

    uint8_t rem = nstreams & 7; // remainder.
    if (rem > 0)
    {
      // write remainders
      switch (blocks)
      {
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
    _mm256_storeu_si256((__m256i *)out, h0);
    _mm256_storeu_si256((__m256i *)(out + 8), h1);
    _mm256_storeu_si256((__m256i *)(out + 16), h2);
    _mm256_storeu_si256((__m256i *)(out + 24), h3);
  }

  /// NOTE: multiples of 32.
  // USING load, INSERT plus unpack is FASTER than i32gather.
  // load with an offset from start of key.
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

    // // load 8 keys at a time, 16 bytes each time,
    // j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + off);     // SSE3  // L3 C0.5 p23
    // j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4) + off); // SSE3
    // j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); // SSE3
    // j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 5) + off); // SSE3

    // k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), (j2), 1); //aa'AA'ee'EE'  // L3 C1 p5
    // k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), (j3), 1); //bb'BB'ff'FF'

    // j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + off); // SSE3
    // j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 6) + off); // SSE3
    // j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + off); // SSE3
    // j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 7) + off); // SSE3

    // // get the 32 byte vector.
    // // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
    // k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), (j2), 1); //CG
    // k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), (j3), 1); //DH

    // load 8 keys at a time, 16 bytes each time,
    j0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + off);     // SSE3  // L3 C0.5 p23
    j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); // SSE3
    j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + off); // SSE3
    j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + off); // SSE3

    // get the 32 byte vector.
    // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
    // y m i version, L4, C0.5, p015 p23
    k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), *(reinterpret_cast<const __m128i *>(key + 4) + off), 1); //aa'AA'ee'EE'
    k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), *(reinterpret_cast<const __m128i *>(key + 5) + off), 1); //bb'BB'ff'FF'
    k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), *(reinterpret_cast<const __m128i *>(key + 6) + off), 1); //CG
    k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), *(reinterpret_cast<const __m128i *>(key + 7) + off), 1); //DH

    
    // MERGED shuffling and update part 1.
    // now unpack and update
    t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'efe'f'                           // L1 C1 p5   
    t2 = _mm256_unpackhi_epi32(k0, k1); 
    t1 = _mm256_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
    
    k0 = _mm256_unpacklo_epi64(t0, t1);   // abcdefgh
    k1 = _mm256_unpackhi_epi64(t0, t1);   // a'b'c'd'e'f'g'h'

    t0 = _mm256_mullo_epi32(k0, this->c1); // avx                                  // L10 C2 p0

    t3 = _mm256_unpackhi_epi32(k2, k3);                                            // L1 C1 p5
    
    t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

    
    k2 = _mm256_unpacklo_epi64(t2, t3);
    k3 = _mm256_unpackhi_epi64(t2, t3);

    t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
    t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2

    // latency:  should be Lat3, C2 for temp
    // update part 2.
    this->template update_part2<4>(t0, t1, t2, t3);

    // loading 8 32-byte keys is slower than loading 8 16-byte keys.
  }

  // USING load, INSERT plus unpack is FASTER than i32gather.
  // also probably going to be faster for non-power of 2 less than 8 (for 9 to 15, this is needed anyways).
  //   because we'd need to shift across lane otherwise.
  // load with an offset from start of key, and load partially.  blocks of 16,
  template <size_t KEY_LEN = sizeof(T), size_t off = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15)>
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
    j1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + off); // SSE3
    j2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + off); // SSE3
    j3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + off); // SSE3

    // get the 32 byte vector.
    // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
    // y m i version, L4, C0.5, p015 p23
    k0 = _mm256_inserti128_si256(_mm256_castsi128_si256(j0), *(reinterpret_cast<const __m128i *>(key + 4) + off), 1); //aa'AA'ee'EE'
    k1 = _mm256_inserti128_si256(_mm256_castsi128_si256(j1), *(reinterpret_cast<const __m128i *>(key + 5) + off), 1); //bb'BB'ff'FF'
    k2 = _mm256_inserti128_si256(_mm256_castsi128_si256(j2), *(reinterpret_cast<const __m128i *>(key + 6) + off), 1); //CG
    k3 = _mm256_inserti128_si256(_mm256_castsi128_si256(j3), *(reinterpret_cast<const __m128i *>(key + 7) + off), 1); //DH

    // ZERO the leading bytes to keep just the lower.
    // latency of 3 and CPI of 1, so can do the masking here...
    k0 = _mm256_and_si256(k0, mask);
    k1 = _mm256_and_si256(k1, mask);
    k2 = _mm256_and_si256(k2, mask);
    k3 = _mm256_and_si256(k3, mask);
    
    // MERGED shuffling and update part 1.
    // now unpack and update
    // RELY ON COMPILER OPTIMIZATION HERE TO REMOVE THE CONDITIONAL CHECKS
    t0 = _mm256_unpacklo_epi32(k0, k1); // aba'b'efe'f'                           // L1 C1 p5    
    if (rem > 8) t2 = _mm256_unpackhi_epi32(k0, k1);
    t1 = _mm256_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
    
    k0 = _mm256_unpacklo_epi64(t0, t1);   // abcdefgh
    if (rem > 4) k1 = _mm256_unpackhi_epi64(t0, t1);   // a'b'c'd'e'f'g'h'

    t0 = _mm256_mullo_epi32(k0, this->c1); // avx                                  // L10 C2 p0

    if (rem > 8) t3 = _mm256_unpackhi_epi32(k2, k3);                                            // L1 C1 p5

    if (rem > 4) t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

    
    if (rem > 8) k2 = _mm256_unpacklo_epi64(t2, t3);
    if (rem > 12) k3 = _mm256_unpackhi_epi64(t2, t3);

    if (rem > 8) t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
    if (rem > 12) t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2
    // now unpack and update

    // latency:  should be Lat3, C2 for temp
    // update part 2.  note that we compute for parts that have non-zero values, determined in blocks of 4 bytes.
    this->template update_part2<((rem + 3) >> 2)>(t0, t1, t2, t3);
  }

  /// NOTE: non-power of 2 length keys ALWAYS use AVX gather, which may be slower.
  // for hasing non multiple of 16 and non power of 2.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T), size_t nblocks = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15),
            typename std::enable_if<((KEY_LEN & (KEY_LEN - 1)) > 0) && ((KEY_LEN & 15) > 0), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
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
        t10, t11, t12, t13,     //t14, t15, t16, t17,
        t20, t21, t22, t23,     //t24, t25, t26, t27,
        t30, t31, t32, t33      //, t34, t35, t36, t37
        ;

    // read input, 8 keys at a time.  need 4 rounds.
    h0 = h1 = h2 = h3 = seed;

    // read input, 2 keys per vector.
    // combined load and update_part1 and update_part2 (data parallel part.)
    size_t i = 0;
    if (i < nblocks) {
      if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
    
      // now do part 3.
      this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
      ++i;
    }
    
    for (; i < nblocks; ++i)
    {
      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
    
      // now do part 3.
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
    }
    // latency: h3: L0, C0.  h0: L1,C2



    // DO LAST ADD FROM UPDATE32
    // do remainder.
    if (rem > 0)
    { // NOT multiple of 16.

      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      if (VEC_CNT >= 1)  this->load_partial16(key, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_partial16(key + 8, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_partial16(key + 16, t20, t21, t22, t23); // , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_partial16(key + 24, t30, t31, t32, t33); // , t34, t35, t36, t37);
    }

    // For the last b < 4 bytes, we do not do full update.
    if (rem >= 4)
    {
      if (i == 0)
        this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
      else
        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
    }
    if (rem >= 8) this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
    if (rem >= 12) this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);

    __m256i t0, t1, t2, t3;
    if ((rem & 3) > 0)
    { // has part of an int.
      //  LAST PART OF UPDATE, which is an xor, then fmix.
      switch (rem >> 2)
      {
      case 0:
        t0 = t00;
        t1 = t10;
        t2 = t20;
        t3 = t30;
        break;
      case 1:
        t0 = t01;
        t1 = t11;
        t2 = t21;
        t3 = t31;
        break;
      case 2:
        t0 = t02;
        t1 = t12;
        t2 = t22;
        t3 = t32;
        break;
      case 3:
        t0 = t03;
        t1 = t13;
        t2 = t23;
        t3 = t33;
        break;
      }
    }


    if (rem >= 4) {                                        // complete the prev update_part3
    // should have 0 idle latency cyles and 0 cpi cycles here.
        if (VEC_CNT >= 1) h0 = _mm256_add_epi32(h0, this->c4);                // avx
        if (VEC_CNT >= 2) h1 = _mm256_add_epi32(h1, this->c4);                // avx
        if (VEC_CNT >= 3) h2 = _mm256_add_epi32(h2, this->c4);                // avx
        if (VEC_CNT >= 4) h3 = _mm256_add_epi32(h3, this->c4);                // avx
      }

      if ((rem & 3) > 0)  {                                  // has partial int
        if (VEC_CNT >= 1) h0 = _mm256_xor_si256(h0, t0);                      // avx
        if (VEC_CNT >= 2) h1 = _mm256_xor_si256(h1, t1);                      // avx
        if (VEC_CNT >= 3) h2 = _mm256_xor_si256(h2, t2);                      // avx
        if (VEC_CNT >= 4) h3 = _mm256_xor_si256(h3, t3);                      // avx
      }

      if (VEC_CNT >= 1) h0 = _mm256_xor_si256(h0, this->length);              // sse
      if (VEC_CNT >= 2) h1 = _mm256_xor_si256(h1, this->length);              // sse
      if (VEC_CNT >= 3) h2 = _mm256_xor_si256(h2, this->length);              // sse
      if (VEC_CNT >= 4) h3 = _mm256_xor_si256(h3, this->length);              // sse

      //    // should have 0 idle latency cyles and 0 cpi cycles here.
      //
      // if (VEC_CNT >= 1) {
      //   if (rem >= 4)                                         // complete the prev update_part3
      //     h0 = _mm256_add_epi32(h0, this->c4);                // avx
      //   if ((rem & 3) > 0)
      //     h0 = _mm256_xor_si256(h0, t0);                      // avx
      //   h0 = _mm256_xor_si256(h0, this->length);              // sse
      //   // h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
      //   // h0 = _mm256_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
      // }
      // if (VEC_CNT >= 2) {
      //   if (rem >= 4)                                         // complete the prev update_part3
      //     h1 = _mm256_add_epi32(h1, this->c4);                // avx
      //   if ((rem & 3) > 0)                                    // has partial int
      //     h1 = _mm256_xor_si256(h1, t1);                      // avx
      //   h1 = _mm256_xor_si256(h1, this->length);              // sse
      //   // h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
      //   // h1 = _mm256_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
      // }
      // if (VEC_CNT >= 3) {
      //   if (rem >= 4)                                         // complete the prev update_part3
      //     h2 = _mm256_add_epi32(h2, this->c4);                // avx
      //   if ((rem & 3) > 0)                                    // has partial int
      //     h2 = _mm256_xor_si256(h2, t2);                      // avx
      //   h2 = _mm256_xor_si256(h2, this->length);              // sse
      //   // h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      //   // h2 = _mm256_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
      // }
      // if (VEC_CNT >= 4) {
      //   if (rem >= 4)                                         // complete the prev update_part3
      //     h3 = _mm256_add_epi32(h3, this->c4);                // avx
      //   if ((rem & 3) > 0)                                    // has partial int
      //     h3 = _mm256_xor_si256(h3, t3);                      // avx
      //   h3 = _mm256_xor_si256(h3, this->length);              // sse
      //   // h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      //   // h3 = _mm256_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
      // }
        
        
        // Latency: h3: L1 C2, h0:L1 C2
        // this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);
        this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
    
  }

  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // multiple of 16 that are greater than 16.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<((KEY_LEN & 15) == 0) && (KEY_LEN > 16), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // we now assume no specific layout, so we need to load 8 at a time.

    // load 16 bytes at a time.
    const int nblocks = KEY_LEN >> 4;

    __m256i t00, t01, t02, t03, //t04, t05, t06, t07,
        t10, t11, t12, t13,     //t14, t15, t16, t17,
        t20, t21, t22, t23,     //t24, t25, t26, t27,
        t30, t31, t32, t33      //, t34, t35, t36, t37
        ;

    // read input, 8 keys at a time.  need 4 rounds.
    h0 = h1 = h2 = h3 = seed;

    int i = 0;
    if (i < nblocks)
    {
      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); //load16 , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
  
      // now do part 3.   ORDER MATTERS. from low to high.
      this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
      ++i;
    }


    for (; i < nblocks; ++i)
    {
      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_stride16(key + 8, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_stride16(key + 16, i, t20, t21, t22, t23); //load16 , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_stride16(key + 24, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
  
      // now do part 3.   ORDER MATTERS. from low to high.
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
    }
    // latency: h3: L0, C0.  h0: L1,C2

    // should have 0 idle latency cyles and 0 cpi cycles here.
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm256_add_epi32(h3, this->c4);
    //   h3 = _mm256_xor_si256(h3, this->length);              // sse
    //   h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    //   h3 = _mm256_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 3:
    //   h2 = _mm256_add_epi32(h2, this->c4);
    //   h2 = _mm256_xor_si256(h2, this->length);              // sse
    //   h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    //   h2 = _mm256_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 2:
    //   h1 = _mm256_add_epi32(h1, this->c4);
    //   h1 = _mm256_xor_si256(h1, this->length);              // sse
    //   h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    //   h1 = _mm256_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 1:
    //   h0 = _mm256_add_epi32(h0, this->c4);
    //   h0 = _mm256_xor_si256(h0, this->length);              // sse
    //   h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    //   h0 = _mm256_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // }
    // //    // should have 0 idle latency cyles and 0 cpi cycles here.
    // //
    // this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);
    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
      this->length, this->length, this->length, this->length);
  
      this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
  
  }


  FSC_FORCE_INLINE void load16(T const *key, __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    __m256i k0, k1, k2, k3;

    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // SSE3  // L3 C0.5 p23
    k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 2)); // SSE3
    k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); // SSE3
    k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 6)); // SSE3

    // MERGED shuffling and update part 1.
    // unpack to get the right set.  require 8 unpack ops
    // start.    aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
    t0 = _mm256_unpacklo_epi32(k0, k1); // aca'c' bdb'd'      
    t1 = _mm256_unpacklo_epi32(k2, k3); // ege'g' fhf'h'
    t2 = _mm256_unpackhi_epi32(k0, k1);  // ACA'C' BDB'D'

    k0 = _mm256_unpacklo_epi64(t0, t1);   // aceg bdfh
    k1 = _mm256_unpackhi_epi64(t0, t1);   // a'c'e'g' b'd'f'h'

    t0 = _mm256_mullo_epi32(k0, this->c1); // avx  // Lat10, CPI2

    t3 = _mm256_unpackhi_epi32(k2, k3); // EGE'G' FHF'H'
    
    t1 = _mm256_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2


    k2 = _mm256_unpacklo_epi64(t2, t3);   // ACEG BDFH
    k3 = _mm256_unpackhi_epi64(t2, t3);   // A'C'E'G' B'D'F'H'

    // one more time.
    t2 = _mm256_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
    t3 = _mm256_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2

    // latency:  should be Lat3, C2 for temp
    // update part 2.
    this->template update_part2<4>(t0, t1, t2, t3);
  }

  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 16), int>::type = 1> // 16 bytes exactly.
  FSC_FORCE_INLINE void
  hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
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
    if (VEC_CNT >= 1)   this->load16(key, t00, t01, t02, t03);
    if (VEC_CNT >= 2)   this->load16(key + 8, t10, t11, t12, t13);
    if (VEC_CNT >= 3)   this->load16(key + 16, t20, t21, t22, t23);
    if (VEC_CNT >= 4)   this->load16(key + 24, t30, t31, t32, t33);

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
    // switch (VEC_CNT)
    // {
    // case 4:
    //   // h3 = _mm256_add_epi32(h3, this->c4);             // L1 C0.5 p15
    //   // h3 = _mm256_xor_si256(h3, this->length);         // L1 C0.33 p015
    //   h3 = _mm256_permutevar8x32_epi32(h3, permute16); // L3 C1 p5
    // case 3:
    //   // h2 = _mm256_add_epi32(h2, this->c4);
    //   // h2 = _mm256_xor_si256(h2, this->length);         // sse
    //   h2 = _mm256_permutevar8x32_epi32(h2, permute16); // latency 3, CPI 1
    // case 2:
    //   // h1 = _mm256_add_epi32(h1, this->c4);
    //   // h1 = _mm256_xor_si256(h1, this->length);         // sse
    //   h1 = _mm256_permutevar8x32_epi32(h1, permute16); // latency 3, CPI 1
    // case 1:
    //   // h0 = _mm256_add_epi32(h0, this->c4);
    //   // h0 = _mm256_xor_si256(h0, this->length);         // sse
    //   h0 = _mm256_permutevar8x32_epi32(h0, permute16); // latency 3, CPI 1
    // }

    if (VEC_CNT >= 1) h0 = _mm256_permutevar8x32_epi32(h0, permute16); // L3 C1 p5
    if (VEC_CNT >= 2) h1 = _mm256_permutevar8x32_epi32(h1, permute16); // L3 C1 p5
    if (VEC_CNT >= 3) h2 = _mm256_permutevar8x32_epi32(h2, permute16); // L3 C1 p5
    if (VEC_CNT >= 4) h3 = _mm256_permutevar8x32_epi32(h3, permute16); // L3 C1 p5
    
    // ADD: L1 C0.5 p15.   XOR: L1 C0.33 p015
    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3,   
      this->length, this->length, this->length, this->length);

    // Latency: h3: L1 C2, h0:L1 C2
    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
  }


  // // this is using a number of different ports with biggest contention over p0.
  // // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  // FSC_FORCE_INLINE void load8(T const *key, __m256i &t0, __m256i &t1) const
  // {
  //   __m256i k0, k1;

  //   k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); 
  //   k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // L3 C0.5 p23
    
  //   // MERGED SHUFFLE AND UPDATE_PARTIAL
  //   // make aebfcgdh and AEBFCGDH .  Order matters.  do lower first.
  //   t0 = _mm256_slli_si256(k1, 4); // 0eEf0gGh                          
  //   t1 = _mm256_srli_si256(k0, 4); // AbB0CdD0                          // L1 C1 p0

  //   // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
  //   t0 = _mm256_blend_epi32(k0, t0, 0xAA);                              // L1 C0.33 p015           
  //   t0 = _mm256_mullo_epi32(t0, this->c1); // avx                       // L10, C2, p0

  //   t1 = _mm256_blend_epi32(t1, k1, 0xAA); // Lat1, cpi 0.3.  // AEBF
  //   t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // L10, C2, p0


  // }

  // this is using a number of different ports with biggest contention over p0.
  // we can probably realistically load all 16 keys at a time to hide the latency of mullo
  FSC_FORCE_INLINE void load8(T const *key, __m256i &t0, __m256i &t1, __m256i &t2, __m256i &t3) const
  {
    __m256i k0, k1, k2, k3;

    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key));     // L3 C0.5 p23
    k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 4)); 
    k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); 
    k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 12)); 

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // make aebfcgdh and AEBFCGDH .  Order matters.  do lower first.
    t1 = _mm256_srli_si256(k0, 4); // AbB0CdD0                          // L1 C1 p0
    t0 = _mm256_slli_si256(k1, 4); // 0eEf0gGh                          
    t3 = _mm256_srli_si256(k2, 4); // AbB0CdD0                          // L1 C1 p0
    t2 = _mm256_slli_si256(k3, 4); // 0eEf0gGh                          

    // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
    t0 = _mm256_blend_epi32(k0, t0, 0xAA);                              // L1 C0.33 p015           
    t1 = _mm256_blend_epi32(t1, k1, 0xAA); // Lat1, cpi 0.3.  // AEBF
    t2 = _mm256_blend_epi32(k2, t2, 0xAA);                              // L1 C0.33 p015           
    t3 = _mm256_blend_epi32(t3, k3, 0xAA); // Lat1, cpi 0.3.  // AEBF

    t0 = _mm256_mullo_epi32(t0, this->c1); // avx                       // L10, C2, p0
    t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // L10, C2, p0
    t2 = _mm256_mullo_epi32(t2, this->c1); // avx                       // L10, C2, p0
    t3 = _mm256_mullo_epi32(t3, this->c1); // avx  // L10, C2, p0


  }

  // hashing 32 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 32 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 8), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
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
    // if (VEC_CNT >= 1) load8(key, t00, t01);
    // if (VEC_CNT >= 2) load8(key + 4, t10, t11);
    // if (VEC_CNT >= 3) load8(key + 8, t20, t21);
    // if (VEC_CNT >= 4) load8(key + 12, t30, t31);
    if (VEC_CNT >= 1) load8(key, t00, t01, t10, t11);
    if (VEC_CNT >= 3) load8(key + 16, t20, t21, t30, t31);


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
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm256_add_epi32(h3, this->c4);
    //   h3 = _mm256_xor_si256(h3, this->length);        // sse
    //   h3 = _mm256_permutevar8x32_epi32(h3, permute1); // latency 3, CPI 1
    // case 3:
    //   h2 = _mm256_add_epi32(h2, this->c4);
    //   h2 = _mm256_xor_si256(h2, this->length);        // sse
    //   h2 = _mm256_permutevar8x32_epi32(h2, permute1); // latency 3, CPI 1
    // case 2:
    //   h1 = _mm256_add_epi32(h1, this->c4);
    //   h1 = _mm256_xor_si256(h1, this->length);        // sse
    //   h1 = _mm256_permutevar8x32_epi32(h1, permute1); // latency 3, CPI 1
    // case 1:
    //   h0 = _mm256_add_epi32(h0, this->c4);
    //   h0 = _mm256_xor_si256(h0, this->length);        // sse
    //   h0 = _mm256_permutevar8x32_epi32(h0, permute1); // latency 3, CPI 1
    // }

    // // Latency: h3: L1 C2, h0:L1 C2
    // this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
    if (VEC_CNT >= 1) h0 = _mm256_permutevar8x32_epi32(h0, permute1); // L3 C1 p5
    if (VEC_CNT >= 2) h1 = _mm256_permutevar8x32_epi32(h1, permute1); // L3 C1 p5
    if (VEC_CNT >= 3) h2 = _mm256_permutevar8x32_epi32(h2, permute1); // L3 C1 p5
    if (VEC_CNT >= 4) h3 = _mm256_permutevar8x32_epi32(h3, permute1); // L3 C1 p5
    
    // ADD: L1 C0.5 p15.   XOR: L1 C0.33 p015
    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3,   
      this->length, this->length, this->length, this->length);

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
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 4 bytes
    //     abcd efgh
    // k0  ---- ----

    __m256i t0, t1, t2, t3;

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2

    // 16 keys per vector. can potentially do 2 iters.
    if (VEC_CNT >= 1) t0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3  L3 C1 p23
    if (VEC_CNT >= 2) t1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); // SSE3  L3 C1 p23
    if (VEC_CNT >= 3) t2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3  L3 C1 p23
    if (VEC_CNT >= 4) t3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24)); // SSE3  L3 C1 p23

    if (VEC_CNT >= 1) t0 = _mm256_mullo_epi32(t0, this->c1); // AVX  L10 C2 p0
    if (VEC_CNT >= 2) t1 = _mm256_mullo_epi32(t1, this->c1); // AVX  L10 C2 p0
    if (VEC_CNT >= 3) t2 = _mm256_mullo_epi32(t2, this->c1); // AVX  L10 C2 p0
    if (VEC_CNT >= 4) t3 = _mm256_mullo_epi32(t3, this->c1); // AVX  L10 C2 p0
    

    // switch (VEC_CNT)
    // {
    // case 4:
    //   t3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24)); // SSE3
    //   t3 = _mm256_mullo_epi32(t3, this->c1);                                // avx  // Lat10, CPI2
    // case 3:
    //   t2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3
    //   t2 = _mm256_mullo_epi32(t2, this->c1);                                // avx  // Lat10, CPI2
    // case 2:
    //   t1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); // SSE3
    //   t1 = _mm256_mullo_epi32(t1, this->c1);                               // avx  // Lat10, CPI2
    // case 1:
      
    //   t0 = _mm256_mullo_epi32(t0, this->c1);                           // avx  // Lat10, CPI2
    // }

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
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm256_add_epi32(h3, this->c4);
    //   h3 = _mm256_xor_si256(h3, this->length);              // sse
    //   h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    //   h3 = _mm256_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 3:
    //   h2 = _mm256_add_epi32(h2, this->c4);
    //   h2 = _mm256_xor_si256(h2, this->length);              // sse
    //   h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    //   h2 = _mm256_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 2:
    //   h1 = _mm256_add_epi32(h1, this->c4);
    //   h1 = _mm256_xor_si256(h1, this->length);              // sse
    //   h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    //   h1 = _mm256_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 1:
    //   h0 = _mm256_add_epi32(h0, this->c4);
    //   h0 = _mm256_xor_si256(h0, this->length);              // sse
    //   h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    //   h0 = _mm256_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // }
    // //    // should have 0 idle latency cyles and 0 cpi cycles here.
    // //
    // this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);

    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
      this->length, this->length, this->length, this->length);

    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);

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
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     abcdefgh ijklmnop
    // k0  -------- --------

    __m256i k0, k1;
    __m256i t0, t1, t2, t3;

    if (VEC_CNT > 2)
    {
      k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3
      k1 = _mm256_permute4x64_epi64(k1, 0xd8);                              // AVX2, latency 3, CPI 1
    }

    // 16 keys per vector. can potentially do 2 iters.
    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3
    // permute across lane, 64bits at a time, with pattern 0 2 1 3 -> 11011000 == 0xD8
    k0 = _mm256_permute4x64_epi64(k0, 0xd8); // AVX2, latency 3, CPI 1

    // result abcd ijkl efgh mnop
    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // transform to a0b0c0d0 e0f0g0h0.  interleave with 0.
    if (VEC_CNT > 2)
    {
      // yz12
      t3 = _mm256_unpackhi_epi16(k1, zeros);  // AVX2, latency 1, CPI 1
      t3 = _mm256_mullo_epi32(t3, this->c1); // avx  // Lat10, CPI2

      t2 = _mm256_unpacklo_epi16(k1, zeros);  // AVX2, latency 1, CPI 1
      t2 = _mm256_mullo_epi32(t2, this->c1); // avx  // Lat10, CPI2
    }
    // ijkl
    t1 = _mm256_unpackhi_epi16(k0, zeros);  // AVX2, latency 1, CPI 1
    t1 = _mm256_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2

    t0 = _mm256_unpacklo_epi16(k0, zeros); // AVX2, latency 1, CPI 1
    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
    t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2
    // qrst
    // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
    h0 = h1 = h2 = h3 = seed;

    // rotl32
    //t0 = rotl32(t0, 15);
    if (VEC_CNT > 2)
    {
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
    if (VEC_CNT > 2)
    {
      h3 = _mm256_xor_si256(h3, t3);                        // avx
      h3 = _mm256_xor_si256(h3, this->length);              // sse
      h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      h3 = _mm256_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

      h2 = _mm256_xor_si256(h2, t2);                        // avx
      h2 = _mm256_xor_si256(h2, this->length);              // sse
      h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      h2 = _mm256_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    h1 = _mm256_xor_si256(h1, t1);                        // avx
    h1 = _mm256_xor_si256(h1, this->length);              // sse
    h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h1 = _mm256_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    h0 = _mm256_xor_si256(h0, t0);                        // avx
    h0 = _mm256_xor_si256(h0, this->length);              // sse
    h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    h0 = _mm256_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // should have 0 idle latency cyles and 0 cpi cycles here.

    //h1 = fmix32(h1); // ***** SSE4.1 **********
    if (VEC_CNT > 2)
    {
      h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13));
      switch (VEC_CNT)
      {
      case 4:
        t3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 24)); // SSE3
        t3 = _mm256_mullo_epi32(t3, this->c1);                                // avx  // Lat10, CPI2
      case 3:
        t2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 16)); // SSE3
        t2 = _mm256_mullo_epi32(t2, this->c1);                                // avx  // Lat10, CPI2
      case 2:
        t1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key + 8)); // SSE3
        t1 = _mm256_mullo_epi32(t1, this->c1);                               // avx  // Lat10, CPI2
      case 1:
        
        t0 = _mm256_mullo_epi32(t0, this->c1);                           // avx  // Lat10, CPI2
      }
   // h ^= h >> 13;      sse2
      h3 = _mm256_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

      h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
      h2 = _mm256_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    }
    h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    h1 = _mm256_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    h0 = _mm256_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    // latencies.
    // h3  Lat 4, cpi 2

    if (VEC_CNT > 2)
    {
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
  FSC_FORCE_INLINE void hash(T const *key, __m256i &h0, __m256i &h1, __m256i &h2, __m256i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 1 bytes
    //     abcdefghijklmnop qrstuvwxyz123456
    // k0  ---------------- ----------------

    __m256i k0, t0, t1, t2, t3;

    // 32 keys per vector, can potentially do 4 rounds.
    k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(key)); // SSE3   // L3 C1 p23

    // 2 extra calls.
    // need to permute with permutevar8x32, idx = [0 2 4 6 1 3 5 7]
    k0 = _mm256_permutevar8x32_epi32(k0, permute1); // AVX2,    // L3 C1
    // abcd ijkl qrst yz12 efgh mnop uvwx 3456

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // USE shuffle_epi8, with mask.
    // transform to a000b000c000d000 e000f000g000h000.  interleave with 0.
    t0 = _mm256_shuffle_epi8(k0, shuffle0); // AVX2, latency 1, CPI 1
    // ijkl
    t1 = _mm256_shuffle_epi8(k0, shuffle1); // AVX2, latency 1, CPI 1
    // qrst
    t2 = _mm256_shuffle_epi8(k0, shuffle2); // AVX2, latency 1, CPI 1
    // yz12
    t3 = _mm256_shuffle_epi8(k0, shuffle3); // AVX2, latency 1, CPI 1

    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
    t0 = _mm256_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2
    t1 = _mm256_mullo_epi32(t1, this->c1);  // avx  // Lat10, CPI2
    t2 = _mm256_mullo_epi32(t2, this->c1);  // avx  // Lat10, CPI2
    t3 = _mm256_mullo_epi32(t3, this->c1);  // avx  // Lat10, CPI2
    
    // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
    h0 = h1 = h2 = h3 = seed;

    // rotl32
    //t0 = rotl32(t0, 15);
    // t3 = _mm256_or_si256(_mm256_slli_epi32(t3, 15), _mm256_srli_epi32(t3, 17));
    // t3 = _mm256_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
    // t2 = _mm256_or_si256(_mm256_slli_epi32(t2, 15), _mm256_srli_epi32(t2, 17));
    // t2 = _mm256_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
    // t1 = _mm256_or_si256(_mm256_slli_epi32(t1, 15), _mm256_srli_epi32(t1, 17));
    // t1 = _mm256_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
    // t0 = _mm256_or_si256(_mm256_slli_epi32(t0, 15), _mm256_srli_epi32(t0, 17));
    // t0 = _mm256_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2
    // merge with existing.
    this->template update_part2<4>(t0, t1, t2, t3);

    // should have 0 idle latency cyles and 0 cpi cycles here.

    // final step of update, xor the length, and fmix32.
    // finalization
    h0 = _mm256_xor_si256(h0, t0);                        // avx
    h0 = _mm256_xor_si256(h0, this->length);              // sse
    h1 = _mm256_xor_si256(h1, t1);                        // avx
    h1 = _mm256_xor_si256(h1, this->length);              // sse
    h2 = _mm256_xor_si256(h2, t2);                        // avx
    h2 = _mm256_xor_si256(h2, this->length);              // sse
    h3 = _mm256_xor_si256(h3, t3);                        // avx
    h3 = _mm256_xor_si256(h3, this->length);              // sse

    // h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    // h3 = _mm256_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    // h2 = _mm256_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    // h1 = _mm256_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    // h0 = _mm256_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // // should have 0 idle latency cyles and 0 cpi cycles here.

    // //    //h1 = fmix32(h1); // ***** SSE4.1 **********
    // h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
    // h3 = _mm256_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    // h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
    // h2 = _mm256_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    // h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    // h1 = _mm256_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    // h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    // h0 = _mm256_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    //                                                       //
    //                                                       //    // latencies.
    //                                                       //    // h3  Lat 4, cpi 2
    //                                                       //
    // h3 = _mm256_xor_si256(h3, _mm256_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    // h2 = _mm256_xor_si256(h2, _mm256_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    // h1 = _mm256_xor_si256(h1, _mm256_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    // h0 = _mm256_xor_si256(h0, _mm256_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    this->template fmix32<4>(h0, h1, h2, h3);
  }
};
template <typename T>
constexpr uint8_t Murmur32AVX<T>::batch_size;

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
class murmur3avx32
{
public:
  static constexpr uint8_t batch_size = ::fsc::hash::sse::Murmur32AVX<T>::batch_size;

protected:
  ::fsc::hash::sse::Murmur32AVX<T> hasher;
  mutable uint32_t temp[batch_size];

public:
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

    if (rem > 0)
    {
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
    if (rem > 0)
    {
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

} // namespace hash

} // namespace fsc

#endif /* MURMUR3_32_AVX_HPP_ */
