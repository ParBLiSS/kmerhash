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
 *        [ ] migrate AVX optimizations over here.
 */
#ifndef MURMUR3_32_SSE_HPP_
#define MURMUR3_32_SSE_HPP_

#include <type_traits> // enable_if
#include <cstring>     // memcpy
#include <stdexcept>   // logic error
#include <stdint.h>    // std int strings
#include <iostream>    // cout

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



#if defined(__SSE4_1__)
// for 32 bit buckets
// original: body: 16 inst per iter of 4 bytes; tail: 15 instr. ; finalization:  8 instr.
// about 4 inst per byte + 8, for each hash value.
template <typename T>
class Murmur32SSE
{

protected:
  // make static so initialization at beginning of class...
  const __m128i mix_const1;
  const __m128i mix_const2;
  const __m128i c1;
  const __m128i c2;
  const __m128i c4;
  const __m128i length;

  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void rotl(uint8_t const & rot, __m128i &t0, __m128i &t1, __m128i &t2, __m128i &t3) const
  {

    __m128i tt0, tt1, tt2, tt3;

    if (VEC_CNT >= 1) {
      tt1 = _mm_slli_epi32(t0, rot);   // SLLI: L1 C1 p0
      tt0 = _mm_srli_epi32(t0, (32 - rot));   // SRLI: L1 C1 p0
      t0 = _mm_or_si128(tt1, tt0);  // OR: L1 C0.33 p015
    }
    if (VEC_CNT >= 2) {
      tt3 = _mm_slli_epi32(t1, rot);   // SLLI: L1 C1 p0
      tt2 = _mm_srli_epi32(t1, (32 - rot));   // SRLI: L1 C1 p0
      t1 = _mm_or_si128(tt3, tt2);  // OR: L1 C0.33 p015
    }
    if (VEC_CNT >= 3) {
      tt1 = _mm_slli_epi32(t2, rot);   // SLLI: L1 C1 p0
      tt0 = _mm_srli_epi32(t2, (32 - rot));   // SRLI: L1 C1 p0
      t2 = _mm_or_si128(tt1, tt0);  // OR: L1 C0.33 p015
    }
    if (VEC_CNT >= 4) {
      tt3 = _mm_slli_epi32(t3, rot);   // SLLI: L1 C1 p0
      tt2 = _mm_srli_epi32(t3, (32 - rot));   // SRLI: L1 C1 p0
      t3 = _mm_or_si128(tt3, tt2);  // OR: L1 C0.33 p015
    }

    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }

  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void mul(__m128i const & mult, __m128i &t0, __m128i &t1, __m128i &t2, __m128i &t3) const
  {
    if (VEC_CNT >= 1) {
      t0 = _mm_mullo_epi32(t0, mult); // sse   // Lat10, CPI2
    }
    if (VEC_CNT >= 2) {
      t1 = _mm_mullo_epi32(t1, mult); // sse   // Lat10, CPI2
    }
    if (VEC_CNT >= 3) {
      t2 = _mm_mullo_epi32(t2, mult); // sse   // Lat10, CPI2
    }
    if (VEC_CNT >= 4) {
      t3 = _mm_mullo_epi32(t3, mult); // sse                          // MULLO: L10 C2 2p0
    }


    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }


  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void update_part2(__m128i &t0, __m128i &t1, __m128i &t2, __m128i &t3) const
  {
    
    // switch (VEC_CNT)
    // {
    // case 4:
    //   t3 = _mm_or_si128(_mm_slli_epi32(t3, 15), _mm_srli_epi32(t3, 17));  // OR: L1 C0.33 p015,  SR/LL: L1 C1 p0
    //   t3 = _mm_mullo_epi32(t3, this->c2); // sse                          // MULLO: L10 C2 2p0
    // case 3:
    //   t2 = _mm_or_si128(_mm_slli_epi32(t2, 15), _mm_srli_epi32(t2, 17));
    //   t2 = _mm_mullo_epi32(t2, this->c2); // sse   // Lat10, CPI2
    // case 2:
    //   t1 = _mm_or_si128(_mm_slli_epi32(t1, 15), _mm_srli_epi32(t1, 17));
    //   t1 = _mm_mullo_epi32(t1, this->c2); // sse   // Lat10, CPI2
    // case 1:
    //   t0 = _mm_or_si128(_mm_slli_epi32(t0, 15), _mm_srli_epi32(t0, 17));
    //   t0 = _mm_mullo_epi32(t0, this->c2); // sse   // Lat10, CPI2
    // }

    rotl<VEC_CNT>(15, t0, t1, t2, t3);
    mul<VEC_CNT>(this->c2, t0, t1, t2, t3);
    
    // note that the next call needs to have at least 4 operations before using t3, 6 before using t2, 8 before t1, 10 before t0
  }



  // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT, bool add_prev_iter>
  FSC_FORCE_INLINE void add_xor(__m128i const & adder, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3,
                                     __m128i const &t0, __m128i const &t1, __m128i const &t2, __m128i const &t3) const
  {
    // first do the add.
    if (add_prev_iter) {
      if (VEC_CNT >= 1) {
        h0 = _mm_add_epi32(h0, adder);
      }
      if (VEC_CNT >= 2) {
        h1 = _mm_add_epi32(h1, adder);
      }
      if (VEC_CNT >= 3) {
        h2 = _mm_add_epi32(h2, adder);
      }
      if (VEC_CNT >= 4) {
        h3 = _mm_add_epi32(h3, adder);                                   // ADD: L1 C0.5 p15
      }
   }
    
    if (VEC_CNT >= 1) {
      h0 = _mm_xor_si128(h0, t0); // sse
    }
    if (VEC_CNT >= 2) {
      h1 = _mm_xor_si128(h1, t1); // sse
    }
    if (VEC_CNT >= 3) { 
      h2 = _mm_xor_si128(h2, t2); // sse
    }
    if (VEC_CNT >= 4) {
      h3 = _mm_xor_si128(h3, t3); // sse                                    // XOR: L1 C0.33 p015
    }
  }



      // part of the update32() function, from second multiply to last multiply.
  template <uint8_t VEC_CNT, bool add_prev_iter>
  FSC_FORCE_INLINE void update_part3(__m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3,
                                     __m128i const &t0, __m128i const &t1, __m128i const &t2, __m128i const &t3) const
  {
    // // first do the add.
    // if (add_prev_iter) {
    //   switch (VEC_CNT)
    //   {
    //   case 4:
    //     h3 = _mm_add_epi32(h3, this->c4);                                   // ADD: L1 C0.5 p15
    //   case 3:
    //     h2 = _mm_add_epi32(h2, this->c4);
    //   case 2:
    //     h1 = _mm_add_epi32(h1, this->c4);
    //   case 1:
    //     h0 = _mm_add_epi32(h0, this->c4);
    //   }
    // }
    
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, t3); // sse                                    // XOR: L1 C0.33 p015
    // case 3:
    //   h2 = _mm_xor_si128(h2, t2); // sse
    // case 2:
    //   h1 = _mm_xor_si128(h1, t1); // sse
    // case 1:
    //   h0 = _mm_xor_si128(h0, t0); // sse
    // }
    add_xor<VEC_CNT, add_prev_iter>(this->c4, h0, h1, h2, h3, t0, t1, t2, t3);


    rotl<VEC_CNT>(13, h0, h1, h2, h3);
    //mul<VEC_CNT>(this->c3, h0, h1, h2, h3);

    __m128i hh0, hh1, hh2, hh3;
    hh0 = _mm_slli_epi32(h0, 2);
    hh1 = _mm_slli_epi32(h1, 2);
    hh2 = _mm_slli_epi32(h2, 2);
    hh3 = _mm_slli_epi32(h3, 2);

    // do 1x + 4x instead of mul by 5.
    h0 = _mm_add_epi32(h0, hh0);
    h1 = _mm_add_epi32(h1, hh1);
    h2 = _mm_add_epi32(h2, hh2);
    h3 = _mm_add_epi32(h3, hh3);

    // switch (VEC_CNT)
    // {
    // case 4:
    //   if (add_prev_iter)
    //     h3 = _mm_add_epi32(h3, this->c4);                                   // ADD: L1 C0.5 p15
    //   h3 = _mm_xor_si128(h3, t3); // sse                                    // XOR: L1 C0.33 p015
    //   h3 = _mm_or_si128(_mm_slli_epi32(h3, 13), _mm_srli_epi32(h3, 19));    // OR: L1 C0.33 p015,  SR/LL: L1 C1 p0
    //   h3 = _mm_mullo_epi32(h3, this->c3);                                   // MULLO: L10 C2 2p0
    // case 3:
    //   if (add_prev_iter)
    //     h2 = _mm_add_epi32(h2, this->c4);
    //   h2 = _mm_xor_si128(h2, t2); // sse
    //   h2 = _mm_or_si128(_mm_slli_epi32(h2, 13), _mm_srli_epi32(h2, 19));
    //   h2 = _mm_mullo_epi32(h2, this->c3);
    // case 2:
    //   if (add_prev_iter)
    //     h1 = _mm_add_epi32(h1, this->c4);
    //   h1 = _mm_xor_si128(h1, t1); // sse
    //   h1 = _mm_or_si128(_mm_slli_epi32(h1, 13), _mm_srli_epi32(h1, 19));
    //   h1 = _mm_mullo_epi32(h1, this->c3);
    // case 1:
    //   if (add_prev_iter)
    //     h0 = _mm_add_epi32(h0, this->c4);
    //   h0 = _mm_xor_si128(h0, t0); // sse
    //   h0 = _mm_or_si128(_mm_slli_epi32(h0, 13), _mm_srli_epi32(h0, 19));
    //   h0 = _mm_mullo_epi32(h0, this->c3);
    // }
  }

    

  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void shift_xor(uint8_t const & shift, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    __m128i tt0, tt1;
    if (VEC_CNT >= 1) {
      tt0 = _mm_srli_epi32(h0, shift);
      h0 = _mm_xor_si128(h0, tt0); // h ^= h >> 16;      sse2
    }
    if (VEC_CNT >= 2) {
      tt1 = _mm_srli_epi32(h1, shift);
    h1 = _mm_xor_si128(h1, tt1); // h ^= h >> 16;      sse2
    }
    if (VEC_CNT >= 3) {
      tt0 = _mm_srli_epi32(h2, shift); 
    h2 = _mm_xor_si128(h2, tt0); // h ^= h >> 16;      sse2
    }
    if (VEC_CNT >= 4) {
      tt1 = _mm_srli_epi32(h3, shift);   //SRLI: L1, C1, p0.  
      h3 = _mm_xor_si128(h3, tt1); // h ^= h >> 16;      sse2  XOR: L1, C0.33, p015
    }

  }


  /// fmix32 for 16 elements at a time.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32(__m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // // should have 0 idle latency cyles and 0 cpi cycles here.
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    //   h3 = _mm_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 3:
    //   h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    //   h2 = _mm_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 2:
    //   h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    //   h1 = _mm_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // case 1:
    //   h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    //   h0 = _mm_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    // }
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const1, h0, h1, h2, h3);

    // // should have 1 idle latency cyles and 2 cpi cycles here.

    // //h1 = fmix32(h1); // ***** SSE4.1 **********
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
    //   h3 = _mm_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 3:
    //   h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
    //   h2 = _mm_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 2:
    //   h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    //   h1 = _mm_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 1:
    //   h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    //   h0 = _mm_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // }
    shift_xor<VEC_CNT>(13, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const2, h0, h1, h2, h3);

    // // latencies.
    // // h3  Lat 1, cpi 2
    // // h0  Lat 4, cpi 2

    // // expect Lat 0, cycle 1
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    // case 3:
    //   h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    // case 2:
    //   h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    // case 1:
    //   h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    // }
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);

  }



  /// fmix32 for 16 elements at a time.
  template <uint8_t VEC_CNT>
  FSC_FORCE_INLINE void fmix32_part2(__m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // should have 1 idle latency cyles and 2 cpi cycles here.

    //h1 = fmix32(h1); // ***** SSE4.1 **********
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
    //   h3 = _mm_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 3:
    //   h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
    //   h2 = _mm_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 2:
    //   h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    //   h1 = _mm_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // case 1:
    //   h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    //   h0 = _mm_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    // }
    shift_xor<VEC_CNT>(13, h0, h1, h2, h3);
    mul<VEC_CNT>(this->mix_const2, h0, h1, h2, h3);


    // latencies.
    // h3  Lat 1, cpi 2
    // h0  Lat 4, cpi 2

    // expect Lat 0, cycle 1
    // switch (VEC_CNT)
    // {
    // case 4:
    //   h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    // case 3:
    //   h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    // case 2:
    //   h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    // case 1:
    //   h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    // }
    shift_xor<VEC_CNT>(16, h0, h1, h2, h3);

  }


protected:    
  // LATENCIES for instruction with latency > 1
  // 1. mullo_epi16 has latency of 5 cycles, CPI of 1 to 0.5 cycles - need unroll 2x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 2. mullo_epi32 has latency of 10 cycles, CPI of 1 to 0.5 cycles - need unroll 3-4x to hide latency, since rotl32 has 3 instructions with 1 cycle latency.
  // 4. note that most extract calls have latency of 3 and CPI of 1, except for _mm_extracti128_si128, which has latency of 1.
  const __m128i shuffle0; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i shuffle1; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i shuffle2; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i shuffle3; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i shuffle20; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i shuffle21; // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
  const __m128i ones;
  const __m128i zeros;
  mutable __m128i seed;
  
//   // input is 4 unsigned ints.
//   FSC_FORCE_INLINE __m128i rotl32(__m128i x, int8_t r) const
//   {
//     // return (x << r) | (x >> (32 - r));
//     return _mm_or_si128(              // sse2
//         _mm_slli_epi32(x, r),         // sse2
//         _mm_srli_epi32(x, (32 - r))); // sse2
//   }

//   FSC_FORCE_INLINE __m128i update32(__m128i h, __m128i k) const
//   {
//     // preprocess the 4 streams
//     k = _mm_mullo_epi32(k, c1); // SSE2
//     k = rotl32(k, 15);          // sse2
//     k = _mm_mullo_epi32(k, c2); // SSE2
//     // merge with existing.
//     h = _mm_xor_si128(h, k); // SSE
//     // this is done per block of 4 bytes.  the last block (smaller than 4 bytes) does not do this.  do for every byte except last,
//     h = rotl32(h, 13);                             // sse2
//     h = _mm_add_epi32(_mm_mullo_epi32(h, c3), c4); // SSE
//     return h;
//   }

//   // count cannot be zero.
//   FSC_FORCE_INLINE __m128i update32_zeroing(__m128i h, __m128i k, uint8_t const &count) const
//   {
//     assert((count > 0) && (count < 4) && "count should be between 1 and 3");

//     unsigned int shift = (4U - count) * 8U;
//     // clear the upper bytes
//     k = _mm_srli_epi32(_mm_slli_epi32(k, shift), shift); // sse2

//     // preprocess the 4 streams
//     k = _mm_mullo_epi32(k, c1); // SSE2
//     k = rotl32(k, 15);          // sse2
//     k = _mm_mullo_epi32(k, c2); // SSE2
//     // merge with existing.
//     h = _mm_xor_si128(h, k); // SSE
//     return h;
//   }

//   // count cannot be zero.
//   FSC_FORCE_INLINE __m128i update32_partial(__m128i h, __m128i k, uint8_t const &count) const
//   {

//     // preprocess the 4 streams
//     k = _mm_mullo_epi32(k, c1); // SSE2
//     k = rotl32(k, 15);          // sse2
//     k = _mm_mullo_epi32(k, c2); // SSE2
//     // merge with existing.
//     h = _mm_xor_si128(h, k); // SSE
//     return h;
//   }

//   // input is 4 unsigned ints.
//   // is ((h ^ f) * c) carryless multiplication with (f = h >> d)?
//   FSC_FORCE_INLINE __m128i fmix32(__m128i h) const
//   {
//     h = _mm_xor_si128(h, _mm_srli_epi32(h, 16)); // h ^= h >> 16;      sse2
//     h = _mm_mullo_epi32(h, mix_const1);          // h *= 0x85ebca6b;   sse4.1
//     h = _mm_xor_si128(h, _mm_srli_epi32(h, 13)); // h ^= h >> 13;      sse2
//     h = _mm_mullo_epi32(h, mix_const2);          // h *= 0xc2b2ae35;   sse4.1
//     h = _mm_xor_si128(h, _mm_srli_epi32(h, 16)); // h ^= h >> 16;      sse2

//     return h;
//   }

  explicit Murmur32SSE(__m128i _seed) : mix_const1(_mm_set1_epi32(0x85ebca6b)),
                                        mix_const2(_mm_set1_epi32(0xc2b2ae35)),
                                        c1(_mm_set1_epi32(0xcc9e2d51)),
                                        c2(_mm_set1_epi32(0x1b873593)),
                                        c4(_mm_set1_epi32(0xe6546b64)),
                                        length(_mm_set1_epi32(sizeof(T))),
                                        shuffle0(_mm_setr_epi32(0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U)),
                                        shuffle1(_mm_setr_epi32(0x80808004U, 0x80808005U, 0x80808006U, 0x80808007U)),
                                        shuffle2(_mm_setr_epi32(0x80808008U, 0x80808009U, 0x8080800AU, 0x8080800BU)),
                                        shuffle3(_mm_setr_epi32(0x8080800CU, 0x8080800DU, 0x8080800EU, 0x8080800FU)),
                                        shuffle20(_mm_setr_epi32(0x80800100U, 0x80800302U, 0x80800504U, 0x80800706U)),
                                        shuffle21(_mm_setr_epi32(0x80800908U, 0x80800B0AU, 0x80800D0CU, 0x80800F0EU)),
                                        ones(_mm_cmpeq_epi32(length, length)),
										zeros(_mm_setzero_si128()),
                                        seed(_seed)
  {
  }

public:
    static constexpr size_t batch_size = 16;

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
  // hash up to 16 elements at a time. at a time.  each is 1 byte
  FSC_FORCE_INLINE void hash(T const *key, uint8_t nstreams, uint32_t *out) const
  {
    // process 4 streams at a time.  all should be the same length.
    // process 4 streams at a time.  all should be the same length.

    assert((nstreams <= 16) && "maximum number of streams is 16");
    assert((nstreams > 0) && "minimum number of streams is 1");

    __m128i h0, h1, h2, h3;


    // do the full ones first.
    uint8_t blocks = (nstreams + 3) >> 2; // divide by 4.
  //  std::cout << "\tStreams to process " << static_cast<size_t>(nstreams) << " in blocks " << static_cast<size_t>(blocks) << std::endl;
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

    blocks = nstreams >> 2;
    if (blocks >= 1) _mm_storeu_si128((__m128i *)out, h0);
    if (blocks >= 2) _mm_storeu_si128((__m128i *)(out + 4), h1);
    if (blocks >= 3) _mm_storeu_si128((__m128i *)(out + 8), h2);
    if (blocks >= 4) _mm_storeu_si128((__m128i *)(out + 12), h3);
    
    uint8_t rem = nstreams & 3; // remainder.
    if (rem > 0)
    {
      // write remainders.  write rem * 4 bytes.
      switch (blocks)
      {
        case 0:
        memcpy(out, reinterpret_cast<uint32_t *>(&h0), rem << 2); // copy bytes
        break;
      case 1:
        memcpy(out + 4, reinterpret_cast<uint32_t *>(&h1), rem << 2); // copy bytes
        break;
        case 2:
        memcpy(out + 8, reinterpret_cast<uint32_t *>(&h2), rem << 2); // copy bytes
        break;
        case 3:
        memcpy(out + 12, reinterpret_cast<uint32_t *>(&h3), rem << 2); // copy bytes
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
    __m128i h0, h1, h2, h3;
//    std::cout << "\tprocess 16 streams  in 4 blocks " << std::endl;
    hash<4>(key, h0, h1, h2, h3);
    _mm_storeu_si128((__m128i *)out, h0);
    _mm_storeu_si128((__m128i *)(out + 4), h1);
    _mm_storeu_si128((__m128i *)(out + 8), h2);
    _mm_storeu_si128((__m128i *)(out + 12), h3);
  }




  /// NOTE: multiples of 32.
  // USING load, INSERT plus unpack is FASTER than i32gather.
  // load with an offset from start of key.
  FSC_FORCE_INLINE void load_stride16(T const *key, size_t const & offset, 
                                      __m128i &t0, __m128i &t1, __m128i &t2, __m128i &t3 //,
                                      // __m128i & t4, __m128i & t5, __m128i & t6, __m128i & t7
                                      ) const
  {

    // faster than gather.

    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    __m128i k0, k1, k2, k3, tt0, tt2;

    // load 4 keys at a time, 16 bytes each time,
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + offset);     // SSE3   // L3, C0.5, p23
    k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + offset); // SSE3

    k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + offset); // SSE3
    k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + offset); // SSE3

    // MERGED shuffling and update part 1.
    // now unpack and update
    tt0 = _mm_unpacklo_epi32(k0, k1); // aba'b'efe'f'                           // L1 C1 p5    
    t1 = _mm_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
    tt2 = _mm_unpackhi_epi32(k0, k1);
    
    k0 = _mm_unpacklo_epi64(tt0, t1);   // abcdefgh
    k1 = _mm_unpackhi_epi64(tt0, t1);   // a'b'c'd'e'f'g'h'

    t0 = _mm_mullo_epi32(k0, this->c1); // avx                                  // L10 C2 p0

    t3 = _mm_unpackhi_epi32(k2, k3);                                            // L1 C1 p5
    
    t1 = _mm_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

    
    k2 = _mm_unpacklo_epi64(tt2, t3);
    k3 = _mm_unpackhi_epi64(tt2, t3);

    t2 = _mm_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
    t3 = _mm_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2


    // latency:  should be Lat3, C2 for temp
    // update part 2.
    this->template update_part2<4>(t0, t1, t2, t3);

    // loading 8 32-byte keys is slower than loading 8 16-byte keys.
  }

  // USING load, INSERT plus unpack is FASTER than i32gather.
  // also probably going to be faster for non-power of 2 less than 8 (for 9 to 15, this is needed anyways).
  //   because we'd need to shift across lane otherwise.
  // load with an offset from start of key, and load partially.  blocks of 16,
  template <size_t KEY_LEN = sizeof(T), size_t offset = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15)>
  FSC_FORCE_INLINE void load_partial16(T const *key,
                                       __m128i &t0, __m128i &t1, __m128i &t2, __m128i &t3 //,
                                       // __m128i & t4, __m128i & t5, __m128i & t6, __m128i & t7
                                       ) const
  {

    // a lot faster than gather.

    static_assert(rem > 0, "ERROR: should not call load_partial when remainder if 0");

    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    __m128i k0, k1, k2, k3, tt0, tt2;

    __m128i mask = _mm_srli_si128(ones, 16 - rem); // shift right to keep just the remainder part

    // load 8 keys at a time, 16 bytes each time,
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key) + offset);     // SSE3
    k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 1) + offset); // SSE3
    k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2) + offset); // SSE3
    k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 3) + offset); // SSE3

    // ZERO the leading bytes to keep just the lower.
    // latency of 3 and CPI of 1, so can do the masking here...
    k0 = _mm_and_si128(k0, mask);
    k1 = _mm_and_si128(k1, mask);
    k2 = _mm_and_si128(k2, mask);
    k3 = _mm_and_si128(k3, mask);
    
    // MERGED shuffling and update part 1.
    // now unpack and update
    // RELY ON COMPILER OPTIMIZATION HERE TO REMOVE THE CONDITIONAL CHECKS
    tt0 = _mm_unpacklo_epi32(k0, k1); // aba'b'efe'f'                           // L1 C1 p5    
    t1 = _mm_unpacklo_epi32(k2, k3); // cdc'd'ghg'h'
    if (rem > 8) tt2 = _mm_unpackhi_epi32(k0, k1);

    k0 = _mm_unpacklo_epi64(tt0, t1);   // abcdefgh
    if (rem > 4) k1 = _mm_unpackhi_epi64(tt0, t1);   // a'b'c'd'e'f'g'h'

    t0 = _mm_mullo_epi32(k0, this->c1); // avx                                  // L10 C2 p0

    if (rem > 8) t3 = _mm_unpackhi_epi32(k2, k3);                                            // L1 C1 p5

    if (rem > 4) t1 = _mm_mullo_epi32(k1, this->c1); // avx  // Lat10, CPI2

    
    if (rem > 8) k2 = _mm_unpacklo_epi64(tt2, t3);
    if (rem > 12) k3 = _mm_unpackhi_epi64(tt2, t3);

    if (rem > 8) t2 = _mm_mullo_epi32(k2, this->c1); // avx  // Lat10, CPI2
    if (rem > 12) t3 = _mm_mullo_epi32(k3, this->c1); // avx  // Lat10, CPI2


    // latency:  should be Lat3, C2 for temp
    // update part 2.  note that we compute for parts that have non-zero values, determined in blocks of 4 bytes.
    this->template update_part2<((rem + 3) >> 2)>(t0, t1, t2, t3);
  }

  /// NOTE: non-power of 2 length keys ALWAYS use AVX gather, which may be slower.
  // for hasing non multiple of 16 and non power of 2.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T), size_t nblocks = (sizeof(T) >> 4), size_t rem = (sizeof(T) & 15),
            typename std::enable_if<((KEY_LEN & (KEY_LEN - 1)) > 0) && ((KEY_LEN & 15) > 0), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
    // and can pipeline 4 at a time, about 40 cycles?
    // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
    // while we still have 8 "update"s, the programming cost is becoming costly.
    // an alternative might be using _mm_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
    // still need to shuffle more than 4 times.

    static_assert(rem > 0, "ERROR remainder is 0.");

    // load 16 bytes at a time.
    __m128i t00, t01, t02, t03, //t04, t05, t06, t07,
        t10, t11, t12, t13,     //t14, t15, t16, t17,
        t20, t21, t22, t23,     //t24, t25, t26, t27,
        t30, t31, t32, t33      //, t34, t35, t36, t37
        ;
#if defined(__clang__)
    t00 = t01 = t02 = t03 = t10 = t11 = t12 = t13 = t20 = t21 = t22 = t23 = t30 = t31 = t32 = t33 = zeros;
#endif

    // read input, 8 keys at a time.  need 4 rounds.
    h0 = h1 = h2 = h3 = seed;

    size_t i = 0;
    for (; i < nblocks; ++i)
    {

      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)

    if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
    if (VEC_CNT >= 2)  this->load_stride16(key + 4, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
    if (VEC_CNT >= 3)  this->load_stride16(key + 8, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
    if (VEC_CNT >= 4)  this->load_stride16(key + 12, i, t30, t31, t32, t33); // , t34, t35, t36, t37);
  
  
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
    if (rem > 0)
    { // NOT multiple of 16.

      // read input, 2 keys per vector.
      // combined load and update_part1 and update_part2 (data parallel part.)
      if (VEC_CNT >= 1)  this->load_partial16(key, t00, t01, t02, t03); // , t04, t05, t06, t07);
      if (VEC_CNT >= 2)  this->load_partial16(key + 4, t10, t11, t12, t13); // , t14, t15, t16, t17);
      if (VEC_CNT >= 3)  this->load_partial16(key + 8, t20, t21, t22, t23); // , t24, t25, t26, t27);
      if (VEC_CNT >= 4)  this->load_partial16(key + 12, t30, t31, t32, t33); // , t34, t35, t36, t37);
    }

    // For the last b < 4 bytes, we do not do full update.
    if (rem >= 4)
    {
      if (i == 0)
        this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
      else
        this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
    }
    if (rem >= 8)
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
    if (rem >= 12)
      this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);

    __m128i t0, t1, t2, t3;
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

    // should have 0 idle latency cyles and 0 cpi cycles here.
    if (VEC_CNT >= 1) {
      if (rem >= 4)                                         // complete the prev update_part3
        h0 = _mm_add_epi32(h0, this->c4);                // avx
      if ((rem & 3) > 0)                                    // has partial int
        h0 = _mm_xor_si128(h0, t0);                      // avx
      h0 = _mm_xor_si128(h0, this->length);              // sse
      // h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
      // h0 = _mm_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    if (VEC_CNT >= 2) {
      if (rem >= 4)                                         // complete the prev update_part3
        h1 = _mm_add_epi32(h1, this->c4);                // avx
      if ((rem & 3) > 0)                                    // has partial int
        h1 = _mm_xor_si128(h1, t1);                      // avx
      h1 = _mm_xor_si128(h1, this->length);              // sse
      // h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
      // h1 = _mm_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    if (VEC_CNT >= 3) {
      if (rem >= 4)                                         // complete the prev update_part3
        h2 = _mm_add_epi32(h2, this->c4);                // avx
      if ((rem & 3) > 0)                                    // has partial int
        h2 = _mm_xor_si128(h2, t2);                      // avx
      h2 = _mm_xor_si128(h2, this->length);              // sse
      // h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      // h2 = _mm_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    if (VEC_CNT >= 4) {
      if (rem >= 4)                                         // complete the prev update_part3
        h3 = _mm_add_epi32(h3, this->c4);                // avx
      if ((rem & 3) > 0)                                    // has partial int
        h3 = _mm_xor_si128(h3, t3);                      // avx
      h3 = _mm_xor_si128(h3, this->length);              // sse
      // h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      // h3 = _mm_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    //    // should have 0 idle latency cyles and 0 cpi cycles here.
    //

    // Latency: h3: L1 C2, h0:L1 C2
    // this->template fmix32_part2<VEC_CNT>(h0, h1, h2, h3);
    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
  }

  // hashing 16 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 16 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // multiple of 16 that are greater than 16.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
  typename std::enable_if<((KEY_LEN & 15) == 0) && (KEY_LEN > 16), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // we now assume no specific layout, so we need to load 8 at a time.

    // load 16 bytes at a time.
    const int nblocks = KEY_LEN >> 4;

    __m128i t00, t01, t02, t03, //t04, t05, t06, t07,
    t10, t11, t12, t13,     //t14, t15, t16, t17,
    t20, t21, t22, t23,     //t24, t25, t26, t27,
    t30, t31, t32, t33      //, t34, t35, t36, t37
    ;

    // read input, 8 keys at a time.  need 4 rounds.
    h0 = h1 = h2 = h3 = seed;

    int i = 0;
    for (; i < nblocks; ++i)
    {

    // read input, 2 keys per vector.
    // combined load and update_part1 and update_part2 (data parallel part.)
    if (VEC_CNT >= 1)  this->load_stride16(key, i, t00, t01, t02, t03); // , t04, t05, t06, t07);
    if (VEC_CNT >= 2)  this->load_stride16(key + 4, i, t10, t11, t12, t13); // , t14, t15, t16, t17);
    if (VEC_CNT >= 3)  this->load_stride16(key + 8, i, t20, t21, t22, t23); // , t24, t25, t26, t27);
    if (VEC_CNT >= 4)  this->load_stride16(key + 12, i, t30, t31, t32, t33); // , t34, t35, t36, t37);

    // now do part 3.   ORDER MATTERS. from low to high.
    if (i == 0)
    this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
    else
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t00, t10, t20, t30);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);
    }
    // latency: h3: L0, C0.  h0: L1,C2

    // should have 0 idle latency cyles and 0 cpi cycles here.
    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
    this->length, this->length, this->length, this->length);

    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);

  }


  // hashing 16 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 16 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 16), int>::type = 1> // 16 bytes exactly.
  FSC_FORCE_INLINE void
  hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {

    // example layout, with each dash representing 4 bytes
    //     aa'AA'  bb'BB' cc'CC' dd'DD' 
    // k0  -- --   
    //             -- --
    // k1                 -- --  
    //                           -- --

    __m128i t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31, t32, t33;

    // read input, 8 keys at a time.  need 4 rounds.

    // read input, 2 keys per vector.
    // combined load and update_part1 and update_part2 (data parallel part.)
    if (VEC_CNT >= 1)   this->load_stride16(key, 0, t00, t01, t02, t03);
    if (VEC_CNT >= 2)   this->load_stride16(key + 4, 0, t10, t11, t12, t13);
    if (VEC_CNT >= 3)   this->load_stride16(key + 8, 0, t20, t21, t22, t23);
    if (VEC_CNT >= 4)   this->load_stride16(key + 12, 0, t30, t31, t32, t33);
    
    h0 = h1 = h2 = h3 = seed;

    // now do part 3.
    this->template update_part3<VEC_CNT, false>(h0, h1, h2, h3, t00, t10, t20, t30);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t01, t11, t21, t31);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t02, t12, t22, t32);
    this->template update_part3<VEC_CNT, true>(h0, h1, h2, h3, t03, t13, t23, t33);

    // latency: h3: L0, C0.  h0: L1,C2

    // DO LAST ADD FROM UPDATE32

    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
      this->length, this->length, this->length, this->length);

    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
  }


  FSC_FORCE_INLINE void load8(T const *key, __m128i &t0, __m128i &t1) const
  {
    __m128i k0, k1;

    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key));     // SSE3  aAbB  L3, C0.5, p23
    k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 2)); // SSE3  cCdD  L3, C0.5, p23
    
    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // shuffle :  [0 2 1 3], or in binary 11011000, or hex 0xD8
    k0 = _mm_shuffle_epi32(k0, 0xD8); // Lat1, Cpi1 p5.  abAB
    k1 = _mm_shuffle_epi32(k1, 0xD8); // Lat1, Cpi1 p5.  cdCD
    

    // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
    
    t0 = _mm_unpacklo_epi64(k0, k1); // Lat1, cpi1, p5.  //abcd 
    t0 = _mm_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2, p0

    t1 = _mm_unpackhi_epi64(k0, k1); // Lat1, cpi1, p5.  // ABCD
    t1 = _mm_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2, p0
    
    // don't run update32_part2 here - need 4 mullo to hide latency.
  }

  // hashing 16 elements worth of keys at a time.
  // 8 8-byte elements fills a cache line.  using 15 to 16 registers, so probably require some reuse..
  // NOTE: the reason for 16 elements is that mullo's 10 cycle latency requires unrolling loop 4 times to fill in the mixing stage.
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 8), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {

    // example layout, with each dash representing 4 bytes
    //     aAbB cCdD eEfF gGhH
    // k0  ---- 
    //          ----
    // k2            ----
    //                    ----

    __m128i t00, t01, t10, t11, t20, t21, t30, t31;

    // read input, 4 keys per vector.
    // do not use unpacklo and unpackhi - interleave would be aeAEcgCG
    // instead use shuffle + 2 blend + another shuffle.
    // OR: shift, shift, blend, blend

    if (VEC_CNT >= 1) load8(key, t00, t01);
    if (VEC_CNT >= 2) load8(key + 4, t10, t11);
    if (VEC_CNT >= 3) load8(key + 8, t20, t21);
    if (VEC_CNT >= 4) load8(key + 12, t30, t31);
    

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

    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
        this->length, this->length, this->length, this->length);

    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
    
  }

  // hashing 16 elements worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // for 4 byte, testing with 50M, on i7-4770, shows 0.0356, 0.0360, 0.0407, 0.0384 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 4), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 4 bytes
    //     ab cd ef gh
    // k0  -- -- 
    //           -- --
    // want
    //     a b  c d
    //     e g  g h  
    //     i j  k l
    //     m n  o p

    __m128i t0, t1, t2, t3;

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2

    // 16 keys per vector. can potentially do 2 iters.
    if (VEC_CNT >= 1) {
      t0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3   // abcd
      t0 = _mm_mullo_epi32(t0, this->c1);                           // avx  // Lat10, CPI2
    }
    if (VEC_CNT >= 2) {
      t1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 4)); // SSE3  // efgh
      t1 = _mm_mullo_epi32(t1, this->c1);                               // avx  // Lat10, CPI2
    }
    if (VEC_CNT >= 3) {
      t2 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 8)); // SSE3
      t2 = _mm_mullo_epi32(t2, this->c1);                                // avx  // Lat10, CPI2
    }
    if (VEC_CNT >= 4) {
      t3 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 12)); // SSE3
      t3 = _mm_mullo_epi32(t3, this->c1);                                // avx  // Lat10, CPI2
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
    add_xor<VEC_CNT, true>(this->c4, h0, h1, h2, h3, 
      this->length, this->length, this->length, this->length);

    this->template fmix32<VEC_CNT>(h0, h1, h2, h3);
}

  // hashing 16 elements worth of keys at a time.  uses 10 to 11 registers.
  // if we go to 64 bytes, then we'd be using 20 to 21 registers
  // first latency cycles are hidden.  the second latency cycles will remain the same for double the number of elements.
  // NOTE: experiment shows that naive loop unrolling, by calling fmix32 and update32_partial directly, and relying on compiler, showed suboptimal improvements.
  // for 2 byte, testing with 50M, on i7-4770, shows 0.0290, 0.0304, 0.0312, 0.0294 secs for
  //    manual, manual with 32element fused update function (for reuse), no unrolling, and simple unrolling
  //   manual is more effective at latency hiding, also confirmed by vtunes hotspots.
  template <uint8_t VEC_CNT, size_t KEY_LEN = sizeof(T),
            typename std::enable_if<(KEY_LEN == 2), int>::type = 1>
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 2 bytes
    //     abcd efgh   ijkl mnop
    // k0  ---- ---- 
    // want
    //     a b  c d
    //     e g  g h  
    //     i j  k l
    //     m n  o p

    __m128i k0, k1;
    __m128i t0, t1, t2, t3;

    if (VEC_CNT > 2)
    {
      k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key + 8)); // SSE3
    }

    // 16 keys per vector. can potentially do 2 iters.
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // transform to a0b0c0d0 e0f0g0h0.  interleave with 0.
    if (VEC_CNT > 2)
    {
      // yz12
      t3 = _mm_shuffle_epi8(k1, shuffle21);  // AVX2, latency 1, CPI 1
      t3 = _mm_mullo_epi32(t3, this->c1); // avx  // Lat10, CPI2

      t2 = _mm_shuffle_epi8(k1, shuffle20);  // AVX2, latency 1, CPI 1
      t2 = _mm_mullo_epi32(t2, this->c1); // avx  // Lat10, CPI2
    }
    // ijkl
    t1 = _mm_shuffle_epi8(k0, shuffle21);  // AVX2, latency 1, CPI 1
    t1 = _mm_mullo_epi32(t1, this->c1); // avx  // Lat10, CPI2

    t0 = _mm_shuffle_epi8(k0, shuffle20); // AVX2, latency 1, CPI 1
    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
    t0 = _mm_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2
    // qrst
    // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
    h0 = h1 = h2 = h3 = seed;

    // rotl32
    //t0 = rotl32(t0, 15);
    if (VEC_CNT > 2)
    {
      t3 = _mm_or_si128(_mm_slli_epi32(t3, 15), _mm_srli_epi32(t3, 17));
      t3 = _mm_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
      t2 = _mm_or_si128(_mm_slli_epi32(t2, 15), _mm_srli_epi32(t2, 17));
      t2 = _mm_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
    }
    t1 = _mm_or_si128(_mm_slli_epi32(t1, 15), _mm_srli_epi32(t1, 17));
    t1 = _mm_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
    t0 = _mm_or_si128(_mm_slli_epi32(t0, 15), _mm_srli_epi32(t0, 17));
    t0 = _mm_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2
    // merge with existing.
    //    this->template update_part2<VEC_CNT>(t0, t1, t2, t3);

    // should have 0 idle latency cyles and 0 cpi cycles here.

    // final step of update, xor the length, and fmix32.
    // finalization
    if (VEC_CNT > 2)
    {
      h3 = _mm_xor_si128(h3, t3);                        // avx
      h3 = _mm_xor_si128(h3, this->length);              // sse
      h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      h3 = _mm_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

      h2 = _mm_xor_si128(h2, t2);                        // avx
      h2 = _mm_xor_si128(h2, this->length);              // sse
      h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
      h2 = _mm_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2
    }
    h1 = _mm_xor_si128(h1, t1);                        // avx
    h1 = _mm_xor_si128(h1, this->length);              // sse
    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h1 = _mm_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    h0 = _mm_xor_si128(h0, t0);                        // avx
    h0 = _mm_xor_si128(h0, this->length);              // sse
    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    h0 = _mm_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // should have 0 idle latency cyles and 0 cpi cycles here.

    //h1 = fmix32(h1); // ***** SSE4.1 **********
    if (VEC_CNT > 2)
    {
      h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
      h3 = _mm_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

      h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
      h2 = _mm_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
    }
    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    h1 = _mm_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    h0 = _mm_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    // latencies.
    // h3  Lat 4, cpi 2

    if (VEC_CNT > 2)
    {
      h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
      h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    }
    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
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
  FSC_FORCE_INLINE void hash(T const *key, __m128i &h0, __m128i &h1, __m128i &h2, __m128i &h3) const
  {
    // process 32 streams at a time, each 1 byte.  all should be the same length.

    // example layout, with each dash representing 1 bytes
    //     abcdefgh ijklmnop 
    // k0  -------- -------- 
    // want
    //     a   b    c   d
    //     e   f    g   h
    //     i   j    k   l
    //     m   n    o   p

    __m128i k0, t0, t1, t2, t3;

    // 32 keys per vector, can potentially do 4 rounds.
    k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(key)); // SSE3

    // MERGED SHUFFLE AND UPDATE_PARTIAL
    // USE shuffle_epi8, with mask.
    // yz12
    t3 = _mm_shuffle_epi8(k0, shuffle3); // AVX2, latency 1, CPI 1
    t3 = _mm_mullo_epi32(t3, this->c1);  // avx  // Lat10, CPI2
    // qrst
    t2 = _mm_shuffle_epi8(k0, shuffle2); // AVX2, latency 1, CPI 1
    t2 = _mm_mullo_epi32(t2, this->c1);  // avx  // Lat10, CPI2
    // ijkl
    t1 = _mm_shuffle_epi8(k0, shuffle1); // AVX2, latency 1, CPI 1
    t1 = _mm_mullo_epi32(t1, this->c1);  // avx  // Lat10, CPI2

    // transform to a000b000c000d000 e000f000g000h000.  interleave with 0.
    t0 = _mm_shuffle_epi8(k0, shuffle0); // AVX2, latency 1, CPI 1
    // h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2
    t0 = _mm_mullo_epi32(t0, this->c1); // avx  // Lat10, CPI2

    // should have 4 idle latency cycles and 2 CPI cycles here.  initialize here.
    h0 = h1 = h2 = h3 = seed;

    // rotl32
    //t0 = rotl32(t0, 15);
    t3 = _mm_or_si128(_mm_slli_epi32(t3, 15), _mm_srli_epi32(t3, 17));
    t3 = _mm_mullo_epi32(t3, this->c2); // avx   // Lat10, CPI2
    t2 = _mm_or_si128(_mm_slli_epi32(t2, 15), _mm_srli_epi32(t2, 17));
    t2 = _mm_mullo_epi32(t2, this->c2); // avx   // Lat10, CPI2
    t1 = _mm_or_si128(_mm_slli_epi32(t1, 15), _mm_srli_epi32(t1, 17));
    t1 = _mm_mullo_epi32(t1, this->c2); // avx   // Lat10, CPI2
    t0 = _mm_or_si128(_mm_slli_epi32(t0, 15), _mm_srli_epi32(t0, 17));
    t0 = _mm_mullo_epi32(t0, this->c2); // avx   // Lat10, CPI2
    // merge with existing.
    //    this->template update_part2<4>(t0, t1, t2, t3);

    // should have 0 idle latency cyles and 0 cpi cycles here.

    // final step of update, xor the length, and fmix32.
    // finalization

    h3 = _mm_xor_si128(h3, t3);                        // avx
    h3 = _mm_xor_si128(h3, this->length);              // sse
    h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    h3 = _mm_mullo_epi32(h3, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    h2 = _mm_xor_si128(h2, t2);                        // avx
    h2 = _mm_xor_si128(h2, this->length);              // sse
    h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    h2 = _mm_mullo_epi32(h2, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    h1 = _mm_xor_si128(h1, t1);                        // avx
    h1 = _mm_xor_si128(h1, this->length);              // sse
    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h1 = _mm_mullo_epi32(h1, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    h0 = _mm_xor_si128(h0, t0);                        // avx
    h0 = _mm_xor_si128(h0, this->length);              // sse
    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
    h0 = _mm_mullo_epi32(h0, this->mix_const1);        // h *= 0x85ebca6b;   Lat10, CPI2

    // should have 0 idle latency cyles and 0 cpi cycles here.

    //    //h1 = fmix32(h1); // ***** SSE4.1 **********
    h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 13)); // h ^= h >> 13;      sse2
    h3 = _mm_mullo_epi32(h3, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 13)); // h ^= h >> 13;      sse2
    h2 = _mm_mullo_epi32(h2, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 13)); // h ^= h >> 13;      sse2
    h1 = _mm_mullo_epi32(h1, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2

    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 13)); // h ^= h >> 13;      sse2
    h0 = _mm_mullo_epi32(h0, this->mix_const2);        // h *= 0xc2b2ae35;   Lat10, CPI2
                                                          //
                                                          //    // latencies.
                                                          //    // h3  Lat 4, cpi 2
                                                          //
    h3 = _mm_xor_si128(h3, _mm_srli_epi32(h3, 16)); // h ^= h >> 16;      sse2
    h2 = _mm_xor_si128(h2, _mm_srli_epi32(h2, 16)); // h ^= h >> 16;      sse2
    h1 = _mm_xor_si128(h1, _mm_srli_epi32(h1, 16)); // h ^= h >> 16;      sse2
    h0 = _mm_xor_si128(h0, _mm_srli_epi32(h0, 16)); // h ^= h >> 16;      sse2
                                                          //    this->template fmix32_part2<4>(h0, h1, h2, h3);
  }

  

};
template <typename T>
constexpr size_t Murmur32SSE<T>::batch_size;


#endif

} // namespace sse


#if defined(__SSE4_1__)
/**
     * @brief MurmurHash.  using lower 64 bits.
     * @details.  prefetching did not help
     */
template <typename T>
class murmur3sse32
{
public:
  static constexpr size_t batch_size = ::fsc::hash::sse::Murmur32SSE<T>::batch_size;

protected:
  ::fsc::hash::sse::Murmur32SSE<T> hasher;
  mutable uint32_t temp[batch_size];

public:
  using result_type = uint32_t;
  using argument_type = T;

  murmur3sse32(uint32_t const &_seed = 43) : hasher(_seed) {
    memset(temp, 0, batch_size * sizeof(uint32_t));
  };

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
    // std::cout << "murmur3 sse hash total count " << count << std::endl;
    size_t rem = count & (batch_size - 1);
    size_t max = count - rem;
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
//      std::cout << "\tpos " << i << std::endl;
      hasher.hash(&(keys[i]), results + i);
    }

    if (rem > 0) {
//      std::cout << "\tlast pos  " << i << std::endl;      
      hasher.hash(&(keys[i]), rem, results + i);
    }
  }

//  // assume consecutive memory layout.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint32_t modulus) const
//  {
//    size_t rem = count & (batch_size - 1);
//    size_t max = count - rem;
//    size_t i = 0, j = 0;
//    for (; i < max; i += batch_size)
//    {
//      hasher.hash(&(keys[i]), temp);
//
//      for (j = 0; j < batch_size; ++j)
//      results[i + j] = temp[j] % modulus;
//    }
//
//    // last part.
//    if (rem > 0) {
//      hasher.hash(&(keys[i]), rem, temp);
//
//      for (j = 0; j < rem; ++j)
//      results[i + j] = temp[j] % modulus;
//    }
//}
//
//  // assume consecutive memory layout.
//  // note that the paremter is modulus bits.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint32_t modulus) const
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
//
//      for (j = 0; j < batch_size; ++j)
//      results[i + j] = temp[j] & modulus;
//    }
//
//    // last part.
//    if (rem > 0) {
//      hasher.hash(&(keys[i]), rem, temp);
//
//      for (j = 0; j < rem; ++j)
//      results[i + j] = temp[j] & modulus;
//    }
//  }

  // TODO: [ ] add a transform_hash_mod.
};
template <typename T>
constexpr size_t murmur3sse32<T>::batch_size;

#endif


} // namespace hash

} // namespace fsc

#endif /* MURMUR3_32_SSE_HPP_ */
