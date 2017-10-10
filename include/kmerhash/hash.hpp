/*
 * Copyright 2015 Georgia Institute of Technology
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
 *  TODO: [ ] proper AVX transition that respects avx constants.
 */
#ifndef HASH_HPP_
#define HASH_HPP_

#include <type_traits>  // enable_if
#include <cstring>  // memcpy
#include <stdexcept>  //logic error
#include <stdint.h>  // std int strings


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

#if defined(_MSC_VER)

#define FSC_FORCE_INLINE  __forceinline

// Other compilers

#else // defined(_MSC_VER)

#define FSC_FORCE_INLINE inline __attribute__((always_inline))

#endif // !defined(_MSC_VER)


// may want to check out CLHash, which uses carryless multiply instead of multiply.

#include <x86intrin.h>

namespace fsc {

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
      template <typename T, size_t bytes = sizeof(T)>
      class Murmur32AVX {

        protected:
          // make static so initialization at beginning of class...
          const __m256i mix_const1;
          const __m256i mix_const2;
          const __m256i c1;
          const __m256i c2;
          const __m256i c3;
          const __m256i c4;
          const __m256i permute1;
          const __m256i permute16;
          const __m256i shuffle1;  // shuffle1 spaces out the lowest 4 bytes to 16 bytes, by inserting 0s. no lane crossing.
          const __m256i length;
          const __m256i offs;
          mutable __m256i seed;

          // input is 4 unsigned ints.
          FSC_FORCE_INLINE __m256i rotl32 ( __m256i x, int8_t r ) const
          {
            // return (x << r) | (x >> (32 - r));
            return _mm256_or_si256(                // sse2
                _mm256_slli_epi32(x, r),           // sse2
                _mm256_srli_epi32(x, (32 - r)));   // sse2
          }

          FSC_FORCE_INLINE __m256i update32( __m256i h, __m256i k) const {
            // preprocess the 4 streams
            k = _mm256_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm256_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm256_xor_si256(h, k);     // SSE
            // this is done per block of 4 bytes.  the last block (smaller than 4 bytes) does not do this.  do for every byte except last,
            h = rotl32(h, 13);           // sse2
            h = _mm256_add_epi32(_mm256_mullo_epi32(h, c3), c4);  // SSE
            return h;
          }

          // count cannot be zero.
          FSC_FORCE_INLINE __m256i update32_zeroing( __m256i h, __m256i k, uint8_t const & count) const {
            assert((count > 0) && (count < 4) && "count should be between 1 and 3");

            unsigned int shift = (4U - count) * 8U;
            // clear the upper bytes
            k = _mm256_srli_epi32(_mm256_slli_epi32(k, shift), shift);	// sse2

            // preprocess the 4 streams
            k = _mm256_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm256_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm256_xor_si256(h, k);     // SSE
            return h;
          }

          // count cannot be zero.
          FSC_FORCE_INLINE __m256i update32_partial( __m256i h, __m256i k, uint8_t const & count) const {

            // preprocess the 4 streams
            k = _mm256_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm256_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm256_xor_si256(h, k);     // SSE
            return h;
          }


          // input is 4 unsigned ints.
          // is ((h ^ f) * c) carryless multiplication with (f = h >> d)?
          FSC_FORCE_INLINE __m256i fmix32 ( __m256i h ) const
          {
            h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));  // h ^= h >> 16;      sse2
            h = _mm256_mullo_epi32(h, mix_const1);           // h *= 0x85ebca6b;   sse4.1
            h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));  // h ^= h >> 13;      sse2
            h = _mm256_mullo_epi32(h, mix_const2);           // h *= 0xc2b2ae35;   sse4.1
            h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));  // h ^= h >> 16;      sse2

            return h;
          }


          explicit Murmur32AVX(__m256i _seed) :
            mix_const1(_mm256_set1_epi32(0x85ebca6bU)),
            mix_const2(_mm256_set1_epi32(0xc2b2ae35U)),
            c1(_mm256_set1_epi32(0xcc9e2d51U)),
            c2(_mm256_set1_epi32(0x1b873593U)),
            c3(_mm256_set1_epi32(0x5U)),
            c4(_mm256_set1_epi32(0xe6546b64U)),
            permute1(_mm256_setr_epi32(0U, 2U, 4U, 6U, 1U, 3U, 5U, 7U)),
            permute16(_mm256_setr_epi32(0U, 4U, 1U, 5U, 2U, 6U, 3U, 7U)),
            shuffle1(_mm256_setr_epi32(	0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U,
            							0x80808000U, 0x80808001U, 0x80808002U, 0x80808003U)),
			length(_mm256_set1_epi32(static_cast<uint32_t>(sizeof(T)))),
            offs(_mm256_setr_epi32(
            		static_cast<uint32_t>(0),
            		static_cast<uint32_t>(sizeof(T)),
					static_cast<uint32_t>(sizeof(T) * 2),
            		static_cast<uint32_t>(sizeof(T) * 3),
            		static_cast<uint32_t>(sizeof(T) * 4),
					static_cast<uint32_t>(sizeof(T) * 5),
            		static_cast<uint32_t>(sizeof(T) * 6),
            		static_cast<uint32_t>(sizeof(T) * 7))),
		    seed(_seed)

        {}

        public:

          explicit Murmur32AVX(uint32_t _seed) :
            Murmur32AVX(_mm256_set1_epi32(_seed))
        {}


          explicit Murmur32AVX( Murmur32AVX const & other) :
            Murmur32AVX(other.seed)
        {}

          explicit Murmur32AVX( Murmur32AVX && other) :
            Murmur32AVX(other.seed)
        {}

          Murmur32AVX& operator=( Murmur32AVX const & other) {
             seed = other.seed;

             return *this;
          }

          Murmur32AVX& operator=( Murmur32AVX && other) {
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
          FSC_FORCE_INLINE void hash(T const * key, uint8_t nstreams, uint32_t * out) const {
            // process 4 streams at a time.  all should be the same length.

            assert((nstreams <= 8) && "maximum number of streams is 8");
            assert((nstreams > 0) && "minimum number of streams is 1");

            __m256i h1 = hash8(key);

            // store all 8 out
            switch (nstreams) {
			  case 8: _mm256_storeu_si256((__m256i*)out, h1);  // sse
				break;
//			  case 7: out[6] = _mm256_extract_epi32(h1, 6);   // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
//			  case 6: *(reinterpret_cast<uint64_t *>(out)) = _mm256_extract_epi64(h1, 0);
//                *(reinterpret_cast<uint64_t *>(out) + 1) = _mm256_extract_epi64(h1, 1);
//                *(reinterpret_cast<uint64_t *>(out) + 2) = _mm256_extract_epi64(h1, 2);
//                break;
//			  case 5: out[4] = _mm256_extract_epi32(h1, 4);
//              case 4: *(reinterpret_cast<uint64_t *>(out)) = _mm256_extract_epi64(h1, 0);
//                *(reinterpret_cast<uint64_t *>(out) + 1) = _mm256_extract_epi64(h1, 1);
//                break;
//              case 3: out[2] = _mm256_extract_epi32(h1, 2);   // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
//              case 2: *(reinterpret_cast<uint64_t *>(out)) = _mm256_extract_epi64(h1, 0);
//                break;
//              case 1: out[0] = _mm256_extract_epi32(h1, 0);
              default:
                  uint32_t *tmp = reinterpret_cast<uint32_t*>(&h1);
                  memcpy(out, tmp, nstreams << 2);   // copy bytes
                break;;
            }
          }


          // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
          // above 2, the finalize and updates will dominate and better to do those in parallel.


          FSC_FORCE_INLINE void hash8(T const *  key, uint32_t * out) const {
            __m256i res = hash8(key);
            _mm256_storeu_si256((__m256i*)out, res);
          }


          /// NOTE: non-power of 2 length keys ALWAYS use AVX gather, which may be slower.
          template <size_t len = bytes,
              typename std::enable_if<((len & (len - 1)) > 0) && ((len & 31) > 0), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 8 streams at a time.  all should be the same length.

            // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
        	  // and can pipeline 4 at a time, about 40 cycles?
            // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
            // while we still have 8 "update"s, the programming cost is becoming costly.
        	// an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
        	// still need to shuffle more than 4 times.

            __m256i k0, k1, k2, k3;
            __m256i h1 = seed;

            //----------
            // do it in blocks of 4 bytes.
            const int nblocks = len >> 2;
            const int nblocks4 = nblocks - (nblocks & 0x3);
            const int nblocks_rem = nblocks & 0x3;

            // first do blocks of 4 blocks
            int i = 0;
            for (; i < nblocks4; i += 4) {
              k0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i, offs, 1);  // avx2
              k1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 1, offs, 1);  // avx2
              k2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 2, offs, 1);  // avx2
              k3 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 3, offs, 1);  // avx2

              // already at the right places, so just call update32.
              // row 0
              h1 = update32(h1, k0); // transpose 4x2  SSE2
              h1 = update32(h1, k1); // transpose 4x2  SSE2
              h1 = update32(h1, k2); // transpose 4x2  SSE2
              h1 = update32(h1, k3); // transpose 4x2  SSE2
            }

            // next do the remaining blocks.
            switch (nblocks_rem) {
				case 3: k2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 2, offs, 1);  // avx2
				case 2: k1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 1, offs, 1);  // avx2
				case 1: k0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i, offs, 1);  // avx2
				default:
					break;
            }
            if (nblocks_rem > 0) h1 = update32(h1, k0);
            if (nblocks_rem > 1) h1 = update32(h1, k1);
            if (nblocks_rem > 2) h1 = update32(h1, k2);

            // next do the last bytes
            if (len & 0x3) {
            	k0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + nblocks, offs, 1);  // avx2

            	h1 = update32_zeroing(h1, k0, len & 0x3);
            }

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }

#if 0
          // USING load, INSERT plus unpack is FASTER than i32gather.
          /// NOTE: multiples of 32.
          template <size_t len = bytes,
              typename std::enable_if<((len & 31) == 0), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8_old(T const *  key) const {
            // process 8 streams at a time.  all should be the same length.

            // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
            // and can pipeline 4 at a time, about 40 cycles?
            // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
            // while we still have 8 "update"s, the programming cost is becoming costly.
          // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
          // still need to shuffle more than 4 times.

            __m256i k0, k1, k2, k3; //, k4, k5, k6, k7;
            __m256i h1 = seed;

            //----------
            // do it in blocks of 4 bytes.
            const int nblocks = len >> 2;

            // gathering 8 keys at a time, 4 bytes each time, so there is 1 gather for each 4 bytes..
            int i = 0;
            for (; i < nblocks; i+=4) {
              k0 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i    , offs, 1);  // avx2
              k1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 1, offs, 1);  // avx2
              k2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 2, offs, 1);  // avx2
              k3 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(key) + i + 3, offs, 1);  // avx2

              // already at the right places, so just call update32.
              // row 0
              h1 = update32(h1, k0); // transpose 4x2  SSE2
              h1 = update32(h1, k1); // transpose 4x2  SSE2
              h1 = update32(h1, k2); // transpose 4x2  SSE2
              h1 = update32(h1, k3); // transpose 4x2  SSE2

            }

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }
#endif

          // faster than gather.
          /// NOTE: multiples of 32.
          template <size_t len = bytes,
              typename std::enable_if<((len & 31) == 0), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 8 streams at a time.  all should be the same length.

            // at this point, i32gather with its 20 cycle latency and 5 to 10 cycle CPI, no additional cost,
            // and can pipeline 4 at a time, about 40 cycles?
            // becomes competitive vs the shuffle/blend/permute approach, which grew superlinearly from 8 to 16 byte elements.
            // while we still have 8 "update"s, the programming cost is becoming costly.
          // an alternative might be using _mm256_set_m128i(_mm_lddqu_si128, _mm_lddqu_si128), which has about 5 cycles latency in total.
          // still need to shuffle more than 4 times.

            __m128i j0, j1, j2, j3, j4, j5, j6, j7;
            __m256i k0, k1, k2, k3, t0, t1, t2;
            __m256i h1 = seed;

            //----------
            // do it in blocks of 16 bytes.
            const int nblocks = len >> 4;

            // gathering 8 keys at a time, 4 bytes each time, so there is 1 gather for each 4 bytes..
            int i = 0;
            for (; i < nblocks; ++i) {
              j0 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key)      + i); // SSE2
              j1 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 1 ) + i);
              j2 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 2 ) + i);
              j3 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 3 ) + i);
              j4 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 4 ) + i);
              j5 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 5 ) + i);
              j6 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 6 ) + i);
              j7 = _mm_lddqu_si128(reinterpret_cast<__m128i const*>(key + 7 ) + i);

              // get the 32 byte vector.
              // mixing 1st and 4th, so don't have to cross boundaries again  // AVX
              k0 = _mm256_insertf128_si256(_mm256_castsi128_si256(j0), (j4), 1);
              k1 = _mm256_insertf128_si256(_mm256_castsi128_si256(j1), (j5), 1);
              k2 = _mm256_insertf128_si256(_mm256_castsi128_si256(j2), (j6), 1);
              k3 = _mm256_insertf128_si256(_mm256_castsi128_si256(j3), (j7), 1);

              // now unpack and update
              t0 = _mm256_unpacklo_epi32(k0, k1);  // abABefEF
              t1 = _mm256_unpacklo_epi32(k2, k3);  // cdCDghGH

              t2 = _mm256_unpacklo_epi64(t0, t1);  // abcdefgh
              h1 = update32(h1, t2);

              t2 = _mm256_unpackhi_epi64(t0, t1);  // ABCDEFGH
              h1 = update32(h1, t2);

              // now unpack and update
              t0 = _mm256_unpackhi_epi32(k0, k1);  // abABefEF
              t1 = _mm256_unpackhi_epi32(k2, k3);  // cdCDghGH

              t2 = _mm256_unpacklo_epi64(t0, t1);  // abcdefgh
              h1 = update32(h1, t2);

              t2 = _mm256_unpackhi_epi64(t0, t1);  // ABCDEFGH
              h1 = update32(h1, t2);

            }

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }


          template <size_t len = bytes,
              typename std::enable_if<(len == 16), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.


            // example layout, with each dash representing 4 bytes
            //     aa'AA'bb'BB' cc'CC'dd'DD' ee'EE'ff'FF' gg'GG'hh'HH'
            // k0  -- -- -- --
            // k1               -- -- -- --
            // k2                            -- -- -- --
            // k3                                         -- -- -- --


            __m256i k0, k1, k2, k3, t0, t1, t2, t3;
            __m256i h1 = seed;

            // read input, 2 keys per vector.
            k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key));  // SSE3
            k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key + 2));  // SSE3
            k2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key + 4));  // SSE3
            k3 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key + 6));  // SSE3

            // 15 extra instructions.  better than overlapping 4 gather operations.
            // if using set(m128i, m128i), 4 loads should take between 8 and 12 cycles, plus the shuffles
            	// and blends, 6, ...etc.
            // using lddqu, total latency should be about 17 from dealing with the input, and 9 2/3 CPI for 128 bytes

            // reorder k1, k2, and k3
            // k1 -> c'cC'Cd'dD'D.  [ 1 0 3 2] -> 10110001 binary == 0xB1
            k1 = _mm256_shuffle_epi32(k1, 0xB1);
            // k2 -> EE'ee'FF'ff'.  [ 2 3 0 1] -> 01001110 binary == 0x4E
            k2 = _mm256_shuffle_epi32(k2, 0x4E);
            // k3 -> G'Gg'gH'Hh'h.  [ 3 2 1 0] -> 00011011 binary == 0x1B
            k3 = _mm256_shuffle_epi32(k3, 0x1B);

            // now blend to get acACbdBD.  [ 0 1 0 1 0 1 0 1 ] -> 10101010 binary = 0xAA
            t0 = _mm256_blend_epi32(k0, k1, 0xAA);
            // now blend to get EGegFHfh.  [ 0 1 0 1 0 1 0 1 ] -> 10101010 binary = 0xAA
            t1 = _mm256_blend_epi32(k2, k3, 0xAA);
            // one more blend to get acegbdfh [ 0 0 1 1 0 0 1 1 ] -> 11001100 binary = 0xCC
            t2 = _mm256_blend_epi32(t0, t1, 0xCC);

            h1 = update32(h1, t2);

            // save t3
            // one more blend to get EGACFHBD [ 0 0 1 1 0 0 1 1 ] -> 11001100 binary = 0xCC
            t3 = _mm256_blend_epi32(t1, t0, 0xCC);
            // and shuffle to get ACEGBDFH [ 2 3 0 1] -> 0x4E
            t3 = _mm256_shuffle_epi32(t3, 0x4E);

            // now blend to get c'a'C'A'd'b'D'B'.
            t0 =  _mm256_blend_epi32(k1, k0, 0xAA);
            // and blend to get G'E'g'e'H'F'h'f'
            t1 =  _mm256_blend_epi32(k3, k2, 0xAA);
            // one more blend to get c'a'g'e'd'b'h'f'  [ 0 0 1 1 0 0 1 1 ] -> 11001100 binary = 0xCC
            t2 = _mm256_blend_epi32(t0, t1, 0xCC);
            // and a shuffle to get a'c'e'g'b'd'f'h' [ 1 0 3 2 ] -> 0xB1
            t2 = _mm256_shuffle_epi32(t2, 0xB1);


            h1 = update32(h1, t2);

            h1 = update32(h1, t3);

            // now compute d2
            // one more blend to get G'E'C'A'H'F'D'B'  [ 0 0 1 1 0 0 1 1 ] -> 11001100 binary = 0xCC
            t2 = _mm256_blend_epi32(t1, t0, 0xCC);
            // and a shuffle to get A'C'E'G'B'D'F'H' [ 3 2 1 0 ] -> 0x1B
            t2 = _mm256_shuffle_epi32(t2, 0x1B);

            h1 = update32(h1, t2);


            // permute to make a"b"c"d" e"f"g"h".  crosses lane here.
            // [ 0 4 1 5 2 6 3 7 ]
            h1 = _mm256_permutevar8x32_epi32(h1, permute16);   // latency 3, CPI 1

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }


          template <size_t len = bytes,
              typename std::enable_if<(len == 8), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // example layout, with each dash representing 4 bytes
            //     aAbBcCdD eEfFgGhH
            // k0  --------
            // k1           --------

            __m256i k0, k1, t0, t1;
            __m256i h1 = seed;

            // read input, 4 keys per vector.
            k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key));  // SSE3
            k1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key + 4));  // SSE3

            // 5 extra inst, 7 latency cycles, and 3 2/3 CPI. for 64 bytes Can this be done faster?

            // make EeFfGgHh.  use shuffle mask [ 1 0 3 2 ] -> 10110001 binary == 0xB1
            k1 = _mm256_shuffle_epi32(k1, 0xB1);   // latency 1, CPI 1

            // then blend to get aebf cgdh  // do it as [0 1 0 1 0 1 0 1] -> 10101010 binary == 0xAA
            t0 = _mm256_blend_epi32(k0, k1, 0xAA);   // latency 1, CPI 0.33
            // second blend gets us EAFB GCHD
            t1 = _mm256_blend_epi32(k1, k0, 0xAA);  // latency 1, CPI 0.33

            // now do one more shuffle to get AEBF CGDH
            t1 = _mm256_shuffle_epi32(t1, 0xB1);   // latency 1, CPI 1


            // update with t1
            h1 = update32(h1, t0); // transpose 4x2  SSE2
            // update with t0
            h1 = update32(h1, t1);

            // permute to make a'b'c'd' e'f'g'h'.  crosses lane here.
            // [ 0 2 4 6 1 3 5 7 ], which is permute1
            h1 = _mm256_permutevar8x32_epi32(h1, permute1);  // latency 3, CPI 1

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }


          template <size_t len = bytes,
              typename std::enable_if<(len == 4), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 4 bytes
            //     abcd efgh
            // k0  ---- ----

            __m256i k0;
            __m256i h1 = seed;

            // no extra inst
            // 8 keys per vector, so perfect match.
            k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key));  // SSE3

            // row 0.  unpacklo puts it to aabbccdd
            h1 = update32(h1, k0); // transpose 4x2  SSE2

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }



          template <size_t len = bytes,
              typename std::enable_if<(len == 2), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.
            __m256i zero = _mm256_setzero_si256();

            // example layout, with each dash representing 2 bytes
            //     abcdefgh ijklmnop
            // k0  -------- --------

            __m256i k0;
            __m256i h1 = seed;

            // 16 keys per vector. can potentially do 2 iters.
            k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key));  // SSE3

            // 2 extra inst

            // permute across lane, 64bits at a time, with pattern 0 2 1 3 -> 11011000 == 0xD8
            k0 = _mm256_permute4x64_epi64(k0, 0xd8);   // AVX2, latency 3, CPI 1
            // result abcd ijkl efgh mnop

            // TODO: can potentially reuse a little more.

            // transform to a0b0c0d0 e0f0g0h0.  interleave with 0.
            h1 = update32_partial(h1, _mm256_unpacklo_epi16(k0, zero), 2); // transpose 4x2  AVX2

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }


          template <size_t len = bytes,
              typename std::enable_if<(len == 1), int>::type = 1>
          FSC_FORCE_INLINE __m256i hash8(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 1 bytes
            //     abcdefghijklmnop qrstuvwxyz123456
            // k0  ---------------- ----------------

            __m256i k0, t0;
            __m256i h1 = seed;

            // 32 keys per vector, can potentially do 4 rounds.
            k0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(key));  // SSE3

            // 2 extra calls.
            // need to permute with permutevar8x32, idx = [0 2 4 6 1 3 5 7]
            k0 = _mm256_permutevar8x32_epi32(k0, permute1); // AVX2, latency of 3, CPI 1
            // abcd ijkl qrst yz12 efgh mnop uvwx 3456

            // TODO: can potentially reuse a little more.

            // USE shuffle_epi8, with mask.
            // transform to a000b000c000d000 e000f000g000h000.  interleave with 0.
            t0 = _mm256_shuffle_epi8(k0, shuffle1);  // AVX2, latency 3, CPI 1

            // row 0.
            h1 = update32_partial(h1, t0, 1); // transpose 4x2  SSE2

            //----------
            // finalization
            // or the length.
            h1 = _mm256_xor_si256(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }

      };

#endif


#if defined(__SSE4_1__)
      // for 32 bit buckets
      // original: body: 16 inst per iter of 4 bytes; tail: 15 instr. ; finalization:  8 instr.
      // about 4 inst per byte + 8, for each hash value.
      template <typename T, size_t bytes = sizeof(T)>
      class Murmur32SSE {

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
          FSC_FORCE_INLINE __m128i rotl32 ( __m128i x, int8_t r ) const
          {
            // return (x << r) | (x >> (32 - r));
            return _mm_or_si128(                // sse2
                _mm_slli_epi32(x, r),           // sse2
                _mm_srli_epi32(x, (32 - r)));   // sse2
          }

          FSC_FORCE_INLINE __m128i update32( __m128i h, __m128i k) const {
            // preprocess the 4 streams
            k = _mm_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm_xor_si128(h, k);     // SSE
            // this is done per block of 4 bytes.  the last block (smaller than 4 bytes) does not do this.  do for every byte except last,
            h = rotl32(h, 13);           // sse2
            h = _mm_add_epi32(_mm_mullo_epi32(h, c3), c4);  // SSE
            return h;
          }

          // count cannot be zero.
          FSC_FORCE_INLINE __m128i update32_zeroing( __m128i h, __m128i k, uint8_t const & count) const {
            assert((count > 0) && (count < 4) && "count should be between 1 and 3");

            unsigned int shift = (4U - count) * 8U;
            // clear the upper bytes
            k = _mm_srli_epi32(_mm_slli_epi32(k, shift), shift);	// sse2

            // preprocess the 4 streams
            k = _mm_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm_xor_si128(h, k);     // SSE
            return h;
          }

          // count cannot be zero.
          FSC_FORCE_INLINE __m128i update32_partial( __m128i h, __m128i k, uint8_t const & count) const {

            // preprocess the 4 streams
            k = _mm_mullo_epi32(k, c1);  // SSE2
            k = rotl32(k, 15); 		   // sse2
            k = _mm_mullo_epi32(k, c2);  // SSE2
            // merge with existing.
            h = _mm_xor_si128(h, k);     // SSE
            return h;
          }


          // input is 4 unsigned ints.
          // is ((h ^ f) * c) carryless multiplication with (f = h >> d)?
          FSC_FORCE_INLINE __m128i fmix32 ( __m128i h ) const
          {
            h = _mm_xor_si128(h, _mm_srli_epi32(h, 16));  // h ^= h >> 16;      sse2
            h = _mm_mullo_epi32(h, mix_const1);           // h *= 0x85ebca6b;   sse4.1
            h = _mm_xor_si128(h, _mm_srli_epi32(h, 13));  // h ^= h >> 13;      sse2
            h = _mm_mullo_epi32(h, mix_const2);           // h *= 0xc2b2ae35;   sse4.1
            h = _mm_xor_si128(h, _mm_srli_epi32(h, 16));  // h ^= h >> 16;      sse2

            return h;
          }


          explicit Murmur32SSE(__m128i _seed) :
            seed(_seed),
            mix_const1(_mm_set1_epi32(0x85ebca6b)),
            mix_const2(_mm_set1_epi32(0xc2b2ae35)),
            c1(_mm_set1_epi32(0xcc9e2d51)),
            c2(_mm_set1_epi32(0x1b873593)),
            c3(_mm_set1_epi32(0x5)),
            c4(_mm_set1_epi32(0xe6546b64)),
            zero(_mm_setzero_si128()),// SSE2
            length(_mm_set1_epi32(bytes))
        {}

        public:


          explicit Murmur32SSE(uint32_t _seed) :
            Murmur32SSE(_mm_set1_epi32(_seed))
        {}

          explicit Murmur32SSE( Murmur32SSE const & other) :
            Murmur32SSE(other.seed)
        {}

          explicit Murmur32SSE( Murmur32SSE && other) :
            Murmur32SSE(other.seed)
        {}

          Murmur32SSE& operator=( Murmur32SSE const & other) {
             seed = other.seed;

             return *this;
          }

          Murmur32SSE& operator=( Murmur32SSE && other) {
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
          FSC_FORCE_INLINE void hash(T const * key, uint8_t nstreams, uint32_t * out) const {
            // process 4 streams at a time.  all should be the same length.

            assert((nstreams <= 4) && "maximum number of streams is 4");
            assert((nstreams > 0) && "minimum number of streams is 1");

            __m128i h1 = hash4(key);

            // store all 4 out
            switch (nstreams) {
              case 4: _mm_storeu_si128((__m128i*)out, h1);  // sse
              break;
              case 3: out[2] = _mm_extract_epi32(h1, 2);   // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
              case 2: *(reinterpret_cast<uint64_t *>(out)) = _mm_extract_epi64(h1, 0);
              break;
              case 1: out[0] = _mm_extract_epi32(h1, 0);
              default:
                break;;
            }
          }

          // TODO: [ ] hash1, do the k transform in parallel.  also use mask to keep only part wanted, rest of update and finalize do sequentially.
          // above 2, the finalize and updates will dominate and better to do those in parallel.

          FSC_FORCE_INLINE void hash4(T const *  key, uint32_t * out) const {
            __m128i res = hash4(key);
            _mm_storeu_si128((__m128i*)out, res);
          }

          // not power of 2, and not multiples of 16.
          template <uint64_t len = bytes,
              typename std::enable_if<((len & (len - 1)) > 0) && ((len & 15) > 0), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            __m128i k0, k1, k2, k3, t0, t1, t2, t3;
            __m128i h1 = seed;

            //----------
            // first do blocks of 16 bytes.
            const int nblocks = len >> 4;

            // init to zero
            for (int i = 0; i < nblocks; ++i) {
              k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key) + i);  // SSE3
              k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 1) + i);  // SSE3
              k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2) + i);  // SSE3
              k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 3) + i);  // SSE3

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
              h1 = update32(h1, _mm_unpacklo_epi64(t2, t3));  // transpose 4x2  SSE2

              // row 3
              h1 = update32(h1, _mm_unpackhi_epi64(t2, t3));   // transpose 4x2  SSE2
            }

            // next do the remainder if any.
            if ((len & 0xF) > 0) {

              // read more stream.  over read, and zero out.
              k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key) + nblocks);  // SSE3
              k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 1) + nblocks);  // SSE3
              k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2) + nblocks);  // SSE3
              k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 3) + nblocks);  // SSE3

              // transpose (8 ops), set missing bytes to 0 (2 extra ops), and do as many words as there are.

              // needed by all cases.
              t0 = _mm_unpacklo_epi32(k0, k1);         // transpose 2x2   SSE2
              t1 = _mm_unpacklo_epi32(k2, k3);         // transpose 2x2   SSE2

              // zeroing out unused bytes takes 2 instructions.
              const uint8_t words = ((len & 15UL) + 3) >> 2;  // use ceiling word count.  else case 3 needs conditional to check for 12 bytes
              const uint8_t rem = len & 0x3;
              //            std::cout << " len " << static_cast<size_t>(len) << " words " << static_cast<size_t>(words) <<
              //                " rem " << static_cast<size_t>(rem) << std::endl;
              switch (words) {
                case 4:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1));
                  // last word needs to be padded with 0.
                  //                h1 = (rem > 0) ?
                  h1 =  update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem);  // remainder  > 0
                  //                    update32(h1, _mm_unpackhi_epi64(t0, t1));
                  break;
                case 3:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
                  // last word needs to be padded with 0.  3 rows only
                  h1 = (rem > 0) ?
                      update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                case 2:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  // last word needs to be padded with 0.  2 rows only.  rem >= 0
                  h1 = (rem > 0) ?
                      update32_zeroing(h1, _mm_unpackhi_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpackhi_epi64(t0, t1));
                  break;
                case 1:
                  // last word needs to be padded with 0.  1 rows only.  remainder must be >= 0
                  h1 = (rem > 0) ?
                      update32_zeroing(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                default:
                  break;
              }

            }

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }


          // multiples of 16
          template <uint64_t len = bytes,
              typename std::enable_if<((len & 15) == 0), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            __m128i k0, k1, k2, k3, t0, t1;
            __m128i h1 = seed;

            //----------
            // do blocks of 16 bytes.
            const int nblocks = len >> 4;

            // init to zero
            for (int i = 0; i < nblocks; ++i) {

              k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key) + i);  // SSE3
              k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 1) + i);  // SSE3
              k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2) + i);  // SSE3
              k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 3) + i);  // SSE3

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
              h1 = update32(h1, _mm_unpacklo_epi64(t0, t1));  // transpose 4x2  SSE2

              // row 3
              h1 = update32(h1, _mm_unpackhi_epi64(t0, t1));   // transpose 4x2  SSE2
            }
            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }



          template <uint64_t len = bytes,
              typename std::enable_if<(len == 8), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 2 bytes
            //     aaAAbbBBccCCddDD
            // k0  --------
            // k1          --------

            __m128i k0, k1, t0, t1;
            __m128i h1 = seed;

            // read input
            k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key));  // SSE3
            k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key + 2));  // SSE3

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
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }


          template <uint64_t len = bytes,
              typename std::enable_if<(len == 4), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 2 bytes
            //     aabbccdd
            // k0  --------

            __m128i k0;
            __m128i h1 = seed;

            // no extra inst
            // blocks of 4
            k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key));  // SSE3

            // row 0.  unpacklo puts it to aabbccdd
            h1 = update32(h1, k0); // transpose 4x2  SSE2

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }

          template <uint64_t len = bytes,
              typename std::enable_if<(len == 2), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 2 bytes
            //     abcdefgh
            // k0  --------

            __m128i k0;
            __m128i h1 = seed;

            // blocks of 2
            k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key));  // SSE3

            // 1 extra inst
            // transform to a0b0c0d0.  interleave with 0.
            // row 0.
            h1 = update32_partial(h1, _mm_unpacklo_epi16(k0, zero), 2); // transpose 4x2  SSE2

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }


          template <uint64_t len = bytes,
              typename std::enable_if<(len == 1), int>::type = 1>
          FSC_FORCE_INLINE __m128i hash4(T const *  key) const {
            // process 4 streams at a time.  all should be the same length.

            // example layout, with each dash representing 1 bytes
            //     abcdefghijklmop
            // k0  ---------------

            __m128i k0, t0;
            __m128i h1 = seed;

            // blocks of 2
            k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key));  // SSE3

            // 2 extra inst.
            // transform to a0b0c0d0e0f0g0h0.  interleave with 0.
            t0 = _mm_unpacklo_epi8(k0, zero);

            // transform to a000b000c000d000.  interleave with 0.
            // row 0.
            h1 = update32_partial(h1, _mm_unpacklo_epi16(t0, zero), 1); // transpose 4x2  SSE2

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, length);  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;
          }

      };


//      NOTE: no _mm_mullo_epi64 below avx512, so no point in this for Broadwell and earlier.
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
    class identity {

      public:
        static constexpr uint8_t batch_size = 1;
        using result_type = uint64_t;
        using argument_type = T;


        /// operator to compute hash value
        inline uint64_t operator()(const T & key) const {
          if (sizeof(T) >= 8)  // more than 64 bits, so use the lower 64 bits.
            return *(reinterpret_cast<uint64_t const *>(&key));
          else if (sizeof(T) == 4)
            return *(reinterpret_cast<uint32_t const *>(&key));
          else if (sizeof(T) == 2)
            return *(reinterpret_cast<uint16_t const *>(&key));
          else if (sizeof(T) == 1)
            return *(reinterpret_cast<uint8_t const *>(&key));
          else {
            // copy into 64 bits
            uint64_t out = 0;
            memcpy(&out, &key, sizeof(T));
            return out;
          }
        }
    };
    template <typename T> constexpr uint8_t identity<T>::batch_size;



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
    class murmur3avx32 {


      protected:
        ::fsc::hash::sse::Murmur32AVX<T> hasher;
        mutable uint32_t temp[8];

      public:
        static constexpr uint8_t batch_size = 8;
        using result_type = uint32_t;
        using argument_type = T;

        murmur3avx32(uint32_t const & _seed = 43 ) : hasher(_seed) {};

        inline uint32_t operator()(const T & key) const
        {
          uint32_t h;
          hasher.hash(&key, 1, &h);

          return h;
        }

        FSC_FORCE_INLINE void operator()(T const * keys, size_t count, uint32_t * results) const
        {
          hash(keys, count, results);
        }

        // results always 32 bit.
        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint32_t * results) const {

          size_t rem = count & 0x7;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 8) {

            hasher.hash8(&(keys[i]), results + i);
          }

          if (rem > 0)
            hasher.hash(&(keys[i]), rem, results + i);

        }

        // assume consecutive memory layout.
        template<typename OT>
        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          size_t rem = count & 0x7;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 8) {
            hasher.hash8(&(keys[i]), temp);
            results[i] = temp[0] % modulus;
            results[i+1] = temp[1] % modulus;
            results[i+2] = temp[2] % modulus;
            results[i+3] = temp[3] % modulus;
            results[i+4] = temp[4] % modulus;
            results[i+5] = temp[5] % modulus;
            results[i+6] = temp[6] % modulus;
            results[i+7] = temp[7] % modulus;
          }


          if (rem > 0)
            hasher.hash(&(keys[i]), rem, temp);

          switch(rem) {
          case 7: results[i+6] = temp[6] % modulus;
          case 6: results[i+5] = temp[5] % modulus;
          case 5: results[i+4] = temp[4] % modulus;
          case 4: results[i+3] = temp[3] % modulus;
            case 3: results[i+2] = temp[2] % modulus;
            case 2: results[i+1] = temp[1] % modulus;
            case 1: results[i] = temp[0] % modulus;
            default:
              break;
          }
        }

        // assume consecutive memory layout.
        // note that the paremter is modulus bits.
        template<typename OT>
        FSC_FORCE_INLINE void hash_and_mod_pow2(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
          --modulus;

          size_t rem = count & 0x4;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 8) {
            hasher.hash8(&(keys[i]), temp);
            results[i]   = temp[0] & modulus;
            results[i+1] = temp[1] & modulus;
            results[i+2] = temp[2] & modulus;
            results[i+3] = temp[3] & modulus;
            results[i+4]   = temp[4] & modulus;
            results[i+5] = temp[5] & modulus;
            results[i+6] = temp[6] & modulus;
            results[i+7] = temp[7] & modulus;
          }

          // last part.
          if (rem > 0)
            hasher.hash(&(keys[i]), rem, temp);
          switch(rem) {
          case 7: results[i+6] = temp[6] & modulus;
          case 6: results[i+5] = temp[5] & modulus;
          case 5: results[i+4] = temp[4] & modulus;
          case 4: results[i+3] = temp[3] & modulus;
            case 3: results[i+2] = temp[2] & modulus;
            case 2: results[i+1] = temp[1] & modulus;
            case 1: results[i]   = temp[0] & modulus;
            default:
              break;
          }
        }

        // TODO: [ ] add a transform_hash_mod.

    };
    template <typename T> constexpr uint8_t murmur3avx32<T>::batch_size;

#endif

#if defined(__SSE4_1__)
    /**
     * @brief MurmurHash.  using lower 64 bits.
     * @details.  prefetching did not help
     */
    template <typename T>
    class murmur3sse32 {


      protected:
        ::fsc::hash::sse::Murmur32SSE<T> hasher;
        mutable uint32_t temp[4];

      public:
        static constexpr uint8_t batch_size = 4;
        using result_type = uint32_t;
        using argument_type = T;

        murmur3sse32(uint32_t const & _seed = 43 ) : hasher(_seed) {};

        inline uint32_t operator()(const T & key) const
        {
          uint32_t h;
          hasher.hash(&key, 1, &h);
          return h;
        }

        FSC_FORCE_INLINE void operator()(T const * keys, size_t count, uint32_t * results) const
        {
          hash(keys, count, results);
        }


        // results always 32 bit.
        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint32_t * results) const {
          size_t rem = count & 0x3;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 4) {
            hasher.hash4(&(keys[i]), results + i);
          }

          if (rem > 0)
            hasher.hash(&(keys[i]), rem, results + i);
        }

        // assume consecutive memory layout.
        template<typename OT>
        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          size_t rem = count & 0x3;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 4) {
            hasher.hash4(&(keys[i]), temp);
            results[i] = temp[0] % modulus;
            results[i+1] = temp[1] % modulus;
            results[i+2] = temp[2] % modulus;
            results[i+3] = temp[3] % modulus;
          }

          // last part.
          if (rem > 0)
            hasher.hash(&(keys[i]), rem, temp);
          switch(rem) {
            case 3: results[i+2] = temp[2] % modulus;
            case 2: results[i+1] = temp[1] % modulus;
            case 1: results[i] = temp[0] % modulus;
            default:
              break;
          }
        }

        // assume consecutive memory layout.
        // note that the paremter is modulus bits.
        template<typename OT>
        FSC_FORCE_INLINE void hash_and_mod_pow2(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
          --modulus;

          size_t rem = count & 0x3;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 4) {
            hasher.hash4(&(keys[i]), temp);
            results[i]   = temp[0] & modulus;
            results[i+1] = temp[1] & modulus;
            results[i+2] = temp[2] & modulus;
            results[i+3] = temp[3] & modulus;
          }

          // last part.
          if (rem > 0)
            hasher.hash(&(keys[i]), rem, temp);
          switch(rem) {
            case 3: results[i+2] = temp[2] & modulus;
            case 2: results[i+1] = temp[1] & modulus;
            case 1: results[i]   = temp[0] & modulus;
            default:
              break;
          }
        }


        // TODO: [ ] add a transform_hash_mod.
    };
    template <typename T> constexpr uint8_t murmur3sse32<T>::batch_size;



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
    class murmur32 {


      protected:
        uint32_t seed;

      public:
        static constexpr uint8_t batch_size = 1;
        using result_type = uint32_t;
        using argument_type = T;

        murmur32(uint32_t const & _seed = 43 ) : seed(_seed) {};

        inline uint32_t operator()(const T & key) const
        {
          // produces 128 bit hash.
          uint32_t h;
          // let compiler optimize out all except one of these.
          MurmurHash3_x86_32(&key, sizeof(T), seed, &h);

          // use the upper 64 bits.
          return h;
        }

    };
    template <typename T> constexpr uint8_t murmur32<T>::batch_size;


    /**
     * @brief MurmurHash.  using lower 64 bits.
     *
     */
    template <typename T>
    class murmur {


      protected:
        uint32_t seed;

      public:
        static constexpr uint8_t batch_size = 1;
        using result_type = uint64_t;
        using argument_type = T;


        murmur(uint32_t const & _seed = 43 ) : seed(_seed) {};

        inline uint64_t operator()(const T & key) const
        {
          // produces 128 bit hash.
          uint64_t h[2];
          // let compiler optimize out all except one of these.
          if (sizeof(void*) == 8)
            MurmurHash3_x64_128(&key, sizeof(T), seed, h);
          else if (sizeof(void*) == 4)
            MurmurHash3_x86_128(&key, sizeof(T), seed, h);
          else
            throw ::std::logic_error("ERROR: neither 32 bit nor 64 bit system");

          // use the upper 64 bits.
          return h[0];
        }

    };
    template <typename T> constexpr uint8_t murmur<T>::batch_size;



#if defined(__SSE4_2__)

    /**
     * @brief crc.  32 bit hash..
     * @details  operator should require sizeof(T)/8  + 6 operations + 2 cycle latencies.
     *            require SSE4.2
     *
     *          prefetching did not help
     *   algo at https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_crc&expand=5247,1244
     */
    template <typename T>
    class crc32c {


      protected:
        uint32_t seed;
        static constexpr size_t blocks = sizeof(T) >> 3;   // divide by 8
        static constexpr size_t rem = sizeof(T) & 0x7;  // remainder.
        static constexpr size_t offset = (sizeof(T) >> 3) << 3;
        uint32_t temp[4];

        uint32_t hash1(const T & key) const {

          // block of 8 bytes
          uint64_t crc64 = seed;
          if (sizeof(T) >= 8) {
            uint64_t const * data64 = reinterpret_cast<uint64_t const *>(&key);
            for (size_t i = 0; i < blocks; ++i) {
              crc64 = _mm_crc32_u64(crc64, data64[i]);
            }
          }

          uint32_t crc = static_cast<uint32_t>(crc64);

          unsigned char const * data = reinterpret_cast<unsigned char const *>(&key);

          // rest.  do it cleanly
          size_t off = offset;  // * 8
          if (rem & 0x4) {  // has 4 bytes
            crc = _mm_crc32_u32(crc, *(reinterpret_cast<uint32_t const *>(data + off)));  off += 4;
          }
          if (rem & 0x2) {  // has 2 bytes extra
            crc = _mm_crc32_u16(crc, *(reinterpret_cast<uint16_t const *>(data + off)));  off += 2;
          }
          if (rem & 0x1) {  // has 1 byte extra
            crc =  _mm_crc32_u8(crc, *(reinterpret_cast< uint8_t const *>(data + off)));
          }

          return crc;
        }

        void hash4(T const * keys, uint32_t * results) const {
          // loop over 3 keys at a time
          uint64_t aa = seed;
          uint64_t bb = seed;
          uint64_t cc = seed;
          uint64_t dd = seed;

          if (sizeof(T) >= 8) {
            // block of 8 bytes
            uint64_t const *data64a = reinterpret_cast<uint64_t const *>(&(keys[0]));
            uint64_t const *data64b = reinterpret_cast<uint64_t const *>(&(keys[1]));
            uint64_t const *data64c = reinterpret_cast<uint64_t const *>(&(keys[2]));
            uint64_t const *data64d = reinterpret_cast<uint64_t const *>(&(keys[3]));

            for (size_t i = 0; i < blocks; ++i) {
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

          unsigned char const * dataa = reinterpret_cast<unsigned char const *>(&(keys[0]));
          unsigned char const * datab = reinterpret_cast<unsigned char const *>(&(keys[1]));
          unsigned char const * datac = reinterpret_cast<unsigned char const *>(&(keys[2]));
          unsigned char const * datad = reinterpret_cast<unsigned char const *>(&(keys[3]));

          // rest.  do it cleanly
          size_t off = offset;  // * 8
          if (rem & 0x4) {  // has 4 bytes
            a = _mm_crc32_u32(a, *(reinterpret_cast<uint32_t const *>(dataa + off)));
            b = _mm_crc32_u32(b, *(reinterpret_cast<uint32_t const *>(datab + off)));
            c = _mm_crc32_u32(c, *(reinterpret_cast<uint32_t const *>(datac + off)));
            d = _mm_crc32_u32(d, *(reinterpret_cast<uint32_t const *>(datad + off)));
            off += 4;
          }
          if (rem & 0x2) {  // has 2 bytes extra
            a = _mm_crc32_u16(a, *(reinterpret_cast<uint16_t const *>(dataa + off)));
            b = _mm_crc32_u16(b, *(reinterpret_cast<uint16_t const *>(datab + off)));
            c = _mm_crc32_u16(c, *(reinterpret_cast<uint16_t const *>(datac + off)));
            d = _mm_crc32_u16(d, *(reinterpret_cast<uint16_t const *>(datad + off)));
            off += 2;
          }
          if (rem & 0x1) {  // has 1 byte extra
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

        crc32c(uint32_t const & _seed = 37 ) : seed(_seed) {};

        // do 1 element.
        FSC_FORCE_INLINE uint32_t operator()(const T & key) const
        {
          return hash1(key);
        }

        FSC_FORCE_INLINE void operator()(T const * keys, size_t count, uint32_t * results) const
        {
          hash(keys, count, results);
        }

        // results always 32 bit.
        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        // however, to limit the modulus, do 4 at a time.
        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint32_t * results) const {
          // loop over 3 keys at a time
          size_t max = count - (count & 3);
          size_t i = 0;
          for (; i < max; i += 4) {
            hash4(keys + i, results + i);
          }

          // handle the remainder
          for (; i < count; ++i) {
            results[i] = hash1(keys[i]);
          }
        }

        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        template <typename OT>
        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          // loop over 3 keys at a time
          size_t max = count - (count & 3);
          size_t i = 0;
          for (; i < max; i += 4) {
            hash4(keys + i, temp);

            results[i]     = temp[0] % modulus;
            results[i + 1] = temp[1] % modulus;
            results[i + 2] = temp[2] % modulus;
            results[i + 3] = temp[3] % modulus;
          }

          // handle the remainder
          for (; i < count; ++i) {
            results[i] = hash1(keys[i]) % modulus;
          }
        }

        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        template <typename OT>
        FSC_FORCE_INLINE void hash_and_mod_pow2(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");

          --modulus;  // convert to mask.

          // loop over 3 keys at a time
          size_t max = count - (count & 3);
          size_t i = 0;
          for (; i < max; i += 4) {
            hash4(keys + i, temp);

            results[i]     = temp[0] & modulus;
            results[i + 1] = temp[1] & modulus;
            results[i + 2] = temp[2] & modulus;
            results[i + 3] = temp[3] & modulus;
          }

          // handle the remainder
          for (; i < count; ++i) {
            results[i] = hash1(keys[i]) & modulus;
          }
        }
    };
    template <typename T> constexpr uint8_t crc32c<T>::batch_size;

#endif



    /**
     * @brief  farm hash
     *
     * MAY NOT WORK CONSISTENTLY between prefetching on and off.
     */
    template <typename T>
    class farm {

      protected:
        uint64_t seed;

      public:
        static constexpr uint8_t batch_size = 1;
        using result_type = uint64_t;
        using argument_type = T;

        farm(uint64_t const & _seed = 43 ) : seed(_seed) {};

        /// operator to compute hash.  64 bit again.
        inline uint64_t operator()(const T & key) const {
          return ::util::Hash64WithSeed(reinterpret_cast<const char*>(&key), sizeof(T), seed);
        }
    };
    template <typename T> constexpr uint8_t farm<T>::batch_size;

    template <typename T>
    class farm32 {

      protected:
        uint32_t seed;

      public:
        static constexpr uint8_t batch_size = 1;
        using result_type = uint32_t;
        using argument_type = T;

        farm32(uint32_t const & _seed = 43 ) : seed(_seed) {};

        /// operator to compute hash.  64 bit again.
        inline uint32_t operator()(const T & key) const {
          return ::util::Hash32WithSeed(reinterpret_cast<const char*>(&key), sizeof(T), seed);
        }
    };
    template <typename T> constexpr uint8_t farm32<T>::batch_size;





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
  class TransformedHash {

  protected:
      using PRETRANS_T = PreTransform<Key>;

      // determine output type of PreTransform.
      using PRETRANS_VAL_TYPE =
    		  decltype(::std::declval<PRETRANS_T>().operator()(::std::declval<Key>()));

      using HASH_T = Hash<PRETRANS_VAL_TYPE>;

  public:
      // determine output type of hash.  could be 64 bit or 32 bit.
      using HASH_VAL_TYPE =
    		  decltype(::std::declval<Hash<PRETRANS_VAL_TYPE> >().operator()(::std::declval<PRETRANS_VAL_TYPE>()));

      // determine output type of return value.
      using result_type =
    		  decltype(::std::declval<PostTransform<HASH_VAL_TYPE> >().operator()(::std::declval<HASH_VAL_TYPE>()));
      using argument_type = Key;

////      TODO: [ ] make below fallback to 1 if batch_size is not defined.
//      constexpr size_t pretrans_batch_size = PRETRANS_T::batch_size;
//      constexpr size_t hash_batch_size = 	 HASH_T::batch_size;
//      constexpr size_t postrans_batch_size = POSTTRANS_T::batch_size;

	  // lowest common multiple of the three.  default to 64byte/sizeof(HASH_VAL_TYPE) for now (cacheline multiple)
      static constexpr uint8_t batch_size = (sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);
      	  // HASH_T::batch_size; //(sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);

      static_assert((batch_size & (batch_size - 1)) == 0, "ERROR: batch_size should be a power of 2.");

  protected:
      using POSTTRANS_T = PostTransform<HASH_VAL_TYPE>;

      // need some buffers
      mutable Key * key_buf;
      mutable PRETRANS_VAL_TYPE * trans_buf;
      mutable HASH_VAL_TYPE * hash_buf;

  public:
      PRETRANS_T trans;
      HASH_T h;
      POSTTRANS_T posttrans;

      TransformedHash(HASH_T const & _hash = HASH_T(),
                      PRETRANS_T const & pre_trans = PRETRANS_T(),
                      POSTTRANS_T const & post_trans = POSTTRANS_T()) :
				//batch_size(lcm(lcm(pretrans_batch_size, hash_batch_size), postrans_batch_size)),
				key_buf(nullptr), trans_buf(nullptr), hash_buf(nullptr),
				trans(pre_trans), h(_hash), posttrans(post_trans) {
    	  key_buf = ::utils::mem::aligned_alloc<Key>(batch_size);
    	  trans_buf = ::utils::mem::aligned_alloc<PRETRANS_VAL_TYPE>(batch_size);
    	  hash_buf = ::utils::mem::aligned_alloc<HASH_VAL_TYPE>(batch_size);
      };

      ~TransformedHash() {
    	  free(key_buf);
    	  free(trans_buf);
    	  free(hash_buf);
      }

      // conditionally defined, there should be just 1 defined methods after compiler resolves all this.
      // note that the compiler may do the same if it notices no-op....
      template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline result_type operator()(Key const& k) const {
        //std::cout << "op 1.1" << std::endl;
        return h(k);
      }
      template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline result_type operator()(Key const& k) const {
        //std::cout << "op 1.2" << std::endl;
        return h(trans(k));
      }
      template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline result_type operator()(Key const& k) const {
        //std::cout << "op 1.3" << std::endl;
        return posttrans(h(k));
      }
      template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline result_type operator()(Key const& k) const {
        //std::cout << "op 1.4" << std::endl;
        return posttrans(h(trans(k)));
      }

      template<typename V>
      inline result_type operator()(::std::pair<Key, V> const& x) const {
        //std::cout << "op 2.1" << std::endl;
        return this->operator()(x.first);
      }
      template<typename V>
      inline result_type operator()(::std::pair<const Key, V> const& x) const {
        //std::cout << "op 2.2" << std::endl;
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
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value, // &&
//					::std::is_same<size_t,
//					 decltype(std::declval<HT>().operator()(
//							 std::declval<PRETRANS_VAL_TYPE const *>(),
//							 std::declval<size_t>(),
//							 std::declval<HASH_VAL_TYPE *>()))>::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, int) const
      -> decltype(::std::declval<HT>()(k, count, out), size_t())
      {
        //std::cout << "op 3.1" << std::endl;
    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0;
    	  for (; i < max; i += batch_size ) {
        	  h(k+i, batch_size, out+i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, int) const
      -> decltype(::std::declval<HT>()(*k), size_t()) {
        //std::cout << "op 3.2" << std::endl;
    	  return 0;
      }
      // pretrans is not identity, post is identity.
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.3" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  h(trans_buf, batch_size, out + i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf), size_t()) {
    	  // first part.
        //std::cout << "op 3.4" << std::endl;
    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  for (j = 0; j < batch_size; ++j) out[i+j] = h(trans_buf[j]);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, int, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.5" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) trans_buf[j] = trans(k[i+j]);
    		  h(trans_buf, batch_size, out + i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				   ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, long, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(*trans_buf), size_t()) {
        //std::cout << "op 3.6" << std::endl;

    	  return 0;
      }
      // posttrans is not identity, post is identity.
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, int) const
      -> decltype(::std::declval<HT>()(k, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.7" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0;
    	  for (; i < max; i += batch_size ) {
    		  h(k + i, batch_size, hash_buf);
    		  posttrans(hash_buf, batch_size, out + i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, int) const
      -> decltype(::std::declval<HT>()(*k), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.8" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) hash_buf[j] = h(k[i+j]);
    		  posttrans(hash_buf, batch_size, out + i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, long) const
      -> decltype(::std::declval<HT>()(k, count, hash_buf), ::std::declval<PoT>()(*hash_buf), size_t()) {
    	  // first part.
        //std::cout << "op 3.9" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  h(k + i, batch_size, hash_buf);
    		  for (j = 0; j < batch_size; ++j) out[i+j] = posttrans(hash_buf[j]);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  ::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, long) const
      -> decltype(::std::declval<HT>()(*k), ::std::declval<PoT>()(*hash_buf), size_t()) {

        //std::cout << "op 3.10" << std::endl;

    	  return 0;
      }
      // ==== none are identity
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.11" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  h(trans_buf, batch_size, hash_buf);
    		  posttrans(hash_buf, batch_size, out+i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, int) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.12" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  for (j = 0; j < batch_size; ++j) hash_buf[j] = h(trans_buf[j]);
    		  posttrans(hash_buf, batch_size, out+i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, int, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, hash_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {
    	  // first part.
        //std::cout << "op 3.13" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) trans_buf[j] = trans(k[i+j]);
    		  h(trans_buf, batch_size, hash_buf);
    		  posttrans(hash_buf, batch_size, out+i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, long, int) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(hash_buf, count, out), size_t()) {

        //std::cout << "op 3.14" << std::endl;


    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) hash_buf[j] = h(trans(k[i+j]));
    		  posttrans(hash_buf, batch_size, out+i);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, int, long) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(trans_buf, count, hash_buf),  ::std::declval<PoT>()(*hash_buf), size_t()) {
    	  // first part.
        //std::cout << "op 3.15" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  h(trans_buf, batch_size, hash_buf);
    		  for (j = 0; j < batch_size; ++j) out[i+j] = posttrans(hash_buf[j]);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, int, long, long) const
      -> decltype(::std::declval<PrT>()(k, count, trans_buf), ::std::declval<HT>()(*trans_buf),  ::std::declval<PoT>()(*hash_buf), size_t()) {
    	  // first part.
        //std::cout << "op 3.16" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  trans(k + i, batch_size, trans_buf);
    		  for (j = 0; j < batch_size; ++j) out[i+j] = posttrans(h(trans_buf[j]));
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, int, long) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(trans_buf, count, hash_buf),  ::std::declval<PoT>()(*hash_buf), size_t()) {
    	  // first part.
        //std::cout << "op 3.17" << std::endl;


    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) trans_buf[j] = trans(k[i+j]);
    		  h(trans_buf, batch_size, hash_buf);
    		  for (j = 0; j < batch_size; ++j) out[i+j] = posttrans(hash_buf[j]);
    	  }
    	  // no last part
    	  return max;
      }
      template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
    		  typename ::std::enable_if<
			  	  !::std::is_same<PrT, ::bliss::transform::identity<Key> >::value &&
				  !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE> >::value,
					int>::type = 1>
      inline auto batch_op(Key const * k, size_t const & count, result_type * out, long, long, long) const
      -> decltype(::std::declval<PrT>()(*k), ::std::declval<HT>()(*trans_buf), ::std::declval<PoT>()(*hash_buf), size_t()) {

        //std::cout << "op 3.18" << std::endl;

    	 return 0;
      }

  public:

      inline void operator()(Key const * k, size_t const & count, result_type* out) const {
    	  size_t max = count - (count & (batch_size - 1) );
    	  max = this->batch_op(k, max, out, 0, 0, 0);  // 0 has type int....
    	  // last part
        //std::cout << "op 3" << std::endl;

    	  for (size_t i = max; i < count; ++i) {
    		  out[i] = this->operator()(k[i]);
    	  }
      }


      template<typename V>
      inline void operator()(::std::pair<Key, V> const * x, size_t const & count, result_type * out) const {
        //std::cout << "op 4.1" << std::endl;


    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) key_buf[j] = x[i+j].first;
    		  this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
    	  }
    	  // last part
    	  for (; i < count; ++i) {
    		  out[i] = this->operator()(x[i].first);
    	  }
      }

      template<typename V>
      inline void operator()(::std::pair<const Key, V> const * x, size_t const & count, result_type * out) const {
        //std::cout << "op 4.2" << std::endl;

    	  size_t max = count - (count & (batch_size - 1) );
    	  size_t i = 0, j = 0;
    	  for (; i < max; i += batch_size ) {
    		  for (j = 0; j < batch_size; ++j) key_buf[j] = x[i+j].first;
    		  this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
    	  }
    	  // last part
    	  for (; i < count; ++i) {
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
