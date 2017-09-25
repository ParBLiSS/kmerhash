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
 */
#ifndef HASH_HPP_
#define HASH_HPP_

#include <type_traits>  // enable_if
#include <cstring>  // memcpy
#include <stdexcept>  //logic error
#include <stdint.h>  // std int strings



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
#include <immintrin.h>

namespace fsc {

  namespace hash
  {
    // code below assumes sse and not avx (yet)
    namespace sse
    {
      template <typename T>
      class Murmur3SSE;


      // for 32 bit buckets
      // original: body: 16 inst per iter of 4 bytes; tail: 15 instr. ; finalization:  8 instr.
      // about 4 inst per byte + 8, for each hash value.
      template <>
      class Murmur3SSE<uint32_t> {

        protected:
          // make static so initialization at beginning of class...
          const __m128i seed;
          const __m128i mix_const1;
          const __m128i mix_const2;
          const __m128i c1;
          const __m128i c2;
          const __m128i c3;
          const __m128i c4;

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
          FSC_FORCE_INLINE __m128i update32_partial( __m128i h, __m128i k, uint8_t const & count) const {
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

        public:
          Murmur3SSE(uint32_t _seed) :
            seed(_mm_set1_epi32(_seed)),
            mix_const1(_mm_set1_epi32(0x85ebca6b)),
            mix_const2(_mm_set1_epi32(0xc2b2ae35)),
            c1(_mm_set1_epi32(0xcc9e2d51)),
            c2(_mm_set1_epi32(0x1b873593)),
            c3(_mm_set1_epi32(0x5)),
            c4(_mm_set1_epi32(0xe6546b64))   // SSE2
        {}



          // useful for computing 4 32bit hashes in 1 pass (for hasing into less than 2^32 buckets)
          // assume 4 streams are available.
          // working with 4 bytes at a time because there are
          // init: 4 instr.
          // body: 13*4 + 12  per iter of 16 bytes
          // tail: about the same
          // finalize: 11 inst. for 4 elements.
          // about 5 inst per byte + 11 inst for 4 elements.
          FSC_FORCE_INLINE void hash( const void ** key, uint64_t len, uint8_t nstreams, uint32_t * out) const {
            // process 4 streams at a time.  all should be the same length.

            assert((nstreams <= 4) && "maximum number of streams is 4");
            assert((nstreams > 0) && "minimum number of streams is 1");

            __m128i k0, k1, k2, k3, t0, t1;
            __m128i h1 = seed;

            //----------
            // first do blocks of 16 bytes.

            const int nblocks = len >> 4;

            // init to zero
            switch (nstreams) {
              case 1: k1 = _mm_setzero_si128();  // SSE
              case 2: k2 = _mm_setzero_si128();  // SSE
              case 3: k3 = _mm_setzero_si128();  // SSE
              default:
                break;
            }

            for (int i = 0; i < nblocks; ++i) {
              // read streams
              switch (nstreams) {
                case 4: k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[3]) + i);  // SSE3
                case 3: k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[2]) + i);  // SSE3
                case 2: k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[1]) + i);  // SSE3
                case 1: k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[0]) + i);  // SSE3
                default:
                  break;
              }

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

            // next do the remainder if any.
            if ((len & 0xF) > 0) {

              // read more stream.  over read, and zero out.
              switch (nstreams) {
                case 4: k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[3]) + nblocks);  // SSE3
                case 3: k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[2]) + nblocks);  // SSE3
                case 2: k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[1]) + nblocks);  // SSE3
                case 1: k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[0]) + nblocks);  // SSE3
                default:
                  break;
              }

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
                  // last word needs to be padded with 0 always.
                  h1 =  update32_partial(h1, _mm_unpackhi_epi64(t0, t1), rem);  // remainder  > 0
                  break;
                case 3:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
                  // last word needs to be padded with 0.  3 rows only
                  h1 = (rem > 0) ?
                      update32_partial(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                case 2:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  // last word needs to be padded with 0.  2 rows only.  rem >= 0
                  h1 = (rem > 0) ?
                      update32_partial(h1, _mm_unpackhi_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpackhi_epi64(t0, t1));
                  break;
                case 1:
                  // last word needs to be padded with 0.  1 rows only.  remainder must be >= 0
                  h1 = (rem > 0) ?
                       update32_partial(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                       update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                default:
                  break;
              }

            }

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, _mm_set1_epi32(len));  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            // store all 4 out
            switch (nstreams) {
              case 4: _mm_storeu_si128((__m128i*)out, h1);  // sse
                break;
              case 3: out[2] = _mm_extract_epi32(h1, 2);   // SSE4.1  2 cycles.  maskmoveu takes 10 (ivybridge)
              case 2: out[1] = _mm_extract_epi32(h1, 1);
              case 1: out[0] = _mm_extract_epi32(h1, 0);
              default:
                break;;
            }


          }

          FSC_FORCE_INLINE void hash4( const void ** key, uint64_t len, uint32_t * out) const {
        	  __m128i res = hash4(key, len);
        	  _mm_storeu_si128((__m128i*)out, res);
          }

          FSC_FORCE_INLINE __m128i hash4( const void ** key, uint64_t len) const {
            // process 4 streams at a time.  all should be the same length.

            __m128i k0, k1, k2, k3, t0, t1;
            __m128i h1 = seed;

            //----------
            // first do blocks of 16 bytes.
            const int nblocks = len >> 4;

            // init to zero
            for (int i = 0; i < nblocks; ++i) {
				k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[3]) + i);  // SSE3
				k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[2]) + i);  // SSE3
				k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[1]) + i);  // SSE3
				k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[0]) + i);  // SSE3

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

            // next do the remainder if any.
            if ((len & 0xF) > 0) {

              // read more stream.  over read, and zero out.
                k3 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[3]) + nblocks);  // SSE3
                k2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[2]) + nblocks);  // SSE3
                k1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[1]) + nblocks);  // SSE3
                k0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(key[0]) + nblocks);  // SSE3

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
                  h1 =  update32_partial(h1, _mm_unpackhi_epi64(t0, t1), rem);  // remainder  > 0
  //                    update32(h1, _mm_unpackhi_epi64(t0, t1));
                  break;
                case 3:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  h1 = update32(h1, _mm_unpackhi_epi64(t0, t1)); // transpose 4x2  SSE2

                  t0 = _mm_unpackhi_epi32(k0, k1);         // transpose 2x2   SSE2
                  t1 = _mm_unpackhi_epi32(k2, k3);         // transpose 2x2   SSE2
                  // last word needs to be padded with 0.  3 rows only
                  h1 = (rem > 0) ?
                      update32_partial(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                case 2:
                  h1 = update32(h1, _mm_unpacklo_epi64(t0, t1)); // transpose 4x2  SSE2
                  // last word needs to be padded with 0.  2 rows only.  rem >= 0
                  h1 = (rem > 0) ?
                      update32_partial(h1, _mm_unpackhi_epi64(t0, t1), rem) :  // remainder  >= 0
                      update32(h1, _mm_unpackhi_epi64(t0, t1));
                  break;
                case 1:
                  // last word needs to be padded with 0.  1 rows only.  remainder must be >= 0
                  h1 = (rem > 0) ?
                       update32_partial(h1, _mm_unpacklo_epi64(t0, t1), rem) :  // remainder  >= 0
                       update32(h1, _mm_unpacklo_epi64(t0, t1));
                  break;
                default:
                  break;
              }

            }

            //----------
            // finalization
            // or the length.
            h1 = _mm_xor_si128(h1, _mm_set1_epi32(len));  // sse

            h1 = fmix32(h1);  // ***** SSE4.1 **********

            return h1;

          }


      };






    } // namespace sse





    /**
     * @brief  returns the least significant 64 bits directly as identity hash.
     * @note   since the number of buckets is not known ahead of time, can't have nbit be a type
     */
    template <typename T>
    class identity {

      public:
        static constexpr uint8_t batch_size = 1;


        /// operator to compute hash value
        inline uint64_t operator()(const T & key) const {
          if (sizeof(T) >= 8)  // more than 64 bits, so use the lower 64 bits.
            return *(reinterpret_cast<uint64_t*>(&key));
          else {
            // copy into 64 bits
            uint64_t out = 0;
            memcpy(&out, &key, sizeof(T));
            return out;
          }
        }
    };






    /**
     * @brief MurmurHash.  using lower 64 bits.
     *
     */
    template <typename T>
    class murmur3sse32 {


      protected:
        ::fsc::hash::sse::Murmur3SSE<uint32_t> hasher;
        mutable void const * kptrs[4];
        mutable uint32_t temp[4];

      public:
        static constexpr uint8_t batch_size = 4;

        murmur3sse32(uint32_t const & _seed = 43 ) : hasher(_seed) {};

        inline uint32_t operator()(const T & key) const
        {
          kptrs[0] = &key;
          uint32_t h;
          hasher.hash(kptrs, sizeof(T), 1, &h);
          return h;
        }

        // results always 32 bit.
        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint32_t * results) const {
          size_t rem = count & 0x3;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 4) {
            kptrs[0] = &(keys[i]);
            kptrs[1] = &(keys[i + 1]);
            kptrs[2] = &(keys[i + 2]);
            kptrs[3] = &(keys[i + 3]);
            hasher.hash4(kptrs, sizeof(T), results + i);
          }

          // last part.
          switch(rem) {
            case 3: kptrs[2] = &(keys[i + 2]);
            case 2: kptrs[1] = &(keys[i + 1]);
            case 1: kptrs[0] = &(keys[i]);
                    hasher.hash(kptrs, sizeof(T), rem, results + i);
            default:
              break;
          }
        }

        // assume consecutive memory layout.
        template<typename OT>
        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint32_t modulus) const {
          size_t rem = count & 0x3;
          size_t max = count - rem;
          size_t i = 0;
          for (; i < max; i += 4) {
            kptrs[0] = &(keys[i]);
            kptrs[1] = &(keys[i + 1]);
            kptrs[2] = &(keys[i + 2]);
            kptrs[3] = &(keys[i + 3]);
            hasher.hash4(kptrs, sizeof(T), temp);
            results[i] = temp[0] % modulus;
            results[i+1] = temp[1] % modulus;
            results[i+2] = temp[2] % modulus;
            results[i+3] = temp[3] % modulus;
          }

          // last part.
          switch(rem) {
            case 3: kptrs[2] = &(keys[i + 2]);
            case 2: kptrs[1] = &(keys[i + 1]);
            case 1: kptrs[0] = &(keys[i]);
              hasher.hash(kptrs, sizeof(T), rem, temp);
            default:
              break;
          }
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
            kptrs[0] = &(keys[i]);
            kptrs[1] = &(keys[i + 1]);
            kptrs[2] = &(keys[i + 2]);
            kptrs[3] = &(keys[i + 3]);
            hasher.hash4(kptrs, sizeof(T), temp);
            results[i]   = temp[0] & modulus;
            results[i+1] = temp[1] & modulus;
            results[i+2] = temp[2] & modulus;
            results[i+3] = temp[3] & modulus;
          }

          // last part.
          switch(rem) {
            case 3: kptrs[2] = &(keys[i + 2]);
            case 2: kptrs[1] = &(keys[i + 1]);
            case 1: kptrs[0] = &(keys[i]);
              hasher.hash(kptrs, sizeof(T), rem, temp);
            default:
              break;
          }
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



    /**
     * @brief crc.  32 bit hash..
     * @details  operator should require sizeof(T)/8  + 6 operations + 2 cycle latencies.
     *            require SSE4.2
     */
    template <typename T>
    class crc32c {


      protected:
        uint32_t seed;
        static constexpr size_t blocks = sizeof(T) >> 3;   // divide by 8
        static constexpr size_t rem = sizeof(T) & 0x7;  // remainder.
        static constexpr size_t offset = (sizeof(T) >> 3) << 3;

        uint32_t hash1(const T & key) const {
            uint64_t crc64 = 0;

            // block of 8 bytes
            uint64_t const * data64 = reinterpret_cast<uint64_t const *>(&key);
            for (size_t i = 0; i < blocks; ++i) {
              crc64 = _mm_crc32_u64(crc64, data64[i]);
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

            return _mm_crc32_u32(crc, seed);   // add the seed in at the end.
        }

        void hash4(T const * keys, uint32_t * results) const {
            // loop over 3 keys at a time
		  uint64_t aa = 0;
		  uint64_t bb = 0;
		  uint64_t cc = 0;
		  uint64_t dd = 0;
		  {
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

		  results[0] = _mm_crc32_u32(a, seed);
		  results[1] = _mm_crc32_u32(b, seed);
		  results[2] = _mm_crc32_u32(c, seed);
		  results[3] = _mm_crc32_u32(d, seed);
        }

      public:
        static constexpr uint8_t batch_size = 4;

        crc32c(uint32_t const & _seed = 37 ) : seed(_seed) {};

        // do 1 element.
        FSC_FORCE_INLINE uint32_t operator()(const T & key) const
        {
        	return hash1(key);
        }

        // results always 32 bit.
        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        // however, to limit the modulus, do 4 at a time.
        FSC_FORCE_INLINE void hash(T const * keys, size_t count, uint32_t * results) const {
          // loop over 3 keys at a time
          size_t max = count - 3;
          size_t j = 0;
          for (; j < max; j += 4) {
        	  hash4(keys + j, results + j);
          }

          // handle the remainder
          for (; j < count; ++j) {
            results[j] = hash1(keys[j]);
          }
        }

        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        template <typename OT>
        FSC_FORCE_INLINE void hash_and_mod(T const * keys, size_t count, OT * results, uint32_t modulus) const {
            uint32_t temp[4];
        	// loop over 3 keys at a time
            size_t max = count - 3;
            size_t j = 0;
            for (; j < max; j += 4) {
          	  hash4(keys + j, temp);

          	  results[j]     = temp[0] % modulus;
          	  results[j + 1] = temp[1] % modulus;
          	  results[j + 2] = temp[2] % modulus;
          	  results[j + 3] = temp[3] % modulus;
            }

            // handle the remainder
            for (; j < count; ++j) {
              results[j] = hash1(keys[j]) % modulus;
            }
        }

        // do 3 at the same time.  since latency of crc32 is 3 cycles.
        template <typename OT>
        FSC_FORCE_INLINE void hash_and_mod_pow2(T const * keys, size_t count, OT * results, uint32_t modulus) const {
        	assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");

            uint32_t temp[4];
            --modulus;  // convert to mask.

        	// loop over 3 keys at a time
            size_t max = count - 3;
            size_t j = 0;
            for (; j < max; j += 4) {
          	  hash4(keys + j, temp);

          	  results[j]     = temp[0] & modulus;
          	  results[j + 1] = temp[1] & modulus;
          	  results[j + 2] = temp[2] & modulus;
          	  results[j + 3] = temp[3] & modulus;
            }

            // handle the remainder
            for (; j < count; ++j) {
              results[j] = hash1(keys[j]) & modulus;
            }
        }
    };




    /**
     * @brief  farm hash
     *
     * MAY NOT WORK CONSISTENTLY between prefetching on and off.
     */
    template <typename T>
    class farm {

      public:
        static constexpr uint8_t batch_size = 1;

        farm(uint32_t const & _seed = 43 ) {};

        /// operator to compute hash.  64 bit again.
        inline uint64_t operator()(const T & key) const {
          return ::util::Hash(reinterpret_cast<const char*>(&key), sizeof(T));
        }
    };

  } // namespace hash
} // namespace bliss



#endif /* HASH_HPP_ */
