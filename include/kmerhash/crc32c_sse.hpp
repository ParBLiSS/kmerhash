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
#ifndef CRC32C_SSE_HPP_
#define CRC32C_SSE_HPP_

#include <type_traits> // enable_if
#include <cstring>     // memcpy
#include <stdexcept>   // logic error
// std int strings
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

//#define HASH_DEBUG

namespace fsc
{

namespace hash
{

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
  static constexpr size_t batch_size = 4;

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

//  // do 3 at the same time.  since latency of crc32 is 3 cycles.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod(T const *keys, size_t count, OT *results, uint32_t modulus) const
//  {
//    // loop over 3 keys at a time
//    size_t max = count - (count & 3);
//    size_t i = 0;
//    for (; i < max; i += 4)
//    {
//      hash4(keys + i, temp);
//
//      results[i] = temp[0] % modulus;
//      results[i + 1] = temp[1] % modulus;
//      results[i + 2] = temp[2] % modulus;
//      results[i + 3] = temp[3] % modulus;
//    }
//
//    // handle the remainder
//    for (; i < count; ++i)
//    {
//      results[i] = hash1(keys[i]) % modulus;
//    }
//  }

//  // do 3 at the same time.  since latency of crc32 is 3 cycles.
//  template <typename OT>
//  FSC_FORCE_INLINE void hash_and_mod_pow2(T const *keys, size_t count, OT *results, uint32_t modulus) const
//  {
//    assert((modulus & (modulus - 1)) == 0 && "modulus should be a power of 2.");
//
//    --modulus; // convert to mask.
//
//    // loop over 3 keys at a time
//    size_t max = count - (count & 3);
//    size_t i = 0;
//    for (; i < max; i += 4)
//    {
//      hash4(keys + i, temp);
//
//      results[i] = temp[0] & modulus;
//      results[i + 1] = temp[1] & modulus;
//      results[i + 2] = temp[2] & modulus;
//      results[i + 3] = temp[3] & modulus;
//    }
//
//    // handle the remainder
//    for (; i < count; ++i)
//    {
//      results[i] = hash1(keys[i]) & modulus;
//    }
//  }
};
template <typename T>
constexpr size_t crc32c<T>::batch_size;

#endif

} // namespace hash

} // namespace fsc

#endif /* CRC32C_SSE_HPP_ */
