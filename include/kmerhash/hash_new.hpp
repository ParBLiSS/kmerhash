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

#if defined(__SSE4_1__)
#include "murmurhash3_32_sse.hpp"

#endif

#if defined(__AVX2__)
#include "murmurhash3_32_avx.hpp"
#include "murmurhash3_64_avx.hpp"

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

/**
     * @brief MurmurHash.  using lower 64 bits.
     *
     */
    template <typename T>
    class murmur_x86
    {
    
    protected:
      uint64_t seed;
    
    public:
      static constexpr uint8_t batch_size = 1;
      using result_type = uint64_t;
      using argument_type = T;
    
      murmur_x86(uint64_t const &_seed = 43) : seed(_seed){};
    
      inline uint64_t operator()(const T &key) const
      {
        // produces 128 bit hash.
        uint64_t h[2];
        // let compiler optimize out all except one of these.
        MurmurHash3_x86_128(&key, sizeof(T), seed, h);
    
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



// /// SFINAE templated class for checking for batch_size.  from  https://stackoverflow.com/questions/1005476/how-to-detect-whether-there-is-a-specific-member-variable-in-class
// /// explanation at https://cpptalk.wordpress.com/2009/09/12/substitution-failure-is-not-an-error-2/
// template<typename T> struct HasBatchSize { 
//     struct Fallback { uint8_t batch_size; }; // introduce member name "x"
//     struct Derived : T, Fallback { };

//     template<typename C, C> struct ChT; 

//     template<typename C> static char (&f(ChT<int Fallback::*, &C::x>*))[1]; 
//     template<typename C> static char (&f(...))[2]; 

//     static bool const value = sizeof(f<Derived>(0)) == 2;
// }; 


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
  // need to take full advantage of batch size.  for now, set to 32, as that is the largest batch size of hash tables. 
  static constexpr uint8_t batch_size = 32; // 64 / sizeof(HASH_VAL_TYPE); //(sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);
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
