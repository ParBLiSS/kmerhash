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
// std int strings
#include <iostream>    // cout

#include "utils/transform_utils.hpp"  //identity
#include "kmerhash/mem_utils.hpp"  // aligned_alloc
#include "kmerhash/math_utils.hpp" // lcm

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
#include "crc32c_sse.hpp"

#endif

#if defined(__AVX2__)
#include "murmurhash3_32_avx.hpp"
#include "murmurhash3_64_avx.hpp"
#include "murmurhash3finalizer_32_avx.hpp"
// no 64 bit finalizer because no mullo for 64 bit.


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

//#define HASH_DEBUG

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
  static constexpr size_t batch_size = 1;
  using result_type = uint64_t;
  using argument_type = T;


  identity (uint32_t const &_seed = 43) {};

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
constexpr size_t identity<T>::batch_size;

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
  static constexpr size_t batch_size = 1;
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
constexpr size_t murmur32<T>::batch_size;

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
  static constexpr size_t batch_size = 1;
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
      static constexpr size_t batch_size = 1;
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
constexpr size_t murmur<T>::batch_size;


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
  static constexpr size_t batch_size = 1;
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
constexpr size_t clhash<T>::batch_size;
#endif

template <typename T>
class farm
{

protected:
  uint64_t seed;

public:
  static constexpr size_t batch_size = 1;
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
constexpr size_t farm<T>::batch_size;

template <typename T>
class farm32
{

protected:
  uint32_t seed;

public:
  static constexpr size_t batch_size = 1;
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
constexpr size_t farm32<T>::batch_size;



/// SFINAE templated class for checking for batch_size.
/// modified from https://stackoverflow.com/questions/11927032/sfinae-check-for-static-member-using-decltype
// info from https://stackoverflow.com/questions/1005476/how-to-detect-whether-there-is-a-specific-member-variable-in-class
//   is not useful here because batch_size should be a static member variable.
// explanation at https://cpptalk.wordpress.com/2009/09/12/substitution-failure-is-not-an-error-2/
// modified from https://stackoverflow.com/questions/36709958/type-trait-check-if-reference-member-variable-is-static-or-not
template <class T>
class batch_traits
{
public:
	// if static class variable exists.  (what about if no variable of that name exists? or if it refers to a function?
    template<class U = T, class = typename std::enable_if<!std::is_member_object_pointer<decltype(&U::batch_size)>::value>::type>
        static constexpr size_t get_batch_size(int) { return U::batch_size; };

    // fallback for all others.  (function, non-existent, non-static member variables)
    template <class U = T>
        static constexpr size_t get_batch_size(...) { return 1ULL; };
};




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

protected:
  using POSTTRANS_T = PostTransform<HASH_VAL_TYPE>;

public:
  //
  static constexpr size_t pretrans_batch_size = batch_traits<PRETRANS_T>::get_batch_size(0);
  static constexpr size_t hash_batch_size = batch_traits<HASH_T>::get_batch_size(0);
  static constexpr size_t posttrans_batch_size = batch_traits<POSTTRANS_T>::get_batch_size(0);

  // lowest common multiple of the three.  default to 64byte/sizeof(HASH_VAL_TYPE) for now (cacheline multiple)
  // need to take full advantage of batch size.  for now, set to 32, as that is the largest batch size of hash tables. 
  static constexpr size_t batch_size =
		  constexpr_lcm(constexpr_lcm(pretrans_batch_size, hash_batch_size),
				  constexpr_lcm(posttrans_batch_size, 128UL / sizeof(HASH_VAL_TYPE)));
  	 // 128/sizeof(HASH_VAL_TYPE) = 16 appear to be a sweet spot for performance.  above and below are not great.
		  // 32;
		  // 64 / sizeof(HASH_VAL_TYPE);
		  //(sizeof(HASH_VAL_TYPE) == 4 ? 8 : 4);
          // HASH_T::batch_size;


  static_assert((batch_size & (batch_size - 1)) == 0, "ERROR: batch_size should be a power of 2.");

protected:

  // need some buffers
  // use local static array instead of dynamic ones so when
  // default copy construction/assignment happens,
  // we are not copying pointers that later gets freed by multiple objects.
//  mutable Key key_buf[batch_size] __attribute__((aligned(64)));
//  mutable PRETRANS_VAL_TYPE trans_buf[batch_size] __attribute__((aligned(64)));
//  mutable HASH_VAL_TYPE hash_buf[batch_size] __attribute__((aligned(64)));

  mutable Key * key_buf;
  mutable PRETRANS_VAL_TYPE * trans_buf;
  mutable HASH_VAL_TYPE * hash_buf;

public:
  // potentially runs into double free issue when the pointers are copied.
  PRETRANS_T trans;
  HASH_T h;
  POSTTRANS_T posttrans;

  TransformedHash(HASH_T const &_hash = HASH_T(),
                  PRETRANS_T const &pre_trans = PRETRANS_T(),
                  POSTTRANS_T const &post_trans = POSTTRANS_T()) : //batch_size(lcm(lcm(pretrans_batch_size, hash_batch_size), postrans_batch_size)),
			key_buf(nullptr),
			trans_buf(nullptr),
			hash_buf(nullptr),
                                                                   trans(pre_trans),
                                                                   h(_hash),
								   posttrans(post_trans){

									key_buf = ::utils::mem::aligned_alloc<Key>(batch_size, 64);
   	                                                                  memset(key_buf, 0, batch_size*sizeof(Key));
									trans_buf = ::utils::mem::aligned_alloc<PRETRANS_VAL_TYPE>(batch_size, 64);
   	                                                                  memset(trans_buf, 0, batch_size*sizeof(PRETRANS_VAL_TYPE));
									hash_buf = ::utils::mem::aligned_alloc<HASH_VAL_TYPE>(batch_size, 64);
   	                                                                  memset(hash_buf, 0, batch_size*sizeof(HASH_VAL_TYPE));
                                                                   };

  ~TransformedHash()
  {
	if (key_buf != nullptr)	::utils::mem::aligned_free(key_buf);
	if (trans_buf != nullptr) ::utils::mem::aligned_free(trans_buf);
	if (hash_buf != nullptr) ::utils::mem::aligned_free(hash_buf);
  }


  TransformedHash(TransformedHash const & other) :
	TransformedHash(other.h, other.trans, other.posttrans)
	{
  }


  TransformedHash(TransformedHash && other) :
	TransformedHash(other.h, other.trans, other.posttrans)
	{
  }


  TransformedHash & operator=(TransformedHash const & other) {
	h = other.h;
	trans = other.trans;
	posttrans = other.posttrans;
	return *this;
  }
  

  TransformedHash & operator=(TransformedHash && other) {
	h = std::move(other.h);
	trans = std::move(other.trans);
	posttrans = std::move(other.posttrans);
	return *this;
  }
  




  // conditionally defined, there should be just 1 defined methods after compiler resolves all this.
  // note that the compiler may do the same if it notices no-op....
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value && ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
#ifdef HASH_DEBUG
	  std::cout << "1" << std::flush;
#endif
    return h(k);
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
#ifdef HASH_DEBUG
	  std::cout << "2" << std::flush;
#endif

    return h(trans(k));
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
#ifdef HASH_DEBUG
	  std::cout << "3" << std::flush;
#endif

    return posttrans(h(k));
  }
  template <typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                !::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline result_type operator()(Key const &k) const
  {
#ifdef HASH_DEBUG
	  std::cout << "4" << std::flush;
#endif

    return posttrans(h(trans(k)));
  }

  template <typename V>
  inline result_type operator()(::std::pair<Key, V> const &x) const
  {
#ifdef HASH_DEBUG
	  std::cout << "5" << std::flush;
#endif
    return this->operator()(x.first);
  }
  template <typename V>
  inline result_type operator()(::std::pair<const Key, V> const &x) const
  {
#ifdef HASH_DEBUG
	  std::cout << "6" << std::flush;
#endif

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
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
				::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
				 // &&
                 //					::std::is_same<size_t,
                 //					 decltype(std::declval<HT>().operator()(
                 //							 std::declval<PRETRANS_VAL_TYPE const *>(),
                 //							 std::declval<size_t>(),
                 //							 std::declval<HASH_VAL_TYPE *>()))>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, int, int) const
      -> decltype(::std::declval<HT>()(k, count, out), size_t())
  {
#ifdef HASH_DEBUG
	  std::cout << "A" << std::flush;
#endif
	  h(k, count, out);

#ifdef HASH_DEBUG
	  std::cout << count << std::flush;
#endif
	  return count;
    // no last part
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
				 ::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, int) const
      -> decltype(::std::declval<HT>()(::std::declval<Key>()), size_t())
  {
#ifdef HASH_DEBUG
	  std::cout << "b0" << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "C" << std::flush;
#endif
    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      h(trans_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "D" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = h(trans_buf[j]);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "E" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        trans_buf[j] = trans(k[i + j]);
      h(trans_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "f0" << std::flush;
#endif

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
#ifdef HASH_DEBUG
	  std::cout << "G" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      h(k + i, batch_size, hash_buf);
      posttrans(hash_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "H" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        hash_buf[j] = h(k[i + j]);
      posttrans(hash_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "I" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      h(k + i, batch_size, hash_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(hash_buf[j]);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
    // no last part
    return max;
  }
  template <typename HT = HASH_T, typename PrT = PRETRANS_T, typename PoT = POSTTRANS_T,
            typename ::std::enable_if<
                ::std::is_same<PrT, ::bliss::transform::identity<Key>>::value &&
                    !::std::is_same<PoT, ::bliss::transform::identity<HASH_VAL_TYPE>>::value,
                int>::type = 1>
  inline auto batch_op(Key const *k, size_t const &count, result_type *out, int, long, long) const
      -> decltype(::std::declval<HT>()(*k), ::std::declval<PoT>()(*hash_buf), size_t())
  {
#ifdef HASH_DEBUG
	  std::cout << "j0" << std::flush;
#endif

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
#ifdef HASH_DEBUG
	  std::cout << "K" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      h(trans_buf, batch_size, hash_buf);
      posttrans(hash_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "L" << std::flush;
#endif

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
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "M" << std::flush;
#endif

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
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "N" << std::flush;
#endif

    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        hash_buf[j] = h(trans(k[i + j]));
      posttrans(hash_buf, batch_size, out + i);
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "O" << std::flush;
#endif

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
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "P" << std::flush;
#endif

    // first part.
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0;
    for (; i < max; i += batch_size)
    {
      trans(k + i, batch_size, trans_buf);
      for (j = 0; j < batch_size; ++j)
        out[i + j] = posttrans(h(trans_buf[j]));
    }
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "Q" << std::flush;
#endif

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
#ifdef HASH_DEBUG
    std::cout << max << std::flush;
#endif
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
#ifdef HASH_DEBUG
	  std::cout << "r0" << std::flush;
#endif

    // no batched part.
    return 0;
  }

public:
  inline void operator()(Key const *k, size_t const &count, result_type *out) const
  {
#ifdef HASH_DEBUG
	    std::cout << count << "/" << static_cast<size_t>(batch_size) << " -" << std::flush;
#endif

    size_t max = count - (count & (batch_size - 1));
    max = this->batch_op(k, max, out, 0, 0, 0); // 0 has type int....

#ifdef HASH_DEBUG
    std::cout << "." << std::flush;
#endif

    for (size_t i = max; i < count; ++i)
    {
      out[i] = this->operator()(k[i]);
    }
#ifdef HASH_DEBUG
    std::cout << ";" << std::endl;
#endif

  }

  template <typename V>
  inline void operator()(::std::pair<Key, V> const *x, size_t const &count, result_type *out) const
  {
#ifdef HASH_DEBUG
	    std::cout << count << "/" << static_cast<size_t>(batch_size) << " =" << std::flush;
#endif
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0, done;
    for (; i < max; ) // i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        key_buf[j] = x[i + j].first;
      done = this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
      if (done == 0) {
#ifdef HASH_DEBUG
    	  std::cout << "X" << std::flush;
#endif
     	  break;
      } else if (done != batch_size) {
#ifdef HASH_DEBUG
    	  std::cout << "x" << std::flush;
#endif
      }
      i += done;
    }
#ifdef HASH_DEBUG
    std::cout << "." << std::flush;
#endif
    // last part
    for (; i < count; ++i)
    {
      out[i] = this->operator()(x[i].first);
    }
#ifdef HASH_DEBUG
    std::cout << ";" << std::endl;
#endif

  }

  template <typename V>
  inline void operator()(::std::pair<const Key, V> const *x, size_t const &count, result_type *out) const
  {
#ifdef HASH_DEBUG
	    std::cout << count << "/" << static_cast<size_t>(batch_size) << " +" << std::flush;
#endif
    size_t max = count - (count & (batch_size - 1));
    size_t i = 0, j = 0, done;
    for (; i < max; ) // i += batch_size)
    {
      for (j = 0; j < batch_size; ++j)
        key_buf[j] = x[i + j].first;
      done = this->batch_op(key_buf, batch_size, out + i, 0, 0, 0);
      if (done == 0) {
#ifdef HASH_DEBUG
    	  std::cout << "X" << std::flush;
#endif
    	  break;
      } else if (done != batch_size) {
#ifdef HASH_DEBUG
    	  std::cout << "x" << std::flush;
#endif
      }
      i += done;
    }
#ifdef HASH_DEBUG
    std::cout << "." << std::flush;
#endif
    // last part
    for (; i < count; ++i)
    {
      out[i] = this->operator()(x[i].first);
    }
#ifdef HASH_DEBUG
    std::cout << ";" << std::endl;
#endif
  }
};
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform,
          template <typename> class PostTransform>
constexpr size_t TransformedHash<Key, Hash, PreTransform, PostTransform>::pretrans_batch_size;
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform,
          template <typename> class PostTransform>
constexpr size_t TransformedHash<Key, Hash, PreTransform, PostTransform>::hash_batch_size;
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform,
          template <typename> class PostTransform>
constexpr size_t TransformedHash<Key, Hash, PreTransform, PostTransform>::posttrans_batch_size;
template <typename Key, template <typename> class Hash,
          template <typename> class PreTransform,
          template <typename> class PostTransform>
constexpr size_t TransformedHash<Key, Hash, PreTransform, PostTransform>::batch_size;

// TODO:  [ ] batch mode transformed_predicate
//		[ ] batch mode transformed_comparator

} // namespace hash

} // namespace fsc

#endif /* HASH_HPP_ */
