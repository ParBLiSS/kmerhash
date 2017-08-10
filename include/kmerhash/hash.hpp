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

// includ the murmurhash code.
#ifndef _MURMURHASH3_H_
#include <smhasher/MurmurHash3.cpp>
#endif

// and farm hash
#ifndef FARM_HASH_H_
#include <farmhash/src/farmhash.cc>
#endif

namespace fsc {

	namespace hash
	{


      /**
       * @brief  returns the least significant 64 bits directly as identity hash.
       * @note   since the number of buckets is not known ahead of time, can't have nbit be a type
       */
      template <typename T>
      class identity {

        public:
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
      class murmur {


        protected:
          uint32_t seed;

        public:
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
       * @brief  farm hash
       *
       * MAY NOT WORK CONSISTENTLY between prefetching on and off.
       */
      template <typename T>
      class farm {

        public:
          /// operator to compute hash.  64 bit again.
          inline uint64_t operator()(const T & key) const {
        	  return ::util::Hash(reinterpret_cast<const char*>(&key), sizeof(T));
          }
      };

    } // namespace hash
} // namespace bliss



#endif /* HASH_HPP_ */
