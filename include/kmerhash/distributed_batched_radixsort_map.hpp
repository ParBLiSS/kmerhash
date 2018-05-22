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
 * @file    distributed_batched_radixsort_map.hpp
 * @ingroup index
 * @author  Tony Pan <tpan7@gatech.edu>
 * @brief   Implements the distributed_multimap, distributed map, and distributed_reduction_map
 *          data structures.
 *
 *          implementation is hash-base (O(1) lookup). later will support sort-based (load balanced).
 *
 *          for now, input and output via local vectors.
 *          (later, allow remote vectors,
 *            which can have remote ranges  (all to all to "sort" remote ranges to src proc,
 *            then src proc compute target procs for each elements in the remote ranges,
 *            communicate remote ranges to target procs.  target proc can then materialize the data.
 *            may not be efficient though if we don't have local spatial coherence..
 *          )
 *
 *          most create-find-delete operations support remote filtering via predicates.
 *          most create-find-delete oeprations support remote transformation.
 *
 *          signature of predicate is bool pred(T&).  if predicate needs to access the local map, it should be done via its constructor.
 *

 */

#ifndef DISTRIBUTED_BATCHED_RADIXSORT_MAP_HPP
#define DISTRIBUTED_BATCHED_RADIXSORT_MAP_HPP


#include "kmerhash/hashmap_radixsort.hpp"  // local storage hash table  // for multimap
#include <utility> 			  // for std::pair

//#include <sparsehash/dense_hash_map>  // not a multimap, where we need it most.
#include <functional> 		// for std::function and std::hash
#include <algorithm> 		// for sort, stable_sort, unique, is_sorted
#include <iterator>  // advance, distance
#include <sstream>  // stringstream for filea
#include <cstdint>  // for uint8, etc.
#include <ostream>  // std::flush

#include <type_traits>

#include <mxx/collective.hpp>
#include <mxx/reduction.hpp>
#include <mxx/algos.hpp> // for bucketing

#include "containers/distributed_map_base.hpp"

#include "utils/benchmark_utils.hpp"  // for timing.
#include "utils/logging.h"
#include "utils/filter_utils.hpp"

#include "common/kmer_transform.hpp"
#include "index/kmer_hash.hpp"   // needed by distributed_map_base...


#include "containers/dsc_container_utils.hpp"

#include "incremental_mxx.hpp"

#include "io_utils.hpp"

#include "mem_utils.hpp"

namespace dsc  // distributed std container
{


	// =================
	// NOTE: when using this, need to further alias so that only Key param remains.
	// =================


  /**
   * @brief  distributed radixsort map following std radixsort map's interface.
   * @details   This class is modeled after the hashmap_batched_radixsort_doubling_offsets.
   *         it has as much of the same methods of hashmap_batched_radixsort_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.  Also since we
   *         are working with 'distributed' data, batched operations are preferred.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed radixsort map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed radixsort map, or it may be done via sorting/lookup or other mapping
   *         mechanisms.  The choice may be constrained by the communication approach, e.g. global sorting  does not work well with
   *         incremental async communication
   *
   *  this class and its subclasses rely on 2 hash function for data distribution and 1 equal and 1 less comparators.  these are specified
   *  as template parameters.  the less comparator is for preprocessing queries and inserts.  keys (e.g.) kmers are transformed before they
   *  are hashed/compared.
   *    an alternative approach is to hold only canonical keys in the map.
   *    yet another alternative approach is to perform 2 queries for every key.- 2x computation but communication is spread out.
   *
   *  note: KeyTransform is applied before Hash, and Equal operators.  These operators should have NO KNOWLEDGE of any transform applied, including kmolecule to kmer mapping.
   *
   *  any operation that uses sort is not going to scale well.  this includes "hash_unique_key, hash_unique_tuple, local_reduction"...
   *  to reduce this, we can try by using a hash set instead.  http://www.vldb.org/pvldb/2/vldb09-257.pdf, http://www.vldb.org/pvldb/vol7/p85-balkesen.pdf
   *
   *  conditional version of insert/erase/find/count supports predicate that operate on INTERMEDIATE RESULTS.  input (key,key-value pair) can be pre-filtered.
   *    output (query result, e.g.) can be post filtered (and optionally reduce comm volume).
   *    intermediate results (such as counting in multimap only if T has certain value) can only be filtered at during local_operation.
   *
   *
   * key to proc assignment can be done as hash or splitters in sorted range.
   * tuples can be sotred in hash table or as sorted array.
   *   hash-hash combination works
   *  sort-sort combination works as well
   *  hash-sort combination can work.  advantage is in range query.
   *  sort-hash combination would be expensive for updating splitters
   *
   * This version is the hash-hash.
   *
   * @tparam Key
   * @tparam T
   * @tparam Container  default to batched_radixsort_map and radixsort multimap, requiring 5 template params.
   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
   */
  template<typename Key, typename T,
    template <typename, typename, template <typename> class, template <typename> class, typename> class Container,
    template <typename> class MapParams,
	typename Reducer = ::fsc::DiscardReducer,
    class Alloc = ::std::allocator< ::std::pair<const Key, T> >
  >
  class batched_radixsort_map_base : public ::dsc::map_base<Key, T, MapParams, Alloc> {

    protected:
      using Base = ::dsc::map_base<Key, T, MapParams, Alloc>;

    	template <typename K>
    	using DistHash = typename MapParams<K>::template DistFunction<K>;
    	template <typename K>
    	using DistTrans = typename MapParams<K>::template DistTransform<K>;

    	using trans_val_type = decltype(::std::declval<DistTrans<Key>>()(::std::declval<Key>()));
    	using hash_val_type = decltype(::std::declval<DistHash<Key>>()(::std::declval<trans_val_type>()));

//    	DistHash<trans_val_type> hash;

  	template <typename IN, typename OUT>
  	struct modulus {
  		static constexpr size_t batch_size = 1; // (sizeof(S) == 4 ? 8 : 4);
  		mutable bool is_pow2;
  		mutable OUT count;

  		modulus(OUT const & _count) : is_pow2((_count & (_count - 1)) == 0), count(_count - (is_pow2 ? 1 : 0)) {}

  		inline OUT operator()(IN const & x) const { return is_pow2 ? (x & count) : (x % count); }

  		// template <typename IN, typename OUT>
  		// inline void operator()(IN const * x, size_t const & _count, OUT * y) const {
  		// 	// TODO: [ ] do SSE version here
  		// 	for (size_t i = 0; i < _count; ++i)  y[i] = is_pow2 ? (x[i] & count) : (x[i] % count);
  		// }
  	};

  	using InternalHash = ::fsc::hash::TransformedHash<Key, DistHash, DistTrans, ::bliss::transform::identity>;
    using transhash_val_type = typename InternalHash::result_type;
    InternalHash key_to_hash;

    template <typename IN>
    using mod_byte = modulus<IN, uint8_t>;
    template <typename IN>
    using mod_short = modulus<IN, uint16_t>;
    template <typename IN>
    using mod_int = modulus<IN, uint32_t>;
    

    // don't use these - they produce only 64 bit hash values, which may not be correct
//      template <typename K>
//      using TransHash = typename MapParams<K>::template StoreTransFuncTemplate<K>;

    // own hyperloglog definition.  separate from the local container's.  this estimates using the transformed distribute hash.
    hyperloglog64<Key, InternalHash, 12> hll;


	template <typename K>
	using StoreHash = typename MapParams<K>::template StorageFunction<K>;
	template <typename K>
	using StoreTrans = typename MapParams<K>::template StorageTransform<K>;


    template <typename K>
    using StoreTransHash = ::fsc::hash::TransformedHash<K, StoreHash, StoreTrans, ::bliss::transform::identity>;
    template <typename K>
    using StoreTransEqual = typename MapParams<K>::template StoreTransEqualTemplate<K>;

    public:
    	// NOTE: if there is a hyperloglog estimator in local container, it is usign the transformed storage hash.
      using local_container_type = Container<Key, T,
    		  StoreTransHash,
    		  StoreTransEqual, Reducer>;

      // std::batched_radixsort_multimap public members.
      using key_type              = typename local_container_type::key_type;
      using mapped_type           = typename local_container_type::mapped_type;
      using value_type            = typename local_container_type::value_type;
      using hasher                = typename local_container_type::hasher;
      using key_equal             = typename local_container_type::key_equal;
//      using allocator_type        = typename local_container_type::allocator_type;
      using iterator              = typename local_container_type::iterator;
      using const_iterator        = typename local_container_type::const_iterator;
      using size_type             = typename local_container_type::size_type;
      using difference_type       = typename local_container_type::difference_type;

      // TODO: should get from the local_container_type...  radixsort does not have a single element count function yet.
      using count_result_type     = uint8_t;

    protected:
      local_container_type c;

      // CASES FOR PERMUTE:
      // appropriate when the input needs to be permuted (count, exists) to match results.
      // appropriate when the input does not need to be permuted (insert, find, erase, update), when no output to match up, or output embeds the keys.

      //==== transform_and_bucket algo that allocates a i2o array and compute hash first, then do permute using the array:
      //     would not perform great because the i2o array is walked 3x,
      //     1x during write for hashing, 1x for counting inside hashed_permute,
      //	 and 1x when actually permuting inside hashed permute.
      //     this is should be done only when hash values are needed somewhere else
      //	 otherwise, i2o can be saved from permute.

      //==== transform_and_bucket algo that transform input via transform iterator
      //	 would not perform as well because the input is transformed 1x during  i2o
      //	 and count, and again during copy for the actual permute.



      /// permute, given the assignment array and bucket counts.
      /// this is the second pass only.
      template <uint8_t prefetch_dist = 8, typename IT, typename MT, typename OT,
      typename ::std::enable_if<::std::is_same<typename ::std::iterator_traits<OT>::iterator_category,
                                               ::std::random_access_iterator_tag >::value, int>::type = 1  >
      void
      permute_by_bucketid(IT _begin, IT _end, MT bucketIds,
			 std::vector<size_t> & bucket_sizes,
			 OT results) const {

      	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
      			typename std::iterator_traits<OT>::value_type>::value,
  				"ERROR: IT and OT should be iterators with same value types");
      	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
  				"ERROR: MT should be an iterator of integral type value");


    	  size_t num_buckets = bucket_sizes.size();

          // no bucket.
          if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");

          if (_begin == _end) {
        	  return;  // no data in question.
          }

          if (num_buckets == 1) {
        	  // copy input to output and be done.
        	  ::std::copy(_begin, _end, results);

        	  return;
          }

          std::vector<size_t> bucket_offsets;
          bucket_offsets.resize(num_buckets, 0);

          // compute exclusive offsets first.
          size_t sum = 0;
          bucket_offsets[0] = 0;
          size_t i = 1;
          for (; i < num_buckets; ++i) {
            sum += bucket_sizes[i-1];
            bucket_offsets[i] = sum;
          }

          size_t input_size = std::distance(_begin, _end);
          // [2nd pass]: saving elements into correct position, and save the final position.
          // not prefetching the bucket offsets - should be small enough to fit in cache.

          // ===========================
          // direct prefetch does not do well because bucketIds has bucket assignments and not offsets.
          // therefore bucket offset is not pointing to the right location yet.
          // instead, use stream write?


          // next prefetch results
          size_t offsets[prefetch_dist];

          MT i2o_it = bucketIds;
          MT i2o_eit = bucketIds;
          std::advance(i2o_eit, ::std::min(input_size, static_cast<size_t>(prefetch_dist)));
          i = 0;
          size_t bid;
          for (; i2o_it != i2o_eit; ++i2o_it, ++i) {
            bid = bucket_offsets[*i2o_it]++;
  //          std::cout << "prefetching = " << static_cast<size_t>(i) << " at " << bid << std::endl;
            offsets[i] = bid;
            KHMXX_PREFETCH((&(*(results + bid))), _MM_HINT_T0);
          }


          // now start doing the work from prefetch_dist to end.
          constexpr size_t mask = prefetch_dist - 1;
          IT it = _begin;
          i = 0;
          i2o_eit = bucketIds + input_size;
          for (; i2o_it != i2o_eit; ++it, ++i2o_it) {
            *(results + offsets[i]) = *it;   // offset decremented by 1 before use.
                // bucekted filled from back to front for each bucket, hence iterators are pre-decremented.

            bid = bucket_offsets[*i2o_it]++;
            offsets[i] = bid;
            KHMXX_PREFETCH((&(*(results + bid))), _MM_HINT_T0);

            i = (i+1) & mask;
          }

          // and finally, finish the last part.
          for (; it != _end; ++it) {
            *(results + offsets[i]) = *it;   // offset decremented by 1 before use.
                // bucekted filled from back to front for each bucket, hence iterators are pre-decremented.
            i = (i+1) & mask;
          }


      }  // permute.




      // Does the first pass of bucketing, ie, assign and first pass in permute.
      // hash, assign to key, save the assignment, count per bucket on each rank,
      //   and update the HLL estimator.   no need for prefetcher.
      //
      // IT: input type
      // Func: key to procesor assignment type
      // ASSIGNMENT_TYPe: assignment data type, determined by the number of assignments.
      // HLL: hyperloglog type.
      // outputs: bucket sizes
      //		  permuted output
      // 		  hyperloglog
      // TODO: [ ] hyperloglog64 with 32 bit hash values....
      template <typename IT, typename ASSIGN_TYPE, typename OT, typename HLL >
      void
      assign_count_estimate_permute(IT _begin, IT _end,
                             ASSIGN_TYPE const num_buckets,
                             std::vector<size_t> & bucket_sizes,
                             OT output,
							 HLL & hll) const {

        static_assert(::std::is_integral<ASSIGN_TYPE>::value,
        		"ASSIGN_TYPE should be integral, preferably unsigned");

        // no bucket.
        if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");

        bucket_sizes.clear();

//        BL_BENCH_INIT(permute_est);
        size_t input_size = std::distance(_begin, _end);

        if (_begin == _end) {

        	return;  // no data in question.
        }

        // do a few cachelines at a time.  probably a good compromise is to do batch_size number of cachelines
        // 64 / sizeof(ASSIGN_TYPE)...
        constexpr size_t block_size = (64 / sizeof(ASSIGN_TYPE)) * InternalHash::batch_size;

//        BL_BENCH_START(permute_est);

        // initialize number of elements per bucket
        bucket_sizes.resize(num_buckets, 0);
//        BL_BENCH_END(permute_est, "alloc_count", num_buckets);


        // single bucket.  still need to estimate.
        if (num_buckets == 1) {
//        	BL_BENCH_START(permute_est);

          // set output buckets sizes
          bucket_sizes[0] = input_size;

          // and compute the hll.
    	  IT it = _begin;
          if (InternalHash::batch_size > 1) {
        	  transhash_val_type hashvals[block_size];

        	  // do blocks
        	  size_t i = 0, j;
        	  size_t max = (input_size / block_size) * block_size;
        	  for (; i < max; i += block_size) {
        		  //this->key_to_rank.hash(&(*it), block_size, hashvals);
        		  this->key_to_hash(&(*it), block_size, hashvals);

        		  for (j = 0; j < block_size; ++j, ++it, ++output) {
        			  hll.update_via_hashval(hashvals[j]);

        			  *output = *it;
        		  }
        	  }

        	  // finish remainder.
        	  size_t rem = input_size - i;
        	  // do remainder.
    		  this->key_to_hash(&(*it), rem, hashvals);
    		  //this->key_to_rank.hash(&(*it), rem, hashvals);

    		  for (j = 0; j < rem; ++j, ++it, ++output) {
    			  hll.update_via_hashval(hashvals[j]);

    			  *output = *it;
    		  }

          } else {  // not batched
        	  transhash_val_type hval;
        	  for (; it != _end; ++it, ++output) {
        		  //this->key_to_rank.hash(*it, hval);
        		  hval = this->key_to_hash(*it);
        		  hll.update_via_hashval(hval);

        		  *output = *it;
        	  }
          }
//          BL_BENCH_END(permute_est, "est_permute", input_size);
//
//          BL_BENCH_REPORT_NAMED(permute_est, "count_permute");

          return;
        }

        bool is_pow2 = (num_buckets & (num_buckets  - 1)) == 0;

//        BL_BENCH_START(permute_est);

        ASSIGN_TYPE* bucketIds = ::utils::mem::aligned_alloc<ASSIGN_TYPE>(input_size + InternalHash::batch_size);
//        BL_BENCH_END(permute_est, "alloc", input_size);


//        BL_BENCH_START(permute_est);

          // 1st pass of 2 pass algo.

          // [1st pass]: compute bucket counts and input2bucket assignment.
          // store input2bucket assignment in bucketIds temporarily.
          ASSIGN_TYPE* i2o_it = bucketIds;
          IT it = _begin;
          size_t i = 0, j, rem;
          ASSIGN_TYPE rank;

          // and compute the hll.
          if (InternalHash::batch_size > 1) {
        	  transhash_val_type hashvals[block_size];

        	  size_t max = (input_size / block_size) * block_size;

        	  if (is_pow2) {
                  ASSIGN_TYPE bucket_mask = num_buckets - 1;
				  for (; i < max; i += block_size, it += block_size) {
					  this->key_to_hash(&(*it), block_size, hashvals);

//					  if ((i == 0) && (this->comm.rank() == 0)) {
//						  std::cout << "modp counts " << num_buckets << std::endl;
//						  for (j = 0; j < block_size; ++j) {
//							  std::cout << hashvals[j] << ", ";
//						  }
//						  std::cout << std::endl;
//					  }
					  for (j = 0; j < block_size; ++j) {
						  hll.update_via_hashval(hashvals[j]);

						  rank = hashvals[j] & bucket_mask; // really (p-1)
						  *i2o_it = rank;
						  ++i2o_it;

						  ++bucket_sizes[rank];
					  }
				  }
	        	  // finish remainder.
				  rem = input_size - i;

				  this->key_to_hash(&(*it), rem, hashvals);

				  for (j = 0; j < rem; ++j) {
					  hll.update_via_hashval(hashvals[j]);

					  rank = hashvals[j] & bucket_mask;  // really (p-1)
					  *i2o_it = rank;
					  ++i2o_it;

					  ++bucket_sizes[rank];
				  }

        	  } else {
				  for (; i < max; i += block_size, it += block_size) {
					  this->key_to_hash(&(*it), block_size, hashvals);

					  for (j = 0; j < block_size; ++j) {
						  hll.update_via_hashval(hashvals[j]);

						  rank = hashvals[j] % num_buckets;
						  *i2o_it = rank;
						  ++i2o_it;

						  ++bucket_sizes[rank];
					  }
				  }
	        	  // finish remainder.
				  rem = input_size - i;

				  this->key_to_hash(&(*it), rem, hashvals);

				  for (j = 0; j < rem; ++j) {
					  hll.update_via_hashval(hashvals[j]);

					  rank = hashvals[j] % num_buckets;
					  *i2o_it = rank;
					  ++i2o_it;

					  ++bucket_sizes[rank];
				  }
        	  } // pow2_p?

          } else {  // batch size of 1.
        	  transhash_val_type h;

        	  if (is_pow2) {
                  ASSIGN_TYPE bucket_mask = num_buckets - 1;
				  for (; it != _end; ++it, ++i2o_it) {
					  h = this->key_to_hash(*it);
					  hll.update_via_hashval(h);

					  rank = h & bucket_mask;
					  *i2o_it = rank;

					  ++bucket_sizes[rank];
				  }
        	  } else {
				  for (; it != _end; ++it, ++i2o_it) {
					  h = this->key_to_hash(*it);
					  hll.update_via_hashval(h);

					  rank = h % num_buckets;
					  *i2o_it = rank;

					  ++bucket_sizes[rank];
				  }
        	  } // pow2_p?
		  } // batching?
//          BL_BENCH_END(permute_est, "est_count", input_size);


//          size_t est = hll.estimate();
//          std::cout << "estimate : " << est << std::endl;

//          BL_BENCH_START(permute_est);
          // pass 2, do the actual permute
          permute_by_bucketid(_begin, _end, bucketIds, bucket_sizes, output );
//          BL_BENCH_END(permute_est, "permute", input_size);

//          BL_BENCH_START(permute_est);
          ::utils::mem::aligned_free(bucketIds);
//          BL_BENCH_END(permute_est, "free", input_size);

//          BL_BENCH_REPORT_NAMED(permute_est, "count_permute");


      }  // end of assign_count_estimate_permute


      /// hash, count and return assignment array and bucket counts.
      /// same as first pass of permute.
      template <typename IT, typename ASSIGN_TYPE, typename OT>
      void
      assign_count_permute(IT _begin, IT _end,
    		  ASSIGN_TYPE const num_buckets,
                             std::vector<size_t> & bucket_sizes,
							 OT output) const {


        // no bucket.
        if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");

        bucket_sizes.clear();

        if (_begin == _end) return;  // no data in question.

//        BL_BENCH_INIT(permute_est);

        using InternalHashMod = 
            typename ::std::conditional<::std::is_same<uint8_t, ASSIGN_TYPE>::value,
                ::fsc::hash::TransformedHash<Key, DistHash, DistTrans, mod_byte>,
                typename ::std::conditional<::std::is_same<uint16_t, ASSIGN_TYPE>::value,
                ::fsc::hash::TransformedHash<Key, DistHash, DistTrans, mod_short>,
                ::fsc::hash::TransformedHash<Key, DistHash, DistTrans, mod_int> >::type>::type;
        InternalHashMod key_to_rank2(DistHash<trans_val_type>(9876543), DistTrans<Key>(), modulus<transhash_val_type, ASSIGN_TYPE>(num_buckets));

//        		decltype(declval<decltype(declval<KeyToRank>().proc_trans_hash)>().h)::batch_size;
        // do a few cachelines at a time.  probably a good compromise is to do batch_size number of cachelines
        // 64 / sizeof(ASSIGN_TYPE)...
        constexpr size_t block_size = (64 / sizeof(ASSIGN_TYPE)) * InternalHash::batch_size;

//        BL_BENCH_START(permute_est);
        // initialize number of elements per bucket
        bucket_sizes.resize(num_buckets, 0);
//        BL_BENCH_END(permute_est, "alloc_count", num_buckets);

        size_t input_size = std::distance(_begin, _end);

        // single bucket.  still need to estimate.
        if (num_buckets == 1) {
//            BL_BENCH_START(permute_est);
          // set output buckets sizes
          bucket_sizes[0] = input_size;

          // set all of bucketIds to 0
          std::copy(_begin, _end, output);
//          BL_BENCH_END(permute_est, "permute", input_size);

//          BL_BENCH_REPORT_NAMED(permute_est, "count_permute");

          return;
        }

//        BL_BENCH_START(permute_est);

        ASSIGN_TYPE* bucketIds = ::utils::mem::aligned_alloc<ASSIGN_TYPE>(input_size + InternalHash::batch_size);
//        BL_BENCH_END(permute_est, "alloc", input_size);


          // 1st pass of 2 pass algo.

//        BL_BENCH_START(permute_est);
          // [1st pass]: compute bucket counts and input2bucket assignment.
          // store input2bucket assignment in bucketIds temporarily.
          ASSIGN_TYPE* i2o_it = bucketIds;
          IT it = _begin;
          size_t i = 0, rem;
          ASSIGN_TYPE rank;
    	  ASSIGN_TYPE* i2o_eit;

          // and compute the hll.
          if (InternalHash::batch_size > 1) {
        	  size_t max = (input_size / block_size) * block_size;

			  for (; i < max; i += block_size, it += block_size) {
				  key_to_rank2(&(*it), block_size, i2o_it);


				  for (i2o_eit = i2o_it + block_size; i2o_it != i2o_eit; ++i2o_it) {
					  ++bucket_sizes[*i2o_it];
				  }
			  }
			  // finish remainder.
			  rem = input_size - i;

			  key_to_rank2(&(*it), rem, i2o_it);

			  for (i2o_eit = i2o_it + rem; i2o_it != i2o_eit; ++i2o_it) {
				  ++bucket_sizes[*i2o_it];
			  }

          } else {
			  for (; it != _end; ++it, ++i2o_it) {
				  rank = key_to_rank2(*it);
				  *i2o_it = rank;

				  ++bucket_sizes[rank];
			  }
		  } // batching?
//          BL_BENCH_END(permute_est, "count", input_size);

//          BL_BENCH_START(permute_est);
          permute_by_bucketid(_begin, _end, bucketIds, bucket_sizes, output);
//          BL_BENCH_END(permute_est, "permiute", input_size);

//          BL_BENCH_START(permute_est);
          ::utils::mem::aligned_free(bucketIds);
//          BL_BENCH_END(permute_est, "free", input_size);

//          BL_BENCH_REPORT_NAMED(permute_est, "count_permute");

      }  // end of assign_estimate_count





    public:

      batched_radixsort_map_base(const mxx::comm& _comm) : Base(_comm),
		  key_to_hash(DistHash<trans_val_type>(9876543), DistTrans<Key>(), ::bliss::transform::identity<hash_val_type>())
		  //hll(ceilLog2(_comm.size()))  // top level hll. no need to ignore bits.
    //	don't bother initializing c.
    {
 //   	  this->c.set_ignored_msb(ceilLog2(_comm.size()));   // NOTE THAT THIS SHOULD MATCH KEY_TO_RANK use of bits in hash table.
      }



      virtual ~batched_radixsort_map_base() {};



      /// returns the local storage.  please use sparingly.
      local_container_type& get_local_container() { return c; }
      local_container_type const & get_local_container() const { return c; }

      // ================ local overrides

      /// clears the batched_radixsort_map
      virtual void local_reset() noexcept {
    	  std::cout << "WARNING: this function is not implemented." << std::endl;
      }

      virtual void local_clear() noexcept {
    	  std::cout << "WARNING: this function is not implemented." << std::endl;
      }

      virtual void local_reserve( size_t b ) {
    	  this->c.reserve(b);
      }

      virtual void local_rehash( size_t b ) {
    	  this->c.resize(b);
      }


      // note that for each method, there is a local version of the operartion.
      // this is for use by the asynchronous version of communicator as callback for any messages received.
      /// check if empty.
      virtual bool local_empty() const {
        return this->c.size() == 0;
      }

      /// get number of entries in local container
      virtual size_t local_size() const {
        return this->c.size();
      }
      virtual size_t local_capacity() const {
    	  return this->c.capacity();
      }
      /// get number of entries in local container
      virtual size_t local_unique_size() const {
        return this->local_size();
      }

      const_iterator cbegin() const {
        return c.cbegin();
      }

      const_iterator cend() const {
        return c.cend();
      }

      using Base::size;

      /// convert the map to a vector
      virtual void to_vector(std::vector<std::pair<Key, T> > & result) const {
        this->c.to_vector().swap(result);
      }
      /// extract the unique keys of a map.
      virtual void keys(std::vector<Key> & result) const {
        this->c.keys().swap(result);
      }



    protected:

  /**
   * @brief insert new elements in the distributed batched_radixsort_multimap.
   * @param input  vector.  will be permuted.
   */
  template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_1(std::vector<std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(insert);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
      return 0;
    }

    // transform once.  bucketing and distribute will read it multiple times.
    BL_BENCH_COLLECTIVE_START(insert, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
    this->transform_input(input);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
    BL_BENCH_END(insert, "transform_input", input.size());


#ifdef DUMP_DISTRIBUTED_INPUT
      // target is reading by benchmark_hashtables, so want whole tuple, and is here only.

      std::stringstream ss;
      ss << "serialized." << this->comm.rank();
      serialize_vector(input, ss.str());
#endif

      using HVT = decltype(::std::declval<hasher>()(::std::declval<key_type>()));
      HVT* hvals;
if (estimate) {
      BL_BENCH_COLLECTIVE_START(insert, "estimate", this->comm);
      // local hash computation and hll update.
      hvals = ::utils::mem::aligned_alloc<HVT>(input.size() + local_container_type::PFD + InternalHash::batch_size);  // 64 byte alignment.
      memset(hvals + input.size(), 0, (local_container_type::PFD + InternalHash::batch_size) * sizeof(HVT) );
      this->c.get_hll().update(input.data(), input.size(), hvals);

      size_t est = this->c.get_hll().estimate();
      if (this->comm.rank() == 0)
      std::cout << "rank " << this->comm.rank() << " estimated size " << est << " capacity " << this->c.capacity() << std::endl;

      BL_BENCH_END(insert, "estimate", est);


      BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);

      if (est > this->c.capacity())
          // add 10% just to be safe.
          this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->c.get_hll().est_error_rate + 0.1)));
      // if (this->comm.rank() == 0)
      //	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
      BL_BENCH_END(insert, "alloc_hashtable", est);
}


      size_t before = this->c.size();
      BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
	if (estimate)
	      this->c.insert(input.data(), hvals, input.size());
	else
      this->c.insert(input.data(), input.size());
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
      BL_BENCH_END(insert, "insert", this->c.size());

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_1", this->comm);
    if (estimate) ::utils::mem::aligned_free(hvals);

    return this->c.size() - before;
  }

  template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_p(std::vector<std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(insert);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
      return 0;
    }

    // alloc buffer
    // transform  input->buffer
    // hash, count, estimate, permute -> hll, count, permuted input.  buffer linear read, i2o linear r/w, rand r/w hll, count, and output
    //             permute with input, i2o, count.  linear read input, i2o.  rand access count and out
    // reduce/a2a - linear
    // free buffer
    // estimate total and received each
    // alloc buffer = recv total
    // a2a
    // insert.  linear read
    // free buffer

    //std::cout << "dist insert: rank " << this->comm.rank() << " insert " << input.size() << std::endl;

        BL_BENCH_COLLECTIVE_START(insert, "alloc", this->comm);
      // get mapping to proc
      // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif

	  	  	int comm_size = this->comm.size();

        std::pair<Key, T>* buffer = ::utils::mem::aligned_alloc<std::pair<Key, T>>(input.size() + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "alloc", input.size());


	        // transform once.  bucketing and distribute will read it multiple times.
	        BL_BENCH_COLLECTIVE_START(insert, "transform", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
    this->transform_input(input.begin(), input.end(), buffer);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
	        BL_BENCH_END(insert, "transform", input.size());


// NOTE: overlap comm is incrementally inserting so we estimate before transmission, thus global estimate, and 64 bit
//            hash is needed.
//       non-overlap comm can estimate just before insertion for more accurate estimate based on actual input received.
//          so 32 bit hashes are sufficient.

    // count and estimate and save the bucket ids.
    BL_BENCH_COLLECTIVE_START(insert, "permute_estimate", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
	// allocate an HLL
	// allocate the bucket sizes array
std::vector<size_t> send_counts(comm_size, 0);

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
	if (estimate) {
	  if (comm_size <= std::numeric_limits<uint8_t>::max())
		  this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
	    			input.data(), this->hll );
	  else if (comm_size <= std::numeric_limits<uint16_t>::max())
    	  this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
    			  input.data(), this->hll );
	  else    // mpi supports only 31 bit worth of ranks.
	    	this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
	    			input.data(), this->hll );
	} else {
#endif
        if (comm_size <= std::numeric_limits<uint8_t>::max())
            this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
                      input.data() );
        else if (comm_size <= std::numeric_limits<uint16_t>::max())
            this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
                    input.data() );
        else    // mpi supports only 31 bit worth of ranks.
              this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
                      input.data() );

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
	}
#endif

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
	::utils::mem::aligned_free(buffer);

    BL_BENCH_END(insert, "permute_estimate", input.size());
    

	BL_BENCH_COLLECTIVE_START(insert, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_resume();
#endif
// merge and do estimate.
std::vector<size_t> recv_counts(this->comm.size());
mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "a2a_count", recv_counts.size());

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
    if (estimate) {
	  	        BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);
	  	        if (this->comm.rank() == 0) std::cout << "local estimated size " << this->hll.estimate() << std::endl;
	  			size_t est = this->hll.estimate_average_per_rank(this->comm);
	  			if (this->comm.rank() == 0)
	  				std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;
	  			if (est > this->c.capacity())
	  				// add 10% just to be safe.
	  				this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->hll.est_error_rate + 0.1)));
				// if (this->comm.rank() == 0)
				//	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
				BL_BENCH_END(insert, "alloc_hashtable", est);
    }
#endif

            size_t before = this->c.size();

#if defined(OVERLAPPED_COMM)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, std::pair<Key, T>* b, std::pair<Key, T>* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert", this->c.size());
#elif defined(OVERLAPPED_COMM_BATCH)

        BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

        ::khmxx::incremental::ialltoallv_and_modify_batch(input.data(), input.data() + input.size(), send_counts,
                                                    [this](int rank, std::pair<Key, T>* b, std::pair<Key, T>* e){
                                                       this->c.insert(b, std::distance(b, e));
                                                    },
                                                    this->comm);

        BL_BENCH_END(insert, "a2av_insert", this->c.size());

#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, std::pair<Key, T>* b, std::pair<Key, T>* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_2phase", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_2phase(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, std::pair<Key, T>* b, std::pair<Key, T>* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert_2phase", this->c.size());



#else

	  	  	  BL_BENCH_START(insert);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif
	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

	          std::pair<Key, T>* distributed = ::utils::mem::aligned_alloc<std::pair<Key, T>>(recv_total + InternalHash::batch_size);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "alloc_output", recv_total);



	  	  	  BL_BENCH_COLLECTIVE_START(insert, "a2a", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_resume();
#endif

#ifdef ENABLE_LZ4_COMM
	  	  	  ::khmxx::lz4::distribute_permuted(input.data(), input.data() + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
		  ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
		  BL_BENCH_END(insert, "a2a", input.size());

#ifdef DUMP_DISTRIBUTED_INPUT
    	// target is reading by benchmark_hashtables, so want whole tuple, and is here only.

		std::stringstream ss;
		ss << "serialized." << this->comm.rank();
		serialize(distributed, distributed + recv_total, ss.str());
#endif

		using HVT = decltype(::std::declval<hasher>()(::std::declval<key_type>()));
		HVT* hvals;
if (estimate) {
// hash and estimate first.

BL_BENCH_COLLECTIVE_START(insert, "estimate", this->comm);
// local hash computation and hll update.
hvals = ::utils::mem::aligned_alloc<HVT>(recv_total + local_container_type::PFD + InternalHash::batch_size);  // 64 byte alignment.
memset(hvals + recv_total, 0, (local_container_type::PFD + InternalHash::batch_size) * sizeof(HVT) );
this->c.get_hll().update(distributed, recv_total, hvals);

size_t est = this->c.get_hll().estimate();
if (this->comm.rank() == 0)
std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;

BL_BENCH_END(insert, "estimate", est);


BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);

if (est > this->c.capacity())
    // add 10% just to be safe.
    this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->c.get_hll().est_error_rate + 0.1)));
// if (this->comm.rank() == 0)
//	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
BL_BENCH_END(insert, "alloc_hashtable", est);
}




    BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.
    // TODO: predicated version.

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
// USE VERSION THAT DOES NOT RECOMPUTE HVALS.
	if (estimate)
		this->c.insert(distributed, hvals, recv_total);
	else
		this->c.insert(distributed, recv_total);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
BL_BENCH_END(insert, "insert", this->c.size());



BL_BENCH_START(insert);
if (estimate) ::utils::mem::aligned_free(hvals);
	::utils::mem::aligned_free(distributed);
    BL_BENCH_END(insert, "clean up", recv_total);

#endif // non overlap

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

    return this->c.size() - before;
  }


    public:
      /**
       * @brief insert new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

          BL_BENCH_INIT(insert);

          BL_BENCH_START(insert);
    	  size_t result;
    	  if (this->comm.size() == 1) {
    		  result = this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  result = this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
          BL_BENCH_END(insert, "insert", 0);


          BL_BENCH_START(insert);
          this->c.finalize_insert();
          BL_BENCH_END(insert, "finalize insert", 0);

          BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);

    	  // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  return result;
      }

      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_no_finalize(std::vector<std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

          BL_BENCH_INIT(insert);

          BL_BENCH_START(insert);
    	  size_t result;
    	  if (this->comm.size() == 1) {
    		  result = this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  result = this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
          BL_BENCH_END(insert, "insert", 0);

          BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_no_finalize", this->comm);

    	  // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  return result;
      }

      void finalize_insert() {

        BL_BENCH_INIT(insert);

        BL_BENCH_START(insert);
        this->c.finalize_insert();
        BL_BENCH_END(insert, "finalize insert", 0);

        BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:finalize_insert", this->comm);
      }



    protected:

      /**
       * @brief count new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t count_1(std::vector<Key >& input,
    		  count_result_type* results,
			  bool sorted_input = false, Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(count);


        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(count, "base_batched_robinhood_map:count", this->comm);
          return 0;
        }

        // transform once.  bucketing and distribute will read it multiple times.
        BL_BENCH_COLLECTIVE_START(count, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
        BL_BENCH_END(count, "transform_input", input.size());



          // local count. memory utilization a potential problem.

          BL_BENCH_COLLECTIVE_START(count, "local_count", this->comm);
          size_t found = 0;
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif
            found = this->c.count(input.data(), input.size(), results);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif
          }
          BL_BENCH_END(count, "local_count", found);

        BL_BENCH_REPORT_MPI_NAMED(count, "base_hashmap:count", this->comm);

        return found;
      }


      /**
       * @brief count new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t count_p(std::vector<Key >& input,
    		  count_result_type * results,
    		  bool sorted_input = false,
          Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(count);


        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(count, "hashmap:count", this->comm);
          return 0;
        }

        // alloc buffer
        // transform  input->buffer
        // hash, count, estimate, permute -> hll, count, permuted input.  buffer linear read, i2o linear r/w, rand r/w hll, count, and output
        //             permute with input, i2o, count.  linear read input, i2o.  rand access count and out
        // reduce/a2a - linear
        // free buffer
        // alloc buffer = recv total
        // alloc results = input
        // a2a
        // count
        // return results
        // free buffer



            BL_BENCH_COLLECTIVE_START(count, "alloc", this->comm);
          // get mapping to proc
          // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif

            int comm_size = this->comm.size();

            Key* buffer = ::utils::mem::aligned_alloc<Key>(input.size() + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
              BL_BENCH_END(count, "alloc", input.size());


            // transform once.  bucketing and distribute will read it multiple times.
            BL_BENCH_COLLECTIVE_START(count, "transform", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input.begin(), input.end(), buffer);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
            BL_BENCH_END(count, "transform", input.size());



        // count and estimate and save the bucket ids.
        BL_BENCH_COLLECTIVE_START(count, "permute", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
      // allocate an HLL
      // allocate the bucket sizes array
    std::vector<size_t> send_counts(comm_size, 0);

      if (comm_size <= std::numeric_limits<uint8_t>::max())
        this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
              input.data() );
      else if (comm_size <= std::numeric_limits<uint16_t>::max())
          this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
              input.data() );
      else    // mpi supports only 31 bit worth of ranks.
          this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
              input.data());

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
      ::utils::mem::aligned_free(buffer);

        BL_BENCH_END(count, "permute", input.size());



    BL_BENCH_COLLECTIVE_START(count, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
  // merge and do estimate.
  std::vector<size_t> recv_counts(this->comm.size());
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);


#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
              BL_BENCH_END(count, "a2a_count", recv_counts.size());





#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH)


          BL_BENCH_COLLECTIVE_START(count, "a2av_count", this->comm);

          ::khmxx::incremental::ialltoallv_and_query_one_to_one(
              input.data(), input.data() + input.size(), send_counts,
                                                      [this, &pred](int rank, Key* b, Key* e, count_result_type * out){
                                                         this->c.count(b, std::distance(b, e), out);
                                                      },
                            results,
                                                      this->comm);

          BL_BENCH_END(count, "a2av_count", this->c.size());

#elif defined(OVERLAPPED_COMM_FULLBUFFER)


      BL_BENCH_COLLECTIVE_START(count, "a2av_count_fullbuf", this->comm);

      ::khmxx::incremental::ialltoallv_and_query_one_to_one_fullbuffer(
          input.data(), input.data() + input.size(), send_counts,
                                                  [this, &pred](int rank, Key* b, Key* e, count_result_type * out){
                                                     this->c.count(b, std::distance(b, e), out);
                                                  },
                        results,
                                                  this->comm);

      BL_BENCH_END(count, "a2av_count_fullbuf", this->c.size());


#elif defined(OVERLAPPED_COMM_2P)

  BL_BENCH_COLLECTIVE_START(count, "a2av_count_2p", this->comm);

  ::khmxx::incremental::ialltoallv_and_query_one_to_one_2phase(
      input.data(), input.data() + input.size(), send_counts,
                                              [this, &pred](int rank, Key* b, Key* e, count_result_type * out){
                                                 this->c.count(b, std::distance(b, e), out);
                                              },
                    results,
                                              this->comm);

  BL_BENCH_END(count, "a2av_count_2p", this->c.size());



#else

              BL_BENCH_START(count);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
              size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

              Key* distributed = ::utils::mem::aligned_alloc<Key>(recv_total + InternalHash::batch_size);
              count_result_type* dist_results = ::utils::mem::aligned_alloc<count_result_type>(recv_total + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
              BL_BENCH_END(count, "alloc_output", recv_total);



              BL_BENCH_COLLECTIVE_START(count, "a2av", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif

#ifdef ENABLE_LZ4_COMM
  	  	  	  ::khmxx::lz4::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
			  ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
			  BL_BENCH_END(count, "a2av", input.size());



        BL_BENCH_COLLECTIVE_START(count, "count", this->comm);
        // local compute part.  called by the communicator.
         // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//          count = this->c.count(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.count(distributed, recv_total, dist_results);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif


    BL_BENCH_END(count, "count", this->c.size());

    BL_BENCH_START(count);
    ::utils::mem::aligned_free(distributed);
        BL_BENCH_END(count, "clean up", recv_total);

        // send back using the constructed recv count
        BL_BENCH_COLLECTIVE_START(count, "a2a2", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
    __itt_resume();
#endif


// this needs to be done.
#ifdef ENABLE_LZ4_RESULT
              ::khmxx::lz4::distribute_permuted(dist_results, dist_results + recv_total,
                  recv_counts, results, send_counts, this->comm);
#else
              ::khmxx::distribute_permuted(dist_results, dist_results + recv_total,
                  recv_counts, results, send_counts, this->comm);
#endif



#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
    __itt_pause();
#endif

    ::utils::mem::aligned_free(dist_results);

        BL_BENCH_END(count, "a2a2", input.size());


#endif // non overlap

        BL_BENCH_REPORT_MPI_NAMED(count, "hashmap:count_p", this->comm);

        return input.size();
      }


    public:

      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<count_result_type > count(::std::vector<Key>& keys, bool sorted_input = false,
                                                        Predicate const& pred = Predicate() ) const {

    	  ::std::vector<count_result_type > results(keys.size(), 0);
        if (this->comm.size() == 1) {
          count_1(keys, results.data(), sorted_input, pred);
        } else {
          count_p(keys, results.data(), sorted_input, pred);
        }
        return results;
      }

      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      size_t count(::std::vector<Key>& keys, count_result_type * results,
    		  bool sorted_input = false,
                                                        Predicate const& pred = Predicate() ) const {

    	  size_t res = 0;
        if (this->comm.size() == 1) {
          res = count_1(keys, results, sorted_input, pred);
        } else {
          res = count_p(keys, results, sorted_input, pred);
        }
        return res;
      }


    protected:

      /**
       * @brief find new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t find_1(std::vector<Key >& input,
    		  mapped_type * results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			  Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(find);

        this->c.set_novalue(nonexistent);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(find, "base_batched_radixsort_map:find", this->comm);
          return 0;
        }

        // transform once.  bucketing and distribute will read it multiple times.
        BL_BENCH_COLLECTIVE_START(find, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
        BL_BENCH_END(find, "transform_input", input.size());



          // local find. memory utilization a potential problem.

          BL_BENCH_COLLECTIVE_START(find, "local_find", this->comm);
          size_t found = 0;
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif
          	found = this->c.find(input.data(), input.size(), results);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif
          }
          BL_BENCH_END(find, "local_find", found);

        BL_BENCH_REPORT_MPI_NAMED(find, "base_hashmap:find", this->comm);

        return found;
      }


      /**
       * @brief find new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t find_p(std::vector<Key >& input,
    		  mapped_type * results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
    		  Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(find);

        this->c.set_novalue(nonexistent);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(find, "hashmap:find", this->comm);
          return 0;
        }

        // alloc buffer
        // transform  input->buffer
        // hash, count, estimate, permute -> hll, count, permuted input.  buffer linear read, i2o linear r/w, rand r/w hll, count, and output
        //             permute with input, i2o, count.  linear read input, i2o.  rand access count and out
        // reduce/a2a - linear
        // free buffer
        // alloc buffer = recv total
        // alloc results = input
        // a2a
        // find
        // return results
        // free buffer



            BL_BENCH_COLLECTIVE_START(find, "alloc", this->comm);
          // get mapping to proc
          // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif

  	  	  	int comm_size = this->comm.size();

  	  	  	Key* buffer = ::utils::mem::aligned_alloc<Key>(input.size() + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(find, "alloc", input.size());


  	        // transform once.  bucketing and distribute will read it multiple times.
  	        BL_BENCH_COLLECTIVE_START(find, "transform", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input.begin(), input.end(), buffer);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
  	        BL_BENCH_END(find, "transform", input.size());



        // count and estimate and save the bucket ids.
        BL_BENCH_COLLECTIVE_START(find, "permute", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
    	// allocate an HLL
    	// allocate the bucket sizes array
    std::vector<size_t> send_counts(comm_size, 0);

		  if (comm_size <= std::numeric_limits<uint8_t>::max())
			  this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
		    			input.data() );
		  else if (comm_size <= std::numeric_limits<uint16_t>::max())
	    	  this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
	    			  input.data() );
		  else    // mpi supports only 31 bit worth of ranks.
		    	this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
		    			input.data());

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
    	::utils::mem::aligned_free(buffer);

        BL_BENCH_END(find, "permute", input.size());



  	BL_BENCH_COLLECTIVE_START(find, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
	// merge and do estimate.
	std::vector<size_t> recv_counts(this->comm.size());
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(find, "a2a_count", recv_counts.size());





#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH)


  	      BL_BENCH_COLLECTIVE_START(find, "a2av_find", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_query_one_to_one(
  	    		  input.data(), input.data() + input.size(), send_counts,
  	                                                  [this, &pred, &nonexistent](int rank, Key* b, Key* e, mapped_type * out){
  	                                                     this->c.find(b, std::distance(b, e), out);
  	                                                  },
													  results,
  	                                                  this->comm);

  	      BL_BENCH_END(find, "a2av_find", this->c.size());

#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(find, "a2av_find_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_fullbuffer(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred, &nonexistent](int rank, Key* b, Key* e, mapped_type * out){
	                                                     this->c.find(b, std::distance(b, e), out);
	                                                  },
												  results,
	                                                  this->comm);

	      BL_BENCH_END(find, "a2av_find_fullbuf", this->c.size());


#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(find, "a2av_find_2p", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_2phase(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred, &nonexistent](int rank, Key* b, Key* e, mapped_type * out){
	                                                     this->c.find(b, std::distance(b, e), out);
	                                                  },
												  results,
	                                                  this->comm);

	      BL_BENCH_END(find, "a2av_find_2p", this->c.size());



#else

  	  	  	  BL_BENCH_START(find);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

  	          Key* distributed = ::utils::mem::aligned_alloc<Key>(recv_total + InternalHash::batch_size);
  	          mapped_type* dist_results = ::utils::mem::aligned_alloc<mapped_type>(recv_total + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(find, "alloc_output", recv_total);



  	  	  	  BL_BENCH_COLLECTIVE_START(find, "a2a", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif

#ifdef ENABLE_LZ4_COMM
  	  	  	  ::khmxx::lz4::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
			  ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
			  BL_BENCH_END(find, "a2av", input.size());




        BL_BENCH_COLLECTIVE_START(find, "find", this->comm);
        // local compute part.  called by the communicator.
        // TODO: predicated version.

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.find(distributed, recv_total, dist_results);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif


    BL_BENCH_END(find, "find", this->c.size());

    BL_BENCH_START(find);
    ::utils::mem::aligned_free(distributed);
        BL_BENCH_END(find, "clean up", recv_total);

    // local find. memory utilization a potential problem.

        // send back using the constructed recv count
        BL_BENCH_COLLECTIVE_START(find, "a2a2", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
    __itt_resume();
#endif


// this needs to be done.
#ifdef ENABLE_LZ4_RESULT
  	  	  	  ::khmxx::lz4::distribute_permuted(dist_results, dist_results + recv_total,
  	  	  			  recv_counts, results, send_counts, this->comm);
#else
  	  	  	  ::khmxx::distribute_permuted(dist_results, dist_results + recv_total,
  	  	  			  recv_counts, results, send_counts, this->comm);
#endif



#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
    __itt_pause();
#endif

		::utils::mem::aligned_free(dist_results);

        BL_BENCH_END(find, "a2a2", input.size());


#endif // non overlap

        BL_BENCH_REPORT_MPI_NAMED(find, "hashmap:find_p", this->comm);

        return input.size();
      }



    public:
      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<mapped_type > find(::std::vector<Key>& keys,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			  Predicate const& pred = Predicate() ) const {

    	  ::std::vector<mapped_type > results(keys.size(), 0);

    	  if (this->comm.size() == 1) {
    		  find_1(keys, results.data(), nonexistent, sorted_input, pred);
    	  } else {
    		  find_p(keys, results.data(), nonexistent, sorted_input, pred);
    	  }

    	  return results;
      }

      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      size_t find(::std::vector<Key>& keys, mapped_type * results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			Predicate const& pred = Predicate() ) const {

    	  size_t res = 0;
        if (this->comm.size() == 1) {
          res = find_1(keys, results, nonexistent, sorted_input, pred);
        } else {
          res = find_p(keys, results, nonexistent, sorted_input, pred);
        }
        return res;
      }


#if 0
      /**
       * @brief find elements with the specified keys in the distributed batched_radixsort_multimap.
       * @param keys  content will be changed and reordered
       * @param last
       */
      template <bool remove_duplicate = false, typename Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, T> > find_existing(::std::vector<Key>& keys, bool sorted_input = false,
                                               Predicate const& pred = Predicate()) const {
          BL_BENCH_INIT(find);

          ::std::vector<::std::pair<Key, T> > results;

          if (this->empty() || ::dsc::empty(keys, this->comm)) {
            BL_BENCH_REPORT_MPI_NAMED(find, "base_batched_radixsort_map:find", this->comm);
            return results;
          }

          BL_BENCH_COLLECTIVE_START(find, "transform_input", this->comm);
          ::fsc::back_emplace_iterator<::std::vector<::std::pair<Key, T> > > emplace_iter(results);
          // even if count is 0, still need to participate in mpi calls.  if (keys.size() == 0) return results;
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
          this->transform_input(keys);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
          BL_BENCH_END(find, "input_transform", keys.size());

          BL_BENCH_COLLECTIVE_START(find, "unique", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_UNIQUE)
      __itt_resume();
#endif
        if (remove_duplicate)
        	::fsc::unique(keys, sorted_input,
        				typename Base::StoreTransformedFunc(),
						typename Base::StoreTransformedEqual());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_UNIQUE)
      __itt_pause();
#endif
        BL_BENCH_END(find, "unique", keys.size());

              if (this->comm.size() > 1) {

                BL_BENCH_COLLECTIVE_START(find, "dist_query", this->comm);
                // distribute (communication part)
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
                std::vector<size_t> recv_counts;
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
                {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
            std::vector<Key > buffer;
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

#ifdef ENABLE_LZ4_COMM
  ::khmxx::lz4::distribute(keys, key_to_rank2, recv_counts, buffer, this->comm);
#else
  ::khmxx::distribute(keys, key_to_rank2, recv_counts, buffer, this->comm);
#endif
  keys.swap(buffer);
      //            ::dsc::distribute_unique(keys, this->key_to_rank, sorted_input, this->comm,
      //                    typename Base::StoreTransformedFunc(),
      //                    typename Base::StoreTransformedEqual()).swap(recv_counts);
                }
                BL_BENCH_END(find, "dist_query", keys.size());


            // local find. memory utilization a potential problem.
            // do for each src proc one at a time.

                BL_BENCH_COLLECTIVE_START(find, "reserve", this->comm);
            BL_BENCH_START(find);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
            results.reserve(keys.size());                   // TODO:  should estimate coverage, but at most same as number of keys.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
            BL_BENCH_END(find, "reserve", results.capacity());

            BL_BENCH_COLLECTIVE_START(find, "local_find", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
            std::vector<size_t> send_counts(this->comm.size(), 0);
            auto start = keys.begin();
            auto end = start;
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

            for (int i = 0; i < this->comm.size(); ++i) {
              ::std::advance(end, recv_counts[i]);

              // work on query from process i.

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
              send_counts[i] = this->c.find_existing(emplace_iter, start, end, pred, pred);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
              // if (this->comm.rank() == 0) BL_DEBUGF("R %d added %d results for %d queries for process %d\n", this->comm.rank(), send_counts[i], recv_counts[i], i);

              start = end;
            }
            BL_BENCH_END(find, "local_find", results.size());
            if (this->comm.rank() == 0) printf("rank %d result size %lu capacity %lu\n", this->comm.rank(), results.size(), results.capacity());


            BL_BENCH_COLLECTIVE_START(find, "a2a2", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
            // send back using the constructed recv count
            auto temp = mxx::all2allv(results, send_counts, this->comm);
            results.swap(temp);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
            BL_BENCH_END(find, "a2a2", results.size());

          } else {


            BL_BENCH_START(find);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
            results.reserve(keys.size());                   // TODO:  should estimate coverage.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
            //printf("reserving %lu\n", keys.size() * this->key_multiplicity);
            BL_BENCH_END(find, "reserve", results.capacity() );

            BL_BENCH_START(find);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
            this->c.find_existing(emplace_iter, keys.data(), keys.data() + keys.size(), pred, pred);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
            BL_BENCH_END(find, "local_find", results.size());

            if (this->comm.rank() == 0) printf("rank %d result size %lu capacity %lu\n", this->comm.rank(), results.size(), results.capacity());

          }

          BL_BENCH_REPORT_MPI_NAMED(find, "base_hashmap:find", this->comm);

          return results;

      }

#endif

    protected:

      /**
       * @brief erase new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase_1(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(erase);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(erase, "base_batched_radixsort_map:erase", this->comm);
          return 0;
        }

        size_t before = this->c.size();

        // transform once.  bucketing and distribute will read it multiple times.
        BL_BENCH_COLLECTIVE_START(erase, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
        BL_BENCH_END(erase, "transform_input", input.size());


          BL_BENCH_COLLECTIVE_START(erase, "local_erase", this->comm);
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif
          	this->c.erase(input.data(), input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif
          }
          BL_BENCH_END(erase, "local_erase", before - this->c.size());

        BL_BENCH_REPORT_MPI_NAMED(erase, "base_hashmap:erase", this->comm);

        return before - this->c.size();
      }


      /**
       * @brief erase new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase_p(std::vector<Key >& input, bool sorted_input = false,
    		  Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(erase);



        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(erase, "hashmap:erase", this->comm);
          return 0;
        }

        // alloc buffer
        // transform  input->buffer
        // hash, count, estimate, permute -> hll, count, permuted input.  buffer linear read, i2o linear r/w, rand r/w hll, count, and output
        //             permute with input, i2o, count.  linear read input, i2o.  rand access count and out
        // reduce/a2a - linear
        // free buffer
        // alloc buffer = recv total
        // alloc results = input
        // a2a
        // erase
        // return results
        // free buffer


        size_t before = this->c.size();

            BL_BENCH_COLLECTIVE_START(erase, "alloc", this->comm);
          // get mapping to proc
          // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif

  	  	  	int comm_size = this->comm.size();

  	  	  	Key* buffer = ::utils::mem::aligned_alloc<Key>(input.size() + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(erase, "alloc", input.size());


  	        // transform once.  bucketing and distribute will read it multiple times.
  	        BL_BENCH_COLLECTIVE_START(erase, "transform", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(input.begin(), input.end(), buffer);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
  	        BL_BENCH_END(erase, "transform", input.size());



        // count and estimate and save the bucket ids.
        BL_BENCH_COLLECTIVE_START(erase, "permute", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
    	// allocate an HLL
    	// allocate the bucket sizes array
    std::vector<size_t> send_counts(comm_size, 0);

		  if (comm_size <= std::numeric_limits<uint8_t>::max())
			  this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
		    			input.data() );
		  else if (comm_size <= std::numeric_limits<uint16_t>::max())
	    	  this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
	    			  input.data() );
		  else    // mpi supports only 31 bit worth of ranks.
		    	this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
		    			input.data());

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
    	::utils::mem::aligned_free(buffer);

        BL_BENCH_END(erase, "permute", input.size());



  	BL_BENCH_COLLECTIVE_START(erase, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
	// merge and do estimate.
	std::vector<size_t> recv_counts(this->comm.size());
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(erase, "a2a_count", recv_counts.size());





#if defined(OVERLAPPED_COMM)


  	      BL_BENCH_COLLECTIVE_START(erase, "a2av_erase", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify(
  	    		  input.data(), input.data() + input.size(), send_counts,
  	                                                  [this, &pred](int rank, Key* b, Key* e){
  	                                                     this->c.erase(b, ::std::distance(b, e));
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(erase, "a2av_erase", this->c.size());

#elif defined(OVERLAPPED_COMM_BATCH)


          BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_batch", this->comm);

          ::khmxx::incremental::ialltoallv_and_modify_batch(
              input.data(), input.data() + input.size(), send_counts,
                                                      [this, &pred](int rank, Key* b, Key* e){
                                                         this->c.erase(b, ::std::distance(b, e));
                                                      },
                                                      this->comm);

          BL_BENCH_END(erase, "a2av_erase_batch", this->c.size());


#elif defined(OVERLAPPED_COMM_FULLBUFFER)


  	      	      BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_fullbuf", this->comm);

  	      	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(
  	      	    		  input.data(), input.data() + input.size(), send_counts,
  	      	                                                  [this, &pred](int rank, Key* b, Key* e){
  	      	                                                     this->c.erase(b, ::std::distance(b, e));
  	      	                                                  },
  	      	                                                  this->comm);

  	      	      BL_BENCH_END(erase, "a2av_erase_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)


  	      BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_2phase", this->comm);

  	    	      ::khmxx::incremental::ialltoallv_and_modify_2phase(
  	    	    		  input.data(), input.data() + input.size(), send_counts,
  	    	                                                  [this, &pred](int rank, Key* b, Key* e){
  	    	                                                     this->c.erase(b, ::std::distance(b, e));
  	    	                                                  },
  	    	                                                  this->comm);

  	      BL_BENCH_END(erase, "a2av_erase_2phase", this->c.size());




#else

  	  	  	  BL_BENCH_START(erase);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

  	          Key* distributed = ::utils::mem::aligned_alloc<Key>(recv_total + InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(erase, "alloc_output", recv_total);



  	  	  	  BL_BENCH_COLLECTIVE_START(erase, "a2a", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif

#ifdef ENABLE_LZ4_COMM
  	  	  	  ::khmxx::lz4::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
			  ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
			  BL_BENCH_END(erase, "a2av", input.size());



        BL_BENCH_COLLECTIVE_START(erase, "erase", this->comm);
        // local compute part.  called by the communicator.
        // TODO: predicated version.

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.erase(distributed, recv_total);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif


    BL_BENCH_END(erase, "erase", this->c.size());

    BL_BENCH_START(erase);
    ::utils::mem::aligned_free(distributed);
        BL_BENCH_END(erase, "clean up", recv_total);

#endif // non overlap

        BL_BENCH_REPORT_MPI_NAMED(erase, "hashmap:erase_p", this->comm);

        return before - this->c.size();
      }


    public:

      /**
       * @brief erase elements with the specified keys in the distributed batched_radixsort_multimap.
       * @param first
       * @param last
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase_no_finish(std::vector<Key>& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
          BL_BENCH_INIT(erase);

    	  BL_BENCH_START(erase);
    	  if (this->comm.size() == 1) {
    		  return erase_1(input, sorted_input, pred);
    	  } else {
    		  return erase_p(input, sorted_input, pred);
    	  }
          BL_BENCH_END(erase, "erase", 0);

          BL_BENCH_REPORT_MPI_NAMED(erase, "hashmap:erase_no_finish", this->comm);
      }
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase(std::vector<Key>& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

          BL_BENCH_INIT(erase);

    	  BL_BENCH_START(erase);
          size_t res = 0;
    	  if (this->comm.size() == 1) {
    		  res =  erase_1(input, sorted_input, pred);
    	  } else {
    		  res =  erase_p(input, sorted_input, pred);
    	  }
          BL_BENCH_END(erase, "erase", 0);

    	  BL_BENCH_START(erase);
          this->c.finalize_erase();
          BL_BENCH_END(erase, "finalize erase", 0);

          BL_BENCH_REPORT_MPI_NAMED(erase, "hashmap:erase", this->comm);

          return res;
      }


      void finalize_erase() {

        BL_BENCH_INIT(erase);

        BL_BENCH_START(erase);
        this->c.finalize_erase();
        BL_BENCH_END(erase, "finalize erase", 0);

        BL_BENCH_REPORT_MPI_NAMED(erase, "hashmap:finalize_erase", this->comm);
      }


      // ================  overrides

  };


  /**
   * @brief  distributed radixsort map following std radixsort map's interface.
   * @details   This class is modeled after the hashmap_batched_radixsort_doubling_offsets.
   *         it has as much of the same methods of hashmap_batched_radixsort_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed radixsort map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed radixsort map, or it may be done via sorting/lookup or other mapping
   *         mechanisms.  The choice may be constrained by the communication approach, e.g. global sorting  does not work well with
   *         incremental async communication
   *
   * @tparam Key
   * @tparam T
   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
   */
  template<typename Key, typename T,
  	  template <typename> class MapParams,
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >
  >
  using batched_radixsort_map = batched_radixsort_map_base<Key, T, ::fsc::hashmap_radixsort, MapParams, ::fsc::DiscardReducer, Alloc>;


  /**
   * @brief  distributed radixsort reduction map following std radixsort map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
   * @details   This class is modeled after the hashmap_batched_radixsort_doubling_offsets, but allows a binary reduction operator to be used during insertion.
   *
   *         the reduction operator is not assumed to be associative.  The operator is called with parameters existing element, then new element to insert.
   *
   *         it has as much of the same methods of hashmap_batched_radixsort_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed radixsort map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed radixsort map, or it may be done via sorting/lookup or other mapping
   *         mechanisms.  The choice may be constrained by the communication approach, e.g. global sorting  does not work well with
   *         incremental async communication
   *
   * @tparam Key
   * @tparam T
   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
   * @tparam Reduc  default to ::std::plus<size_t>    reduction operator
   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
   */
  template<typename Key, typename T,
  	  template <typename> class MapParams,
  typename Reduc = ::std::plus<T>,
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >
  >
  using reduction_batched_radixsort_map = batched_radixsort_map_base<Key, T, ::fsc::hashmap_radixsort, MapParams, Reduc, Alloc>;




  /**
   * @brief  distributed radixsort counting map following std radixsort map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
   * @details   This class is modeled after the hashmap_batched_radixsort_doubling_offsets, but allows a binary reduction operator to be used during insertion.
   *
   *         the reduction operator is not assumed to be associative.  The operator is called with parameters existing element, then new element to insert.
   *
   *         it has as much of the same methods of hashmap_batched_radixsort_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed radixsort map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed radixsort map, or it may be done via sorting/lookup or other mapping
   *         mechanisms.  The choice may be constrained by the communication approach, e.g. global sorting  does not work well with
   *         incremental async communication
   *
   * @tparam Key
   * @tparam T
   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
   */
  template<typename Key, typename T,
  template <typename> class MapParams,
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >
  >
  class counting_batched_radixsort_map : public reduction_batched_radixsort_map<Key, T,
  	  MapParams, ::std::plus<T>, Alloc > {
      static_assert(::std::is_integral<T>::value, "count type has to be integral");

    protected:
      using Base = reduction_batched_radixsort_map<Key, T, MapParams, ::std::plus<T>, Alloc>;

    public:
      using local_container_type = typename Base::local_container_type;

      // std::batched_radixsort_multimap public members.
      using key_type              = typename local_container_type::key_type;
      using mapped_type           = typename local_container_type::mapped_type;
      using value_type            = typename local_container_type::value_type;
      using hasher                = typename local_container_type::hasher;
      using key_equal             = typename local_container_type::key_equal;
//      using allocator_type        = typename local_container_type::allocator_type;
//      using reference             = typename local_container_type::reference;
//      using const_reference       = typename local_container_type::const_reference;
//      using pointer               = typename local_container_type::pointer;
//      using const_pointer         = typename local_container_type::const_pointer;
//      using iterator              = typename local_container_type::iterator;
//      using const_iterator        = typename local_container_type::const_iterator;
      using size_type             = typename local_container_type::size_type;
      using difference_type       = typename local_container_type::difference_type;



      counting_batched_radixsort_map(const mxx::comm& _comm) : Base(_comm) {}

      virtual ~counting_batched_radixsort_map() {};

//      using Base::insert;
      using Base::count;
      using Base::find;
      using Base::erase;
      using Base::unique_size;

protected:

  /**
   * @brief insert new elements in the distributed batched_radixsort_multimap.
   * @param input  vector.  will be permuted.
   */
  template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_1(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(insert);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
      return 0;
    }

    // transform once.  bucketing and distribute will read it multiple times.
    BL_BENCH_COLLECTIVE_START(insert, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
    this->transform_input(input);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
    BL_BENCH_END(insert, "transform_input", input.size());


#ifdef DUMP_DISTRIBUTED_INPUT
      // target is reading by benchmark_hashtables, so want whole tuple, and is here only.

      std::stringstream ss;
      ss << "serialized." << this->comm.rank();
      serialize_vector(input, ss.str());
#endif

      using HVT = decltype(::std::declval<hasher>()(::std::declval<key_type>()));
      HVT* hvals;
if (estimate) {
      BL_BENCH_COLLECTIVE_START(insert, "estimate", this->comm);
      // local hash computation and hll update.
      hvals = ::utils::mem::aligned_alloc<HVT>(input.size() + local_container_type::PFD + Base::InternalHash::batch_size);  // 64 byte alignment.
      memset(hvals + input.size(), 0, (local_container_type::PFD + Base::InternalHash::batch_size) * sizeof(HVT) );
      this->c.get_hll().update(input.data(), input.size(), hvals);

      size_t est = this->c.get_hll().estimate();
      if (this->comm.rank() == 0)
      std::cout << "rank " << this->comm.rank() << " estimated size " << est << " capacity " << this->c.capacity() << std::endl;

      BL_BENCH_END(insert, "estimate", est);


      BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);

      if (est > this->c.capacity())
          // add 10% just to be safe.
          this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->c.get_hll().est_error_rate + 0.1)));
      // if (this->comm.rank() == 0)
      //	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
      BL_BENCH_END(insert, "alloc_hashtable", est);
}


      size_t before = this->c.size();
      BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
	if (estimate)
	      this->c.insert(input.data(), hvals, input.size());
	else
      this->c.insert(input.data(), input.size());
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
      BL_BENCH_END(insert, "insert", this->c.size());

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_1", this->comm);
    if (estimate) ::utils::mem::aligned_free(hvals);

    return this->c.size() - before;
  }

  template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_p(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(insert);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
      return 0;
    }

    // alloc buffer
    // transform  input->buffer
    // hash, count, estimate, permute -> hll, count, permuted input.  buffer linear read, i2o linear r/w, rand r/w hll, count, and output
    //             permute with input, i2o, count.  linear read input, i2o.  rand access count and out
    // reduce/a2a - linear
    // free buffer
    // estimate total and received each
    // alloc buffer = recv total
    // a2a
    // insert.  linear read
    // free buffer

    //std::cout << "dist insert: rank " << this->comm.rank() << " insert " << input.size() << std::endl;

        BL_BENCH_COLLECTIVE_START(insert, "alloc", this->comm);
      // get mapping to proc
      // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif

	  	  	int comm_size = this->comm.size();

        Key* buffer = ::utils::mem::aligned_alloc<Key>(input.size() + Base::InternalHash::batch_size);

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "alloc", input.size());


	        // transform once.  bucketing and distribute will read it multiple times.
	        BL_BENCH_COLLECTIVE_START(insert, "transform", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
    this->transform_input(input.begin(), input.end(), buffer);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
	        BL_BENCH_END(insert, "transform", input.size());


// NOTE: overlap comm is incrementally inserting so we estimate before transmission, thus global estimate, and 64 bit
//            hash is needed.
//       non-overlap comm can estimate just before insertion for more accurate estimate based on actual input received.
//          so 32 bit hashes are sufficient.

    // count and estimate and save the bucket ids.
    BL_BENCH_COLLECTIVE_START(insert, "permute_estimate", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
	// allocate an HLL
	// allocate the bucket sizes array
std::vector<size_t> send_counts(comm_size, 0);

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
	if (estimate) {
	  if (comm_size <= std::numeric_limits<uint8_t>::max())
		  this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
	    			input.data(), this->hll );
	  else if (comm_size <= std::numeric_limits<uint16_t>::max())
    	  this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
    			  input.data(), this->hll );
	  else    // mpi supports only 31 bit worth of ranks.
	    	this->assign_count_estimate_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
	    			input.data(), this->hll );
	} else {
#endif
        if (comm_size <= std::numeric_limits<uint8_t>::max())
            this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint8_t>(comm_size), send_counts,
                      input.data() );
        else if (comm_size <= std::numeric_limits<uint16_t>::max())
            this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
                    input.data() );
        else    // mpi supports only 31 bit worth of ranks.
              this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
                      input.data() );

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
	}
#endif

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
	::utils::mem::aligned_free(buffer);

    BL_BENCH_END(insert, "permute_estimate", input.size());
    

	BL_BENCH_COLLECTIVE_START(insert, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_resume();
#endif
// merge and do estimate.
std::vector<size_t> recv_counts(this->comm.size());
mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "a2a_count", recv_counts.size());

#if defined(OVERLAPPED_COMM) || defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
    if (estimate) {
	  	        BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);
	  	        if (this->comm.rank() == 0) std::cout << "local estimated size " << this->hll.estimate() << std::endl;
	  			size_t est = this->hll.estimate_average_per_rank(this->comm);
	  			if (this->comm.rank() == 0)
	  				std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;
	  			if (est > this->c.capacity())
	  				// add 10% just to be safe.
	  				this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->hll.est_error_rate + 0.1)));
				// if (this->comm.rank() == 0)
				//	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
				BL_BENCH_END(insert, "alloc_hashtable", est);
    }
#endif

            size_t before = this->c.size();

#if defined(OVERLAPPED_COMM)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert", this->c.size());
#elif defined(OVERLAPPED_COMM_BATCH)

        BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

        ::khmxx::incremental::ialltoallv_and_modify_batch(input.data(), input.data() + input.size(), send_counts,
                                                    [this](int rank, Key* b, Key* e){
                                                       this->c.insert(b, std::distance(b, e));
                                                    },
                                                    this->comm);

        BL_BENCH_END(insert, "a2av_insert", this->c.size());

#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_2phase", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_2phase(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert(b, std::distance(b, e));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert_2phase", this->c.size());



#else

	  	  	  BL_BENCH_START(insert);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif
	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

	          Key* distributed = ::utils::mem::aligned_alloc<Key>(recv_total + Base::InternalHash::batch_size);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "alloc_output", recv_total);



	  	  	  BL_BENCH_COLLECTIVE_START(insert, "a2a", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_resume();
#endif

#ifdef ENABLE_LZ4_COMM
	  	  	  ::khmxx::lz4::distribute_permuted(input.data(), input.data() + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
		  ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
		  BL_BENCH_END(insert, "a2a", input.size());

#ifdef DUMP_DISTRIBUTED_INPUT
    	// target is reading by benchmark_hashtables, so want whole tuple, and is here only.

		std::stringstream ss;
		ss << "serialized." << this->comm.rank();
		serialize(distributed, distributed + recv_total, ss.str());
#endif

		using HVT = decltype(::std::declval<hasher>()(::std::declval<key_type>()));
		HVT* hvals;
if (estimate) {
// hash and estimate first.

BL_BENCH_COLLECTIVE_START(insert, "estimate", this->comm);
// local hash computation and hll update.
hvals = ::utils::mem::aligned_alloc<HVT>(recv_total + local_container_type::PFD + Base::InternalHash::batch_size);  // 64 byte alignment.
memset(hvals + recv_total, 0, (local_container_type::PFD + Base::InternalHash::batch_size) * sizeof(HVT) );
this->c.get_hll().update(distributed, recv_total, hvals);

size_t est = this->c.get_hll().estimate();
if (this->comm.rank() == 0)
std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;

BL_BENCH_END(insert, "estimate", est);


BL_BENCH_COLLECTIVE_START(insert, "alloc_hashtable", this->comm);

if (est > this->c.capacity())
    // add 10% just to be safe.
    this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->c.get_hll().est_error_rate + 0.1)));
// if (this->comm.rank() == 0)
//	std::cout << "rank " << this->comm.rank() << " reserved " << this->c.capacity() << std::endl;
BL_BENCH_END(insert, "alloc_hashtable", est);
}




    BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.
    // TODO: predicated version.

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
// USE VERSION THAT DOES NOT RECOMPUTE HVALS.
	if (estimate)
		this->c.insert(distributed, hvals, recv_total);
	else
		this->c.insert(distributed, recv_total);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
BL_BENCH_END(insert, "insert", this->c.size());



BL_BENCH_START(insert);
if (estimate) ::utils::mem::aligned_free(hvals);
	::utils::mem::aligned_free(distributed);
    BL_BENCH_END(insert, "clean up", recv_total);

#endif // non overlap

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

    return this->c.size() - before;
  }



public:

    using Base::finalize_insert;
    using Base::insert;
    using Base::insert_no_finalize;
      /**
       * @brief insert new elements in the distributed batched_radixsort_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

          BL_BENCH_INIT(insert);

          BL_BENCH_START(insert);
    	  size_t result;
    	  if (this->comm.size() == 1) {
    		  result = this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  result = this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
          BL_BENCH_END(insert, "insert", 0);


          BL_BENCH_START(insert);
          this->c.finalize_insert();
          BL_BENCH_END(insert, "finalize insert", 0);

          BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);

    	  // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  return result;
      }

      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_no_finalize(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

          BL_BENCH_INIT(insert);

          BL_BENCH_START(insert);
    	  size_t result;
    	  if (this->comm.size() == 1) {
    		  result = this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  result = this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
          BL_BENCH_END(insert, "insert", 0);

          BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_no_finalize", this->comm);

    	  // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  return result;
      }


  };


} /* namespace dsc */


#endif // DISTRIBUTED_BATCHED_ROBINHOOD_MAP_HPP
