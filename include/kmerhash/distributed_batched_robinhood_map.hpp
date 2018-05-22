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
 * @file    distributed_batched_robinhood_map.hpp
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

#ifndef DISTRIBUTED_BATCHED_ROBINHOOD_MAP_HPP
#define DISTRIBUTED_BATCHED_ROBINHOOD_MAP_HPP


#include "kmerhash/robinhood_offset_hashmap_ptr.hpp"  // local storage hash table  // for multimap
#include <utility> 			  // for std::pair

//#include <sparsehash/dense_hash_map>  // not a multimap, where we need it most.
#include <functional> 		// for std::function and std::hash
#include <algorithm> 		// for sort, stable_sort, unique, is_sorted
#include <iterator>  // advance, distance
#include <sstream>  // stringstream for filea.  for debugging...
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
   * @brief  distributed robinhood map following std robinhood map's interface.
   * @details   This class is modeled after the hashmap_batched_robinhood_doubling_offsets.
   *         it has as much of the same methods of hashmap_batched_robinhood_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.  Also since we
   *         are working with 'distributed' data, batched operations are preferred.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed robinhood map, or it may be done via sorting/lookup or other mapping
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
   * @tparam Container  default to batched_robinhood_map and robinhood multimap, requiring 5 template params.
   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
   */
  template<typename Key, typename T,
  template <typename, typename, template <typename> class, template <typename> class, typename...> class Container,
  template <typename> class MapParams,
	typename Reducer = ::fsc::DiscardReducer,
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >
  >
  class batched_robinhood_map_base :
		  public ::dsc::map_base<Key, T, MapParams, Alloc> {

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
    		  StoreTransEqual, Reducer,
    		  Alloc>;

      // std::batched_robinhood_multimap public members.
      using key_type              = typename local_container_type::key_type;
      using mapped_type           = typename local_container_type::mapped_type;
      using value_type            = typename local_container_type::value_type;
      using hasher                = typename local_container_type::hasher;
      using key_equal             = typename local_container_type::key_equal;
      using allocator_type        = typename local_container_type::allocator_type;
      using iterator              = typename local_container_type::iterator;
      using const_iterator        = typename local_container_type::const_iterator;
      using size_type             = typename local_container_type::size_type;
      using difference_type       = typename local_container_type::difference_type;

      using count_result_type     = decltype(::std::declval<local_container_type>().count(::std::declval<key_type>(),
                                                                                          ::std::declval<::bliss::filter::TruePredicate>(),
                                                                                           ::std::declval<::bliss::filter::TruePredicate>()));

    protected:
      local_container_type c;

      mutable bool local_changed;

      /// local reduction via a copy of local container type (i.e. batched_robinhood_map).
      /// this takes quite a bit of memory due to use of batched_robinhood_map, but is significantly faster than sorting.
      virtual void local_reduction(::std::vector<::std::pair<Key, T> >& input, bool & sorted_input) {

        if (input.size() == 0) return;

        // sort is slower.  use robinhood map.
        BL_BENCH_INIT(reduce_tuple);

        BL_BENCH_START(reduce_tuple);
        local_container_type temp;  // reserve with buckets.
        BL_BENCH_END(reduce_tuple, "reserve", input.size());

        BL_BENCH_START(reduce_tuple);
        temp.insert(input);
        BL_BENCH_END(reduce_tuple, "reduce", temp.size());

        BL_BENCH_START(reduce_tuple);
        temp.to_vector().swap(input);
        BL_BENCH_END(reduce_tuple, "copy", input.size());

        //local_container_type().swap(temp);   // doing the swap to clear helps?

        BL_BENCH_REPORT_NAMED(reduce_tuple, "reduction_hashmap:local_reduce");
      }

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


          bool is_pow2 = (num_buckets & (num_buckets - 1)) == 0;
        
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

      batched_robinhood_map_base(const mxx::comm& _comm) : Base(_comm),
		  key_to_hash(DistHash<trans_val_type>(9876543), DistTrans<Key>(), ::bliss::transform::identity<hash_val_type>())
		  //hll(ceilLog2(_comm.size()))  // top level hll. no need to ignore bits.
    //	don't bother initializing c.
    {
 //   	  this->c.set_ignored_msb(ceilLog2(_comm.size()));   // NOTE THAT THIS SHOULD MATCH KEY_TO_RANK use of bits in hash table.
      }



      virtual ~batched_robinhood_map_base() {};



      /// returns the local storage.  please use sparingly.
      local_container_type& get_local_container() { return c; }
      local_container_type const & get_local_container() const { return c; }

      // ================ local overrides

      /// clears the batched_robinhood_map
      virtual void local_reset() noexcept {
    	  this->c.clear();
    	  this->c.rehash(128);
      }

      virtual void local_clear() noexcept {
        this->c.clear();
      }

      /// reserve space.  n is the local container size.  this allows different processes to individually adjust its own size.
      virtual void local_reserve( size_t n ) {
    	  this->c.reserve(n);
      }

      virtual void local_rehash( size_t b ) {
    	  this->c.rehash(b);
      }


      // note that for each method, there is a local version of the operartion.
      // this is for use by the asynchronous version of communicator as callback for any messages received.
      /// check if empty.
      virtual bool local_empty() const {
        return this->c.size() == 0;
      }

      /// get number of entries in local container
      virtual size_t local_size() const {
//        if (this->comm.rank() == 0) printf("rank %d hashmap_base local size %lu\n", this->comm.rank(), this->c.size());

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
      using Base::unique_size;
      using Base::get_multiplicity;

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
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_1(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(insert);

        if (input.size() == 0) {
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


          size_t before = this->c.size();
          BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
        // local compute part.  called by the communicator.
          // do the version with size estimates.

        // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    if (estimate)
    	this->c.insert(input);
    else
    	this->c.insert_no_estimate(input);

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
          BL_BENCH_END(insert, "insert", this->c.size());

        BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_1", this->comm);

        return this->c.size() - before;
      }


      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_p(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
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

            ::std::pair<Key, T>* buffer = ::utils::mem::aligned_alloc<::std::pair<Key, T> >(input.size() + InternalHash::batch_size);

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
                        input.data());
          else if (comm_size <= std::numeric_limits<uint16_t>::max())
              this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint16_t>(comm_size), send_counts,
                      input.data() );
          else    // mpi supports only 31 bit worth of ranks.
                this->assign_count_permute(buffer, buffer + input.size(), static_cast<uint32_t>(comm_size), send_counts,
                        input.data());

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
  			size_t est = this->hll.estimate_average_per_rank(this->comm);
  			if (est > (this->c.get_max_load_factor() * this->c.capacity()))
  				// add 10% just to be safe.
  	        this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->hll.est_error_rate + 0.1)));
  	        if (this->comm.rank() == 0) std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;
  	        BL_BENCH_END(insert, "alloc_hashtable", est);
  	  	  	  }
#endif

  	        size_t before = this->c.size();

#if defined(OVERLAPPED_COMM)

  	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify(input.data(), input.data() + input.size(), send_counts,
  	                                                  [this](int rank, ::std::pair<Key, T>* b, ::std::pair<Key, T>* e){
  	                                                     this->c.insert_no_estimate(b, e);
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(insert, "a2av_insert", this->c.size());

#elif defined(OVERLAPPED_COMM_BATCH)

          BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

          ::khmxx::incremental::ialltoallv_and_modify_batch(input.data(), input.data() + input.size(), send_counts,
                                                      [this](int rank, ::std::pair<Key, T>* b, ::std::pair<Key, T>* e){
                                                         this->c.insert_no_estimate(b, e);
                                                      },
                                                      this->comm);

          BL_BENCH_END(insert, "a2av_insert", this->c.size());


#elif defined(OVERLAPPED_COMM_FULLBUFFER)

  	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_fullbuf", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(input.data(), input.data() + input.size(), send_counts,
  	                                                  [this](int rank, ::std::pair<Key, T>* b, ::std::pair<Key, T>* e){
  	                                                     this->c.insert_no_estimate(b, e);
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(insert, "a2av_insert_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

  	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_2pass", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify_2phase(input.data(), input.data() + input.size(), send_counts,
  	                                                  [this](int rank, ::std::pair<Key, T>* b, ::std::pair<Key, T>* e){
  	                                                     this->c.insert_no_estimate(b, e);
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(insert, "a2av_insert_2pass", this->c.size());


#else

  	  	  	  BL_BENCH_START(insert);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

  	          ::std::pair<Key, T>* distributed = ::utils::mem::aligned_alloc<::std::pair<Key, T> >(recv_total + InternalHash::batch_size);

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



        BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
        // local compute part.  called by the communicator.
        // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
// NOTE: with local cardinality estimation.
    if (estimate)
        this->c.insert(distributed, distributed + recv_total);
    else
        this->c.insert_no_estimate(distributed, distributed + recv_total);

#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(insert, "insert", this->c.size());



    BL_BENCH_START(insert);
    	::utils::mem::aligned_free(distributed);
        BL_BENCH_END(insert, "clean up", recv_total);

#endif // non overlap

        BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

        return this->c.size() - before;
      }

    public:
      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

    	  if (this->comm.size() == 1) {
    		  return this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  return this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
      }

    protected:

      /**
       * @brief count new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t count_1(std::vector<Key >& input,
    		  count_result_type * results,
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
          // do for each src proc one at a time.

          BL_BENCH_COLLECTIVE_START(count, "local_count", this->comm);
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif

          	this->c.count(results, input.data(), input.data() + input.size(), pred, pred);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif
          }
          BL_BENCH_END(count, "local_count", input.size());

        BL_BENCH_REPORT_MPI_NAMED(count, "base_hashmap:count", this->comm);

        return input.size();
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
  	                                                     this->c.count(out, b, e, pred, pred);
  	                                                  },
													  results,
  	                                                  this->comm);

  	      BL_BENCH_END(count, "a2av_count", this->c.size());

#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(count, "a2av_count_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_fullbuffer(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred](int rank, Key* b, Key* e, count_result_type * out){
	                                                     this->c.count(out, b, e, pred, pred);
	                                                  },
												  results,
	                                                  this->comm);

	      BL_BENCH_END(count, "a2av_count_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(count, "a2av_count", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_2phase(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred](int rank, Key* b, Key* e, count_result_type * out){
	                                                     this->c.count(out, b, e, pred, pred);
	                                                  },
												  results,
	                                                  this->comm);

	      BL_BENCH_END(count, "a2av_count", this->c.size());


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
//        	count = this->c.count(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.count(dist_results, distributed, distributed + recv_total, pred, pred);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif


    BL_BENCH_END(count, "count", this->c.size());

    BL_BENCH_START(count);
    ::utils::mem::aligned_free(distributed);
        BL_BENCH_END(count, "clean up", recv_total);

    // local count. memory utilization a potential problem.

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
      size_t count(::std::vector<Key>& keys,
    		  count_result_type * results,
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


//      template <typename Predicate = ::bliss::filter::TruePredicate>
//      ::std::vector<::std::pair<Key, size_type> > count(Predicate const & pred = Predicate()) const {
//
//        ::std::vector<::std::pair<Key, size_type> > results = this->c.count(pred);
//
//        if (this->comm.size() > 1) this->comm.barrier();
//        return results;
//      }

    protected:

      /**
       * @brief find new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t find_1(std::vector<Key >& input, mapped_type * results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			  Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(find);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(find, "base_batched_robinhood_map:find", this->comm);
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
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif
		this->c.find(results, input.data(), input.data() + input.size(), nonexistent, pred, pred);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif
          }
          BL_BENCH_END(find, "local_find", input.size());

        BL_BENCH_REPORT_MPI_NAMED(find, "base_hashmap:find", this->comm);

        return input.size();
      }


      /**
       * @brief find new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t find_p(std::vector<Key >& input, mapped_type* results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
    		  Predicate const & pred = Predicate()) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(find);

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
  	                                                     this->c.find(out, b, e, nonexistent, pred, pred);
  	                                                  },
													  results,
  	                                                  this->comm);

  	      BL_BENCH_END(find, "a2av_find", this->c.size());


#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(find, "a2av_find_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_fullbuffer(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred, &nonexistent](int rank, Key* b, Key* e, mapped_type * out){
	                                                     this->c.find(out, b, e, nonexistent, pred, pred);
	                                                  },
												  results,
	                                                  this->comm);

	      BL_BENCH_END(find, "a2av_find_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(find, "a2av_find_2p", this->comm);

	      ::khmxx::incremental::ialltoallv_and_query_one_to_one_2phase(
	    		  input.data(), input.data() + input.size(), send_counts,
	                                                  [this, &pred, &nonexistent](int rank, Key* b, Key* e, mapped_type * out){
	                                                     this->c.find(out, b, e, nonexistent, pred, pred);
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
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.find(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.find(dist_results, distributed, distributed + recv_total, nonexistent, pred, pred);
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


#if 0  // TODO: temporarily retired.
      /**
       * @brief find elements with the specified keys in the distributed batched_robinhood_multimap.
       * @param keys  content will be changed and reordered
       * @param last
       */
      template <bool remove_duplicate = false, typename Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, T> > find_existing(::std::vector<Key>& keys, bool sorted_input = false,
                                               Predicate const& pred = Predicate()) const {
          BL_BENCH_INIT(find);

          ::std::vector<::std::pair<Key, T> > results;

          if (this->empty() || ::dsc::empty(keys, this->comm)) {
            BL_BENCH_REPORT_MPI_NAMED(find, "base_batched_robinhood_map:find", this->comm);
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
       * @brief erase new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase_1(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(erase);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(erase, "base_batched_robinhood_map:erase", this->comm);
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
          size_t erased;
          {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_resume();
#endif
          	erased = this->c.erase(input.data(), input.data() + input.size(), pred, pred);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COUNT)
      __itt_pause();
#endif

#ifndef NDEBUG
  printf("rank %d of %d erase. before %ld after %ld\n", this->comm.rank(), this->comm.size(), before, this->c.size());
#endif

          }
          BL_BENCH_END(erase, "local_erase", erased);

        BL_BENCH_REPORT_MPI_NAMED(erase, "base_hashmap:erase", this->comm);

        return before - this->c.size();
      }


      /**
       * @brief erase new elements in the distributed batched_robinhood_multimap.
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
  	                                                     this->c.erase(b, e, pred, pred);
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(erase, "a2av_erase", this->c.size());

#elif defined(OVERLAPPED_COMM_BATCH)


          BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_batch", this->comm);

          ::khmxx::incremental::ialltoallv_and_modify_batch(
              input.data(), input.data() + input.size(), send_counts,
                                                      [this, &pred](int rank, Key* b, Key* e){
                                                         this->c.erase(b, e, pred, pred);
                                                      },
                                                      this->comm);

          BL_BENCH_END(erase, "a2av_erase_batch", this->c.size());



#elif defined(OVERLAPPED_COMM_FULLBUFFER)


  	      	      BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_fullbuf", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(
  	    		  input.data(), input.data() + input.size(), send_counts,
  	                                                  [this, &pred](int rank, Key* b, Key* e){
  	                                                     this->c.erase(b, e, pred, pred);
  	                                                  },
  	                                                  this->comm);

  	      BL_BENCH_END(erase, "a2av_erase_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)


  	      BL_BENCH_COLLECTIVE_START(erase, "a2av_erase_2phase", this->comm);

  	      ::khmxx::incremental::ialltoallv_and_modify_2phase(
  	    		  input.data(), input.data() + input.size(), send_counts,
  	                                                  [this, &pred](int rank, Key* b, Key* e){
  	                                                     this->c.erase(b, e, pred, pred);
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
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.erase(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif

    this->c.erase(distributed, distributed + recv_total, pred, pred);
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

#ifndef NDEBUG
        printf("rank %d of %d erase.  before %ld after %ld\n", this->comm.rank(), this->comm.size(), before, this->c.size());
#endif
        return before - this->c.size();
      }


    public:

      /**
       * @brief erase elements with the specified keys in the distributed batched_robinhood_multimap.
       * @param first
       * @param last
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase(std::vector<Key>& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

    	  if (this->comm.size() == 1) {
    		  return erase_1(input, sorted_input, pred);
    	  } else {
    		  return erase_p(input, sorted_input, pred);
    	  }
      }

      // ================  overrides

  };


  /**
   * @brief  distributed robinhood map following std robinhood map's interface.
   * @details   This class is modeled after the hashmap_batched_robinhood_doubling_offsets.
   *         it has as much of the same methods of hashmap_batched_robinhood_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed robinhood map, or it may be done via sorting/lookup or other mapping
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
  using batched_robinhood_map = batched_robinhood_map_base<Key, T, ::fsc::hashmap_robinhood_offsets_reduction, MapParams, ::fsc::DiscardReducer, Alloc>;


  /**
   * @brief  distributed robinhood reduction map following std robinhood map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
   * @details   This class is modeled after the hashmap_batched_robinhood_doubling_offsets, but allows a binary reduction operator to be used during insertion.
   *
   *         the reduction operator is not assumed to be associative.  The operator is called with parameters existing element, then new element to insert.
   *
   *         it has as much of the same methods of hashmap_batched_robinhood_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed robinhood map, or it may be done via sorting/lookup or other mapping
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
  using reduction_batched_robinhood_map = batched_robinhood_map_base<Key, T, ::fsc::hashmap_robinhood_offsets_reduction, MapParams, Reduc, Alloc>;




  /**
   * @brief  distributed robinhood counting map following std robinhood map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
   * @details   This class is modeled after the hashmap_batched_robinhood_doubling_offsets, but allows a binary reduction operator to be used during insertion.
   *
   *         the reduction operator is not assumed to be associative.  The operator is called with parameters existing element, then new element to insert.
   *
   *         it has as much of the same methods of hashmap_batched_robinhood_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using distributed robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local distributed robinhood map, or it may be done via sorting/lookup or other mapping
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
  class counting_batched_robinhood_map : public reduction_batched_robinhood_map<Key, T,
  	  MapParams, ::std::plus<T>, Alloc > {
      static_assert(::std::is_integral<T>::value, "count type has to be integral");

    protected:
      using Base = reduction_batched_robinhood_map<Key, T, MapParams, ::std::plus<T>, Alloc>;

    public:
      using local_container_type = typename Base::local_container_type;

      // std::batched_robinhood_multimap public members.
      using key_type              = typename local_container_type::key_type;
      using mapped_type           = typename local_container_type::mapped_type;
      using value_type            = typename local_container_type::value_type;
      using hasher                = typename local_container_type::hasher;
      using key_equal             = typename local_container_type::key_equal;
      using allocator_type        = typename local_container_type::allocator_type;
      using reference             = typename local_container_type::reference;
      using const_reference       = typename local_container_type::const_reference;
      using pointer               = typename local_container_type::pointer;
      using const_pointer         = typename local_container_type::const_pointer;
      using iterator              = typename local_container_type::iterator;
      using const_iterator        = typename local_container_type::const_iterator;
      using size_type             = typename local_container_type::size_type;
      using difference_type       = typename local_container_type::difference_type;



      counting_batched_robinhood_map(const mxx::comm& _comm) : Base(_comm) {}

      virtual ~counting_batched_robinhood_map() {};

      using Base::insert;
      using Base::count;
      using Base::find;
      using Base::erase;
      using Base::unique_size;

protected:

  /**
   * @brief insert new elements in the distributed batched_robinhood_multimap.
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


      size_t before = this->c.size();
      BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.

    // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
if (estimate)
      this->c.insert(input, T(1));
else
	  this->c.insert_no_estimate(input, T(1));

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
      BL_BENCH_END(insert, "insert", this->c.size());

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_1", this->comm);

#ifndef NDEBUG
    printf("rank %d of %d insert %ld, recv %ld. before %ld after %ld\n", this->comm.rank(), this->comm.size(), input.size(), input.size(), before, this->c.size());
#endif
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
	  	        size_t est = this->hll.estimate_average_per_rank(this->comm);
	  			if (est > (this->c.get_max_load_factor() * this->c.capacity()))
	  				// add 10% just to be safe.
	  	        	this->c.reserve(static_cast<size_t>(static_cast<double>(est) * (1.0 + this->hll.est_error_rate + 0.1)));
	  	        if (this->comm.rank() == 0) std::cout << "rank " << this->comm.rank() << " estimated size " << est << std::endl;
	  	        BL_BENCH_END(insert, "alloc_hashtable", est);
	  	  	  }
#endif
	        size_t before = this->c.size();

#if defined(OVERLAPPED_COMM)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert_no_estimate(b, e, T(1));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert", this->c.size());

#elif defined(OVERLAPPED_COMM_BATCH)

        BL_BENCH_COLLECTIVE_START(insert, "a2av_insert", this->comm);

        ::khmxx::incremental::ialltoallv_and_modify_batch(input.data(), input.data() + input.size(), send_counts,
                                                    [this](int rank, Key* b, Key* e){
                                                       this->c.insert_no_estimate(b, e, T(1));
                                                    },
                                                    this->comm);

        BL_BENCH_END(insert, "a2av_insert", this->c.size());


#elif defined(OVERLAPPED_COMM_FULLBUFFER)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_fullbuf", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_fullbuffer(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert_no_estimate(b, e, T(1));
	                                                  },
	                                                  this->comm);

	      BL_BENCH_END(insert, "a2av_insert_fullbuf", this->c.size());

#elif defined(OVERLAPPED_COMM_2P)

	      BL_BENCH_COLLECTIVE_START(insert, "a2av_insert_2phase", this->comm);

	      ::khmxx::incremental::ialltoallv_and_modify_2phase(input.data(), input.data() + input.size(), send_counts,
	                                                  [this](int rank, Key* b, Key* e){
	                                                     this->c.insert_no_estimate(b, e, T(1));
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


    BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);
    // local compute part.  called by the communicator.
    // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif

if (estimate)
// NOTE: local cardinality estimation.
	this->c.insert(distributed, distributed + recv_total, T(1));
else
	this->c.insert_no_estimate(distributed, distributed + recv_total, T(1));
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
BL_BENCH_END(insert, "insert", this->c.size());



BL_BENCH_START(insert);
	::utils::mem::aligned_free(distributed);
    BL_BENCH_END(insert, "clean up", recv_total);

#ifndef NDEBUG
    printf("rank %d of %d insert %ld, recv %ld. before %ld after %ld\n", this->comm.rank(), this->comm.size(), input.size(), recv_total, before, this->c.size());
#endif


#endif // non overlap

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

    return this->c.size() - before;
  }



public:

      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  if (this->comm.size() == 1) {
    		  return this->template insert_1<estimate>(input, sorted_input, pred);
    	  } else {
    		  return this->template insert_p<estimate>(input, sorted_input, pred);
    	  }
      }

  };







} /* namespace dsc */


#endif // DISTRIBUTED_BATCHED_ROBINHOOD_MAP_HPP


//
//  /**
//   * @brief  distributed robinhood multimap following std robinhood multimap's interface.
//   * @details   This class is modeled after the std::batched_robinhood_multimap.
//   *         it does not have all the methods of std::batched_robinhood_multimap.  Whatever methods that are present considers the fact
//   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
//   *
//   *         Iterators are assumed to be local rather than distributed, so methods that returns iterators are not provided.
//   *         as an alternative, vectors are returned.
//   *         methods that accept iterators as input assume that the input data is local.
//   *
//   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
//   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
//   *
//   *         This allows the possibility of using distributed robinhood map as local storage for coarser grain distributed container.
//   *
//   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
//   *         may be done using a hash, similar to the local distributed robinhood map, or it may be done via sorting/lookup or other mapping
//   *         mechanisms.  The choice may be constrained by the communication approach, e.g. global sorting  does not work well with
//   *         incremental async communication
//   *
//   * @tparam Key
//   * @tparam T
//   * @tparam Comm   default to mpi_collective_communicator       communicator for global communication. may hash or sort.
//   * @tparam KeyTransform   transform function for the key.  can supply identity.  requires a single template argument (Key).  useful for mapping kmolecule to kmer.
//   * @tparam Hash   hash function for local and distribution.  requires a template arugment (Key), and a bool (prefix, chooses the MSBs of hash instead of LSBs)
//   * @tparam Equal   default to ::std::equal_to<Key>   equal function for the local storage.
//   * @tparam Alloc  default to ::std::allocator< ::std::pair<const Key, T> >    allocator for local storage.
//   */
//  template<typename Key, typename T,
//  template <typename> class MapParams,
//  class Alloc = ::std::allocator< ::std::pair<const Key, T> >
//  >
//  class batched_robinhood_multimap : public batched_robinhood_map_base<Key, T, ::std::batched_robinhood_multimap, MapParams, Alloc> {
//    protected:
//      using Base = batched_robinhood_map_base<Key, T, ::std::batched_robinhood_multimap, MapParams, Alloc>;
//
//
//    public:
//      using local_container_type = typename Base::local_container_type;
//
//      // std::batched_robinhood_multimap public members.
//      using key_type              = typename local_container_type::key_type;
//      using mapped_type           = typename local_container_type::mapped_type;
//      using value_type            = typename local_container_type::value_type;
//      using hasher                = typename local_container_type::hasher;
//      using key_equal             = typename local_container_type::key_equal;
//      using allocator_type        = typename local_container_type::allocator_type;
//      using reference             = typename local_container_type::reference;
//      using const_reference       = typename local_container_type::const_reference;
//      using pointer               = typename local_container_type::pointer;
//      using const_pointer         = typename local_container_type::const_pointer;
//      using iterator              = typename local_container_type::iterator;
//      using const_iterator        = typename local_container_type::const_iterator;
//      using size_type             = typename local_container_type::size_type;
//      using difference_type       = typename local_container_type::difference_type;
//
//    protected:
//
//      struct LocalFind {
//        // unfiltered.
//        template<class DB, typename Query, class OutputIter>
//        size_t operator()(DB &db, Query const &v, OutputIter &output) const {
//            auto range = db.equal_range(v);
//
//            // range's iterators are not random access iterators, so insert calling distance uses ++, slowing down the process.
//            // manually insert improves performance here.
//            size_t count = 0;
//            for (auto it2 = range.first; it2 != range.second; ++it2) {
//              *output = *it2;
//              ++output;
//              ++count;
//            }
//            return count;
//        }
//        // filtered element-wise.
//        template<class DB, typename Query, class OutputIter, class Predicate = ::bliss::filter::TruePredicate>
//        size_t operator()(DB &db, Query const &v, OutputIter &output,
//                          Predicate const& pred) const {
//            auto range = db.equal_range(v);
//
//            // add the output entry.
//            size_t count = 0;
//            if (pred(range.first, range.second)) {
//              for (auto it2 = range.first; it2 != range.second; ++it2) {
//                if (pred(*it2)) {
//                  *output = *it2;
//                  ++output;
//                  ++count;
//                }
//              }
//            }
//            return count;
//        }
//        // no filter by range AND elemenet for now.
//      } find_element;
//
//      mutable size_t local_unique_count;
//
//    public:
//
//
//      batched_robinhood_multimap(const mxx::comm& _comm) : Base(_comm), local_unique_count(0) {}
//
//      virtual ~batched_robinhood_multimap() {}
//
//      using Base::count;
//      using Base::erase;
//      using Base::unique_size;
//
//
//
//  /**
//   * @brief find elements with the specified keys in the distributed batched_robinhood_multimap.
//   *
//   * this version is more relevant for multimap.  was called find_overlap.
//   * why this version that uses isend and irecv?  because all2all version requires all result data to be in memory.
//   * this one can do it one source process at a time.
//   *
//   * @param keys    content will be changed and reordered.
//   * @param last
//   */
//  template <bool remove_duplicate = false, typename Predicate = ::bliss::filter::TruePredicate>
//  ::std::vector<::std::pair<Key, T> > find_overlap(::std::vector<Key>& keys, bool sorted_input = false, Predicate const& pred = Predicate()) const {
//      BL_BENCH_INIT(find);
//
//      ::std::vector<::std::pair<Key, T> > results;
//
//      if (this->empty() || ::dsc::empty(keys, this->comm)) {
//        BL_BENCH_REPORT_MPI_NAMED(find, "base_batched_robinhood_map:find_overlap", this->comm);
//        return results;
//      }
//
//
//      BL_BENCH_START(find);
//      // even if count is 0, still need to participate in mpi calls.  if (keys.size() == 0) return results;
//
//      this->transform_input(keys);
//      BL_BENCH_END(find, "transform_input", keys.size());
//
//  BL_BENCH_START(find);
//  if (remove_duplicate)
//	  ::fsc::unique(keys, sorted_input,
//          typename Base::StoreTransformedFunc(),
//          typename Base::StoreTransformedEqual());
//  BL_BENCH_END(find, "unique", keys.size());
//
//        if (this->comm.size() > 1) {
//
//          BL_BENCH_COLLECTIVE_START(find, "dist_query", this->comm);
//          // distribute (communication part)
//          std::vector<size_t> recv_counts;
//          {
//      std::vector<Key > buffer;
//      ::khmxx::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
//      keys.swap(buffer);
////            ::dsc::distribute_unique(keys, this->key_to_rank, sorted_input, this->comm,
////                    typename Base::StoreTransformedFunc(),
////                    typename Base::StoreTransformedEqual()).swap(recv_counts);
//          }
//          BL_BENCH_END(find, "dist_query", keys.size());
//
//
//        //======= local count to determine amount of memory to allocate at destination.
//        BL_BENCH_START(find);
//        ::std::vector< uint8_t > count_results;
//        size_t max_key_count = *(::std::max_element(recv_counts.begin(), recv_counts.end()));
//        count_results.reserve(max_key_count);
//        ::fsc::back_emplace_iterator<::std::vector< uint8_t > > count_emplace_iter(count_results);
//
//        std::vector<size_t> send_counts(this->comm.size(), 0);
//
//        auto start = keys.begin();
//        auto end = start;
//        size_t total = 0;
//        for (int i = 0; i < this->comm.size(); ++i) {
//          ::std::advance(end, recv_counts[i]);
//
//          // count results for process i
//          count_results.clear();
//          this->c.count(count_emplace_iter, start, end, pred, pred);
//
//          send_counts[i] =
//              ::std::accumulate(count_results.begin(), count_results.end(), static_cast<size_t>(0));
//          total += send_counts[i];
//          start = end;
//          //printf("Rank %d local count for src rank %d:  recv %d send %d\n", this->comm.rank(), i, recv_counts[i], send_counts[i]);
//        }
//        ::std::vector<uint8_t >().swap(count_results);
//        BL_BENCH_END(find, "local_count", total);
//
//
//        BL_BENCH_COLLECTIVE_START(find, "a2a_count", this->comm);
//        std::vector<size_t> resp_counts = mxx::all2all(send_counts, this->comm);  // compute counts of response to receive
//        BL_BENCH_END(find, "a2a_count", keys.size());
//
//
//        //==== reserve
//        BL_BENCH_START(find);
//        auto resp_displs = mxx::impl::get_displacements(resp_counts);  // compute response displacements.
//
//        auto resp_total = resp_displs[this->comm.size() - 1] + resp_counts[this->comm.size() - 1];
//        auto max_send_count = *(::std::max_element(send_counts.begin(), send_counts.end()));
//        results.resize(resp_total);   // allocate, not just reserve
//        ::std::vector<::std::pair<Key, T> > local_results(2 * max_send_count);
//        size_t local_offset = 0;
//        auto local_results_iter = local_results.begin();
//
//        //printf("reserving %lu\n", keys.size() * this->key_multiplicity);
//        BL_BENCH_END(find, "reserve", resp_total);
//
//        //=== process queries and send results.  O(p) iterations
//        BL_BENCH_START(find);
//        auto recv_displs = mxx::impl::get_displacements(recv_counts);  // compute response displacements.
//        int recv_from, send_to;
//        size_t found;
//        total = 0;
//        std::vector<MPI_Request> recv_reqs(this->comm.size());
//        std::vector<MPI_Request> send_reqs(this->comm.size());
//
//
//        mxx::datatype dt = mxx::get_datatype<::std::pair<Key, T> >();
//
//        for (int i = 0; i < this->comm.size(); ++i) {
//
//          recv_from = (this->comm.rank() + (this->comm.size() - i)) % this->comm.size(); // rank to recv data from
//          // set up receive.
//          MPI_Irecv(&results[resp_displs[recv_from]], resp_counts[recv_from], dt.type(),
//                    recv_from, i, this->comm, &recv_reqs[i]);
//
//        }
//
//        for (int i = 0; i < this->comm.size(); ++i) {
//          send_to = (this->comm.rank() + i) % this->comm.size();    // rank to send data to
//
//          local_offset = (i % 2) * max_send_count;
//          local_results_iter = local_results.begin() + local_offset;
//
//          //== get data for the dest rank
//          start = keys.begin();                                   // keys for the query for the dest rank
//          ::std::advance(start, recv_displs[send_to]);
//          end = start;
//          ::std::advance(end, recv_counts[send_to]);
//
//          found = this->c.find(local_results_iter, start, end, pred, pred);
//          // if (this->comm.rank() == 0) BL_DEBUGF("R %d added %d results for %d queries for process %d\n", this->comm.rank(), send_counts[i], recv_counts[i], i);
//          total += found;
//          //== now send the results immediately - minimizing data usage so we need to wait for both send and recv to complete right now.
//
//
//          MPI_Isend(&(local_results[local_offset]), found, dt.type(), send_to,
//                    i, this->comm, &send_reqs[i]);
//
//          // wait for previous requests to complete.
//          if (i > 0) MPI_Wait(&send_reqs[(i - 1)], MPI_STATUS_IGNORE);
//
//          //printf("Rank %d local find send to %d:  query %d result sent %d (%d).  recv from %d received %d\n", this->comm.rank(), send_to, recv_counts[send_to], found, send_counts[send_to], recv_from, resp_counts[recv_from]);
//        }
//        // last pair
//        MPI_Wait(&send_reqs[(this->comm.size() - 1)], MPI_STATUS_IGNORE);
//
//        // wait for all the receives
//        MPI_Waitall(this->comm.size(), &(recv_reqs[0]), MPI_STATUSES_IGNORE);
//
//
//        //printf("Rank %d total find %lu\n", this->comm.rank(), total);
//        BL_BENCH_END(find, "find_send", results.size());
//
//      } else {
//
//        BL_BENCH_START(find);
//        // keep unique keys
//        if (remove_duplicate)
//        	::fsc::unique(keys, sorted_input,
//        			typename Base::StoreTransformedFunc(),
//					typename Base::StoreTransformedEqual());
//        BL_BENCH_END(find, "uniq1", keys.size());
//
//        // memory is constrained.  find EXACT count.
//        BL_BENCH_START(find);
//        ::std::vector<uint8_t> count_results;
//        count_results.reserve(keys.size());
//        ::fsc::back_emplace_iterator<::std::vector<uint8_t > > count_emplace_iter(count_results);
//        ::fsc::back_emplace_iterator<::std::vector<::std::pair<Key, T> > > emplace_iter(results);
//
//        // count now.
//        this->c.count(count_emplace_iter, keys.begin(), keys.end(), pred, pred);
//        size_t count = ::std::accumulate(count_results.begin(), count_results.end(), static_cast<size_t>(0));
//        BL_BENCH_END(find, "local_count", count);
//
//        BL_BENCH_START(find);
//        results.reserve(count);                   // TODO:  should estimate coverage.
//        //printf("reserving %lu\n", keys.size() * this->key_multiplicity);
//        BL_BENCH_END(find, "reserve", results.capacity());
//
//        BL_BENCH_START(find);
//        this->c.find(emplace_iter, keys.begin(), keys.end(), pred, pred);
//        BL_BENCH_END(find, "local_find", results.size());
//      }
//
//      BL_BENCH_REPORT_MPI_NAMED(find, "base_hashmap:find_overlap", this->comm);
//
//      return results;
//
//  }

//
//      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
//      ::std::vector<::std::pair<Key, T> > find(::std::vector<Key>& keys, bool sorted_input = false,
//                                               Predicate const& pred = Predicate()) const {
//          return Base::find_overlap<remove_duplicate>(find_element, keys, sorted_input, pred);
//      }
////      template <class Predicate = ::bliss::filter::TruePredicate>
////      ::std::vector<::std::pair<Key, T> > find(::std::vector<Key>& keys, bool sorted_input = false,
////                                                          Predicate const& pred = Predicate()) const {
////          return Base::find(find_element, keys, sorted_input, pred);
////      }
////      template <class Predicate = ::bliss::filter::TruePredicate>
////      ::std::vector<::std::pair<Key, T> > find_collective(::std::vector<Key>& keys, bool sorted_input = false,
////                                                          Predicate const& pred = Predicate()) const {
////          return Base::find_a2a(find_element, keys, sorted_input, pred);
////      }
////      template <class Predicate = ::bliss::filter::TruePredicate>
////      ::std::vector<::std::pair<Key, T> > find_sendrecv(::std::vector<Key>& keys, bool sorted_input = false,
////                                                          Predicate const& pred = Predicate()) const {
////          return Base::find_sendrecv(find_element, keys, sorted_input, pred);
////      }
//
//
//
//      template <class Predicate = ::bliss::filter::TruePredicate>
//      ::std::vector<::std::pair<Key, T> > find(Predicate const& pred = Predicate()) const {
//          return Base::find(find_element, pred);
//      }
//      /// access the current the multiplicity.  only multimap needs to override this.
//      virtual float get_multiplicity() const {
//        // multimaps would add a collective function to change the multiplicity
//        if (this->comm.rank() == 0) printf("rank %d batched_robinhood_multimap get_multiplicity called\n", this->comm.rank());
//
//
//        // one approach is to add up the number of repeats for the key of each entry, then divide by total count.
//        //  sum(count per key) / c.size.
//        // problem with this approach is that for robinhood map, to get the count for a key is essentially O(count), so we get quadratic time.
//        // The approach is VERY SLOW for large repeat count.  - (0.0078125 human: 52 sec, synth: FOREVER.)
//
//        // a second approach is to count the number of unique key then divide the map size by that.
//        //  c.size / #unique.  requires unique set
//        // To find unique set, we take each bucket, copy to vector, sort it, and then count unique.
//        // This is precise, and is faster than the approach above.  (0.0078125 human: 54 sec.  synth: 57sec.)
//        // but the n log(n) sort still grows with the duplicate count
//
//        size_t n_unique = this->local_unique_size();
//        float multiplicity = 1.0f;
//        if (n_unique > 0) {
//          // local unique
//          multiplicity =
//              static_cast<float>(this->local_size()) /
//              static_cast<float>(n_unique);
//        }
//
//
//        //        ::std::vector< ::std::pair<Key, T> > temp;
//        //        KeyTransform<Key> trans;
//        //        for (int i = 0, max = this->c.capacity(); i < max; ++i) {
//        //          if (this->c.bucket_size(i) == 0) continue;  // empty bucket. move on.
//        //
//        //          // copy and sort.
//        //          temp.assign(this->c.begin(i), this->c.end(i));  // copy the bucket
//        //          // sort the bucket
//        //          ::std::sort(temp.begin(), temp.end(), [&] ( ::std::pair<Key, T> const & x,  ::std::pair<Key, T> const & y){
//        //            return trans(x.first) < trans(y.first);
//        //          });
//        // //          auto end = ::std::unique(temp.begin(), temp.end(), this->key_equal_op);
//        // //          uniq_count += ::std::distance(temp.begin(), end);
//        //
//        //          // count via linear scan..
//        //          auto x = temp.begin();
//        //          ++uniq_count;  // first entry.
//        //          // compare pairwise.
//        //          auto y = temp.begin();  ++y;
//        //          while (y != temp.end()) {
//        //            if (trans(x->first) != trans(y->first)) {
//        //              ++uniq_count;
//        //              x = y;
//        //            }
//        //            ++y;
//        //          }
//        //        }
//        //        printf("%lu elements, %lu buckets, %lu unique\n", this->c.size(), this->c.capacity(), uniq_count);
//        // alternative approach to get number of unique keys is to use an batched_robinhood_set.  this will take more memory but probably will be faster than sort for large buckets (high repeats).
//
//
//        //        // third approach is to assume each bucket contains only 1 kmer/kmolecule.
//        //        // This is not generally true for all hash functions, so this is an over estimation of the repeat count.
//        //        // we equate bucket size to the number of repeats for that key.
//        //        // we can use mean, max, or mean+stdev.
//        //        // max overestimates significantly with potentially value > 1000, so don't use max.  (0.0078125 human: 50 sec. synth  32 sec)
//        //        // mean may be underestimating for well behaving hash function.   (0.0078125 human: 50 sec. synth  32 sec)
//        //        // mean + 2 stdev gets 95% of all entries.  1 stdev covers 67% of all entries, which for high coverage genome is probably better.
//        //        //    (1 stdev:  0.0078125 human: 49 sec. synth  32 sec;  2stdev: 0.0078125 human 49s synth: 33 sec)
//        //        double nBuckets = 0.0;
//        //        for (size_t i = 0, max = this->c.capacity(); i < max; ++i) {
//        //          if (this->c.bucket_size(i) > 0) nBuckets += 1.0;
//        //        }
//        //        double mean = static_cast<double>(this->c.size()) / nBuckets;
//        //        // do stdev = sqrt((1/nBuckets)  * sum((x - u)^2)).  value is more centered compared to summing the square of x.
//        //        double stdev = 0.0;
//        //        double entry = 0;
//        //        for (size_t i = 0, max = this->c.capacity(); i < max; ++i) {
//        //          if (this->c.bucket_size(i) == 0) continue;
//        //          entry = static_cast<double>(this->c.bucket_size(i)) - mean;
//        //          stdev += (entry * entry);
//        //        }
//        //        stdev = ::std::sqrt(stdev / nBuckets);
//        //        this->key_multiplicity = ::std::ceil(mean + 1.0 * stdev);  // covers 95% of data.
//        //        printf("%lu elements, %lu buckets, %f occupied, mean = %f, stdev = %f, key multiplicity = %lu\n", this->c.size(), this->c.capacity(), nBuckets, mean, stdev, this->key_multiplicity);
//
//        // finally, hard coding.  (0.0078125 human:  50 sec.  synth:  32 s)
//        // this->key_multiplicity = 50;
//
//        return multiplicity;
//      }
//
//
//      /**
//       * @brief insert new elements in the distributed batched_robinhood_multimap.
//       * @param first
//       * @param last
//       */
//      template <typename Predicate = ::bliss::filter::TruePredicate>
//      size_t insert(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
//        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
//        BL_BENCH_INIT(insert);
//
//        if (::dsc::empty(input, this->comm)) {
//          BL_BENCH_REPORT_MPI_NAMED(insert, "hash_multimap:insert", this->comm);
//          return 0;
//        }
//
//
//        BL_BENCH_START(insert);
//        this->transform_input(input);
//        BL_BENCH_END(insert, "transform_input", input.size());
//
//
//        //        printf("r %d key size %lu, val size %lu, pair size %lu, tuple size %lu\n", this->comm.rank(), sizeof(Key), sizeof(T), sizeof(::std::pair<Key, T>), sizeof(::std::tuple<Key, T>));
//        //        count_unique(input);
//        //        count_unique(bucketing(input, this->key_to_rank, this->comm));
//
//        // communication part
//        if (this->comm.size() > 1) {
//          BL_BENCH_START(insert);
//          // first remove duplicates.  sort, then get unique, finally remove the rest.  may not be needed
//
//          std::vector<size_t> recv_counts;
//          std::vector<::std::pair<Key, T> > buffer;
//          ::khmxx::distribute(input, this->key_to_rank, recv_counts, buffer, this->comm);
//          input.swap(buffer);
//
//          //auto recv_counts = ::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm);
//          //BLISS_UNUSED(recv_counts);
//          BL_BENCH_END(insert, "dist_data", input.size());
//        }
//
//        //        count_unique(input);
//
//        BL_BENCH_START(insert);
//        // local compute part.  called by the communicator.
//        size_t count = 0;
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value)
//          count = this->Base::local_insert(input.begin(), input.end(), pred);
//        else
//          count = this->Base::local_insert(input.begin(), input.end());
//        BL_BENCH_END(insert, "insert", this->c.size());
//
//        BL_BENCH_REPORT_MPI_NAMED(insert, "hash_multimap:insert", this->comm);
//        return count;
//      }
//
//
//      /// get the size of unique keys in the current local container.
//      virtual size_t local_unique_size() const {
//        if (this->local_changed) {
//
//          typename Base::template UniqueKeySetUtilityType<Key> unique_set(this->c.size());
//          auto max = this->c.end();
//          for (auto it = this->c.begin(); it != max; ++it) {
//            unique_set.emplace(it->first);
//          }
//          local_unique_count = unique_set.size();
//
//          this->local_changed = false;
//        }
//        return local_unique_count;
//      }
//  };
