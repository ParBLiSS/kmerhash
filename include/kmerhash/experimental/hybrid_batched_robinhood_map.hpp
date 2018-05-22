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
 * @file    hybrid_batched_robinhood_map.hpp
 * @ingroup index
 * @author  Tony Pan <tpan7@gatech.edu>
 * @brief   Implements the hybrid_multimap, hybrid map, and hybrid_reduction_map
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
 * general: specify OMP_NUM_THREADS,  OMP_PROC_BIND=true, OMP_PLACES=cores
 * OpenMPI: specify for mpirun --map-by ppr:1:node:pe=64 or ppr:4:socket:pe=16
 */

#ifndef HYBRID_BATCHED_ROBINHOOD_MAP_HPP
#define HYBRID_BATCHED_ROBINHOOD_MAP_HPP


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

#include "iterators/concatenating_iterator.hpp"

#include "containers/dsc_container_utils.hpp"

#include "kmerhash/incremental_mxx.hpp"

#include "kmerhash/io_utils.hpp"

#include "kmerhash/mem_utils.hpp"

#include "omp.h"

namespace hsc  // hybrid std container
{


	// =================
	// NOTE: when using this, need to further alias so that only Key param remains.
	// =================


  /**
   * @brief  hybrid robinhood map following some of std unordered map's interface.
   * @details
   *         it has as much of the same methods of distributed_batched_robinhood_map as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.  Also since we
   *         are working with 'distributed' data, batched operations are preferred.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using hybrid robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local hybrid robinhood map, or it may be done via sorting/lookup or other mapping
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
    std::vector<hyperloglog64<Key, InternalHash, 12> > hlls;


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
    		  StoreTransEqual, 
              Reducer,
    		  Alloc>;

      // std::batched_robinhood_multimap public members.
      using key_type              = typename local_container_type::key_type;
      using mapped_type           = typename local_container_type::mapped_type;
      using value_type            = typename local_container_type::value_type;
      using hasher                = typename local_container_type::hasher;
      using key_equal             = typename local_container_type::key_equal;
      using allocator_type        = typename local_container_type::allocator_type;
      using iterator              = ::bliss::iterator::ConcatenatingIterator<typename local_container_type::iterator>;
      using const_iterator        = ::bliss::iterator::ConcatenatingIterator<typename local_container_type::const_iterator>;
      using size_type             = typename local_container_type::size_type;
      using difference_type       = typename local_container_type::difference_type;

      using count_result_type     = decltype(::std::declval<local_container_type>().count(::std::declval<key_type>(),
                                                                                          ::std::declval<::bliss::filter::TruePredicate>(),
                                                                                           ::std::declval<::bliss::filter::TruePredicate>()));

    protected:
      std::vector<local_container_type> c;


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



      /// permute, given the assignment array and bucket counts.
      /// this is the second pass only.
      template <uint8_t prefetch_dist = 8, typename IT, typename MT, typename OT,
      typename ::std::enable_if<::std::is_same<typename ::std::iterator_traits<OT>::iterator_category,
                                               ::std::random_access_iterator_tag >::value, int>::type = 1  >
      void
      permute_by_bucketid(IT _begin, IT _end, MT bucketIds, 
			 std::vector<size_t> & bucket_sizes, std::vector<size_t> & bucket_offsets,
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

          // compute exclusive offsets first - passed in..

          size_t input_size = std::distance(_begin, _end);
          // [2nd pass]: saving elements into correct position, and save the final position.
          // not prefetching the bucket offsets - should be small enough to fit in cache.

          // ===========================
          // direct prefetch does not do well because bucketIds has bucket assignments and not offsets.
          // therefore bucket offset is not pointing to the right location yet.
          // instead, use stream write?


          // next prefetch output
          size_t offsets[prefetch_dist];

          MT i2o_it = bucketIds;
          MT i2o_eit = bucketIds;
          std::advance(i2o_eit, ::std::min(input_size, static_cast<size_t>(prefetch_dist)));
          size_t i = 0;
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
      template <typename IT, typename ASSIGN_TYPE, typename HLL >
      void
      assign_count_estimate(IT _begin, IT _end,
                             ASSIGN_TYPE const num_buckets,
                             std::vector<size_t> & bucket_sizes,
                             ASSIGN_TYPE* bucketIds,
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


        constexpr size_t batch_size = InternalHash::batch_size;

        // do a few cachelines at a time.  probably a good compromise is to do batch_size number of cachelines
        // 64 / sizeof(ASSIGN_TYPE)...
        constexpr size_t block_size = (64 / sizeof(ASSIGN_TYPE)) * batch_size;

//        BL_BENCH_START(permute_est);

        // initialize number of elements per bucket
        bucket_sizes.resize(num_buckets, 0);
//        BL_BENCH_END(permute_est, "alloc_count", num_buckets);


        // single bucket.  still need to estimate.
        if (num_buckets == 1) {
//        	BL_BENCH_START(permute_est);

          // set output buckets sizes
          bucket_sizes.resize(1, input_size);

          // estimate
          hll.update(_begin, input_size);

          // set the bucket ids.
          memset(bucketIds, 0, input_size * sizeof(ASSIGN_TYPE));

          return;
        }

        bool is_pow2 = (num_buckets & (num_buckets - 1)) == 0;
//        BL_BENCH_START(permute_est);

          // 1st pass of 2 pass algo.

          // [1st pass]: compute bucket counts and input2bucket assignment.
          // store input2bucket assignment in bucketIds temporarily.
          ASSIGN_TYPE* i2o_it = bucketIds;
          IT it = _begin;
          size_t i = 0, j, rem;
          ASSIGN_TYPE rank;

          // and compute the hll.
          if (batch_size > 1) {
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
// //          BL_BENCH_REPORT_NAMED(permute_est, "count_est_permute");

      }  // end of assign_count_estimate_permute


      /// hash, count and return assignment array and bucket counts.
      /// same as first pass of permute.
      template <typename IT, typename ASSIGN_TYPE>
      void assign_count(IT _begin, IT _end,
    		  ASSIGN_TYPE const num_buckets,
              std::vector<size_t> & bucket_sizes,
				ASSIGN_TYPE * bucketIds) const {


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


        constexpr size_t batch_size = InternalHashMod::batch_size;
//        		decltype(declval<decltype(declval<KeyToRank>().proc_trans_hash)>().h)::batch_size;
        // do a few cachelines at a time.  probably a good compromise is to do batch_size number of cachelines
        // 64 / sizeof(ASSIGN_TYPE)...
        constexpr size_t block_size = (64 / sizeof(ASSIGN_TYPE)) * batch_size;

//        BL_BENCH_START(permute_est);
        // initialize number of elements per bucket
        bucket_sizes.resize(num_buckets, 0);
//        BL_BENCH_END(permute_est, "alloc_count", num_buckets);

        size_t input_size = std::distance(_begin, _end);

        // single bucket.  still need to estimate.
        if (num_buckets == 1) {
//            BL_BENCH_START(permute_est);
          // set output buckets sizes
          bucket_sizes.resize(1, input_size);

          memset(bucketIds, 0, input_size * sizeof(ASSIGN_TYPE));

          return;
        }

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
          if (batch_size > 1) {
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
// //          BL_BENCH_REPORT_NAMED(permute_est, "count_permute");

      }  // end of assign_estimate_count




    public:

      batched_robinhood_map_base(const mxx::comm& _comm) : Base(_comm),
		  key_to_hash(DistHash<trans_val_type>(9876543), DistTrans<Key>(), ::bliss::transform::identity<hash_val_type>())
		  //hll(ceilLog2(_comm.size()))  // top level hll. no need to ignore bits.
        {

    	  if (_comm.rank() == 0)
    		  printf("rank %d initializing for %d threads\n", _comm.rank(), omp_get_max_threads());
//		c = new local_container_type[omp_get_max_threads()];
//		hlls = new hyperloglog64<Key, InternalHash, 12>[omp_get_max_threads()];
  	  c.resize(omp_get_max_threads());
  	  hlls.resize(omp_get_max_threads());

 //   	  this->c.set_ignored_msb(ceilLog2(_comm.size()));   // NOTE THAT THIS SHOULD MATCH KEY_TO_RANK use of bits in hash table.

	#pragma omp parallel
	{
			int tid = omp_get_thread_num();
			c[tid].swap(local_container_type());  // get thread local allocation
			hlls[tid].swap(hyperloglog64<Key, InternalHash, 12>());
	}
      }



      virtual ~batched_robinhood_map_base() {
//	delete [] c;
//	delete [] hlls;
	};



      /// returns the local storage.  please use sparingly.
      local_container_type& get_local_container(int tid) { return c[tid]; }
      local_container_type const & get_local_container(int tid) const { return c[tid]; }

      local_container_type * get_local_containers() { return c; }
      local_container_type const * get_local_containers() const { return c; }

      // ================ local overrides

      /// clears the batched_robinhood_map
      virtual void local_reset() noexcept {
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            this->c[i].clear();
    	    this->c[i].rehash(128);
          }
      }

      virtual void local_clear() noexcept {
        for (int i = 0; i < omp_get_max_threads(); ++i)
            this->c[i].clear();
    
      }

      /// reserve space.  n is the local container size.  this allows different processes to individually adjust its own size.
      virtual void local_reserve( size_t n ) {
          #pragma omp parallel
          {
        	  this->c[omp_get_thread_num()].reserve(n);
          }
      }

      virtual void local_rehash( size_t b ) {
        #pragma omp parallel
        {
    	  this->c[omp_get_thread_num()].rehash(b);
        }
      }


      // note that for each method, there is a local version of the operartion.
      // this is for use by the asynchronous version of communicator as callback for any messages received.
      /// check if empty.
      virtual bool local_empty() const {
        bool res = true;
        for (int i = 0; i < omp_get_max_threads(); ++i)
          res &= (this->c[i].size() == 0);
          
        return res;
      }

      /// get number of entries in local container
      virtual size_t local_size() const {
        size_t res = 0;
        for (int i = 0; i < omp_get_max_threads(); ++i)
            res += this->c[i].size();
        return res;
      }

      virtual size_t local_capacity() const {
          size_t res = 0;
        for (int i = 0; i < omp_get_max_threads(); ++i)
            res += this->c[i].capacity();
        return res;
      }

      /// get number of entries in local container
      virtual size_t local_unique_size() const {
        return this->local_size();
      }



      virtual std::vector<bool> local_empties() const {
        std::vector<bool> res(omp_get_max_threads());
        for (int i = 0; i < omp_get_max_threads(); ++i)
          res[i] = (this->c[i].size() == 0);
          
        return res;
      }

      /// get number of entries in local container
      virtual std::vector<size_t> local_sizes() const {
        std::vector<size_t> res(omp_get_max_threads());
        for (int i = 0; i < omp_get_max_threads(); ++i)
            res[i] = this->c[i].size();
        return res;
      }

      virtual std::vector<size_t> local_capacitys() const {
        std::vector<size_t> res(omp_get_max_threads());
        for (int i = 0; i < omp_get_max_threads(); ++i)
            res[i] = this->c[i].capacity();
        return res;
      }


      /// get number of entries in local container
      virtual std::vector<size_t> local_unique_sizes() const {
        return this->local_sizes();
      }


      virtual double get_max_load_factor() {
        return this->c[0].get_max_load_factor();
      }
      virtual double get_min_load_factor() {
        return this->c[0].get_min_load_factor();
      }

      virtual void set_max_load_factor(double const & max_load) {
        for (int i = 0; i < omp_get_max_threads(); ++i)
        {
              this->c[i].set_max_load_factor(max_load);
        }
      }
      virtual void set_min_load_factor(double const & min_load) {
        for (int i = 0; i < omp_get_max_threads(); ++i)
        {
              this->c[i].set_min_load_factor(min_load);
          }
      }
      virtual void set_insert_lookahead(uint8_t insert_prefetch) {
        for (int i = 0; i < omp_get_max_threads(); ++i)
        {
              this->c[i].set_insert_lookahead(insert_prefetch);
        }
      }
      virtual void set_query_lookahead(uint8_t query_prefetch) {
        for (int i = 0; i < omp_get_max_threads(); ++i)
        {
              this->c[i].set_query_lookahead(query_prefetch);
        }
      }


      typename local_container_type::const_iterator cbegin(int tid) const {
        return this->c[tid].cbegin();
      }

      typename local_container_type::const_iterator cend(int tid) const {
        return this->c[tid].cend();
      }

      const_iterator cbegin() const {
        const_iterator iter; 

        for (int i = 0; i < omp_get_max_threads(); ++i)
        {  
            iter.addRange(this->c[i].cbegin(), this->c[i].cend() );
        }
        return iter;
      }

      const_iterator cend() const {
        return const_iterator(this->c[omp_get_max_threads() - 1].cend());
      }

      using Base::size;
      using Base::unique_size;
      using Base::get_multiplicity;

      /// convert the map to a vector
      virtual void to_vector(std::vector<std::pair<Key, T> > & results) const {
        std::vector<size_t> sizes = this->local_sizes();
        std::vector<size_t> offsets(omp_get_max_threads());
        size_t sum = 0;
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            offsets[i] = sum;
            sum += sizes[i];
        }
        results.clear();
        results.resize(sum);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::copy(this->c[tid].cbegin(), this->c[tid].cend(), results.begin() + offsets[tid]);
        }
      }
      /// extract the unique keys of a map.
      virtual void keys(std::vector<Key> & results) const {
        std::vector<size_t> sizes = this->local_sizes();
        std::vector<size_t> offsets(omp_get_max_threads());
        size_t sum = 0;
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            offsets[i] = sum;
            sum += sizes[i];
        }
        results.clear();
        results.resize(sum);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::transform(this->c[tid].cbegin(), this->c[tid].cend(), results.begin() + offsets[tid],
                [](std::pair<Key,T> const & x){
                    return x.first;
                });
        }
      }

    protected:


  /**
   * @brief insert new elements in the hybrid batched_robinhood_multimap.
   * @param input  vector.  will be permuted.
   */
  template <bool estimate, typename V, typename OP, typename Predicate = ::bliss::filter::TruePredicate>
  int64_t modify_1(std::vector<V>& input, OP const & compute) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(modify);

    int64_t cnt = 0; 

    if (::dsc::empty(input, this->comm)) {
        BL_BENCH_REPORT_MPI_NAMED(modify, "hashmap:modify_1", this->comm);
        return 0;
    }

    BL_BENCH_START(modify);

    // get some common variables
    size_t in_size = input.size();
    size_t nthreads_global = omp_get_max_threads();
    int comm_size = 1;
    int batch_size = InternalHash::batch_size;

    // set up per thread storage
    std::vector< std::vector<size_t> > thread_bucket_sizes(omp_get_max_threads());
    std::vector< std::vector<size_t> > thread_bucket_offsets(omp_get_max_threads());
    std::vector<size_t> node_bucket_sizes(nthreads_global, 0);
    std::vector<size_t> node_bucket_offsets(nthreads_global, 0);
    std::vector<size_t> thread_total(omp_get_max_threads(), 0);

    BL_BENCH_END(modify, "alloc", nthreads_global);

#ifdef MT_DEBUG
    V *transformed = ::utils::mem::aligned_alloc<V>(input.size(), 64);

    this->transform_input(input.begin(), input.end(), transformed);


    std::vector<size_t> test_sizes(nthreads_global, 0);
    uint32_t *test_bids = ::utils::mem::aligned_alloc<uint32_t>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint8_t>(nthreads_global),
         test_sizes, reinterpret_cast<uint8_t*>(test_bids) );
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint16_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint16_t*>(test_bids) );
    } else {   // mpi supports only 31 bit worth of ranks.
        this->assign_count(transformed, transformed + input.size(), static_cast<uint32_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint32_t*>(test_bids) );
    }
    // above is same.

    std::vector<size_t> test_offsets(nthreads_global, 0);
    for (size_t i = 1; i < nthreads_global; ++i) {
    	test_offsets[i] = test_offsets[i-1]+test_sizes[i-1];
    }
    V* test_res = ::utils::mem::aligned_alloc<V>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint8_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint16_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint32_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    }

    ::utils::mem::aligned_free(transformed);
#endif


    BL_BENCH_START(modify);
    #pragma omp parallel reduction(+ : cnt)
    {
        int tid = omp_get_thread_num();
        int tcnt = omp_get_num_threads();

        size_t block = in_size / tcnt;
        size_t rem = in_size % tcnt;
        size_t r_start = block * tid + std::min(rem, static_cast<size_t>(tid));
        block += (static_cast<size_t>(tid) < rem ? 1: 0);
        size_t r_end = r_start + block;

        auto it = input.data() + r_start;
        auto et = input.data() + r_end;

        if (tcnt > 1) { // only do if more than 1 thread.  then we need to permute
            V* buffer = ::utils::mem::aligned_alloc<V>(r_end - r_start + batch_size);

            // transform once.  bucketing and distribute will read it multiple times.
            this->transform_input(it, et, buffer);

            // ===== assign to get bucket ids.
            thread_bucket_sizes[tid].resize(nthreads_global, 0);
            uint32_t* bid_buf = ::utils::mem::aligned_alloc<uint32_t>(r_end - r_start + batch_size);


            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                this->assign_count(buffer, buffer + block, static_cast<uint8_t>(nthreads_global),
                 thread_bucket_sizes[tid], reinterpret_cast<uint8_t*>(bid_buf) );
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->assign_count(buffer, buffer + block, static_cast<uint16_t>(nthreads_global),
                 thread_bucket_sizes[tid], reinterpret_cast<uint16_t*>(bid_buf) );
            } else {   // mpi supports only 31 bit worth of ranks.
                this->assign_count(buffer, buffer + block, static_cast<uint32_t>(nthreads_global),
                 thread_bucket_sizes[tid], reinterpret_cast<uint32_t*>(bid_buf) );
            }
            thread_bucket_offsets[tid].resize(nthreads_global, 0);

#ifdef MT_DEBUG   //verified same bucket assignment
            {
            	bool same = true;
				int i = 0;
	            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {

					uint8_t* bit = reinterpret_cast<uint8_t*>(test_bids) + r_start;
					uint8_t* bet = reinterpret_cast<uint8_t*>(test_bids) + r_end;
					uint8_t* bit2 = reinterpret_cast<uint8_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
					uint16_t* bit = reinterpret_cast<uint16_t*>(test_bids) + r_start;
					uint16_t* bet = reinterpret_cast<uint16_t*>(test_bids) + r_end;
					uint16_t* bit2 = reinterpret_cast<uint16_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else {   // mpi supports only 31 bit worth of ranks.
					uint32_t* bit = reinterpret_cast<uint32_t*>(test_bids) + r_start;
					uint32_t* bet = reinterpret_cast<uint32_t*>(test_bids) + r_end;
					uint32_t* bit2 = reinterpret_cast<uint32_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            }
				printf("thread %d bucket ids same ? %s : %d\n", tid, (same ? "yes":"no"), i );
            }
#endif


            #pragma omp barrier

            // ===== now compute the offsets 
            // exclusive scan of everyhing.  proceed in 3 step
            //  1. thread local scan for each bucket, each thread do a block of comm_size buckets, across all thread., store back into each thread's thread_bucket_offsets
            //     also store count for each bucket
            size_t offset = 0;
            for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
                node_bucket_offsets[j] = offset;  // exscan within block for this thread
                // iterate over a block of bucket sizes.
                for (int t = 0; t < tcnt; ++t) {
                    // iterate over each thread's data
                    thread_bucket_offsets[t][j] = offset;     // exscan within block for this thread
                    offset += thread_bucket_sizes[t][j];
                }
                // store the bucket count
                node_bucket_sizes[j] = offset - node_bucket_offsets[j];
            }   // when finished, has per bucket offsets for each thread, and total 
            // when finished, has per bucket offsets for each thread, and total 
            thread_total[tid] = offset;   // save the total element count, for prefix scan next and update.
            // wait for all to be done with the scan.
            #pragma omp barrier

			//  2. 1 thread to scan thread total to get thread offsets.  O(C) instead of O(CP)
			#pragma omp single
			{
				size_t curr = thread_total[0];
				size_t sum = 0;
				for (int t = 1; t < tcnt; ++t) {
					thread_total[t - 1] = sum;
					sum += curr;
					curr = thread_total[t];
				}
				thread_total[tcnt - 1] = sum;
			}

			// wait for all to finish
			#pragma omp barrier
			//  3. thread local update of per thread offsets.
			offset = thread_total[tid];
			for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
				// update the per thread prefix scan to node prefix scan
				node_bucket_offsets[j] += offset;
				for (int t = 0; t < tcnt; ++t) {
					// iterate over each thread's data, update offsets
					thread_bucket_offsets[t][j] += offset;
				}
			}


            //===============  now the offsets are ready.  we can permute.
            // since each thread updates a portion of every other thread's thread buckewt sizes, need this barrier here.
   			#pragma omp barrier

            // then call permute_by_bucketId and computed offsets. 
            // NOTE: permute back into input
            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint8_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint16_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint32_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            }

            ::utils::mem::aligned_free(bid_buf);
            ::utils::mem::aligned_free(buffer);

            #pragma omp barrier

            // update it and et.
            it = input.data() + node_bucket_offsets[tid];
            et = it + node_bucket_sizes[tid]; 


#ifdef MT_DEBUG
#pragma omp barrier
            #pragma omp single
            {
            	printf("node offests from thread %d", tid);
            	for (int j = 0; j < nthreads_global; ++j) {
            		printf("\t[%ld, %ld)", node_bucket_offsets[j], node_bucket_offsets[j] + node_bucket_sizes[j] );
            	}
            	printf("\n");

            }
			#pragma omp barrier
#endif
        } else {
            // 1 thread, no permute needed, so just transform inplace..
            this->transform_input(it, et, it);
        }

        // insert into the right places
        size_t before = this->c[tid].size();
        // local compute part.  called by the communicator.

#ifdef MT_DEBUG
        {
			// compare to single threaded
			bool same = true;
			auto iit2 = test_res + node_bucket_offsets[tid];
			int i = 0;
			for (auto iit = it; iit != et; ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("thread %d permutation same ? %s : %d\n", tid, (same ? "yes":"no"), i );
        }
#endif

        compute(tid, it, et, estimate);
        //printf("rank %d of %d, thread %d of %d before %ld after count %ld\n", 0, 1, tid, tcnt, before, this->c[tid].size());

        cnt = static_cast<int64_t>(this->c[tid].size()) - static_cast<int64_t>(before);
    } // end parallel section
    BL_BENCH_END(modify, "base:trans_permute_modify", cnt);

#ifdef MT_DEBUG
        {
        // compare to single threaded
        bool same = true;
        auto iit2 = test_res;
        int i = 0;
        for (auto iit = input.begin(); iit != input.end(); ++iit, ++iit2, ++i) {
        	same &= (*iit == *iit2);
        	if (!same) break;
        }
        printf("ALL THREADS permutation same ? %s : %d\n", (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_res);
        ::utils::mem::aligned_free(test_bids);

#endif


    BL_BENCH_REPORT_MPI_NAMED(modify, "hashmap:modify_1", this->comm);

    return cnt;
  }

  template <bool estimate, typename V, typename C1, typename C2, typename Predicate = ::bliss::filter::TruePredicate>
  int64_t modify_p(std::vector<V >& input, C1 const & c1, C2 const & c2) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(modify);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(modify, "hashmap:modify", this->comm);
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
    // modify.  linear read
    // free buffer

    BL_BENCH_START(modify);

    // get some common variables
    int comm_size = this->comm.size();
    int comm_rank = this->comm.rank();
    size_t in_size = input.size();
    size_t nthreads_global = comm_size * omp_get_max_threads();
    int batch_size = InternalHash::batch_size;

    // set up per thread storage
    std::vector< std::vector<size_t> > thread_bucket_sizes(omp_get_max_threads());
    std::vector< std::vector<size_t> > thread_bucket_offsets(omp_get_max_threads());
    std::vector<size_t> node_bucket_sizes(nthreads_global, 0);
    std::vector<size_t> node_bucket_offsets(nthreads_global, 0);
    std::vector<size_t> thread_total(omp_get_max_threads(), 0);

    BL_BENCH_END(modify, "alloc", nthreads_global);


    if (comm_rank == 0) printf("estimating ? %s\n", (estimate ? "y" : "n"));

#ifdef MT_DEBUG
    V *transformed = ::utils::mem::aligned_alloc<V>(input.size(), 64);

    this->transform_input(input.begin(), input.end(), transformed);


    std::vector<size_t> test_sizes(nthreads_global, 0);
    uint32_t *test_bids = ::utils::mem::aligned_alloc<uint32_t>(input.size(), 64);

    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint8_t>(nthreads_global),
         test_sizes, reinterpret_cast<uint8_t*>(test_bids) );
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint16_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint16_t*>(test_bids) );
    } else {   // mpi supports only 31 bit worth of ranks.
        this->assign_count(transformed, transformed + input.size(), static_cast<uint32_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint32_t*>(test_bids) );
    }
    // above is same.

    std::vector<size_t> test_offsets(nthreads_global, 0);
    for (size_t i = 1; i < nthreads_global; ++i) {
    	test_offsets[i] = test_offsets[i-1]+test_sizes[i-1];
    }
    V* test_res = ::utils::mem::aligned_alloc<V>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint8_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint16_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint32_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    }

    ::utils::mem::aligned_free(transformed);

    std::vector<size_t> rtest_sizes(nthreads_global, 0);
    mxx::all2all(test_sizes.data(), omp_get_max_threads(), rtest_sizes.data(), this->comm);

    std::vector<size_t> test_sendcounts(this->comm.size(), 0);
    std::vector<size_t> test_recvcounts(this->comm.size(), 0);
    std::vector<size_t> test_recvcounts2(this->comm.size(), 0);
    size_t test_recv_total = 0;
    for (size_t i = 0; i < this->comm.size(); ++i) {
    	for (size_t j = 0; j < omp_get_max_threads(); ++j) {
    		test_sendcounts[i] += test_sizes[i * omp_get_max_threads() + j];
    		test_recvcounts[i] += rtest_sizes[i * omp_get_max_threads() + j];
    		test_recv_total += rtest_sizes[i * omp_get_max_threads() + j];
    	}
    }

    // send the test data.
    V* test_distributed = ::utils::mem::aligned_alloc<V>(test_recv_total);

    ::khmxx::distribute_permuted(test_res, test_res + input.size(),
                test_sendcounts, test_distributed, test_recvcounts, this->comm);


    // assert that test_recvcounts and test_recvcounts2 are the same.
//    {
//    	bool same = true;
//    	for (size_t i = 0; i < omp_get_max_threads(); ++i) {
//    		same &= test_recvcounts[i] == test_recvcounts2[i];
//    		if (!same) break;
//    	}
//    	printf("rank %d recv counts via permute same? %s \n", this->comm.rank(), (same ? "yes" : "no"));
//    }


#endif

    BL_BENCH_COLLECTIVE_START(modify, "permute_estimate", this->comm);
    
    // THREADED code to permute.  note that the partitioning is even, per thread memory access is random within ranges of data, 
    // and there should be no contention or fine grain synchronization.

    size_t before = 0;
    #pragma omp parallel reduction(+: before)
    {
        int tid = omp_get_thread_num();
        int tcnt = omp_get_num_threads();

        size_t block = in_size / tcnt;
        size_t rem = in_size % tcnt;
        size_t r_start = block * tid + std::min(rem, static_cast<size_t>(tid));
        block += (static_cast<size_t>(tid) < rem ? 1: 0);
        size_t r_end = r_start + block;


      // get mapping to proc
      // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
        V* buffer = ::utils::mem::aligned_alloc<V>(r_end - r_start + batch_size);

//        BL_BENCH_END(modify, "alloc", input.size());
        auto it = input.data() + r_start;
        auto et = input.data() + r_end;

	        // transform once.  bucketing and distribute will read it multiple times.
//	        BL_BENCH_COLLECTIVE_START(modify, "transform", this->comm);
        this->transform_input(it, et, buffer);
//	        BL_BENCH_END(modify, "transform", input.size());


// NOTE: overlap comm is incrementally inserting so we estimate before transmission, thus global estimate, and 64 bit
//            hash is needed.
//       non-overlap comm can estimate just before insertion for more accurate estimate based on actual input received.
//          so 32 bit hashes are sufficient.
            // count and estimate and save the bucket ids.
//    BL_BENCH_COLLECTIVE_START(modify, "permute_estimate", this->comm);
        // allocate an HLL
        // allocate the bucket sizes array

        thread_bucket_sizes[tid].resize(nthreads_global, 0);
    
        uint32_t* bid_buf = ::utils::mem::aligned_alloc<uint32_t>(r_end - r_start + batch_size);

    #if defined(OVERLAPPED_COMM) //|| defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
        if (estimate) {
            if ( nthreads_global <= std::numeric_limits<uint8_t>::max()) {
            
                this->assign_count_estimate(buffer, buffer + block, static_cast<uint8_t>(nthreads_global), thread_bucket_sizes[tid],
                            reinterpret_cast<uint8_t*>(bid_buf), hlls[tid] );
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->assign_count_estimate(buffer, buffer + block, static_cast<uint16_t>(nthreads_global), thread_bucket_sizes[tid],
                        reinterpret_cast<uint16_t*>(bid_buf), hlls[tid] );
            } else {   // mpi supports only 31 bit worth of ranks.
                    
                    this->assign_count_estimate(buffer, buffer + block, static_cast<uint32_t>(nthreads_global), thread_bucket_sizes[tid],
                            reinterpret_cast<uint32_t*>(bid_buf), hlls[tid] );
            }

            //==== now that hll is done, merge in binary way.
            if (tcnt > 1) {  // merge if more than 1 thread.
                int mask = 0;
                for (int i = 1; i < tcnt; i <<= 1) {
                    mask = (mask << 1) | 1; 
                    if (((tid & mask) == 0) &  // select only power of 2 to merge into.
                        ((tid ^ i) < tcnt)) {  // flip the bit to get the current peer.
                            this->hlls[tid].merge(this->hlls[tid ^ i]);
                        }
                        #pragma omp barrier
                }
            }

        } else {
    #endif
            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                this->assign_count(buffer, buffer + block, static_cast<uint8_t>(nthreads_global), thread_bucket_sizes[tid],
                            reinterpret_cast<uint8_t*>(bid_buf) );
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->assign_count(buffer, buffer + block, static_cast<uint16_t>(nthreads_global), thread_bucket_sizes[tid],
                        reinterpret_cast<uint16_t*>(bid_buf) );
            } else {   // mpi supports only 31 bit worth of ranks.
                    this->assign_count(buffer, buffer + block, static_cast<uint32_t>(nthreads_global), thread_bucket_sizes[tid],
                            reinterpret_cast<uint32_t*>(bid_buf) );
            }
    #if defined(OVERLAPPED_COMM) //|| defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)
        }
    #endif
        // do some calc with thread_bucket_sizes to get offsets for each bucket for each thread in node-wide permuted input array.
        thread_bucket_offsets[tid].resize(nthreads_global,0);
        

#ifdef MT_DEBUG   //verified same bucket assignment
            {
            	bool same = true;
				int i = 0;
	            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {

					uint8_t* bit = reinterpret_cast<uint8_t*>(test_bids) + r_start;
					uint8_t* bet = reinterpret_cast<uint8_t*>(test_bids) + r_end;
					uint8_t* bit2 = reinterpret_cast<uint8_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
					uint16_t* bit = reinterpret_cast<uint16_t*>(test_bids) + r_start;
					uint16_t* bet = reinterpret_cast<uint16_t*>(test_bids) + r_end;
					uint16_t* bit2 = reinterpret_cast<uint16_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else {   // mpi supports only 31 bit worth of ranks.
					uint32_t* bit = reinterpret_cast<uint32_t*>(test_bids) + r_start;
					uint32_t* bet = reinterpret_cast<uint32_t*>(test_bids) + r_end;
					uint32_t* bit2 = reinterpret_cast<uint32_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            }
				printf("rank %d thread %d bucket ids same ? %s : %d\n",
						this->comm.rank(), tid, (same ? "yes":"no"), i );
            }
#endif



        // make sure all threads have reached hererehashing
        #pragma omp barrier

        // exclusive scan of everyhing.  proceed in 3 step
        //  1. thread local scan for each bucket, each thread do a block of comm_size buckets, across all thread., store back into each thread's thread_bucket_offsets
        //     also store count for each bucket
            size_t offset = 0;
            for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
                node_bucket_offsets[j] = offset;  // exscan within block for this thread
                // iterate over a block of bucket sizes.
                for (int t = 0; t < tcnt; ++t) {
                    // iterate over each thread's data
                    thread_bucket_offsets[t][j] = offset;     // exscan within block for this thread
                    offset += thread_bucket_sizes[t][j];
                }
                // store the bucket count
                node_bucket_sizes[j] = offset - node_bucket_offsets[j];
            }   // when finished, has per bucket offsets for each thread, and total 
            // when finished, has per bucket offsets for each thread, and total 
            thread_total[tid] = offset;   // save the total element count, for prefix scan next and update.
            // // wait for all to be done with the scan.
            // #pragma omp barrier
// 
            // #pragma omp critical
            // {
            //     printf("rank %d of %d, thread %d of %d bucket offsets: ", comm_rank, comm_size, tid, tcnt);
            //     for (size_t i = 0; i < nthreads_global; ++i) {
            //         printf("%ld ", thread_bucket_offsets[tid][i]);
            //     }
            //     printf("\n");
            // }
            // #pragma omp barrier
            // #pragma omp single
            // {
            //     printf("rank %d of %d, thread %d of %d node bucket sizes: ", comm_rank, comm_size, tid, tcnt);
            //     for (size_t i = 0; i < nthreads_global; ++i) {
            //         printf("%ld ", node_bucket_sizes[i]);
            //     }
            //     printf("\n");
            //     printf("rank %d of %d, thread %d of %d node bucket offsets: ", comm_rank, comm_size, tid, tcnt);
            //     for (size_t i = 0; i < nthreads_global; ++i) {
            //         printf("%ld ", node_bucket_offsets[i]);
            //     }
            //     printf("\n");
            //     printf("rank %d of %d, thread %d of %d thread_totals: ", comm_rank, comm_size, tid, tcnt);
            //     for (int i = 0; i < tcnt; ++i) {
            //         printf("%ld ", thread_total[i]);
            //     }
            //     printf("\n");
            // }

            if (tcnt > 1) {  // update offsets if more than 1 thread.
                // wait for all to be done with the scan.
                #pragma omp barrier
                //  2. 1 thread to scan thread total to get thread offsets.  O(C) instead of O(CP)
                #pragma omp single
                {
                    size_t curr = thread_total[0];
                    size_t sum = 0;
                    for (int t = 1; t < tcnt; ++t) {
                        thread_total[t - 1] = sum;
                        sum += curr;
                        curr = thread_total[t];
                    }
                    thread_total[tcnt - 1] = sum;
                }

                // #pragma omp barrier
                // #pragma omp single
                // {

                //     printf("rank %d of %d, thread %d of %d exscan thread_totals: ", comm_rank, comm_size, tid, tcnt);
                //     for (int i = 0; i < tcnt; ++i) {
                //         printf("%ld ", thread_total[i]);
                //     }
                //     printf("\n");
                // }

                // wait for all to finish
                #pragma omp barrier
                //  3. thread local update of per thread offsets.
                offset = thread_total[tid];
                for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
                    // update the per thread prefix scan to node prefix scan
                    node_bucket_offsets[j] += offset;
                    for (int t = 0; t < tcnt; ++t) {
                        // iterate over each thread's data, update offsets
                        thread_bucket_offsets[t][j] += offset;
                    }
                }
                // #pragma omp barrier
                // #pragma omp critical
                // {
                //     printf("rank %d of %d, thread %d of %d exscan bucket offsets: ", comm_rank, comm_size, tid, tcnt);
                //     for (size_t i = 0; i < nthreads_global; ++i) {
                //         printf("%ld ", thread_bucket_offsets[tid][i]);
                //     }
                //     printf("\n");
                // }
                // #pragma omp barrier
                // #pragma omp single
                // {
                //     printf("rank %d of %d, thread %d of %d post exscan node bucket offsets: ", comm_rank, comm_size, tid, tcnt);
                //     for (size_t i = 0; i < nthreads_global; ++i) {
                //         printf("%ld ", node_bucket_offsets[i]);
                //     }
                //     printf("\n");
                // }
            }            
        //===============  now the offsets are ready.  we can permute.
            // since each thread updates a portion of every other thread's thread buckewt sizes, need this barrier here.
			#pragma omp barrier

        // then call permute_by_bucketId and computed offsets.
            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint8_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint16_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint32_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            }

        ::utils::mem::aligned_free(bid_buf);
        ::utils::mem::aligned_free(buffer);

#ifdef MT_DEBUG
        {
			#pragma omp barrier

			// update it and et.
			it = input.data() + node_bucket_offsets[tid];
			et = it + node_bucket_sizes[tid];

			// compare to single threaded
			bool same = true;
			auto iit2 = test_res + node_bucket_offsets[tid];
			int i = 0;
			for (auto iit = it; iit != et; ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("rank %d thread %d permutation same ? %s : %d\n",
					this->comm.rank(), tid, (same ? "yes":"no"), i );
        }
#endif




        before = this->c[tid].size();
    }  // omp parallel for permuting.  done.

    BL_BENCH_END(modify, "permute_estimate", input.size());
            


#ifdef MT_DEBUG
		{
			printf("rank %d node offests from ALL THREADS ", this->comm.rank());
			for (int j = 0; j < nthreads_global; ++j) {
				printf("\t[%ld, %ld)", node_bucket_offsets[j], node_bucket_offsets[j] + node_bucket_sizes[j] );
			}
			printf("\n");

		}

        {
			// compare to single threaded
			bool same = true;
			auto iit2 = test_res;
			int i = 0;
			for (auto iit = input.begin(); iit != input.end(); ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("rank %d ALL THREADS permutation same ? %s : %d\n",
					this->comm.rank(), (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_res);
        ::utils::mem::aligned_free(test_bids);
#endif

	BL_BENCH_COLLECTIVE_START(modify, "a2a_count", this->comm);
    // send off the node_bucket_sizes - for per recv thread traversal.
    std::vector<size_t> rnode_bucket_sizes(nthreads_global, 0);
    std::vector<size_t> rnode_bucket_offsets(nthreads_global, 0);
    mxx::all2all(node_bucket_sizes.data(), omp_get_max_threads(), rnode_bucket_sizes.data(), this->comm);

    // now compute the send_counts
    // single thread to do this, so don't have to worry about comm_size / tcnt not being even.
    std::vector<size_t> send_counts(this->comm.size());
    std::vector<size_t> recv_counts(this->comm.size());

    std::vector<size_t> rthread_total(omp_get_max_threads(), 0);


    size_t send_cnt, recv_cnt, recv_offset = 0;
    int jmax =  omp_get_max_threads();
    for (int i = 0; i < this->comm.size(); ++i) {
        send_cnt = 0;
        recv_cnt = 0;
#if defined(OVERLAPPED_COMM)
        // if overlap, then we reset the offset for each rank-pair's message.
        recv_offset = 0;  
#endif
        for (int j = 0; j < jmax; ++j) {
            // get send count and recv count
            send_cnt += node_bucket_sizes[i * jmax + j];
            recv_cnt += rnode_bucket_sizes[i * jmax + j];

            // compute recv bucket offsets.
            rnode_bucket_offsets[i * jmax + j] = recv_offset;
            recv_offset += rnode_bucket_sizes[i * jmax + j];
            
            // per thread total recv
            rthread_total[j] += rnode_bucket_sizes[i * jmax + j];
        }
        send_counts[i] = send_cnt;
        recv_counts[i] = recv_cnt;
    }
    BL_BENCH_END(modify, "a2a_count", recv_counts.size());

    size_t after = 0;

#if defined(OVERLAPPED_COMM) //|| defined(OVERLAPPED_COMM_BATCH) || defined(OVERLAPPED_COMM_FULLBUFFER) || defined(OVERLAPPED_COMM_2P)

    if (estimate) {
        BL_BENCH_COLLECTIVE_START(modify, "alloc_hashtable", this->comm);
        size_t est = this->hlls[0].estimate_average_per_rank(this->comm);
        printf("rank %d estimated size %ld\n", this->comm.rank(), est);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int tcnt = omp_get_num_threads();

            // further divide by number of threads.
            size_t lest = (est + tcnt - 1) / tcnt;
            if (lest > (this->c[tid].get_max_load_factor() * this->c[tid].capacity()))
            // add 10% just to be safe.
                this->c[tid].reserve(static_cast<size_t>(static_cast<double>(lest) * (1.0 + this->hlls[tid].est_error_rate + 0.1)));
        }
        BL_BENCH_END(modify, "alloc_hashtable", est);
    }  // allocation threads.
#endif

#if defined(OVERLAPPED_COMM)

    BL_BENCH_COLLECTIVE_START(modify, "a2av_modify", this->comm);

    // need to be threaded.
    ::khmxx::incremental::ialltoallv_and_modify(
        input.data(), input.data() + input.size(),
        send_counts,
        [this, &rnode_bucket_offsets, &rnode_bucket_sizes, &c2](int rank, V* b, V* e){
            // compute just the offsets within a thread block.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int tcnt = omp_get_num_threads();
                V* bb = b + rnode_bucket_offsets[rank * tcnt + tid];
                V* ee = bb + rnode_bucket_sizes[rank * tcnt + tid];

                c2(tid, bb, ee); // this->c[tid].insert_no_estimate(bb, ee, T(1));
            }  // finished parallel modify.
        },
        this->comm);
    #pragma omp parallel reduction(+: after)
    {
        int tid = omp_get_thread_num();
        after = this->c[tid].size();
    }
                                                    
    BL_BENCH_END(modify, "a2av_modify", after);

#else
    size_t recv_total = recv_offset;  // for overlap, this would be incorrect but that is okay.

    BL_BENCH_START(modify);
    V* distributed = ::utils::mem::aligned_alloc<V>(recv_total + batch_size);
    BL_BENCH_END(modify, "alloc_output", recv_total);


    BL_BENCH_COLLECTIVE_START(modify, "a2a", this->comm);
    ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
                send_counts, distributed, recv_counts, this->comm);
    BL_BENCH_END(modify, "a2a", input.size());

    BL_BENCH_COLLECTIVE_START(modify, "modify", this->comm);
    #pragma omp parallel reduction(+: after)
    {
        int tid = omp_get_thread_num();
        int tcnt = omp_get_num_threads();

        //======= shuffle received to get contiguous memory.  (CN buckets)
        V* shuffled;
        
        if (tcnt > 1) {   // only need to shuffle if more than 1 thread
            shuffled = ::utils::mem::aligned_alloc<V>(rthread_total[tid] + batch_size);
            V* it = shuffled;
            for (int i = 0; i < this->comm.size(); ++i) {
                // copy from one src rank at a time
                memcpy(it, distributed + rnode_bucket_offsets[i * tcnt + tid], 
                    rnode_bucket_sizes[i * tcnt + tid] * sizeof(V));

                it += rnode_bucket_sizes[i * tcnt + tid];
            }

        } else {
            shuffled = distributed;
        }




        // local compute part.  called by the communicator.
        // TODO: predicated version.

        // if (estimate)
        // NOTE: local cardinality estimation.
        c1(tid, shuffled, shuffled + rthread_total[tid], estimate); 
            // this->c[tid].insert(shuffled, shuffled + rthread_total[tid], T(1));
        // else
        //     c2(tid, shuffled, shuffled + rthread_total[tid]); // this->c[tid].insert_no_estimate(shuffled, shuffled + rthread_total[tid], T(1));

        after = this->c[tid].size();

        //printf("rank %d of %d, thread %d of %d before %ld after count %ld\n", comm_rank, comm_size, tid, tcnt, before, after);

        if (tcnt > 1) {
            ::utils::mem::aligned_free(shuffled);
        }

    } // parallel modify.
    BL_BENCH_END(modify, "modify", after);

#ifdef MT_DEBUG
        {
        // compare to single threaded
        bool same = true;
        auto iit2 = test_distributed;
        int i = 0;
        for (auto iit = distributed; iit != distributed + recv_total; ++iit, ++iit2, ++i) {
        	same &= (*iit == *iit2);
        	if (!same) break;
        }
        printf("rank %d ALL THREADS distribution same ? %s : %d\n",
        		this->comm.rank(), (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_distributed);

#endif


    BL_BENCH_START(modify);
	::utils::mem::aligned_free(distributed);
    BL_BENCH_END(modify, "clean up", recv_total);

#endif // non overlap

    BL_BENCH_REPORT_MPI_NAMED(modify, "hashmap:modify_p", this->comm);

    return static_cast<int64_t>(after) - static_cast<int64_t>(before);
  }




    public:
      /**
       * @brief insert new elements in the hybrid batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

            auto insert_functor = [this](int tid, ::std::pair<Key, T> * it, ::std::pair<Key, T> * et, bool est = true) {
                if (est)
                    this->c[tid].insert(it, et);
                else
                    this->c[tid].insert_no_estimate(it, et);
            };
            auto insert_no_est_functor = [this](int tid, ::std::pair<Key, T> * it, ::std::pair<Key, T> * et) {
                // don't call the estimate
                this->c[tid].insert_no_estimate(it, et);
            };


    	  if (this->comm.size() == 1) {
    		  return this->template modify_1<estimate>(input, insert_functor);
    	  } else {
    		  return this->template modify_p<estimate>(input, insert_functor, insert_no_est_functor);
    	  }
      }

    protected:


      /**
       * @brief query new elements in the hybrid batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename V, typename OP>
      void query_1(std::vector<Key> & input,
    		  V* results,
              OP compute) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(query);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(query, "base_batched_robinhood_map:query", this->comm);
          return;
        }

        BL_BENCH_START(query);

        // get some common variables
        size_t in_size = input.size();
        size_t nthreads_global = omp_get_max_threads();
        int comm_size = 1;
	size_t batch_size = InternalHash::batch_size;

        // set up per thread storage
        std::vector< std::vector<size_t> > thread_bucket_sizes(omp_get_max_threads());
        std::vector< std::vector<size_t> > thread_bucket_offsets(omp_get_max_threads());
        std::vector<size_t> node_bucket_sizes(nthreads_global, 0);
        std::vector<size_t> node_bucket_offsets(nthreads_global, 0);
        std::vector<size_t> thread_total(omp_get_max_threads(), 0);

        BL_BENCH_END(query, "alloc", nthreads_global);

#ifdef MT_DEBUG
    Key *transformed = ::utils::mem::aligned_alloc<Key>(input.size(), 64);

    this->transform_input(input.begin(), input.end(), transformed);


    std::vector<size_t> test_sizes(nthreads_global, 0);
    uint32_t *test_bids = ::utils::mem::aligned_alloc<uint32_t>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint8_t>(nthreads_global),
         test_sizes, reinterpret_cast<uint8_t*>(test_bids) );
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint16_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint16_t*>(test_bids) );
    } else {   // mpi supports only 31 bit worth of ranks.
        this->assign_count(transformed, transformed + input.size(), static_cast<uint32_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint32_t*>(test_bids) );
    }
    // above is same.

    std::vector<size_t> test_offsets(nthreads_global, 0);
    for (size_t i = 1; i < nthreads_global; ++i) {
    	test_offsets[i] = test_offsets[i-1]+test_sizes[i-1];
    }
    Key* test_res = ::utils::mem::aligned_alloc<Key>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint8_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint16_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint32_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    }

    ::utils::mem::aligned_free(transformed);
#endif


        BL_BENCH_START(query);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int tcnt = omp_get_num_threads();

            size_t block = in_size / tcnt;
            size_t rem = in_size % tcnt;
            size_t r_start = block * tid + std::min(rem, static_cast<size_t>(tid));
            block += (static_cast<size_t>(tid) < rem ? 1: 0);
            size_t r_end = r_start + block;

            auto it = input.data() + r_start;
            auto et = input.data() + r_end;
            auto rt = results;

            // transform once.  bucketing and distribute will read it multiple times.
            if (tcnt > 1) { // only do if more than 1 thread.  then we need to permute
                // require permuting.  leave the input in the permuted order after.
                Key* buffer = ::utils::mem::aligned_alloc<Key>(r_end - r_start + batch_size);

                // transform once.  bucketing and distribute will read it multiple times.
                this->transform_input(it, et, buffer);

                // ===== assign to get bucket ids.
                thread_bucket_sizes[tid].resize(nthreads_global, 0);
                uint32_t* bid_buf = ::utils::mem::aligned_alloc<uint32_t>(r_end - r_start + batch_size);

                if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                    this->assign_count(buffer, buffer + block, static_cast<uint8_t>(nthreads_global),
                    thread_bucket_sizes[tid], reinterpret_cast<uint8_t*>(bid_buf) );
                } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                    this->assign_count(buffer, buffer + block, static_cast<uint16_t>(nthreads_global),
                    thread_bucket_sizes[tid], reinterpret_cast<uint16_t*>(bid_buf) );
                } else {   // mpi supports only 31 bit worth of ranks.
                    this->assign_count(buffer, buffer + block, static_cast<uint32_t>(nthreads_global),
                    thread_bucket_sizes[tid], reinterpret_cast<uint32_t*>(bid_buf) );
                }
                thread_bucket_offsets[tid].resize(nthreads_global, 0);

#ifdef MT_DEBUG   //verified same.
            {
            	bool same = true;
				int i = 0;
	            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {

					uint8_t* bit = reinterpret_cast<uint8_t*>(test_bids) + r_start;
					uint8_t* bet = reinterpret_cast<uint8_t*>(test_bids) + r_end;
					uint8_t* bit2 = reinterpret_cast<uint8_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
					uint16_t* bit = reinterpret_cast<uint16_t*>(test_bids) + r_start;
					uint16_t* bet = reinterpret_cast<uint16_t*>(test_bids) + r_end;
					uint16_t* bit2 = reinterpret_cast<uint16_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else {   // mpi supports only 31 bit worth of ranks.
					uint32_t* bit = reinterpret_cast<uint32_t*>(test_bids) + r_start;
					uint32_t* bet = reinterpret_cast<uint32_t*>(test_bids) + r_end;
					uint32_t* bit2 = reinterpret_cast<uint32_t*>(bid_buf);
					printf("thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            }
				printf("thread %d bucket ids same ? %s : %d\n", tid, (same ? "yes":"no"), i );
            }
#endif


                #pragma omp barrier

                // ===== now compute the offsets 
                // exclusive scan of everyhing.  proceed in 3 step
                //  1. thread local scan for each bucket, each thread do a block of comm_size buckets, across all thread., store back into each thread's thread_bucket_offsets
                //     also store count for each bucket
                size_t offset = 0;
                for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
                    node_bucket_offsets[j] = offset;  // exscan within block for this thread
                    // iterate over a block of bucket sizes.
                    for (int t = 0; t < tcnt; ++t) {
                        // iterate over each thread's data
                        thread_bucket_offsets[t][j] = offset;     // exscan within block for this thread
                        offset += thread_bucket_sizes[t][j];
                    }
                    // store the bucket count
                    node_bucket_sizes[j] = offset - node_bucket_offsets[j];
                }   // when finished, has per bucket offsets for each thread, and total 
                // when finished, has per bucket offsets for each thread, and total 
                thread_total[tid] = offset;   // save the total element count, for prefix scan next and update.
                // wait for all to be done with the scan.
                #pragma omp barrier

				//  2. 1 thread to scan thread total to get thread offsets.  O(C) instead of O(CP)
				#pragma omp single
				{
					size_t curr = thread_total[0];
					size_t sum = 0;
					for (int t = 1; t < tcnt; ++t) {
						thread_total[t - 1] = sum;
						sum += curr;
						curr = thread_total[t];
					}
					thread_total[tcnt - 1] = sum;
				}

				// wait for all to finish
				#pragma omp barrier
				//  3. thread local update of per thread offsets.
				offset = thread_total[tid];
				for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
					// update the per thread prefix scan to node prefix scan
					node_bucket_offsets[j] += offset;
					for (int t = 0; t < tcnt; ++t) {
						// iterate over each thread's data, update offsets
						thread_bucket_offsets[t][j] += offset;
					}
				}
                    //===============  now the offsets are ready.  we can permute.
	            // since each thread updates a portion of every other thread's thread buckewt sizes, need this barrier here.
	   			#pragma omp barrier

                    // then call permute_by_bucketId and computed offsets. 
                    // NOTE: permute back into input
				if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                    this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint8_t*>(bid_buf),
                    thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
                } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                    this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint16_t*>(bid_buf),
                    thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
                } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
                    this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint32_t*>(bid_buf),
                    thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
                }

                ::utils::mem::aligned_free(bid_buf);
                ::utils::mem::aligned_free(buffer);

                #pragma omp barrier

                // update it and et.
                it = input.data() + node_bucket_offsets[tid];
                et = it + node_bucket_sizes[tid];
                rt = results + node_bucket_offsets[tid];

#ifdef MT_DEBUG
#pragma omp barrier
            #pragma omp single
            {
            	printf("node offests from thread %d", tid);
            	for (int j = 0; j < nthreads_global; ++j) {
            		printf("\t[%ld, %ld)", node_bucket_offsets[j], node_bucket_offsets[j] + node_bucket_sizes[j] );
            	}
            	printf("\n");

            }
			#pragma omp barrier
#endif
            } else {
                this->transform_input(it, et, it);
                rt = results + r_start;
            }

#ifdef MT_DEBUG
        {
			// compare to single threaded
			bool same = true;
			auto iit2 = test_res + node_bucket_offsets[tid];
			int i = 0;
			for (auto iit = it; iit != et; ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("thread %d permutation same ? %s : %d\n", tid, (same ? "yes":"no"), i );
        }
#endif

            compute(tid, it, et, rt);

        } // end parallel section
        BL_BENCH_END(query, "local_find", in_size);

#ifdef MT_DEBUG
        {
        // compare to single threaded
        bool same = true;
        auto iit2 = test_res;
        int i = 0;
        for (auto iit = input.begin(); iit != input.end(); ++iit, ++iit2, ++i) {
        	same &= (*iit == *iit2);
        	if (!same) break;
        }
        printf("ALL THREADS permutation same ? %s : %d\n", (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_res);
        ::utils::mem::aligned_free(test_bids);

#endif

        BL_BENCH_REPORT_MPI_NAMED(query, "base_hashmap:query", this->comm);

        return;
      }


      /**
       * @brief query new elements in the hybrid batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename V, typename OP>
      void query_p(std::vector<Key >& input,
    		  V * results,
              OP compute) const {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(query);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(query, "hashmap:query", this->comm);
          return;
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
        // query
        // return results
        // free buffer

        BL_BENCH_START(query);

        // get some common variables
        int comm_size = this->comm.size();
        size_t in_size = input.size();
        size_t nthreads_global = comm_size * omp_get_max_threads();
	size_t batch_size = InternalHash::batch_size;

        // set up per thread storage
        std::vector< std::vector<size_t> > thread_bucket_sizes(omp_get_max_threads());
        std::vector< std::vector<size_t> > thread_bucket_offsets(omp_get_max_threads());
        std::vector<size_t> node_bucket_sizes(nthreads_global, 0);
        std::vector<size_t> node_bucket_offsets(nthreads_global, 0);
        std::vector<size_t> thread_total(omp_get_max_threads(), 0);

        BL_BENCH_END(query, "alloc", nthreads_global);


#ifdef MT_DEBUG
    Key *transformed = ::utils::mem::aligned_alloc<Key>(input.size(), 64);

    this->transform_input(input.begin(), input.end(), transformed);


    std::vector<size_t> test_sizes(nthreads_global, 0);
    uint32_t *test_bids = ::utils::mem::aligned_alloc<uint32_t>(input.size(), 64);

    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint8_t>(nthreads_global),
         test_sizes, reinterpret_cast<uint8_t*>(test_bids) );
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->assign_count(transformed, transformed + input.size(), static_cast<uint16_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint16_t*>(test_bids) );
    } else {   // mpi supports only 31 bit worth of ranks.
        this->assign_count(transformed, transformed + input.size(), static_cast<uint32_t>(nthreads_global),
        		test_sizes, reinterpret_cast<uint32_t*>(test_bids) );
    }
    // above is same.

    std::vector<size_t> test_offsets(nthreads_global, 0);
    for (size_t i = 1; i < nthreads_global; ++i) {
    	test_offsets[i] = test_offsets[i-1]+test_sizes[i-1];
    }
    Key* test_res = ::utils::mem::aligned_alloc<Key>(input.size(), 64);
    if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint8_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint16_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
        this->permute_by_bucketid(transformed, transformed + input.size(), reinterpret_cast<uint32_t*>(test_bids),
        		test_sizes, test_offsets, test_res);
    }

    ::utils::mem::aligned_free(transformed);

    std::vector<size_t> rtest_sizes(nthreads_global, 0);
    mxx::all2all(test_sizes.data(), omp_get_max_threads(), rtest_sizes.data(), this->comm);

    std::vector<size_t> test_sendcounts(this->comm.size(), 0);
    std::vector<size_t> test_recvcounts(this->comm.size(), 0);
    std::vector<size_t> test_recvcounts2(this->comm.size(), 0);
    size_t test_recv_total = 0;
    for (size_t i = 0; i < this->comm.size(); ++i) {
    	for (size_t j = 0; j < omp_get_max_threads(); ++j) {
    		test_sendcounts[i] += test_sizes[i * omp_get_max_threads() + j];
    		test_recvcounts[i] += rtest_sizes[i * omp_get_max_threads() + j];
    		test_recv_total += rtest_sizes[i * omp_get_max_threads() + j];
    	}
    }

    // send the test data.
    Key* test_distributed = ::utils::mem::aligned_alloc<Key>(test_recv_total);

    ::khmxx::distribute_permuted(test_res, test_res + input.size(),
                test_sendcounts, test_distributed, test_recvcounts, this->comm);


    // assert that test_recvcounts and test_recvcounts2 are the same.
//    {
//    	bool same = true;
//    	for (size_t i = 0; i < omp_get_max_threads(); ++i) {
//    		same &= test_recvcounts[i] == test_recvcounts2[i];
//    		if (!same) break;
//    	}
//    	printf("rank %d recv counts via permute same? %s \n", this->comm.rank(), (same ? "yes" : "no"));
//    }


#endif


    BL_BENCH_COLLECTIVE_START(query, "permute", this->comm);
    
    // THREADED code to permute.  note that the partitioning is even, per thread memory access is random within ranges of data, 
    // and there should be no contention or fine grain synchronization.

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tcnt = omp_get_num_threads();

        size_t block = in_size / tcnt;
        size_t rem = in_size % tcnt;
        size_t r_start = block * tid + std::min(rem, static_cast<size_t>(tid));
        block += (static_cast<size_t>(tid) < rem ? 1: 0);
        size_t r_end = r_start + block;


      // get mapping to proc
      // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
        Key* buffer = ::utils::mem::aligned_alloc<Key>(r_end - r_start + batch_size);

//        BL_BENCH_END(modify, "alloc", input.size());
        auto it = input.data() + r_start;
        auto et = input.data() + r_end;

	        // transform once.  bucketing and distribute will read it multiple times.
//	        BL_BENCH_COLLECTIVE_START(modify, "transform", this->comm);
        this->transform_input(it, et, buffer);
//	        BL_BENCH_END(modify, "transform", input.size());


// NOTE: overlap comm is incrementally inserting so we estimate before transmission, thus global estimate, and 64 bit
//            hash is needed.
//       non-overlap comm can estimate just before insertion for more accurate estimate based on actual input received.
//          so 32 bit hashes are sufficient.
            // count and estimate and save the bucket ids.
//    BL_BENCH_COLLECTIVE_START(modify, "permute_estimate", this->comm);
        // allocate an HLL
        // allocate the bucket sizes array

        thread_bucket_sizes[tid].resize(nthreads_global, 0);
    
        uint32_t* bid_buf = ::utils::mem::aligned_alloc<uint32_t>(r_end - r_start + batch_size);

        if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
            this->assign_count(buffer, buffer + block, static_cast<uint8_t>(nthreads_global), thread_bucket_sizes[tid],
                        reinterpret_cast<uint8_t*>(bid_buf) );
        } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
            this->assign_count(buffer, buffer + block, static_cast<uint16_t>(nthreads_global), thread_bucket_sizes[tid],
                    reinterpret_cast<uint16_t*>(bid_buf) );
        } else {   // mpi supports only 31 bit worth of ranks.
                this->assign_count(buffer, buffer + block, static_cast<uint32_t>(nthreads_global), thread_bucket_sizes[tid],
                        reinterpret_cast<uint32_t*>(bid_buf) );
        }
        // do some calc with thread_bucket_sizes to get offsets for each bucket for each thread in node-wide permuted input array.
        thread_bucket_offsets[tid].resize(nthreads_global, 0);
        
#ifdef MT_DEBUG   //verified same bucket assignment
            {
            	bool same = true;
				int i = 0;
	            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {

					uint8_t* bit = reinterpret_cast<uint8_t*>(test_bids) + r_start;
					uint8_t* bet = reinterpret_cast<uint8_t*>(test_bids) + r_end;
					uint8_t* bit2 = reinterpret_cast<uint8_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
					uint16_t* bit = reinterpret_cast<uint16_t*>(test_bids) + r_start;
					uint16_t* bet = reinterpret_cast<uint16_t*>(test_bids) + r_end;
					uint16_t* bit2 = reinterpret_cast<uint16_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            } else {   // mpi supports only 31 bit worth of ranks.
					uint32_t* bit = reinterpret_cast<uint32_t*>(test_bids) + r_start;
					uint32_t* bet = reinterpret_cast<uint32_t*>(test_bids) + r_end;
					uint32_t* bit2 = reinterpret_cast<uint32_t*>(bid_buf);
					printf("rank %d thread %d bucket ids gold: %d, %d, %d, %d, %d, %d, %d, %d.  computed %d, %d, %d, %d, %d, %d, %d, %d\n",
							this->comm.rank(), tid, *bit, *(bit + 1), *(bit + 2), *(bit + 3), *(bit + 4), *(bit + 5), *(bit + 6), *(bit + 7),
							*bit2, *(bit2 + 1), *(bit2 + 2), *(bit2 + 3), *(bit2 + 4), *(bit2 + 5), *(bit2 + 6), *(bit2 + 7));
					for (; bit != bet; ++bit, ++bit2, ++i) {
						same &= (*bit == *bit2);
						if (!same) break;
					}
	            }
				printf("rank %d thread %d bucket ids same ? %s : %d\n",
						this->comm.rank(), tid, (same ? "yes":"no"), i );
            }
#endif


        // make sure all threads have reached here
        #pragma omp barrier

        // exclusive scan of everyhing.  proceed in 3 step
        //  1. thread local scan for each bucket, each thread do a block of comm_size buckets, across all thread., store back into each thread's thread_bucket_offsets
        //     also store count for each bucket
        size_t offset = 0;
        for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
            node_bucket_offsets[j] = offset;  // exscan within block for this thread
            // iterate over a block of bucket sizes.
            for (int t = 0; t < tcnt; ++t) {
                // iterate over each thread's data
                thread_bucket_offsets[t][j] = offset;     // exscan within block for this thread
                offset += thread_bucket_sizes[t][j];
            }
            // store the bucket count
            node_bucket_sizes[j] = offset - node_bucket_offsets[j];
        }   // when finished, has per bucket offsets for each thread, and total 
        thread_total[tid] = offset;   // save the total element count, for prefix scan next and update.

        if (tcnt > 1) {  // update offsets if more than 1 thread.
            // wait for all to be done with the scan.
            #pragma omp barrier
            //  2. 1 thread to scan thread total to get thread offsets.  O(C) instead of O(CP)
            #pragma omp single
            {
                size_t curr = thread_total[0];
                size_t sum = 0;
                for (int t = 1; t < tcnt; ++t) {
                    thread_total[t - 1] = sum;
                    sum += curr;
                    curr = thread_total[t];
                }
                thread_total[tcnt - 1] = sum;
            }

            // wait for all to finish
            #pragma omp barrier
            //  3. thread local update of per thread offsets.
            offset = thread_total[tid];
            for (size_t j = tid * comm_size, jmax = j + comm_size; j < jmax; ++j) {
                // update the per thread prefix scan to node prefix scan
                node_bucket_offsets[j] += offset;
                for (int t = 0; t < tcnt; ++t) {
                    // iterate over each thread's data, update offsets
                    thread_bucket_offsets[t][j] += offset;
                }
            }
        }
        //===============  now the offsets are ready.  we can permute.
#pragma omp barrier

        // then call permute_by_bucketId and computed offsets.
            if (nthreads_global <= std::numeric_limits<uint8_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint8_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint16_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint16_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            } else if (nthreads_global <= std::numeric_limits<uint32_t>::max()) {
                this->permute_by_bucketid(buffer, buffer + block, reinterpret_cast<uint32_t*>(bid_buf),
                 thread_bucket_sizes[tid], thread_bucket_offsets[tid], input.data());
            }

        // merge hll in binary fashion.

        ::utils::mem::aligned_free(bid_buf);
        ::utils::mem::aligned_free(buffer);

#ifdef MT_DEBUG
        {
			#pragma omp barrier

			// update it and et.
			it = input.data() + node_bucket_offsets[tid];
			et = it + node_bucket_sizes[tid];

			// compare to single threaded
			bool same = true;
			auto iit2 = test_res + node_bucket_offsets[tid];
			int i = 0;
			for (auto iit = it; iit != et; ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("rank %d thread %d permutation same ? %s : %d\n",
					this->comm.rank(), tid, (same ? "yes":"no"), i );
        }
#endif


    }  // omp parallel for permuting.  done.

    BL_BENCH_END(query, "permute", input.size());

#ifdef MT_DEBUG
		{
			printf("rank %d node offests from ALL THREADS ", this->comm.rank());
			for (int j = 0; j < nthreads_global; ++j) {
				printf("\t[%ld, %ld)", node_bucket_offsets[j], node_bucket_offsets[j] + node_bucket_sizes[j] );
			}
			printf("\n");

		}

        {
			// compare to single threaded
			bool same = true;
			auto iit2 = test_res;
			int i = 0;
			for (auto iit = input.begin(); iit != input.end(); ++iit, ++iit2, ++i) {
				same &= (*iit == *iit2);
				if (!same) break;
			}
			printf("rank %d ALL THREADS permutation same ? %s : %d\n",
					this->comm.rank(), (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_res);
        ::utils::mem::aligned_free(test_bids);
#endif

  	BL_BENCH_COLLECTIVE_START(query, "a2a_count", this->comm);

    // send off the node_bucket_sizes - for per recv thread traversal.
    std::vector<size_t> rnode_bucket_sizes(nthreads_global, 0);
    std::vector<size_t> rnode_bucket_offsets(nthreads_global, 0);
    mxx::all2all(node_bucket_sizes.data(), omp_get_max_threads(), rnode_bucket_sizes.data(), this->comm);

    // now compute the send_counts
    // single thread to do this, so don't have to worry about comm_size / tcnt not being even.
    std::vector<size_t> send_counts(this->comm.size());
    std::vector<size_t> recv_counts(this->comm.size());

    std::vector<size_t> rthread_total(omp_get_max_threads(), 0);


    size_t send_cnt, recv_cnt, recv_offset = 0;
    int jmax =  omp_get_max_threads();
    for (int i = 0; i < this->comm.size(); ++i) {
        send_cnt = 0;
        recv_cnt = 0;
#if defined(OVERLAPPED_COMM)
        // if overlap, then we reset the offset for each rank-pair's message.
        recv_offset = 0;  
#endif
        for (int j = 0; j < jmax; ++j) {
            // get send count and recv count
            send_cnt += node_bucket_sizes[i * jmax + j];
            recv_cnt += rnode_bucket_sizes[i * jmax + j];

            // compute recv bucket offsets.
            rnode_bucket_offsets[i * jmax + j] = recv_offset;
            recv_offset += rnode_bucket_sizes[i * jmax + j];
            
            // per thread total recv
            rthread_total[j] += rnode_bucket_sizes[i * jmax + j];
        }
        send_counts[i] = send_cnt;
        recv_counts[i] = recv_cnt;
    }
    BL_BENCH_END(query, "a2a_count", recv_counts.size());


#if defined(OVERLAPPED_COMM) 

    BL_BENCH_COLLECTIVE_START(query, "a2av_query", this->comm);

    ::khmxx::incremental::ialltoallv_and_query_one_to_one(
        input.data(), input.data() + input.size(), send_counts,
        [this, &rnode_bucket_offsets, &rnode_bucket_sizes, &compute](int rank, 
                                                        Key* b, Key* e, V* r){
            // compute just the offsets within a thread block.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int tcnt = omp_get_num_threads();
                Key* bb = b + rnode_bucket_offsets[rank * tcnt + tid];
                Key* ee = bb + rnode_bucket_sizes[rank * tcnt + tid];
                V* rr = r + rnode_bucket_offsets[rank * tcnt + tid];

                compute(tid, bb, ee, rr); // this->c[tid].insert_no_estimate(bb, ee, T(1));
            }  // finished parallel modify.
        },
        results,
        this->comm);

    BL_BENCH_END(query, "a2av_query", input.size());

#else
    size_t recv_total = recv_offset;  // for overlap, this would be incorrect but that is okay.

    BL_BENCH_START(query);
    Key* distributed = ::utils::mem::aligned_alloc<Key>(recv_total + batch_size);
    V* dist_results = ::utils::mem::aligned_alloc<V>(recv_total + batch_size);
    BL_BENCH_END(query, "alloc_intermediates", recv_total);

    BL_BENCH_COLLECTIVE_START(query, "a2a", this->comm);
    ::khmxx::distribute_permuted(input.data(), input.data() + input.size(),
            send_counts, distributed, recv_counts, this->comm);
    BL_BENCH_END(query, "a2av", input.size());



    BL_BENCH_COLLECTIVE_START(query, "query", this->comm);
    // local compute part.  called by the communicator.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tcnt = omp_get_num_threads();

        //======= no shuffling, to avoid 2 memcopies.

        // local compute part.  called by the communicator.
        Key * it; 
        V * rt; 
        for (int i = 0; i < this->comm.size(); ++i) {
            it = distributed + rnode_bucket_offsets[i * tcnt + tid];
            rt = dist_results + rnode_bucket_offsets[i * tcnt + tid];
            compute(tid, it,
              it + rnode_bucket_sizes[i * tcnt + tid], rt); 
        }
    } // parallel query.
    BL_BENCH_END(query, "query", recv_total);

#ifdef MT_DEBUG
        {
        // compare to single threaded
        bool same = true;
        auto iit2 = test_distributed;
        int i = 0;
        for (auto iit = distributed; iit != distributed + recv_total; ++iit, ++iit2, ++i) {
        	same &= (*iit == *iit2);
        	if (!same) break;
        }
        printf("rank %d ALL THREADS distribution same ? %s : %d\n",
        		this->comm.rank(), (same ? "yes":"no"), i );
        }

        // free memory.
        ::utils::mem::aligned_free(test_distributed);

#endif


    BL_BENCH_START(query);
    ::utils::mem::aligned_free(distributed);
    BL_BENCH_END(query, "cleanup", recv_total);

    // local query. memory utilization a potential problem.
    // do for each src proc one at a time.

    // send back using the constructed recv count
    BL_BENCH_COLLECTIVE_START(query, "a2a2", this->comm);
    ::khmxx::distribute_permuted(dist_results, dist_results + recv_total,
            recv_counts, results, send_counts, this->comm);

    ::utils::mem::aligned_free(dist_results);

    BL_BENCH_END(query, "a2a2", input.size());


#endif // non overlap

    BL_BENCH_REPORT_MPI_NAMED(query, "hashmap:query_p", this->comm);

    return;
    }



    public:

      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<count_result_type > count(::std::vector<Key>& keys,
    		  bool sorted_input = false,
			  Predicate const& pred = Predicate() ) const {

        std::vector<count_result_type> results(keys.size());

        auto count_functor =
            [this, &pred](int tid, Key* b, Key* e, count_result_type * out) {
  	            this->c[tid].count(out, b, e, pred, pred);
            };

        if (this->comm.size() == 1) {
            query_1(keys, results.data(), count_functor);
        } else {
            query_p(keys, results.data(), count_functor);
        }

        return results;
      }

      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      size_t count(::std::vector<Key>& keys, count_result_type* results,
    		  bool sorted_input = false,
			  Predicate const& pred = Predicate() ) const {
        
        auto count_functor =  
            [this, &pred](int tid, Key* b, Key* e, count_result_type * out) {
  	            this->c[tid].count(out, b, e, pred, pred);
            };

        if (this->comm.size() == 1) {
            query_1(keys, results, count_functor);
        } else {
            query_p(keys, results, count_functor);
        }

        return keys.size();
      }


//      template <typename Predicate = ::bliss::filter::TruePredicate>
//      ::std::vector<::std::pair<Key, size_type> > count(Predicate const & pred = Predicate()) const {
//
//        ::std::vector<::std::pair<Key, size_type> > results = this->c.count(pred);
//
//        if (this->comm.size() > 1) this->comm.barrier();
//        return results;
//      }


      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<mapped_type > find(::std::vector<Key>& keys,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			  Predicate const& pred = Predicate() ) const {

        std::vector<mapped_type> results(keys.size());
        
        auto find_functor =  
            [this, &pred, &nonexistent](int tid, Key* b, Key* e, mapped_type * out) {
  	            this->c[tid].find(out, b, e, nonexistent, pred, pred);
            };

        if (this->comm.size() == 1) {
            query_1(keys, results.data(), find_functor);
        } else {
            query_p(keys, results.data(), find_functor);
        }

        return results;
      }


      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      size_t find(::std::vector<Key>& keys, mapped_type * results,
    		  mapped_type const & nonexistent = mapped_type(),
    		  bool sorted_input = false,
			  Predicate const& pred = Predicate() ) const {

        auto find_functor =
            [this, &pred, &nonexistent](int tid, Key* b, Key* e, mapped_type * out) {
  	            this->c[tid].find(out, b, e, nonexistent, pred, pred);
            };

        if (this->comm.size() == 1) {
            query_1(keys, results, find_functor);
        } else {
            query_p(keys, results, find_functor);
        }

        return keys.size();
      }


#if 0  // TODO: temporarily retired.
      /**
       * @brief find elements with the specified keys in the hybrid batched_robinhood_multimap.
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
  ::khmxx::lz4::distribute(keys, this->key_to_rank2, recv_counts, buffer, this->comm);
#else
  ::khmxx::distribute(keys, this->key_to_rank2, recv_counts, buffer, this->comm);
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


      /**
       * @brief erase elements with the specified keys in the hybrid batched_robinhood_multimap.
       * @param first
       * @param last
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase(std::vector<Key>& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

        auto erase_functor = [this, &pred](int tid, Key * it, Key * et, bool est = false){
            this->c[tid].erase(it, et, pred, pred);
        };
        auto erase_functor2 = [this, &pred](int tid, Key * it, Key * et){
            this->c[tid].erase(it, et, pred, pred);
        };

        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  if (this->comm.size() == 1) {
              // single node, possibly multi-thread.
    		  return this->template modify_1<false>(input, erase_functor);
    	  } else {
    		  return this->template modify_p<false>(input, erase_functor, erase_functor2);
    	  }
      }

      // ================  overrides


      template <typename SERIALIZER>
      size_t serialize(unsigned char * out, SERIALIZER const & kvs) const {
        size_t bytes = 0;

            size_t out_elem_size = sizeof(Key) + sizeof(T);
            vector<size_t> offsets(omp_get_max_threads());

            #pragma omp parallel reduction(+ : bytes)
            {
                int tid = omp_get_thread_num();
                int tcnt = omp_get_num_threads();

                // get the sizes
                bytes = offsets[tid] = this->c[tid].size();
                bytes *= out_elem_size;

                // compute the offsets
                #pragma omp barrier
                #pragma omp single 
                {
                    size_t sum = 0;
                    size_t curr = offsets[0];
                    for (int i = 1; i < tcnt; ++i) {
                        offsets[i-1] = sum;
                        sum += curr;
                        curr = offsets[i];
                    }
                    offsets[tcnt - 1] = sum;
                }

                // get the starting position of the output
                unsigned char * data = out + bytes;

                // now serialize
                auto it_end = this->c[tid].cend();
                for (auto it = this->c[tid].cbegin(); it != it_end; ++it) {
				    data = kvs(*it, data);
			    }

            }  // end parallel section
            return bytes;

      }
  };


  /**
   * @brief  hybrid robinhood map following std robinhood map's interface.
   * @details   This class is modeled after the hashmap_batched_robinhood_doubling_offsets.
   *         it has as much of the same methods of hashmap_batched_robinhood_doubling_offsets as possible.  however, all methods consider the fact
   *         that the data are in distributed memory space, so to access the data, "communication" is needed.
   *
   *         Note that "communication" is a weak concept here meaning that we are accessing a different local container.
   *         as such, communicator may be defined for MPI, UPC, OpenMP, etc.
   *
   *         This allows the possibility of using hybrid robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local hybrid robinhood map, or it may be done via sorting/lookup or other mapping
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
   * @brief  hybrid robinhood reduction map following std robinhood map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
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
   *         This allows the possibility of using hybrid robinhood map as local storage for coarser grain distributed container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local hybrid robinhood map, or it may be done via sorting/lookup or other mapping
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
   * @brief  hybrid robinhood counting map following std robinhood map's interface.  Insertion applies the binary reduction operator between the existing and inserted element (in that order).
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
   *         This allows the possibility of using hybrid robinhood map as local storage for coarser grain hybrid container.
   *
   *         Note that communicator requires a mapping strategy between a key and the target processor/thread/partition.  The mapping
   *         may be done using a hash, similar to the local hybrid robinhood map, or it may be done via sorting/lookup or other mapping
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
  	  MapParams, ::std::plus<T>, Alloc> {
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



    using Base::modify_1;
    using Base::modify_p;


public:

      /**
       * @brief insert new elements in the hybrid batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <bool estimate = true, typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

        auto insert_key_functor = [this](int tid, Key * it, Key * et, bool est = true){
#ifdef MT_DEBUG
           size_t before = this->c[tid].size();
#endif
            if (estimate)
                this->c[tid].insert(it, et, T(1));
            else
                this->c[tid].insert_no_estimate(it, et, T(1));
#ifdef MT_DEBUG
            printf("rank %d of %d thread %d inserting %ld, before %ld, after %ld\n", this->comm.rank(), this->comm.size(), tid, std::distance(it, et), before, this->c[tid].size());
#endif
        };
        auto insert_key_no_est_functor = [this](int tid, Key * it, Key * et){
#ifdef MT_DEBUG
          size_t before = this->c[tid].size();
#endif
            this->c[tid].insert_no_estimate(it, et, T(1));
#ifdef MT_DEBUG
            printf("rank %d of %d thread %d inserting %ld, before %ld, after %ld\n", this->comm.rank(), this->comm.size(), tid, std::distance(it, et), before, this->c[tid].size());
#endif
        };
        

        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  if (this->comm.size() == 1) {
              // single node, possibly multi-thread.
    		  return this->template modify_1<estimate>(input, insert_key_functor);
    	  } else {
    		  return this->template modify_p<estimate>(input, insert_key_functor, insert_key_no_est_functor);
    	  }
      }

  };


} /* namespace hsc */


#endif // HYBRID_BATCHED_ROBINHOOD_MAP_HPP
