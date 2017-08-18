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


#include "kmerhash/robinhood_offset_hashmap.hpp"  // local storage hash table  // for multimap
#include <utility> 			  // for std::pair

//#include <sparsehash/dense_hash_map>  // not a multimap, where we need it most.
#include <functional> 		// for std::function and std::hash
#include <algorithm> 		// for sort, stable_sort, unique, is_sorted
#include <iterator>  // advance, distance
#include <sstream>  // stringstream for filea
#include <cstdint>  // for uint8, etc.

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
  template <typename, typename, typename, typename, typename...> class Container,
  template <typename> class MapParams,
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >,
	typename Reducer = ::fsc::DiscardReducer
  >
  class batched_robinhood_map_base :
		  public ::dsc::map_base<Key, T, MapParams, Alloc> {

    protected:
      using Base = ::dsc::map_base<Key, T, MapParams, Alloc>;

//      using TransformedHash = ::fsc::TransformedHash<Key, Hash<Key, false>, KeyTransform>;
//      TransformedHash hash;

      struct KeyToRank {
          typename Base::DistTransformedFunc proc_trans_hash;
          const int p;

          // 2x comm size to allow more even distribution?
          KeyToRank(int comm_size) :
        	  proc_trans_hash(typename Base::DistFunc(ceilLog2(comm_size)),
        			  	  	  typename Base::DistTrans()),
        			  p(comm_size) {};

          inline int operator()(Key const & x) const {
            //            printf("KeyToRank operator. commsize %d  key.  hashed to %d, mapped to proc %d \n", p, proc_hash(Base::trans(x)), proc_hash(Base::trans(x)) % p);
            return proc_trans_hash(x) % p;
          }
          template<typename V>
          inline int operator()(::std::pair<Key, V> const & x) const {
            return this->operator()(x.first);
          }
          template<typename V>
          inline int operator()(::std::pair<const Key, V> const & x) const {
            return this->operator()(x.first);
          }
      } key_to_rank;


//      /**
//       * @brief count elements with the specified keys in the distributed sorted_multimap.
//       * @note  input cannot have duplicate elements.
//       *
//       * @param first
//       * @param last
//       */
//      struct QueryProcessor {  // assume unique, always.
//
//          // assumes that container is sorted. and exact overlap region is provided.  do not filter output here since it's an output iterator.
//          template <class DB, class QueryIter, class OutputIter, class Operator, class Predicate = ::bliss::filter::TruePredicate>
//          static size_t process(DB &db,
//                                QueryIter query_begin, QueryIter query_end,
//                                OutputIter &output, Operator & op,
//                                bool sorted_query = false, Predicate const &pred = Predicate()) {
//
//              if (query_begin == query_end) return 0;
//
//              size_t count = 0;  // before size.
//              if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value)
//                for (auto it = query_begin; it != query_end; ++it) {
//                  count += op(db, *it, output, pred);
//                }
//              else
//                for (auto it = query_begin; it != query_end; ++it) {
//                  count += op(db, *it, output);
//                }
//              return count;
//          }
//
//      };




    public:
      using local_container_type = Container<Key, T,
    		  typename Base::StoreTransformedFunc,
    		  typename Base::StoreTransformedEqual,
    		  Alloc, Reducer>;

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

        BL_BENCH_REPORT_MPI_NAMED(reduce_tuple, "reduction_hashmap:local_reduce", this->comm);
      }


      /// performs transform and bucketing using key_to_rank.
      /// permutes input array in place, and returns bucket counts
      /// appropriate when the input needs to be permuted (count, exists) to match results.
      // for multiple processes only
      template <typename IT>
      std::vector<size_t> transform_and_bucket(IT _begin, IT _end) {

    	  assert((this->comm.size() > 1) && "this function is for comm size larger than 1 only.");

        BL_BENCH_INIT(transform_bucket);

        using V = typename ::std::iterator_traits<IT>::value_type;


        // if transform inplace - 1 read 1 write, but has to make a copy in order to have _begin be bucketed, so 2 more reads and 1 more write.
        // so here we don't transform in place.


          BL_BENCH_COLLECTIVE_START(transform_bucket, "alloc", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
		  size_t input_size = ::std::distance(_begin, _end);
		  std::vector<size_t> send_counts(this->comm.size());

          V* transformed = nullptr;
          int ret = posix_memalign(reinterpret_cast<void **>(&transformed), 64, input_size * sizeof(V));
          if (ret) {
            free(transformed);
            throw std::length_error("failed to allocate aligned memory");
          }

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
          BL_BENCH_END(transform_bucket, "alloc", input_size);

        // transform once.  bucketing and distribute will read it multiple times.
          BL_BENCH_START(transform_bucket);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
          this->transform_input(_begin, _end, transformed);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
          BL_BENCH_END(transform_bucket, "transform_input", input_size);

          BL_BENCH_COLLECTIVE_START(transform_bucket, "permute", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
              size_t comm_size = this->comm.size();
              if (comm_size <= std::numeric_limits<uint8_t>::max()) {
                ::khmxx::local::assign_and_permute(transformed, transformed + input_size,
                                                   this->key_to_rank, static_cast<uint8_t>(comm_size),
                                                   send_counts, _begin);
              } else if (comm_size <= std::numeric_limits<uint16_t>::max()) {
                ::khmxx::local::assign_and_permute(transformed, transformed + input_size,
                                                   this->key_to_rank, static_cast<uint16_t>(comm_size),
                                                   send_counts, _begin);
              } else if (comm_size <= std::numeric_limits<uint32_t>::max()) {
                ::khmxx::local::assign_and_permute(transformed, transformed + input_size,
                                                   this->key_to_rank, static_cast<uint32_t>(comm_size),
                                                   send_counts, _begin);
              } else {
                ::khmxx::local::assign_and_permute(transformed, transformed + input_size,
                                                   this->key_to_rank, static_cast<uint64_t>(comm_size),
                                                   send_counts, _begin);
              }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif
  	  	  free(transformed);
  	  	  BL_BENCH_END(transform_bucket, "permute", input_size);

              BL_BENCH_REPORT_MPI_NAMED(transform_bucket, "hashmap:transform_bucket", this->comm);


        return send_counts;
      }



      /// performs transform and bucketing using key_to_rank.
      /// transforms input array in place but permuted is stored in bucketed. and returns bucket counts
      /// appropriate when the input does not need to be permuted (insert, find, erase, update), when no output to match up, or output embeds the keys.
      template <typename IT, typename OT>
      std::vector<size_t> transform_and_bucket(IT _begin, IT _end, OT bucketed) {

    	  assert((this->comm.size() > 1) && "this function is for comm size larger than 1 only.");


        BL_BENCH_INIT(transform_bucket);


        BL_BENCH_COLLECTIVE_START(transform_bucket, "alloc", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
    __itt_resume();
#endif
		  size_t input_size = ::std::distance(_begin, _end);
		  std::vector<size_t> send_counts(this->comm.size());

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
    __itt_pause();
#endif
        BL_BENCH_END(transform_bucket, "alloc", input_size);


        // Since the bucketed is stored sepratedly, we can transform the input inplace, 1R+1W.
        //
        // transform once.  bucketing and distribute will read it multiple times.
        BL_BENCH_COLLECTIVE_START(transform_bucket, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        this->transform_input(_begin, _end, _begin);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
        BL_BENCH_END(transform_bucket, "transform_input", input_size);


          BL_BENCH_COLLECTIVE_START(transform_bucket, "permute", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
              size_t comm_size = this->comm.size();
              if (comm_size <= std::numeric_limits<uint8_t>::max()) {
                ::khmxx::local::assign_and_permute(_begin, _end,
                                                   this->key_to_rank, static_cast<uint8_t>(comm_size),
                                                   send_counts, bucketed);
              } else if (comm_size <= std::numeric_limits<uint16_t>::max()) {
                ::khmxx::local::assign_and_permute(_begin, _end,
                                                   this->key_to_rank, static_cast<uint16_t>(comm_size),
                                                   send_counts, bucketed);
              } else if (comm_size <= std::numeric_limits<uint32_t>::max()) {
                ::khmxx::local::assign_and_permute(_begin, _end,
                                                   this->key_to_rank, static_cast<uint32_t>(comm_size),
                                                   send_counts, bucketed);
              } else {
                ::khmxx::local::assign_and_permute(_begin, _end,
                                                   this->key_to_rank, static_cast<uint64_t>(comm_size),
                                                   send_counts, bucketed);
              }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif

  	  	  BL_BENCH_END(transform_bucket, "permute", input_size);


        BL_BENCH_REPORT_MPI_NAMED(transform_bucket, "hashmap:transform_bucket", this->comm);


        return send_counts;
      }



    public:

      batched_robinhood_map_base(const mxx::comm& _comm) : Base(_comm),
          key_to_rank(_comm.size()), local_changed(false) {
    	  this->c.set_ignored_msb(ceilLog2(_comm.size()));   // NOTE THAT THIS SHOULD MATCH KEY_TO_RANK use of bits in hash table.
      }



      virtual ~batched_robinhood_map_base() {};



      /// returns the local storage.  please use sparingly.
      local_container_type& get_local_container() { return c; }

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
      template <typename Predicate = ::bliss::filter::TruePredicate>
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

        // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
          this->c.insert(input);
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
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_p(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(insert);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
          return 0;
        }


            BL_BENCH_COLLECTIVE_START(insert, "alloc", this->comm);
          // get mapping to proc
          // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif

            ::std::pair<Key, T>* permuted = nullptr;
            int ret = posix_memalign(reinterpret_cast<void **>(&permuted), 64, input.size() * sizeof(::std::pair<Key, T>));
            if (ret) {
              free(permuted);
              throw std::length_error("failed to allocate aligned memory");
            }

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(insert, "alloc", input.size());



        // transform once.  bucketing and distribute will read it multiple times.
        BL_BENCH_COLLECTIVE_START(insert, "transform_bucket", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
        std::vector<size_t> send_counts = this->transform_and_bucket(input.begin(), input.end(), permuted);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
        BL_BENCH_END(insert, "transform_bucket", input.size());

        size_t before = this->c.size();




  	BL_BENCH_COLLECTIVE_START(insert, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
  	  std::vector<size_t> recv_counts(this->comm.size());
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
  	  	  	  BL_BENCH_END(insert, "a2a_count", recv_counts.size());


  	  	  	  BL_BENCH_START(insert);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

  	          ::std::pair<Key, T>* distributed = nullptr;
  	          ret = posix_memalign(reinterpret_cast<void **>(&distributed), 64, recv_total * sizeof(::std::pair<Key, T>));
  	          if (ret) {
  	            free(distributed);
  	            throw std::length_error("failed to allocate aligned memory");
  	          }
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
  	  	  	  ::khmxx::lz4::distribute_permuted(permuted, permuted + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
			  ::khmxx::distribute_permuted(permuted, permuted + input.size(),
  	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
  	  	  	  free(permuted);
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
        	this->c.insert(distributed, distributed + recv_total);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    	free(distributed);
        BL_BENCH_END(insert, "insert", this->c.size());

        BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

        return this->c.size() - before;
      }

    public:
      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<::std::pair<Key, T> >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {

    	  if (this->comm.size() == 1) {
    		  return insert_1(input, sorted_input, pred);
    	  } else {
    		  return insert_p(input, sorted_input, pred);
    	  }
      }

      /**
       * @brief count elements with the specified keys in the distributed batched_robinhood_multimap.
       * @param first
       * @param last
       */
      template <bool remove_duplicate = false, class Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, size_type> > count(::std::vector<Key>& keys, bool sorted_input = false,
                                                        Predicate const& pred = Predicate() ) const {
          BL_BENCH_INIT(count);
          ::std::vector<::std::pair<Key, size_type> > results;

          if (::dsc::empty(keys, this->comm)) {
            BL_BENCH_REPORT_MPI_NAMED(count, "base_batched_robinhood_map:count", this->comm);
            return results;
          }
//
//          if (this->empty()) {
//            BL_BENCH_REPORT_MPI_NAMED(count, "base_batched_robinhood_map:count", this->comm);
//            return results;
//          }

          BL_BENCH_COLLECTIVE_START(count, "transform_input", this->comm);
          ::fsc::back_emplace_iterator<::std::vector<::std::pair<Key, size_type> > > emplace_iter(results);
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
          BL_BENCH_END(count, "transform_input", keys.size());


          BL_BENCH_COLLECTIVE_START(count, "unique", this->comm);
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
            BL_BENCH_END(count, "unique", keys.size());

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
              std::vector<size_t> recv_counts;
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

          if (this->comm.size() > 1) {

              BL_BENCH_COLLECTIVE_START(count, "dist_query", this->comm);
              // distribute (communication part)

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
  	  	  	  ::khmxx::lz4::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
#else
              ::khmxx::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
#endif
                keys.swap(buffer);
	  //            ::dsc::distribute_unique(keys, this->key_to_rank, sorted_input, this->comm,
	  //            				typename Base::StoreTransformedFunc(),
	  //            				typename Base::StoreTransformedEqual()).swap(recv_counts);
              }
              BL_BENCH_END(count, "dist_query", keys.size());
          }

            // local count. memory utilization a potential problem.
            // do for each src proc one at a time.
          BL_BENCH_COLLECTIVE_START(count, "reserve", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
            results.reserve(keys.size() );                   // TODO:  should estimate coverage.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
            BL_BENCH_END(count, "reserve", results.capacity());

            BL_BENCH_COLLECTIVE_START(count, "local_count", this->comm);
            {
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
            	this->c.count(emplace_iter, keys.begin(), keys.end(), pred, pred);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
            }
            BL_BENCH_END(count, "local_count", results.size());

            if (this->comm.size() > 1) {

            // send back using the constructed recv count
            BL_BENCH_COLLECTIVE_START(count, "a2a2", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_A2A)
        __itt_resume();
#endif

            auto temp = mxx::all2allv(results, recv_counts, this->comm);
            results.swap(temp);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_A2A)
        __itt_pause();
#endif
            BL_BENCH_END(count, "a2a2", results.size());
          }


          BL_BENCH_REPORT_MPI_NAMED(count, "base_hashmap:count", this->comm);

          return results;

      }


      template <typename Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, size_type> > count(Predicate const & pred = Predicate()) const {

        ::std::vector<::std::pair<Key, size_type> > results = this->c.count(pred);

        if (this->comm.size() > 1) this->comm.barrier();
        return results;
      }

// TODO: exists,
// TODO: update



      /**
       * @brief find elements with the specified keys in the distributed batched_robinhood_multimap.
       * @param keys  content will be changed and reordered
       * @param last
       */
      template <bool remove_duplicate = false, typename Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, T> > find(::std::vector<Key>& keys, bool sorted_input = false,
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
  ::khmxx::lz4::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
#else
  ::khmxx::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
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
              send_counts[i] = this->c.find(emplace_iter, start, end, pred, pred);
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
            this->c.find(emplace_iter, keys.begin(), keys.end(), pred, pred);
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


      template <typename Predicate = ::bliss::filter::TruePredicate>
      ::std::vector<::std::pair<Key, T> > find(Predicate const& pred = Predicate()) const {
        ::std::vector<::std::pair<Key, T> > results = this->c.find(pred);

        if (this->comm.size() > 1) this->comm.barrier();

          return results;
      }



      /**
       * @brief erase elements with the specified keys in the distributed batched_robinhood_multimap.
       * @param first
       * @param last
       */

      template <bool remove_duplicate = false, typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase(::std::vector<Key>& keys, bool sorted_input = false, Predicate const& pred = Predicate()) {
          // even if count is 0, still need to participate in mpi calls.  if (keys.size() == 0) return;
          size_t before = this->c.size();
          BL_BENCH_INIT(erase);

          if (this->empty() || ::dsc::empty(keys, this->comm)) {
            BL_BENCH_REPORT_MPI_NAMED(erase, "base_batched_robinhood_map:erase", this->comm);
            return 0;
          }

          BL_BENCH_COLLECTIVE_START(erase, "transform_input", this->comm);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_resume();
#endif
          this->transform_input(keys);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_TRANSFORM)
        __itt_pause();
#endif
          BL_BENCH_END(erase, "transform_input", keys.size());

          if (this->comm.size() > 1) {

              BL_BENCH_COLLECTIVE_START(erase, "dist_query", this->comm);
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
  ::khmxx::lz4::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
#else
          ::khmxx::distribute(keys, this->key_to_rank, recv_counts, buffer, this->comm);
#endif
          //::khmxx::destructive_distribute(input, this->key_to_rank, recv_counts, buffer, this->comm);
          keys.swap(buffer);
              }
              BL_BENCH_END(erase, "dist_query", keys.size());

            // don't try to run unique further - have to use a set so might as well just have erase_element handle it.
            sorted_input = false;
          }

          BL_BENCH_COLLECTIVE_START(erase, "unique", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_UNIQUE)
      __itt_resume();
#endif
          if (remove_duplicate)
			  // then call local remove.
			  ::fsc::unique(keys, sorted_input,
                                                  typename Base::StoreTransformedFunc(),
                                                  typename Base::StoreTransformedEqual());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_UNIQUE)
      __itt_pause();
#endif
          BL_BENCH_END(erase, "unique", keys.size());


          BL_BENCH_COLLECTIVE_START(erase, "erase", this->comm);
          // then call local remove.
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
          this->c.erase(keys.begin(), keys.end(), pred, pred);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
          BL_BENCH_END(erase, "erase", keys.size());

          BL_BENCH_REPORT_MPI_NAMED(erase, "base_hashmap:erase", this->comm);

          if (before != this->c.size()) local_changed = true;

          return before - this->c.size();
      }

      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t erase(Predicate const & pred = Predicate()) {
        size_t before = this->c.size();

        this->c.erase(pred);

        if (before != this->c.size()) local_changed = true;

        return before - this->c.size();
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
  using batched_robinhood_map = batched_robinhood_map_base<Key, T, ::fsc::hashmap_robinhood_offsets_reduction, MapParams, Alloc, ::fsc::DiscardReducer>;


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
  class Alloc = ::std::allocator< ::std::pair<const Key, T> >,
  typename Reduc = ::std::plus<T>
  >
  using reduction_batched_robinhood_map = batched_robinhood_map_base<Key, T, ::fsc::hashmap_robinhood_offsets_reduction, MapParams, Alloc, Reduc>;




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
  	  MapParams, Alloc, ::std::plus<T> > {
      static_assert(::std::is_integral<T>::value, "count type has to be integral");

    protected:
      using Base = reduction_batched_robinhood_map<Key, T, MapParams, Alloc, ::std::plus<T>>;

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
protected:

  /**
   * @brief insert new elements in the distributed batched_robinhood_multimap.
   * @param input  vector.  will be permuted.
   */
  template <typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_1(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
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


      auto converter = [](Key const & x) {
        return ::std::make_pair(x, T(1));
      };

      using trans_iter_type = ::bliss::iterator::transform_iterator<typename std::vector< Key >::iterator, decltype(converter)>;
      trans_iter_type local_start(input.begin(), converter);
      trans_iter_type local_end(input.end(), converter);

    // TODO: predicated version.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//          std::cerr << "WARNING: not implemented to filter by predicate." << std::endl;
//        	count = this->c.insert(input, pred);
//        } else
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_resume();
#endif
      this->c.insert(local_start, local_end);
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
  template <typename Predicate = ::bliss::filter::TruePredicate>
  size_t insert_p(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
    // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    BL_BENCH_INIT(insert);

    if (::dsc::empty(input, this->comm)) {
      BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert", this->comm);
      return 0;
    }


        BL_BENCH_COLLECTIVE_START(insert, "alloc", this->comm);
      // get mapping to proc
      // TODO: keep unique only may not be needed - comm speed may be faster than we can compute unique.
//          auto recv_counts(::dsc::distribute(input, this->key_to_rank, sorted_input, this->comm));
//          BLISS_UNUSED(recv_counts);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif

        Key* permuted = nullptr;
        int ret = posix_memalign(reinterpret_cast<void **>(&permuted), 64, input.size() * sizeof(Key));
        if (ret) {
          free(permuted);
          throw std::length_error("failed to allocate aligned memory");
        }

#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "alloc", input.size());



    // transform once.  bucketing and distribute will read it multiple times.
    BL_BENCH_COLLECTIVE_START(insert, "transform_bucket", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_resume();
#endif
    std::vector<size_t> send_counts = this->transform_and_bucket(input.begin(), input.end(), permuted);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_TRANSFORM)
    __itt_pause();
#endif
    BL_BENCH_END(insert, "transform_bucket", input.size());

    size_t before = this->c.size();




	BL_BENCH_COLLECTIVE_START(insert, "a2a_count", this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_resume();
#endif
	  std::vector<size_t> recv_counts(this->comm.size());
mxx::all2all(send_counts.data(), 1, recv_counts.data(), this->comm);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
	  	  	  BL_BENCH_END(insert, "a2a_count", recv_counts.size());


	  	  	  BL_BENCH_START(insert);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_RESERVE)
  __itt_resume();
#endif
	  	  	  size_t recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));

	          Key* distributed = nullptr;
	          ret = posix_memalign(reinterpret_cast<void **>(&distributed), 64, recv_total * sizeof(Key));
	          if (ret) {
	            free(distributed);
	            throw std::length_error("failed to allocate aligned memory");
	          }
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
	  	  	  ::khmxx::lz4::distribute_permuted(permuted, permuted + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#else
		  ::khmxx::distribute_permuted(permuted, permuted + input.size(),
	  	  			  send_counts, distributed, recv_counts, this->comm);
#endif
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_A2A)
  __itt_pause();
#endif
	  	  	  free(permuted);
		  BL_BENCH_END(insert, "a2a", input.size());

#ifdef DUMP_DISTRIBUTED_INPUT
    	// target is reading by benchmark_hashtables, so want whole tuple, and is here only.

		std::stringstream ss;
		ss << "serialized." << this->comm.rank();
		serialize(distributed, distributed + recv_total, ss.str());
#endif


    BL_BENCH_COLLECTIVE_START(insert, "insert", this->comm);

    auto converter = [](Key const & x) {
      return ::std::make_pair(x, T(1));
    };

    using trans_iter_type = ::bliss::iterator::transform_iterator<Key*, decltype(converter)>;
    trans_iter_type local_start(distributed, converter);
    trans_iter_type local_end(distributed + recv_total, converter);

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
    	this->c.insert(local_start, local_end);
#ifdef VTUNE_ANALYSIS
if (measure_mode == MEASURE_INSERT)
    __itt_pause();
#endif
	free(distributed);
    BL_BENCH_END(insert, "insert", this->c.size());

    BL_BENCH_REPORT_MPI_NAMED(insert, "hashmap:insert_p", this->comm);

    return this->c.size() - before;
  }


public:

      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param input  vector.  will be permuted.
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert(std::vector<Key >& input, bool sorted_input = false, Predicate const & pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
    	  if (this->comm.size() == 1) {
    		  return this->insert_1(input, sorted_input, pred);
    	  } else {
    		  return this->insert_p(input, sorted_input, pred);
    	  }
      }


      /**
       * @brief insert new elements in the distributed batched_robinhood_multimap.
       * @param first
       * @param last
       */
      template <typename Predicate = ::bliss::filter::TruePredicate>
      size_t insert_old(std::vector< Key >& input, bool sorted_input = false, Predicate const &pred = Predicate()) {
        // even if count is 0, still need to participate in mpi calls.  if (input.size() == 0) return;
        BL_BENCH_INIT(insert);

        if (::dsc::empty(input, this->comm)) {
          BL_BENCH_REPORT_MPI_NAMED(insert, "count_hashmap:insert_key", this->comm);
          return 0;
        }

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

        // then send the raw k-mers.
        // communication part
        if (this->comm.size() > 1) {
            BL_BENCH_COLLECTIVE_START(insert, "dist_data", this->comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
          // first remove duplicates.  sort, then get unique, finally remove the rest.  may not be needed
          std::vector<size_t> recv_counts;
          std::vector<Key > buffer;
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

#ifdef ENABLE_LZ4_COMM
  ::khmxx::lz4::distribute(input, this->key_to_rank, recv_counts, buffer, this->comm);
#else
          ::khmxx::distribute(input, this->key_to_rank, recv_counts, buffer, this->comm);
#endif
          input.swap(buffer);

          BL_BENCH_END(insert, "dist_data", input.size());
        }

//        BL_BENCH_START(insert);
//#ifdef VTUNE_ANALYSIS
//  if (measure_mode == MEASURE_RESERVE)
//      __itt_resume();
//#endif
//        ::std::vector<::std::pair<Key, T> > temp;
//        temp.reserve(input.size());
//        ::fsc::back_emplace_iterator<::std::vector<::std::pair<Key, T> > > emplace_iter(temp);
//#ifdef VTUNE_ANALYSIS
//  if (measure_mode == MEASURE_RESERVE)
//      __itt_pause();
//#endif
//
//#ifdef VTUNE_ANALYSIS
//  if (measure_mode == MEASURE_TRANSFORM)
//      __itt_resume();
//#endif
//        ::std::transform(input.begin(), input.end(), emplace_iter, [](Key const & x) {
//          return ::std::make_pair(x, T(1));
//        });
//#ifdef VTUNE_ANALYSIS
//  if (measure_mode == MEASURE_TRANSFORM)
//      __itt_pause();
//#endif
//        BL_BENCH_END(insert, "convert", input.size());




        // local compute part.  called by the communicator.
        BL_BENCH_COLLECTIVE_START(insert, "local_insert", this->comm);

        auto converter = [](Key const & x) {
          return ::std::make_pair(x, T(1));
        };

        using trans_iter_type = ::bliss::iterator::transform_iterator<typename std::vector< Key >::iterator, decltype(converter)>;
        trans_iter_type local_start(input.begin(), converter);
        trans_iter_type local_end(input.end(), converter);


        size_t before = this->c.size();
// TODO: filter on insert.
//        if (!::std::is_same<Predicate, ::bliss::filter::TruePredicate>::value) {
//        	std::cerr << "WARNING, predicated insert not implemented yet. using normal" << std::endl;
//          count = this->c.insert(temp, pred);
//        } else
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
          //this->c.insert(temp);
    		this->c.insert(local_start, local_end);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
        BL_BENCH_END(insert, "local_insert", (this->c.size() - before));


//        ::std::vector<::std::pair<Key, T> >().swap(temp);  // clear the temp.


        BL_BENCH_REPORT_MPI_NAMED(insert, "count_hashmap:insert_key", this->comm);

        return this->c.size() - before;

      }
  };

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



} /* namespace dsc */


#endif // DISTRIBUTED_BATCHED_ROBINHOOD_MAP_HPP
