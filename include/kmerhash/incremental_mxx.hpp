/*
 * Copyright 2016 Georgia Institute of Technology
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
 * @file    incremental mxx.hpp
 * @ingroup
 * @author  tpan
 * @brief   extends mxx to send/recv incrementally, so as to minimize memory use and allocation
 * @details
 *
 */

#ifndef KHMXX_HPP
#define KHMXX_HPP


#include <algorithm>
#include <mxx/datatypes.hpp>
#include <mxx/comm.hpp>
#include <mxx/collective.hpp>
#include <mxx/samplesort.hpp>

#include "utils/benchmark_utils.hpp"
#include "utils/function_traits.hpp"

#include "containers/fsc_container_utils.hpp"

#ifndef LZ4_H_2983827168210
#include "lz4.c"
#endif

#include <stdlib.h>  // for posix_memalign.

namespace khmxx
{

  // local version of MPI
  namespace local {



    /* ====
     *
     * options:
     * 	3 possibilities
     *
     * 	1x hashing	|	stable	|	in place	|	algo			|	comment
     * 	======================================================================================================
     * 	n			      |	n		|	n			|					|	we can be stable for free with non-in-place, so this does not make sense.  also 2x hash is hard to offset in computation
     * 	n			      |	n		|	y			|	mxx inplace		|	not stable
     * 	n			      |	y		|	n			|	mxx 			|
     * 	n			      |	y		|	y			|		 			|	no possible strictly.  to be stable means we need ordering information, especially with multiple overlapping ranges (for each bucket entries.)
     *
     * 		1x hash means we need temp storage O(n) for bucket ids, so can be stable automatically
     *
     * 	y						|	y		|	n			|	tony's       	|	in place means to not have a copy of the data, just index.  strictly speaking it's not in place.
     * 	y						|	y		|	y   	|	tony's  			|	possible, if one-to-one mapping between in and out is available.  (extra O(n) memory)
     *
     *  TODO: minimize memory by comparing temp storage element sizes vs index array size - then choose to do "in place" or "not".
     *
     *
     */



    /**
     * @brief   implementation function for use by assign_and_bucket
     * @details uses the smallest data type (ASSIGN_TYPE) given the number of buckets.
     *          this version requires extra O(n) for the mapping, and extra O(n) for data movement.
     *
     *          for version that calls Func 2x (no additional mapping space) and O(n) for data movement, use mxx::bucket
     *          basically a counting sort impl.
     */
    template <typename T, typename Func, typename ASSIGN_TYPE, typename SIZE>
    void
    bucketing_impl(std::vector<T>& input,
                           Func const & key_func,
                           ASSIGN_TYPE const num_buckets,
                           std::vector<SIZE> & bucket_sizes,
                           size_t first = 0,
                           size_t last = std::numeric_limits<size_t>::max()) {

      static_assert(::std::is_integral<ASSIGN_TYPE>::value, "ASSIGN_TYPE should be integral, preferably unsigned");

      bucket_sizes.clear();

      // no bucket.
      if (num_buckets == 0) return;

      // initialize number of elements per bucket
      bucket_sizes.resize(num_buckets, 0);

      // ensure valid range
      size_t f = std::min(first, input.size());
      size_t l = std::min(last, input.size());
      assert((f <= l) && "first should not exceed last" );

      if (f == l) return;  // no data in question.

      size_t len = l - f;

      // single bucket.
      if (num_buckets == 1) {
        bucket_sizes[0] = len;

        return;
      }

      // output to input mapping
      std::vector<ASSIGN_TYPE> i2o;
      i2o.reserve(len);
//      int ret;
//      ASSIGN_TYPE* i2o = nullptr;
//      ret = posix_memalign(reinterpret_cast<void **>(&i2o), 64, len * sizeof(ASSIGN_TYPE));
//      if (ret) {
//        free(i2o);
//        throw std::length_error("failed to allocate aligned memory");
//      }
////        ASSIGN_TYPE* i2o = nullptr;
////        ptrdiff_t i2o_size = 0;
////        std::tie(i2o, i2o_size) = ::std::get_temporary_buffer<ASSIGN_TYPE>(len);
////        if (i2o_size < static_cast<ptrdiff_t>(len)) {
////          return_temporary_buffer(i2o);
////          throw std::length_error("failed to allocate aligned memory");
////        }


      // [1st pass]: compute bucket counts and input to bucket assignment.
      ASSIGN_TYPE p;
//      ASSIGN_TYPE* i2o_it = i2o;
//      for (size_t i = f; i < l; ++i, ++i2o_it) {
      for (size_t i = f; i < l; ++i) {
          p = key_func(input[i]);

          assert(((0 <= p) && ((size_t)p < num_buckets)) && "assigned bucket id is not valid");

          i2o.emplace_back(p);
//          *i2o_it = p;
          ++bucket_sizes[p];
      }

      // get offsets of where buckets start (= exclusive prefix sum)
      // use bucket_sizes temporarily.
      bucket_sizes.back() = len - bucket_sizes.back();
      // checking for >= 0 with unsigned and decrement is a bad idea.  need to check for > 0
      for (size_t i = num_buckets - 1; i > 0; --i) {
        bucket_sizes[i-1] = bucket_sizes[i] - bucket_sizes[i-1];
      }
      assert(bucket_sizes.front() == 0);  // first one should be 0 at this point.


      // [2nd pass]: saving elements into correct position, and save the final position.
      if (len = input.size()) {
        // use swap
        std::vector<T> tmp_result(len);
        for (size_t i = f; i < l; ++i) {
            tmp_result[bucket_sizes[i2o[i-f]]++] = input[i];
//      i2o_it = i2o;
//        for (size_t i = f; i < l; ++i, ++i2o_it) {
//            tmp_result[bucket_sizes[*i2o_it]++] = input[i];
        }
        input.swap(tmp_result);

      } else {
        T* tmp_result = nullptr;
        int ret = posix_memalign(reinterpret_cast<void **>(&tmp_result), 64, len * sizeof(T));
        if (ret) {
          free(tmp_result);
          throw std::length_error("failed to allocate aligned memory");
        }

        for (size_t i = f; i < l; ++i) {
          tmp_result[bucket_sizes[i2o[i-f]]++] = input[i];
//      i2o_it = i2o;
//        for (size_t i = f; i < l; ++i, ++i2o_it) {
//            tmp_result[bucket_sizes[*i2o_it]++] = input[i];
        }
        memcpy(input.data() + f, tmp_result, len * sizeof(T));   // else memcpy.

        free(tmp_result);
      }

//      free(i2o);
////      return_temporary_buffer(i2o);

      // this process should have turned bucket_sizes to an inclusive prefix sum
      assert(bucket_sizes.back() == len);
      // convert inclusive prefix sum back to counts.
      // hopefully this is a fast process when compared to allocating memory.
      for (size_t i = num_buckets - 1; i > 0; --i) {
        bucket_sizes[i] -= bucket_sizes[i-1];
      }
    }

    // writes into separate array of results.
    template <typename T, typename Func, typename ASSIGN_TYPE, typename SIZE>
    void
    bucketing_impl(std::vector<T>const & input,
                           Func const & key_func,
                           ASSIGN_TYPE const num_buckets,
                           std::vector<SIZE> & bucket_sizes,
                           std::vector<T> & results,
                           size_t first = 0,
                           size_t last = std::numeric_limits<size_t>::max()) {

      static_assert(::std::is_integral<ASSIGN_TYPE>::value, "ASSIGN_TYPE should be integral, preferably unsigned");

      assert(((input.size() == 0) || (input.data() != results.data())) &&
  			"input and output should not be the same.");

      bucket_sizes.clear();

      // no bucket.
      if (num_buckets == 0) return;

      // initialize number of elements per bucket
      bucket_sizes.resize(num_buckets, 0);

      // ensure valid range
      size_t f = std::min(first, input.size());
      size_t l = std::min(last, input.size());
      assert((f <= l) && "first should not exceed last" );

      if (f == l) return;  // no data in question.

      size_t len = l - f;

      // single bucket.
      if (num_buckets == 1) {
    	// set output buckets sizes
        bucket_sizes[0] = len;

        // set output values
        memcpy(results.data() + f, input.data() + f, len * sizeof(T));

        return;
      }

      // output to input mapping
      std::vector<ASSIGN_TYPE> i2o;
      i2o.reserve(len);
      //int ret;
//      ASSIGN_TYPE* i2o = nullptr;
//      ret = posix_memalign(reinterpret_cast<void **>(&i2o), 64, len * sizeof(ASSIGN_TYPE));
//      if (ret) {
//        free(i2o);
//        throw std::length_error("failed to allocate aligned memory");
//      }
////      ASSIGN_TYPE* i2o = nullptr;
////      ptrdiff_t i2o_size = 0;
////      std::tie(i2o, i2o_size) = ::std::get_temporary_buffer<ASSIGN_TYPE>(len);
////      if (i2o_size < static_cast<ptrdiff_t>(len)) {
////        return_temporary_buffer(i2o);
////        throw std::length_error("failed to allocate aligned memory");
////      }

      // [1st pass]: compute bucket counts and input to bucket assignment.
      ASSIGN_TYPE p;
      for (size_t i = f; i < l; ++i) {
//      ASSIGN_TYPE* i2o_it = i2o;
//      for (size_t i = f; i < l; ++i, ++i2o_it) {
          p = key_func(input[i]);

          assert(((0 <= p) && ((size_t)p < num_buckets)) && "assigned bucket id is not valid");

          i2o.emplace_back(p);
//          *i2o_it = p;
          ++bucket_sizes[p];
      }

      // get offsets of where buckets start (= exclusive prefix sum)
      // use bucket_sizes temporarily.
      bucket_sizes.back() = l - bucket_sizes.back();
      // checking for >= 0 with unsigned and decrement is a bad idea.  need to check for > 0
      for (size_t i = num_buckets - 1; i > 0; --i) {
        bucket_sizes[i-1] = bucket_sizes[i] - bucket_sizes[i-1];
      }
      assert(bucket_sizes.front() == f);  // first one should be 0 at this point.

      // [2nd pass]: saving elements into correct position, and save the final position.
      results.resize(input.size());
      for (size_t i = f; i < l; ++i) {
          results[bucket_sizes[i2o[i-f]]++] = input[i];
      }
//      i2o_it = i2o;
//      for (size_t i = f; i < l; ++i, ++i2o_it) {
//          results[bucket_sizes[*i2o_it]++] = input[i];
//      }

//      free(i2o);
////      return_temporary_buffer(i2o);

      // this process should have turned bucket_sizes to an inclusive prefix sum
      assert(bucket_sizes.back() == l);
      // convert inclusive prefix sum back to counts.
      // hopefully this is a fast process when compared to allocating memory.
      for (size_t i = num_buckets - 1; i > 0; --i) {
        bucket_sizes[i] -= bucket_sizes[i-1];
      }
      bucket_sizes[0] -= f;

    }


    // writes into separate array of results..  similar to bucketing_impl
    template <typename IT, typename Func, typename ASSIGN_TYPE, typename OT,
    typename ::std::enable_if<::std::is_same<typename ::std::iterator_traits<OT>::iterator_category,
                                             ::std::random_access_iterator_tag >::value, int>::type = 1  >
    void
    assign_and_permute(IT _begin, IT _end,
                           Func const & key_func,
                           ASSIGN_TYPE const num_buckets,
                           std::vector<size_t> & bucket_sizes,
                           OT results) {

      static_assert(::std::is_integral<ASSIGN_TYPE>::value, "ASSIGN_TYPE should be integral, preferably unsigned");

      bucket_sizes.clear();

      std::vector<size_t> bucket_offsets;

      if (_begin == _end) return;  // no data in question.

      // no bucket.
      if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");

      // initialize number of elements per bucket
      bucket_sizes.resize(num_buckets, 0);
      bucket_offsets.resize(num_buckets, 0);

      size_t input_size = std::distance(_begin, _end);

      // single bucket.
      if (num_buckets == 1) {
        // set output buckets sizes
        bucket_sizes[0] = input_size;

        // set output values
        std::copy(_begin, _end, results);

        return;
      } else {

        // 2 pass algo.

        // first get the mapping array.
        ASSIGN_TYPE * i2o = nullptr;
        int ret = posix_memalign(reinterpret_cast<void **>(&i2o), 64, input_size * sizeof(ASSIGN_TYPE));
        if (ret) {
          free(i2o);
          throw std::length_error("failed to allocate aligned memory");
        }

        // [1st pass]: compute bucket counts and input2bucket assignment.
        // store input2bucket assignment in i2o temporarily.
        ASSIGN_TYPE p;
        ASSIGN_TYPE* i2o_it = i2o;
        for (auto it = _begin; it != _end; ++it, ++i2o_it) {
            p = key_func(*it);

            assert(((0 <= p) && ((size_t)p < num_buckets)) && "assigned bucket id is not valid");

            *i2o_it = p;
            ++bucket_sizes[p];
        }

        // since we decrement the offsets, we are filling from back.  to maintain stable, input is iterated from back.

        // compute exclusive offsets
        size_t sum = 0;
        bucket_offsets[0] = 0;
        for (size_t i = 1; i < num_buckets; ++i) {
        	sum += bucket_sizes[i-1];
        	bucket_offsets[i] = sum;
        }

////         [2nd pass]: saving elements into correct position, and save the final position.
////		below is for use with inclusive offsets.
//        i2o_it = i2o;
//        std::advance(i2o_it, input_size);
//        for (IT it = _end; it != _begin; ) {
//          *(results + (--bucket_offsets[*(--i2o_it)])) = *(--it);   // offset decremented by 1 before use.
//              // bucekted filled from back to front for each bucket, hence iterators are pre-decremented.
//        }

        // [2nd pass]: saving elements into correct position, and save the final position.
        i2o_it = i2o;
//        std::advance(i2o_it, input_size);
        for (IT it = _begin; it != _end; ++it, ++i2o_it) {
          *(results + (bucket_offsets[*i2o_it]++)) = *it;   // offset decremented by 1 before use.
              // bucekted filled from back to front for each bucket, hence iterators are pre-decremented.
        }

        free(i2o);

      }

    }


    /**
     * @brief   compute the element index mapping between input and bucketed output.
     *
     *  good for a2av, all at once.
     * this is an "deconstructed" version of assign_to_buckets
     *
     * It returns an array with the output position for the input entries, i.e.
     *    an array x, such that input[i] = output[x[i]]; and the mapping and input share the same coordinates.
     *
     *    To reconstruct input, we can walk through the mapping and random access output array,
     *    which should be no slower than before as the writes are consecutive in cache while the reads are random.
     *
     *    also, this is more friendly with emplacing into a vector.
     *    the mapping is useful to avoid recomputing key_func.
     *
     * this is faster than mxx by about 2x (for hashing k-mers 1x.), but uses another vector of size 8N.
     *
     *    ** operate only within the input range [first, last), and assign to same output range.
     *
     * @tparam T            Input type
     * @tparam Func         Type of the key function.
     * @param input         Contains the values to be bucketed.
     * @param key_func      A function taking a type T and returning the bucket index
     *                      in the range [0, num_buckets).
     * @param num_buckets     number of buckets.  cannot be more than num procs in MPI
     * @param bucket_sizes[out]  The number of elements in each bucket.  reset during each call.
     * @param i2o[out]      input to output position mapping.  updated between first and last.  value is between first and last
     * @param first         first position to bucket in input
     * @param last          last position to bucket in input
     */
    template <typename T, typename Func, typename SIZE>
    void
    assign_to_buckets(::std::vector<T> const & input,
                      Func const & key_func,
                      size_t const & num_buckets,
                      std::vector<size_t> & bucket_sizes,
                      ::std::vector<SIZE> & i2o,
                      size_t first = 0,
                      size_t last = std::numeric_limits<size_t>::max()) {
        bucket_sizes.clear();

        // no bucket.
        if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");

        // initialize number of elements per bucket
        bucket_sizes.resize(num_buckets, 0);

        // ensure valid range
        size_t f = std::min(first, input.size());
        size_t l = std::min(last, input.size());
        assert((f <= l) && "first should not exceed last" );

        if (f == l) return;  // no data in question.

        i2o.resize(input.size(), 0);  // need exactly input size.

        // single bucket.
        if (num_buckets == 1) {
          bucket_sizes[0] = l - f;

          memset(&(i2o[f]), 0, (l-f) * sizeof(SIZE));

          return;
        }


        // [1st pass]: compute bucket counts and input2bucket assignment.
        // store input2bucket assignment in i2o temporarily.
        size_t p;
        for (size_t i = f; i < l; ++i) {
            p = key_func(input[i]);

            assert(((0 <= p) && ((size_t)p < num_buckets)) && "assigned bucket id is not valid");

            i2o[i] = p;
            ++bucket_sizes[p];
        }

    }

//    template <typename IT, typename Func, typename ST>
//    void
//    assign_to_buckets(IT _begin, IT _end,
//                      Func const & key_func,
//                      size_t const & num_buckets,
//                      std::vector<size_t> & bucket_sizes,
//                      ST i2o_begin) {
//
//      bucket_sizes.clear();
//
//        if (_begin == _end) return;  // no data in question.
//
//        // no bucket.
//        if (num_buckets == 0) throw std::invalid_argument("ERROR: number of buckets is 0");
//
//        // initialize number of elements per bucket
//        bucket_sizes.resize(num_buckets, 0);
//
//
//
//        size_t input_size = std::distance(_begin, _end);
//
//
//        // single bucket.
//        if (num_buckets == 1) {
//          bucket_sizes[0] = input_size;
//
//
//          ST i2o_end = i2o_begin;
//			std::advance(i2o_end, input_size);
//          for (auto it = i2o; it != i2o_end; ++it) {
//            *it = 0;
//          }
//
//          return;
//        } else {
//
//          // [1st pass]: compute bucket counts and input2bucket assignment.
//          // store input2bucket assignment in i2o temporarily.
//          size_t p;
//          ST i2o_it = i2o_begin;
//          for (auto it = _begin; it != _end; ++it, ++i2o_it) {
//              p = key_func(*it);
//
//              assert(((0 <= p) && ((size_t)p < num_buckets)) && "assigned bucket id is not valid");
//
//              *i2p_it = p;
//              ++bucket_sizes[p];
//          }
//        }
//    }
//
//    /**
//     * @brief  given a mapping from unbucketed to bucket id, permute the input so that the output has bucket order.
//     * @details
//     *    combined bucketId_to_pos and permute.
//     *    only enabled if OT is random iterator.
//     *    compute inclusive prefix sum for offsets.
//     *    when permuting, decrement and then insert at decremented position.
//     *
//     *    since random access, it's okay to fill from larger pos to smaller.
//     *
//     *    at the end, have the offsets for buckets to return.
//     */
//    template <typename IT, typename MT, typename OT,
//      typename ::std::enable_if<::std::is_same<typename ::std::iterator_traits<OT>::iterator_category,
//                                               ::std::random_access_iterator_tag >::value, int>::type = 1  >
//    std::vector<size_t> permute_to_buckets(IT unbucketed, IT unbucketed_end,
//                             std::vector<size_t> const & bucket_sizes,
//        MT i2o, OT bucketed) {
//
//      std::vector<size_t> offsets(bucket_sizes.size(), 0);
//
//      if (unbucketed == unbucketed_end) {
//        return offsets;
//      }
//
//      if (bucket_sizes.size() == 0) {
//        throw std::logic_error("bucket_sizes should not be 0");
//      }
//
//      // compute inclusive offsets
//      offsets[0] = bucket_sizes[0];
//      size_t max = bucket_sizes.size();
//      for (size_t i = 1; i < max; ++i) {
//        offsets[i] = offsets[i-1] + bucket_sizes[i];
//      }
//
//      // [2nd pass]: saving elements into correct position, and save the final position.
//      MT it2 = i2o;
//      for (IT it = unbucketed; it != unbucketed_end; ++it, ++it2) {
//        *(bucketed + (--offsets[*it2])) = *it;   // offset decremented by 1 before use.   bucekted filled from back to front for each bucket.
//      }
//
//      return offsets;
//
//    }
//    /// stable version of permute to bucket.  approach is to iterate input in reverse.
//    template <typename IT, typename MT, typename OT,
//      typename ::std::enable_if<::std::is_same<typename ::std::iterator_traits<OT>::iterator_category,
//                                               ::std::random_access_iterator_tag >::value, int>::type = 1  >
//    std::vector<size_t> stable_permute_to_buckets(IT unbucketed, IT unbucketed_end,
//                             std::vector<size_t> const & bucket_sizes,
//        MT i2o, OT bucketed) {
//
//      std::vector<size_t> offsets(bucket_sizes.size(), 0);
//
//      if (unbucketed == unbucketed_end) {
//        return offsets;
//      }
//
//      if (bucket_sizes.size() == 0) {
//        throw std::logic_error("bucket_sizes should not be 0");
//      }
//
//      // compute inclusive offsets
//      offsets[0] = bucket_sizes[0];
//      size_t max = bucket_sizes.size();
//      for (size_t i = 1; i < max; ++i) {
//        offsets[i] = offsets[i-1] + bucket_sizes[i];
//      }
//
//      // [2nd pass]: saving elements into correct position, and save the final position.
//      MT it2 = i2o;
//		std::advance(it2, std::distance(unbucketed, unbucketed_end));
//      for (IT it = unbucketed_end; it != unbucketed; ) {
//        *(bucketed + (--offsets[*(--it2)])) = *(--it);   // offset decremented by 1 before use.   bucekted filled from back to front for each bucket.
//      }
//
//      return offsets;
//    }


    // operate only within the input range [first, last), and assign to same output range.
    template <typename SIZE = size_t>
    void
    bucketId_to_pos(std::vector<SIZE> & bucket_sizes,
                          std::vector<SIZE> & i2o,
                          size_t first = 0,
                          size_t last = std::numeric_limits<size_t>::max()) {

      // no bucket.
      if (bucket_sizes.size() == 0) throw std::invalid_argument("bucket_sizes has 0 buckets.");

      // ensure valid range
      size_t f = std::min(first, i2o.size());
      size_t l = std::min(last, i2o.size());
      assert((f <= l) && "first should not exceed last" );

      if (f == l) return;  // no data in question.


      // get offsets of where buckets start (= exclusive prefix sum), offset by the range.
      // use bucket_sizes temporarily.
      bucket_sizes.back() = l - bucket_sizes.back();  // offset from f, or alternatively, l.
      // checking for >= 0 with unsigned and decrement is a bad idea.  need to check for > 0
      for (size_t i = bucket_sizes.size() - 1; i > 0; --i) {
        bucket_sizes[i-1] = bucket_sizes[i] - bucket_sizes[i-1];
      }
      assert((bucket_sizes.front() == f) && "prefix sum resulted in incorrect starting position");  // first one should be 0 at this point.

      // [2nd pass]: saving elements into correct position, and save the final position.
      for (size_t i = f; i < l; ++i) {
        i2o[i] = bucket_sizes[i2o[i]]++;  // output position from bucket id (i2o[i]).  post increment.  already offset by f.
      }

      // this process should have turned bucket_sizes to an inclusive prefix sum
      assert((bucket_sizes.back() == l) && "mapping assignment resulted in incorrect ending position");
      // convert inclusive prefix sum back to counts.
      // hopefully this is a fast process when compared to allocating memory.
      for (size_t i = bucket_sizes.size() - 1; i > 0; --i) {
        bucket_sizes[i] -= bucket_sizes[i-1];
      }
      bucket_sizes[0] -= f;
    }

//    template <typename SIZE = size_t>
//    void
//    permutation_to_bucket(std::vector<SIZE> & bucket_sizes,
//                          std::vector<SIZE> & i2o,
//                          size_t first = 0,
//                          size_t last = std::numeric_limits<size_t>::max()) {
//
//      // TO IMPLEMENT?
//    }


    /**
     * @brief perform permutation on i2o to produce x number of blocks, each with p number of buckets with specified per-bucket size s.  remainder are placed at end
     *        and output bucket sizes are the remainders for each bucket.
     * @details  kind of a division on the buckets.
     *
     *        operates within range [first, last) only.
     *
     * @param in_block_bucket_size   size of a bucket inside a block - all buckets are same size.
     * @param bucket_sizes[in/out]   entire size of local buckets.
     * @param i2o[in/out]            from bucket assignment, to permutation index.
     * @return number of FULL blocks in the permutation.
     */
    template <typename SIZE = size_t>
    void
    blocked_bucketId_to_pos(SIZE const & block_bucket_size, SIZE const & nblocks,
                                std::vector<SIZE> & bucket_sizes,
                                std::vector<SIZE> & i2o,
                                size_t first = 0,
                                size_t last = std::numeric_limits<size_t>::max()) {

      if (bucket_sizes.size() == 0)
        throw std::invalid_argument("bucket_sizes is 0");


      // if block_bucket_size is 0, use normal stuff
      if ((block_bucket_size * nblocks * bucket_sizes.size()) > i2o.size()) {
        throw std::invalid_argument("block is larger than total size");
      }

        // no block.  so do it without blocks
      if ((block_bucket_size == 0) || (nblocks == 0)) {
        // no blocks. use standard permutation
        bucketId_to_pos(bucket_sizes, i2o, first, last);

        return;
      }

      if (bucket_sizes.size() == 1)
      {
        // all go into the block.  no left overs.
        bucketId_to_pos(bucket_sizes, i2o, first, last);
        bucket_sizes[0] = i2o.size() - block_bucket_size * nblocks;
        return;
      }

      SIZE min_bucket_size = *(std::min_element(bucket_sizes.begin(), bucket_sizes.end()));  // find the minimum
      if (nblocks > (min_bucket_size / block_bucket_size)) {
        // no blocks. use standard permutation
        throw std::invalid_argument("min bucket is smaller than the number of requested blocks. ");
      }

      // else we do blocked permutation

      size_t block_size = block_bucket_size * bucket_sizes.size();
      size_t front_size = block_size * nblocks;

      // ensure valid range
      size_t f = std::min(first, i2o.size());
      size_t l = std::min(last, i2o.size());
      assert((f <= l) && "first should not exceed last" );

      if (f == l) return;  // no data in question.


      // initialize the offset, and reduce the bucket_sizes
      std::vector<SIZE> offsets(bucket_sizes.size(), 0);
      for (size_t i = 0; i < bucket_sizes.size(); ++i) {
        offsets[i] = i * block_bucket_size;
        bucket_sizes[i] -= block_bucket_size * nblocks;
      }

      //== convert bucket sizes to offsets as well.
      // get offsets of where buckets start (= exclusive prefix sum), offset by the range.
      // use bucket_sizes temporarily.
      bucket_sizes.back() = l - bucket_sizes.back();  // offset from f, or alternatively, l.
      // checking for >= 0 with unsigned and decrement is a bad idea.  need to check for > 0
      for (size_t i = bucket_sizes.size() - 1; i > 0; --i) {
        bucket_sizes[i-1] = bucket_sizes[i] - bucket_sizes[i-1];
      }
      assert((bucket_sizes.front() == (f + front_size)) && "prefix sum resulted in incorrect starting position");  // first one should be 0 at this point.


      // walk through all.
      size_t bucket;
      for (size_t i = f; i < l; ++i) {
        bucket = i2o[i];
        i2o[i] = offsets[bucket]++;  // output position from bucket id (i2o[i]).  post increment.  already offset by f.

        // if the offset entry indicates full, move the offset to next block, unless we already have max number of blocks.
        if (offsets[bucket] > front_size) {
          continue;  // last part with NONBLOCKS, so let it continue;
//        } else if (offsets[bucket] == front_size) {  // last element of last bucket in last block in the front portion
//          offsets[bucket] = bucket_sizes[bucket];
        } else if (offsets[bucket] % block_bucket_size == 0) {  // full bucket.
          if (((offsets[bucket] + block_size - 1) / block_size) == nblocks) { // in last block
            offsets[bucket] = bucket_sizes[bucket];
          } else {
            offsets[bucket] += block_size - block_bucket_size;
          }
        }
      }

      // convert inclusive prefix sum back to counts.
      // hopefully this is a fast process when compared to allocating memory.
      for (size_t i = 0; i < bucket_sizes.size() - 1; ++i) {
        bucket_sizes[i] = bucket_sizes[i+1] - bucket_sizes[i];
      }
      bucket_sizes.back() = l - bucket_sizes.back();

    }


    //===  permute and unpermute are good for communication bucketing for partitions of INPUT data, using A2Av for communication (not for A2A)

    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       permute a range in input.  (entire output range may be affected.)
     *
     *		 if OT is not random access iterator, this will be very slow.
     *
     * @param unbucketed	input to be bucketed.  only processed between first and last.
     * @param i2o			mapping.  only the part between first and last is used.  values should be between first and last.
     * @param results		bucketed output, only processed between first and last.
     */
    template <typename IT, typename OT, typename MT>
    void permute_for_input_range(IT unbucketed, IT unbucketed_end,
    		MT i2o, OT bucketed, OT bucketed_end,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t in_len = std::distance(unbucketed, unbucketed_end);  // input range.
    	size_t out_len = std::distance(bucketed, bucketed_end);   // number of output to look for.

        // if input is empty, simply return
        if ((in_len == 0) || (out_len == 0)) return;

        assert((out_len >= in_len) && "input range is larger than output range");

        //size_t bucketed_max = bucketed_pos_offset + out_len - 1;
        assert((*(std::min_element(i2o, i2o + in_len)) >= bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + in_len)) <= (bucketed_pos_offset + out_len - 1)) &&
				"ERROR, i2o [0, len) does not map to itself");

        // saving elements into correct position
        for (; unbucketed != unbucketed_end; ++i2o, ++unbucketed) {
            *(bucketed + (*i2o - bucketed_pos_offset)) = *unbucketed;
        }
    }


    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       permute a range in output.  (entire input range may be read..)
     *
     * @param unbucketed	input to be bucketed.  only processed between first and last.
     * @param i2o			mapping.  only the part between first and last is used.  values should be between first and last.
     * @param results		bucketed output, only processed between first and last.
     */
    template <typename IT, typename OT, typename MT>
    void permute_for_output_range(IT unbucketed, IT unbucketed_end,
    		MT i2o, OT bucketed, OT bucketed_end,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t in_len = std::distance(unbucketed, unbucketed_end);  // input range.
    	size_t out_len = std::distance(bucketed, bucketed_end);   // number of output to look for.

        // if input is empty, simply return
        if ((in_len == 0) || (out_len == 0)) return;

        assert((in_len >= out_len) && "output range is larger than input range");

        size_t bucketed_max = bucketed_pos_offset + out_len - 1;
        assert((*(std::min_element(i2o, i2o + in_len)) <= bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + in_len)) >= (bucketed_pos_offset + out_len - 1) ) &&
				"ERROR, i2o [0, len) does not map to itself");

        // saving elements into correct position
        size_t pos = 0;
        size_t count = 0;
        for (; unbucketed != unbucketed_end; ++i2o, ++unbucketed) {
        	pos = *i2o;
        	if ((pos < bucketed_pos_offset) || (pos > bucketed_max)) continue;

			*(bucketed + (pos - bucketed_pos_offset)) = *unbucketed;
			++count;

        	if (count == out_len) break;  // filled.  done.
        }
    }

    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       permute a range in input.  (the range maps to itself.)
     *
     * @param unbucketed	input to be bucketed.  only processed between first and last.
     * @param i2o			mapping.  only the part between first and last is used.  values should be between first and last.
     * @param results		bucketed output, only processed between first and last.
     */
    template <typename IT, typename OT, typename MT>
    void permute(IT unbucketed, IT unbucketed_end,
    		MT i2o, OT bucketed,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t len = std::distance(unbucketed, unbucketed_end);
        // if input is empty, simply return
        if (len == 0) return;

        assert((*(std::min_element(i2o, i2o + len)) == bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + len)) == ( bucketed_pos_offset + len - 1)) &&
				"ERROR, i2o [first, last) does not map to itself");

        // saving elements into correct position
        for (; unbucketed != unbucketed_end; ++unbucketed, ++i2o) {
            *(bucketed + (*i2o - bucketed_pos_offset)) = *unbucketed;
        }

    }



    /**
     * @brief inplace permute.  at most 2n steps, as we need to check each entry for completion.
     * @details	also need i2o to be a signed integer, so we can use the sign bit to check
     * 			that an entry has been visited.
     *
     *      permute_inplace uses cycle following, so entire range is affected, unless it is known a priori
     *      that [first, last) maps to itself.
     *
     * 			profiling:  heaptrack shows that there are unlikely unnecessary allocation
     * 						cachegrind and gprof show that bulk of time is in inplace_permute, and there are no further details
     * 						perf stats show that cache miss rate is about the same for permute as is for inplace permute
     * 						perf record shows that the majority of the cost is in swap (or in the case of inplace_unpermute, in copy assignment)
     * 						checking counter shows that the array is visited approximately 2x. reducing this is unlikely to improve performance
     * 							due to move/copy cost.
     * 						move/copy cost is associated with std::pair.
     *				loop unrolling did not help much.  trying to change read/write dependence does not seem to help much either.
     *				issue is random read and write, probably.  perf profiling identifies the problem as such.
     *
     */
    template <typename T>
    void permute_inplace(std::vector<T> & unbucketed, std::vector<size_t> & i2o,
    		size_t first = 0,
			size_t last = std::numeric_limits<size_t>::max()) {

    	if (unbucketed.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    		throw std::invalid_argument("input is limited to max signed long in size");
    	// TODO: separate logic using a bit vector.

    	// ensure valid range
        size_t f = std::min(first, unbucketed.size());
        size_t l = std::min(last, unbucketed.size());
        assert((f <= l) && "first should not exceed last" );
        size_t len = l-f;

        // if input is empty, simply return
        if ((len == 0) || (unbucketed.size() == 0) || (i2o.size() == 0)) return;

        assert((*(std::min_element(i2o.begin() + f, i2o.begin() + l)) == f) &&
        		(*(std::max_element(i2o.begin() + f, i2o.begin() + l)) == (l - 1)) &&
				"ERROR, i2o [first, last) does not map to itself");

        assert(unbucketed.size() == i2o.size());

        // saving elements into correct position
        // modeled after http://stackoverflow.com/questions/7365814/in-place-array-reordering
        // and https://blog.merovius.de/2014/08/12/applying-permutation-in-constant.html
        size_t j, k;
        T v;
        constexpr size_t mask = ~(0UL) << (sizeof(size_t) * 8 - 1);
//        size_t counter = 0;
        for (size_t i = f; i < l; ++i) {
        	k = i2o[i];
//        	++counter;

        	if (k >= mask) {
        		// previously visited.  unmark and move on.
        		i2o[i] = ~k;  // negate
        		continue;
        	}

        	// get the current position and value
        	j = i;
        	v = unbucketed[j];
        	while (k != i) {  // stop when cycle completes (return to original position)
        		std::swap(unbucketed[k], v);
        		j = k;
        		k = i2o[j];
//        		++counter;

        		i2o[j] = ~k;  			  // mark the last position as visited.
        	}
        	unbucketed[i] = v;				  // cycle complete, swap back the current value (from some other location)
        }

//        std::cout << "i2o read " << counter << " times.  size= " << i2o.size() << std::endl;

    }


    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       unpermute input range (entire output range is involved, last - first entries are written.)
     *       i2o and unbucketed should be same size.
     */
    template <typename IT, typename OT, typename MT>
    void unpermute_by_input_range(IT bucketed, IT bucketed_end,
    		MT i2o, OT unbucketed, OT unbucketed_end,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t in_len = std::distance(bucketed, bucketed_end);   // number of output to look for.
    	size_t out_len = std::distance(unbucketed, unbucketed_end);  // input range.

        // if input is empty, simply return
        if ((in_len == 0) || (out_len == 0)) return;

        assert((out_len >= in_len) && "input range is larger than output range");

        size_t bucketed_max = bucketed_pos_offset + in_len - 1;
        assert((*(std::min_element(i2o, i2o + out_len)) <= bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + out_len)) >= bucketed_max) &&
				"ERROR, i2o [0, len) does not map to itself");

        MT i2o_end = i2o + out_len;

        size_t pos = 0;
        size_t count = 0;
        for (; i2o != i2o_end; ++i2o, ++unbucketed) {
        	pos = *i2o;
        	if ((pos < bucketed_pos_offset) || (pos > bucketed_max)) continue;

        	*unbucketed = *(bucketed + (pos - bucketed_pos_offset));
			++count;

        	if (count == in_len) break;  // filled.  done.
        }

    }


    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       unpermute output range (entire input range is involved, last - first entries are read.)
     */
    template <typename IT, typename OT, typename MT>
    void unpermute_by_output_range(IT bucketed, IT bucketed_end,
    		MT i2o, OT unbucketed, OT unbucketed_end,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t out_len = std::distance(unbucketed, unbucketed_end);  // input range.
    	size_t in_len = std::distance(bucketed, bucketed_end);   // number of output to look for.

        // if input is empty, simply return
        if ((in_len == 0) || (out_len == 0)) return;

        assert((out_len <= in_len) && "input range is larger than output range");

        //size_t bucketed_max = bucketed_pos_offset + in_len - 1;
        assert((*(std::min_element(i2o, i2o + out_len)) >= bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + out_len)) <= (bucketed_pos_offset + in_len - 1)) &&
				"ERROR, i2o [0, len) does not map to itself");

        // saving elements into correct position
        for (; unbucketed < unbucketed_end; ++i2o, ++unbucketed) {
            *unbucketed = *(bucketed + (*i2o - bucketed_pos_offset));
        }
    }

    /**
     * @brief  given a bucketed array and the mapping from unbucketed to bucketed, undo the bucketing.
     * @details  note that output and i2o are traversed in order (write and read respectively),
     *       while input is randomly read.
     *
     *       unpermute output range (requires [first, last) map to itself.)
     */
    template <typename IT, typename OT, typename MT>
    void unpermute(IT bucketed, IT bucketed_end,
    		MT i2o, OT unbucketed,
    		size_t const & bucketed_pos_offset) {

    	static_assert(std::is_same<typename std::iterator_traits<IT>::value_type,
    			typename std::iterator_traits<OT>::value_type>::value,
				"ERROR: IT and OT should be iterators with same value types");
    	static_assert(std::is_integral<typename std::iterator_traits<MT>::value_type>::value,
				"ERROR: MT should be an iterator of integral type value");

    	size_t len = std::distance(bucketed, bucketed_end);
        // if input is empty, simply return
        if (len == 0) return;

        assert((*(std::min_element(i2o, i2o + len)) == bucketed_pos_offset) &&
        		(*(std::max_element(i2o, i2o + len)) == ( bucketed_pos_offset + len - 1)) &&
				"ERROR, i2o [first, last) does not map to itself");

        // saving elements into correct position
        OT unbucketed_end = unbucketed + len;
        for (; unbucketed != unbucketed_end; ++unbucketed, ++i2o) {
            *unbucketed = *(bucketed + (*i2o - bucketed_pos_offset));
        }
    }


    /**
     *
    * 			profiling:  heaptrack shows that there are unlikely unnecessary allocation
    * 						cachegrind and gprof show that bulk of time is in inplace_permute, and there are no further details
    * 						perf stats show that cache miss rate is about the same for permute as is for inplace permute
    * 						perf record shows that the majority of the cost is in swap (or in the case of inplace_unpermute, in copy assignment)
    * 						checking counter shows that the array is visited approximately 2x. reducing this is unlikely to improve performance
    * 							due to move/copy cost.
    * 						move/copy cost is associated with std::pair.
    *
    * 						unpermute input range (entire bucketed range is used.)
    *
     */
    template <typename T>
    void unpermute_inplace(std::vector<T> & bucketed, std::vector<size_t> & i2o,
    		size_t first = 0,
    					size_t last = std::numeric_limits<size_t>::max()) {

    	if (bucketed.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    		throw std::invalid_argument("input is limited to max signed long in size");
    	// TODO: separate logic using a bit vector.

    	// ensure valid range
        size_t f = std::min(first, bucketed.size());
        size_t l = std::min(last, bucketed.size());
        assert((f <= l) && "first should not exceed last" );
        size_t len = l-f;

        // if input is empty, simply return
        if ((len == 0) || (bucketed.size() == 0) || (i2o.size() == 0)) return;

        assert(bucketed.size() == i2o.size());

        // saving elements into correct position
        // modeled after http://stackoverflow.com/questions/7365814/in-place-array-reordering
        // and https://blog.merovius.de/2014/08/12/applying-permutation-in-constant.html
        size_t j, k;
        T v;
//        size_t counter = 0;
        constexpr size_t mask = ~(0UL) << (sizeof(size_t) * 8 - 1);
        for (size_t i = f; i < l; ++i) {
        	k = i2o[i];				// position that the unbucketed was moved to in the bucketed list; source data
//        	++counter;

        	if (k >= mask) {
        		// previously visited.  unmark and move on.
        		i2o[i] = ~k;  // negate
        		continue;
        	}

        	// get the current position and value
        	v = bucketed[i];		// curr value - to be moved forward until its correct position is found
        	j = i;  				// curr position in bucketed list == , where the value is to be replaced.
        	while (k != i) {  // stop when cycle completes (return to original position)
//        	while ((k & mask) == 0) {  // avoid previously visited, including i.  each entry visited just once.  THIS DOES NOT WORK - k may not be set to negative yet.
        		// copy assign operator of the data type.  could be costly.
        		bucketed[j] = bucketed[k];   // move value back from the bucketed list to the unbucketed.
        		i2o[j] = ~k;  			  // mark the curr position as visited.

        		j = k;  				  // advance the curr position
        		k = i2o[j];				  // get the next position
//        		++counter;
        	}
        	bucketed[j] = v;		// cycle complete, swap back the current value (from some other location)
        	i2o[j] = ~k;		// cycle complete

        	i2o[i] = ~i2o[i];   // unmark the i2o map entry- never visiting prev i again.
        }

//        std::cout << "i2o read " << counter << " times.  size= " << i2o.size() << std::endl;
    }



  } // local namespace


  //== get bucket assignment. - complete grouping by processors (for a2av), or grouping by processors in communication blocks (for a2a, followed by 1 a2av)
  // parameter: in block per bucket send count (each block is for an all2all operation.  max block size is capped by max int.
  // 2 sets of send counts


  // need to 1. get assignment + bucket count
  //         2. globally compute block size
  //         3. do a blocks at a time ()
  // once the block permutation matrix is available, we need to permute the input completely (e.g. using the inplace permute and unpermute algorithm),
  //     or gather the part we need into local buffers (inplace gather and scatter does NOT make sense.  also, local buffers allows inplace MPI_alltoall)
  //

  /*
     *        from this data can be processed in a few ways:
     *          1. permute wholly in place, then communicate (wholly or in blocks) not in place.  up to 2x memory.
     *              communicate wholly is wasteful because of complete intermediate buffer is needed but may be faster.
     *              comm in blocks saves memory but may be slower.
     *              origin not preserved, permuted is preserved.  okay for count and exists.
     *              permute slower, comm faster
     *          2. permute wholly in place, then communicate (wholly or in blocks) in place for the first blocks.
     *              original not preserved. permuted not preserved.  okay for erase, update, insert, even find.
     *              permute slower, comm probably slower
     *          3. permute wholly not in place, then communicate not in place - might be faster.  3x memory now.
     *              original and permuted preserved.  output may be in permuted order.
     *              permute fast, comm fast
     *          4. permute wholly not in place, then communicate in place for first blocks.
     *              original preserved, permuted not preserved.  no ordering for count and exists
     *              permute fast, comm probably slower.
     *          5. gather locally then communicate .  still need local buffer.  probably slower than permute wholly in place since we need to search.  orig order preserved.
   *
   */




  /**
   * @brief       distribute.  speed version.  no guarantee of output ordering, but actually is same.
   * @param[IN] vals    vals to distribute.  should already be bucketed.
   * @param[IN/OUT] send_counts    count of each target bucket of input, and each source bucket after call.
   *
   * @return distributed values.
   */
//      template <typename V>
//      ::std::vector<V> distribute_bucketed(::std::vector<V> const & vals,
//              ::std::vector<size_t> & send_counts,
//                                       mxx::comm const &_comm) {
//        ::std::vector<size_t> temp(send_counts);
//        mxx::all2all(send_counts, _comm).swap(send_counts);
//
//          return mxx::all2allv(vals, temp, _comm);
//      }



// specialized for lower memory, incremental operations
// also  later, non-blocking mxx.
// requirements:  in place in original or fixed-size buffer(s)
// iterative processing




//== bucket actually.

//== all2allv using in place all2all as first step, then all2allv + buffer as last part.  pure communication.
// to use all2all, the data has to be contiguous.  2 ways to do this:  1. in place.  2. separate buffer
  /**
   * @brief use all2all to send 1 block of data, block = send_count * comm.size()
   *
   * @param input should be already bucketed and arranged as blocks with p * b, where b is size of each bucket.
   * @param send_count   number to send to each processor
   * @param output    buffer to store the result
   * @param send_offset    offset in input from which to start sending.  end of range is offset + send_count * comm.size()
   * @param recv_offset    offset in output from which to start writing.  end of range is offset + send_count * comm.size()
   * @param comm    communicator
   */
  template <typename T>
  void block_all2all(std::vector<T> const & input, size_t send_count, std::vector<T> & output,
                     size_t send_offset = 0, size_t recv_offset = 0, mxx::comm const & comm = mxx::comm()) {

	    bool empty = ((input.size() == 0) || (send_count == 0));
	    empty = mxx::all_of(empty);
	    if (empty) {
	      return;
	    }

    // ensure enough space
    assert((input.size() >= (send_offset + send_count * comm.size())) && "input for block_all2all not big enough");
    assert((output.size() >= (recv_offset + send_count * comm.size())) && "output for block_all2all not big enough");

    // send via mxx all2all - leverage any large message support from mxx.  (which uses datatype.contiguous() to increase element size and reduce element count to 1)
    ::mxx::all2all(&(input[send_offset]), send_count, &(output[recv_offset]), comm);
  }

  /**
   * @brief use all2all and all2allv to send data in blocks
   * @details   first part of the vector is sent via alltoall.  second part sent by all2allv.
   *            sends 1 block.
   *
   * @param input[in|out] should be already bucketed and arranged as blocks with p * b, where b is size of each bucket.
   * @param send_count   number to send to each processor
   * @param send_offset    offset in input from which to start sending.  end of range is offset + send_count * comm.size()
   * @param comm    communicator
   */
  template <typename T>
  void block_all2all_inplace(std::vector<T> & input, size_t send_count,
                     size_t offset = 0, mxx::comm const & comm = mxx::comm()) {

	    bool empty = ((input.size() == 0) || (send_count == 0));
	    empty = mxx::all_of(empty);
	    if (empty) {
	      return;
	    }


    // ensure enough space
    assert((input.size() >= (offset + send_count * comm.size())) && "input for block_all2all not big enough");
    assert((send_count < mxx::max_int) && "send count too large for mpi ");
//    assert((send_count * comm.size() < mxx::max_int) && "send count too large for mpi ");

    // get datatype based on size.
    mxx::datatype dt;
    if ((send_count * comm.size()) < mxx::max_int) {
      dt = mxx::get_datatype<T>();
    } else {
      // create a contiguous data type to support large messages (send_count * comm.size() > mxx::max_int).
      dt = mxx::get_datatype<T>().contiguous(send_count);
      send_count = 1;
    }

    // send using special mpi keyword MPI_IN_PLACE. should work for MPI_Alltoall.
    MPI_Alltoall(MPI_IN_PLACE, send_count, dt.type(), const_cast<T*>(&(input[offset])), send_count, dt.type(), comm);
  }


  /**
   * @brief distribute function.  input is transformed, but remains the original input with original order.  buffer is used for output.
   * @details
   * @tparam SIZE     type for the i2o mapping and recv counts.  should be large enough to represent max of input.size() and output.size()
   */
  template <typename V, typename ToRank, typename SIZE>
  void distribute(::std::vector<V>& input, ToRank const & to_rank,
                  ::std::vector<SIZE> & recv_counts,
                  ::std::vector<SIZE> & i2o,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm, bool const & preserve_input = false) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    std::vector<SIZE> send_counts(_comm.size(), 0);
    i2o.resize(input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_map", input.size());

    // bucketing
    BL_BENCH_COLLECTIVE_START(distribute, "bucket", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
    khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "bucket", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_resume();
#endif
    // distribute (communication part)
    khmxx::local::bucketId_to_pos(send_counts, i2o, 0, input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "to_pos", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    if (output.capacity() < input.size()) output.clear();
    output.resize(input.size());
    output.swap(input);  // swap the 2.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_permute", output.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_resume();
#endif
    // distribute (communication part)
    khmxx::local::permute(output.begin(), output.end(), i2o.begin(), input.begin(), 0);  // input now holds permuted entries.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "permute", input.size());

    // distribute (communication part)
    BL_BENCH_COLLECTIVE_START(distribute, "a2a_count", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    recv_counts.resize(_comm.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_count", recv_counts.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  size_t total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // now resize output
    if (output.capacity() < total) output.clear();
    output.resize(total);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "realloc_out", output.size());

//    std::cout << "send counts " << std::endl;
//    std::cout << _comm.rank() << ",";
//    for (int ii = 0; ii < _comm.size(); ++ii) {
//      std::cout << send_counts[ii] << ",";
//    }
//    std::cout << std::endl;
//    std::cout << "recv counts " << std::endl;
//    std::cout << _comm.rank() << ",";
//    for (int ii = 0; ii < _comm.size(); ++ii) {
//      std::cout << recv_counts[ii] << ",";
//    }
//    std::cout << std::endl;



    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2allv(input.data(), send_counts, output.data(), recv_counts, _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", output.size());

    if (preserve_input) {
        BL_BENCH_COLLECTIVE_START(distribute, "unpermute_inplace", _comm);
      // unpermute.  may be able to work around this so leave it as "_inplace"
      khmxx::local::unpermute_inplace(input, i2o, 0, input.size());
      BL_BENCH_END(distribute, "unpermute_inplace", input.size());
    }
    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);

  }

  template <typename V, typename ToRank, typename SIZE>
  void distribute(::std::vector<V>& input, ToRank const & to_rank,
                  ::std::vector<SIZE> & recv_counts,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    std::vector<SIZE> send_counts(_comm.size(), 0);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_map", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    if (output.capacity() < input.size()) output.clear();
    output.resize(input.size());
    output.swap(input);  // swap the 2.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_permute", output.size());

    // bucketing
    BL_BENCH_COLLECTIVE_START(distribute, "bucket", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
    size_t comm_size = _comm.size();
    if (comm_size <= std::numeric_limits<uint8_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast< uint8_t>(comm_size), send_counts, input, 0, output.size());
    } else if (comm_size <= std::numeric_limits<uint16_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint16_t>(comm_size), send_counts, input, 0, output.size());
    } else if (comm_size <= std::numeric_limits<uint32_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint32_t>(comm_size), send_counts, input, 0, output.size());
    } else {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint64_t>(comm_size), send_counts, input, 0, output.size());
    }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "bucket", input.size());


    // distribute (communication part)
    BL_BENCH_COLLECTIVE_START(distribute, "a2a_count", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    recv_counts.resize(_comm.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_count", recv_counts.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    size_t total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // now resize output
    if (output.capacity() < total) output.clear();
    output.resize(total);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "realloc_out", output.size());

    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2allv(input.data(), send_counts, output.data(), recv_counts, _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", output.size());

    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute_bucket", _comm);

  }


  /// distribute data that has been permuted and the counts are in send_counts.  recv_counts should contain the target counts
  template <typename T>
  void distribute_permuted(T* _begin, T* _end,
                   ::std::vector<size_t> const & send_counts,
				  T* output,
				   ::std::vector<size_t> & recv_counts,
                  ::mxx::comm const &_comm) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    size_t input_size = std::distance(_begin, _end);
    bool empty = (input_size == 0);
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input_size);

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2allv(_begin, send_counts, output, recv_counts, _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", input_size);

    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute_permuted", _comm);

  }


  template <typename V, typename SIZE>
  void undistribute(::std::vector<V> const & input,
                  ::std::vector<SIZE> const & recv_counts,
                  ::std::vector<SIZE> & i2o,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm, bool const & restore_order = true) {
    BL_BENCH_INIT(undistribute);

    BL_BENCH_COLLECTIVE_START(undistribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(undistribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(undistribute, "khmxx:undistribute", _comm);
      return;
    }


    BL_BENCH_START(undistribute);
    std::vector<size_t> send_counts(recv_counts.size());
    mxx::all2all(recv_counts.data(), 1, send_counts.data(), _comm);
    size_t total = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));
    BL_BENCH_END(undistribute, "recv_counts", input.size());

    BL_BENCH_START(undistribute);
    if (output.capacity() < total) output.clear();
    output.resize(total);
    BL_BENCH_COLLECTIVE_END(undistribute, "realloc_out", output.size(), _comm);

    BL_BENCH_START(undistribute);
    mxx::all2allv(input.data(), recv_counts, output.data(), send_counts, _comm);
    BL_BENCH_END(undistribute, "a2av", input.size());

    if (restore_order) {
      BL_BENCH_START(undistribute);
      // distribute (communication part)
      khmxx::local::unpermute_inplace(output, i2o, 0, output.size());
      BL_BENCH_END(undistribute, "unpermute_inplace", output.size());

    }
    BL_BENCH_REPORT_MPI_NAMED(undistribute, "khmxx:undistribute", _comm);

  }

  /**
   * @brief distribute function.  input is transformed, but remains the original input with original order.  buffer is used for output.
   *
   */
  template <typename V, typename ToRank, typename SIZE>
  void distribute_2part(::std::vector<V>& input, ToRank const & to_rank,
                        ::std::vector<SIZE> & recv_counts,
                        ::std::vector<SIZE> & i2o,
                        ::std::vector<V>& output,
                        ::mxx::comm const &_comm, bool const & preserve_input = false) {
    BL_BENCH_INIT(distribute);

      // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.
    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute_2", _comm);
      return;
    }

      // do assignment.
      BL_BENCH_START(distribute);
      std::vector<SIZE> send_counts(_comm.size(), 0);
      i2o.resize(input.size());
      BL_BENCH_END(distribute, "alloc_map", input.size());

      // bucketing
      BL_BENCH_START(distribute);
      khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
      BL_BENCH_END(distribute, "bucket", input.size());

      // compute minimum block size.
      BL_BENCH_START(distribute);
      SIZE min_bucket_size = *(::std::min_element(send_counts.begin(), send_counts.end()));
      min_bucket_size = ::mxx::allreduce(min_bucket_size, mxx::min<SIZE>(), _comm);
      BL_BENCH_END(distribute, "min_bucket_size", min_bucket_size);

      // compute the permutations from block size and processor mapping.  send_counts modified to the remainders.
      BL_BENCH_START(distribute);
      ::khmxx::local::blocked_bucketId_to_pos(min_bucket_size, 1UL, send_counts, i2o, 0, input.size());
      SIZE first_part = _comm.size() * min_bucket_size;
      BL_BENCH_END(distribute, "to_pos", input.size());

      // compute receive counts and total
      BL_BENCH_START(distribute);
      recv_counts.resize(_comm.size());
      mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
      SIZE total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
      total += first_part;
      BL_BENCH_END(distribute, "a2av_count", total);

      BL_BENCH_START(distribute);
      if (output.capacity() < input.size()) output.clear();
      output.resize(input.size());
      output.swap(input);
      BL_BENCH_END(distribute, "alloc_out", output.size());

      // permute
      BL_BENCH_START(distribute);
      khmxx::local::permute(output.begin(), output.end(), i2o.begin(), input.begin(), 0);
      BL_BENCH_END(distribute, "permute", input.size());

      BL_BENCH_START(distribute);
      if (output.capacity() < total) output.clear();
      output.resize(total);
      BL_BENCH_END(distribute, "alloc_out", output.size());

      BL_BENCH_START(distribute);
      block_all2all(input, min_bucket_size, output, 0, 0, _comm);
      BL_BENCH_END(distribute, "a2a", first_part);


      BL_BENCH_START(distribute);
      mxx::all2allv(input.data() + first_part, send_counts,
                    output.data() + first_part, recv_counts, _comm);
      BL_BENCH_END(distribute, "a2av", total - first_part);

      // permute
      if (preserve_input) {
        BL_BENCH_START(distribute);
        ::khmxx::local::unpermute_inplace(input, i2o, 0, input.size());
        BL_BENCH_END(distribute, "unpermute_inplace", input.size());
      }

      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute_2", _comm);
  }


  /**
   * @param recv_counts  counts for each bucket that is NOT PART OF FIRST BLOCK.
   */
  template <typename V, typename SIZE>
  void undistribute_2part(::std::vector<V> const & input,
                  ::std::vector<SIZE> const & recv_counts,
                  ::std::vector<SIZE> & i2o,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm, bool const & restore_order = true) {
    BL_BENCH_INIT(undistribute);

    BL_BENCH_COLLECTIVE_START(undistribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(undistribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(undistribute, "khmxx:undistribute_2", _comm);
      return;
    }


    BL_BENCH_START(undistribute);
    size_t second_part = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    size_t first_part = input.size() - second_part;
    assert((first_part % _comm.size() == 0) && "the first block should be evenly distributed to buckets.");

    std::vector<size_t> send_counts(recv_counts.size());
    mxx::all2all(recv_counts.data(), 1, send_counts.data(), _comm);
    second_part = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));
    BL_BENCH_COLLECTIVE_END(undistribute, "recv_counts", second_part, _comm);

    BL_BENCH_START(undistribute);
    if (output.capacity() < (first_part + second_part)) output.clear();
    output.resize(first_part + second_part);
    BL_BENCH_COLLECTIVE_END(undistribute, "realloc_out", output.size(), _comm);

    BL_BENCH_START(undistribute);
    mxx::all2all(input.data(), first_part / _comm.size(), output.data(), _comm);
    BL_BENCH_END(undistribute, "a2a", first_part);

    BL_BENCH_START(undistribute);
    mxx::all2allv(input.data() + first_part, recv_counts, output.data() + first_part, send_counts, _comm);
    BL_BENCH_END(undistribute, "a2av", second_part);

    if (restore_order) {
      BL_BENCH_START(undistribute);
      // distribute (communication part)
      khmxx::local::unpermute_inplace(output, i2o, 0, output.size());
      BL_BENCH_END(undistribute, "unpermute_inplace", output.size());

    }
    BL_BENCH_REPORT_MPI_NAMED(undistribute, "khmxx:undistribute_2", _comm);

  }

  namespace lz4 {



  /**
   * @brief distribute function.  input is transformed, but remains the original input with original order.  buffer is used for output.
   * @details
   * @tparam SIZE     type for the i2o mapping and recv counts.  should be large enough to represent max of input.size() and output.size()
   */
  template <typename V, typename ToRank, typename SIZE>
  void distribute(::std::vector<V>& input, ToRank const & to_rank,
                  ::std::vector<SIZE> & recv_counts,
                  ::std::vector<SIZE> & i2o,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm, bool const & preserve_input = false) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    std::vector<SIZE> send_counts(_comm.size(), 0);
    i2o.resize(input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_map", input.size());

    // bucketing
    BL_BENCH_COLLECTIVE_START(distribute, "bucket", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
    khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "bucket", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_resume();
#endif
    // distribute (communication part)
    khmxx::local::bucketId_to_pos(send_counts, i2o, 0, input.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "to_pos", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    if (output.capacity() < input.size()) output.clear();
    output.resize(input.size());
    output.swap(input);  // swap the 2.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_permute", output.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_resume();
#endif
    // distribute (communication part)
    khmxx::local::permute(output.begin(), output.end(), i2o.begin(), input.begin(), 0);  // input now holds permuted entries.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_PERMUTE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "permute", input.size());

    //============= compress each input block now...
    BL_BENCH_COLLECTIVE_START(distribute, "alloc_lz4", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    // allocate the compressed byte count
    std::vector<int> compressed_send_bytes;
    compressed_send_bytes.reserve(_comm.size());
    std::vector<int> aligned_send_counts;
    aligned_send_counts.reserve(_comm.size());

    // compute for each target rank the estimated max size.
    size_t max_comp_total = 0;
    for (size_t i = 0; i < send_counts.size(); ++i) {
    	if ((send_counts[i] * sizeof(V)) >= (1ULL << 31))
    		throw std::logic_error("individual block size is more than 2^31 bytes (int)");

    	compressed_send_bytes.emplace_back(LZ4_compressBound(send_counts[i] * sizeof(V)));
    	aligned_send_counts.emplace_back((compressed_send_bytes.back() + 7) & 0xFFFFFFF8);  // remove trailing 3 bits - same as removing remainders.
    	max_comp_total += aligned_send_counts.back();
    }

    // then allocate the output space
	char* compressed = nullptr;
	int ret = posix_memalign(reinterpret_cast<void **>(&compressed), 64, max_comp_total);
	if (ret) {
		free(compressed);
		throw std::length_error("failed to allocate aligned memory");
	}
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_lz4", max_comp_total);

    // now compress block by block
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_comp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    size_t offset = 0;   // element offset
    size_t compressed_offset = 0;  // byte offset
    for (size_t i = 0; i < send_counts.size(); ++i) {
    	compressed_send_bytes[i] =
    			LZ4_compress_default(reinterpret_cast<const char *>(input.data() + offset),
    					compressed + compressed_offset, send_counts[i] * sizeof(V), compressed_send_bytes[i]);

    	if (compressed_send_bytes[i] < 0) {
    		throw std::logic_error("failed to compress");
    	} else if (compressed_send_bytes[i] == 0) {
    		throw std::logic_error("out of space for compression");
    	}

    	aligned_send_counts[i] = (compressed_send_bytes[i] + 7) & 0xFFFFFFF8;

    	compressed_offset += aligned_send_counts[i];
    	offset += send_counts[i];  // advance to next block
    }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_comp", offset);


    // distribute (communication part)
    BL_BENCH_COLLECTIVE_START(distribute, "a2a_count", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    recv_counts.resize(_comm.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);

    std::vector<int> compressed_recv_bytes = mxx::all2all(compressed_send_bytes.data(), 1, _comm);
    std::vector<int> aligned_recv_counts = mxx::all2all(aligned_send_counts.data(), 1, _comm);

    std::vector<int> aligned_send_displs = mxx::impl::get_displacements(aligned_send_counts);
    std::vector<int> aligned_recv_displs = mxx::impl::get_displacements(aligned_recv_counts);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_count", recv_counts.size());


//    std::cout << "send counts " << std::endl;
//    std::cout << _comm.rank() << ",";
//    for (int ii = 0; ii < _comm.size(); ++ii) {
//      std::cout << send_counts[ii] << ",";
//    }
//    std::cout << std::endl;
//    std::cout << "recv counts " << std::endl;
//    std::cout << _comm.rank() << ",";
//    for (int ii = 0; ii < _comm.size(); ++ii) {
//      std::cout << recv_counts[ii] << ",";
//    }
//    std::cout << std::endl;


    // alloc compressed output.
    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    max_comp_total = 0;
    for (size_t i = 0; i < aligned_recv_counts.size(); ++i) {
    	max_comp_total += aligned_recv_counts[i];
    }
    // allocate recv_compressed
	char* recv_compressed = nullptr;
	ret = posix_memalign(reinterpret_cast<void **>(&recv_compressed), 64, max_comp_total);
	if (ret) {
		free(recv_compressed);
		throw std::length_error("failed to allocate aligned memory");
	}
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_alloc", max_comp_total);

    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
  	MPI_Alltoallv(compressed, aligned_send_counts.data(),
  			aligned_send_displs.data(),
  			MPI_BYTE,
            recv_compressed, aligned_recv_counts.data(),
			aligned_recv_displs.data(),
			MPI_BYTE, _comm);
    free(compressed);
    //mxx::all2allv(compressed, compressed_send_bytes, recv_compressed, compressed_recv_bytes, _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", max_comp_total);

//	clear the src.


    // allocate output
    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
  	size_t total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // now resize output
    if (output.capacity() < total) output.clear();
    output.resize(total);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "realloc_out", output.size());

    // decompress received
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_decomp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    offset = 0;   // element offset
    compressed_offset = 0;  // byte offset
    int decomp_size;
    for (size_t i = 0; i < compressed_recv_bytes.size(); ++i) {
    	decomp_size = LZ4_decompress_safe(recv_compressed + compressed_offset,
    			reinterpret_cast<char *>(output.data() + offset), compressed_recv_bytes[i],
				recv_counts[i] * sizeof(V));

    	if (decomp_size < 0) {
    		throw std::logic_error("failed to decompress");
    	} else if (decomp_size == 0) {
    		throw std::logic_error("out of space for decompression");
    	} else if (decomp_size != static_cast<int>(recv_counts[i] * sizeof(V)))
    		throw std::logic_error("decompression generated different size than expected.");

    	compressed_offset += aligned_recv_counts[i];
    	offset += recv_counts[i];  // advance to next block
    }

    free(recv_compressed);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_decomp", offset);


    if (preserve_input) {
        BL_BENCH_COLLECTIVE_START(distribute, "unpermute_inplace", _comm);
      // unpermute.  may be able to work around this so leave it as "_inplace"
      khmxx::local::unpermute_inplace(input, i2o, 0, input.size());
      BL_BENCH_END(distribute, "unpermute_inplace", input.size());
    }
    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);

  }


  template <typename V, typename ToRank, typename SIZE>
  void distribute(::std::vector<V>& input, ToRank const & to_rank,
                  ::std::vector<SIZE> & recv_counts,
                  ::std::vector<V>& output,
                  ::mxx::comm const &_comm) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    bool empty = input.size() == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input.size());

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    std::vector<SIZE> send_counts(_comm.size(), 0);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_map", input.size());

    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    if (output.capacity() < input.size()) output.clear();
    output.resize(input.size());
    output.swap(input);  // swap the 2.
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_permute", output.size());

    // bucketing
    BL_BENCH_COLLECTIVE_START(distribute, "bucket", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_resume();
#endif
    size_t comm_size = _comm.size();
    if (comm_size <= std::numeric_limits<uint8_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast< uint8_t>(comm_size), send_counts, input, 0, output.size());
    } else if (comm_size <= std::numeric_limits<uint16_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint16_t>(comm_size), send_counts, input, 0, output.size());
    } else if (comm_size <= std::numeric_limits<uint32_t>::max()) {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint32_t>(comm_size), send_counts, input, 0, output.size());
    } else {
      khmxx::local::bucketing_impl(output, to_rank, static_cast<uint64_t>(comm_size), send_counts, input, 0, output.size());
    }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_BUCKET)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "bucket", input.size());


    //============= compress each input block now...
    BL_BENCH_COLLECTIVE_START(distribute, "alloc_lz4", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    // allocate the compressed byte count
    std::vector<int> compressed_send_bytes;
    compressed_send_bytes.reserve(_comm.size());
    std::vector<int> aligned_send_counts;
    aligned_send_counts.reserve(_comm.size());

    // compute for each target rank the estimated max size.
    size_t max_comp_total = 0;
    for (size_t i = 0; i < send_counts.size(); ++i) {
      if ((send_counts[i] * sizeof(V)) >= (1ULL << 31))
        throw std::logic_error("individual block size is more than 2^31 bytes (int)");

      compressed_send_bytes.emplace_back(LZ4_compressBound(send_counts[i] * sizeof(V)));
      aligned_send_counts.emplace_back((compressed_send_bytes.back() + 7) & 0xFFFFFFF8);  // remove trailing 3 bits - same as removing remainders.
      max_comp_total += aligned_send_counts.back();
    }

    // then allocate the output space
  char* compressed = nullptr;
  int ret = posix_memalign(reinterpret_cast<void **>(&compressed), 64, max_comp_total);
  if (ret) {
    free(compressed);
    throw std::length_error("failed to allocate aligned memory");
  }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_lz4", max_comp_total);

    // now compress block by block
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_comp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    size_t offset = 0;   // element offset
    size_t compressed_offset = 0;  // byte offset
    for (size_t i = 0; i < send_counts.size(); ++i) {
      compressed_send_bytes[i] =
          LZ4_compress_default(reinterpret_cast<const char *>(input.data() + offset),
              compressed + compressed_offset, send_counts[i] * sizeof(V), compressed_send_bytes[i]);

      if (compressed_send_bytes[i] < 0) {
        throw std::logic_error("failed to compress");
      } else if (compressed_send_bytes[i] == 0) {
        throw std::logic_error("out of space for compression");
      }

      aligned_send_counts[i] = (compressed_send_bytes[i] + 7) & 0xFFFFFFF8;

      compressed_offset += aligned_send_counts[i];
      offset += send_counts[i];  // advance to next block
    }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_comp", offset);



    // distribute (communication part)
    BL_BENCH_COLLECTIVE_START(distribute, "a2a_count", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    recv_counts.resize(_comm.size());
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);

    std::vector<int> compressed_recv_bytes = mxx::all2all(compressed_send_bytes.data(), 1, _comm);
    std::vector<int> aligned_recv_counts = mxx::all2all(aligned_send_counts.data(), 1, _comm);

    std::vector<int> aligned_send_displs = mxx::impl::get_displacements(aligned_send_counts);
    std::vector<int> aligned_recv_displs = mxx::impl::get_displacements(aligned_recv_counts);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_count", recv_counts.size());


    // alloc compressed output.
    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    max_comp_total = 0;
    for (size_t i = 0; i < aligned_recv_counts.size(); ++i) {
      max_comp_total += aligned_recv_counts[i];
    }
    // allocate recv_compressed
  char* recv_compressed = nullptr;
  ret = posix_memalign(reinterpret_cast<void **>(&recv_compressed), 64, max_comp_total);
  if (ret) {
    free(recv_compressed);
    throw std::length_error("failed to allocate aligned memory");
  }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_alloc", max_comp_total);


    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
  MPI_Alltoallv(compressed, aligned_send_counts.data(),
      aligned_send_displs.data(),
      MPI_BYTE,
          recv_compressed, aligned_recv_counts.data(),
    aligned_recv_displs.data(),
    MPI_BYTE, _comm);
  free(compressed);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", output.size());



    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    size_t total = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // now resize output
    if (output.capacity() < total) output.clear();
    output.resize(total);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "realloc_out", output.size());



    // decompress received
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_decomp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    offset = 0;   // element offset
    compressed_offset = 0;  // byte offset
    int decomp_size;
    for (size_t i = 0; i < compressed_recv_bytes.size(); ++i) {
      decomp_size = LZ4_decompress_safe(recv_compressed + compressed_offset,
          reinterpret_cast<char *>(output.data() + offset), compressed_recv_bytes[i],
        recv_counts[i] * sizeof(V));

      if (decomp_size < 0) {
        throw std::logic_error("failed to decompress");
      } else if (decomp_size == 0) {
        throw std::logic_error("out of space for decompression");
      } else if (decomp_size != static_cast<int>(recv_counts[i] * sizeof(V)))
        throw std::logic_error("decompression generated different size than expected.");

      compressed_offset += aligned_recv_counts[i];
      offset += recv_counts[i];  // advance to next block
    }

    free(recv_compressed);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_decomp", offset);


    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute_bucket", _comm);

  }



  template <typename T>
  void distribute_permuted(T* _begin, T* _end,
                  ::std::vector<size_t> & send_counts,
	                  T* output,
	              ::std::vector<size_t> & recv_counts,
                  ::mxx::comm const &_comm) {
    BL_BENCH_INIT(distribute);

    BL_BENCH_COLLECTIVE_START(distribute, "empty", _comm);
    size_t input_size = std::distance(_begin, _end);
    bool empty = input_size == 0;
    empty = mxx::all_of(empty);
    BL_BENCH_END(distribute, "empty", input_size);

    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:distribute", _comm);
      return;
    }
    // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.

    //============= compress each input block now...
    BL_BENCH_COLLECTIVE_START(distribute, "alloc_lz4", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    // allocate the compressed byte count
    std::vector<int> compressed_send_bytes;
    compressed_send_bytes.reserve(_comm.size());
    std::vector<int> aligned_send_counts;
    aligned_send_counts.reserve(_comm.size());

    // compute for each target rank the estimated max size.
    size_t max_comp_total = 0;
    for (size_t i = 0; i < send_counts.size(); ++i) {
      if ((send_counts[i] * sizeof(T)) >= (1ULL << 31))
        throw std::logic_error("individual block size is more than 2^31 bytes (int)");

      compressed_send_bytes.emplace_back(LZ4_compressBound(send_counts[i] * sizeof(T)));
      aligned_send_counts.emplace_back((compressed_send_bytes.back() + 7) & 0xFFFFFFF8);  // remove trailing 3 bits - same as removing remainders.
      max_comp_total += aligned_send_counts.back();
    }

    // then allocate the output space
  char* compressed = nullptr;
  int ret = posix_memalign(reinterpret_cast<void **>(&compressed), 64, max_comp_total);
  if (ret) {
    free(compressed);
    throw std::length_error("failed to allocate aligned memory");
  }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "alloc_lz4", max_comp_total);

    // now compress block by block
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_comp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    size_t offset = 0;   // element offset
    size_t compressed_offset = 0;  // byte offset
    for (size_t i = 0; i < send_counts.size(); ++i) {
      compressed_send_bytes[i] =
          LZ4_compress_default(reinterpret_cast<const char *>(_begin + offset),
              compressed + compressed_offset, send_counts[i] * sizeof(T), compressed_send_bytes[i]);

      if (compressed_send_bytes[i] < 0) {
        throw std::logic_error("failed to compress");
      } else if (compressed_send_bytes[i] == 0) {
        throw std::logic_error("out of space for compression");
      }

      aligned_send_counts[i] = (compressed_send_bytes[i] + 7) & 0xFFFFFFF8;

      compressed_offset += aligned_send_counts[i];
      offset += send_counts[i];  // advance to next block
    }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_comp", offset);



    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
    std::vector<int> compressed_recv_bytes = mxx::all2all(compressed_send_bytes.data(), 1, _comm);
    std::vector<int> aligned_recv_counts = mxx::all2all(aligned_send_counts.data(), 1, _comm);

    std::vector<int> aligned_send_displs = mxx::impl::get_displacements(aligned_send_counts);
    std::vector<int> aligned_recv_displs = mxx::impl::get_displacements(aligned_recv_counts);

#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_count", recv_counts.size());


    // alloc compressed output.
    BL_BENCH_START(distribute);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_resume();
#endif
    max_comp_total = 0;
    for (size_t i = 0; i < aligned_recv_counts.size(); ++i) {
      max_comp_total += aligned_recv_counts[i];
    }
    // allocate recv_compressed
  char* recv_compressed = nullptr;
  ret = posix_memalign(reinterpret_cast<void **>(&recv_compressed), 64, max_comp_total);
  if (ret) {
    free(recv_compressed);
    throw std::length_error("failed to allocate aligned memory");
  }
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_RESERVE)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a_alloc", max_comp_total);


    BL_BENCH_COLLECTIVE_START(distribute, "a2a", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_resume();
#endif
  MPI_Alltoallv(compressed, aligned_send_counts.data(),
      aligned_send_displs.data(),
      MPI_BYTE,
          recv_compressed, aligned_recv_counts.data(),
    aligned_recv_displs.data(),
    MPI_BYTE, _comm);
  free(compressed);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_A2A)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "a2a", compressed_offset);


    // decompress received
    BL_BENCH_COLLECTIVE_START(distribute, "lz4_decomp", _comm);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_resume();
#endif
    offset = 0;   // element offset
    compressed_offset = 0;  // byte offset
    int decomp_size;
    for (size_t i = 0; i < compressed_recv_bytes.size(); ++i) {
      decomp_size = LZ4_decompress_safe(recv_compressed + compressed_offset,
          reinterpret_cast<char *>(output + offset), compressed_recv_bytes[i],
        recv_counts[i] * sizeof(T));

      if (decomp_size < 0) {
        throw std::logic_error("failed to decompress");
      } else if (decomp_size == 0) {
        throw std::logic_error("out of space for decompression");
      } else if (decomp_size != static_cast<int>(recv_counts[i] * sizeof(T)))
        throw std::logic_error("decompression generated different size than expected.");

      compressed_offset += aligned_recv_counts[i];
      offset += recv_counts[i];  // advance to next block
    }

    free(recv_compressed);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_COMPRESS)
      __itt_pause();
#endif
    BL_BENCH_END(distribute, "lz4_decomp", offset);


    BL_BENCH_REPORT_MPI_NAMED(distribute, "khmxx:lz4_distribute_permuted", _comm);

  }

  }  // namespace lz4


  /**
   * @brief distribute, compute, send back.  one to one.  result matching input in order at then end.
   * @detail   this is the memory inefficient version
   *
   *
   */
  template <typename V, typename ToRank, typename Operation, typename SIZE = size_t,
      typename T = typename bliss::functional::function_traits<Operation, V>::return_type>
  void scatter_compute_gather(::std::vector<V>& input, ToRank const & to_rank,
                              Operation const & op,
                              ::std::vector<SIZE> & i2o,
                              ::std::vector<T>& output,
                              ::std::vector<V>& in_buffer, std::vector<T>& out_buffer,
                              ::mxx::comm const &_comm,
                              bool const & preserve_input = false) {
      BL_BENCH_INIT(scat_comp_gath);

      // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.
      BL_BENCH_COLLECTIVE_START(scat_comp_gath, "empty", _comm);
      bool empty = input.size() == 0;
      empty = mxx::all_of(empty);
      BL_BENCH_END(scat_comp_gath, "empty", input.size());

      if (empty) {
        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath, "khmxx:scat_comp_gath", _comm);
        return;
      }

      // do assignment.
      BL_BENCH_START(scat_comp_gath);
      std::vector<SIZE> recv_counts(_comm.size(), 0);
      i2o.resize(input.size());
      BL_BENCH_END(scat_comp_gath, "alloc_map", input.size());

      // distribute
      BL_BENCH_START(scat_comp_gath);
      distribute(input, to_rank, recv_counts, i2o, in_buffer, _comm, false);
      BL_BENCH_END(scat_comp_gath, "distribute", in_buffer.size());

      // allocate out_buffer - output is same size as input
      BL_BENCH_START(scat_comp_gath);
      if (out_buffer.capacity() < (in_buffer.size())) out_buffer.clear();
      out_buffer.resize(in_buffer.size());
      BL_BENCH_END(scat_comp_gath, "alloc_outbuf", out_buffer.size());

      // process
      BL_BENCH_START(scat_comp_gath);
      op(in_buffer.begin(), in_buffer.end(), out_buffer.begin());
      BL_BENCH_END(scat_comp_gath, "compute", out_buffer.size());

      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath);
      if (output.capacity() < (input.size())) output.clear();
      output.resize(input.size());
      BL_BENCH_END(scat_comp_gath, "alloc_out", output.size());

      // distribute data back to source
      BL_BENCH_START(scat_comp_gath);
      undistribute(out_buffer, recv_counts, i2o, output, _comm, false);
      BL_BENCH_END(scat_comp_gath, "undistribute", output.size());


      // permute
      if (preserve_input) {
        BL_BENCH_START(scat_comp_gath);
        ::khmxx::local::unpermute(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), 0);
        in_buffer.swap(input);
        ::khmxx::local::unpermute(output.begin(), output.end(), i2o.begin(), out_buffer.begin(), 0);
        out_buffer.swap(output);
        BL_BENCH_END(scat_comp_gath, "unpermute_inplace", output.size());
      }

      BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath, "khmxx:scat_comp_gath", _comm);
  }


  /**
   * @brief distribute, compute, send back.  one to one.  result matching input in order at then end.
   * @details  this is the memory efficient version.  this has to be incremental.
   *            this version uses a permute buffer. (in_buffer)
   */
  template <typename V, typename ToRank, typename Operation, typename SIZE = size_t,
      typename T = typename bliss::functional::function_traits<Operation, V>::return_type>
  void scatter_compute_gather_2part(::std::vector<V>& input, ToRank const & to_rank,
                              Operation const & op,
                              ::std::vector<SIZE> & i2o,
                              ::std::vector<T>& output,
                              ::std::vector<V>& in_buffer, std::vector<T>& out_buffer,
                              ::mxx::comm const &_comm,
                              bool const & preserve_input = false) {
      BL_BENCH_INIT(scat_comp_gath_2);

      // speed over mem use.  mxx all2allv already has to double memory usage. same as stable scat_comp_gath_2.
      BL_BENCH_COLLECTIVE_START(scat_comp_gath_2, "empty", _comm);
      bool empty = input.size() == 0;
      empty = mxx::all_of(empty);
      BL_BENCH_END(scat_comp_gath_2, "empty", input.size());

      if (empty) {
        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_2, "khmxx:scat_comp_gath_2", _comm);
        return;
      }

      // do assignment.
      BL_BENCH_START(scat_comp_gath_2);
      std::vector<SIZE> send_counts(_comm.size(), 0);
      std::vector<SIZE> recv_counts(_comm.size(), 0);
      i2o.resize(input.size());
      BL_BENCH_END(scat_comp_gath_2, "alloc_map", input.size());

      // first bucketing
      BL_BENCH_START(scat_comp_gath_2);
      khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
      BL_BENCH_END(scat_comp_gath_2, "bucket", input.size());

      // then compute minimum block size.
      BL_BENCH_START(scat_comp_gath_2);
      SIZE min_bucket_size = *(::std::min_element(send_counts.begin(), send_counts.end()));
      min_bucket_size = ::mxx::allreduce(min_bucket_size, mxx::min<SIZE>(), _comm);
      SIZE first_part = _comm.size() * min_bucket_size;
      BL_BENCH_END(scat_comp_gath_2, "min_bucket_size", first_part);

      // compute the permutations from block size and processor mapping.  send_counts modified to the remainders.
      BL_BENCH_START(scat_comp_gath_2);
      ::khmxx::local::blocked_bucketId_to_pos(min_bucket_size, 1UL, send_counts, i2o, 0, input.size());
      BL_BENCH_END(scat_comp_gath_2, "to_pos", input.size());

      // compute receive counts and total
      BL_BENCH_START(scat_comp_gath_2);
      recv_counts.resize(_comm.size());
      mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
      SIZE second_part = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
      BL_BENCH_END(scat_comp_gath_2, "a2av_count", second_part);

      // allocate input buffer (for permuted data.)
      BL_BENCH_START(scat_comp_gath_2);
      if (in_buffer.capacity() < input.size()) in_buffer.clear();
      in_buffer.resize(input.size());
      BL_BENCH_END(scat_comp_gath_2, "alloc_inbuf", in_buffer.size());

      // permute
      BL_BENCH_START(scat_comp_gath_2);
      khmxx::local::permute(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), 0);
      in_buffer.swap(input);       // input is now permuted.
//      in_buffer.resize(second_part);
      BL_BENCH_END(scat_comp_gath_2, "permute", input.size());


      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath_2);
      if (output.capacity() < (input.size())) output.clear();
      output.resize(input.size());
      BL_BENCH_END(scat_comp_gath_2, "alloc_out", output.size());

      //== process first part.  communicate in place
      BL_BENCH_START(scat_comp_gath_2);
      block_all2all(input, min_bucket_size, in_buffer, 0, 0, _comm);
      BL_BENCH_END(scat_comp_gath_2, "a2a_inplace", first_part);

      // process
      BL_BENCH_START(scat_comp_gath_2);
      op(in_buffer.begin(), in_buffer.begin() + first_part, output.begin());
      BL_BENCH_END(scat_comp_gath_2, "compute1", first_part);

      // send the results back.  and reverse the input
      // undo a2a, so that result data matches.
      BL_BENCH_START(scat_comp_gath_2);
      block_all2all_inplace(output, min_bucket_size, 0, _comm);
      BL_BENCH_END(scat_comp_gath_2, "inverse_a2a_inplace", first_part);

      //======= process the second part
      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath_2);
      if (out_buffer.capacity() < second_part) out_buffer.clear();
      out_buffer.resize(second_part);
      in_buffer.resize(second_part);
      BL_BENCH_END(scat_comp_gath_2, "alloc_outbuf", out_buffer.size());

      // send second part.  reuse entire in_buffer
      BL_BENCH_START(scat_comp_gath_2);
	  mxx::all2allv(input.data() + first_part, send_counts,
                    in_buffer.data(), recv_counts, _comm);
      BL_BENCH_END(scat_comp_gath_2, "a2av", in_buffer.size());

      // process the second part.
      BL_BENCH_START(scat_comp_gath_2);
      op(in_buffer.begin(), in_buffer.begin() + second_part, out_buffer.begin());
      BL_BENCH_END(scat_comp_gath_2, "compute2", out_buffer.size());

      // send the results back
      BL_BENCH_START(scat_comp_gath_2);
	  mxx::all2allv(out_buffer.data(), recv_counts,
                    output.data() + first_part, send_counts, _comm);
      BL_BENCH_END(scat_comp_gath_2, "inverse_a2av", output.size());


      // permute
      if (preserve_input) {
        BL_BENCH_START(scat_comp_gath_2);
        // in_buffer was already allocated to be same size as input.
        ::khmxx::local::unpermute(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), 0);
        in_buffer.swap(input);
        // out_buffer is small, so should do this inplace.
        ::khmxx::local::unpermute_inplace(output, i2o, 0, output.size());
        BL_BENCH_END(scat_comp_gath_2, "unpermute_inplace", output.size());
      }

      BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_2, "khmxx:scat_comp_gath_2", _comm);
  }

  /**
   * @brief distribute, compute, send back.  one to one.  result matching input in order at then end.
   * @details  this is the memory efficient version.  this has to be incremental.
   *            this version uses a permute buffer. (in_buffer)
   *
   *            low mem version.
   *
   *            the second half (a2av) require the space it requires.  for in_buffer and out_buffer.
   *            the first half can be reduced in space usage by incrementally populate the in_buffer,
   *            	compute the output, then populate the output.  then 1 all2all inplace for the output
   *            	at the requester, then unpermute somehow.
   *
      // compute the size of the buffers to use.  set a maximum limit.
       * let the final block_bucket_size be bbs. then the first_part x = bbs * blocks * p., p = comm_size.
       * let the sum of UNEVEN parts in orig input be y1, and after a2av be y2.  the minimum space needed here is y1+y2 for the a2av part.
       *   the uneven part is defined by the difference between a bucket and the global min bucket.
       *
       * APPROACH:  use either y1+y2 or 1/2 min_bucket, whichever is larger, as buffer
       * 	if y1+y2 > 1/2 (input + y2), use traditional
       * 	else
       * 		use larger of y1+y2, and 1/2 min bucket.
       *
       *
       * however, bbs may not perfectly divide the min_bucket..  let the remainder be r.  0 <= r < bbs
       * the UNEVEN part then needs to be appended with r in each bucket.  Y1 = y1 + r * p, and Y2 = y2 + r * p.  post communication, the remainder size remains same.
       * minimum space needed is revised to Y1 + Y2 = y1 + y2 + 2 rp.
       *
       * if y1 + y2 > input_size, then we should just the normal scatter_compute_gather, or the 2 part version, since buffer requirement is high.
       *
       * case 1
       * bbs * p > Y1 + Y2.  each block still needs to got through a2a, so makes no sense to have more than 1 blocks in the buffer and we should reduce the number of iterations
       * to have some form of savings, we should use bbs * p < input.  then input > Y2 + Y2.  if this does not hold, then use full memory version.
       *
       * for all processors, the following must hold to have buffer savings.
       * y1 + rp + bbs * blocks * p > bbs * p > y1 + y2 + 2rp.   r + bbs * blocks = min_bucket_size.
       * 	p * min_bucket_size > y2 + 2rp,  so r < min_bucket_size - y2 / 2p,
       * 	mod(min_bucket_size, bbs) = r < bbs.
       * so let bbs = min_bucket_size - y2 / 2p.  (bbs can be at most this quantity, else r may exceed bound)
       * 	now this can be pretty big, or really small.
       *
       * what is the lower bound of bbs?
       *
       *
       *
       *
       *
       * for there to be space savings, y1+y2 is the minimum buffer size and the maximum buffer size should not exceed 1/2 of input
       * buffer size should be std::max(y1+y2, comm_size * block_bucket_size)   1 block...
       * where block_bucket_size = std::min(min_bucket_size, 1/2 max_bucket_size)    // at most min_bucket_size, at least 1/2 max_bucket_size
       * this will affect y1 and y2, of course.
       * 	y1 has size input % (comm_size * block_bucket_size), with max value of (comm_size * block_bucket_size - 1)
       * 	y2 has max comm_size * (comm_size  * block_bucket_size - 1 ) - 1
       *
       * note at the end, we'd still need to permute either the input or the output to make them consistent with each other.
      // max of "second_part" is the minimum required space for the a2av portion.  let this be y.
      //    reducing this requires O(p) type communication rather than O(log(p))
      // to effect some space savings, we should use max(1/2 input_size, y).
      //	we can reduce the 1/2 input size, at the expense of more iterations, each requiring a complete scan of input during permuting.
      //	if y > 1/2 input, then on one processor the min bucket is less than 1/2 input,  we'd be using this min bucket anyways.
      //	if y < 1/2 input, then we can potentially do more in a2a phase, but that would use more mem.
      // so configure the buffer size to be max(1/2 input_size, y) = Y.
      //    for bucket size, this means min(min_bucket, (Y + comm_size - 1) / comm_size))
      // a simpler way to look at bucket size is to do min(max_bucket/2, min_bucket), and the buffer size is max(1/2 input_size, y)
       *
       * the buffer is set to largest of first part, local second part, remote second part.
       *
       * break up the input into smaller chunks imply iterative process.  each iteration requires full input scan.  if we do logarithmic number of iterations, we can't scale very large.
       * instead we break up into 2 pieces only, (or maybe break it up to 3 or 4), a low number, and accept the overhead incurred.
   */
  template <typename V, typename ToRank, typename Operation, typename SIZE = size_t,
      typename T = typename bliss::functional::function_traits<Operation, V>::return_type>
  void scatter_compute_gather_lowmem(::std::vector<V>& input, ToRank const & to_rank,
                              Operation const & op,
                              ::std::vector<SIZE> & i2o,
                              ::std::vector<T>& output,
                              ::std::vector<V>& in_buffer, std::vector<T>& out_buffer,
                              ::mxx::comm const &_comm,
                              bool const & preserve_input = false) {
      BL_BENCH_INIT(scat_comp_gath_lm);

      BL_BENCH_COLLECTIVE_START(scat_comp_gath_lm, "empty", _comm);
      bool empty = input.size() == 0;
      empty = mxx::all_of(empty);
      BL_BENCH_END(scat_comp_gath_lm, "empty", input.size());

      if (empty) {
        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_lm, "khmxx:scat_comp_gath_lm", _comm);
        return;
      }



      // do assignment.
      BL_BENCH_START(scat_comp_gath_lm);
      std::vector<SIZE> send_counts(_comm.size(), 0);
      std::vector<SIZE> recv_counts(_comm.size(), 0);
      i2o.resize(input.size());
      BL_BENCH_END(scat_comp_gath_lm, "alloc_map", input.size());

      // first bucketing
      BL_BENCH_START(scat_comp_gath_lm);
      khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
      BL_BENCH_END(scat_comp_gath_lm, "bucket", input.size());

      // then compute minimum block size.
      BL_BENCH_START(scat_comp_gath_lm);
      SIZE min_bucket_size = *(::std::min_element(send_counts.begin(), send_counts.end()));
      min_bucket_size = ::mxx::allreduce(min_bucket_size, mxx::min<SIZE>(), _comm);
      SIZE block_bucket_size = min_bucket_size / 2;  // block_bucket_size is at least 1/2 as large as the largest bucket.
      SIZE block_size = _comm.size() * block_bucket_size;
      SIZE first_part = 2 * block_size;   // this is at least 1/2 of the largest input.
      SIZE second_part_local = input.size() - first_part;
      recv_counts.resize(_comm.size());
      ::mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
      SIZE second_part_remote = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0)) - first_part;   // second part size.

      SIZE second = second_part_local + second_part_remote;
      SIZE input_plus = input.size() + second_part_remote;
      bool traditional = (second > (input_plus / 2));
      traditional = mxx::any_of(traditional, _comm);
      BL_BENCH_END(scat_comp_gath_lm, "a2av_count", first_part);

      if (traditional) {
    	  BL_BENCH_START(scat_comp_gath_lm);

    	  scatter_compute_gather_2part(input, to_rank, op, i2o, output, in_buffer, out_buffer, _comm, preserve_input);
          BL_BENCH_END(scat_comp_gath_lm, "switch_to_trad", output.size());


        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_lm, "khmxx:scat_comp_gath_lm", _comm);
        return;
      }

      // compute the permutations from block size and processor mapping.  send_counts modified to the remainders.
      BL_BENCH_START(scat_comp_gath_lm);
      ::khmxx::local::blocked_bucketId_to_pos(block_bucket_size, 2UL, send_counts, i2o, 0, input.size());
      BL_BENCH_END(scat_comp_gath_lm, "to_pos", input.size());

      // allocate input buffer (for permuted data.)
      BL_BENCH_START(scat_comp_gath_lm);
      SIZE buffer_size = std::max(block_size, second_part_local + second_part_remote);
      if (in_buffer.capacity() < buffer_size) in_buffer.clear();
      in_buffer.resize(buffer_size);
      BL_BENCH_END(scat_comp_gath_lm, "alloc_inbuf", in_buffer.size());

      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath_lm);
      if (output.capacity() < (input.size())) output.clear();
      output.resize(input.size());
      BL_BENCH_END(scat_comp_gath_lm, "alloc_out", output.size());

      //== process first part - 2 iterations..  communicate in place
      for (size_t i = 0; i < 2; ++i) {
		  // permute
		  BL_BENCH_START(scat_comp_gath_lm);
		  ::khmxx::local::permute_for_output_range(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), in_buffer.begin() + block_size, i * block_size);
		  BL_BENCH_END(scat_comp_gath_lm, "permute_block", block_size);

		  BL_BENCH_START(scat_comp_gath_lm);
		  block_all2all_inplace(in_buffer, block_bucket_size, 0, _comm);
		  BL_BENCH_END(scat_comp_gath_lm, "a2a_inplace", block_size);

		  // process
		  BL_BENCH_START(scat_comp_gath_lm);
		  op(in_buffer.begin(), in_buffer.begin() + block_size, output.begin() + i * block_size);
		  BL_BENCH_END(scat_comp_gath_lm, "compute1", block_size);

		  // send the results back.  and reverse the input
		  // undo a2a, so that result data matches.
		  BL_BENCH_START(scat_comp_gath_lm);
		  block_all2all_inplace(output, block_bucket_size, i * block_size, _comm);
		  BL_BENCH_END(scat_comp_gath_lm, "inverse_a2a_inplace", block_size);

      }

      //======= process the second part
      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath_lm);
      if (out_buffer.capacity() < second_part_remote) out_buffer.clear();
      out_buffer.resize(second_part_remote);
      BL_BENCH_END(scat_comp_gath_lm, "alloc_outbuf", out_buffer.size());

	  // permute
	  BL_BENCH_START(scat_comp_gath_lm);
	  ::khmxx::local::permute_for_output_range(input.begin(), input.end(), i2o.begin(),
			  in_buffer.begin(), in_buffer.begin() + second_part_local, first_part);
	  BL_BENCH_END(scat_comp_gath_lm, "permute_block", second_part_local);

      // send second part.  reuse entire in_buffer
      BL_BENCH_START(scat_comp_gath_lm);
      ::mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
	  mxx::all2allv(in_buffer.data(), send_counts,
                    in_buffer.data() + second_part_local, recv_counts, _comm);
      BL_BENCH_END(scat_comp_gath_lm, "a2av", second_part_local);

      // process the second part.
      BL_BENCH_START(scat_comp_gath_lm);
      op(in_buffer.begin() + second_part_local, in_buffer.begin() + second_part_local + second_part_remote, out_buffer.begin());
      BL_BENCH_END(scat_comp_gath_lm, "compute2", second_part_remote);

      // send the results back
      BL_BENCH_START(scat_comp_gath_lm);
	  mxx::all2allv(out_buffer.data(), recv_counts,
                    output.data() + first_part, send_counts, _comm);
      BL_BENCH_END(scat_comp_gath_lm, "inverse_a2av", second_part_remote);

      // permute
      if (preserve_input) {
        // in_buffer was already allocated to be same size as input.
        ::khmxx::local::unpermute_inplace(input, i2o, 0, input.size());
        // out_buffer is small, so should do this inplace.
        ::khmxx::local::unpermute_inplace(output, i2o, 0, output.size());
        BL_BENCH_END(scat_comp_gath_lm, "unpermute_inplace", output.size());
      }

      BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_lm, "khmxx:scat_comp_gath_lm", _comm);
  }

  // TODO: non-one-to-one version.

  /**
   * @brief distribute, compute, send back.  one to one.  result matching input in order at then end.
   * @detail   this is the memory inefficient version
   *
   *
   */
  template <typename V, typename ToRank, typename Operation, typename SIZE = size_t,
      typename T = typename bliss::functional::function_traits<Operation, V>::return_type>
  void scatter_compute_gather_v(::std::vector<V>& input, ToRank const & to_rank,
                              Operation const & op,
                              ::std::vector<SIZE> & i2o,
                              ::std::vector<T>& output,
                              ::std::vector<V>& in_buffer, std::vector<T>& out_buffer,
                              ::mxx::comm const &_comm,
                              bool const & preserve_input = false) {
      BL_BENCH_INIT(scat_comp_gath_v);

      // speed over mem use.  mxx all2allv already has to double memory usage. same as stable distribute.
      BL_BENCH_COLLECTIVE_START(scat_comp_gath_v, "empty", _comm);
      bool empty = input.size() == 0;
      empty = mxx::all_of(empty);
      BL_BENCH_END(scat_comp_gath_v, "empty", input.size());

      if (empty) {
        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_v, "khmxx:scat_comp_gath_v", _comm);
        return;
      }

      // do assignment.
      BL_BENCH_START(scat_comp_gath_v);
      std::vector<SIZE> recv_counts(_comm.size(), 0);
      i2o.resize(input.size());
      BL_BENCH_END(scat_comp_gath_v, "alloc_map", input.size());

      // distribute
      BL_BENCH_START(scat_comp_gath_v);
      distribute(input, to_rank, recv_counts, i2o, in_buffer, _comm, false);
      BL_BENCH_END(scat_comp_gath_v, "distribute", in_buffer.size());

      // allocate out_buffer - output is same size as input
      BL_BENCH_START(scat_comp_gath_v);
      if (out_buffer.capacity() < (in_buffer.size())) out_buffer.clear();
      out_buffer.reserve(in_buffer.size());
      ::fsc::back_emplace_iterator<std::vector<T> > emplacer(out_buffer);
      BL_BENCH_END(scat_comp_gath_v, "alloc_outbuf", out_buffer.size());

      // process
      BL_BENCH_START(scat_comp_gath_v);
      size_t s;
      auto it = in_buffer.begin();
      for (size_t i = 0; i < static_cast<size_t>(_comm.size()); ++i) {
    	  s = recv_counts[i];

          recv_counts[i] = op(it, it + s, emplacer);
          std::advance(it, s);
      }
      BL_BENCH_END(scat_comp_gath_v, "compute", out_buffer.size());

      // allocate output - output is same size as input
      BL_BENCH_START(scat_comp_gath_v);
      std::vector<SIZE> send_counts(_comm.size(), 0);
      ::mxx::all2all(recv_counts.data(), 1, send_counts.data(), _comm);
      size_t total = ::std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));
      if (output.capacity() < total) output.clear();
      output.resize(total);
      BL_BENCH_END(scat_comp_gath_v, "alloc_out", output.size());

      // distribute data back to source
      BL_BENCH_START(scat_comp_gath_v);
      undistribute(out_buffer, recv_counts, i2o, output, _comm, false);
      BL_BENCH_END(scat_comp_gath_v, "undistribute", output.size());

      // permute
      if (preserve_input) {
        BL_BENCH_START(scat_comp_gath_v);
        ::khmxx::local::unpermute(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), 0);
        in_buffer.swap(input);
        ::khmxx::local::unpermute(output.begin(), output.end(), i2o.begin(), out_buffer.begin(), 0);
        out_buffer.swap(output);
        BL_BENCH_END(scat_comp_gath_v, "unpermute_inplace", output.size());
      }

      BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_v, "khmxx:scat_comp_gath_v", _comm);
  }

  //TODO:
//
//  /**
//   * @brief distribute, compute, send back.  one to one.  result matching input in order at then end.
//   * @details  this is the memory efficient version.  this has to be incremental.
//   *            this version uses a permute buffer. (in_buffer)
//   *
//   *            low mem version.
//   *
//   *            the second half (a2av) require the space it requires.  for in_buffer and out_buffer.
//   *            the first half can be reduced in space usage by incrementally populate the in_buffer,
//   *            	compute the output, then populate the output.  then 1 all2all inplace for the output
//   *            	at the requester, then unpermute somehow.
//   *
//      // compute the size of the buffers to use.  set a maximum limit.
//       * let the final block_bucket_size be bbs. then the first_part x = bbs * blocks * p., p = comm_size.
//       * let the sum of UNEVEN parts in orig input be y1, and after a2av be y2.  the minimum space needed here is y1+y2 for the a2av part.
//       *   the uneven part is defined by the difference between a bucket and the global min bucket.
//       *
//       * APPROACH:  use either y1+y2 or 1/2 min_bucket, whichever is larger, as buffer
//       * 	if y1+y2 > 1/2 (input + y2), use traditional
//       * 	else
//       * 		use larger of y1+y2, and 1/2 min bucket.
//       *
//       *
//       * however, bbs may not perfectly divide the min_bucket..  let the remainder be r.  0 <= r < bbs
//       * the UNEVEN part then needs to be appended with r in each bucket.  Y1 = y1 + r * p, and Y2 = y2 + r * p.  post communication, the remainder size remains same.
//       * minimum space needed is revised to Y1 + Y2 = y1 + y2 + 2 rp.
//       *
//       * if y1 + y2 > input_size, then we should just the normal scatter_compute_gather, or the 2 part version, since buffer requirement is high.
//       *
//       * case 1
//       * bbs * p > Y1 + Y2.  each block still needs to got through a2a, so makes no sense to have more than 1 blocks in the buffer and we should reduce the number of iterations
//       * to have some form of savings, we should use bbs * p < input.  then input > Y2 + Y2.  if this does not hold, then use full memory version.
//       *
//       * for all processors, the following must hold to have buffer savings.
//       * y1 + rp + bbs * blocks * p > bbs * p > y1 + y2 + 2rp.   r + bbs * blocks = min_bucket_size.
//       * 	p * min_bucket_size > y2 + 2rp,  so r < min_bucket_size - y2 / 2p,
//       * 	mod(min_bucket_size, bbs) = r < bbs.
//       * so let bbs = min_bucket_size - y2 / 2p.  (bbs can be at most this quantity, else r may exceed bound)
//       * 	now this can be pretty big, or really small.
//       *
//       * what is the lower bound of bbs?
//       *
//       *
//       *
//       *
//       *
//       * for there to be space savings, y1+y2 is the minimum buffer size and the maximum buffer size should not exceed 1/2 of input
//       * buffer size should be std::max(y1+y2, comm_size * block_bucket_size)   1 block...
//       * where block_bucket_size = std::min(min_bucket_size, 1/2 max_bucket_size)    // at most min_bucket_size, at least 1/2 max_bucket_size
//       * this will affect y1 and y2, of course.
//       * 	y1 has size input % (comm_size * block_bucket_size), with max value of (comm_size * block_bucket_size - 1)
//       * 	y2 has max comm_size * (comm_size  * block_bucket_size - 1 ) - 1
//       *
//       * note at the end, we'd still need to permute either the input or the output to make them consistent with each other.
//      // max of "second_part" is the minimum required space for the a2av portion.  let this be y.
//      //    reducing this requires O(p) type communication rather than O(log(p))
//      // to effect some space savings, we should use max(1/2 input_size, y).
//      //	we can reduce the 1/2 input size, at the expense of more iterations, each requiring a complete scan of input during permuting.
//      //	if y > 1/2 input, then on one processor the min bucket is less than 1/2 input,  we'd be using this min bucket anyways.
//      //	if y < 1/2 input, then we can potentially do more in a2a phase, but that would use more mem.
//      // so configure the buffer size to be max(1/2 input_size, y) = Y.
//      //    for bucket size, this means min(min_bucket, (Y + comm_size - 1) / comm_size))
//      // a simpler way to look at bucket size is to do min(max_bucket/2, min_bucket), and the buffer size is max(1/2 input_size, y)
//       *
//       * the buffer is set to largest of first part, local second part, remote second part.
//       *
//       * break up the input into smaller chunks imply iterative process.  each iteration requires full input scan.  if we do logarithmic number of iterations, we can't scale very large.
//       * instead we break up into 2 pieces only, (or maybe break it up to 3 or 4), a low number, and accept the overhead incurred.
//   */
//  template <typename V, typename ToRank, typename Operation, typename SIZE = size_t,
//      typename T = typename bliss::functional::function_traits<Operation, V>::return_type>
//  void scatter_compute_gather_v_lowmem(::std::vector<V>& input, ToRank const & to_rank,
//                              Operation const & op,
//                              ::std::vector<SIZE> & i2o,
//                              ::std::vector<T>& output,
//                              ::std::vector<V>& in_buffer, std::vector<T>& out_buffer,
//                              ::mxx::comm const &_comm,
//                              bool const & preserve_input = false) {
//      BL_BENCH_INIT(scat_comp_gath_v_lm);
//
//      bool empty = input.size() == 0;
//      empty = mxx::all_of(empty);
//      if (empty) {
//        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_v_lm, "map_base:scat_comp_gath_v_lm", _comm);
//        return;
//      }
//
//
//
//      // do assignment.
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      std::vector<SIZE> send_counts(_comm.size(), 0);
//      std::vector<SIZE> recv_counts(_comm.size(), 0);
//      i2o.resize(input.size());
//      BL_BENCH_END(scat_comp_gath_v_lm, "alloc_map", input.size());
//
//      // first bucketing
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      khmxx::local::assign_to_buckets(input, to_rank, _comm.size(), send_counts, i2o, 0, input.size());
//      BL_BENCH_END(scat_comp_gath_v_lm, "bucket", input.size());
//
//      // then compute minimum block size.
//      BL_BENCH_COLLECTIVE_START(scat_comp_gath_v_lm, "a2av_count", _comm);
//      SIZE min_bucket_size = *(::std::min_element(send_counts.begin(), send_counts.end()));
//      min_bucket_size = ::mxx::allreduce(min_bucket_size, mxx::min<SIZE>(), _comm);
//      SIZE block_bucket_size = min_bucket_size / 2;  // block_bucket_size is at least 1/2 as large as the largest bucket.
//      SIZE block_size = _comm.size() * block_bucket_size;
//      SIZE first_part = 2 * block_size;   // this is at least 1/2 of the largest input.
//      SIZE second_part_local = input.size() - first_part;
//      recv_counts.resize(_comm.size());
//      ::mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
//      SIZE second_part_remote = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0)) - first_part;   // second part size.
//
//      SIZE second = second_part_local + second_part_remote;
//      SIZE input_plus = input.size() + second_part_remote;
//      bool traditional = (second > (input_plus / 2));
//      traditional = mxx::any_of(traditional, _comm);
//      BL_BENCH_END(scat_comp_gath_v_lm, "a2av_count", first_part);
//
//      if (traditional) {
//    	  BL_BENCH_START(scat_comp_gath_v_lm);
//
//    	  scatter_compute_gather_2part(input, to_rank, op, i2o, output, in_buffer, out_buffer, _comm, preserve_input);
//          BL_BENCH_END(scat_comp_gath_v_lm, "switch_to_trad", output.size());
//
//
//        BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_v_lm, "map_base:scat_comp_gath_v_lm", _comm);
//        return;
//      }
//
//      // compute the permutations from block size and processor mapping.  send_counts modified to the remainders.
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      ::khmxx::local::blocked_bucketId_to_pos(block_bucket_size, 2UL, send_counts, i2o, 0, input.size());
//      BL_BENCH_END(scat_comp_gath_v_lm, "to_pos", input.size());
//
//      // allocate input buffer (for permuted data.)
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      SIZE buffer_size = std::max(block_size, second_part_local + second_part_remote);
//      if (in_buffer.capacity() < buffer_size) in_buffer.clear();
//      in_buffer.resize(buffer_size);
//      BL_BENCH_END(scat_comp_gath_v_lm, "alloc_inbuf", in_buffer.size());
//
//      // allocate output - output is same size as input
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      if (output.capacity() < (input.size())) output.clear();
//      output.resize(input.size());
//      BL_BENCH_END(scat_comp_gath_v_lm, "alloc_out", output.size());
//
//      //== process first part - 2 iterations..  communicate in place
//      for (size_t i = 0; i < 2; ++i) {
//		  // permute
//		  BL_BENCH_START(scat_comp_gath_v_lm);
//		  ::khmxx::local::permute_for_output_range(input.begin(), input.end(), i2o.begin(), in_buffer.begin(), in_buffer.begin() + block_size, i * block_size);
//		  BL_BENCH_END(scat_comp_gath_v_lm, "permute_block", block_size);
//
//		  BL_BENCH_COLLECTIVE_START(scat_comp_gath_v_lm, "a2a_inplace", _comm);
//		  block_all2all_inplace(in_buffer, block_bucket_size, 0, _comm);
//		  BL_BENCH_END(scat_comp_gath_v_lm, "a2a_inplace", block_size);
//
//		  // process
//		  BL_BENCH_START(scat_comp_gath_v_lm);
//		  op(in_buffer.begin(), in_buffer.begin() + block_size, output.begin() + i * block_size);
//		  BL_BENCH_END(scat_comp_gath_v_lm, "compute1", block_size);
//
//		  // send the results back.  and reverse the input
//		  // undo a2a, so that result data matches.
//		  BL_BENCH_COLLECTIVE_START(scat_comp_gath_v_lm, "inverse_a2a_inplace", _comm);
//		  block_all2all_inplace(output, block_bucket_size, i * block_size, _comm);
//		  BL_BENCH_END(scat_comp_gath_v_lm, "inverse_a2a_inplace", block_size);
//
//      }
//
//
//      //======= process the second part
//      // allocate output - output is same size as input
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      if (out_buffer.capacity() < second_part_remote) out_buffer.clear();
//      out_buffer.resize(second_part_remote);
//      BL_BENCH_END(scat_comp_gath_v_lm, "alloc_outbuf", out_buffer.size());
//
//	  // permute
//	  BL_BENCH_START(scat_comp_gath_v_lm);
//	  ::khmxx::local::permute_for_output_range(input.begin(), input.end(), i2o.begin(),
//			  in_buffer.begin(), in_buffer.begin() + second_part_local, first_part);
//	  BL_BENCH_END(scat_comp_gath_v_lm, "permute_block", second_part_local);
//
//      // send second part.  reuse entire in_buffer
//      BL_BENCH_COLLECTIVE_START(scat_comp_gath_v_lm, "a2av", _comm);
//      ::mxx::all2all(send_counts.data(), 1, recv_counts.data(), _comm);
//	  mxx::all2allv(in_buffer.data(), send_counts,
//                    in_buffer.data() + second_part_local, recv_counts, _comm);
//      BL_BENCH_END(scat_comp_gath_v_lm, "a2av", second_part_local);
//
//      // process the second part.
//      BL_BENCH_START(scat_comp_gath_v_lm);
//      op(in_buffer.begin() + second_part_local, in_buffer.begin() + second_part_local + second_part_remote, out_buffer.begin());
//      BL_BENCH_END(scat_comp_gath_v_lm, "compute2", second_part_remote);
//
//      // send the results back
//      BL_BENCH_COLLECTIVE_START(scat_comp_gath_v_lm, "inverse_a2av", _comm);
//	  mxx::all2allv(out_buffer.data(), recv_counts,
//                    output.data() + first_part, send_counts, _comm);
//      BL_BENCH_END(scat_comp_gath_v_lm, "inverse_a2av", second_part_remote);
//
//      // permute
//      if (preserve_input) {
//        // in_buffer was already allocated to be same size as input.
//        ::khmxx::local::unpermute_inplace(input, i2o, 0, input.size());
//        // out_buffer is small, so should do this inplace.
//        ::khmxx::local::unpermute_inplace(output, i2o, 0, output.size());
//        BL_BENCH_END(scat_comp_gath_v_lm, "unpermute_inplace", output.size());
//      }
//
//      BL_BENCH_REPORT_MPI_NAMED(scat_comp_gath_v_lm, "map_base:scat_comp_gath_v_lm", _comm);
//  }
//

//== TODO: transform before or after communication


//== communicate, process (not one to one), return communication  - order does not matter


// TODO mpi3 versions?


// ============== specialized parallel sample sorting for indexing. adapted from mxx =============


  template <typename V, typename _Compare>
  std::vector<size_t> stable_split(::std::vector<V>& input, _Compare comp,
                                   const std::vector<V>& splitters,
                                   const mxx::comm& comm) {
      // 5. locally find splitter positions in data
      //    (if an identical splitter appears at least three times (or more),
      //    then split the intermediary buckets evenly) => send_counts
      MXX_ASSERT(splitters.size() == (size_t) comm.size() - 1);

      // check if there are repeated entries in local_splitters.
      if (splitters.size() > 0) {
        auto it = splitters.cbegin();
        auto next = it;  ++next;
        while (next != splitters.cend()) {
          if (!comp(*it, *next)) {  // equal. since local_splitters are sorted, don't need to check for next < it
            throw std::invalid_argument("ERROR: kmers for indexing should be somewhat evenly distributed so that there is no range spanning an entire processor.");
          }
          it = next;
          ++next;
        }
      }


      std::vector<size_t> send_counts(comm.size(), 0);
      auto pos = input.cbegin();
      auto splitter_end = pos;
      for (std::size_t i = 0; i < splitters.size(); ++i) {
          // get the range of equal elements
          splitter_end = std::upper_bound(pos, input.cend(), splitters[i], comp);

          // assign smaller elements to processor left of splitter (= `i`)
          send_counts[i] = std::distance(pos, splitter_end);
          pos = splitter_end;
      }

      // send last elements to last processor
      send_counts[comm.size() - 1] = std::distance(pos, input.cend());
      MXX_ASSERT(std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0)) == input.size());
      return send_counts;
  }


  /**
   * @brief modified version of mxx::samplesort with some optimizations
   * @details  This version does not attempt to rebalance after parallel sort.
   *    NOTE: if we want load balance, balance BEFORE calling this routine.
   *    ASSUMPTION: data is close to uniformly distributed
   *    ASSUMPTION: where data is not uniformly distributed, sampling results in good enough splits
   *
   *    return local splitters
   */
  template<bool _Stable = false, typename V, typename _Compare>
  std::vector<V> samplesort(::std::vector<V>& input, ::std::vector<V>& output, _Compare comp, const mxx::comm& comm) {
      // get value type of underlying data



    BL_BENCH_INIT(khmxx_samplesort);


    bool empty = ::mxx::all_of(input.size() == 0, comm);
    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "noop-samplesort", comm);

      return std::vector<V>();
    }


      int p = comm.size();



      BL_BENCH_COLLECTIVE_START(khmxx_samplesort, "init", comm);

      // perform local (stable) sorting
      if (_Stable)
          std::stable_sort(input.begin(), input.end(), comp);
      else
          std::sort(input.begin(), input.end(), comp);

      BL_BENCH_COLLECTIVE_END(khmxx_samplesort, "local sort", input.size(), comm);

      // sequential case: we're done
      if (p == 1) {
          output.resize(input.size());
          std::copy(input.begin(), input.end(), output.begin());
          BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "p1-samplesort", comm);
          return std::vector<V>();
      }



      BL_BENCH_START(khmxx_samplesort);

      // local size
      std::size_t local_size = input.size();

      // check if we have a perfect block decomposition
      std::size_t global_size = ::mxx::allreduce(local_size, comm);
      ::mxx::partition::block_decomposition<std::size_t> mypart(global_size, p, comm.rank());
      bool _AssumeBlockDecomp = ::mxx::all_of(local_size == mypart.local_size(), comm);

      // sample sort
      // 1. local sort
      // 2. pick `s` samples regularly spaced on each processor
      // 3. bitonic sort samples
      // 4. allgather the last sample of each process -> splitters
      // 5. locally find splitter positions in data
      //    //NO (if an identical splitter appears twice, then split evenly)
      //    => send_counts
      // 6. distribute send_counts with all2all to get recv_counts
      // 7. allocate enough space (may be more than previously allocated) for receiving
      // 8. all2allv
      // 9. local reordering (multiway-merge or again std::sort)
      // A. equalizing distribution into original size (e.g.,block decomposition)
      //    by sending elements to neighbors

      // get splitters, using the method depending on whether the input consists
      // of arbitrary decompositions or not
      std::vector<V> local_splitters;
      // number of samples
      size_t s = p-1;
      if(_AssumeBlockDecomp)
          local_splitters = ::mxx::impl::sample_block_decomp(input.cbegin(), input.cend(), comp, s, comm);
      else
          local_splitters = ::mxx::impl::sample_arbit_decomp(input.cbegin(), input.cend(), comp, s, comm);

      BL_BENCH_END(khmxx_samplesort, "get_splitters", s);

      BL_BENCH_START(khmxx_samplesort);


      // 5. locally find splitter positions in data
      //    (if an identical splitter appears at least three times (or more),
      //    then split the intermediary buckets evenly) => send_counts
      std::vector<size_t> send_counts = khmxx::stable_split(input, comp, local_splitters, comm);  // always do stable split
      BL_BENCH_END(khmxx_samplesort, "send_counts", send_counts.size());

      BL_BENCH_START(khmxx_samplesort);


      std::vector<size_t> recv_counts = ::mxx::all2all(send_counts, comm);
      std::size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
      MXX_ASSERT(!_AssumeBlockDecomp || (local_size <= (size_t)p || recv_n <= 2* local_size));

      // reserve
      if (output.capacity() < recv_n) output.clear();
      output.resize(recv_n);
      BL_BENCH_END(khmxx_samplesort, "reserve", recv_n);

      BL_BENCH_START(khmxx_samplesort);


      // TODO: use collective with iterators [begin,end) instead of pointers!
      ::mxx::all2allv(input.data(), send_counts, output.data(), recv_counts, comm);
      BL_BENCH_END(khmxx_samplesort, "all2all", local_size);

      BL_BENCH_START(khmxx_samplesort);

      // 9. local reordering
      /* multiway-merge (using the implementation in __gnu_parallel) */
      // if p = 2, then merge
      if (p == 2) {
        // always stable.
        std::inplace_merge(output.begin(), output.begin() + recv_counts[0], output.end(), comp);

      } else {

        // allocate half of the total space.  merge for first half, compact, copy, run the second half, copy.

  #ifdef MXX_USE_GCC_MULTIWAY_MERGE
        // NOTE: uses recv_n / 2 temp storage.

      if (recv_n > (size_t)p*p) {  // should not be local_size.

        BL_BENCH_INIT(khmxx_samplesort_merge);


        BL_BENCH_START(khmxx_samplesort_merge);


        // prepare the sequence offsets
          typedef typename std::vector<V>::iterator val_it;

          std::vector<std::pair<val_it, val_it> > seqs(p);
          std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
          for (int i = 0; i < p; ++i) {
              seqs[i].first = output.begin() + recv_displs[i];
              seqs[i].second = seqs[i].first + recv_counts[i];
          }

          BL_BENCH_END(khmxx_samplesort_merge, "merge_ranges", seqs.size());

          BL_BENCH_START(khmxx_samplesort_merge);

          // allocate the size.
          // TODO: reasonable values for the buffer?
          std::size_t merge_n = (recv_n + 1) / 2;
          std::vector<V> merge_buf(merge_n);
          // auto tmp = std::get_temporary_buffer<V>(merge_n);

          BL_BENCH_END(khmxx_samplesort_merge, "merge_buf_alloc", merge_n);

          BL_BENCH_START(khmxx_samplesort_merge);


          val_it start_merge_it = output.begin();

          V* merge_buf_begin = merge_buf.data();

          __gnu_parallel::sequential_tag seq_tag;

          // 2 iterations.  unroll loop.
          // first half.
          // i)   merge at most `merge_n` many elements sequentially
          if (_Stable)
            __gnu_parallel::stable_multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, merge_n, comp, seq_tag);
          else
            __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, merge_n, comp, seq_tag);

          BL_BENCH_END(khmxx_samplesort_merge, "merge1", merge_n);

          BL_BENCH_START(khmxx_samplesort_merge);


          // ii)  compact the remaining elements in `output`
          // TCP:  doing this a fixed number of times makes it linear in data size.  if the number of iterations is dependent on data size, then it becomes quadratic.
          for (int i = p-1; i > 0; --i)
          {
              seqs[i-1].first = std::move_backward(seqs[i-1].first, seqs[i-1].second, seqs[i].first);
              seqs[i-1].second = seqs[i].first;
          }

          BL_BENCH_END(khmxx_samplesort_merge, "compact", std::distance(seqs[0].first, seqs[p-1].second));

          BL_BENCH_START(khmxx_samplesort_merge);


          // iii) copy the output buffer `merge_n` elements back into `output`
          //      `recv_elements`.
          start_merge_it = std::copy(merge_buf.begin(), merge_buf.begin() + merge_n, start_merge_it);
          assert(start_merge_it == seqs[0].first);
          BL_BENCH_END(khmxx_samplesort_merge, "copy1", merge_n);

          BL_BENCH_START(khmxx_samplesort_merge);


          // now the second half.
          merge_n = recv_n  - merge_n;

          // i)   merge at most `local_size` many elements sequentially
          if (_Stable)
            __gnu_parallel::stable_multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, merge_n, comp, seq_tag);
          else
            __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, merge_n, comp, seq_tag);

          BL_BENCH_END(khmxx_samplesort_merge, "merge2", merge_n);

          BL_BENCH_START(khmxx_samplesort_merge);

          // iii) copy the output buffer `merge_n` elements back into `output`
          //      `recv_elements`.
          start_merge_it = std::move(merge_buf.begin(), merge_buf.begin() + merge_n, start_merge_it);
          assert(start_merge_it == output.end());

          BL_BENCH_END(khmxx_samplesort_merge, "copy2", merge_n);

          BL_BENCH_START(khmxx_samplesort_merge);


          // clean up
          merge_buf.clear(); std::vector<V>().swap(merge_buf);

          BL_BENCH_END(khmxx_samplesort_merge, "merge_buf_dealloc", merge_n);

          BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort_merge, "khmxx_samplesort_merge", comm);

      } else
  #endif
      { // sorting based.
          if (_Stable)
              std::stable_sort(output.begin(), output.end(), comp);
          else
              std::sort(output.begin(), output.end(), comp);
      }
      } // p > 2

      BL_BENCH_COLLECTIVE_END(khmxx_samplesort, "local_merge", recv_n, comm);


//      // A. equalizing distribution into original size (e.g.,block decomposition)
//      //    by elements to neighbors
//      //    and save elements into the original iterator positions
//      if (_AssumeBlockDecomp)
//          ::mxx::stable_distribute(recv_elements.begin(), recv_elements.end(), begin, comm);
//      else
//          ::mxx::redo_arbit_decomposition(recv_elements.begin(), recv_elements.end(), begin, local_size, comm);
//
//      SS_TIMER_END_SECTION("fix_partition");

      BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "khmxx_samplesort", comm);

      return local_splitters;
  }



  /**
   * @brief modified version of mxx::samplesort with some optimizations
   * @details  This version does not attempt to rebalance after parallel sort.
   *    NOTE: if we want load balance, balance BEFORE calling this routine.
   *    ASSUMPTION: data is close to uniformly distributed
   *    ASSUMPTION: where data is not uniformly distributed, sampling results in good enough splits
   *
   *    return local splitters
   */
  template<bool _Stable = false, typename V, typename _Compare>
  std::vector<V> samplesort_buf(::std::vector<V>& input, ::std::vector<V>& output, _Compare comp, const mxx::comm& comm,
		  char use_sort_override = 0, char full_buffer_override = 0) {
      // get value type of underlying data

    BL_BENCH_INIT(khmxx_samplesort);


    bool empty = ::mxx::all_of(input.size() == 0, comm);
    if (empty) {
      BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "noop-samplesort", comm);

      return std::vector<V>();
    }


      int p = comm.size();


      BL_BENCH_COLLECTIVE_START(khmxx_samplesort, "init", comm);

      // perform local (stable) sorting
      if (_Stable)
          std::stable_sort(input.begin(), input.end(), comp);
      else
          std::sort(input.begin(), input.end(), comp);

      BL_BENCH_COLLECTIVE_END(khmxx_samplesort, "local sort", input.size(), comm);

      // sequential case: we're done
      if (p == 1) {
          output.resize(input.size());
          std::copy(input.begin(), input.end(), output.begin());
          BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "p1-samplesort", comm);
          return std::vector<V>();
      }


      BL_BENCH_START(khmxx_samplesort);

      // local size
      std::size_t local_size = input.size();

      // check if we have a perfect block decomposition
      std::size_t global_size = ::mxx::allreduce(local_size, comm);
      ::mxx::partition::block_decomposition<std::size_t> mypart(global_size, p, comm.rank());
      bool _AssumeBlockDecomp = ::mxx::all_of(local_size == mypart.local_size(), comm);

      // sample sort
      // 1. local sort
      // 2. pick `s` samples regularly spaced on each processor
      // 3. bitonic sort samples
      // 4. allgather the last sample of each process -> splitters
      // 5. locally find splitter positions in data
      //    //NO (if an identical splitter appears twice, then split evenly)
      //    => send_counts
      // 6. distribute send_counts with all2all to get recv_counts
      // 7. allocate enough space (may be more than previously allocated) for receiving
      // 8. all2allv
      // 9. local reordering (multiway-merge or again std::sort)
      // A. equalizing distribution into original size (e.g.,block decomposition)
      //    by sending elements to neighbors

      // get splitters, using the method depending on whether the input consists
      // of arbitrary decompositions or not
      std::vector<V> local_splitters;
      // number of samples
      size_t s = p-1;
      if(_AssumeBlockDecomp)
          local_splitters = ::mxx::impl::sample_block_decomp(input.cbegin(), input.cend(), comp, s, comm);
      else
          local_splitters = ::mxx::impl::sample_arbit_decomp(input.cbegin(), input.cend(), comp, s, comm);

      BL_BENCH_END(khmxx_samplesort, "get_splitters", s);

      BL_BENCH_START(khmxx_samplesort);


      // 5. locally find splitter positions in data
      //    (if an identical splitter appears at least three times (or more),
      //    then split the intermediary buckets evenly) => send_counts
      std::vector<size_t> send_counts = khmxx::stable_split(input, comp, local_splitters, comm);  // always do stable split
      BL_BENCH_END(khmxx_samplesort, "send_counts", send_counts.size());

      BL_BENCH_START(khmxx_samplesort);


      std::vector<size_t> recv_counts = ::mxx::all2all(send_counts, comm);
      std::size_t recv_n = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
      MXX_ASSERT(!_AssumeBlockDecomp || (local_size <= (size_t)p || recv_n <= 2* local_size));

      // reserve output directly.
      if (output.capacity() < recv_n) output.clear();
      output.resize(recv_n);

      BL_BENCH_END(khmxx_samplesort, "reserve", recv_n);

      BL_BENCH_START(khmxx_samplesort);

      bool full_buffer = true;  // is a full buffer availabel for use?
      bool use_sort = (p > 2);

#if defined(__GNUG__)
	// if not gnu, then leave use_sort as is, since we can't really use k-way merge.
#ifdef MXX_USE_GCC_MULTIWAY_MERGE
      use_sort &= (recv_n <= (size_t)p * (size_t)p);  // if merge is defined, then use merge if p = 2, or if have enough entries.
#endif
#endif

      V* buf = nullptr;
      std::ptrdiff_t buf_size = 0;
      if (use_sort) {
        full_buffer = false;
      } else {
        // not using sort, then let's check if we can alloc temp buffer
        std::tie(buf, buf_size) = std::get_temporary_buffer<V>(recv_n);
        full_buffer = (static_cast<size_t>(buf_size) >= recv_n);
 
	// get_temporary_buffer does not seem to work correctly with more than 2 processses under clang.

      std::cout << "rank " << comm.rank() << " SAMPLESORT allocated " << (full_buffer ? "full" : "partial") << " buffer. " << buf_size << "/" << 
		recv_n << " override " << (full_buffer_override ? "full" : "partial") << std::endl;

     }

// for testing only.
      if (use_sort_override != 0) {
    	  use_sort |= use_sort_override == 1;  // use merge only when allowed by compiler and data size and not overriden

    	  if (comm.rank() == 0) std::cout << "SAMPLESORT using " << (use_sort ? "sort" : "merge") << std::endl;
      }
      if (full_buffer_override != 0) {
    	  full_buffer &= full_buffer_override == 1;  // can only override if there is full buffer.

    	  if (comm.rank() == 0) std::cout << "SAMPLESORT using " << (full_buffer ? "full" : "partial") << " buffer" << std::endl;
      }



      // if no gcc multiway merge, then use 2-way merge for p=2, and sort otherwise.

      BL_BENCH_END(khmxx_samplesort, "res_buf", recv_n);

      BL_BENCH_START(khmxx_samplesort);

      // TODO: use collective with iterators [begin,end) instead of pointers!
      if (use_sort || !full_buffer)
        ::mxx::all2allv(input.data(), send_counts, output.data(), recv_counts, comm);
      else
        ::mxx::all2allv(input.data(), send_counts, buf, recv_counts, comm);

      BL_BENCH_END(khmxx_samplesort, "all2all", local_size);

      BL_BENCH_START(khmxx_samplesort);

      // 9. local reordering
      /* multiway-merge (using the implementation in __gnu_parallel) */
      // if p = 2, then merge
      if (use_sort) {
        if (_Stable)
            std::stable_sort(output.begin(), output.end(), comp);
        else
            std::sort(output.begin(), output.end(), comp);

      } else {
        if (p == 2) { // use 2-way merge.  always stable.
          if (full_buffer) {
            std::merge(buf, buf + recv_counts[0],
                       buf + recv_counts[0], buf + recv_n,
                       output.begin(), comp);

          } else {
            std::inplace_merge(output.begin(), output.begin() + recv_counts[0], output.end(), comp);
          }
#ifdef MXX_USE_GCC_MULTIWAY_MERGE

        } else { // using multiway merge, and p > 2.
          BL_BENCH_INIT(khmxx_samplesort_merge);
          __gnu_parallel::sequential_tag seq_tag;

          if (full_buffer) {  // if we have full buffer, then all2all result is in the buf.  merge into output.
            // buffer -> output

            BL_BENCH_START(khmxx_samplesort_merge);

            std::vector<std::pair<V*, V*> > seqs(p);

            std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
            for (int i = 0; i < p; ++i) {
              seqs[i].first = buf + recv_displs[i];   // point to buffer.
              seqs[i].second = seqs[i].first + recv_counts[i];
            }
            BL_BENCH_END(khmxx_samplesort_merge, "merge_ranges", seqs.size());

            BL_BENCH_START(khmxx_samplesort_merge);
            if (_Stable)
              __gnu_parallel::stable_multiway_merge(seqs.begin(), seqs.end(), output.data(), recv_n, comp, seq_tag);
            else
              __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), output.data(), recv_n, comp, seq_tag);

            BL_BENCH_END(khmxx_samplesort_merge, "merge", recv_n);

          } else {  // buffer was not big enough, so all2all went into output.  now need to use buffer to merge and then copy into output.

            // output -> temp buffer -> back to output, in iterations.
            BL_BENCH_START(khmxx_samplesort_merge);

            // prepare the sequence offsets
			typedef typename std::vector<V>::iterator val_it;

			std::vector<std::pair<val_it, val_it> > seqs(p);
			std::vector<size_t> recv_displs = mxx::impl::get_displacements(recv_counts);
			for (int i = 0; i < p; ++i) {
			  seqs[i].first = output.begin() + recv_displs[i];
			  seqs[i].second = seqs[i].first + recv_counts[i];
			}
			BL_BENCH_END(khmxx_samplesort_merge, "merge_ranges", seqs.size());


			BL_BENCH_START(khmxx_samplesort_merge);
			size_t remain_n = recv_n;
			size_t target_size = buf_size;

            val_it start_merge_it = output.begin();
            V* merge_buf_begin = buf;

            BL_BENCH_LOOP_START(khmxx_samplesort_merge, 0);
            BL_BENCH_LOOP_START(khmxx_samplesort_merge, 1);
            BL_BENCH_LOOP_START(khmxx_samplesort_merge, 2);
            size_t count0 = 0;
            size_t count1 = 0;
            size_t count2 = 0;


            while (remain_n > 0) {

                BL_BENCH_LOOP_RESUME(khmxx_samplesort_merge, 0);
            	if (remain_n < target_size) target_size = remain_n;

              // 2 iterations.  unroll loop.
              // first half.
              // i)   merge at most `merge_n` many elements sequentially
              if (_Stable)
                __gnu_parallel::stable_multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, target_size, comp, seq_tag);
              else
                __gnu_parallel::multiway_merge(seqs.begin(), seqs.end(), merge_buf_begin, target_size, comp, seq_tag);

              BL_BENCH_LOOP_PAUSE(khmxx_samplesort_merge, 0);
              count0 += target_size;

              BL_BENCH_LOOP_RESUME(khmxx_samplesort_merge, 1);

              // ii)  compact the remaining elements in `output`
              // TCP:  doing this a fixed number of times makes it linear in data size.  if the number of iterations is dependent on data size, then it becomes quadratic.
              for (int i = p-1; i > 0; --i)
              {
                  seqs[i-1].first = std::move_backward(seqs[i-1].first, seqs[i-1].second, seqs[i].first);
                  seqs[i-1].second = seqs[i].first;
              }
              BL_BENCH_LOOP_PAUSE(khmxx_samplesort_merge, 1);
              count1 += std::distance(seqs[0].first, seqs[p-1].second);

              BL_BENCH_LOOP_RESUME(khmxx_samplesort_merge, 2);

              // iii) copy the output buffer `merge_n` elements back into `output`
              //      `recv_elements`.
              start_merge_it = std::copy(merge_buf_begin, merge_buf_begin + target_size, start_merge_it);
              assert(start_merge_it == seqs[0].first);

              // now the second half.
              remain_n -= target_size;
              BL_BENCH_LOOP_PAUSE(khmxx_samplesort_merge, 2);
              count2 += target_size;
              }
            BL_BENCH_LOOP_END(khmxx_samplesort_merge, 0, "merge", count0);
            BL_BENCH_LOOP_END(khmxx_samplesort_merge, 1, "compact", count1);
            BL_BENCH_LOOP_END(khmxx_samplesort_merge, 2, "copy", count2);
          }



          BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort_merge, "khmxx_samplesort_merge", comm);

#else
	} else {
	  throw ::std::logic_error("ERROR: specified multiway merge but is not using gnu compiler");	
#endif

        }  // else p > 2, in the absense of MXX_USE_GCC_MULTIWAY_MERGE, would be using SORT and handled earlier.



      }

      BL_BENCH_COLLECTIVE_END(khmxx_samplesort, "local_merge", recv_n, comm);

      BL_BENCH_START(khmxx_samplesort);


      // CLEAR the temp buffer.
      if (!use_sort) std::return_temporary_buffer(buf);

      BL_BENCH_END(khmxx_samplesort, "rel_buf", buf_size);






//      // A. equalizing distribution into original size (e.g.,block decomposition)
//      //    by elements to neighbors
//      //    and save elements into the original iterator positions
//      if (_AssumeBlockDecomp)
//          ::mxx::stable_distribute(recv_elements.begin(), recv_elements.end(), begin, comm);
//      else
//          ::mxx::redo_arbit_decomposition(recv_elements.begin(), recv_elements.end(), begin, local_size, comm);
//
//      SS_TIMER_END_SECTION("fix_partition");

      BL_BENCH_REPORT_MPI_NAMED(khmxx_samplesort, "khmxx_samplesort", comm);

      return local_splitters;
  }



} // namespace khmxx


#endif // KHMXX_HPP
