/*
 * Copyright 2017 Georgia Institute of Technology
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

/*
 * hashtable_OA_RH_do_prefix.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS3_HPP_
#define KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS3_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"

//#define REPROBE_STAT

// should be easier for prefetching

namespace fsc {

//#define make_missing_bucket_id(POS) POS
//#define make_existing_bucket_id(POS) (POS | bid_pos_exists)


/**
 * @brief Open Addressing hashmap that uses Robin Hood hashing, with doubling for resizing, circular internal array.  modified from standard robin hood hashmap to use bucket offsets, in attempt to improve speed.
 * @details  at the moment, nothing special for k-mers yet.
 * 		This class has the following implementation characteristics
 * 			vector of structs
 * 			open addressing with robin hood hashing.
 * 			doubling for reallocation
 * 			circular internal array
 *
 *  like standard robin hood hashmap implementation in "hashmap_robinhood_doubling.hpp", we use an auxillary array.  however, the array is interpreted differently.
 *  whereas standard imple stores the distance of the current array position from the bucket that the key is "hashed to",
 *  the "offsets" implementation stores at each position the starting offset of entries for the current bucket, and the empty/occupied bit indicates
 *  whether the current bucket has content or not (to distinguish between an empty bucket from an occupied bucket with offset 0.).
 *
 *  this is possible as robin hood hashing places entries for a bucket consecutively
 *
 *  benefits:
 *  1. no need to scan in order to find start of a bucket.
 *  2. end of a bucket is determined by the next bucket's offset.
 *
 *  Auxillary array interpretation:
 *
 *  where as RH standard aux array element represents the distance from source for the corresponding element in container.
 *    idx   ----a---b------------
 *    aux   ===|=|=|=|=|4|3|=====   _4_ is recorded at position hash(X) + _4_ of aux array.
 *    data  -----------|X|Y|-----,  X is inserted into bucket hash(X) = a.  in container position hash(X) + _4_.
 *
 *    empty positions are set to 0x80, and same for all info entries.
 *
 *  this aux array instead has each element represent the offset for the first entry of this bucket.
 *    idx   ----a---b------------
 *    aux   ===|4|=|3|=|=|=|=====   _4_ is recorded at position hash(X) of aux array.
 *    data  -----------|X|Y|-----,  X is inserted into bucket hash(X), in container position hash(X) + _4_.
 *
 *    empty positions are set with high bit 1, but can has distance larger than 0.
 *
 *  in standard, we linear scan info_container from position hash(Y) and must go to hash(Y) + 4 linearly.
 *    each aux entry is essentailyl independent of others.
 *    have to check and compare each data entry.
 *
 *  in this class, we just need to look at position hash(Y) and hash(Y) + 1 to know where to start and end in the container
 *    for find/count/update, those 2 are the only ones we need.
 *    for insert, we need to find the end of the range to modify, even after the insertion point.
 *      first search for end of range from current bucket (linear scan)
 *      then move from insertion point to end of range to right by 1.
 *      then insert at insertion point
 *      finally update the aux array from hash(Y) +1, to end of bucket range (ends on empty or dist = 0), by adding 1...
 *    for deletion, we need to again find the end of the range to modify, from the point of the deletion
 *      search for end of range from current bucket
 *      then move from deletion point to end of range to left by 1
 *      finally update the aux array from hash(Y) +1 to end of bucket range(ends on dist == 0), by subtracting 1...
 *    for rehash, from the first non-empty, to the next empty
 *      call insert on the entire range.
 *
 *  need AT LEAST 1 extra element, since copy_downsize and copy_upsize rely on those.
 *
 *  TODO:
 *  [ ] batch mode operations to allow more opportunities for optimization including SIMD
 *  [ ] predicated version of operations
 *  [ ] macros for repeated code.
 *  [x] use bucket offsets instead of distance to bucket.
 *  [ ] faster insertion in batch
 *  [ ] faster find?
 *
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_robinhood_doubling_offsets {

public:

    using key_type              = Key;
    using mapped_type           = T;
    using value_type            = ::std::pair<Key, T>;
    using hasher                = Hash;
    using key_equal             = Equal;

protected:

    //=========  start INFO_TYPE definitions.
    // MSB is to indicate if current BUCKET is empty.  rest 7 bits indicate offset for the first BUCKET entry if not empty, or where it would have been.
	// i.e. if empty, flipping the bit should indicate offset position of where the bucket entry goes.
    // relationship to prev and next:
    //    empty, pos == 0: prev: 0 & (empty or full);    1 & empty.       next: 0 & (empty or full)
    //    empty, pos >  0: prev: ( 0<= p <= pos) & full; (pos+1) & empty  next: (pos-1) & (empty or full)
    //	  full,  pos == 0: prev: 0 & (empty or full);    1 & empty.       next: (p >= 0) & (empty or full)
    //	  full,  pos >  0: prev: ( 0<= p <= pos) & full; (pos+1) & empty  next: (p >= pos) & (empty or full)
    // container has a valid entry for each of last 3.
    using info_type = uint8_t;

	static constexpr info_type info_empty = 0x80;
	static constexpr info_type info_mask = 0x7F;
	static constexpr info_type info_normal = 0x00;   // this is used to initialize the reprobe distances.

	inline bool is_empty(info_type const & x) const {
		return x >= info_empty;  // empty 0x80
	}
	inline bool is_normal(info_type const & x) const {
		return x < info_empty;  // normal. both top bits are set. 0xC0
	}
	inline void set_empty(info_type & x) {
		x |= info_empty;  // nothing here.
	}
	inline void set_normal(info_type & x) {
		x &= info_mask;  // nothing here.
	}
	inline info_type get_distance(info_type const & x) const {
		return x & info_mask;  // nothing here.
	}
	// make above explicit by preventing automatic type conversion.
	template <typename TT> inline bool is_empty(TT const & x) const  = delete;
	template <typename TT> inline bool is_normal(TT const & x) const = delete;
	template <typename TT> inline void set_empty(TT & x) = delete;
	template <typename TT> inline void set_normal(TT & x) = delete;
	template <typename TT> inline info_type get_distance(TT const & x) const = delete;

	//=========  end INFO_TYPE definitions.
    // filter
    struct valid_entry_filter {
    	inline bool operator()(info_type const & x) {   // a container entry is empty only if the corresponding info is empty (0x80), not just have empty flag set.
    		return x != info_empty;   // (curr bucket is empty and position is also empty.  otherwise curr bucket is here or prev bucket is occupying this)
    	};
    };



    using container_type		= ::std::vector<value_type, Allocator>;
    using info_container_type	= ::std::vector<info_type, Allocator>;


public:

    using allocator_type        = typename container_type::allocator_type;
    using reference 			= typename container_type::reference;
    using const_reference	    = typename container_type::const_reference;
    using pointer				= typename container_type::pointer;
    using const_pointer		    = typename container_type::const_pointer;
    using iterator              = ::bliss::iterator::aux_filter_iterator<typename container_type::iterator, typename info_container_type::iterator, valid_entry_filter>;
    using const_iterator        = ::bliss::iterator::aux_filter_iterator<typename container_type::const_iterator, typename info_container_type::const_iterator, valid_entry_filter>;
    using size_type             = typename container_type::size_type;
    using difference_type       = typename container_type::difference_type;


protected:


    //=========  start BUCKET_ID_TYPE definitions.
    // find_pos returns 2 pieces of information:  assigned bucket, and the actual position.  actual position is set to all bits on if not found.
    // so 3 pieces of information needs to be incorporated.
    //   note that pos - bucket = offset, where offset just needs 1 bytes.
    //   we can try to limit bucket id or pos to only 56 bits, to avoid constructing a pair.
    // 56 bits gives 64*10^15, or 64 peta entries (local entries).  should be enough for now...
    // majority - looking for position, not bucket, but sometimes do need bucket pos.
    // also need to know if NOT FOUND - 1. bucket is empty, 2. bucket does not contain this entry.
    //    if bucket empty, then the offset has the empty flag on.  pos should bucket + offset.
    //    if bucket does not contain entry, then pos should be start pos of next bucket.  how to indicate nothing found? use the MSB of the 56 bits.
    // if found, then return pos, and offset of original bucket (for convenience).
    // NOTE that the pos part may not point to the first entry of the bucket.

    // failed insert (pos_flag set) means the same as successful find.

    using bucket_id_type = size_t;
//    static constexpr bucket_id_type bid_pos_mask = ~(static_cast<bucket_id_type>(0)) >> 9;   // lower 55 bits set.
//    static constexpr bucket_id_type bid_pos_exists = 1UL << 55;  // 56th bit set.
      static const bucket_id_type bid_pos_mask = ~(static_cast<bucket_id_type>(0)) >> 1;   // lower 63 bits set.
      static const bucket_id_type bid_pos_exists = 1ULL << 63;  // 64th bit set.
//    static constexpr bucket_id_type bid_info_mask = static_cast<bucket_id_type>(info_mask) << 56;   // lower 55 bits set.
//    static constexpr bucket_id_type bid_info_empty = static_cast<bucket_id_type>(info_empty) << 56;  // 56th bit set.

    // failed is speial, correspond to all bits set (max distnace failed).  not using 0x800000... because that indicates failed inserting due to occupied.
// NOT USED    static constexpr bucket_id_type bid_failed = ~(static_cast<bucket_id_type>(0));


    inline bucket_id_type make_missing_bucket_id(size_t const & pos) const { //, info_type const & info) const {
      //      assert(pos <= bid_pos_mask);
      //      return (static_cast<bucket_id_type>(info) << 56) | pos;
      assert(pos < bid_pos_exists);
      return static_cast<bucket_id_type>(pos);
    }
    inline bucket_id_type make_existing_bucket_id(size_t & pos) const { //, info_type const & info) const {
      //      return make_missing_bucket_id(pos, info) | bid_pos_exists;
      // return static_cast<bucket_id_type>(*pos) | bid_pos_exists;
      reinterpret_cast<uint32_t*>(&pos)[1] |= 0x80000000U;
      return static_cast<bucket_id_type>(pos);
    }

    // NOT USED
//    inline bool is_empty(bucket_id_type const & x) const {
//      return x >= bid_info_empty;  // empty 0x80....
//    }
    // NOT USED
//    inline bool is_normal(bucket_id_type const & x) const {
//      return x < bid_info_empty;  // normal. both top bits are set. 0xC0
//    }
    inline bool exists(bucket_id_type const & x) const {
      //return (x & bid_pos_exists) > 0;
      return x > bid_pos_mask;
    }
    inline bool missing(bucket_id_type const & x) const {
      // return (x & bid_pos_exists) == 0;
      return x < bid_pos_exists;
    }
    // NOT USED
//    inline info_type get_info(bucket_id_type const & x) const {
//      return static_cast<info_type>(x >> 56);
//    }

    inline size_t get_pos(bucket_id_type const & x) const {
      return x & bid_pos_mask;
    }
    // NOT USED
//    inline size_t get_offset(bucket_id_type const & x) const {
//      return (x & bid_info_mask) >> 56;
//    }

	// make above explicit by preventing automatic type conversion.
//    template <typename SS, typename TT>
//    inline bucket_id_type make_missing_bucket_id(SS const & pos /*, TT const & info */) const  = delete;
//    template <typename SS, typename TT>
//    inline bucket_id_type make_existing_bucket_id(SS const & pos /*, TT const & info */) const  = delete;
	template <typename TT> inline bool exists(TT const & x) const = delete;
	template <typename TT> inline bool missing(TT const & x) const = delete;
// NOT USED.	template <typename TT> inline info_type get_info(TT const & x) const = delete;
	template <typename TT> inline size_t get_pos(TT const & x) const = delete;
// NOT USED.	template <typename TT> inline size_t get_offset(TT const & x) const = delete;


    //=========  end BUCKET_ID_TYPE definitions.


    size_t lsize;
    mutable size_t buckets;
    mutable size_t mask;
    mutable size_t min_load;
    mutable size_t max_load;
    mutable float min_load_factor;
    mutable float max_load_factor;

#if defined(REPROBE_STAT)
    // some stats.
    size_type upsize_count;
    size_type downsize_count;
    mutable size_type reprobes;   // for use as temp variable
    mutable info_type max_reprobes;
    mutable size_type moves;
    mutable size_type max_moves;
    mutable size_type shifts;
    mutable size_type max_shifts;
#endif


    valid_entry_filter filter;
    hasher hash;
    key_equal eq;

    container_type container;
    info_container_type info_container;

public:

    /**
     * _capacity is the number of usable entries, not the capacity of the underlying container.
     */
	explicit hashmap_robinhood_doubling_offsets(size_t const & _capacity = 128,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
			lsize(0), buckets(next_power_of_2(_capacity)), mask(buckets - 1),
#if defined (REPROBE_STAT)
      upsize_count(0), downsize_count(0),
#endif
			container(buckets + info_empty), info_container(buckets + info_empty, info_empty)
			{
		// set the min load and max load thresholds.  there should be a good separation so that when resizing, we don't encounter a resize immediately.
		set_min_load_factor(_min_load_factor);
		set_max_load_factor(_max_load_factor);
	};

	/**
	 * initialize and insert, allocate about 1/4 of input, and resize at the end to bound the usage.
	 */
	template <typename Iter, typename = typename std::enable_if<
			::std::is_constructible<value_type, typename ::std::iterator_traits<Iter>::value_type>::value  ,int>::type >
	hashmap_robinhood_doubling_offsets(Iter begin, Iter end,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
		hashmap_robinhood_doubling_offsets(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

    insert(begin, end);
	}

	~hashmap_robinhood_doubling_offsets() {
#if defined(REPROBE_STAT)
		::std::cout << "SUMMARY:" << std::endl;
		::std::cout << "  upsize\t= " << upsize_count << std::endl;
		::std::cout << "  downsize\t= " << downsize_count << std::endl;
#endif
		}

	/**
	 * @brief set the load factors.
	 */
	inline void set_min_load_factor(float const & _min_load_factor) {
		min_load_factor = _min_load_factor;
		min_load = static_cast<size_t>(static_cast<float>(buckets) * min_load_factor);

	}

	inline void set_max_load_factor(float const & _max_load_factor) {
		max_load_factor = _max_load_factor;
		max_load = static_cast<size_t>(static_cast<float>(buckets) * max_load_factor);
	}

	/**
	 * @brief get the load factors.
	 */
	inline float get_load_factor() {
		return static_cast<float>(lsize) / static_cast<float>(buckets);
	}

	inline float get_min_load_factor() {
		return min_load_factor;
	}

	inline float get_max_load_factor() {
		return max_load_factor;
	}

	size_t capacity() {
		return buckets;
	}


	/**
	 * @brief iterators
	 */
	iterator begin() {
		return iterator(container.begin(), info_container.begin(), info_container.end(), filter);
	}

	iterator end() {
		return iterator(container.end(), info_container.end(), filter);
	}

	const_iterator cbegin() const {
		return const_iterator(container.cbegin(), info_container.cbegin(), info_container.cend(), filter);
	}

	const_iterator cend() const {
		return const_iterator(container.cend(), info_container.cend(), filter);
	}



  void print() const {
	  std::cout << "lsize " << lsize << " buckets " << buckets << " max load factor " << max_load_factor << std::endl;
    for (size_type i = 0; i < info_container.size() - 1; ++i) {
      std::cout << "i " << i << " key " << container[i].first << " val " <<
          container[i].second << " info " <<
          static_cast<size_t>(info_container[i]) << " offset = " << static_cast<size_t>(get_distance(info_container[i])) <<
          " pos = " << (i + get_distance(info_container[i])) <<
          " count " << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1));
      std::cout << std::endl;
    }


  }


	std::vector<std::pair<key_type, mapped_type> > to_vector() const {
		std::vector<std::pair<key_type, mapped_type> > output(lsize);

		auto it = std::copy(this->cbegin(), this->cend(), output.begin());
		output.erase(it, output.end());

		return output;
	}

	std::vector<key_type > keys() const {
		std::vector<key_type > output(lsize);

		auto it = std::transform(this->cbegin(), this->cend(), output.begin(),
				[](value_type const & x){ return x.first; });
		output.erase(it, output.end());

		return output;
	}


	size_t size() const {
		return this->lsize;
	}

	/**
	 * @brief  mark all entries as clear.
	 */
	void clear() {
		this->lsize = 0;
    std::fill(this->info_container.begin(), this->info_container.end(), info_empty);
	}

	/**
	 * @brief reserve space for specified entries.
	 */
  void reserve(size_type n) {
//    if (n > this->max_load) {   // if requested more than current max load, then we need to resize up.
      rehash(static_cast<size_t>(static_cast<float>(n) / this->max_load_factor));
      // rehash to the new size.    current bucket count should be less than next_power_of_2(n).
//    }  // do not resize down.  do so only when erase.
  }

  /**
   * @brief reserve space for specified buckets.
   * @details note that buckets > entries.
   */
  void rehash(size_type const & b) {

    // check it's power of 2
    size_type n = next_power_of_2(b);

//    std::cout << "REHASH b " << b << " n " << n << " lsize " << lsize << std::endl;

//    print();

    if ((n != buckets) && (lsize < (max_load_factor * n))) {

      container_type tmp(n + info_empty);
      info_container_type tmp_info(n + info_empty, info_empty);

      if (lsize > 0) {
        if (n > buckets) {
          this->copy_upsize(tmp, tmp_info, n);
    #if defined(REPROBE_STAT)
          ++upsize_count;
    #endif
        } else {
          this->copy_downsize2(tmp, tmp_info, n);
    #if defined(REPROBE_STAT)
          ++downsize_count;
    #endif
        }
      }

      // new size and mask
      buckets = n;
      mask = n - 1;

      min_load = static_cast<size_type>(static_cast<float>(n) * min_load_factor);
      max_load = static_cast<size_type>(static_cast<float>(n) * max_load_factor);

      // swap in.
      container.swap(tmp);
      info_container.swap(tmp_info);
    }
  }


protected:

  /**
   *  @brief copy from hash table the supplied target (size should be already specified.)
   *  @details  Note that we essentially are merging partitions from the source.
   *
   *        iterating over source means that when writing into target, we will need to shift the entire target container, potentially.
   *
   *        prob better to iterate over multiple partitions, so that target is filled sequentially.
   *
   *        figure out the scaling factor, and create an array to track iteration position as we go.   note that each may need to be at diff points...
   *
   */
  void copy_downsize(container_type & target, info_container_type & target_info,
                     size_type const & target_buckets) {
    size_type m = target_buckets - 1;
    assert((target_buckets & m) == 0);   // assert this is a power of 2.

    size_t id = 0, bid;
    size_t pos;
    size_t endd;

//    std::cout << "RESIZE DOWN " << target_buckets << std::endl;

    std::vector<size_t> offsets(target_buckets, 0);
    size_t iterations = buckets / target_buckets;

    // initialize as size storage for now.
    std::fill(target_info.begin(), target_info.begin() + target_buckets, 0);

    // compute the sizes of the new bucket first.  can go in target_info for now.
    for (size_t it = 0; it < iterations; ++it) {
    	for (bid = 0; bid < target_buckets; ++bid, ++id) {
    		if (is_normal(info_container[id])) {
    			target_info[bid] += get_distance(info_container[id + 1]) + 1 - get_distance(info_container[id]);
    		}
    	}
    }

    // then get the EXCLUSIVE prefix sum.  in vector of size_t
    offsets[0] = 0;
    for (bid = 0; bid < m; ++bid) {
    	offsets[bid + 1] = std::max(offsets[bid] + target_info[bid], bid + 1);  // empty slot points to self, and nonempty still need to advance..
    }

    // then copy to the right positions, using the offsets to keep track.
    id = 0;
    for (size_t it = 0; it < iterations; ++it) {
    	for (bid = 0; bid < target_buckets; ++bid, ++id) {
    		if (is_normal(info_container[id])) {
    			// get the range
    			pos = id + get_distance(info_container[id]);
    			endd = id + 1 + get_distance(info_container[id + 1]);

    			// copy the range.
    			memmove(&(target[offsets[bid]]), &(container[pos]), sizeof(value_type) * (endd - pos));

    			// adjust the offset
    			offsets[bid] += endd - pos;
    		}
    	}
    }

    // and setup the target_info offsets.  offsets now have inclusive prefix sum
    target_info[0] = target_info[0] == 0 ? info_empty : info_normal;
    for (bid = 1; bid < target_buckets; ++bid) {
    	// offsets - target_info gets back the exclusive scan.  offsets[bid] is pos bucket end from 0.  need offset from start of bucket, so
    	// subtract count (in target_info[bid], gives position of bucket start from 0), and subtract bid (gives offset from bid.)
    	target_info[bid] = (target_info[bid] == 0 ? info_empty : info_normal) + static_cast<info_type>(offsets[bid] - bid - static_cast<size_t>(target_info[bid]));
    }
    // adjust the target_info at the end, in the padding region.
    for (bid = target_buckets; bid < offsets[target_buckets - 1]; ++bid) {
    	target_info[bid] = target_info[bid - 1] - 1;
    	set_empty(target_info[bid]);
    }


//    // iterate through the entire input.
//    for (size_t i = 0; i < buckets; ++i) {
//      if (is_normal(info_container[i])) {  // copy bucket ranges for non-empty buckets.
//    	  //doing a range allows memcpy, or at least to skip rehash.
//        // get start and end of bucket.  exclusive end.
//        pos = i + static_cast<size_t>(get_distance(info_container[i]));
//        endd = i + 1 + static_cast<size_t>(get_distance(info_container[i + 1]));
//
//        // compute the new id from current bucket.  since downsize and power of 2, modulus would do.
//        id = i & m;
//
//        // do batch copy to bucket 'id', copy from pos to endd.  the shifting may be compacting out empty spaces.
//        if ((endd - pos) > 1) copy_with_hint(target, target_info, id, container, pos, endd);
//        else copy_with_hint(target, target_info, id, container[pos]);
//
//      }  // else is empty, so continue.
//    }
//    std::cout << "RESIZE DOWN DONE " << target_buckets << std::endl;
  }
  void copy_downsize2(container_type & target, info_container_type & target_info,
                     size_type const & target_buckets) {
    size_type m = target_buckets - 1;
    assert((target_buckets & m) == 0);   // assert this is a power of 2.

    size_t id = 0, bid;
    size_t pos;
    size_t endd;

//    std::cout << "RESIZE DOWN " << target_buckets << std::endl;

    size_t new_start = 0, new_end = 0;

    size_t blocks = buckets / target_buckets;

    // iterate over all matching blocks.  fill one target bucket at a time and immediately fill the target info.
    size_t bl;
    for (bid = 0; bid < target_buckets; ++bid) {
		// starting offset is maximum of bid and prev offset.
		new_start = std::max(bid, new_end);
		new_end = new_start;

		for (bl = 0; bl < blocks; ++bl) {
			id = bid + bl * target_buckets;


			if (is_normal(info_container[id])) {
				// get the range
				pos = id + get_distance(info_container[id]);
				endd = id + 1 + get_distance(info_container[id + 1]);

				// copy the range.
//				std::cout << " copy from " << pos << " to " << new_end << " length " << (endd - pos) << std::endl;
				memmove(&(target[new_end]), &(container[pos]), sizeof(value_type) * (endd - pos));

				new_end += (endd - pos);

//				if (bid == target_buckets - 1)
//					std::cout << " last: " << bid << " from id " << id << " old " << pos << "-" << endd << " new " << new_start << "-" << new_end << std::endl;
			}
    	}

		// offset - current bucket id.
		target_info[bid] = ((new_end - new_start) == 0 ? info_empty : info_normal) + new_start - bid;
//		if (bid == target_buckets - 1)
//			std::cout << " info: " << bid << " from id " << id << " info " << static_cast<size_t>(target_info[bid]) << std::endl;
    }
    // adjust the first padding target info.
//    if ((new_end - new_start) > 0) target_info[target_buckets] = info_empty + new_end - target_buckets;
//    else target_info[target_buckets] = target_info[target_buckets - 1] - 1;

//	std::cout << " info: " << (target_buckets - 1) << " info " << static_cast<size_t>(target_info[target_buckets - 1]) << " entry " << target[target_buckets - 1].first << std::endl;
    // adjust the target_info at the end, in the padding region.
    for (bid = target_buckets; bid < new_end; ++bid) {
    	new_start = std::max(bid, new_end);  // fixed new_end.  get new start.
    	// if last one is not empty, then first padding position is same distance with
    	target_info[bid] = info_empty + new_start - bid;
//		std::cout << " info: " << bid << " info " << static_cast<size_t>(target_info[bid]) << " entry " << target[bid].first << std::endl;
    }


//    // iterate through the entire input.
//    for (size_t i = 0; i < buckets; ++i) {
//      if (is_normal(info_container[i])) {  // copy bucket ranges for non-empty buckets.
//    	  //doing a range allows memcpy, or at least to skip rehash.
//        // get start and end of bucket.  exclusive end.
//        pos = i + static_cast<size_t>(get_distance(info_container[i]));
//        endd = i + 1 + static_cast<size_t>(get_distance(info_container[i + 1]));
//
//        // compute the new id from current bucket.  since downsize and power of 2, modulus would do.
//        id = i & m;
//
//        // do batch copy to bucket 'id', copy from pos to endd.  the shifting may be compacting out empty spaces.
//        if ((endd - pos) > 1) copy_with_hint(target, target_info, id, container, pos, endd);
//        else copy_with_hint(target, target_info, id, container[pos]);
//
//      }  // else is empty, so continue.
//    }
//    std::cout << "RESIZE DOWN DONE " << target_buckets << std::endl;
  }




  /**
  * @brief inserts a range into the current hash table.
  * @details
  * essentially splitting the source into multiple non-overlapping ranges.
  *    each partition is filled nearly sequentially, so figure out the scaling factor, and create an array as large as the scaling factor.
  *
  */
  void copy_upsize(container_type & target, info_container_type & target_info,
                   size_type const & target_buckets) {
    size_type m = target_buckets - 1;
    assert((target_buckets & m) == 0);   // assert this is a power of 2.

//    std::cout << "RESIZE UP " << target_buckets << std::endl;

    size_t id, bid, p;
    size_t pos;
    size_t endd;
    value_type v;

    size_t bl;
    size_t blocks = target_buckets / buckets;
    std::vector<size_t> offsets(blocks + 1, 0);
    std::vector<size_t> len(blocks, 0);

    // let's store the hash in order to avoid redoing hash.  This is needed only because we need to first count the number in a block,
    //  so that at block boundaries we have the right offsets.
    std::vector<size_t> hashes(lsize);
    size_t j = 0;
    // compute and store all hashes, and at the same time compute end of last bucket in each block.
    for (bid = 0; bid < buckets; ++bid) {
    	if (is_normal(info_container[bid])) {

    		pos = bid + get_distance(info_container[bid]);
    		endd = bid + 1 + get_distance(info_container[bid + 1]);

    		for (p = pos; p < endd; ++p, ++j) {
        		// eval the target id.
    			hashes[j] = hash(container[p].first);
    			id = hashes[j] & m;

    			// figure out which block it is in.
    			bl = id / buckets;

    			// count.
    			offsets[bl + 1] = std::max(offsets[bl + 1], id) + 1;
    		}
    	}
    }

//    for (bl = 0; bl <= blocks; ++bl) {
//    	std::cout << "OFFSETS "  << offsets[bl] << std::endl;
//    }

    // now that we have the right offsets,  start moving things.
    j = 0;
    size_t pp;
    for (bid = 0; bid < buckets; ++bid) {
    	if (is_normal(info_container[bid])) {

    		pos = bid + get_distance(info_container[bid]);
    		endd = bid + 1 + get_distance(info_container[bid + 1]);

    		std::fill(len.begin(), len.end(), 0);

    		for (p = pos; p < endd; ++p, ++j) {
        		// eval the target id.
    			id = hashes[j] & m;

    			// figure out which block it is in.
    			bl = id / buckets;

    			// now copy
    			pp = std::max(offsets[bl], id);
    			target[pp] = container[p];

//    			std::cout << " moved from " << p << " to " << pp << " block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;

    			// count.
    			offsets[bl] = pp + 1;
    			++len[bl];

    		}

    		// update all positive ones.
    		for (bl = 0; bl < blocks; ++bl) {
    			id = bid + bl * buckets;
    			target_info[id] = (len[bl] == 0 ? info_empty : info_normal) + static_cast<info_type>(std::max(offsets[bl], id) - id - len[bl]);
//    			std::cout << " updated info at " << id << " to " << static_cast<size_t>(target_info[id]) << ". block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;
    		}
    	} else {
    		for (bl = 0; bl < blocks; ++bl) {
    			id = bid + bl * buckets;
    			target_info[id] = info_empty + static_cast<info_type>(std::max(offsets[bl], id) - id);
//    			std::cout << " updated empty info at " << id << " to " << static_cast<size_t>(target_info[id]) << ". block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;
    		}

    	}
    }
    // clean up the last part.
    size_t new_start;
    for (bid = target_buckets; bid < offsets[blocks]; ++bid) {
    	new_start = std::max(bid, offsets[blocks]);  // fixed new_end.  get new start.
    	// if last one is not empty, then first padding position is same distance with
    	target_info[bid] = info_empty + new_start - bid;
//		std::cout << " info: " << bid << " info " << static_cast<size_t>(target_info[bid]) << " entry " << target[bid].first << std::endl;
    }


//	for (size_t i = 0; i < buckets; ++i) {
//		if (info_container[i] != info_empty) {
//			// any full entry, or empty entry that with offset greater than 0, has non-zero container entry.
//			// comparing to info_empty lets us use a single loop and access info_continer just once.
//			// FOR PREFETCH, can use "full" entries to get a range to preload.
//
//			v = container[i];
//
//			// compute the new id via hash.
//			id = hash(v.first) & m;
//
//			// does not do equal comparisons.
//			copy_with_hint(target, target_info, id, v);
//		}  // else is empty, so continue.
//	}
//    std::cout << "RESIZE UP DONE " << target_buckets << std::endl;
  }


	/**
	 * return the position in container where the current key is found.  if not found, max is returned.
	 */
	bucket_id_type find_pos_with_hint(key_type const & k, size_t const & bid) const {

		assert(bid < buckets);

		info_type offset = info_container[bid];
		size_t start = bid + get_distance(offset);  // distance is at least 0, and definitely not empty

		// no need to check for empty?  if i is empty, then this one should be.
		// otherwise, we are checking distance so doesn't matter if empty.

		// first get the bucket id
		if (is_empty(offset) ) {
			// std::cout << "Empty entry at " << i << " val " << static_cast<size_t>(info_container[i]) << std::endl;
			//return make_missing_bucket_id(start, offset);
		  return make_missing_bucket_id(start);
		}

		// get the next bucket id
		size_t end = bid + 1 + get_distance(info_container[bid + 1]);   // distance is at least 0, and can be empty.

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		// now we scan through the current.
		for (; start < end; ++start) {

			if (eq(k, container[start].first)) {
//				return make_existing_bucket_id(start, offset);
	        return make_existing_bucket_id(start);
			}

#if defined(REPROBE_STAT)
			++reprobe;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

    return make_missing_bucket_id(start);
//		return make_missing_bucket_id(end, offset);
	}

	/**
	 * return the bucket id where the current key is found.  if not found, max is returned.
	 */
	inline bucket_id_type find_pos(key_type const & k) const {
	  size_t i = hash(k) & mask;
	  return find_pos_with_hint(k, i);

	}

	/**
	 * return the next position in "container" that is empty - i.e. info has 0x80.
	 *
	 * one property to use - we can jump the "distance" in the current info_container, there are guaranteed no
	 *   empty entries within that range.
	 *
	 *   remember that distances can be non-zero for empty BUCKETs.
	 *
	 *   container should not be completely full because of rehashing, so there should be no need to break search into 2 parts.
	 *
	 */
	inline size_t find_next_empty_pos(info_container_type const & target_info, size_t const & pos) const {
		size_t end = pos;
		for (; (end < target_info.size()) && (target_info[end] != info_empty); ) {
			// can skip ahead with target_info[end]
			end += get_distance(target_info[end]);
			end += (target_info[end] == info_normal);
		}

		return end;
	}

	/**
	 * return the next position in "container" that is empty - i.e. info has 0x80.
	 *
	 * one property to use - we can jump the "distance" in the current info_container, there are guaranteed no
	 *   empty entries within that range.
	 *
	 *   remember that distances can be non-zero for empty BUCKETs.
	 *
	 *   container should not be completely full because of rehashing, so there should be no need to break search into 2 parts.
	 *
	 */
	inline size_t find_next_zero_offset_pos(info_container_type const & target_info, size_t const & pos) const {
		info_type dist;
		size_t end = pos;
		for (; end < target_info.size(); ) {
			dist = get_distance(target_info[end]);
			if (dist == 0) return end;
			// can skip ahead with target_info[end]
			end += dist;
		}

		return end;
	}


	/**
	 * find next non-empty position.  including self.
	 */
	inline size_type find_next_non_empty_pos(info_container_type const & target_info, size_t const & pos) const {
		size_t end = pos;
		for (; end < target_info.size(); ) {
			if (is_normal(target_info[end])) return end;
			// can skip ahead with target_info[end]
			end += get_distance(target_info[end]);
			end += (target_info[end] == info_empty);
		}

		return end;
	}



	/**
	 * @brief insert a single key-value pair into container at the desired position
	 *
	 * note that swap only occurs at bucket boundaries.
	 *
	 * return insertion position, and update id to end of
			// insert if empty, or if reprobe distance is larger than current position's.  do this via swap.
			// continue to probe if we swapped out a normal entry.
			// this logic relies on choice of bits for empty entries.

			// we want the reprobe distance to be larger than empty, so we need to make
			// normal to have high bits 1 and empty 0.

			// save the insertion position (or equal position) for the first insert or match, but be aware that after the swap
			//  there will be other inserts or swaps. BUT NOT OTHER MATCH since this is a MAP.

		// 4 cases:
		// A. empty bucket, offset == 0        use this bucket. offset at original bucket converted to non-empty.  move vv in.  next bucket unchanged.  done
		// B. empty bucket, offset > 0         use this bucket. offset at original bucket converted to non-empty.  swap vv in.  go to next bucket
		// C. non empty bucket, offset == 0.   use next bucket. offset at original bucket not changed              swap vv in.  go to next bucket
		// D. non empty bucket, offset > 0.    use next bucket. offset at original bucket not changed.             swap vv in.  go to next bucket

//      // alternative for each insertion: vertical curr, horizontal next.
//      // operations.  Op X: move vv to curr pos
//      //				Op Y: swap vv with curr pos
//      //				Op Z: no swap
//      //				Op R: change curr info to non empty
//      //				Op S: increment next info by 1
//      //				Op T: increment next infoS by 1, up till a full entry, or info_empty.
//      //				Op Y2: move vv to next pos, change curr info to non empty, next infoS +1 until either full, or info_empty.  done
//      //				Op Z: swap vv to next pos, change curr info to non empty, next infos +1,  go to next bucket (using next next bucket).  repeat
//      //				Op NA: not possible and
//      // 		A		B		C		D
//      // A	X		NA		X 		NA
//      // B	Y		Y2		Z		Z
//      // C
//      // D
//      //
//      // need operation to shift empty bucket offsets.

		// alternative, scan forward in info_container for info_empty slots, as many as needed, (location where we can insert).  also PREFETCH here
		// then compact container back to front while updating the associated info_container
		//		(update is from start of next bucket to empty slot, shift all those by 1, each empty space encounter increases shift distance by 1.)
		//      increased shift distance then increases the number of swaps.  possibly memmove would be better.
		//      each swap is 2 reads and 2 writes.  whole cacheline access might be simpler when bucket is small, and swap may be better when shift amount is small.
		//		  use large buckets to break up the shift.  everything in between just use memmove (catch multiple small buckets with 1 memmove).
		//   treat as optimization.  initially, just memmove for all.
		// finally move in the new data.

		// return bucket_id_type with info_type of CURRENT info_type
	 */
	inline bucket_id_type insert_with_hint(container_type & target,
			info_container_type & target_info,
			size_t const & id,
			value_type const & v) {

		assert(id < buckets);

		// get the starting position
		info_type info = target_info[id];

//		std::cout << "info " << static_cast<size_t>(info) << std::endl;
		set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.

		// if this is empty and no shift, then insert and be done.
		if (info == info_empty) {
			target[id] = v;
			return make_missing_bucket_id(id);
//			return make_missing_bucket_id(id, target_info[id]);
		}

		// the bucket is either non-empty, or empty but offset by some amount

		// get the range for this bucket.
		size_t start = id + get_distance(info);
		size_t next = id + 1 + get_distance(target_info[id + 1]);

		// now search within bucket to see if already present.
		if (is_normal(info)) {  // only for full bucket, of course.

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
			for (size_t i = start; i < next; ++i) {
				if (eq(v.first, target[i].first)) {
					// check if value and what's in container match.
//					std::cout << "EXISTING.  " << v.first << ", " << target[i].first << std::endl;

				  //return make_existing_bucket_id(i, info);
          return make_existing_bucket_id(i);
				}
#if defined(REPROBE_STAT)
			++reprobe;
#endif
			}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		}

		// now for the non-empty, or empty with offset, shift and insert, starting from NEXT bucket.


		// first swap in at start of bucket, then deal with the swapped.
		// swap until empty info
		// then insert into last empty,
		// then update info until info_empty

		// scan for the next empty position
		size_t end = find_next_empty_pos(target_info, next);

//		std::cout << "val " << v.first << " id " << id <<
//				" start " << static_cast<size_t>(start) <<
//				" next " << static_cast<size_t>(next) <<
//				" end " << static_cast<size_t>(end) << std::endl;

		// now compact backwards.  first do the container via MEMMOVE
		// can potentially be optimized to use only swap, if distance is long enough.
		memmove(&(target[next + 1]), &(target[next]), sizeof(value_type) * (end - next));
		// and increment the infos.
		for (size_t i = id + 1; i <= end; ++i) {
			++(target_info[i]);
		}
#if defined(REPROBE_STAT)
		this->shifts += (end - next);
		this->max_shifts = std::max(this->max_shifts, (end - next));
		this->moves += (end - next);
		this->max_moves = std::max(this->max_moves, (end - next));
#endif

		// that's it.
		target[next] = v;
//		return make_missing_bucket_id(next, target_info[id]);
    return make_missing_bucket_id(next);

	}

	inline void copy_with_hint(container_type & target,
			info_container_type & target_info,
			size_t const & id,
			value_type const & v) {

		assert(id < buckets);

		// get the starting position
		info_type info = target_info[id];
		set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.

		// if this is empty and no shift, then insert and be done.
		if (info == info_empty) {
			target[id] = v;
			return;
		}

		// the bucket is either non-empty, or empty but offset by some amount

		// get the range for this bucket.
		size_t next = id + 1 + get_distance(target_info[id + 1]);

		// now for the non-empty, or empty with offset, shift and insert, starting from NEXT bucket.

		// first swap in at start of bucket, then deal with the swapped.
		// swap until empty info
		// then insert into last empty,
		// then update info until info_empty

		// scan for the next empty position
		size_t end = find_next_empty_pos(target_info, next);

		// now compact backwards.  first do the container via MEMMOVE.
		// can potentially be optimized to use only swap, if distance is long enough.
		memmove(&(target[next + 1]), &(target[next]), sizeof(value_type) * (end - next));
		// and increment the infos.
		for (size_t i = id + 1; i <= end; ++i) {
			++(target_info[i]);
		}
#if defined(REPROBE_STAT)
		this->shifts += (end - next);
		this->max_shifts = std::max(this->max_shifts, (end - next));
		this->moves += (end - next);
		this->max_moves = std::max(this->max_moves, (end - next));
#endif

		// that's it.
		target[next] = v;
		return;
	}


//	/**
//	 * insert multiple positions at the same time.
//	 *
//	 * no repeats in sstart to send, and between input and target.
//	 *
//	 * shift the current bucket and insert at front of bucket.
//	 */
//	inline void copy_with_hint(container_type & target,
//			info_container_type & target_info,
//			size_t const & id,
//			container_type const & source,
//			size_t const & sstart, size_t const & send) {
//
//      BROKEN.
//
//		assert(id < buckets);
//
//		if (sstart == send) return;
//
//		size_t count = send - sstart;
//
//		// get the starting position
//		info_type info = target_info[id];
//		set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.
//
//		// get the range for this bucket.
//		size_t start = id + get_distance(info);
//
//		// scan for the next X number of empty positions
//		size_t end = id + 1 + get_distance(target_info[id + 1]);
//		std::vector<size_t> empties;
//		empties.reserve(count);
//		size_t c = 0;
//		for (; (end < target_info.size()) && (c < count);) {
//			if (target_info[end] == info_empty) {
//				empties.emplace_back(end);
//
//				std::cout << "from id " << id << " moving " << count << " empty at " << end << std::endl;
//				++c;
//			}
//			// can skip ahead with target_info[end]
//			end += std::max(get_distance(target_info[end]), static_cast<info_type>(1));  // by at least 1.
//		}
//
//		assert(empties.size() == count);
//
//		// now compact backwards.  first do the container via MEMMOVE
//		int i = count - 1;
//		end = empties[i];
//		size_t next;
//		for (; i > 0; --i) {
//			next = empties[i - 1];
//
//			std::cout << " moving " << next << " to " << end << " with i " << i << std::endl;
//
//			memmove(&(target[next + 1 + count - i]), &(target[next + 1]), sizeof(value_type) * (end - next - 1));
//			// and increment the infos.
//			for (size_t j = next + 1; j <= end; ++j) {
//				target_info[j] += count - i;
//			}
//
//			end = next;
//		}
//		// can potentially be optimized to use only swap, if distance is long enough.
//		memmove(&(target[start + count]), &(target[start]), sizeof(value_type) * (end - start));
//		// and increment the infos from id+1 to first empty entry.
//		for (size_t j = id + 1; j <= end; ++j) {
//			target_info[j] += count;
//		}
//
//		// now copy in.
//		memmove(&(target[end]), &(source[sstart]), sizeof(value_type) * count);
//
//#if defined(REPROBE_STAT)
//		this->shifts += (end - start);
//		this->max_shifts = std::max(this->max_shifts, (end - start));
//		this->moves += (end - start);
//		this->max_moves = std::max(this->max_moves, (end - start));
//#endif
//
//		return;
//	}



#if defined(REPROBE_STAT)
	void reset_reprobe_stats() {
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
		this->shifts = 0;
		this->max_shifts = 0;

	}

	void print_reprobe_stats(std::string const & operation, size_t input_size, size_t success_count) {
		std::cout << "hash table stat: lsize " << lsize << " buckets " << buckets << std::endl;

		std::cout << "hash table op stat: " << operation << ":" <<
				"\tsuccess=" << success_count << "\ttotal=" << input_size << std::endl;

		std::cout << "hash table reprobe stat: " << operation << ":" <<
				"\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
				"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
				"\tshift scanned max=" << static_cast<unsigned int>(this->max_shifts) << "\tshift scan total=" << this->shifts << std::endl;
	}
#endif







public:

	/**
	 * @brief insert a single key-value pair into container.
	 *
	 * note that swap only occurs at bucket boundaries.
	 */
	std::pair<iterator, bool> insert(value_type const & vv) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif

		// first check if we need to resize.
		if (lsize >= max_load) rehash(buckets << 1);

		// first get the bucket id
		bucket_id_type id = hash(vv.first) & mask;  // target bucket id.

		id = insert_with_hint(container, info_container, id, vv);
		bool success = missing(id);
		size_t bid = get_pos(id);

		if (success) ++lsize;

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT 1", 1, (success ? 1 : 0));
#endif

//		std::cout << "insert 1 lsize " << lsize << std::endl;
		return std::make_pair(iterator(container.begin() + bid, info_container.begin()+ bid, info_container.end(), filter), success);

	}

	std::pair<iterator, bool> insert(key_type const & key, mapped_type const & val) {
		return insert(std::make_pair(key, val));
	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	void insert(Iter begin, Iter end) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif
		bucket_id_type id;
		bucket_id_type insert_pos;

		// iterate based on size between rehashes
		for (auto it = begin; it != end ; ++it) {

			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			// first get the bucket id
			id = hash((*it).first) & mask;  // target bucket id.

			insert_pos = insert_with_hint(container, info_container, id, *it);
			if (missing(insert_pos))
				++lsize;

//			std::cout << "insert iter lsize " << lsize << " key " << (*it).first << " id " << id << " result " <<
//					std::hex << insert_pos << std::dec << std::endl;
		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER", std::distance(begin, end), (lsize - before));
#endif

		// NOT needed until we are estimating reservation size.
//		reserve(lsize);  // resize down as needed
	}



	/// batch insert not using iterator
	void insert(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif
		bucket_id_type id;

		// iterate based on size between rehashes
		for (size_t i = 0; i < input.size(); ++i) {

			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			// first get the bucket id
			id = hash(input[i].first) & mask;  // target bucket id.

			if (missing(insert_with_hint(container, info_container, id, input[i])))
				++lsize;

//			std::cout << "insert vec lsize " << lsize << std::endl;

		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT VEC", input.size(), (lsize - before));
#endif

		// NOT needed until we are estimating reservation size.
//		reserve(lsize);  // resize down as needed
	}




	/// batch insert using sorting.  This is about 4x slower on core i5-4200U (haswell) than integrated batch insertion above, even just for sort.
	void insert_integrated(::std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif
		bucket_id_type id;

		value_type v;
		info_type reprobe;
		info_type curr_info;
		bool found;

		// iterate based on size between rehashes
		for (size_t j = 0; j < input.size(); ++j) {

			found = false;

//			std::cout << "insert integrated lsize " << lsize << std::endl;
			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			// first get the bucket id
			v = input[j];
			id = hash(v.first) & mask;  // target bucket id.

			assert(id < buckets);

			// get the starting position
			info_type info = info_container[id];
			set_normal(info_container[id]);   // if empty, change it.  if normal, same anyways.

			// if this is empty and no shift, then insert and be done.
			if (info == info_empty) {
				container[id] = v;
				++lsize;
				continue;
			}

			// the bucket is either non-empty, or empty but offset by some amount

			// get the range for this bucket.
			size_t start = id + get_distance(info);
			size_t next = id + 1 + get_distance(info_container[id + 1]);

			// now search within bucket to see if already present.
			if (is_normal(info)) {  // only for full bucket, of course.

	#if defined(REPROBE_STAT)
			size_t reprobe = 0;
	#endif
				for (size_t i = start; i < next; ++i) {
					if (eq(v.first, container[i].first)) {
//						std::cout << "Insert Integrated EXISTING.  " << v.first << ", " << container[i].first << std::endl;
						// check if value and what's in container match.
						found = true;
						break;
					}
	#if defined(REPROBE_STAT)
				++reprobe;
	#endif
				}


	#if defined(REPROBE_STAT)
			this->reprobes += reprobe;
			this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
	#endif

				if (found) continue;   // skip the remaining.
			}

			// now for the non-empty, or empty with offset, shift and insert, starting from NEXT bucket.


			// first swap in at start of bucket, then deal with the swapped.
			// swap until empty info
			// then insert into last empty,
			// then update info until info_empty

			// scan for the next empty position
			size_t end = find_next_empty_pos(info_container, next);

			// now compact backwards.  first do the container via MEMMOVE
			// can potentially be optimized to use only swap, if distance is long enough.
			memmove(&(container[next + 1]), &(container[next]), sizeof(value_type) * (end - next));
			// and increment the infos.
			for (size_t i = id + 1; i <= end; ++i) {
				++(info_container[i]);
			}
			// that's it.
			container[next] = v;
			++lsize;

	#if defined(REPROBE_STAT)
			this->shifts += (end - next);
			this->max_shifts = std::max(this->max_shifts, (end - next));
			this->moves += (end - next);
			this->max_moves = std::max(this->max_moves, (end - next));
	#endif


		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT INTEGRATED", input.size(), (lsize - before));
#endif

	}


	/// batch insert using sorting.  This is about 4x slower on core i5-4200U (haswell) than integrated batch insertion above, even just for sort.
	template <typename LESS = ::std::less<key_type> >
	void insert_sort(::std::vector<value_type> const & input) {

	  throw ::std::logic_error("ERROR: DISABLED FOR NONCIRC VERSION");

	}

  void insert_shuffled(::std::vector<value_type> const & input) {

    throw ::std::logic_error("ERROR: DISABLED FOR NONCIRC VERSION");

  }


	/**
	 * @brief count the presence of a key
	 */
	inline size_type count( key_type const & k ) const {

		return exists(find_pos(k)) ? 1 : 0;

	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif
		::std::vector<size_type> counts;
		counts.reserve(std::distance(begin, end));

		size_t id;
		key_type k;
		bucket_id_type found;

		for (auto it = begin; it != end; ++it) {

			k = (*it).first;
			id = hash(k) & mask;

			found = find_pos_with_hint(k, id);

			counts.emplace_back(exists(found));
		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT iter", std::distance(begin, end), (lsize - before));
#endif
		return counts;
	}


	template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif
		::std::vector<size_type> counts;
		counts.reserve(std::distance(begin, end));


		size_t id;
		key_type k;
		bucket_id_type found;

		for (auto it = begin; it != end; ++it) {

			k = *it;
			id = hash(k) & mask;

			found = find_pos_with_hint(k, id);

			counts.emplace_back(exists(found));
		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT iter", std::distance(begin, end), (lsize - before));
#endif
		return counts;
	}

	/**
	 * @brief find the iterator for a key
	 */
	iterator find(key_type const & k) {

		bucket_id_type idx = find_pos(k);

		if (exists(idx))
      return iterator(container.begin() + get_pos(idx), info_container.begin()+ get_pos(idx),
          info_container.end(), filter);
		else
      return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		bucket_id_type idx = find_pos(k);

		if (exists(idx))
      return const_iterator(container.cbegin() + get_pos(idx), info_container.cbegin()+ get_pos(idx),
          info_container.cend(), filter);
		else
      return const_iterator(container.cend(), info_container.cend(), filter);


	}

	/**
	 * @brief.  updates current value.  behaves like insert, but overwrites the exists.
	 */
	iterator update(key_type const & k, mapped_type const & val) {
		// find and update.  if not present, insert.
		std::pair<iterator, bool> result = insert(k, val);

		if (! result.second) {  // not inserted and no exception, so an equal entry has been found.
			result.first->second = val;   // so update.

		} // else inserted. so updated.

		return result.first;
	}

	iterator update(value_type const & vv) {
		// find and update.  if not present, insert.
		std::pair<iterator, bool> result = insert(vv);

		if (! result.second) {  // not inserted and no exception, so an equal entry has been found.
			result.first->second = vv.second;   // so update.

		} // else inserted. so updated.

		return result.first;
	}


protected:
	/**
	 * @brief erases a key.  performs backward shift.  swap at bucket boundaries only.
	 */
	size_type erase_and_compact(key_type const & k) {
		size_t bid = hash(k) & mask;
		bucket_id_type found = find_pos_with_hint(k, bid);  // get the matching position

		if (missing(found)) {
			// did not find. done
			return 0;
		}

		--lsize;   // reduce the size by 1.

		size_t pos = get_pos(found);   // get bucket id
		size_t pos1 = pos + 1;
		// get the end of the non-empty range, starting from the next position.
		size_type bid1 = bid + 1;  // get the next bucket, since bucket contains offset for current bucket.

		size_type end = find_next_zero_offset_pos(info_container, bid1);

// debug		std::cout << "erasing " << k << " hash " << bid << " at " << found << " end is " << end << std::endl;

		// move to backward shift.  move [found+1 ... end-1] to [found ... end - 2].  end is excluded because it has 0 dist.
			memmove(&(container[pos]), &(container[pos1]), (end - pos1) * sizeof(value_type));



// debug		print();

		// now change the offsets.
			// if that was the last entry for the bucket, then need to change this to say empty.
			if (get_distance(info_container[bid]) == get_distance(info_container[bid1])) {  // both have same distance, so bid has only 1 entry
				set_empty(info_container[bid]);
			}


		// start from bid+1, end at end - 1.
			for (size_t i = bid1; i < end; ++i ) {
				--(info_container[i]);
			}
		return 1;

	}

	//============ ERASE


	public:

		/// single element erase with key.
		size_type erase_no_resize(key_type const & k) {
	#if defined(REPROBE_STAT)
			reset_reprobe_stats();
	#endif
			size_t erased = erase_and_compact(k);

	#if defined(REPROBE_STAT)
			print_reprobe_stats("ERASE 1", 1, erased);
	#endif
			return erased;
		}

		/// batch erase with iterator of value pairs.
		template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
			typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
		size_type erase_no_resize(Iter begin, Iter end) {

	#if defined(REPROBE_STAT)
			reset_reprobe_stats();
	#endif

			size_type before = lsize;

			for (auto it = begin; it != end; ++it) {
				erase_and_compact((*it).first);
			}

	#if defined(REPROBE_STAT)
			print_reprobe_stats("ERASE PAIR ITER", std::distance(begin, end), before - lsize);
	#endif
			return before - lsize;
		}

		/// batch erase with iterator of keys.
		template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
			typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
		size_type erase_no_resize(Iter begin, Iter end) {
	#if defined(REPROBE_STAT)
			reset_reprobe_stats();
	#endif

			size_type before = lsize;

			for (auto it = begin; it != end; ++it) {
				erase_and_compact(*it);
			}

	#if defined(REPROBE_STAT)
			print_reprobe_stats("ERASE KEY ITER", std::distance(begin, end), before - lsize);
	#endif
			return before - lsize;
		}

		/**
		 * @brief erases a key.
		 */
		size_type erase(key_type const & k) {

			size_type res = erase_no_resize(k);

	//		if (lsize < min_load) rehash(buckets >> 1);

			return res;
		}

		template <typename Iter>
		size_type erase(Iter begin, Iter end) {

			size_type erased = erase_no_resize(begin, end);

	//		if (lsize < min_load) reserve(lsize);

			return erased;
		}

};

template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_empty;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_mask;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_normal;

//template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
//constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_failed;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
const typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_pos_mask;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
const typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_pos_exists;
//template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
//constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_info_mask;
//template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
//constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_info_empty;


}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS_HPP_ */
