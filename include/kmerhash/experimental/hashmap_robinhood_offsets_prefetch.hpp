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
 * hashmap_robinhood_offsets_prefetch.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_HASHMAP_ROBINHOOD_OFFSETS_PREFETCH_HPP_
#define KMERHASH_HASHMAP_ROBINHOOD_OFFSETS_PREFETCH_HPP_

#include <vector>   // for vector.
#include <array>
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"
#include "mmintrin.h"  // emm: _mm_stream_si64

#include <stdlib.h> // posix_memalign
#include <stdexcept>

#include "kmerhash/hyperloglog64.hpp"

#include <xmmintrin.h>

//#define REPROBE_STAT

// should be easier for prefetching
#define LOOK_AHEAD 16

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
 *  [ ] estimate distinct element counts in input.
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
    hyperloglog64<key_type, hasher, 12> hll;  // precision of 6bits  uses 64 bytes, which should fit in a cache line.  sufficient precision.


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
      static constexpr bucket_id_type bid_pos_mask = ~(static_cast<bucket_id_type>(0)) >> 1;   // lower 63 bits set.
      static constexpr bucket_id_type bid_pos_exists = 1ULL << 63;  // 64th bit set.
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
      return static_cast<bucket_id_type>(pos) | bid_pos_exists;
//      reinterpret_cast<uint32_t*>(&pos)[1] |= 0x80000000U;
//      return static_cast<bucket_id_type>(pos);
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


	// =========  prefetch constants.
  static constexpr uint32_t info_per_cacheline = 64 / sizeof(info_type);
  static constexpr uint32_t value_per_cacheline = 64 / sizeof(value_type);
  static constexpr uint32_t info_prefetch_iters = (LOOK_AHEAD + info_per_cacheline - 1) / info_per_cacheline;
  static constexpr uint32_t value_prefetch_iters = (LOOK_AHEAD + value_per_cacheline - 1) / value_per_cacheline;
	// =========  END prefetech constants.


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


	hashmap_robinhood_doubling_offsets(hashmap_robinhood_doubling_offsets const & other) :
		hll(other.hll),
		lsize(other.lsize),
    	buckets(other.buckets),
    	mask(other.mask),
		min_load(other.min_load),
		max_load(other.max_load),
		min_load_factor(other.min_load_factor),
		max_load_factor(other.max_load_factor),
#if defined(REPROBE_STAT)
    // some stats.
    upsize_count(other.upsize_count),
    downsize_count(other.downsize_count),
    reprobes(other.reprobes),
    max_reprobes(other.max_reprobes),
    moves(other.moves),
    max_moves(other.max_moves),
    shifts(other.shifts),
    max_shifts(other.max_shifts),
#endif
    filter(other.filter),
    hash(other.hash),
    eq(other.eq),
    container(other.container),
    info_container(other.info_container) {};

    hashmap_robinhood_doubling_offsets & operator=(hashmap_robinhood_doubling_offsets const & other) {
		hll = other.hll;
		lsize = other.lsize;
    	buckets = other.buckets;
    	mask = other.mask;
		min_load = other.min_load;
		max_load = other.max_load;
		min_load_factor = other.min_load_factor;
		max_load_factor = other.max_load_factor;
#if defined(REPROBE_STAT)
    // some stats.
    upsize_count = other.upsize_count;
    downsize_count = other.downsize_count;
    reprobes = other.reprobes;
    max_reprobes = other.max_reprobes;
    moves = other.moves;
    max_moves = other.max_moves;
    shifts = other.shifts;
    max_shifts = other.max_shifts;
#endif
    filter = other.filter;
    hash = other.hash;
    eq = other.eq;
    container = other.container;
    info_container = other.info_container;
    }

	hashmap_robinhood_doubling_offsets(hashmap_robinhood_doubling_offsets && other) :
		hll(std::move(other.hll)),
		lsize(std::move(other.lsize)),
    	buckets(std::move(other.buckets)),
    	mask(std::move(other.mask)),
		min_load(std::move(other.min_load)),
		max_load(std::move(other.max_load)),
		min_load_factor(std::move(other.min_load_factor)),
		max_load_factor(std::move(other.max_load_factor)),
#if defined(REPROBE_STAT)
    // some stats.
    upsize_count(std::move(other.upsize_count)),
    downsize_count(std::move(other.downsize_count)),
    reprobes(std::move(other.reprobes)),
    max_reprobes(std::move(other.max_reprobes)),
    moves(std::move(other.moves)),
    max_moves(std::move(other.max_moves)),
    shifts(std::move(other.shifts)),
    max_shifts(std::move(other.max_shifts)),
#endif
    filter(std::move(other.filter)),
    hash(std::move(other.hash)),
    eq(std::move(other.eq)),
    container(std::move(other.container)),
    info_container(std::move(other.info_container)) {}

	hashmap_robinhood_doubling_offsets & operator=(hashmap_robinhood_doubling_offsets && other) {
		hll = std::move(other.hll);
		lsize = std::move(other.lsize);
    	buckets = std::move(other.buckets);
    	mask = std::move(other.mask);
		min_load = std::move(other.min_load);
		max_load = std::move(other.max_load);
		min_load_factor = std::move(other.min_load_factor);
		max_load_factor = std::move(other.max_load_factor);
#if defined(REPROBE_STAT)
    // some stats.
    upsize_count = std::move(other.upsize_count);
    downsize_count = std::move(other.downsize_count);
    reprobes = std::move(other.reprobes);
    max_reprobes = std::move(other.max_reprobes);
    moves = std::move(other.moves);
    max_moves = std::move(other.max_moves);
    shifts = std::move(other.shifts);
    max_shifts = std::move(other.max_shifts);
#endif
    filter = std::move(other.filter);
    hash = std::move(other.hash);
    eq = std::move(other.eq);
    container = std::move(other.container);
    info_container = std::move(other.info_container);
	}

	void swap(hashmap_robinhood_doubling_offsets && other) {
		std::swap(hll, std::move(other.hll));
		std::swap(lsize, other.lsize);
    	std::swap(buckets, other.buckets);
    	std::swap(mask, other.mask);
		std::swap(min_load, other.min_load);
		std::swap(max_load, other.max_load);
		std::swap(min_load_factor, other.min_load_factor);
		std::swap(max_load_factor, other.max_load_factor);
#if defined(REPROBE_STAT)
    // some stats.
    std::swap(upsize_count, other.upsize_count);
    std::swap(downsize_count, other.downsize_count);
    std::swap(reprobes, other.reprobes);
    std::swap(max_reprobes, other.max_reprobes);
    std::swap(moves, other.moves);
    std::swap(max_moves, other.max_moves);
    std::swap(shifts, other.shifts);
    std::swap(max_shifts, other.max_shifts);
#endif
    std::swap(filter, std::move(other.filter));
    std::swap(hash, std::move(other.hash));
    std::swap(eq, std::move(other.eq));
    std::swap(container, std::move(other.container));
    std::swap(info_container, std::move(other.info_container));
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


//
//  void print() const {
//	  std::cout << "lsize " << lsize << " buckets " << buckets << " max load factor " << max_load_factor << std::endl;
//    for (size_type i = 0; i < info_container.size() - 1; ++i) {
//      std::cout << "i " << i << " key " << container[i].first << " val " <<
//          container[i].second << " info " <<
//          static_cast<size_t>(info_container[i]) << " offset = " << static_cast<size_t>(get_distance(info_container[i])) <<
//          " pos = " << (i + get_distance(info_container[i])) <<
//          " count " << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1));
//      std::cout << std::endl;
//    }
//
//
//  }

	void print() const {
		std::cout << "lsize " << lsize << "\tbuckets " << buckets << "\tmax load factor " << max_load_factor << std::endl;
		size_type i = 0, j = 0;

		container_type tmp;
		size_t offset = 0;
		for (; i < buckets; ++i) {
			std::cout << "buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					std::endl;


			if (! is_empty(info_container[i])) {
				offset = i + get_distance(info_container[i]);
				tmp.clear();
				tmp.insert(tmp.end(), container.begin() + offset,
						container.begin() + i + 1 + get_distance(info_container[i + 1]));
				std::sort(tmp.begin(), tmp.end(), [](typename container_type::value_type const & x,
						typename container_type::value_type const & y){
					return x.first < y.first;
				});
				for (j = 0; j < tmp.size(); ++j) {
					std::cout << std::setw(72) << (offset + j) <<
							", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
							", key: " << std::setw(22) << tmp[j].first <<
							", val: " << std::setw(22) << tmp[j].second <<
							std::endl;
				}
			}
		}

		for (i = buckets; i < info_container.size(); ++i) {
			std::cout << "PAD: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}
	}

	void print_raw() const {
		std::cout << "lsize " << lsize << "\tbuckets " << buckets << "\tmax load factor " << max_load_factor << std::endl;
		size_type i = 0;

		for (i = 0; i < buckets; ++i) {
			std::cout << "buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}

		for (i = buckets; i < info_container.size(); ++i) {
			std::cout << "PAD: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}
	}

	void print_raw(size_t const & first, size_t const &last, std::string prefix) const {
		std::cout << prefix <<
				" lsize " << lsize <<
				"\tbuckets " << buckets <<
				"\tmax load factor " << max_load_factor <<
				"\t printing [" << first << " .. " << last << "]" << std::endl;
		size_type i = 0;

		for (i = first; i <= last; ++i) {
			std::cout << prefix <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}
	}

	void print(size_t const & first, size_t const &last, std::string prefix) const {
		std::cout << prefix <<
				" lsize " << lsize <<
				"\tbuckets " << buckets <<
				"\tmax load factor " << max_load_factor <<
				"\t printing [" << first << " .. " << last << "]" << std::endl;
		size_type i = 0, j = 0;

		container_type tmp;
		size_t offset = 0;
		for (i = first; i <= last; ++i) {
			std::cout << prefix <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_distance(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_distance(info_container[i+1]) - get_distance(info_container[i]) + 1)) <<
					std::endl;


			if (! is_empty(info_container[i])) {
				offset = i + get_distance(info_container[i]);
				tmp.clear();
				tmp.insert(tmp.end(), container.begin() + offset,
						container.begin() + i + 1 + get_distance(info_container[i + 1]));
				std::sort(tmp.begin(), tmp.end(), [](typename container_type::value_type const & x,
						typename container_type::value_type const & y){
					return x.first < y.first;
				});
				for (j = 0; j < tmp.size(); ++j) {
					std::cout << prefix <<
							" " << std::setw(72) << (offset + j) <<
							", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
							", key: " << std::setw(22) << tmp[j].first <<
							", val: " << std::setw(22) << tmp[j].second <<
							std::endl;
				}
			}
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

#if defined(REPROBE_STAT)
    std::cout << "REHASH current " << buckets << " b " << b << " n " << n << " lsize " << lsize << std::endl;
#endif

//    print();

    if ((n != buckets) && (lsize < (max_load_factor * n))) {  // don't resize if lsize is larger than the new max load.

      container_type tmp(n + std::numeric_limits<info_type>::max() + 1);
      info_container_type tmp_info(n + std::numeric_limits<info_type>::max() + 1, info_empty);

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
  void copy_downsize_prefetch(container_type & target, info_container_type & target_info,
                      size_type const & target_buckets) {
    size_type m = target_buckets - 1;
    assert((target_buckets & m) == 0);   // assert this is a power of 2.

    size_t id = 0, bid;
    size_t pos;
    size_t endd;

    //    std::cout << "RESIZE DOWN " << target_buckets << std::endl;

    size_t new_start = 0, new_end = 0;
    size_t blocks = buckets / target_buckets;
    size_t bl;


    //prefetch only if target_buckets is larger than LOOK_AHEAD
    size_t next_info_prefetch_bid;
    size_t next_value_prefetch_bid;
    if (target_buckets > (2 * LOOK_AHEAD)) {
      for (bl = 0; bl < blocks; ++bl) {
        // prefetch 2*LOOK_AHEAD of the info_container.
        for (bid = 0; bid <= (2 * LOOK_AHEAD); bid += info_per_cacheline) {
          _mm_prefetch((const char *)&(info_container[bid + bl * target_buckets]), _MM_HINT_T0);
        }
        next_info_prefetch_bid = bid;
        // for now, also prefetch 2 * LOOK_AHEAD number of entries from the container
        for (bid = 0; bid <= LOOK_AHEAD; bid += value_per_cacheline) {
          _mm_prefetch((const char *)&(container[bid + bl * target_buckets]), _MM_HINT_T0);
        }
        // save position
        next_value_prefetch_bid = bid;
        // rest of the 2*LOOK_AHEAD - this is just to be sure that the region in container between LOOK_AHEAD and info_container[LOOK_AHEAD] (the offset) is covered.
        for (; bid <= (2*LOOK_AHEAD); bid += value_per_cacheline) {
          _mm_prefetch((const char *)&(container[bid + bl * target_buckets]), _MM_HINT_T0);
        }

        // NEED TO Prefetch for write.

      }
    }

    // iterate over all matching blocks.  fill one target bucket at a time and immediately fill the target info.
    info_type prefetched_info;
    for (bid = 0; bid < target_buckets; ++bid) {
      // starting offset is maximum of bid and prev offset.
      new_start = std::max(bid, new_end);
      new_end = new_start;

      // prefetch info_container at 2 LOOK_AHEAD ahead.
      if (next_info_prefetch_bid < target_buckets) {
        for (bl = 0; bl < blocks; ++bl) {
          _mm_prefetch((const char *)&(info_container[next_info_prefetch_bid + bl * target_buckets]), _MM_HINT_T0);
        }
        next_info_prefetch_bid += info_per_cacheline;
      }
      // prefetch container 1 LOOK_AHEAD ahead, and only if info_container indicates content.
      if (next_value_prefetch_bid < target_buckets) {
        for (bl = 0; bl < blocks; ++bl) {
          prefetched_info = info_container[next_value_prefetch_bid + bl * target_buckets];
          if (is_normal(prefetched_info)) _mm_prefetch((const char *)&(container[next_value_prefetch_bid + bl * target_buckets + get_distance(prefetched_info)]), _MM_HINT_T0);
        }
      }


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

  }
  void copy_downsize2(container_type & target, info_container_type & target_info,
                     size_type const & target_buckets) {
    assert((target_buckets & (target_buckets - 1)) == 0);   // assert this is a power of 2.

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
//        std::cout << id << " infos " << static_cast<size_t>(info_container[id]) << "," << static_cast<size_t>(info_container[id + 1]) << ", " <<
//        		" copy from " << pos << " to " << new_end << " length " << (endd - pos) << std::endl;
        memmove(&(target[new_end]), &(container[pos]), sizeof(value_type) * (endd - pos));

        new_end += (endd - pos);

//        if (bid == target_buckets - 1)
//          std::cout << " last: " << bid << " from id " << id << " old " << pos << "-" << endd << " new " << new_start << "-" << new_end << std::endl;
      }
      }

    // offset - current bucket id.
    target_info[bid] = ((new_end - new_start) == 0 ? info_empty : info_normal) + new_start - bid;
//    if (bid == target_buckets - 1)
//      std::cout << " info: " << bid << " from id " << id << " info " << static_cast<size_t>(target_info[bid]) << std::endl;
    }
    // adjust the first padding target info.
//    if ((new_end - new_start) > 0) target_info[target_buckets] = info_empty + new_end - target_buckets;
//    else target_info[target_buckets] = target_info[target_buckets - 1] - 1;

//  std::cout << " info: " << (target_buckets - 1) << " info " << static_cast<size_t>(target_info[target_buckets - 1]) << " entry " << target[target_buckets - 1].first << std::endl;
    // adjust the target_info at the end, in the padding region.
    for (bid = target_buckets; bid < new_end; ++bid) {
      new_start = std::max(bid, new_end);  // fixed new_end.  get new start.
      // if last one is not empty, then first padding position is same distance with
      target_info[bid] = info_empty + new_start - bid;
//    std::cout << " info: " << bid << " info " << static_cast<size_t>(target_info[bid]) << " entry " << target[bid].first << std::endl;
    }

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

    			// count.  at least the bucket id + 1, or last insert target position + 1.
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
#if defined(REPROBE_STAT)
    this->reprobes += reprobe;
    this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
#endif
//				return make_existing_bucket_id(start, offset);
	        return make_existing_bucket_id(start);
			}

#if defined(REPROBE_STAT)
      ++reprobe;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
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
	 * return the next position in "container" that is pointing to self - i.e. offset == 0.
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
		for (; (end < target_info.size()) && !is_normal(target_info[end]); ) {
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
	// old insert_with_hint, searches for empty position, move, then update info
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
#if defined(REPROBE_STAT)
    this->reprobes += reprobe;
    this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
#endif

				  //return make_existing_bucket_id(i, info);
          return make_existing_bucket_id(i);
				}
#if defined(REPROBE_STAT)
			++reprobe;
#endif
			}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
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
//				" end " << static_cast<size_t>(end) <<
//				" buckets " << buckets <<
//				" actual " << target_info.size() << std::endl;
//
		// now compact backwards.  first do the container via MEMMOVE
		// can potentially be optimized to use only swap, if distance is long enough.
		memmove(&(target[next + 1]), &(target[next]), sizeof(value_type) * (end - next));
		// and increment the infos.
		for (size_t i = id + 1; i <= end; ++i) {
			++(target_info[i]);
		}
#if defined(REPROBE_STAT)
		this->shifts += (end - id);
		this->max_shifts = std::max(this->max_shifts, (end - id));
		this->moves += (end - next);
		this->max_moves = std::max(this->max_moves, (end - next));
#endif

		// that's it.
		target[next] = v;
//		return make_missing_bucket_id(next, target_info[id]);
    return make_missing_bucket_id(next);

	}
  inline bucket_id_type insert_with_hint2(container_type & target,
      info_container_type & target_info,
      size_t const & id,
      value_type const & v) {

    assert(id < buckets);

    // get the starting position
    info_type info = target_info[id];

//    std::cout << "info " << static_cast<size_t>(info) << std::endl;
    set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.

    // if this is empty and no shift, then insert and be done.
    if (info == info_empty) {
      target[id] = v;
      return make_missing_bucket_id(id);
//      return make_missing_bucket_id(id, target_info[id]);
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
//          std::cout << "EXISTING.  " << v.first << ", " << target[i].first << std::endl;
#if defined(REPROBE_STAT)
    this->reprobes += reprobe;
    this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
#endif

          //return make_existing_bucket_id(i, info);
          return make_existing_bucket_id(i);
        }
#if defined(REPROBE_STAT)
      ++reprobe;
#endif
      }

#if defined(REPROBE_STAT)
    this->reprobes += reprobe;
    this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
#endif
    }

    // now for the non-empty, or empty with offset, shift and insert, starting from NEXT bucket.


    // first swap in at start of bucket, then deal with the swapped.
    // swap until empty info
    // then insert into last empty,
    // then update info until info_empty

    // scan for the next empty position
    size_t end = find_next_empty_pos(target_info, next);
    size_t i = id+1;
    value_type vv = v;
#if defined(REPROBE_STAT)
    size_t m = 0;
#endif
    for (; i < end; ++i) {
      if (is_normal(target_info[i])) {
        std::swap(target[i + get_distance(target_info[i])], vv);
#if defined(REPROBE_STAT)
        ++m;
#endif
      }

      ++target_info[i];
    }
    target[i] = vv;  // last one.
    ++target_info[i];


#if defined(REPROBE_STAT)
    this->shifts += (end - id);
    this->max_shifts = std::max(this->max_shifts, (end - id));
    this->moves += m;
    this->max_moves = std::max(this->max_moves, m);
#endif

//    return make_missing_bucket_id(next, target_info[id]);
    return make_missing_bucket_id(next);

  }

  inline bucket_id_type insert_with_hint3(container_type & target,
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
#if defined(REPROBE_STAT)
  this->reprobes += reprobe;
  this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
#endif

				  //return make_existing_bucket_id(i, info);
        return make_existing_bucket_id(i);
				}
#if defined(REPROBE_STAT)
			++reprobe;
#endif
			}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
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
		size_t i = id + 1;
		size_t i8 = std::min(end, (i+7) & ~(0x7UL));
		size_t e8 = std::max(i8, end & ~(0x7UL));   // ((end + 1) - 1) & ~(0x8UL)

//		std::cout << "increment for id " << id << " from " << i << " to " << i8 << " to " << e8 << " end " << end << std::endl;

		for (; i < i8; ++i) {
			++(target_info[i]);
		}
		for (uint64_t* ptr; i < e8; i += 8) {   //start i8
			ptr = reinterpret_cast<uint64_t*>(&(target_info[i]));
			*ptr += 0x0101010101010101UL;
//			std::cout << static_cast<size_t>(target_info[i]) << std::endl;
		}
		for (; i <= end; ++i) {   // start e8
			++(target_info[i]);
		}

#if defined(REPROBE_STAT)
		this->shifts += (end - id);
		this->max_shifts = std::max(this->max_shifts, (end - id));
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
		this->shifts += (end - id);
		this->max_shifts = std::max(this->max_shifts, (end - id));
		this->moves += (end - next);
		this->max_moves = std::max(this->max_moves, (end - next));
#endif

		// that's it.
		target[next] = v;
		return;
	}




#if defined(REPROBE_STAT)
	void reset_reprobe_stats() const {
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
		this->shifts = 0;
		this->max_shifts = 0;

	}

	void print_reprobe_stats(std::string const & operation, size_t input_size, size_t success_count) const {
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
	void insert_prefetch(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif
		bucket_id_type id, bid, bid1;

		size_t ii;

		std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

    //prefetch only if target_buckets is larger than LOOK_AHEAD
		size_t max_prefetch2 = std::min(info_container.size(), static_cast<size_t>(2 * LOOK_AHEAD));
    // prefetch 2*LOOK_AHEAD of the info_container.
    for (ii = 0; ii < max_prefetch2; ++ii) {
      hashes[ii] = hash(input[ii].first);
      // prefetch the info_container entry for ii.
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

      // prefetch container as well - would be NEAR but may not be exact.
      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
    }

		// iterate based on size between rehashes
    constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;
		for (size_t i = 0, i1 = LOOK_AHEAD, i2 = 2*LOOK_AHEAD; i < input.size(); ++i, ++i1, ++i2) {

			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);  // TODO: SHOULD PREFETCH AGAIN

			// first get the bucket id
			id = hashes[i & hash_mask] & mask;  // target bucket id.

			// prefetch info_container.
			if (i2 < input.size()) {
      ii = i2 & hash_mask;
      hashes[ii] = hash(input[i2].first);
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
        }
      // prefetch container
      if (i1 < input.size()) {
      bid = hashes[i1 & hash_mask] & mask;
      if (is_normal(info_container[bid])) {
        bid1 = bid + 1 + get_distance(info_container[bid + 1]);
        bid += get_distance(info_container[bid]);

          for (size_t j = bid; j < bid1; ++j) {
          _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
        }
      }
    }

      if (missing(insert_with_hint(container, info_container, id, input[i])))
        ++lsize;

//      std::cout << "insert vec lsize " << lsize << std::endl;

    }


#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT VEC", input.size(), (lsize - before));
#endif

		// NOT needed until we are estimating reservation size.
//		reserve(lsize);  // resize down as needed
	}




	/// batch insert.  integrated code in insert_with_hint
	void insert_integrated(::std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif
		bucket_id_type id;

		value_type v;
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
			this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
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
			this->shifts += (end - id);
			this->max_shifts = std::max(this->max_shifts, (end - id));
			this->moves += (end - next);
			this->max_moves = std::max(this->max_moves, (end - next));
	#endif


		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT INTEGRATED", input.size(), (lsize - before));
#endif

	}


	/// batch insert 2.  try to avoid too many extra mem moves by allocating if needed.  can't do this until have some grouping because we can't determine duplicates directly
	void insert_integrated2(::std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
		size_type before = lsize;
#endif


		// ----------- first generate a hash index array.  linear mem access.

		// if existing is empty, insert 1 entry each first, then do the following.


		// ----------- compute insertion count (only previously non-existent entries)  (at most 127 per bucket)
		// ---------------- compare within buckets.  for update, do it here.  for update insert, update here too


		// ----------- if need to resize, resize, and recompute insertion count.


		// ----------- create updated info array  (reuse insertion count?)


		// ----------- use orig info array and new info array to memmove each bucket, back to front so can do it inplace.


		// done...

		bucket_id_type id;

		value_type v;
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
			this->max_reprobes = std::max(this->max_reprobes, static_cast<info_type>(reprobe));
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
			this->shifts += (end - id);
			this->max_shifts = std::max(this->max_shifts, (end - id));
			this->moves += (end - next);
			this->max_moves = std::max(this->max_moves, (end - next));
	#endif


		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT INTEGRATED", input.size(), (lsize - before));
#endif

	}


	/// batch insert, minimizing number of loop conditionals and rehash checks.
	template <typename LESS = ::std::less<key_type> >
	void insert(::std::vector<value_type> const & input) {


		//    throw ::std::logic_error("ERROR: DISABLED FOR NONCIRC VERSION");
//	  bucket_id_type info_align = reinterpret_cast<size_t>(info_container.data()) % 64;  // cacheline size = 64


		#if defined(REPROBE_STAT)
      if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
        std::cout << "WARNING: container alignment not on value boundary" << std::endl;
      } else {
        std::cout << "STATUS: container alignment on value boundary" << std::endl;
      }
		    reset_reprobe_stats();
		    size_type before = lsize;
		#endif
		    bucket_id_type id, bid1, bid;

		    size_t ii;
		    size_t hash_val;

		    std::array<size_t, 2 * LOOK_AHEAD>  hashes;

		    //prefetch only if target_buckets is larger than LOOK_AHEAD
		    size_t max_prefetch2 = std::min(info_container.size(), static_cast<size_t>(2 * LOOK_AHEAD));
		    // prefetch 2*LOOK_AHEAD of the info_container.
		    for (ii = 0; ii < max_prefetch2; ++ii) {
		      hash_val = hash(input[ii].first);
		      hashes[ii] = hash_val;
		      id = hash_val & mask;
		      // prefetch the info_container entry for ii.
		      _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
//		      if (((id + 1) % 64) == info_align)
//		        _mm_prefetch((const char *)&(info_container[id + 1]), _MM_HINT_T0);

		      // prefetch container as well - would be NEAR but may not be exact.
		      _mm_prefetch((const char *)&(container[id]), _MM_HINT_T0);

		    }

		    // iterate based on size between rehashes
		    constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;
		    size_t max2 = (input.size() > (2*LOOK_AHEAD)) ? input.size() - (2*LOOK_AHEAD) : 0;
		    size_t max1 = (input.size() > LOOK_AHEAD) ? input.size() - LOOK_AHEAD : 0;
		    size_t i = 0; //, i1 = LOOK_AHEAD, i2 = 2*LOOK_AHEAD;

		    size_t to_insert = max2 - i, lmax;

		    while (to_insert > 0) {

#if defined(REPROBE_STAT)
		      std::cout << "checking if rehash needed.  i = " << i << std::endl;
#endif

          // first check if we need to resize.
		      if (lsize >= max_load) {
		        rehash(buckets << 1);

#if defined(REPROBE_STAT)
            std::cout << "rehashed.  size = " << buckets << std::endl;
		        if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type))  > 0) {
		          std::cout << "WARNING: container alignment not on value boundary" << std::endl;
		        } else {
		          std::cout << "STATUS: container alignment on value boundary" << std::endl;
		        }
#endif


		      }

	        to_insert = max2 - i;
	        lmax = i + std::min(max_load - lsize, to_insert);


          for (; i < lmax; ++i) {
            _mm_prefetch((const char *)&(hashes[(i + 2 * LOOK_AHEAD) & hash_mask]), _MM_HINT_T0);
            // prefetch input
            _mm_prefetch((const char *)&(input[i + 2 * LOOK_AHEAD]), _MM_HINT_T0);


            // prefetch container
            bid = hashes[(i + LOOK_AHEAD) & hash_mask] & mask;
            if (is_normal(info_container[bid])) {
              bid1 = bid + 1;
              bid += get_distance(info_container[bid]);
              bid1 += get_distance(info_container[bid1]);

              for (size_t j = bid; j < bid1; ++j) {
                _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
              }
            }

            // first get the bucket id
            id = hashes[i & hash_mask] & mask;  // target bucket id.
            if (missing(insert_with_hint(container, info_container, id, input[i])))
              ++lsize;

      //      std::cout << "insert vec lsize " << lsize << std::endl;
            // prefetch info_container.
            hash_val = hash(input[(i + 2 * LOOK_AHEAD)].first);
            bid = hash_val & mask;
            _mm_prefetch((const char *)&(info_container[bid]), _MM_HINT_T0);
//            if (((bid + 1) % 64) == info_align)
//              _mm_prefetch((const char *)&(info_container[bid + 1]), _MM_HINT_T0);

            hashes[(i + 2 * LOOK_AHEAD)  & hash_mask] = hash_val;
          }

		    }
        //if ((lsize + 2 * LOOK_AHEAD) >= max_load) rehash(buckets << 1);  // TODO: SHOULD PREFETCH AGAIN


		    // second to last LOOK_AHEAD
		    for (; i < max1; ++i) {


		      // === same code as in insert(1)..

		      // first check if we need to resize.
		      if (lsize >= max_load) rehash(buckets << 1);  // TODO: SHOULD PREFETCH AGAIN

		      bid = hashes[(i + LOOK_AHEAD) & hash_mask] & mask;

		      // first get the bucket id
		      id = hashes[i & hash_mask] & mask;  // target bucket id.

		      // prefetch container
		      if (is_normal(info_container[bid])) {
            bid1 = bid + 1;
            bid += get_distance(info_container[bid]);
            bid1 += get_distance(info_container[bid1]);

		        for (size_t j = bid; j < bid1; ++j) {
		          _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
		        }
		      }

		      if (missing(insert_with_hint(container, info_container, id, input[i])))
		        ++lsize;

		//      std::cout << "insert vec lsize " << lsize << std::endl;

		    }


		    // last LOOK_AHEAD
		    for (; i < input.size(); ++i) {

		      // === same code as in insert(1)..

		      // first check if we need to resize.
		      if (lsize >= max_load) rehash(buckets << 1);  // TODO: SHOULD PREFETCH AGAIN

		      // first get the bucket id
		      id = hashes[i & hash_mask] & mask;  // target bucket id.

		      if (missing(insert_with_hint(container, info_container, id, input[i])))
		        ++lsize;

		//      std::cout << "insert vec lsize " << lsize << std::endl;

		    }


		#if defined(REPROBE_STAT)
		    print_reprobe_stats("INSERT VEC", input.size(), (lsize - before));
		#endif

	}

	template <typename LESS = ::std::less<key_type> >
	void insert_sort(::std::vector<value_type> const & input) {
		insert(input);

	}

	// batch insert, minimizing number of loop conditionals and rehash checks.
	// provide a set of precomputed hash values, contains the bucket id to go into.
	// because this hash value is fixed by number of buckets, this function does not allow resize.
	template <typename LESS = ::std::less<key_type> >
	void insert_with_hint_no_resize(value_type const * const input, size_t const * const bids, size_t input_size) {


		#if defined(REPROBE_STAT)
		  if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
		  } else {
			std::cout << "STATUS: container alignment on value boundary" << std::endl;
		  }
		    reset_reprobe_stats();
		    size_type before = lsize;
		#endif
		    bucket_id_type id, bid1, bid;

		    size_t ii;

		    //prefetch only if target_buckets is larger than LOOK_AHEAD
		    size_t max_prefetch = std::min(input_size, static_cast<size_t>(2 * LOOK_AHEAD));
		    // prefetch 2*LOOK_AHEAD of the info_container.
		    for (ii = 0; ii < max_prefetch; ++ii) {
	            _mm_prefetch((const char *)&(bids[ii]), _MM_HINT_T0);
	            // prefetch input
	            _mm_prefetch((const char *)&(input[ii]), _MM_HINT_T0);

		    }

		    for (ii = 0; ii < max_prefetch; ++ii) {

	            id = bids[ii];
		      // prefetch the info_container entry for ii.
		      _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
//		      if (((id + 1) % 64) == info_align)
//		        _mm_prefetch((const char *)&(info_container[id + 1]), _MM_HINT_T0);

		      // prefetch container as well - would be NEAR but may not be exact.
		      _mm_prefetch((const char *)&(container[id]), _MM_HINT_T0);

		    }


		    // iterate based on size between rehashes
		    size_t max2 = (input_size > (2*LOOK_AHEAD)) ? input_size - (2*LOOK_AHEAD) : 0;
		    size_t max1 = (input_size > LOOK_AHEAD) ? input_size - LOOK_AHEAD : 0;
		    size_t i = 0; //, i1 = LOOK_AHEAD, i2 = 2*LOOK_AHEAD;

          for (; i < max2; ++i) {
            _mm_prefetch((const char *)&(bids[(i + 2 * LOOK_AHEAD)]), _MM_HINT_T0);
            // prefetch input
            _mm_prefetch((const char *)&(input[i + 2 * LOOK_AHEAD]), _MM_HINT_T0);


            // prefetch container
            bid = bids[(i + LOOK_AHEAD)];
            if (is_normal(info_container[bid])) {
              bid1 = bid + 1;
              bid += get_distance(info_container[bid]);
              bid1 += get_distance(info_container[bid1]);

              for (size_t j = bid; j < bid1; ++j) {
                _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
              }
            }

            // first get the bucket id
            if (missing(insert_with_hint(container, info_container, bids[i], input[i])))
              ++lsize;

      //      std::cout << "insert vec lsize " << lsize << std::endl;
            // prefetch info_container.
            bid = bids[(i + 2 * LOOK_AHEAD)];
            _mm_prefetch((const char *)&(info_container[bid]), _MM_HINT_T0);
//            if (((bid + 1) % 64) == info_align)
//              _mm_prefetch((const char *)&(info_container[bid + 1]), _MM_HINT_T0);

          }


        //if ((lsize + 2 * LOOK_AHEAD) >= max_load) rehash(buckets << 1);  // TODO: SHOULD PREFETCH AGAIN


		    // second to last LOOK_AHEAD
		    for (; i < max1; ++i) {


		      // === same code as in insert(1)..

		      bid = bids[(i + LOOK_AHEAD)];


		      // prefetch container
		      if (is_normal(info_container[bid])) {
            bid1 = bid + 1;
            bid += get_distance(info_container[bid]);
            bid1 += get_distance(info_container[bid1]);

		        for (size_t j = bid; j < bid1; ++j) {
		          _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
		        }
		      }

		      if (missing(insert_with_hint(container, info_container, bids[i], input[i])))
		        ++lsize;

		//      std::cout << "insert vec lsize " << lsize << std::endl;


		    }


		    // last LOOK_AHEAD
		    for (; i < input_size; ++i) {

		      // === same code as in insert(1)..

		      if (missing(insert_with_hint(container, info_container, bids[i], input[i])))
		        ++lsize;

		//      std::cout << "insert vec lsize " << lsize << std::endl;

		    }


		#if defined(REPROBE_STAT)
		    print_reprobe_stats("INSERT VEC", input_size, (lsize - before));
		#endif

	}



  void insert_shuffled(::std::vector<value_type> const & input) {


    #if defined(REPROBE_STAT)
	    if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
	      std::cout << "WARNING: container alignment not on value boundary" << std::endl;
	    } else {
	      std::cout << "STATUS: container alignment on value boundary" << std::endl;
	    }

	    reset_reprobe_stats();
        size_type before = lsize;
    #endif

        // compute hash value array, and estimate the number of unique entries in current.  then merge with current and get the after count.
        //std::vector<size_t> hash_vals;
        //hash_vals.reserve(input.size());
        size_t* hash_vals;
        int ret = posix_memalign(reinterpret_cast<void **>(&hash_vals), 16, sizeof(size_t) * input.size());
		if (ret)
			throw std::length_error("failed to allocate aligned memory");


        hyperloglog64<key_type, hasher, 12> hll_local;

        size_t hval;
        for (size_t i = 0; i < input.size(); ++i) {
        	hval = hash(input[i].first);
        	hll_local.update_via_hashval(hval);
        	// using mm_stream here does not make a differnece.
        	//_mm_stream_si64(reinterpret_cast<long long int*>(hash_vals + i), *(reinterpret_cast<long long int*>(&hval)));
        	hash_vals[i] = hval;
        }

        // estimate the number of unique entries in input.
        double distinct_input_est = hll_local.estimate();

        hll_local.merge(hll);
        double distinct_total_est = hll_local.estimate();

        std::cout << " estimate input cardinality as " << distinct_input_est << " total after insertion " << distinct_total_est << std::endl;

        // assume one element per bucket as ideal, resize now.  should not resize if don't need to.
        reserve(distinct_total_est);   // this updates the bucket counts also.

#if 0
   // not correct...
        // ---- shuffle data.  Later separate this into a different function, 2 versions, one for insert and one for all others.
        // want each 64 consecutive buckets's input to be together.
        // first compute the input bucket counts. each bucket has at most 127 entries, times 64, so 13 bits.
        std::vector<uint16_t> shuffle_bucket_counts((buckets >> 6) + 1);
        // random access to compute shuffle buccket counts
        size_t i = 0;
        size_t shuffle_max1 = std::min(input.size(), static_cast<size_t>(LOOK_AHEAD));
        for (; i < shuffle_max1; ++i) {
        	hash_vals[i] &= mask;  // get the final target bucket id and save it.
        	_mm_prefetch((const char *)&(shuffle_bucket_counts[((hash_vals[i] >> 6) + 1)]), _MM_HINT_T0);
        }
        size_t shuffle_max2 = std::max(input.size(), static_cast<size_t>(LOOK_AHEAD)) - LOOK_AHEAD;
        for (i = 0; i < shuffle_max2; ++i) {
        	++shuffle_bucket_counts[((hash_vals[i] >> 6) + 1)];

        	hash_vals[i + LOOK_AHEAD] &= mask;
        	_mm_prefetch((const char *)&(shuffle_bucket_counts[((hash_vals[i + LOOK_AHEAD] >> 6) + 1)]), _MM_HINT_T0);
        }
		for (; i < input.size(); ++i) {
			++shuffle_bucket_counts[((hash_vals[i] >> 6) + 1)];
        }
        // convert to excl prefix sum. sequential
		for (i = 1; i < shuffle_bucket_counts.size(); ++i) {
			shuffle_bucket_counts[i] += shuffle_bucket_counts[i-1];
		}
		// now perform actual shuffle.  random access of shuffle_bucket_counts, sh_input[pos], and sh_hash_vals[pos]
        value_type* sh_input;
        ret = posix_memalign(reinterpret_cast<void **>(&sh_input), 16, sizeof(value_type) * input.size());
		if (ret)
			throw std::length_error("failed to allocate aligned memory");

        size_t* sh_hash_val;
        ret = posix_memalign(reinterpret_cast<void **>(&sh_hash_val), 16, sizeof(size_t) * input.size());
		if (ret)
			throw std::length_error("failed to allocate aligned memory");
//		std::vector<value_type> sh_input(input.size());
//		std::vector<size_t> sh_hash_val(input.size());
		size_t pos;

		shuffle_max1 = std::min(input.size(), static_cast<size_t>(LOOK_AHEAD));
		for (i = 0; i < shuffle_max1; ++i) {
			_mm_prefetch((const char *)&(shuffle_bucket_counts[(hash_vals[i] >> 6)]), _MM_HINT_T0);
		}
//		for (i = 0; i < shuffle_max1; ++i) {
//			_mm_prefetch((const char *)&(shuffle_bucket_counts[(hash_vals[i + LOOK_AHEAD] >> 6)]), _MM_HINT_T0);
//			// hash_vals have the bucket id for each input
//			pos = shuffle_bucket_counts[hash_vals[i] >> 6];
//			_mm_prefetch((const char *)&(sh_input[pos]), _MM_HINT_T0);
//			_mm_prefetch((const char *)&(sh_hash_val[pos]), _MM_HINT_T0);
//		}
//		shuffle_max1 = std::max(input.size(), static_cast<size_t>(LOOK_AHEAD)) - LOOK_AHEAD;  // for mm_stream only
//		for (i = 0; i < shuffle_max1; ++i) {
//			_mm_prefetch((const char *)&(shuffle_bucket_counts[(hash_vals[i + LOOK_AHEAD] >> 6)]), _MM_HINT_T0);
//			// hash_vals have the bucket id for each input
////			pos = shuffle_bucket_counts[hash_vals[i + LOOK_AHEAD] >> 6];
////			_mm_prefetch((const char *)&(sh_input[pos]), _MM_HINT_T0);
////			_mm_prefetch((const char *)&(sh_hash_val[pos]), _MM_HINT_T0);
//
//
//			// hash_vals have the bucket id for each input
//			pos = shuffle_bucket_counts[hash_vals[i] >> 6]++;  // random
////			sh_input[pos] = input[i];  // random write, seq rea.
////			sh_hash_val[pos] = hash_vals[i];
//			_mm_stream_si128(reinterpret_cast<__m128i*>(sh_input + pos), *(reinterpret_cast<const __m128i*>(input.data() + i)));
//			_mm_stream_si64(reinterpret_cast<long long int*>(sh_hash_val + pos), *(reinterpret_cast<long long int*>(hash_vals + i)) & mask);
//		}
//		shuffle_max1 = std::max(input.size(), static_cast<size_t>(LOOK_AHEAD)) - LOOK_AHEAD;
//		for (; i < shuffle_max1; ++i) {
//			// hash_vals have the bucket id for each input
//			pos = shuffle_bucket_counts[hash_vals[i + LOOK_AHEAD] >> 6];
//			_mm_prefetch((const char *)&(sh_input[pos]), _MM_HINT_T0);
//			_mm_prefetch((const char *)&(sh_hash_val[pos]), _MM_HINT_T0);
//
//			// hash_vals have the bucket id for each input
//			pos = shuffle_bucket_counts[hash_vals[i] >> 6]++;  // random
//			sh_input[pos] = input[i];  // random write, seq rea.
//			sh_hash_val[pos] = hash_vals[i];
//		}
		for (; i < input.size(); ++i) {
			// hash_vals have the bucket id for each input
			pos = shuffle_bucket_counts[hash_vals[i] >> 6]++;  // random
//			sh_input[pos] = input[i];  // random write, seq rea.
//			sh_hash_val[pos] = hash_vals[i];
			_mm_stream_si128(reinterpret_cast<__m128i*>(sh_input + pos), *(reinterpret_cast<const __m128i*>(input.data() + i)));
			_mm_stream_si64(reinterpret_cast<long long int*>(sh_hash_val + pos), *(reinterpret_cast<long long int*>(hash_vals + i)) & mask);
		}

		// now try to insert.  hashing done already.
		insert_with_hint_no_resize(sh_input, sh_hash_val, input.size());
#else

		for (size_t i = 0; i < input.size(); ++i) {
			hash_vals[i] &= mask;
		}
		insert_with_hint_no_resize(input.data(), hash_vals, input.size());
#endif

//        for (size_t i = 0; i < input.size(); ++i) {
//        	hash_vals[i] &= mask;
//        }
//        insert_with_hint_no_resize(input.data(), hash_vals.data(), input.size());
//        insert_with_hint_no_resize(sh_input.data(), sh_hash_val.data(), sh_input.size());
        // finally, update the hyperloglog estimator.  just swap.
        hll.swap(std::move(hll_local));
        free(hash_vals);

    #if defined(REPROBE_STAT)
        print_reprobe_stats("INSERT VEC", input.size(), (lsize - before));
    #endif

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

    std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

    //prefetch only if target_buckets is larger than LOOK_AHEAD
    size_t ii = 0;
    // prefetch 2*LOOK_AHEAD of the info_container.
    for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
      hashes[ii] = hash((*it).first);
      // prefetch the info_container entry for ii.
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

      // prefetch container as well - would be NEAR but may not be exact.
      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
    }

    size_t total = std::distance(begin, end);
		::std::vector<size_type> counts;
		counts.reserve(total);

		size_t id, bid, bid1;
		bucket_id_type found;
		Iter it2 = begin;
		std::advance(it2, 2 * LOOK_AHEAD);
		    size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
		    constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

		for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

      // first get the bucket id
      id = hashes[i & hash_mask] & mask;  // target bucket id.

      // prefetch info_container.
      if (i2 < total) {
        ii = i2 & hash_mask;
        hashes[ii] = hash((*it2).first);
        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
      }
      // prefetch container
      if (i1 < total) {
        bid = hashes[i1 & hash_mask] & mask;
        if (is_normal(info_container[bid])) {
          bid1 = bid + 1 + get_distance(info_container[bid + 1]);
          bid += get_distance(info_container[bid]);

          for (size_t j = bid; j < bid1; ++j) {
            _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
          }
        }
      }

			found = find_pos_with_hint((*it).first, id);

			counts.emplace_back(exists(found));
		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT ITER PAIR", std::distance(begin, end), counts);
#endif
		return counts;
	}


	template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif
    std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

    //prefetch only if target_buckets is larger than LOOK_AHEAD
    size_t ii = 0;
    // prefetch 2*LOOK_AHEAD of the info_container.
    for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
      hashes[ii] = hash(*it);
      // prefetch the info_container entry for ii.
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

      // prefetch container as well - would be NEAR but may not be exact.
      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
    }


    size_t total = std::distance(begin, end);
    ::std::vector<size_type> counts;
    counts.reserve(total);

    size_t id, bid, bid1;
    bucket_id_type found;
    Iter it2 = begin;
    std::advance(it2, 2 * LOOK_AHEAD);
        size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
        constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

    for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

      // first get the bucket id
      id = hashes[i & hash_mask] & mask;  // target bucket id.

      // prefetch info_container.
      if (i2 < total) {
        ii = i2 & hash_mask;
        hashes[ii] = hash(*it2);
        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
      }
      // prefetch container
      if (i1 < total) {
        bid = hashes[i1 & hash_mask] & mask;
        if (is_normal(info_container[bid])) {
          bid1 = bid + 1 + get_distance(info_container[bid + 1]);
          bid += get_distance(info_container[bid]);

          for (size_t j = bid; j < bid1; ++j) {
            _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
          }
        }
      }

      found = find_pos_with_hint(*it, id);

      counts.emplace_back(exists(found));
    }


#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT ITER KEY", std::distance(begin, end), counts.size());
#endif
		return counts;
	}


	/**
	 * @brief find the iterator for a key
	 */
	iterator find(key_type const & k) {
#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif

		bucket_id_type idx = find_pos(k);

#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND 1 KEY", 1, ( exists(idx) ? 1: 0));
#endif

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
#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif

		bucket_id_type idx = find_pos(k);

#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND 1 KEY", 1, ( exists(idx) ? 1: 0));
#endif

		if (exists(idx))
      return const_iterator(container.cbegin() + get_pos(idx), info_container.cbegin()+ get_pos(idx),
          info_container.cend(), filter);
		else
      return const_iterator(container.cend(), info_container.cend(), filter);


	}

  template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
  std::vector<value_type> find(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif

    std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

    //prefetch only if target_buckets is larger than LOOK_AHEAD
    size_t ii = 0;
    // prefetch 2*LOOK_AHEAD of the info_container.
    for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
      hashes[ii] = hash((*it).first);
      // prefetch the info_container entry for ii.
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

      // prefetch container as well - would be NEAR but may not be exact.
      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
    }


    size_t total = std::distance(begin, end);
    ::std::vector<value_type> counts;
    counts.reserve(total);

    size_t id, bid, bid1;
    bucket_id_type found;
    Iter it2 = begin;
    std::advance(it2, 2 * LOOK_AHEAD);
        size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
        constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

    for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

      // first get the bucket id
      id = hashes[i & hash_mask] & mask;  // target bucket id.

      // prefetch info_container.
      if (i2 < total) {
        ii = i2 & hash_mask;
        hashes[ii] = hash((*it2).first);
        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
      }
      // prefetch container
      if (i1 < total) {
        bid = hashes[i1 & hash_mask] & mask;
        if (is_normal(info_container[bid])) {
          bid1 = bid + 1 + get_distance(info_container[bid + 1]);
          bid += get_distance(info_container[bid]);

          for (size_t j = bid; j < bid1; ++j) {
            _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
          }
        }
      }

      found = find_pos_with_hint((*it).first, id);

      counts.emplace_back(container[get_pos(found)]);
    }

#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND ITER PAIR", std::distance(begin, end), counts.size());
#endif
    return counts;
  }


  template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
  std::vector<value_type> find(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif
    std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

    //prefetch only if target_buckets is larger than LOOK_AHEAD
    size_t ii = 0;
    // prefetch 2*LOOK_AHEAD of the info_container.
    for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
      hashes[ii] = hash(*it);
      // prefetch the info_container entry for ii.
      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

      // prefetch container as well - would be NEAR but may not be exact.
      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
    }


    size_t total = std::distance(begin, end);
    ::std::vector<value_type> counts;
    counts.reserve(total);

    size_t id, bid, bid1;
    bucket_id_type found;
    Iter it2 = begin;
    std::advance(it2, 2 * LOOK_AHEAD);
        size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
        constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

    for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

      // first get the bucket id
      id = hashes[i & hash_mask] & mask;  // target bucket id.

      // prefetch info_container.
      if (i2 < total) {
        ii = i2 & hash_mask;
        hashes[ii] = hash(*it2);
        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
      }
      // prefetch container
      if (i1 < total) {
        bid = hashes[i1 & hash_mask] & mask;
        if (is_normal(info_container[bid])) {
          bid1 = bid + 1 + get_distance(info_container[bid + 1]);
          bid += get_distance(info_container[bid]);

          for (size_t j = bid; j < bid1; ++j) {
            _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
          }
        }
      }

      found = find_pos_with_hint(*it, id);

      counts.emplace_back(container[get_pos(found)]);
    }


#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND ITER PAIR", std::distance(begin, end), counts.size());
#endif
    return counts;
  }



	/**
	 * @brief.  updates current value.  behaves like insert, but overwrites the exists.
	 */
  template <typename Reducer>
	iterator update(key_type const & k, mapped_type const & val, Reducer const & r) {
		// find and update.  if not present, insert.
		std::pair<iterator, bool> result = insert(k, val);

		if (! result.second) {  // not inserted and no exception, so an equal entry has been found.
			result.first->second = r(result.first->second, val);   // so update.

		} // else inserted. so updated.

		return result.first;
	}
  template <typename Reducer>
	iterator update(value_type const & vv, Reducer const & r) {
		// find and update.  if not present, insert.
		std::pair<iterator, bool> result = insert(vv);

		if (! result.second) {  // not inserted and no exception, so an equal entry has been found.
			result.first->second = r(result.first->second, vv.second);   // so update.

		} // else inserted. so updated.

		return result.first;
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
	size_type erase_and_compact(key_type const & k, bucket_id_type const & bid) {
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

#if defined(REPROBE_STAT)
    this->shifts += (end - bid1);
    this->max_shifts = std::max(this->max_shifts, (end - bid1));
    this->moves += (end - pos1);
    this->max_moves = std::max(this->max_moves, (end - pos1));
#endif

		return 1;

	}

	//============ ERASE


	public:

		/// single element erase with key.
		size_type erase_no_resize(key_type const & k) {
	#if defined(REPROBE_STAT)
			reset_reprobe_stats();
	#endif
	    size_t bid = hash(k) & mask;

			size_t erased = erase_and_compact(k, bid);

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

	    std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

	    //prefetch only if target_buckets is larger than LOOK_AHEAD
	    size_t ii = 0;
	    // prefetch 2*LOOK_AHEAD of the info_container.
	    for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
	      hashes[ii] = hash((*it).first);
	      // prefetch the info_container entry for ii.
	      _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

	      // prefetch container as well - would be NEAR but may not be exact.
	      _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
	    }

	    size_t total = std::distance(begin, end);

	    size_t id, bid, bid1;
	    Iter it2 = begin;
	    std::advance(it2, 2 * LOOK_AHEAD);
	        size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
	        constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

	    for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

	      // first get the bucket id
	      id = hashes[i & hash_mask] & mask;  // target bucket id.

	      // prefetch info_container.
	      if (i2 < total) {
	        ii = i2 & hash_mask;
	        hashes[ii] = hash((*it2).first);
	        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
	      }
	      // prefetch container
	      if (i1 < total) {
	        bid = hashes[i1 & hash_mask] & mask;
	        if (is_normal(info_container[bid])) {
	          bid1 = bid + 1 + get_distance(info_container[bid + 1]);
	          bid += get_distance(info_container[bid]);

	          for (size_t j = bid; j < bid1; ++j) {
	            _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
	          }
	        }
	      }

	      erase_and_compact((*it).first, id);
	    }


	#if defined(REPROBE_STAT)
			print_reprobe_stats("ERASE ITER PAIR", std::distance(begin, end), before - lsize);
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

      std::vector<size_t>  hashes(2 * LOOK_AHEAD, 0);

      //prefetch only if target_buckets is larger than LOOK_AHEAD
      size_t ii = 0;
      // prefetch 2*LOOK_AHEAD of the info_container.
      for (Iter it = begin; (ii < (2* LOOK_AHEAD)) && (it != end); ++it, ++ii) {
        hashes[ii] = hash(*it);
        // prefetch the info_container entry for ii.
        _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);

        // prefetch container as well - would be NEAR but may not be exact.
        _mm_prefetch((const char *)&(container[hashes[ii] & mask]), _MM_HINT_T0);
      }

      size_t total = std::distance(begin, end);

      size_t id, bid, bid1;
      Iter it2 = begin;
      std::advance(it2, 2 * LOOK_AHEAD);
      size_t i = 0, i1 = LOOK_AHEAD, i2=2 * LOOK_AHEAD;
      constexpr size_t hash_mask = 2 * LOOK_AHEAD - 1;

      for (auto it = begin; it != end; ++it, ++it2, ++i, ++i1, ++i2) {

        // first get the bucket id
        id = hashes[i & hash_mask] & mask;  // target bucket id.

        // prefetch info_container.
        if (i2 < total) {
          ii = i2 & hash_mask;
          hashes[ii] = hash(*it2);
          _mm_prefetch((const char *)&(info_container[hashes[ii] & mask]), _MM_HINT_T0);
        }
        // prefetch container
        if (i1 < total) {
          bid = hashes[i1 & hash_mask] & mask;
          if (is_normal(info_container[bid])) {
            bid1 = bid + 1 + get_distance(info_container[bid + 1]);
            bid += get_distance(info_container[bid]);

            for (size_t j = bid; j < bid1; ++j) {
              _mm_prefetch((const char *)&(container[j]), _MM_HINT_T0);
            }
          }
        }

        erase_and_compact(*it, id);
      }


	#if defined(REPROBE_STAT)
			print_reprobe_stats("ERASE ITER KEY", std::distance(begin, end), before - lsize);
	#endif
			return before - lsize;
		}

		/**
		 * @brief erases a key.
		 */
		size_type erase(key_type const & k) {

			size_type res = erase_no_resize(k);

			if (lsize < min_load) rehash(buckets >> 1);

			return res;
		}

		template <typename Iter>
		size_type erase(Iter begin, Iter end) {

			size_type erased = erase_no_resize(begin, end);

			if (lsize < min_load) reserve(lsize);

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
constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_pos_mask;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_pos_exists;
//template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
//constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_info_mask;
//template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
//constexpr typename hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::bid_info_empty;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr uint32_t hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_per_cacheline;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr uint32_t hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::value_per_cacheline;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr uint32_t hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::info_prefetch_iters;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr uint32_t hashmap_robinhood_doubling_offsets<Key, T, Hash, Equal, Allocator>::value_prefetch_iters;


}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_OFFSETS_PREFETCH_HPP_ */
