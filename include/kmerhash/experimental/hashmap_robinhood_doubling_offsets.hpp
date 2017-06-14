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

#ifndef KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS_HPP_
#define KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"

namespace fsc {

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
    struct info_type {
    	info_type(unsigned char val) : info(val) {}

    	unsigned char info;

    	static constexpr unsigned char empty = 0x80;
    	static constexpr unsigned char dist_mask = 0x7F;
    	static constexpr unsigned char normal = 0x00;

    	inline bool is_empty() const {
    		return info >= empty;  // empty 0x80
    	}
    	inline bool is_normal() const {
    		return info < empty;  // normal. both top bits are set. 0xC0
    	}
    	inline void set_empty() {
    		info |= empty;  // nothing here.
    	}
    	inline void set_normal() {
    		info &= dist_mask;  // nothing here.
    	}

    	inline unsigned char get_offset() const {
    		return info & dist_mask;  // nothing here.
    	}

    	inline void operator++() {
    		++info;
    	}
    	inline void operator--() {
    		if (info & dist_mask) --info;
    		else info = empty;  // if already empty, no change.  if normal (1 entry), then now empty.
    	}

    };


    using container_type		= ::std::vector<value_type, Allocator>;
    using info_container_type	= ::std::vector<info_type, Allocator>;

    // filter
    struct empty_deleted_filter {
    	bool operator()(info_type const & x) {   // a container entry is empty only if the corresponding info is empty (0x80), not just have empty flag set.
    		return x.info != info_type::empty;   // (curr bucket is empty and position is also empty.  otherwise curr bucket is here or prev bucket is occupying this)
    	};
    };

public:

    using allocator_type        = typename container_type::allocator_type;
    using reference 			= typename container_type::reference;
    using const_reference	    = typename container_type::const_reference;
    using pointer				= typename container_type::pointer;
    using const_pointer		    = typename container_type::const_pointer;
    using iterator              = ::bliss::iterator::aux_filter_iterator<typename container_type::iterator, typename info_container_type::iterator, empty_deleted_filter>;
    using const_iterator        = ::bliss::iterator::aux_filter_iterator<typename container_type::const_iterator, typename info_container_type::const_iterator, empty_deleted_filter>;
    using size_type             = typename container_type::size_type;
    using difference_type       = typename container_type::difference_type;


protected:

    size_t lsize;
    mutable size_t buckets;
    mutable size_t mask;
    mutable size_t min_load;
    mutable size_t max_load;
    mutable float min_load_factor;
    mutable float max_load_factor;


    empty_deleted_filter filter;
    hasher hash;
    key_equal eq;

    container_type container;
    info_container_type info_container;

    // some stats.
    size_t upsize_count;
    size_t downsize_count;
    mutable size_t reprobes;   // for use as temp variable
    mutable size_t max_reprobes;

public:

    /**
     * _capacity is the number of usable entries, not the capacity of the underlying container.
     */
	explicit hashmap_robinhood_doubling_offsets(size_t const & _capacity = 128,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
			lsize(0), buckets(next_power_of_2(_capacity)), mask(buckets - 1),
			container(buckets), info_container(buckets, info_type(info_type::empty)),
			upsize_count(0), downsize_count(0)
			{
		// get the nearest power of 2 above specified capacity.
		// buckets = next_power_of_2(_capacity);

//		container.resize(buckets);
//		info_container.resize(buckets, 0x40);
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

		for (auto it = begin; it != end; ++it) {
			insert(value_type(*it));  //local insertion.  this requires copy construction...
		}
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
		for (size_t i = 0; i < buckets; ++i) {
			std::cout << i << ": [" << container[i].first << "->" <<
					container[i].second << "] info " <<
					static_cast<size_t>(info_container[i].info) <<
					" offset = " <<
					static_cast<size_t>(info_container[i].get_offset()) <<
					" pos = " <<
					(info_container[i].get_offset() + i) << std::endl;
		}
	}

	std::vector<std::pair<key_type, mapped_type> > to_vector() const {
		std::vector<std::pair<key_type, mapped_type> > output(lsize);

		std::copy(this->cbegin(), this->cend(), output.begin());

		return output;
	}

	std::vector<key_type > keys() const {
		std::vector<key_type > output(lsize);

		std::transform(this->cbegin(), this->cend(), output.begin(),
				[](value_type const & x){ return x.first; });

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
    std::fill(this->info_container.begin(), this->info_container.end(), info_type(info_type::empty));
	}

	/**
	 * @brief reserve space for specified entries.
	 */
  void reserve(size_type n) {
    if (n > this->max_load) {   // if requested more than current max load, then we need to resize up.
      rehash(static_cast<size_t>(static_cast<float>(n) / this->max_load_factor));
      // rehash to the new size.    current bucket count should be less than next_power_of_2(n).
    }  // do not resize down.  do so only when erase.
  }

  /**
   * @brief reserve space for specified buckets.
   * @details note that buckets > entries.
   */
  void rehash(size_type const & b) {
    // check it's power of 2
    size_t n = next_power_of_2(b);

#if defined(REPROBE_STAT)
		if (n > buckets) {
			++upsize_count;
		}
		else if (n < buckets) {
			++downsize_count;
		}
#endif

		if (n != buckets) {
#if defined(REPROBE_STAT)
			std::cout << "REHASH before copy lsize = " << lsize << " capacity = " << buckets << std::endl;
#endif
			buckets = n;
			mask = buckets - 1;
			container_type tmp(buckets);
			info_container_type tmp_info(buckets, info_type(info_type::empty));
			container.swap(tmp);
			info_container.swap(tmp_info);

	    if (lsize == 0) return;   // nothing to copy.
	    lsize = 0;  // reset lsize since we will be inserting.

			min_load = static_cast<size_t>(static_cast<float>(buckets) * min_load_factor);
			max_load = static_cast<size_t>(static_cast<float>(buckets) * max_load_factor);


			copy(tmp, tmp_info);

#if defined(REPROBE_STAT)
			std::cout << "REHASH after copy lsize = " << lsize << " capacity = " << buckets << std::endl;
#endif
		}
	}



protected:


	/**
	 * return the position in container where the current key is found.  if not found, max is returned.
	 */
	std::pair<size_type, size_type> find_pos(key_type const & k) const {

		// first get the bucket id
		size_t i = hash(k) & mask;   // get bucket id
		if (info_container[i].is_empty() ) {
			// std::cout << "Empty entry at " << i << " val " << static_cast<size_t>(info_container[i].info) << std::endl;
			return std::make_pair(i, std::numeric_limits<size_type>::max());  // empty BUCKET.  so done.
		}

		// get the next bucket id
		size_t i1 = (i+1) & mask;
		// no need to check for empty.  if i is empty, then this one should be.
		// otherwise, we are checking distance so doesn't matter if empty.

		// now we scan through the current.
		size_t start = (i + info_container[i].get_offset()) & mask;  // distance is at least 0, and definitely not empty
		size_t end = (i1 + info_container[i1].get_offset()) & mask;   // distance is at least 0, and can be empty.
		size_t result = std::numeric_limits<size_type>::max();

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		size_t j;
		for (j = 0; j < buckets; ++j) {
		  if (start == end) break;  // finished scanning through

			if (eq(k, container[start].first)) {
				result = start;
				break; //found it.
			}

			start = (start + 1) & mask;

#if defined(REPROBE_STAT)
			++reprobe;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

		return std::make_pair(i, result);
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
	size_type find_next_empty_pos(size_t const & pos) const {

		// first get the bucket id
		size_t i = pos;

// this is not yet completely done.
//		size_t result = ::std::numeric_limits<size_t>::max();
//		size_t total_dist = 0;
//		// check between current and end of array.
//		while (i < buckets) {  // first check between i and end of list
//			if (info_container[i].info == info_type::empty)  {
//				// not completely empty,
//				result = i;
//				break;
//			}
//			i = (i+1 + info_container[i].get_offset());
//		}
//
//		// if i is less than buckets, then done.
//		// else if i is greater or equal, then wrapped around.
//		if (i >= buckets) {
//			i &= mask;  // wrap around
//			while (i < pos) {
//				if (info_container[i].info == info_type::empty) { // not completely empty,
//					result = i;
//					break;
//				}
//				i = (i+1 + info_container[i].get_offset());
//			}
//
//			if (result >= pos) {
//				// wrapped around, so found nothing.
//				total_dist += buckets;
//				throw std::logic_error("ERROR: could not find empty slot. container completely full.  should not happen");
//			} else {
//				total_dist += buckets - pos + i;
//			}
//
//		} else {
//			total_dist += (result - pos);
//			result = i;
//		}
//		std::cout << "found next empty position at " << i << " search dist = " << total_dist << std::endl;

		assert(lsize < buckets);

//		size_t j = i;
//		size_t offset = info_container[i].get_offset();
		while (info_container[i].info != info_type::empty) {
//			j = (i + 1) & mask;
// 			offset = info_container[j].get_offset();
//			i = (j + offset) & mask;  // jump by distance indicated in info, and increment by 1.
			if (info_container[i].info == info_type::normal) i = (i + 1) & mask;
			else i = (i + info_container[i].get_offset()) & mask;
		}
#if defined(REPROBE_STAT)
		size_t reprobe;
		if (i < pos) { // wrapped around
			reprobe = buckets - pos + i;
		} else {
			reprobe = i - pos;
		}
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		return i;

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
	size_type find_next_zero_offset_pos(size_t const & pos) const {

		// first get the bucket id
		size_t i = pos;

// this is not yet completely done.
//		size_t result = ::std::numeric_limits<size_t>::max();
//		size_t total_dist = 0;
//		// check between current and end of array.
//		while (i < buckets) {  // first check between i and end of list
//			if (info_container[i].info == info_type::empty)  {
//				// not completely empty,
//				result = i;
//				break;
//			}
//			i = (i+1 + info_container[i].get_offset());
//		}
//
//		// if i is less than buckets, then done.
//		// else if i is greater or equal, then wrapped around.
//		if (i >= buckets) {
//			i &= mask;  // wrap around
//			while (i < pos) {
//				if (info_container[i].info == info_type::empty) { // not completely empty,
//					result = i;
//					break;
//				}
//				i = (i+1 + info_container[i].get_offset());
//			}
//
//			if (result >= pos) {
//				// wrapped around, so found nothing.
//				total_dist += buckets;
//				throw std::logic_error("ERROR: could not find empty slot. container completely full.  should not happen");
//			} else {
//				total_dist += buckets - pos + i;
//			}
//
//		} else {
//			total_dist += (result - pos);
//			result = i;
//		}
//		std::cout << "found next empty position at " << i << " search dist = " << total_dist << std::endl;

		assert(lsize < buckets);

		unsigned char offset = info_container[i].get_offset();
//		if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
		while (offset > 0) {   // should try linear scan
			i = (i + offset) & mask;  // jump by distance indicated in info, and increment by 1.

			offset = info_container[i].get_offset();
		}


//		size_t j = 0;
//		unsigned char offset = info_container[i].get_offset();
//		if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//		while (offset > 0) {   // should try linear scan
//			j = (i + 1) & mask;
//			if (pos == 9) std::cout << "B i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//			offset = info_container[j].get_offset();
//			if (pos == 9) std::cout << "C i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//			i = (j + offset) & mask;  // jump by distance indicated in info, and increment by 1.
//
//			if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//		}
#if defined(REPROBE_STAT)
		size_t reprobe;
		if (i < pos) { // wrapped around
			reprobe = buckets - pos + i;
		} else {
			reprobe = i - pos;
		}
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		return i;

	}


	/**
	 * find next non-empty position.  meant to be with input that points to an empty position.
	 */
	size_type find_next_non_empty_pos(size_t const & pos) const {
		size_t i = (pos + 1) & mask;
		while (info_container[i].info == info_type::empty) {
			i = (i+1) & mask;
		}
		return i;
	}

	/**
	 * @brief inserts a range into the current hash table.
	 */
	void copy(container_type const & tmp, info_container_type const & info_tmp) {

//		size_t count = 0;

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif


		// insert using the iterators.
		insert(const_iterator(tmp.cbegin(), info_tmp.cbegin(), info_tmp.cend(), filter),
				const_iterator(tmp.cend(), info_tmp.cend(), filter));


#if defined(REPROBE_STAT)
		std::cout << "REHASH copy:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) <<
				"\treprobe total=" << this->reprobes <<
					"\ttotal=" << tmp.size() << "\tlsize=" << lsize <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
	}


public:

	/**
	 * @brief insert a single key-value pair into container.
	 */
	std::pair<iterator, bool> insert(value_type && vv) {

		if (lsize >= max_load) {
//			std::cout << "lsize " << lsize << " max_load " << max_load << " new size " << (buckets << 1) << std::endl;
			rehash(buckets << 1);  // double in size.
		}

		size_type bid;
		size_type found;
		std::tie(bid, found) = find_pos(vv.first); // container insertion point.

		// empty, so insert.
		if (info_container[bid].info == info_type::empty) {
			// empty, can insert directly.
			info_container[bid].info = info_type::normal;
			container[bid] = std::move(vv);
			++lsize;


//		std::cout << "insert in bucket " << bid << std::endl;
			return std::make_pair(iterator(container.begin() + bid, info_container.begin()+ bid, info_container.end(), filter), true);
		}

		// check for insertion point.
		//size_type found = find_pos(vv.first); // container insertion point.

		if (found < buckets) {  // found it.  so no insert.
//			std::cout << "duplicate found at " << found << std::endl;
			return std::make_pair(iterator(container.begin() + found, info_container.begin()+ found, info_container.end(), filter), false);
		}

		// did not find, so insert.
		++lsize;

		// else this is already occupied.  insert at the end of bucket == start of next bucket
		size_t bid1 = (bid + 1) & mask;
		found = (bid1 + info_container[bid1].get_offset()) & mask; // insert location

		// end of region to move.
		size_type end = find_next_empty_pos(found);

//		std::cout << "insert in bucket " << bid << " bucket + 1 " << bid1 << " position " << found << " update to " << end << std::endl;
	
		size_t j = 0;
		// first increment from [bid1 .. found1)
		size_t pos = bid1;
		for (; j < buckets; ++j) {
		  if (pos == found) break;

		  ++(info_container[pos]);
		  pos = (pos + 1) & mask;
		}

//		std::cout << "updated offsets [" << bid1 << " .. " << pos << ")" << std::endl;

		// next shift right from [found1 .. end)
		value_type uu;
		size_t pos1 = (end + mask) & mask;
		pos = end;
		for (; j < buckets; ++j) {
      if (pos == found) break;

      //manual swap.
      container[pos] = std::move(container[pos1]);
      ++(info_container[pos]);  // increment current.

      pos = pos1;
	pos1 = (pos + mask) & mask;
		}

//		std::cout << "shifted [" << found << " .. " << end << ")" << std::endl;

		// now handle last one.
		container[found] = std::move(vv);
		++(info_container[found]);  // increment insertion point.
    

// if the current bucket is empty, then change to say occupied.
    info_container[bid].set_normal();

//		std::cout << "inserted at " << found << ", marked as non-empty at  " << bid << std::endl;
		
		return std::make_pair(iterator(container.begin() + found, info_container.begin()+ found, info_container.end(), filter), true);

	}

	std::pair<iterator, bool> insert(key_type && key, mapped_type && val) {
		auto result = insert(std::make_pair(key, val));
		return result;
	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	void insert(Iter begin, Iter end) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		size_t count = 0;
		size_t fail_count = 0;
		iterator dummy;
		bool success;

		for (auto it = begin; it != end; ++it) {
			std::tie(dummy, success) = insert(std::move(value_type(*it)));  //local insertion.  this requires copy construction...

			if (success) {
				++count;
			} else {
				++fail_count;
			}
		}


#if defined(REPROBE_STAT)
		std::cout << "INSERT batch success = " << count << " failed = " << fail_count << std::endl;
		std::cout << "INSERT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << count << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		reserve(lsize);  // resize down as needed
	}

	/// batch insert not using iterator
	void insert(std::vector<value_type> && input) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		size_t count = 0;

		for (size_t i = 0; i < input.size(); ++i) {
			if ( insert(std::move(input[i])).second)   //local insertion.  this requires copy construction...
				++count;
		}

#if defined(REPROBE_STAT)
    std::cout << "lsize " << lsize << std::endl;

    std::cout << "INSERT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << count << "\ttotal=" << input.size() <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		reserve(lsize);  // resize down as needed
	}

	/**
	 * @brief count the presence of a key
	 */
	size_type count( key_type const & k ) const {
		size_t bucket;
		size_t pos;
		std::tie(bucket, pos) = find_pos(k);
	
//		std::cout << "found at bucket " << bucket << " pos " << pos << std::endl;
	
		return (pos < buckets) ? 1 : 0;

	}


	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
		::std::vector<size_type> counts;
		counts.reserve(std::distance(begin, end));

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			counts.emplace_back(count((*it).first));
		}

#if defined(REPROBE_STAT)
		std::cout << "COUNT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << counts.size() << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		return counts;
	}


	template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
		::std::vector<size_type> counts;
		counts.reserve(std::distance(begin, end));

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			counts.emplace_back(count(*it));
		}

#if defined(REPROBE_STAT)
		std::cout << "COUNT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << counts.size() << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		return counts;
	}

	/**
	 * @brief find the iterator for a key
	 */
	iterator find(key_type const & k) {

		size_type idx = find_pos(k).second;

		if (idx < buckets)
			return iterator(container.begin() + idx, info_container.begin()+ idx, info_container.end(), filter);
		else
			return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		size_type idx = find_pos(k).second;

		if (idx < buckets)
			return const_iterator(container.cbegin() + idx, info_container.cbegin()+ idx, info_container.cend(), filter);
		else
			return const_iterator(container.cend(), info_container.cend(), filter);

	}

	/**
	 * @brief.  updates current value.  behaves like insert, but overwrites the existing.
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


	/**
	 * @brief erases a key.  performs backward shift.  swap at bucket boundaries only.
	 */
	size_type erase_no_resize(key_type const & k) {
		size_t bid;
		size_type found;

		std::tie(bid, found) = find_pos(k);  // get the matching position

		if (found >= buckets) {
			// did not find. done
// debug			std::cout << "erasing " << k << " did not find one " << std::endl;
			return 0;
		}

		--lsize;   // reduce the size by 1.

		//size_t bid = hash(k) & mask;   // get bucket id
		// get the end of the non-empty range, starting from the next position.
    size_type bid1 = (bid + 1) & mask;  // get the next bucket, since bucket contains offset for current bucket.


		size_type found1 = (found + 1) & mask;
		size_type end = find_next_zero_offset_pos(found1);

// debug		std::cout << "erasing " << k << " hash " << bid << " at " << found << " end is " << end << std::endl;

		// move to backward shift.  move [found+1 ... end-1] to [found ... end - 2].  end is excluded because it has 0 dist.
		if (end < found) {  // wrapped around
			// first move to the end.
			memmove(&(container[found]), &(container[found1]), ((buckets - 1) - found) * sizeof(value_type));

			// definitely wrapped around, so copy first entry to last
			if (end > 0) {
				container[buckets - 1] = std::move(container[0]);

				// now if there is still more, than do it
				memmove(&(container[0]), &(container[1]), (end - 1) * sizeof(value_type));
			}
		} else if (end > found) {  // no wrap around.
			memmove(&(container[found]), &(container[found1]), (end - 1 - found) * sizeof(value_type));
		} // else target == found, nothing to shift.

// debug		print();

		// now change the offsets.
		// start from bid+1, end at end - 1.


		// if that was the last entry for the bucket, then need to change this to say empty.
		if (info_container[bid].get_offset() == info_container[bid1].get_offset()) {  // both have same distance, so bid has only 1 entry
			info_container[bid].set_empty();
		}

//
//		if (bid1 == end) {
//			// if bid and end are same now, then nothing to change.  return.
//			return 1;
//		}
//
//		// get last valid one (i.e. not 0x80)
//		end = (end == 0) ? (buckets - 1) : (end - 1);

//	debug 	std::cout << "adjusting " << bid1 << " to " << end << std::endl;

		// the for each operator than should never encounter 0x80.  exclude end...
		if (end < bid1) {  // wrapped around
			// first move to the end.
			std::for_each(info_container.begin() + bid1, info_container.end(),
					[](info_type & x){ --x; });

			std::for_each(info_container.begin(), info_container.begin() + end,
					[](info_type & x){ --x; });

		} else {  // no wrap around.
			std::for_each(info_container.begin() + bid1, info_container.begin() + end,
					[](info_type & x){ --x; });

		}

		// clear the last one.  The last one HAS to be next to an empty one, so distance has to be 1.
// debug		print();

		return 1;

	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	size_type erase_no_resize(Iter begin, Iter end) {
		size_type erased = 0;

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			erased += erase_no_resize((*it).first);
		}

#if defined(REPROBE_STAT)
		std::cout << "ERASE batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << erased << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		return erased;
	}

	template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	size_type erase_no_resize(Iter begin, Iter end) {
		size_type erased = 0;

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			erased += erase_no_resize(*it);
		}

#if defined(REPROBE_STAT)
		std::cout << "ERASE batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << erased << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		return erased;
	}

	/**
	 * @brief erases a key.
	 */
	size_type erase(key_type const & k) {

		size_type res = erase_no_resize(k);

		if (lsize < min_load) {
			std::cout << "lsize " << lsize << " min_load " << min_load << " new size " << (buckets >> 1) << std::endl;
			rehash(buckets >> 1);
		}

		return res;
	}

	template <typename Iter>
	size_type erase(Iter begin, Iter end) {

		size_type erased = erase_no_resize(begin, end);

		//std::cout << "erase resize: curr size is " << lsize << " target max_load is " << (static_cast<float>(lsize) / max_load_factor) << " buckets is " <<
		//		next_power_of_2(static_cast<size_t>(static_cast<float>(lsize) / max_load_factor)) << std::endl;
		if (lsize < min_load) {
			
			std::cout << "lsize " << lsize << " min_load " << min_load << " new size " << 
				next_power_of_2(static_cast<size_t>(static_cast<float>(lsize) / max_load_factor)) << std::endl;
			rehash(static_cast<size_t>(static_cast<float>(lsize) / max_load_factor));
		}
		return erased;
	}


};


}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS_HPP_ */
