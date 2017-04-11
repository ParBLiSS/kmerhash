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
 * hashtable_OA_RH_DO_noncircular.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_HASHTABLE_OA_RH_DO_NONCIRCULAR_HPP_
#define KMERHASH_HASHTABLE_OA_RH_DO_NONCIRCULAR_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"


namespace fsc {

/**
 * @brief Open Addressing hashmap that uses robinhood hashing, with doubling for allocation, and noncircular internal array.  Modified from "noncircular"
 * @details
 *    This class is a modified version of "noncircular" in an attempt to make the insert/erase/find operations faster through mostly early termination.
 *        however, performance actually degraded.
 * at the moment, nothing special for k-mers yet.
 * 		This class has the following implementation characteristics
 * 			vector of structs
 * 			open addressing with robinhood hashing
 * 			doubling for reallocation
 * 			linear array for storage.
 *
 *
 *
 *  TODO:
 *  [x] early termination of loops.  did not help..
 *
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_robinhood_doubling_noncircular {

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

    	static constexpr unsigned char empty = 0x00;
    	static constexpr unsigned char dist_mask = 0x7F;
    	static constexpr unsigned char normal = 0x80;   // this is used to initialize the reprobe distances.


    	inline bool is_empty() const {
    		return info == empty;  // empty 0x40
    	}
    	inline bool is_normal() const {
    		return info >= normal;  // normal. both top bits are set. 0xC0
    	}
    	inline void set_normal() {
    		info |= normal;  // set the top bits.
    	}
    	inline void set_empty() {
    		info = empty;  // nothing here.
    	}

    };


    using container_type		= ::std::vector<value_type, Allocator>;
    using info_container_type	= ::std::vector<info_type, Allocator>;

    // filter
    struct empty_deleted_filter {
    	bool operator()(info_type const & x) { return x.is_normal(); };
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
    mutable size_t buckets;  // always add info_type::normal number of entries, then we don't need to circularize.  buckets exclude this part.
    mutable size_t mask;
    mutable size_t min_load;
    mutable size_t max_load;
    mutable float min_load_factor;
    mutable float max_load_factor;   // load factors are computed using buckets, not with the extra


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
	explicit hashmap_robinhood_doubling_noncircular(size_t const & _capacity = 128,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
			lsize(0), buckets(next_power_of_2(_capacity)), mask(buckets - 1),
			container(buckets + info_type::normal), info_container(buckets + info_type::normal, info_type(info_type::empty)),
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
	hashmap_robinhood_doubling_noncircular(Iter begin, Iter end,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
		hashmap_robinhood_doubling_noncircular(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

		insert(begin, end);
	}

	~hashmap_robinhood_doubling_noncircular() {
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

  void print() const {
    for (size_t i = 0; i < container.size(); ++i) {
      std::cout << i << ": [" << container[i].first << "->" <<
          container[i].second << "] info " <<
          static_cast<size_t>(info_container[i].info) <<
          " offset = " <<
          static_cast<size_t>(info_container[i].info & info_type::dist_mask) <<
          " pos = " <<
          ((info_container[i].info & info_type::dist_mask) + i) << std::endl;
    }
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
	 * @details	note that buckets > entries.
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

			buckets = n;
			mask = buckets - 1;
			container_type tmp(buckets + info_type::normal);
			info_container_type tmp_info(buckets  + info_type::normal, info_type(info_type::empty));
			container.swap(tmp);
			info_container.swap(tmp_info);
	    lsize = 0;  // insert increments the lsize.  this ensures that it is correct.

			min_load = static_cast<size_t>(static_cast<float>(buckets) * min_load_factor);
			max_load = static_cast<size_t>(static_cast<float>(buckets) * max_load_factor);

			copy(tmp.begin(), tmp.end(), tmp_info.begin());
		}
	}



protected:
	/**
	 * @brief inserts a range into the current hash table.
	 */
	void copy(typename container_type::iterator begin, typename container_type::iterator end, typename info_container_type::iterator info_begin) {


#if defined(REPROBE_STAT)
    size_t count = 0;
		this->reprobes = 0;
		this->max_reprobes = 0;
    iterator dummy;
    bool success;
#endif

		auto it = begin;
		auto iit = info_begin;
		for (; it != end; ++it, ++iit) {
			if ((*iit).is_normal()) {
#if defined(REPROBE_STAT)
				std::tie(dummy, success) = insert(::std::move(*it));
				if (success) ++count;
#else
				insert(::std::move(*it));
#endif
			}
		}

#if defined(REPROBE_STAT)
		std::cout << "REHASH copy:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << count << "\ttotal=" << std::distance(begin, end) << "\tlsize=" << lsize <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
	}



public:



	/**
	 * @brief insert a single key-value pair into container.
	 *
	 * note that swap only occurs at bucket boundaries.
	 */
	std::pair<iterator, bool> insert(value_type && vv) {

    // implementing back shifting, so don't worry about deleted entries.

    // insert if empty, or if reprobe distance is larger than current position's.  do this via swap.
    // continue to probe if we swapped out a normal entry.
    // this logic relies on choice of bits for empty entries.

    // we want the reprobe distance to be larger than empty, so we need to make
    // normal to have high bits 1 and empty 0.

    // save the insertion position (or equal position) for the first insert or match, but be aware that after the swap
    //  there will be other inserts or swaps. BUT NOT OTHER MATCH since this is a MAP.

	  unsigned char reprobe = info_type::normal;

		// first check if we need to resize.
		if (lsize >= max_load) rehash(buckets << 1);

		// first get the bucket id
		size_type i = hash(vv.first) & mask;  // target bucket id.
		size_type bid = i;

		// shortcutting.
		if (info_container[i].info == info_type::empty) {
		  info_container[i].info = reprobe;
		  container[i] = std::move(vv);
		  ++lsize;
		  return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), true);
		}

		size_type insert_pos = std::numeric_limits<size_type>::max();
		bool success = false;

		// scan through the part before bucket i
		for (; reprobe < info_container[i].info; ++reprobe, ++i) {};

		// compare entries in bucket i, see if there is a match.
		for (; reprobe == info_container[i].info; ++reprobe, ++i) {
		  if (eq(container[i].first, vv.first)) {
		    // matched, so found.

		    return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), false);
		  }
		}

		// insert the original operand.
		if (info_container[i].info == info_type::empty) {  // if currently empty, move it in.
      info_container[i].info = reprobe;
      container[i] = std::move(vv);
      ++lsize;
      return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), true);
		} else {  // if occupied, swap.
		  std::swap(reprobe, info_container[i].info);
		  std::swap(container[i], vv);
		  insert_pos = i;
		  // go to next pos.
		  ++i;
		  ++reprobe;
		}

		// loop until reach an empty slot.
		for (; i < container.size(); ) {
	    // skip entries of same bucket
	    for (; reprobe == info_container[i].info; ++reprobe, ++i) {};

	    // now swap back in.

	    // insert the original operand.
	    if (info_container[i].info == info_type::empty) {  // if currently empty
	      info_container[i].info = reprobe;
	      container[i] = std::move(vv);
	      ++lsize;
	      return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), true);
	    } else {
	      // not empty, so swap
	      std::swap(reprobe, info_container[i].info);
	      std::swap(container[i], vv);
	      insert_pos = i;
	      // go to next pos.
	      ++i;
	      ++reprobe;
	    }

		}

		// if here, then we failed to insert, probably due to array being full.
		return std::make_pair(iterator(container.end(), info_container.end(), filter), false);


#if defined(REPROBE_STAT)

		this->reprobes += i-bid;
		this->max_reprobes = std::max(this->max_reprobes, (i-bid));
#endif
		return std::make_pair(iterator(container.begin() + insert_pos, info_container.begin()+ insert_pos, info_container.end(), filter), success);

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
		iterator dummy;
		bool success;

		for (auto it = begin; it != end; ++it) {
			std::tie(dummy, success) = insert(std::move(value_type(*it)));  //local insertion.  this requires copy construction...

			if (success) {
				++count;
			}
		}

#if defined(REPROBE_STAT)
    std::cout << "lsize " << lsize << std::endl;

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


protected:

	/**
	 * return the bucket id where the current key is found.  if not found, max is returned.
	 */
	size_type find_pos(key_type const & k) const {

		unsigned char reprobe = info_type::normal;

		// first get the bucket id
		size_t i = hash(k) & mask;

		size_type result = std::numeric_limits<size_type>::max();

    // Exit when either empty, OR query's distance is larger than the current entries, which indicates
    // this would have been swapped during insertion, and can also indicate that the current
    // entry is from a different bucket.
    // so did not find one.

		// closer to the old way.  see if this is slightly faster...
		for (; i < info_container.size(); ++i, ++reprobe) {
			if (reprobe > info_container[i].info) break;

			// else do check
			if ((reprobe == info_container[i].info) && (eq(k, container[i].first)))
			{
				result = i;
				break;
			}
		}

//		Cleaner code. really should be faster with less branching, but it's not...
//		// first scan for equal or greater
//		for (; reprobe < info_container[i].info; ++reprobe, ++i) {};
//
//		// now scan within the equal region for kmer match.
//		for (; reprobe == info_container[i].info; ++reprobe, ++i) {
//      // possibly matched.  compare.
//      if (eq(k, container[i].first)) {
//        result = i;
//        break;
//      }
//		}

#if defined(REPROBE_STAT)
		reprobe &= info_type::dist_mask;
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

		return result;
	}

public:

	/**
	 * @brief count the presence of a key
	 */
	size_type count( key_type const & k ) const {

		return (find_pos(k) < container.size()) ? 1 : 0;

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

		size_type idx = find_pos(k);

		if (idx < container.size())
			return iterator(container.begin() + idx, info_container.begin()+ idx, info_container.end(), filter);
		else
			return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		size_type idx = find_pos(k);

		if (idx < container.size())
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


	/**
	 * @brief erases a key.  performs backward shift.
	 * @details note that entries for the same bucket are always consecutive, so we don't have to shift, but rather, move elements at boundaries only.
	 *    boundaries are characterized by non-decreasing info dist, or by empty.
	 *
	 */
	size_type erase_no_resize(key_type const & k) {

		size_type found = find_pos(k);

		if (found >= container.size()) {  // did not find.
			return 0;
		}

    --lsize;   // reduce the size by 1.

    size_type target = found;  // save for later swapping.
    size_type i = (found + 1);  // curr pos.

    // for safety.  check last entry.
    if (i == container.size()) return 1;

    // save original target info so can compare (++) to actual
    unsigned char reprobe = info_container[found].info;
    // move to next position
    ++reprobe;  // next entry SHOULD be ...

    // update the info at target
    unsigned char target1_info = info_container[i].info;
    info_container[target].info = (target1_info <= info_type::normal) ? info_type::empty : (target1_info - 1);

    // shortcutting
    if (target1_info <= info_type::normal) return 1;


    // loop at most to end of container.
    while (i < container.size()) {

      // scan to end of bucket (reprobe == info)
      for (; reprobe == info_container[i].info; ++reprobe, ++i) {};

      // update the info at i-1 (new target)
      reprobe = info_container[i].info;
      info_container[i-1].info = (reprobe <= info_type::normal) ? info_type::empty : (reprobe - 1);

      // move the entry from new target to old target.
      container[target] = std::move(container[i-1]);

      // if this is the last one, then done.
      if (reprobe <= info_type::normal) return 1;

      // else set target
      target = i-1;
      ++i;
      ++reprobe;

    }

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

		if (lsize < min_load) rehash(buckets >> 1);

		return res;
	}

	template <typename Iter>
	size_type erase(Iter begin, Iter end) {

		size_type erased = erase_no_resize(begin, end);

		//std::cout << "erase resize: curr size is " << lsize << " target max_load is " << (static_cast<float>(lsize) / max_load_factor) << " buckets is " <<
		//		next_power_of_2(static_cast<size_t>(static_cast<float>(lsize) / max_load_factor)) << std::endl;
		if (lsize < min_load) rehash(static_cast<size_t>(static_cast<float>(lsize) / max_load_factor));

		return erased;
	}


};

}  // namespace fsc
#endif /* KMERHASH_HASHTABLE_OA_RH_DO_NONCIRCULAR_HPP_ */
