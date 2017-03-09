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
 * hashtable_OA_RH_doubling.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_HASHTABLE_OA_RH_DOUBLING_HPP_
#define KMERHASH_HASHTABLE_OA_RH_DOUBLING_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "math_utils.hpp"



namespace fsc {

/**
 * @brief Open Addressing hashmap that uses linear addressing, with doubling for hashing, and with specialization for k-mers.
 * @details  at the moment, nothing special for k-mers yet.
 * 		This class has the following implementation characteristics
 * 			vector of structs
 * 			open addressing with linear probing
 * 			doubling for reallocation
 *
 *
 *
 *
 *
 * 			using vector internally.
 * 			tracking empty and deleted via special bit instead of using empty and deleted keys (potentially has to compare whole key)
 * 				use bit array instead of storing flags with value type
 * 				Q: should the key and value be stored in separate arrays?  count, exist, and erase may be faster (no need to touch value array).
 *
 *			MPI friendliness - would never send the vector directly - need to permute, which requires extra space and time, and only send value when insert (and possibly update),
 *				neither are technically sending the vector really.
 *
 *				so assume we always are copying then sending, which means we can probably construct as needed.
 *
 *		WANT a simple array of key, value pair.  then have auxillary arrays to support scanning through, ordering, and reordering.
 *			memorizing hashes can be done as well in aux array....
 *
 *
 * 			somethings to put in the aux array:  empty bit.  deleted bit.  probe distance.
 * 				memorizing the hash requires more space, else would need to compute hash anyways.
 *
 * 			TODO: bijective hash could be useful....
 *
 *  requirements
 *		support iterate
 *		no special keys
 *		incremental allocation (perhaps not doubling right now...
 *
 *
 *
 *
 *
 *  array of tuples vs tuple of arrays
 *  start with array of tuples
 *  then do tuple of arrays
 *
 *  hashset may be useful for organizing the query input.
 *  how to order the hash table accesses so that there is maximal cache reuse?  order things by hash?
 *
 *  using a C array may be better - better memory allocation control.
 *
 *
 *
 *  linear probing - if reached end, then wrap around?  or just allocate some extra, but still hash to first 2^x positions.
 *
 *  when rehashing under doubling/halving, since always power of 2 in capacity, can just merge the higher half to the lower half by index.
 *  batch process during insertion?
 *
 *  TODO: when using power-of-2 sizes, using low-to-high hash value bits means that when we grow, a bucket is split between buckets i and i+2^x.
 *  		if we use high-to-low bits, then resize up split into adjacent buckets.
 *  		it is possible that some buckets may not need to move.	 It is possible to grow incrementally potentially.
 *
 *  FIX:
 *  [x] remove max_probe - treat as if circular array.
 *  [x] separate info from rest of struct, so as to remove use of transform_iterator, thus allowing us to change values through iterator. requires a new aux_filter_iterator.
 *  [ ] batch mode operations to allow more opportunities for optimization including SIMD
 *  [ ] predicated version of operations
 *  [ ] macros for repeated code.
 *  [x] testing with k-mers
 *  [x] measure reprobe count for insert
 *  [x] measure reprobe count for find and update.
 *  [x] measure reprobe count for count and erase.
 *  [x] robin hood hashing with back shifting during deletion.
 *  [ ] memmove?  probably not.  current approach swaps only when dist increases, not when they are the same, which is what most of a bucket would contain, so total mem access would be low.
 *
 *  Robin Hood Hashing logic follows
 *  	http://www.sebastiansylvan.com/post/robin-hood-hashing-should-be-your-default-hash-table-implementation/
 *
 *  use an auxillary array.
 *  where as RH standard aux array element represents the distance from source for the corresponding element in container.
 *      idx   ----a---b------------
 *    aux   ===|-|=|-|=|4|3|=====   _4_ is recorded at position hash(X) + _4_ of aux array.
 *    data  -----------|X|Y|-----,  X is inserted into bucket hash(X) = a.  in container position hash(X) + _4_.
 *
 *    empty positions are set to 0x80, and same for all info entries.
 *
 *  this aux array instead has each element represent the offset for the first entry of this bucket.
 *      idx   ----a---b------------
 *    aux   ===|4|=|3|=|-|-|=====   _4_ is recorded at position hash(X) of aux array.
 *    data  -----------|X|Y|-----,  X is inserted into bucket hash(X), in container position hash(X) + _4_.
 *
 *    empty positions are set with high bit 1, but can has distance larger than 0.
 *
 *  in traditional, we linear scan info_container from position hash(Y) and must go to hash(Y) + 4 linearly.
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
 *  oa = open addressing
 *  rh = robin hood
 *  do = doubling
 *  tuple = array of tuples (instead of tuple of arrays)
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_oa_rh_do_tuple {

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
    mutable unsigned char max_reprobes;

public:

    /**
     * _capacity is the number of usable entries, not the capacity of the underlying container.
     */
	explicit hashmap_oa_rh_do_tuple(size_t const & _capacity = 128,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
			lsize(0), buckets(next_power_of_2(_capacity) ), mask(buckets - 1),
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
	hashmap_oa_rh_do_tuple(Iter begin, Iter end,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
		hashmap_oa_rh_do_tuple(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

		for (auto it = begin; it != end; ++it) {
			insert(value_type(*it));  //local insertion.  this requires copy construction...
		}
	}

	~hashmap_oa_rh_do_tuple() {
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

			buckets = n;
			mask = buckets - 1;
			container_type tmp(buckets);
			info_container_type tmp_info(buckets, info_type(info_type::empty));
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

		unsigned char reprobe = info_type::normal;


		// first check if we need to resize.
		if (lsize >= max_load) rehash(buckets << 1);

		// first get the bucket id
		size_type i = hash(vv.first) & mask;  // target bucket id.

		size_type insert_pos = std::numeric_limits<size_type>::max();
		bool success = false;
		size_type j = 0;

		for (; j < buckets; ++j) {  // limit to 1 complete loop

			// implementing back shifting, so don't worry about deleted entries.

			// insert if empty, or if reprobe distance is larger than current position's.  do this via swap.
			// continue to probe if we swapped out a normal entry.
			// this logic relies on choice of bits for empty entries.

			// we want the reprobe distance to be larger than empty, so we need to make
			// normal to have high bits 1 and empty 0.

			// save the insertion position (or equal position) for the first insert or match, but be aware that after the swap
			//  there will be other inserts or swaps. BUT NOT OTHER MATCH since this is a MAP.

			// if current distance is larger than target's distance, (including empty cell), then swap.
			assert(reprobe >= info_type::normal);  // if distance is over 128, then wrap around would get us less than normal.

			if (reprobe > info_container[i].info) {
				::std::swap(info_container[i].info, reprobe);

				if (insert_pos == std::numeric_limits<size_type>::max()) {
					insert_pos = i;  // set insert_pos to first positin of insertion.
					success = true;
					++lsize;
				}  // subsequent "shifts" do not change insert_pos or success.

				// then decide what to do given the swapped out distance.
				if (reprobe == info_type::empty) {  // previously it was empty,
					// then we can simply set.
					container[i] = std::move(vv);

					break;

				} else {
					// there was a real entry, so need to swap
					::std::swap(container[i], vv);

					// and continue
				}
			} else if (reprobe == info_container[i].info) {
			  // check for equality, only if haven't inserted (if don't check success, there could be a lot of equality checks.

//			  if (success)  std::cout << ".";

				// same distance, then possibly same value.  let's check.
				if (!success && eq(container[i].first, vv.first)) {  // note that a previously swapped vv would not match again and can be skipped.
					// same, then we found it and need to return.
					insert_pos = i;
					success = false;
					break;
				}  // note that the loop terminates within here so don't need to worry about another insertion point.

			}  // note that it's not possible for a match to occur with shorter current reprobe distance -
				// that would mean same element is hashed to multiple buckets.

			// increment probe distance
			++reprobe;

			// circular array.
			i = (i+1) & mask;   // for bucket == 2^x, this acts like modulus.
		}
		assert(j < buckets);

#if defined(REPROBE_STAT)
		this->reprobes += j;
		this->max_reprobes = std::max(this->max_reprobes, j);
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

protected:

	/**
	 * return the bucket id where the current key is found.  if not found, max is returned.
	 */
	size_type find_pos(key_type const & k) const {

		unsigned char reprobe = info_type::normal;

		// first get the bucket id
		size_t i = hash(k) & mask;

		size_type result = std::numeric_limits<size_type>::max();

		size_t j;
		for (j = 0; j < buckets; ++j) {
			if (reprobe > info_container[i].info) {
				// either empty, OR query's distance is larger than the current entries, which indicates
				// this would have been swapped during insertion, and can also indicate that the current
				// entry is from a different bucket.
				// so did not find one.
				break;
			} else if (reprobe == info_container[i].info) {
				// possibly matched.  compare.
				if (eq(k, container[i].first)) {
					result = i;
					break;
				}
			} // else still traversing a previous bucket.

			++reprobe;
			i = (i+1) & mask;   // again power of 2 modulus.
		}
		assert(j < buckets);

#if defined(REPROBE_STAT)
		reprobe &= info_type::dist_mask;
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, reprobe);
#endif

		return result;
	}

public:

	/**
	 * @brief count the presence of a key
	 */
	size_type count( key_type const & k ) const {

		return (find_pos(k) < buckets) ? 1 : 0;

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

		if (idx < buckets)
			return iterator(container.begin() + idx, info_container.begin()+ idx, info_container.end(), filter);
		else
			return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		size_type idx = find_pos(k);

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



	/**
	 * @brief erases a key.  performs backward shift.
	 * @details note that entries for the same bucket are always consecutive, so we don't have to shift, but rather, move elements at boundaries only.
	 *    boundaries are characterized by non-decreasing info dist, or by empty.
	 *
	 */
	size_type erase_no_resize(key_type const & k) {



		size_type found = find_pos(k);

		if (found >= buckets) {  // did not find.
			return 0;
		}

    --lsize;   // reduce the size by 1.

    // for iterating to find boundaries
		size_type curr = found;
		size_type next = (found + 1) & mask;

		//    unsigned char curr_info = info_container[curr].info;
		unsigned char next_info = info_container[next].info;

		// a little more short circuiting.
		if (next_info <= info_type::normal) {
			info_container[curr].info = info_type::empty;
			return 1;
		}

		size_type target = found;

		for (size_t j = 0; j < buckets - 1; ++j) {
			// terminate when next entry is empty or has distance 0.
			if (next_info <= info_type::normal) break;

			// do something at bucket boundaries
			if (next_info <= info_container[curr].info) {
				// change the curr info
				info_container[curr].info = next_info - 1;				

				// and move the current value entry.
				container[target] = std::move(container[curr]);

				// store the curr position as the next target of move.
				target = curr;
			}

			// advance
			curr = next;
			next = (next+1) & mask;

			next_info = info_container[next].info;

		}
		info_container[curr].info = info_type::empty;  // set last to empty.
		container[target] = std::move(container[curr]);


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
#endif /* KMERHASH_HASHTABLE_OA_RH_DOUBLING_HPP_ */
