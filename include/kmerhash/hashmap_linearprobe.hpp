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
 * hashtable_OA_LP_doubling.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 */

#ifndef KMERHASH_HASHMAP_LINEARPROBE_HPP_
#define KMERHASH_HASHMAP_LINEARPROBE_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"



namespace fsc {

/**
 * @brief Open Addressing hashmap that uses linear addressing, with doubling for allocation, and with specialization for k-mers.
 * @details
 *    at the moment, nothing special for k-mers yet.
 * 		This class has the following implementation characteristics
 * 			vector of structs
 * 			open addressing with linear probing
 * 			doubling for reallocation
 *      array of tuples
 *      circular array for storage.
 *
 *
 *   Implementation Details
 * 			using vector internally.
 * 			tracking empty and deleted via special bit instead of using empty and deleted keys (potentially has to compare whole key)
 * 				use byte array instead of storing flags with value type
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
 * 			somethings to put in the aux array:  empty bit.  deleted bit.  probe distance.
 * 				memorizing the hash requires more space, else would need to compute hash anyways.
 *
 *  requirements
 *		[x] support iterate
 *		[x] no special keys
 *		[ ] incremental allocation (perhaps not doubling right now...
 *
 *
 *  how to order the hash table accesses so that there is maximal cache reuse?  order things by hash?
 *
 *
 *  MAYBE:
 *      hashset may be useful for organizing the query input.
 *      when using power-of-2 sizes, using low-to-high hash value bits means that when we grow, a bucket is split between buckets i and i+2^x.
 *      when rehashing under doubling/halving, since always power of 2 in capacity, can just merge the higher half to the lower half by index.
 *  		  if we use high-to-low bits, then resize up split into adjacent buckets.
 *  	  	it is possible that some buckets may not need to move.	 It is possible to grow incrementally potentially.
 *      using a C array may be better - better memory allocation control.
 *
 *  TODO:
 *  [x] remove max_probe - treat as if circular array.
 *  [x] separate info from rest of struct, so as to remove use of transform_iterator, thus allowing us to change values through iterator. requires a new aux_filter_iterator.
 *  [ ] batch mode operations to allow more opportunities for optimization including SIMD
 *  [ ] predicated version of operations
 *  [ ] macros for repeated code.
 *  [x] testing with k-mers
 *
 *  first do the stupid simple implementation.
 *
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_linearprobe_doubling {

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

    	static constexpr unsigned char empty = 0x40;
    	static constexpr unsigned char deleted = 0x80;


    	inline bool is_empty() const {
    		return info == empty;  // empty 0x40
    	}
    	inline bool is_deleted() const {
    		return info == deleted;  // deleted. 0x80
    	}
    	inline bool is_normal() const {  // upper bits == 0
    		return info < empty;  // lower 6 bits are set.
      	}
    	inline void set_deleted() {
    		info = deleted;  // set the top bit.
    	}
    	inline void set_empty() {
    		info = empty;  // nothing here.
    	}
    	inline void set_normal() {
    		info &= 0x3F;  // clear the upper bits;
    	}



    };


    using container_type		= ::std::vector<value_type, Allocator>;
    using info_container_type	= ::std::vector<info_type, Allocator>;

    // filter
    struct valid_entry_filter {
    	bool operator()(info_type const & x) { return x.is_normal(); };
    };

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

    size_t lsize;
    mutable size_t buckets;
    mutable size_t min_load;
    mutable size_t max_load;
    mutable float min_load_factor;
    mutable float max_load_factor;


    valid_entry_filter filter;
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
	explicit hashmap_linearprobe_doubling(size_t const & _capacity = 128,
			float const & _min_load_factor = 0.2,
			float const & _max_load_factor = 0.6) :
			lsize(0), buckets(next_power_of_2(_capacity)),
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
	hashmap_linearprobe_doubling(Iter begin, Iter end, float const & _min_load_factor = 0.2, float const & _max_load_factor = 0.6) :
		hashmap_linearprobe_doubling(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

		for (auto it = begin; it != end; ++it) {
			insert(value_type(*it));  //local insertion.  this requires copy construction...
		}
	}

	~hashmap_linearprobe_doubling() {
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
			container_type tmp(buckets);
			info_container_type tmp_info(buckets, info_type(info_type::empty));
			container.swap(tmp);
			info_container.swap(tmp_info);

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

		size_t count = 0;

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		auto it = begin;
		auto iit = info_begin;
		for (; it != end; ++it, ++iit) {
			if ((*iit).is_normal()) {
				count += copy_one(*it, *iit);
			}
		}

#if defined(REPROBE_STAT)
		std::cout << "REHASH copy:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << count << "\ttotal=" << std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
	}


	/**
	 * @brief insert at the specified position.  does NOT check if already exists.  for internal use during rehash only.
	 * @details passed in info should have "normal" flag.
	 * @return reprobe count
	 */
	size_type copy_one(value_type const & value, info_type const & info) {

		if (buckets == 0) return 0;

		size_type pos = hash(value.first) % buckets;
		size_type i = pos;

		while ((i < buckets) && info_container[i].is_normal()) ++i;  // find a place to insert.
#if defined(REPROBE_STAT)
		size_t reprobe = i - pos;
#endif
		if (i == buckets) {
			// did not find one, so search from beginning
			i = 0;
			while ((i < pos) && info_container[i].is_normal()) ++i;  // find a place to insert.
#if defined(REPROBE_STAT)
			reprobe += i;
#endif
			if (i == pos) // nothing was found.  this should not happen
				throw std::logic_error("ERROR: did not find any place to insert.  should not have happend");

			// else 0 <= i < pos
		} // else  pos <= i < buckets.

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		// insert at i.
		container[i] = value;
		info_container[i].info = 0;   // higest 2 bits are already set.

		return 1;
	}

public:

// HERE:  TODO: change insert to remove reprobe distance and associated distance.
	/**
	 * @brief insert a single key-value pair into container.
	 */
	std::pair<iterator, bool> insert(key_type const & key, mapped_type const & val) {

#if defined(REPROBE_STAT)
		size_type reprobe = 0;
#endif

		if (buckets == 0) buckets = 1;

		// first check if we need to resize.
		if (lsize >= max_load) rehash(buckets << 1);

		// first get the bucket id
		size_type pos = hash(key) % buckets;
		size_type i;
		size_type insert_pos = buckets;

		for (i = pos; i < buckets; ++i) {
			if (info_container[i].is_empty()) {
				insert_pos = i;
				break;
			}

			if (info_container[i].is_deleted() && (insert_pos == buckets)) {
//				std::cout << " check deleted pos = " << pos << " buckets = " << buckets << " i = " << i << " insert_pot " << insert_pos << std::endl;

				insert_pos = i;

			} else if (info_container[i].is_normal() && (eq(key, container[i].first))) {

#if defined(REPROBE_STAT)
				this->reprobes += i - pos;
				this->max_reprobes = std::max(this->max_reprobes, (i-pos));
#endif
				// found a match
				return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), false);
			}
		}
//		std::cout << " first part insertion pos = " << pos << " buckets = " << buckets << " i = " << i << " insert_pot " << insert_pos << std::endl;
#if defined(REPROBE_STAT)
		reprobe = i-pos;
#endif

		if (i == buckets) {
			// now repeat for first part
			for (i = 0; i < pos; ++i) {
				if (info_container[i].is_empty()) {
					insert_pos = i;
					break;
				}

				if (info_container[i].is_deleted() && (insert_pos == buckets))
					insert_pos = i;
				else if (info_container[i].is_normal() && (eq(key, container[i].first))) {
#if defined(REPROBE_STAT)
					this->reprobes += reprobe + i;
					this->max_reprobes = std::max(this->max_reprobes, (reprobe + i));
#endif
					// found a match
					return std::make_pair(iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter), false);

				}
			}

#if defined(REPROBE_STAT)
			reprobe += i;
#endif
		}
#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		// finally check if we need to insert.
		if (insert_pos == buckets) {  // went through the whole container.  must be completely full.
			throw std::logic_error("ERROR: did not find a slot to insert into.  container must be full.  should not happen.");
		}

		container[insert_pos].first = key;
		container[insert_pos].second = val;
		info_container[insert_pos].info = 0;   // high bits are cleared.
		++lsize;

		return std::make_pair(iterator(container.begin() + insert_pos, info_container.begin()+ insert_pos, info_container.end(), filter), true);

	}

	std::pair<iterator, bool> insert(value_type const & vv) {
		return insert(vv.first, vv.second);
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
			std::tie(dummy, success) = insert(value_type(*it));  //local insertion.  this requires copy construction...

			if (success) ++count;
		}


#if defined(REPROBE_STAT)
		std::cout << "INSERT batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << count << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif
		reserve(lsize);  // resize down as needed
	}

	void insert(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

		size_t count = 0;

		for (size_t i = 0; i < input.size(); ++i) {
			if( insert(input[i]).second)  //local insertion.  this requires copy construction...
				++count;

		}


#if defined(REPROBE_STAT)
		std::cout << "INSERT batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
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

		if (buckets == 0) return 0;

		// first get the bucket id
		size_t pos = hash(k) % buckets;

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (info_container[i].is_empty())  // done
				break;

			if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
				this->reprobes += i - pos;
				this->max_reprobes = std::max(this->max_reprobes, (i-pos));
#endif
				return 1;
			}
		}  // ends when i == buckets or empty node.
#if defined(REPROBE_STAT)
		reprobe = i - pos;
#endif

		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (info_container[i].is_empty())  // done
					break;

				if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
					this->reprobes += reprobe + i;
					this->max_reprobes = std::max(this->max_reprobes, (reprobe + i));
#endif
					return 1;
				}
			}  // ends when i == buckets or empty node.
#if defined(REPROBE_STAT)
			reprobe += i;
#endif
		}
#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		// if we are here, then we did not find it.  return 0.
		return 0;
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
		std::cout << "COUNT batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
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
		std::cout << "COUNT batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
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

		if (buckets == 0) return iterator(container.end(), info_container.end(), filter);

		// first get the bucket id
		size_t pos = hash(k) % buckets;

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (info_container[i].is_empty())  // done
				break;

			if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
				this->reprobes += i - pos;
				this->max_reprobes = std::max(this->max_reprobes, (i-pos));
#endif
				return iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter);
			}
		}  // ends when i == buckets or empty node.
#if defined(REPROBE_STAT)
		reprobe = i-pos;
#endif
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (info_container[i].is_empty())  // done
					break;

				if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
					this->reprobes += reprobe + i;
					this->max_reprobes = std::max(this->max_reprobes, (reprobe + i));
#endif
					return iterator(container.begin() + i, info_container.begin()+ i, info_container.end(), filter);
				}
			}  // ends when i == buckets or empty node.

#if defined(REPROBE_STAT)
			reprobe += i;
#endif
		}
#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		// if we are here, then we did not find it.  return  end iterator.
		return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		if (buckets == 0) return const_iterator(container.cend(), info_container.cend(), filter);

		// first get the bucket id
		size_t pos = hash(k) % buckets;

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif

		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (info_container[i].is_empty())  // done
				break;

			if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
				this->reprobes += i - pos;
				this->max_reprobes = std::max(this->max_reprobes, (i-pos));
#endif
				return const_iterator(container.cbegin() + i, info_container.cbegin() + i, info_container.cend(), filter);
			}
		}  // ends when i == buckets or empty node.
#if defined(REPROBE_STAT)
		reprobe = i-pos;
#endif
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (info_container[i].is_empty())  // done
					break;

				if (info_container[i].is_normal() && (eq(k, container[i].first))) {
#if defined(REPROBE_STAT)
					this->reprobes += reprobe + i;
					this->max_reprobes = std::max(this->max_reprobes, (reprobe + i));
#endif

					return const_iterator(container.cbegin() + i, info_container.cbegin() + i, info_container.cend(), filter);
				}
			}  // ends when i == buckets or empty node.
#if defined(REPROBE_STAT)
			reprobe += i;
#endif
			}
#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

		// if we are here, then we did not find it.  return 0.
		return const_iterator(container.cend(), info_container.cend(), filter);

	}



	  template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
	    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	  std::vector<value_type> find(Iter begin, Iter end) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

	    size_t total = std::distance(begin, end);

	    iterator map_end = this->end();
	    iterator found;

	    ::std::vector<value_type> counts;
	    counts.reserve(total);

	    // iterate based on size between rehashes
	    for (Iter it = begin; it != end; ++it) {

	      // === same code as in insert(1)..
	      found = find((*it).first);
	      if (found != map_end) counts.emplace_back(*found);
	    }

#if defined(REPROBE_STAT)
		std::cout << "FIND ITER:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << counts.size() << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif



	    return counts;
	  }


	  template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
	    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	  std::vector<value_type> find(Iter begin, Iter end) {
#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif

	    size_t total = std::distance(begin, end);

	    iterator map_end = this->end();
	    iterator found;

	    ::std::vector<value_type> counts;
	    counts.reserve(total);

	    // iterate based on size between rehashes
	    for (Iter it = begin; it != end; ++it) {

	      // === same code as in insert(1)..
	      found = find(*it);
	      if (found != map_end) counts.emplace_back(*found);
	    }

#if defined(REPROBE_STAT)
		std::cout << "FIND ITER:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
					"\tvalid=" << counts.size() << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
#endif


		    return counts;
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
	 * @brief erases a key.
	 */
	size_type erase_no_resize(key_type const & k) {

		if (buckets == 0) return 0;

		// first get the bucket id
		size_t pos = hash(k) % buckets;

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (info_container[i].is_empty()) // done
				break;

			if (info_container[i].is_normal() && (eq(k, container[i].first))) {

				info_container[i].set_deleted();

				--lsize;
#if defined(REPROBE_STAT)
				this->reprobes += i - pos;
				this->max_reprobes = std::max(this->max_reprobes, (i - pos));
#endif
				return 1;
			}
		}  // ends when i == buckets or empty node.
//		std::cout << "first: pos = " << pos << "\ti = " << i << std::endl;
#if defined(REPROBE_STAT)
		reprobe = i-pos;
#endif
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (info_container[i].is_empty())  // done
					break;

				if (info_container[i].is_normal() && (eq(k, container[i].first))) {

					info_container[i].set_deleted();

					--lsize;

#if defined(REPROBE_STAT)
					this->reprobes += reprobe + i;
					this->max_reprobes = std::max(this->max_reprobes, (reprobe + i));
#endif
					return 1;
				}
			}  // ends when i == buckets or empty node.
//			std::cout << "second: pos = " << pos << "\ti = " << i << std::endl;

#if defined(REPROBE_STAT)
			reprobe += i;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		// if we are here, then we did not find it.  return 0.
		return 0;

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
		std::cout << "ERASE batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
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
		std::cout << "ERASE batch:\treprobe max=" << this->max_reprobes << "\treprobe total=" << this->reprobes <<
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
#endif /* KMERHASH_HASHMAP_LINEARPROBE_HPP_ */
