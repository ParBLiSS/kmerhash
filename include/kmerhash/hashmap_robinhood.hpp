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

#ifndef KMERHASH_HASHMAP_ROBINHOOD_HPP_
#define KMERHASH_HASHMAP_ROBINHOOD_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"

#include "io/incremental_mxx.hpp"

namespace fsc {

/**
 * @brief Open Addressing hashmap that uses lienar addressing with robin hood hashing strategy, with doubling for reallocation, and with specialization for k-mers.
 * @details
 * @details
 *    at the moment, nothing special for k-mers yet.
 *    This class has the following implementation characteristics
 *      vector of structs
 *      open addressing with robin hood hashing
 *      doubling for reallocation
 *      array of tuples
 *      circular array for storage.
 *
 *
 *   Implementation Details
 *      using vector internally.
 *      tracking empty via special bit instead of using empty key (potentially has to compare whole key).
 *      No separate key for deleted as we perform backward shift to remove deleted slots.
 *        use byte array instead of storing flags with value type
 *        Q: should the key and value be stored in separate arrays?  count, exist, and erase may be faster (no need to touch value array).
 *
 *      MPI friendliness - would never send the vector directly - need to permute, which requires extra space and time, and only send value when insert (and possibly update),
 *        neither are technically sending the vector really.
 *
 *        so assume we always are copying then sending, which means we can probably construct as needed.
 *
 *    WANT a simple array of key, value pair.  then have auxillary arrays to support scanning through, ordering, and reordering.
 *      memorizing hashes can be done as well in aux array....
 *
 *      somethings to put in the aux array:  empty bit.  deleted bit.  probe distance.
 *        memorizing the hash requires more space, else would need to compute hash anyways.
 *
 *   tuple of arrays is probably not a good idea.
 *      checking for key equality is the only reason for doing that
 *      at the cost of constructing pairs all the time, and having iterators that are zip iterators so harder to "update" values.
 *
 *
 *  requirements
 *    [x] support iterate
 *    [x] no special keys
 *    [ ] incremental allocation (perhaps not doubling right now...
 *
 *
 *  how to order the hash table accesses so that there is maximal cache reuse?  order things by hash?
 *
 *
 *  MAYBE:
 *      hashset may be useful for organizing the query input.
 *      when using power-of-2 sizes, using low-to-high hash value bits means that when we grow, a bucket is split between buckets i and i+2^x.
 *      when rehashing under doubling/halving, since always power of 2 in capacity, can just merge the higher half to the lower half by index.
 *        if we use high-to-low bits, then resize up split into adjacent buckets.
 *        it is possible that some buckets may not need to move.   It is possible to grow incrementally potentially.
 *      using a C array may be better - better memory allocation control.
 *
 *  TODO:
 *  [x] remove max_probe - treat as if circular array.
 *  [x] separate info from rest of struct, so as to remove use of transform_iterator, thus allowing us to change values through iterator. requires a new aux_filter_iterator.
 *  [ ] batch mode operations to allow more opportunities for optimization including SIMD
 *  [ ] predicated version of operations
 *  [ ] macros for repeated code.
 *  [x] testing with k-mers
 *  [x] measure reprobe count for insert
 *  [x] measure reprobe count for find and update.
 *  [x] measure reprobe count for count and erase.
 *  [x] robin hood hashing
 *  [x] backward shifting during deletion.
 *  [x] no memmove.  current approach swaps only when dist increases, not when they are the same (in a bucket), so total mem access would be low.
 *
 *  Robin Hood Hashing logic follows
 *  	http://www.sebastiansylvan.com/post/robin-hood-hashing-should-be-your-default-hash-table-implementation/
 *
 *  use an auxillary array.
 *  where aux array element represents the distance from source for the corresponding element in container.
 *      idx   ----a---b------------
 *    aux   ===|-|=|-|=|4|3|=====   _4_ is recorded at position hash(X) + _4_ of aux array.
 *    data  -----------|X|Y|-----,  X is inserted into bucket hash(X) = a.  in container position hash(X) + _4_.
 *
 *    empty positions are set to 0x80, and same for all info entries.
 *
 *  We linear scan info_container from position hash(Y) and must go to hash(Y) + 4 linearly.
 *    each aux entry is essentailyl independent of others.
 *    have to check and compare each data entry.
 *
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_robinhood_doubling {

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

    	inline size_t get_offset() const {
    		return info & dist_mask;
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
    mutable size_t mask;
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
    mutable size_t moves;
    mutable size_t max_moves;

public:

    /**
     * _capacity is the number of usable entries, not the capacity of the underlying container.
     */
	explicit hashmap_robinhood_doubling(size_t const & _capacity = 128,
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
	hashmap_robinhood_doubling(Iter begin, Iter end,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
		hashmap_robinhood_doubling(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

		for (auto it = begin; it != end; ++it) {
			insert(value_type(*it));  //local insertion.  this requires copy construction...
		}
	}

	~hashmap_robinhood_doubling() {
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
		print_raw();
	}

	void print_raw() const {
		std::cout << "lsize " << lsize << "\tbuckets " << buckets << "\tmax load factor " << max_load_factor << std::endl;
		size_type i = 0;

		for (i = 0; i < buckets; ++i) {
			std::cout <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i].info) <<
					", off: " << std::setw(3) << static_cast<size_t>(info_container[i].get_offset()) <<
					", pos: " << std::setw(10) << ((i + mask - info_container[i].get_offset()) & mask) <<
					"\n" << std::setw(62) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}

		for (i = buckets; i < info_container.size(); ++i) {
			std::cout <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i].info) <<
					", off: " << std::setw(3) << static_cast<size_t>(info_container[i].get_offset()) <<
					", pos: " << std::setw(10) << ((i + mask - info_container[i].get_offset()) & mask) <<
					"\n" << std::setw(62) << i <<
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
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i].info) <<
					", off: " << std::setw(3) << static_cast<size_t>(info_container[i].get_offset()) <<
					", pos: " << std::setw(10) << ((i + mask - info_container[i].get_offset()) & mask) <<
					"\n" << std::setw(62) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}
	}

	void print(size_t const & first, size_t const &last, std::string prefix) const {
		print_raw(first, last, prefix);
	}

//
//	void print() const {
//		std::cout << "buckets " << buckets << " lsize " << lsize << " max load factor " << max_load_factor << std::endl;
//
//		for (size_t i = 0; i < buckets; ++i) {
//			std::cout << i << ": [" << container[i].first << "->" <<
//					container[i].second << "] info " <<
//					static_cast<size_t>(info_container[i].info) <<
//					" offset = " <<
//					static_cast<size_t>(info_container[i].get_offset()) <<
//					" pos = " <<
//					(info_container[i].get_offset() + i) << std::endl;
//		}
//	}

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

//    std::cout << std::endl << "REHASH " << b << std::endl;

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

//	    std::cout << "REHASH DONE " << b << std::endl;

	}



protected:
	/**
	 * @brief inserts a range into the current hash table.
	 */
	void copy(typename container_type::iterator begin, typename container_type::iterator end, typename info_container_type::iterator info_begin) {


#if defined(REPROBE_STAT)
    size_t count = 0;
		this->reprobes = 0;
		this->moves = 0;
		this->max_reprobes = 0;
		this->max_moves = 0;
    iterator dummy;
    bool success;
#endif

		auto it = begin;
		auto iit = info_begin;
		for (; it != end; ++it, ++iit) {
			if ((*iit).is_normal()) {
#if defined(REPROBE_STAT)
				std::tie(dummy, success) = insert(*it);
				if (success) ++count;
#else
				insert(*it);
#endif
			}
		}

#if defined(REPROBE_STAT)
		std::cout << "REHASH copy:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
	    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << count << "\ttotal=" << std::distance(begin, end) << "\tlsize=" << lsize <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->moves = 0;
		this->max_reprobes = 0;
		this->max_moves = 0;
#endif
	}



public:



	/**
	 * @brief insert a single key-value pair into container.
	 *
	 * note that swap only occurs at bucket boundaries.
	 */
	std::pair<iterator, bool> insert(value_type const & v) {

		unsigned char reprobe = info_type::normal;
#if defined(REPROBE_STAT)
		unsigned char probe_count = 0;
		size_t move_count = 0;
#endif
		// first check if we need to resize.
		if (lsize >= max_load) rehash(buckets << 1);

		value_type vv = v;

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

//			std::cout << " iter " << j << " reprobe " << static_cast<size_t>(reprobe) <<
//					" id " << i << " info " << static_cast<size_t>(info_container[i].info) << std::endl;

			if (reprobe > info_container[i].info) {
				::std::swap(info_container[i].info, reprobe);

				if (insert_pos == std::numeric_limits<size_type>::max()) {
					insert_pos = i;  // set insert_pos to first positin of insertion.
					success = true;
					++lsize;
#if defined(REPROBE_STAT)
					probe_count = j;
				} else {
					// subsequent "shifts" do not change insert_pos or success.
					++move_count;
#endif
				}

				// then decide what to do given the swapped out distance.
				if (reprobe == info_type::empty) {  // previously it was empty,
					// then we can simply set.
					container[i] = vv;
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
#if defined(REPROBE_STAT)
					probe_count = j;
#endif
					break;
				}  // note that the loop terminates within here so don't need to worry about another insertion point.

			}  // note that it's not possible for a match to occur with shorter current reprobe distance -
				// that would mean same element is hashed to multiple buckets.

			// increment probe distance from assigned slot.
			++reprobe;

			// circular array.
			i = (i+1) & mask;   // for bucket == 2^x, this acts like modulus.
		}
		assert(j < buckets);

#if defined(REPROBE_STAT)
		this->reprobes += probe_count;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(probe_count));

		this->moves += move_count;
		this->max_moves = std::max(this->max_moves, move_count);
#endif
		return std::make_pair(iterator(container.begin() + insert_pos, info_container.begin()+ insert_pos, info_container.end(), filter), success);

	}

	std::pair<iterator, bool> insert(key_type const & key, mapped_type const & val) {
		auto result = insert(std::make_pair(key, val));
		return result;
	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	void insert(Iter begin, Iter end) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
		size_t count = 0;
		iterator dummy;
		bool success;
#endif

//		size_t i = 0;
		for (auto it = begin; it != end; ++it) { //, ++i) {
#if defined(REPROBE_STAT)
			std::tie(dummy, success) =
#endif
//					std::cout << " element " << i;
					insert(value_type(*it));  //local insertion.  this requires copy construction...

#if defined(REPROBE_STAT)
			if (success) {
				++count;
			}
#endif
		}

#if defined(REPROBE_STAT)
    std::cout << "lsize " << lsize << std::endl;

    std::cout << "INSERT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << count << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
#endif
		reserve(lsize);  // resize down as needed
	}



	/// batch insert not using iterator
	void insert(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
		size_t count = 0;
		iterator dummy;
		bool success;
#endif

		for (size_t i = 0; i < input.size(); ++i) {
#if defined(REPROBE_STAT)
			std::tie(dummy, success) =
#endif
//			std::cout << " element " << i;
			insert(input[i]);   //local insertion.  this requires copy construction...

#if defined(REPROBE_STAT)
			if (success) {
				++count;
			}
#endif
		}

#if defined(REPROBE_STAT)
    std::cout << "lsize " << lsize << std::endl;

    std::cout << "INSERT batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << count << "\ttotal=" << input.size() <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
#endif
		reserve(lsize);  // resize down as needed
	}


	/// batch insert not using iterator
	void insert_integrated(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
		unsigned char probe_count = 0;
		size_t move_count = 0;
		size_t before = lsize;
#endif
		unsigned char reprobe = info_type::normal;
		size_t id;
		size_type j;

		value_type vv;
		size_type insert_pos = std::numeric_limits<size_type>::max();

		for (size_t i = 0; i < input.size(); ++i) {

			reprobe = info_type::normal;
#if defined(REPROBE_STAT)
			probe_count = std::numeric_limits<unsigned char>::max();
			move_count = 0;
#endif

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			vv = input[i];
			insert_pos = std::numeric_limits<size_type>::max();

			// first get the bucket id
			id = hash(vv.first) & mask;  // target bucket id.

			for (j = 0; j < buckets; ++j, ++reprobe) {  // limit to 1 complete loop

				// if current distance is larger than target's distance, (including empty cell), then swap.
				assert(reprobe >= info_type::normal);  // if distance is over 128, then wrap around would get us less than normal.
//				std::cout << " element i " << i << " iter " << j << " reprobe " << static_cast<size_t>(reprobe) <<
//						" id " << id << " info " << static_cast<size_t>(info_container[id].info) << std::endl;

				if (reprobe > info_container[id].info) {

					::std::swap(info_container[id].info, reprobe);
					if (insert_pos == std::numeric_limits<size_type>::max()) {
						insert_pos = id;  // set insert_pos to first positin of insertion.
						++lsize;
#if defined(REPROBE_STAT)
						probe_count = j;
					} else {
						// subsequent "shifts" do not change insert_pos or success.
						++move_count;
#endif
					}

					// then decide what to do given the swapped out distance.
					if (reprobe == info_type::empty) {  // previously it was empty,
						// then we can simply set.
						container[id] = vv;
						break;

					} else {
						// there was a real entry, so need to swap
						::std::swap(container[id], vv);
						// and continue
					}

				} else if (reprobe == info_container[id].info) {
				  // check for equality, only if haven't inserted (if don't check success, there could be a lot of equality checks.

	//			  if (success)  std::cout << ".";

					// same distance, then possibly same value.  let's check.
					if (eq(container[id].first, vv.first)) {  // note that a previously swapped vv would not match again and can be skipped.
						// same, then we found it and need to return.
#if defined(REPROBE_STAT)
						probe_count = j;
#endif
						break;
					}  // note that the loop terminates within here so don't need to worry about another insertion point.

				}  // note that it's not possible for a match to occur with shorter current reprobe distance -
					// that would mean same element is hashed to multiple buckets.

				// circular array.
				id = (id+1) & mask;   // for bucket == 2^x, this acts like modulus.
			}
			assert(j < buckets);

	#if defined(REPROBE_STAT)
			this->reprobes += probe_count;
			this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(probe_count));

			this->moves += move_count;
			this->max_moves = std::max(this->max_moves, move_count);
	#endif

		}

#if defined(REPROBE_STAT)
    std::cout << "lsize " << lsize << std::endl;

    std::cout << "INSERT_I batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << (lsize - before) << "\ttotal=" << input.size() <<
					"\tbuckets=" << buckets <<std::endl;
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
#endif


//		reserve(lsize);  // resize down as needed
	}




	/// batch insert using sorting.  This is about 4x slower on core i5-4200U (haswell) than integrated batch insertion above, even just for sort.
	template <typename LESS = ::std::less<key_type> >
	void insert_sort(::std::vector<value_type> & input) {

		std::cerr << "sort adds a stable sort, but otherwise same as integrated." << std::endl;

//		assert(lsize == 0);   // currently only works with empty hashmap.
//
//#if defined(REPROBE_STAT)
//		this->reprobes = 0;
//		this->max_reprobes = 0;
//		this->moves = 0;
//		this->max_moves = 0;
//		size_t total = input.size();
//#endif
		LESS less;

		// overestimate the size first.
		size_t local_mask = next_power_of_2(static_cast<size_t>(static_cast<float>(input.size() + this->lsize) / this->max_load_factor)) - 1;

		// first sort by hash value.  two with same hash then are compared by key.  this last part is needed
		// for the unique.  stable sort to ensure the right entry is kept.
		std::stable_sort(input.begin(), input.end(),
				[this, &local_mask, &less](value_type const & x, value_type const & y){
			size_t id_x = this->hash(x.first) & local_mask;
			size_t id_y = this->hash(y.first) & local_mask;

			return (id_x < id_y) || ( (id_x == id_y) && less(x.first, y.first) );
		});

#if 0
// THIS HAS ISSUES.  in particular, it needs estimated size, correct merge, and insert code is incorrect.
// DO NOT USE. RIGHT NOW.
		// then do unique by key_value, so we have an accurate count
		// THIS IS NOT SUITABLE FOR COUNT OR REDUCTION IN GENERAL.
		auto new_end = std::unique(input.begin(), input.end(),
				[this](value_type const & x, value_type const & y){
			return this->eq(x.first, y.first);
		});
		input.erase(new_end, input.end());

		// now allocate for the new size.
		// TODO:  we need at least same number as the number of unique entries.
		//        if there are existing entries, then we've overprovisioned.
		//		  but that may be okay?
		//		  alternatively, insert into a local hash, then merge
		reserve(input.size() + lsize);  // include current size.  overestimated. but input is now smaller.

		auto merge_comparator = [this, &less](value_type const & x, value_type const & y){
			size_t id_x = this->hash(x.first) & this->mask;
			size_t id_y = this->hash(y.first) & this->mask;

			return (id_x < id_y) || ( (id_x == id_y) && less(x.first, y.first) );
		};

		// do one scan, and get all the break points (at (& mask) == 0).
		// local mask should always be >= mask, since only input.size is possibly reduced.
		if (local_mask > this->mask) { // if original size is larger than target size, then need to condense.
			// first get the splitters.
			std::vector<typename ::std::vector<value_type>::iterator> iterators;
			auto it = input.begin();
			iterators.push_back(it);
			++it;
			for (; it !=  input.end(); ++it) {
				if ((hash(it->first) & mask) == 0) {
					// wrapped around again.
					iterators.push_back(it);
				}
			}
			iterators.push_back(input.end());

			// next merge.  recall that lower k bits are same every 2^k buckets
			// also stopping has to be at mask count.  should not need to compare when merging.
			::std::vector<value_type> sorted_input(input.size());
			while (iterators.size() > buckets) {
				// merge pairwise.
				for (size_t j = 0; j < iterators.size() / 2; ++j) {
					// this merge is not right, should be merging buckets in steps of "buckets"
					std::inplace_merge(iterators[j], iterators[j+1], iterators[j+2], merge_comparator);
				}

				// remove every other entry from iterators.
				std::vector<typename ::std::vector<value_type>::iterator> new_iters;
				for (size_t j = 0; j < iterators.size() - 1; j+=2) {
					new_iters.push_back(iterators[j]);
				}
				new_iters.push_back(iterators.back());

				iterators.swap(new_iters);
			}
		} else if ( local_mask < this->mask) {
			// should not happen
		} else {
			// same.  so buckets already matched up....
		}

		// set up the hash table, using only unique entries (by key value)  THIS PART PROBABLY IS CAUSING SEGV.
		std::fill(info_container.begin(), info_container.end(), info_type::empty);
		size_t id = 0;
		size_t pos = 0;
		size_t last_id = std::numeric_limits<size_t>::max();
		unsigned char dist = info_type::normal;
//		unsigned char last_dist = info_type::empty;
		for (size_t i = 0; i < input.size(); ++i) {
			id = hash(input[i].first) & mask;

			if (id == last_id) {
				// same bucket, so increment the position by 1.
				++pos;
				++dist;
			} else {
				// recall this is sorted, so "else" implies id > last_id
				if (id > pos) {  // write to future position, no dependency to last pos.
					pos = id;
					dist = info_type::normal;

				} else if (id == pos) {  // write to current position, which must be occupied, so need to increment by 1.
					++pos;
					dist = info_type::normal + 1;
					// this fails at first position
				} else {  // a previous bucket.
					++pos;
					dist = info_type::normal + (pos - id);
				}


				last_id = id;

			}
				info_container[pos] = dist;
				container[pos] = input[i];
		}
		 lsize += input.size();

#else
		insert_integrated(input);

#endif

//
//
//#if defined(REPROBE_STAT)
//    std::cout << "lsize " << lsize << std::endl;
//
//    std::cout << "INSERT_I batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
//    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
//					"\tvalid=" << input.size() << "\ttotal=" << total <<
//					"\tbuckets=" << buckets <<std::endl;
//		this->reprobes = 0;
//		this->max_reprobes = 0;
//		this->moves = 0;
//		this->max_moves = 0;
//#endif
//
//		reserve(lsize);  // resize down as needed
	}




	/// batch insert using extra memory and do shuffling
	void insert_shuffled(::std::vector<value_type> & input) {

		insert_integrated(input);
		std::cerr << "shuffle not implemented" << std::endl;

//		assert(lsize == 0);   // currently only works with empty hashmap.
//
//#if defined(REPROBE_STAT)
//		this->reprobes = 0;
//		this->max_reprobes = 0;
//		this->moves = 0;
//		this->max_moves = 0;
//		size_t total = input.size();
//#endif
//
//		// NEED ESTIMATE.
//		size_t local_mask = next_power_of_2(static_cast<size_t>(static_cast<float>(input.size()) / this->max_load_factor)) - 1;
//		local_mask = ::std::max(local_mask, mask);
//
//		::std::vector<size_t> bucket_sizes;
//		bucket_sizes.reserve(local_mask);
//
//		// first compute hash values
//		// now shuffle to bucket them together.
//		::imxx::local::bucketing_impl(input, [this, &local_mask](value_type const & x){
//			return this->hash(x.first) & local_mask;
//		}, local_mask, bucket_sizes);
//
//
//		// then insert
//		insert_integrated(input);
//
//
//#if defined(REPROBE_STAT)
//    std::cout << "lsize " << lsize << std::endl;
//
//    std::cout << "INSERT_I batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
//    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
//					"\tvalid=" << input.size() << "\ttotal=" << total <<
//					"\tbuckets=" << buckets <<std::endl;
//		this->reprobes = 0;
//		this->max_reprobes = 0;
//		this->moves = 0;
//		this->max_moves = 0;
//#endif
//
//		reserve(lsize);  // resize down as needed
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
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
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



	  template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
	    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	  std::vector<value_type> find(Iter begin, Iter end) {

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
#endif


	    size_t total = std::distance(begin, end);

	    ::std::vector<value_type> counts;
	    counts.reserve(total);

	    size_type id;

	    // iterate based on size between rehashes
	    for (Iter it = begin; it != end; ++it) {

	      // === same code as in insert(1)..
	      id = find_pos((*it).first);
	      if (id < buckets) counts.emplace_back(container[id]);
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

	    ::std::vector<value_type> counts;
	    counts.reserve(total);

	    size_type id;

	    // iterate based on size between rehashes
	    for (Iter it = begin; it != end; ++it) {

	      // === same code as in insert(1)..
	      id = find_pos(*it);
	      if (id < buckets) counts.emplace_back(container[id]);
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
		size_t move_count = 0;

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
				container[target] = container[curr];

				++move_count;

				// store the curr position as the next target of move.
				target = curr;
			}

			// advance
			curr = next;
			next = (next+1) & mask;

			next_info = info_container[next].info;

		}
		info_container[curr].info = info_type::empty;  // set last to empty.
		container[target] = container[curr];
		++move_count;

#if defined(REPROBE_STAT)
		this->moves += move_count;
		this->max_moves = std::max(this->max_moves, move_count);
#endif
		return 1;
	}

	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	size_type erase_no_resize(Iter begin, Iter end) {
		size_type erased = 0;

#if defined(REPROBE_STAT)
		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			erased += erase_no_resize((*it).first);
		}

#if defined(REPROBE_STAT)
		std::cout << "ERASE batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
	    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << erased << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
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
		this->moves = 0;
		this->max_moves = 0;
#endif

		for (auto it = begin; it != end; ++it) {
			erased += erase_no_resize(*it);
		}

#if defined(REPROBE_STAT)
		std::cout << "ERASE batch:\treprobe max=" << static_cast<unsigned int>(this->max_reprobes) << "\treprobe total=" << this->reprobes <<
	    		"\tmove max=" << static_cast<unsigned int>(this->max_moves) << "\tmove total=" << this->moves <<
					"\tvalid=" << erased << "\ttotal=" << ::std::distance(begin, end) <<
					"\tbuckets=" << buckets <<std::endl;

		this->reprobes = 0;
		this->max_reprobes = 0;
		this->moves = 0;
		this->max_moves = 0;
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
		if (lsize < min_load) reserve(lsize);

		return erased;
	}


};

}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_HPP_ */
