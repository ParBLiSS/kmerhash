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
 * hashtable_OA_RH_prefetch.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_HASHMAP_ROBINHOOD_PREFETCH_HPP_
#define KMERHASH_HASHMAP_ROBINHOOD_PREFETCH_HPP_

#include <vector>   // for vector.
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include <cstdlib>   // aligned_alloc

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"

#include "io/incremental_mxx.hpp"

#include "kmerhash/hyperloglog64.hpp"  // for size estimation.


#include <mmintrin.h>
#include <xmmintrin.h>


namespace fsc {

#define LOOK_AHEAD 16

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
 *  [ ] simplified info_type
 *  [ ] change insert return type
 *  [ ] refactor for separate upsize and downsize, where downsize does not require hashing
 *  [ ] change upsize and downsize during rehash to better use spatial locality (insert adjacent source entries should induce some spatial locality).
 *  [ ] change to no wrap around.
 *  [ ] refactor so to avoid checking for rehashing during each iteration of batch insertion - break up into multiple inserts, each up to current size.
 *  [ ] prefetching insertion of a range of data.
 *  [ ] try erase without compact, then compact in 1 pass.  however, erase without compact may take more time....
 *
 *  [x] resize when reprobe distance is greater than 127 during insert_with_hint
 *  [x] resize when reprobe distance is greater than 127 during insert_integrated
 *  [ ] resize when resize down and reprobe distance is greater than 127. - can only throw
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
class hashmap_robinhood_prefetch {

public:

    using key_type              = Key;
    using mapped_type           = T;
    using value_type            = ::std::pair<Key, T>;
    using hasher                = Hash;
    using key_equal             = Equal;

protected:

    //=========  start INFO_TYPE definitions.

    // info_type, and helpers for it.
    using info_type = uint8_t;

	static constexpr info_type info_empty = 0;
	static constexpr info_type info_normal = 0x80;   // this is used to initialize the reprobe distances.

	inline bool is_empty(info_type const & x) const {
		return x < info_normal;
	}
	inline bool is_normal(info_type const & x) const {
		return x >= info_normal;
	}
	inline info_type get_distance(info_type const & x) const {
		assert(is_normal(x));   // should not call with empty entry.
		return x & 0x7F;
	}
	//=========  end INFO_TYPE definitions.

    // filter
    struct valid_entry_filter {
    	inline bool operator()(info_type const & x) { return x >= info_normal; };
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
    // difference type should be signed.
    //  use the sign bit for indication of success/failure of insertion.
    // failed insert (MSB set) and successful find have same meaning.
    //  remaining bits indicate insertion position.
    using bucket_id_type = size_t;
    static constexpr bucket_id_type bid_mask = ~(static_cast<bucket_id_type>(0)) >> 1;   // lower bits set.
    static constexpr bucket_id_type bid_exists = 1UL << 63;
    // failed is speial, correspond to all bits set (max distnace failed).  not using 0x800000... because that indicates failed inserting due to occupied.
    static constexpr bucket_id_type bid_failed = ~(static_cast<bucket_id_type>(0));

	inline bool exists(bucket_id_type const & x) const {
		return x > bid_mask;
	}
  inline bool missing(bucket_id_type const & x) const {
    return x <= bid_mask;
  }
	inline bucket_id_type get_bucket_id(bucket_id_type const & x) const {
		return x & bid_mask;
	}
	inline bucket_id_type mark_as_existing(bucket_id_type const & x) const {
		return x | bid_exists;
	}
    //=========  end BUCKET_ID_TYPE definitions.


    size_type lsize;
    mutable size_type buckets;
    mutable size_type mask;
    mutable size_type min_load;
    mutable size_type max_load;
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
	explicit hashmap_robinhood_prefetch(size_type const & _capacity = 128,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
			lsize(0), buckets(next_power_of_2(_capacity) ), mask(buckets - 1),
#if defined (REPROBE_STAT)
			upsize_count(0), downsize_count(0),
#endif
			container(buckets + std::numeric_limits<info_type>::max() + 1),
			info_container(buckets + std::numeric_limits<info_type>::max() + 1, info_empty)
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
	hashmap_robinhood_prefetch(Iter begin, Iter end,
			float const & _min_load_factor = 0.4,
			float const & _max_load_factor = 0.9) :
		hashmap_robinhood_prefetch(::std::distance(begin, end) / 4, _min_load_factor, _max_load_factor) {

		insert(begin, end);
	}

	~hashmap_robinhood_prefetch() {
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
		min_load = static_cast<size_type>(static_cast<float>(buckets) * min_load_factor);

	}

	inline void set_max_load_factor(float const & _max_load_factor) {
		max_load_factor = _max_load_factor;
		max_load = static_cast<size_type>(static_cast<float>(buckets) * max_load_factor);
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

	size_type capacity() {
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
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (is_normal(info_container[i]) ? (i - get_distance(info_container[i])) : -i) <<
					"\n" << std::setw(62) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}

		for (i = buckets; i < info_container.size(); ++i) {
			std::cout <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (is_normal(info_container[i]) ? (i - get_distance(info_container[i])) : -i) <<
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
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_distance(info_container[i])) <<
					", pos: " << std::setw(10) << (is_normal(info_container[i]) ? (i - get_distance(info_container[i])) : -i) <<
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

//	void print() const {
//		for (size_type i = 0; i < info_container.size(); ++i) {
//			std::cout << i << ": [" << container[i].first << "->" <<
//					container[i].second << "] info " <<
//					static_cast<size_t>(info_container[i]) <<
//					" bucket = ";
//			if (is_normal(info_container[i])) {
//				std::cout << static_cast<size_t>(i - get_distance(info_container[i]));
//			} else {
//				std::cout << static_cast<int64_t>(-i);
//			}
//			std::cout << std::endl;
//		}
//	}

	std::vector<value_type > to_vector() const {
		std::vector<value_type > output(lsize);

		std::copy(this->cbegin(), this->cend(), output.begin());

		return output;
	}

	std::vector<key_type > keys() const {
		std::vector<key_type > output(lsize);

		std::transform(this->cbegin(), this->cend(), output.begin(),
				[](value_type const & x){ return x.first; });

		return output;
	}


	size_type size() const {
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
	 * @brief reserve space for specified entries (below max load factor)
	 */
  void reserve(size_type n) {
//    if (n > this->max_load) {   // if requested more than current max load, then we need to resize up.
      rehash(static_cast<size_type>(static_cast<float>(n) / this->max_load_factor));
      // rehash to the new size.    current bucket count should be less than next_power_of_2(n).
//    }  // do not resize down.  do so only when erase.
  }

  /**
   * @brief reserve space for specified BUCKETS.
   * @details note that buckets > entries.
   */
  void rehash(size_type const & b) {
    // check it's power of 2
    size_type n = next_power_of_2(b);

    if ((n != buckets) && (lsize < (max_load_factor * n))) {

      container_type tmp(n + std::numeric_limits<info_type>::max() + 1);
      info_container_type tmp_info(n + std::numeric_limits<info_type>::max() + 1, info_empty);

      if (lsize > 0) {
        if (n > buckets) {
          this->copy_upsize(tmp, tmp_info, n);
    #if defined(REPROBE_STAT)
          ++upsize_count;
    #endif
        } else {
          this->copy_downsize(tmp, tmp_info, n);
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
  	   *  			iterating over source means that when writing into target, we will need to shift the entire target container, potentially.
  	   *
  	   *  			prob better to iterate over multiple partitions, so that target is filled sequentially.
  	   *
  	   *  			figure out the scaling factor, and create an array to track iteration position as we go.   note that each may need to be at diff points...
  	   *
  	   */
  	  void copy_downsize(container_type & target, info_container_type & target_info, size_type const & target_buckets) {
  		  size_type m = target_buckets - 1;
  		  assert((target_buckets & m) == 0);   // assert this is a power of 2.

  		  bucket_id_type id, last;
  		  last = 0;

  		  // iterate through the entire input.
  		  for (size_t i = 0; i < info_container.size(); ++i) {
  			  if (is_normal(info_container[i])) {
  				  // compute the new id
  				  // first part is original bucket, &m converts to new bucket.  &m is modulus.
  				  // mask & m == 0.  hash & mask & m == hash & m
  				  id = (static_cast<bucket_id_type>(i) - get_distance(info_container[i])) & m;

  				  last = insert_with_hint(target, target_info, id, last, container[i]);

  				  if (last == bid_failed) {
  				    throw std::logic_error("ERROR: max reprobe reached during downsizing");
  				  }

  			  }  // else is empty, so continue.
  		  }
  	  }


	/**
	 * @brief inserts a range into the current hash table.
	 * @details
	 * essentially splitting the source into multiple non-overlapping ranges.
	 * 		each partition is filled nearly sequentially, so figure out the scaling factor, and create an array as large as the scaling factor.
	 *
	 */
	void copy_upsize(container_type & target, info_container_type & target_info, size_type const & target_buckets) {
		  size_type m = target_buckets - 1;
		  assert((target_buckets & m) == 0);   // assert this is a power of 2.

		  // get the scale down factor.
		  bucket_id_type id, last;
		  last = 0;

  		  for (size_t i = 0; i < info_container.size(); ++i) {
  			  if (is_normal(info_container[i])) {
  				  // compute the new id via hash.
  				  id = hash(container[i].first) & m;

  				  last = insert_with_hint(target, target_info, id, last, container[i]);

            if (last == bid_failed) {
              throw std::logic_error("ERROR: max reprobe reached during upsizing");
            }


  			  }  // else is empty, so continue.
  		  }
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
	 */
	inline bucket_id_type insert_with_hint(container_type & target,
			info_container_type & target_info,
			bucket_id_type const & id, bucket_id_type const & last_insert_pos,
			value_type const & v) {

		assert(id != bid_failed);

		value_type vv = v;

		info_type reprobe = info_normal;
		info_type curr_info;

		bucket_id_type insert_pos = bid_failed;
		bucket_id_type i = get_bucket_id(id);
		bucket_id_type i_max = i + static_cast<bucket_id_type>(std::numeric_limits<info_type>::max() - info_normal) + 1;
		i_max = std::min(i_max, static_cast<bucket_id_type>(target_info.size()));

		// first take care of inserting vv, search for the first entry of the next bucket, reprobe > target_info.
		// if found duplicate before that or empty slot, then finish.
		// scan through at most max of info_type.
		for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.
			curr_info = target_info[i];

			if (is_empty(curr_info)) {
				// if current position is empty, then insert here and return.  no shifting is needed.
				target_info[i] = reprobe;
				target[i] = vv;

#if defined(REPROBE_STAT)
		this->reprobes += get_distance(reprobe);
		this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
#endif

				return i;  // success in insertion, and nothing to shift.

			} else if (reprobe > curr_info) {
				// if current position has less than reprobe dist, then swap
				::std::swap(target_info[i], reprobe);
				::std::swap(target[i], vv);

				// inserted, save position.  done with first loop.  next is to shift.
				insert_pos = i;

#if defined(REPROBE_STAT)
		this->reprobes += get_distance(target_info[i]);
		this->max_reprobes = std::max(this->max_reprobes, get_distance(target_info[i]));
#endif
				break;
			} else if (reprobe == curr_info) {
				// if current position is equal to reprobe, then check for equality.

				// same distance, then possibly same value.  let's check.
				// note that a previously swapped vv would not match again and can be skipped.
				if (eq(target[i].first, vv.first)) {
					// same, then we found it and need to return.
#if defined(REPROBE_STAT)
		this->reprobes += get_distance(reprobe);
		this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
#endif

					return mark_as_existing(i);   // indicate that this is a duplicate.
				}
			}  // else reprobe < target_info[i], increase reprobe and i and try again.

		}

		// kicks in only for third case, and no match.
		if (i == i_max) {
      std::cout << " id " << get_bucket_id(id) << " curr pos " << i << ", max " << i_max << " info size " << static_cast<bucket_id_type>(info_container.size()) << std::endl;
//      print();
//		  throw std::logic_error("ERROR: searching for insertion position reached max reprobe distance (128)");
		  // max reprobe reached.  return failed, resize and retry.

		  return bid_failed;
		}

		// if here, then swapped and breaked. reprobe and i need to advance 1.
		++reprobe;
		++i;

#if defined(REPROBE_STAT)
		size_t move_count = 0;
		bucket_id_type orig_i = i;
#endif

		// now shift.  note that since we are incrementing reprobe, consecutive entries for the same bucket
		// when shifting would have reprobe == target_info[i], until changing bucket.
		// therefore swap only occurs at bucket bundaries.
		// this MAY save some memory access for now, but story may be different we are bandwidth bound (and hardware prefetching becomes important).

		// this shift may go all the way to end of buckets.  no wrapping around.
		i_max = static_cast<bucket_id_type>(target_info.size());
		for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.

			curr_info = target_info[i];
			if (is_empty(curr_info)) {
				// if current position is empty, then insert here and stop.
				target_info[i] = reprobe;
				target[i] = vv;

#if defined(REPROBE_STAT)
				++move_count;
#endif

				break;
			} else if (reprobe > curr_info) {
				// if current position has less than reprobe dist (boundary), then swap
				::std::swap(target_info[i], reprobe);
				::std::swap(target[i], vv);

#if defined(REPROBE_STAT)
				++move_count;
#endif
				// continue until reaching an empty slot
			} // since we are just shifting and swapping at boundaries, in the else cluase we have reprobe == target_info[i] and never reprobe < target_info[i]
		      // we should also never get the case of duplicate entries,
		}

#if defined(REPROBE_STAT)
		this->moves += move_count;
		this->max_moves = std::max(this->max_moves, move_count);
		this->shifts += (i - orig_i);
		this->max_shifts = std::max(this->max_shifts, (i-orig_i));
#endif

		if (i == i_max) {
			std::cout << "i is " << i << " i_max at " << i_max << std::endl;
		  throw std::logic_error("ERROR: shifting entries to the right ran out of room in container.");
		}


		return insert_pos;

	}


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
		size_t h = hash(vv.first);

		bucket_id_type id = insert_with_hint(container, info_container, h & mask, h & mask, vv);
		while (id == bid_failed) {
	    	  std::cout << "RESIZE AND RE_INSERT: " << vv.first << std::endl;

		  rehash(buckets << 1);
		  id = insert_with_hint(container, info_container, h & mask, h & mask, vv);
		}
		bool success = missing(id);
		size_t bid = get_bucket_id(id);

		if (success) ++lsize;

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT 1", 1, (success ? 1 : 0));
#endif

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
		size_t h;
		bucket_id_type id;

		// iterate based on size between rehashes
		for (auto it = begin; it != end ; ++it) {

			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			// first get the bucket id
			h = hash((*it).first);  // target bucket id.

	    id = insert_with_hint(container, info_container, h & mask, h & mask, *it);
	    while (id == bid_failed) {
	    	  std::cout << "RESIZE AND RE_INSERT: " << it->first << std::endl;

	      rehash(buckets << 1);
	      id = insert_with_hint(container, info_container, h & mask, h & mask, *it);
	    }

			if (missing(id))
				++lsize;
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
		bucket_id_type id, id_la;

		// USE power of 2 look ahead...


		// prefetch.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), input.size());  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (input.size() > LOOK_AHEAD) ? (input.size() - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      for (size_t i = 0; i < block1_max; ++i) {
        // compute hash and store it.
        hashes[i] = hash(input[i].first);  // no need to prefetch here - sequential access.

        id = hashes[i] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_NTA);
        // adding a second does not help/
        // _mm_prefetch((const char *)&(container[id + (64 / sizeof(value_type))]), _MM_HINT_T0);

      }

		size_t hashes_mask = LOOK_AHEAD - 1;
		size_t h;

		// iterate based on size between rehashes
		for (size_t i = 0; i < block2_max; ++i) {

			// === same code as in insert(1)..

			// first check if we need to resize.
			if (lsize >= max_load) rehash(buckets << 1);

			// first get the bucket id
			hash_pos = i & hashes_mask;
			h = hashes[hash_pos];

      //=== prefetch
      hashes[hash_pos] = hash(input[i + LOOK_AHEAD].first);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_NTA);
      // adding a second does not help/
      //_mm_prefetch((const char *)&(container[id_la + (64 / sizeof(value_type))]), _MM_HINT_T0);

      // first get the bucket id
      id = insert_with_hint(container, info_container, h & mask, h & mask, input[i]);
      while (id == bid_failed) {
    	  std::cout << "RESIZE AND RE_INSERT: " << input[i].first << std::endl;
        rehash(buckets << 1);
        id = insert_with_hint(container, info_container, h & mask, h & mask, input[i]);
      }

      if (missing(id))
        ++lsize;

		}

		for (size_t i = block3_min; i < input.size(); ++i) {
      // === same code as in insert(1)..

      // first check if we need to resize.
      if (lsize >= max_load) rehash(buckets << 1);

      // first get the bucket id
      hash_pos = i & hashes_mask;
      h = hashes[hash_pos];

      // first get the bucket id
      id = insert_with_hint(container, info_container, h & mask, h & mask, input[i]);
      while (id == bid_failed) {
    	  std::cout << "RESIZE AND RE_INSERT: " << input[i].first << std::endl;

        rehash(buckets << 1);
        id = insert_with_hint(container, info_container, h & mask, h & mask, input[i]);
      }

      if (missing(id))
        ++lsize;
		}

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT VEC", input.size(), (lsize - before));
#endif


    free(hashes);

		// NOT needed until we are estimating reservation size.
//		reserve(lsize);  // resize down as needed
	}



  /// batch insert not using iterator
  void insert_integrated(std::vector<value_type> const & input) {

#if defined(REPROBE_STAT)
    reset_reprobe_stats();
    size_type before = lsize;
#endif
    bucket_id_type id, id_la;

    value_type vv;
    info_type reprobe;
    info_type curr_info;

    bucket_id_type i;
    bucket_id_type i_max;
    bool no_shift;


    // ================ PREFETCH.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), input.size());  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (input.size() > LOOK_AHEAD) ? (input.size() - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      for (size_t j = 0; j < block1_max; ++j) {
        // compute hash and store it.
        hashes[j] = hash(input[j].first);  // no need to prefetch here - sequential access.

        id = hashes[j] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_T0);
        // adding a second does not help/
        //_mm_prefetch((const char *)&(container[id + (64 / sizeof(value_type))]), _MM_HINT_NTA);

      }

      // =================  PROCESS MAIN BLOCK


      size_t hashes_mask = LOOK_AHEAD - 1;
      size_t h; // old hash value
      // ------------- INSERT

    // iterate based on size between rehashes

    for (size_t j = 0; j < block2_max; ++j) {

      // === same code as in insert(1)..

      // first check if we need to resize.
      if (lsize >= max_load) rehash(buckets << 1);

      // first get the bucket id
      hash_pos = j & hashes_mask;
      h = hashes[hash_pos];
      id = h & mask;  // target bucket id.

      //=== prefetch
      hashes[hash_pos] = hash(input[j + LOOK_AHEAD].first);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_T0);
      //_mm_prefetch((const char *)&(container[id_la + (64 / sizeof(value_type))]), _MM_HINT_NTA);

      //==== now do some work to insert.
      // first get the bucket id
      vv = input[j];
      no_shift = false;

      reprobe = info_normal;

      i = get_bucket_id(id);
      i_max = i + static_cast<bucket_id_type>(std::numeric_limits<info_type>::max() - info_normal) + 1;
      i_max = std::min(i_max, static_cast<bucket_id_type>(info_container.size()));

      // first take care of inserting vv, search for the first entry of the next bucket, reprobe > target_info.
      // if found duplicate before that or empty slot, then finish.
      // scan through at most max of info_type.
      for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.

        curr_info = info_container[i];

        if (is_empty(curr_info)) {
          // if current position is empty, then insert here and return.  no shifting is needed.
          info_container[i] = reprobe;
          container[i] = vv;
          ++lsize;

  #if defined(REPROBE_STAT)
      this->reprobes += get_distance(reprobe);
      this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
  #endif
          no_shift = true;
          break;
        } else if (reprobe > curr_info) {
          // if current position has less than reprobe dist, then swap
          ::std::swap(info_container[i], reprobe);
          ::std::swap(container[i], vv);

          ++lsize;

          // prefetch container.
          for (size_t l = sizeof(value_type); l <= 64; l += sizeof(value_type)) {
            _mm_prefetch((const char *)&(container[i + l]), _MM_HINT_T0);
          }


  #if defined(REPROBE_STAT)
      this->reprobes += get_distance(info_container[i]);
      this->max_reprobes = std::max(this->max_reprobes, get_distance(info_container[i]));
  #endif
          break;
        } else if (reprobe == curr_info) {
          // if current position is equal to reprobe, then check for equality.

          // same distance, then possibly same value.  let's check.
          // note that a previously swapped vv would not match again and can be skipped.
          if (eq(container[i].first, vv.first)) {
            // same, then we found it and need to return.
  #if defined(REPROBE_STAT)
      this->reprobes += get_distance(reprobe);
      this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
  #endif

            no_shift = true;
            break;
          }
        }  // else reprobe < info_container[i], increase reprobe and i and try again.

      }

      if (i == i_max) {
        std::cout << "block 2 entry " << j << ", id " << get_bucket_id(id) << " curr pos " << i << ", max " << i_max << " info size " << static_cast<bucket_id_type>(info_container.size()) << std::endl;
//        ;
//        throw std::logic_error("ERROR: searching for insertion position reached max reprobe distance (128)");
  	  std::cout << "RESIZE AND RE_INSERT: " << vv.first << std::endl;

        rehash(buckets << 1);  // resize.
        --j;  // roll back one and restart this iterator.
        hashes[hash_pos] = h;  // also roll back hash code.
        continue;  //
      }

      if (no_shift) {   // skip rest of loop if no need to shift.
        continue;
      }

      // --------- SHIFT
        // if here, then swapped and breaked. reprobe and i need to advance 1.
        ++reprobe;
        ++i;

    #if defined(REPROBE_STAT)
        size_t move_count = 0;
        bucket_id_type orig_i = i;
    #endif

        // now shift.  note that since we are incrementing reprobe, consecutive entries for the same bucket
        // when shifting would have reprobe == info_container[i], until changing bucket.
        // therefore swap only occurs at bucket bundaries.
        // this MAY save some memory access for now, but story may be different we are bandwidth bound (and hardware prefetching becomes important).

        // this shift may go all the way to end of buckets.  no wrapping around.
        i_max = static_cast<bucket_id_type>(info_container.size());
        for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.

          curr_info = info_container[i];

          if (is_empty(curr_info)) {
            // if current position is empty, then insert here and stop.
            info_container[i] = reprobe;
            container[i] = vv;

    #if defined(REPROBE_STAT)
            ++move_count;
    #endif

            break;
          } else if (reprobe > curr_info) {
            // if current position has less than reprobe dist (boundary), then swap
            ::std::swap(info_container[i], reprobe);
            ::std::swap(container[i], vv);

    #if defined(REPROBE_STAT)
            ++move_count;
    #endif
            // continue until reaching an empty slot
          } // since we are just shifting and swapping at boundaries, in the else cluase we have reprobe == info_container[i] and never reprobe < info_container[i]
            // we should also never get the case of duplicate entries,
        }

        if (i == i_max) {
          std::cout << i << ", " << i_max << std::endl;
          throw std::logic_error("ERROR: shifting entries to the right ran out of room in container.");
        }


#if defined(REPROBE_STAT)
    this->moves += move_count;
    this->max_moves = std::max(this->max_moves, move_count);
    this->shifts += (i - orig_i);
    this->max_shifts = std::max(this->max_shifts, (i-orig_i));
#endif
    }

        // ===========  LAST PART ==============.

        // ----------  INSERT

    for (size_t j = block3_min; j < input.size(); ++j) {
          // === same code as in insert(1)..

          // first check if we need to resize.
          if (lsize >= max_load) rehash(buckets << 1);

          // first get the bucket id
          hash_pos = j & hashes_mask;
          id = hashes[hash_pos] & mask; // target bucket id.

          //==== now do some work to insert.
          // first get the bucket id
          vv = input[j];
          no_shift = false;


          reprobe = info_normal;

          i = get_bucket_id(id);
          i_max = i + static_cast<bucket_id_type>(std::numeric_limits<info_type>::max() - info_normal) + 1;
          i_max = std::min(i_max, static_cast<bucket_id_type>(info_container.size()));

          // first take care of inserting vv, search for the first entry of the next bucket, reprobe > target_info.
          // if found duplicate before that or empty slot, then finish.
          // scan through at most max of info_type.
          for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.

            curr_info = info_container[i];

            if (is_empty(curr_info)) {
              // if current position is empty, then insert here and return.  no shifting is needed.
              info_container[i] = reprobe;
              container[i] = vv;
              ++lsize;

      #if defined(REPROBE_STAT)
          this->reprobes += get_distance(reprobe);
          this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
      #endif
              no_shift = true;
              break;
            } else if (reprobe > curr_info) {
              // if current position has less than reprobe dist, then swap
              ::std::swap(info_container[i], reprobe);
              ::std::swap(container[i], vv);

              ++lsize;

              // prefetch container.
              for (size_t l = sizeof(value_type); l <= 64; l += sizeof(value_type)) {
                _mm_prefetch((const char *)&(container[i + l]), _MM_HINT_T0);
              }

      #if defined(REPROBE_STAT)
          this->reprobes += get_distance(info_container[i]);
          this->max_reprobes = std::max(this->max_reprobes, get_distance(info_container[i]));
      #endif
              break;
            } else if (reprobe == curr_info) {
              // if current position is equal to reprobe, then check for equality.

              // same distance, then possibly same value.  let's check.
              // note that a previously swapped vv would not match again and can be skipped.
              if (eq(container[i].first, vv.first)) {
                // same, then we found it and need to return.
      #if defined(REPROBE_STAT)
          this->reprobes += get_distance(reprobe);
          this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
      #endif

                no_shift = true;
                break;
              }
            }  // else reprobe < info_container[i], increase reprobe and i and try again.

          }

          if (i == i_max) {
//            std::cout << "block 3 entry " << j << ", id " << get_bucket_id(id) << " curr pos " << i << ", max " << i_max << " info size " << static_cast<bucket_id_type>(info_container.size()) << std::endl;
//            print();
//            throw std::logic_error("ERROR: searching for insertion position reached max reprobe distance (128)");
            rehash(buckets << 1);  // resize.
            --j;  // roll back one and restart this iterator.
            continue;  //
          }

          if (no_shift) {   // skip rest of loop if no need to shift.
            continue;
          }

          // --------- SHIFT

            // if here, then swapped and breaked. reprobe and i need to advance 1.
            ++reprobe;
            ++i;

        #if defined(REPROBE_STAT)
            size_t move_count = 0;
            bucket_id_type orig_i = i;
        #endif

            // now shift.  note that since we are incrementing reprobe, consecutive entries for the same bucket
            // when shifting would have reprobe == info_container[i], until changing bucket.
            // therefore swap only occurs at bucket bundaries.
            // this MAY save some memory access for now, but story may be different we are bandwidth bound (and hardware prefetching becomes important).

            // this shift may go all the way to end of buckets.  no wrapping around.
            i_max = static_cast<bucket_id_type>(info_container.size());
            for (; i < i_max; ++i, ++reprobe) {  // limit to max info_type reprobe positions for vv insert.

              curr_info = info_container[i];

              if (is_empty(curr_info)) {
                // if current position is empty, then insert here and stop.
                info_container[i] = reprobe;
                container[i] = vv;

        #if defined(REPROBE_STAT)
                ++move_count;
        #endif

                break;
              } else if (reprobe > curr_info) {
                // if current position has less than reprobe dist (boundary), then swap
                ::std::swap(info_container[i], reprobe);
                ::std::swap(container[i], vv);

        #if defined(REPROBE_STAT)
                ++move_count;
        #endif
                // continue until reaching an empty slot
              } // since we are just shifting and swapping at boundaries, in the else cluase we have reprobe == info_container[i] and never reprobe < info_container[i]
                // we should also never get the case of duplicate entries,
            }

            if (i == i_max) {
              std::cout << i << ", " << i_max << std::endl;
              throw std::logic_error("ERROR: shifting entries to the right ran out of room in container.");
            }


  #if defined(REPROBE_STAT)
      this->moves += move_count;
      this->max_moves = std::max(this->max_moves, move_count);
      this->shifts += (i - orig_i);
      this->max_shifts = std::max(this->max_shifts, (i-orig_i));
  #endif
    }

    free(hashes);

#if defined(REPROBE_STAT)
    print_reprobe_stats("INSERT PREFETCH", input.size(), (lsize - before));
#endif

  }




	/// batch insert using sorting.  This is about 4x slower on core i5-4200U (haswell) than integrated batch insertion above, even just for sort.
  /// THIS NOT WORKING RIGHT NOW.
	template <typename LESS = ::std::less<key_type> >
	void insert_sort(::std::vector<value_type> const & input) {

    std::cerr << "sort adds a stable sort, but otherwise same as integrated." << std::endl;

//#if defined(REPROBE_STAT)
//		this->reprobes = 0;
//		this->max_reprobes = 0;
//		this->moves = 0;
//		this->max_moves = 0;
//		size_t total = input.size();
//#endif

		// need hyperloglog to estimate the final size, in order to sort by bucket.


    size_t local_mask = next_power_of_2(static_cast<size_t>(static_cast<float>(input.size() + this->lsize) / this->max_load_factor)) - 1;

    // 1/2 local mask is okay, but using mask is not, only for uint32 and uint64.  DONT KNOW WHY
    local_mask >>= 1;

		::std::vector<value_type> orig(input);
		::std::vector<value_type> sorted(input);

		// sort by last binary digits of hash value
		std::stable_sort(sorted.begin(), sorted.end(),
				[this, &local_mask](value_type const & x, value_type const & y){
			 //return ((this->hash(x.first) & this->mask) < (this->hash(y.first) & this->mask));  // this line is NOT working.
			 return (this->hash(x.first) & local_mask) < (this->hash(y.first) & local_mask);  // this line is working...

			 // do not want below.  messes up original order.
//			size_t id_x = this->hash(x.first) & local_mask;
//			size_t id_y = this->hash(y.first) & local_mask;
//
//			return (id_x < id_y) || ( (id_x == id_y) && less(x.first, y.first) );
		});


    std::stable_sort(orig.begin(), orig.end(), [](value_type const & x, value_type const & y){
          return (x.first < y.first);
    });


    std::cout << " LOCAL MASK " << local_mask << std::endl;
//    std::cout << "in_k,in_v,in_h,in_p,s_k,s_v,s_h,s_p,hs_k,hs_v,hs_h,hs_p" << std::endl;
//    for (size_t i = 0; i < input.size(); ++i) {
//      std::cout <<
//          input[i].first << "," << input[i].second << "," << this->hash(input[i].first)  << "," << (this->hash(input[i].first) & local_mask) << "," <<
//          orig[i].first << "," << orig[i].second << "," << this->hash(orig[i].first)  << "," << (this->hash(orig[i].first) & local_mask) << "," <<
//          sorted[i].first << "," << sorted[i].second << "," << this->hash(sorted[i].first) << "," << (this->hash(sorted[i].first) & local_mask) <<
//          std::endl;
//    }

    // no difference in (in)correctness between insert(iter,iter), insert(vec), and insert_integrated
    insert_integrated(sorted);
    print();

		std::stable_sort(sorted.begin(), sorted.end(), [](value_type const & x, value_type const & y){
	        return (x.first < y.first);
		});

		bool same = std::equal(orig.begin(), orig.end(), sorted.begin());

		std::cout << "INSERT SORT:  SORTING CHANGED CONTENT? " << (same ? "no" : "yes") << std::endl;


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


	}

  void insert_shuffled(::std::vector<value_type> const & input) {

//    std::random_shuffle(input.begin(), input.end());
    insert(input);
	  std::cerr << "ERROR: no shuffling in prefetch version" << std::endl;
	  //throw ::std::logic_error("ERROR: DISABLED FOR PREFETCH VERSION");
	  print();


  }


protected:

	/**
	 * return the bucket id where the current key is found.  if not found, max is returned.
	 */
	inline bucket_id_type find_pos(key_type const & k) const {
	  bucket_id_type i = hash(k) & mask;
	  return find_pos_with_hint(k, i);

	}

  /**
   * return the bucket id where the current key is found.  if not found, max is returned.
   */
  inline bucket_id_type find_pos_with_hint(key_type const & k, bucket_id_type const & id) const {

    info_type reprobe = info_normal;

    // first get the bucket id
    bucket_id_type i = id;
    info_type curr_info;

    // can only search within max of info_type...
    bucket_id_type i_max = i + static_cast<bucket_id_type>(std::numeric_limits<info_type>::max()- info_normal) + 1;
    i_max = std::max(i_max, static_cast<bucket_id_type>(info_container.size()));

    for (; i < i_max; ++i, ++reprobe) {
      curr_info = info_container[i];

      if (reprobe > curr_info) {
        // indicate that the current
        // entry is from a different bucket.
        // so did not find one.
#if defined(REPROBE_STAT)
    this->reprobes += get_distance(reprobe);
    this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
#endif
        return i;
      } else if (reprobe == curr_info) {
        // possibly matched.  compare.
        if (eq(k, container[i].first)) {
#if defined(REPROBE_STAT)
    this->reprobes += get_distance(reprobe);
    this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
#endif
          return mark_as_existing(i);
        }
      } // else still traversing a previous bucket.
    }

#if defined(REPROBE_STAT)
    this->reprobes += get_distance(reprobe);
    this->max_reprobes = std::max(this->max_reprobes, get_distance(reprobe));
#endif

    // reaching here means it was not found.
    //return i;
    throw std::logic_error("ERROR: exhausted search range without finding a match.  THIS SHOULD NOT HAPPEN");
  }

public:

	/**
	 * @brief count the presence of a key
	 */
	size_type count( key_type const & k ) const {

		return exists(find_pos(k)) ? 1 : 0;

	}


	template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
		::std::vector<size_type> counts(std::distance(begin, end), 0);

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif

    bucket_id_type id, id_la;

    // USE power of 2 look ahead...

    // prefetch.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), counts.size());  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (counts.size() > LOOK_AHEAD) ? (counts.size() - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      size_t i = 0;
      for (Iter it = begin; it != begin + block1_max; ++it, ++i) {
        // compute hash and store it.
        hashes[i] = hash((*it).first);  // no need to prefetch here - sequential access.

        id = hashes[i] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_NTA);
      }

    size_t hashes_mask = LOOK_AHEAD - 1;

    // iterate based on size between rehashes
    i = 0;
    for (Iter it = begin, it2 = begin + block1_max; it != begin + block2_max; ++it, ++it2, ++i) {

      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      //=== prefetch
      hashes[hash_pos] = hash((*it2).first);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_NTA);

      counts[i] = exists(find_pos_with_hint((*it).first, id)) ? 1 : 0;
    }

    for (Iter it = begin + block3_min; it != end; ++it, ++i) {
      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      counts[i] = exists(find_pos_with_hint((*it).first, id)) ? 1 : 0;
    }

#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT ITER KEY", std::distance(begin, end), counts.size());
#endif

		free(hashes);
		return counts;
	}


	template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
		typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
	std::vector<size_type> count(Iter begin, Iter end) {
		::std::vector<size_type> counts(std::distance(begin, end), 0);

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif
    bucket_id_type id, id_la;

    // USE power of 2 look ahead...

    // prefetch.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), counts.size());  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (counts.size() > LOOK_AHEAD) ? (counts.size() - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      size_t i = 0;
      for (Iter it = begin; it != begin + block1_max; ++it, ++i) {
        // compute hash and store it.
        hashes[i] = hash(*it);  // no need to prefetch here - sequential access.

        id = hashes[i] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_NTA);
      }

    size_t hashes_mask = LOOK_AHEAD - 1;

    // iterate based on size between rehashes
    i = 0;
    for (Iter it = begin, it2 = begin + block1_max; it != begin + block2_max; ++it, ++it2, ++i) {

      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      //=== prefetch
      hashes[hash_pos] = hash(*it2);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_NTA);

      counts[i] = exists(find_pos_with_hint(*it, id)) ? 1 : 0;
    }

    for (Iter it = begin + block3_min; it != end; ++it, ++i) {
      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      counts[i] = exists(find_pos_with_hint(*it, id)) ? 1 : 0;
    }



#if defined(REPROBE_STAT)
		print_reprobe_stats("COUNT ITER KEY", std::distance(begin, end), counts.size());
#endif
		free(hashes);

		return counts;
	}

	/**
	 * @brief find the iterator for a key
	 */
	iterator find(key_type const & k) {

		bucket_id_type idx = find_pos(k);

		if (exists(idx))
      return iterator(container.begin() + get_bucket_id(idx), info_container.begin()+ get_bucket_id(idx),
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
      return const_iterator(container.cbegin() + get_bucket_id(idx), info_container.cbegin()+ get_bucket_id(idx),
          info_container.cend(), filter);
		else
      return const_iterator(container.cend(), info_container.cend(), filter);


	}




  template <typename Iter, typename std::enable_if<std::is_constructible<value_type,
    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
  std::vector<value_type> find(Iter begin, Iter end) {
    size_t total = std::distance(begin, end);

    ::std::vector<value_type> counts;
    counts.reserve(total);

#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif

    bucket_id_type id, id_la;
    bucket_id_type found;

    // USE power of 2 look ahead...

    // prefetch.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), total);  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (total > LOOK_AHEAD) ? (total - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      size_t i = 0;
      for (Iter it = begin; it != begin + block1_max; ++it, ++i) {
        // compute hash and store it.
        hashes[i] = hash((*it).first);  // no need to prefetch here - sequential access.

        id = hashes[i] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_NTA);
      }

    size_t hashes_mask = LOOK_AHEAD - 1;

    // iterate based on size between rehashes
    i = 0;
    for (Iter it = begin, it2 = begin + block1_max; it != begin + block2_max; ++it, ++it2, ++i) {

      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      //=== prefetch
      hashes[hash_pos] = hash((*it2).first);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_NTA);

      found = find_pos_with_hint((*it).first, id);
      if (exists(found)) counts.emplace_back(container[get_bucket_id(found)]);
    }

    for (Iter it = begin + block3_min; it != end; ++it, ++i) {
      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      found = find_pos_with_hint((*it).first, id);
      if (exists(found)) counts.emplace_back(container[get_bucket_id(found)]);
    }

#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND ITER PAIR", std::distance(begin, end), counts.size());
#endif

    free(hashes);
    return counts;
  }


  template <typename Iter, typename std::enable_if<std::is_constructible<key_type,
    typename std::iterator_traits<Iter>::value_type >::value, int >::type = 1>
  std::vector<value_type> find(Iter begin, Iter end) {
    size_t total = std::distance(begin, end);

    ::std::vector<value_type> counts;
    counts.reserve(total);

#if defined(REPROBE_STAT)
    reset_reprobe_stats();
#endif

    bucket_id_type id, id_la;
    bucket_id_type found;

    // USE power of 2 look ahead...

    // prefetch.
    // allocate a circular buffer of size 64 (8 cache lines)
      size_t *hashes = (size_t *)(malloc(LOOK_AHEAD * sizeof(size_t)));
      info_type hash_pos = 0;

      // first preload LOOK_AHEAD number of items.
      size_t block1_max = std::min(static_cast<size_t>(LOOK_AHEAD), total);  // if fewer than LOOK_AHEAD, no point in using?
      size_t block2_max = (total > LOOK_AHEAD) ? (total - LOOK_AHEAD) : 0;  // if fewer than LOOK_AHEAD, no block 2.
      size_t block3_min = block2_max;  // may not need to allocate this...

      // try _MM_HINT_NT and _MM_HINT_T0.  try simple loop first.

      // ====  prefetch.  also save the hash function.
      size_t i = 0;
      for (Iter it = begin; it != begin + block1_max; ++it, ++i) {
        // compute hash and store it.
        hashes[i] = hash(*it);  // no need to prefetch here - sequential access.

        id = hashes[i] & mask;  //

        // prefetch info container
        _mm_prefetch((const char *)&(info_container[id]), _MM_HINT_T0);
        // prefetch container
        _mm_prefetch((const char *)&(container[id]), _MM_HINT_NTA);
      }

    size_t hashes_mask = LOOK_AHEAD - 1;

    // iterate based on size between rehashes
    i = 0;
    for (Iter it = begin, it2 = begin + block1_max; it != begin + block2_max; ++it, ++it2, ++i) {

      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      //=== prefetch
      hashes[hash_pos] = hash(*it2);
      id_la = hashes[hash_pos] & mask;
      _mm_prefetch((const char *)&(info_container[id_la]), _MM_HINT_T0);
      _mm_prefetch((const char *)&(container[id_la]), _MM_HINT_NTA);

      found = find_pos_with_hint(*it, id);
      if (exists(found)) counts.emplace_back(container[get_bucket_id(found)]);
    }

    for (Iter it = begin + block3_min; it != end; ++it, ++i) {
      // === same code as in insert(1)..

      // first get the bucket id
      hash_pos = i & hashes_mask;
      id = hashes[hash_pos] & mask;  // target bucket id.

      found = find_pos_with_hint(*it, id);
      if (exists(found)) counts.emplace_back(container[get_bucket_id(found)]);
    }

#if defined(REPROBE_STAT)
    print_reprobe_stats("FIND ITER KEY", std::distance(begin, end), counts.size());
#endif

    free(hashes);
    return counts;
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

//============ ERASE

protected:
	/**
	 * @brief erases a key.  performs backward shift.
	 * @details note that entries for the same bucket are always consecutive, so we don't have to shift, but rather, move elements at boundaries only.
	 *    boundaries are characterized by non-decreasing info dist, or by empty.
	 *
	 */
	inline size_type erase_and_compact(key_type const & k) {

		bucket_id_type found = find_pos(k);

		if (missing(found)) {  // did not find.
			return 0;
		}

		//========= if next is empty, just mark current as empty as well.

		--lsize;   // reduce the size by 1.
		bucket_id_type curr = get_bucket_id(found);
		bucket_id_type next = curr + static_cast<bucket_id_type>(1);


		//=========== CHECK IF WE NEED TO SHIFT

		// last entry of array.  nothing to move.  done.
		if (next == static_cast<bucket_id_type>(info_container.size())) {
			info_container[curr] = info_empty;
			return 1;
		}

		//=========== SHIFTING NOW...
#if defined(REPROBE_STAT)
		size_type move_count = 0;
#endif

		// otherwise, save the position where we are deleting - we will move the last entry of the bucket to here.
		bucket_id_type target = get_bucket_id(found);

		// for iterating to find boundaries
		info_type next_info;
		info_type curr_info = info_container[curr];

		bucket_id_type next_max = static_cast<bucket_id_type>(info_container.size());
		for (; next < next_max; ++next) {
			next_info = info_container[next];

			// terminate when next entry is empty or has distance 0.0
			if (next_info <= info_normal) {   // covers all empty values, and occupied with dist of 0.
				break;
			}

			// next is occupied with distance greater than 0 (so is curr_info)

			// at bucket boundaries, the reprobe value is not increasing
			if (next_info <= curr_info) {
				// change the curr info
				info_container[curr] = next_info - 1;  // for the next iteration.

				// and move the current value entry.
				container[target] = container[curr];

				// store the curr position as the next target of move.
				target = curr;

#if defined(REPROBE_STAT)
				++move_count;
#endif
			}

			// advance
			curr = next;
			curr_info = next_info;

		}
		// very last iteration.  the last "target" already has the right info
		// but is missing the value, so move that here.
		// also the curr info container needs to be reset to empty.
		info_container[curr] = info_empty;  // set last to empty.
		container[target] = container[curr];

#if defined(REPROBE_STAT)
		++move_count;

		this->moves += move_count;
		this->max_moves = std::max(this->max_moves, move_count);
		this->shifts += curr - get_bucket_id(found);
		this->max_shifts = std::max(this->max_shifts, curr - get_bucket_id(found));
#endif
		return 1;
	}

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

		for (auto it = begin; it != end; ++it) {
			erase_and_compact(*it);
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
constexpr typename hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::info_type hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::info_empty;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::info_type hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::info_normal;

template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bid_failed;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bid_exists;
template <typename Key, typename T, typename Hash, typename Equal, typename Allocator >
constexpr typename hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bucket_id_type hashmap_robinhood_prefetch<Key, T, Hash, Equal, Allocator>::bid_mask;

}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_prefetch_HPP_ */
