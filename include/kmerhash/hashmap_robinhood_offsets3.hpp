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
    // MSB is to indicate if current BUCKET is empty.  rest 7 bits indicate offset for the first BUCKET entry.
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
    static constexpr bucket_id_type bid_pos_mask = ~(static_cast<bucket_id_type>(0)) >> 9;   // lower 55 bits set.
    static constexpr bucket_id_type bid_pos_exists = 1UL << 55;  // 56th bit set.
    static constexpr bucket_id_type bid_info_mask = static_cast<bucket_id_type>(info_mask) << 56;   // lower 55 bits set.
    static constexpr bucket_id_type bid_info_empty = static_cast<bucket_id_type>(info_empty) << 56;  // 56th bit set.

    // failed is speial, correspond to all bits set (max distnace failed).  not using 0x800000... because that indicates failed inserting due to occupied.
    static constexpr bucket_id_type bid_failed = ~(static_cast<bucket_id_type>(0));

    inline bucket_id_type make_missing_bucket_id(size_t const & pos, info_type const & info) const {
      assert(pos <= bid_pos_mask);
      return (static_cast<bucket_id_type>(info) << 55) | pos;
    }
    inline bucket_id_type make_existing_bucket_id(size_t const & pos, info_type const & info) const {
      return make_missing_bucket_id(pos, info) | bid_pos_exists;
    }

    inline bool is_empty(bucket_id_type const & x) const {
      return x >= bid_info_empty;  // empty 0x80....
    }
    inline bool is_normal(bucket_id_type const & x) const {
      return x < bid_info_empty;  // normal. both top bits are set. 0xC0
    }
    inline bool exists(bucket_id_type const & x) const {
      return (x & bid_pos_exists) > 0;
    }
//    inline bucket_id_type mark_as_existing(bucket_id_type const & x) const {
//      return x | bid_pos_exists;
//    }
    inline info_type get_info(bucket_id_type const & x) const {
      return static_cast<info_type>(x >> 56);
    }

    inline size_t get_pos(bucket_id_type const & x) const {
      return x & bid_pos_mask;
    }
    inline size_t get_offset(bucket_id_type const & x) const {
      return (x & bid_info_mask) >> 56;
    }

//    inline size_t get_bucket_id(bucket_id_type const & x) const {
//      return get_pos(x) - get_offset(x);
//    }
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
    for (size_type i = 0; i < info_container.size(); ++i) {
      std::cout << i << ": [" << container[i].first << "->" <<
          container[i].second << "] info " <<
          static_cast<size_t>(info_container[i]) <<
          " bucket = ";
      std::cout << i << " offset = " << get_offset(info_container[i]) <<
          " pos = " << (i + get_distance(info_container[i])) <<
          " empty? " << (is_empty(info_container[i]) ? "true" : "false");
      std::cout << std::endl;
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
    std::fill(this->info_container.begin(), this->info_container.end(), info_empty);
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
    size_type n = next_power_of_2(b);

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

    bucket_id_type id;
    size_t pos;
    size_t endd;

    // iterate through the entire input.
    for (size_t i = 0; i < buckets; ++i) {
      if (is_normal(info_container[i])) {
        // get start and end of bucket.  exclusive end.
        pos = i + static_cast<size_t>(get_distance(info_container[i]));
        endd = i + 1 + static_cast<size_t>(get_distance(info_container[i + 1]));

        // compute the new id from current bucket.  since downsize and power of 2, modulus would do.
        id = i & m;

        // do batch copy to bucket 'id', copy from pos to endd
        insert_with_hint(target, target_info, id, container, info_container, pos, endd);

      }  // else is empty, so continue.
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

    bucket_id_type id, last;
    size_t pos;
    size_t endd;
    last = 0;

      for (size_t i = 0; i < buckets; ++i) {
        if (is_normal(info_container[i])) {

          // get start and end of bucket.  exclusive end.
          pos = i + static_cast<size_t>(get_distance(info_container[i]));
          endd = i + 1 + static_cast<size_t>(get_distance(info_container[i + 1]));


          for (; pos < endd; ++pos) {
            // compute the new id via hash.
            id = hash(container[pos].first) & m;

            last = insert_with_hint(target, target_info, id, last, container[pos]);
          }
        }  // else is empty, so continue.
      }
  }


	/**
	 * return the position in container where the current key is found.  if not found, max is returned.
	 */
	bucket_id_type find_pos_with_hint(key_type const & k, size_t const & bid) const {

		assert(bid != bid_failed);

		info_type offset = info_container[bid];
		size_t start = bid + get_distance(offset);  // distance is at least 0, and definitely not empty

		// no need to check for empty?  if i is empty, then this one should be.
		// otherwise, we are checking distance so doesn't matter if empty.

		// first get the bucket id
		if (is_empty(offset) ) {
			// std::cout << "Empty entry at " << i << " val " << static_cast<size_t>(info_container[i]) << std::endl;
			return make_missing_bucket_id(start, offset);
		}

		// get the next bucket id
		size_t end = bid + 1 + get_distance(info_container[bid + 1]);   // distance is at least 0, and can be empty.

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		// now we scan through the current.
		for (; start < end; ++start) {

			if (eq(k, container[start].first)) {
				return make_existing_bucket_id(start, offset);
			}

#if defined(REPROBE_STAT)
			++reprobe;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

		return make_missing_bucket_id(end, offset);
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

		info_type offset = info_container[id];

		// handle target bucket first.
		// then handle shifting.

		// 4 cases:
		// empty bucket, offset == 0        use this bucket. offset at original bucket converted to non-empty.  move vv in.  next bucket unchanged.  done
		// empty bucket, offset > 0         use this bucket. offset at original bucket converted to non-empty.  swap vv in.  go to next bucket
		// non empty bucket, offset == 0.   use next bucket. offset at original bucket not changed              swap vv in.  go to next bucket
		// non empty bucket, offset > 0.    use next bucket. offset at original bucket not changed.             swap vv in.  go to next bucket

		// first get the bucket id
		if ( offset == info_empty ) {
			container[id] = v;
			info_container[id] = info_normal;

			// std::cout << "Empty entry at " << i << " val " << static_cast<size_t>(info_container[i]) << std::endl;
			return make_missing_bucket_id(id, offset);
		}



		size_t start = id + get_distance(offset);  // distance is at least 0, and definitely not empty

		// no need to check for empty.  if i is empty, then this one should be.
		// otherwise, we are checking distance so doesn't matter if empty.


		// get the next bucket id
		size_t end = bid + 1 + get_distance(info_container[bid + 1]);   // distance is at least 0, and can be empty.

#if defined(REPROBE_STAT)
		size_t reprobe = 0;
#endif
		// now we scan through the current.
		for (; start < end; ++start) {

			if (eq(k, container[start].first)) {
				return make_existing_bucket_id(start, offset);
			}

#if defined(REPROBE_STAT)
			++reprobe;
#endif
		}

#if defined(REPROBE_STAT)
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif

		bucket_id_type bpos = find_pos_with_hint(v.first, id); // container insertion point.

		if (exists(bpos)) {  // found it.  so no insert.
	//      std::cout << "duplicate found at " << found << std::endl;
	//      std::cout << "  existing at " << container[found].first << "->" << container[found].second << std::endl;
	//      std::cout << "  inserting " << vv.first << "->" << vv.second << std::endl;
		  return bpos;
		}

		// did not find, so insert.
		++lsize;


		// first swap in at start of bucket, then deal with the swapped.

		// swap until empty info
		// then insert into last empty,
		// then update info until info_empty


		// incrementally swap.
		size_t target_bid = bid;
		size_t target = get_pos(bpos);
		// now iterate and try to swap, terminate when we've reached an empty slot that points to self.
		while (info_container[target_bid] != info_empty) {
		  if (is_normal(info_container[target_bid])) {
			// swap for occupied buckets only
			target = target_bid + get_distance(info_container[target_bid]);
			::std::swap(container[target], vv);
		  }

		  // increment everyone's info except for the first one., but don't change the occupied bit.
		  info_container[target_bid] += (target_bid == bid) ? 0 : 1;

		  ++target_bid;

		}

		// reached end, insert last

		found = target_bid + get_distance(info_container[target_bid]);
		container[found] = std::move(vv);  // empty slot.
		// if we are at the original bucket, then set it to normal (1 entry)

		set_normal(info_container[bid]);  // head is always marked as occupied.
		if (target_bid != bid) {  // tail, originally == empty
		  ++info_container[target_bid];  // increment tail end if it's not same as head.
		}

		found = bid + get_distance(info_container[bid]);


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
//			if (info_container[i] == info_empty)  {
//				// not completely empty,
//				result = i;
//				break;
//			}
//			i += 1 + get_distance(info_container[i]);
//		}
//
//		// if i is less than buckets, then done.
//		// else if i is greater or equal, then wrapped around.
//		if (i >= buckets) {
//			i &= mask;  // wrap around
//			while (i < pos) {
//				if (info_container[i] == info_empty) { // not completely empty,
//					result = i;
//					break;
//				}
		//			i += 1 + get_distance(info_container[i]);
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

		assert(lsize < max_load);

//		size_t j = i;
//		size_t offset = get_distance(info_container[i]);
		while (info_container[i] != info_empty) {
//			j = i + 1;
// 			offset = get_distance(info_container[j]);
//			i = j + offset;  // jump by distance indicated in info, and increment by 1.
			if (info_container[i] == info_normal) ++i;
			else i += get_distance(info_container[i]);
		}
#if defined(REPROBE_STAT)
		size_t reprobe;
//		if (i < pos) { // wrapped around
//			reprobe = buckets - pos + i;
//		} else {
			reprobe = i - pos;
//		}
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
//			if (info_container[i] == info_empty)  {
//				// not completely empty,
//				result = i;
//				break;
//			}
//			i += 1 + get_distance(info_container[i]);
//		}
//
//		// if i is less than buckets, then done.
//		// else if i is greater or equal, then wrapped around.
//		if (i >= buckets) {
//			i &= mask;  // wrap around
//			while (i < pos) {
//				if (info_container[i] == info_empty) { // not completely empty,
//					result = i;
//					break;
//				}
		//			i += 1 + get_distance(info_container[i]);
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

		assert(lsize < max_load);

		unsigned char offset = get_distance(info_container[i]);
//		if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
		while (offset > 0) {   // should try linear scan
			i += offset;  // jump by distance indicated in info, and increment by 1.

			offset = get_distance(info_container[i]);
		}


//		size_t j = 0;
//		unsigned char offset = get_distance(info_container[i]);
//		if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//		while (offset > 0) {   // should try linear scan
//			j = i + 1;
//			if (pos == 9) std::cout << "B i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//			offset = get_distance(info_container[j]);
//			if (pos == 9) std::cout << "C i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//			i = j + offset;  // jump by distance indicated in info, and increment by 1.
//
//			if (pos == 9) std::cout << "A i=" << i << " j=" << j << " offset=" << static_cast<size_t>(offset) << std::endl;
//		}
#if defined(REPROBE_STAT)
		size_t reprobe;
//		if (i < pos) { // wrapped around
//			reprobe = buckets - pos + i;
//		} else {
			reprobe = i - pos;
//		}
		this->reprobes += reprobe;
		this->max_reprobes = std::max(this->max_reprobes, static_cast<size_t>(reprobe));
#endif
		return i;

	}


	/**
	 * find next non-empty position.  meant to be with input that points to an empty position.
	 */
	size_type find_next_non_empty_pos(size_t const & pos) const {
		size_t i = pos + 1;
		while (info_container[i] == info_empty) {
			++i;
		}
		return i;
	}



public:

	/**
	 * @brief insert a single key-value pair into container.
	 */
	std::pair<iterator, bool> insert(value_type && vv) {

		if (lsize >= max_load) {
		  std::cout << "RESIZE lsize " << lsize << " max_load " << max_load << " new size " << (buckets << 1) << std::endl;

			rehash(buckets << 1);  // double in size.
			std::cout << "RESIZE DONE" << std::endl;
		}

		size_type bid;
		size_type found;
		std::tie(bid, found) = find_pos(vv.first); // container insertion point.

		if (found != std::numeric_limits<size_type>::max()) {  // found it.  so no insert.
	//      std::cout << "duplicate found at " << found << std::endl;
	//      std::cout << "  existing at " << container[found].first << "->" << container[found].second << std::endl;
	//      std::cout << "  inserting " << vv.first << "->" << vv.second << std::endl;
		  return std::make_pair(iterator(container.begin() + found, info_container.begin()+ found, info_container.end(), filter), false);
		}

		// did not find, so insert.
		++lsize;

		// 4 cases:
		// empty bucket, offset == 0        use this bucket. offset at original bucket converted to non-empty.  move vv in.  done
		// empty bucket, offset > 0         use this bucket. offset at original bucket converted to non-empty.  swap vv in.  go to next bucket
		// non empty bucket, offset == 0.   use next bucket. offset at original bucket not changed              swap vv in.  go to next bucket
		// non empty bucket, offset > 0.    use next bucket. offset at original bucket not changed.             swap vv in.  go to next bucket


		// first swap in at start of bucket, then deal with the swapped.


		// incrementally swap.
		size_t target_bid = bid;
		// now iterate and try to swap, terminate when we've reached an empty slot that points to self.
		while (info_container[target_bid] != info_empty) {
		  if (is_normal(info_container[target_bid])) {
			// swap for occupied buckets only
			found = target_bid + get_distance(info_container[target_bid]);
			::std::swap(container[found], vv);
		  }

		  // increment everyone's info except for the first one., but don't change the occupied bit.
		  info_container[target_bid] += (target_bid == bid) ? 0 : 1;

		  ++target_bid;

		}

		// reached end, insert last

		found = target_bid + get_distance(info_container[target_bid]);
		container[found] = std::move(vv);  // empty slot.
		// if we are at the original bucket, then set it to normal (1 entry)

		set_normal(info_container[bid]);  // head is always marked as occupied.
		if (target_bid != bid) {  // tail, originally == empty
		  ++info_container[target_bid];  // increment tail end if it's not same as head.
		}

		found = bid + get_distance(info_container[bid]);
		
		return std::make_pair(iterator(container.begin() + found, info_container.begin() + found, info_container.end(), filter), true);

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
	
		return (pos != std::numeric_limits<size_type>::max()) ? 1 : 0;

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

		if (idx != std::numeric_limits<size_type>::max())
			return iterator(container.begin() + idx, info_container.begin()+ idx, info_container.end(), filter);
		else
			return iterator(container.end(), info_container.end(), filter);

	}

	/**
	 * @brief find the iterator for a key
	 */
	const_iterator find(key_type const & k) const {

		size_type idx = find_pos(k).second;

		if (idx != std::numeric_limits<size_type>::max())
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

		if (found == std::numeric_limits<size_type>::max()) {
			// did not find. done
// debug			std::cout << "erasing " << k << " did not find one " << std::endl;
			return 0;
		}

		--lsize;   // reduce the size by 1.

		//size_t bid = hash(k) & mask;   // get bucket id
		// get the end of the non-empty range, starting from the next position.
    size_type bid1 = bid + 1;  // get the next bucket, since bucket contains offset for current bucket.


		size_type found1 = found + 1;
		size_type end = find_next_zero_offset_pos(found1);

// debug		std::cout << "erasing " << k << " hash " << bid << " at " << found << " end is " << end << std::endl;

		// move to backward shift.  move [found+1 ... end-1] to [found ... end - 2].  end is excluded because it has 0 dist.
//		if (end < found) {  // wrapped around
//			// first move to the end.
//			memmove(&(container[found]), &(container[found1]), ((buckets - 1) - found) * sizeof(value_type));
//
//			// definitely wrapped around, so copy first entry to last
//			if (end > 0) {
//				container[buckets - 1] = std::move(container[0]);
//
//				// now if there is still more, than do it
//				memmove(&(container[0]), &(container[1]), (end - 1) * sizeof(value_type));
//			}
//		} else if (end > found) {  // no wrap around.
			memmove(&(container[found]), &(container[found1]), (end - 1 - found) * sizeof(value_type));
//		} // else target == found, nothing to shift.

// debug		print();

		// now change the offsets.
		// start from bid+1, end at end - 1.


		// if that was the last entry for the bucket, then need to change this to say empty.
		if (get_distance(info_container[bid]) == get_distance(info_container[bid1])) {  // both have same distance, so bid has only 1 entry
			set_empty(info_container[bid]);
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
//		if (end < bid1) {  // wrapped around
//			// first move to the end.
//			std::for_each(info_container.begin() + bid1, info_container.end(),
//					[](info_type & x){ --x; });
//
//			std::for_each(info_container.begin(), info_container.begin() + end,
//					[](info_type & x){ --x; });
//
//		} else {  // no wrap around.
			std::for_each(info_container.begin() + bid1, info_container.begin() + end,
					[](info_type & x){ --x; });

//		}

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

#undef REPROBE_STAT

}  // namespace fsc
#endif /* KMERHASH_HASHMAP_ROBINHOOD_DOUBLING_OFFSETS_HPP_ */
