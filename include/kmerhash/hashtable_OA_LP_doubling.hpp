/*
 * hashtable_OA_LP_doubling.hpp
 *
 *  Created on: Feb 27, 2017
 *      Author: tpan
 */

#ifndef KMERHASH_HASHTABLE_OA_LP_DOUBLING_HPP_
#define KMERHASH_HASHTABLE_OA_LP_DOUBLING_HPP_

#include <vector>
#include <type_traits>

#include "filter_iterator.hpp"
#include "transform_iterator.hpp"

/*
 * get the next power of 2 for unsigned integer type.  based on http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
template <typename T>
inline constexpr T next_power_of_2(T x) {
	static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value, "ERROR: can only find power of 2 for unsigned integers.");

	--x;
	switch (sizeof(T)) {
	case 8:
		x |= x >> 32;
	case 4:
		x |= x >> 16;
	case 2:
		x |= x >> 8;
	default:
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
		break;
	}
	++x;
	return x;
}


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
 *  [ ] separate info from rest of struct, so as to remove use of transform_iterator, thus allowing us to change values through iterator.
 *  [ ] batch mode operations to allow more opportunities for optimization including SIMD
 *  [ ] predicated version of operations
 *  [ ] macros for repeated code.
 *
 *  first do the stupid simple implementation.
 */
template <typename Key, typename T, typename Hash = ::std::hash<Key>,
		typename Equal = ::std::equal_to<Key>, typename Allocator = ::std::allocator<std::pair<const Key, T> > >
class hashmap_oa_lp_do_tuple {

public:

    using key_type              = Key;
    using mapped_type           = T;
    using value_type            = ::std::pair<const Key, T>;
    using hasher                = Hash;
    using key_equal             = Equal;

protected:
    struct internal_value_type {
    	key_type k;
    	mapped_type v;
    	unsigned char info;


    	inline bool is_empty() {
    		return info == 0x80;
    	}
    	inline bool is_deleted() {
    		return info == 0x40;
    	}
    	inline bool is_normal() {  // upper bits == 0
    		return info < 0x40;
    	}
    	inline void set_normal() {
    		info &= 0x3F;  // clear the bits;
    	}
    	inline void set_deleted() {
    		info = 0x40;
    	}
    	inline void set_empty() {
    		info = 0x80;  // nothing here.
    	}

    	inline key_type getKey() { return k; }
    	inline mapped_type getVal() { return v; }
    	inline value_type getKV() { return std::make_pair<const key_type, mapped_type>(k, v); }

    };

    struct to_value_type {
    	value_type operator()(internal_value_type const & x) {
    		return x.getKV();
    	}
    };


    using container_type		= ::std::vector<internal_value_type, Allocator>;

    // filter
    struct empty_deleted_filter {
    	bool operator()(internal_value_type const & x) { return x.is_normal(); };
    };

    using filter_iter = ::bliss::iterator::filter_iterator<empty_deleted_filter, typename container_type::iterator>;
    using filter_const_iter = ::bliss::iterator::filter_iterator<empty_deleted_filter, typename container_type::const_iterator>;

public:

    using allocator_type        = typename container_type::allocator_type;
    using reference 			= typename container_type::reference;
    using const_reference	    = typename container_type::const_reference;
    using pointer				= typename container_type::pointer;
    using const_pointer		    = typename container_type::const_pointer;
    using iterator              = ::bliss::iterator::transform_iterator<filter_iter, to_value_type>;
    using const_iterator        = ::bliss::iterator::transform_iterator<filter_const_iter, to_value_type>;
    using size_type             = typename container_type::size_type;
    using difference_type       = typename container_type::difference_type;


protected:


    empty_deleted_filter filter;
    hasher hash;
    key_equal eq;
    to_value_type internal_to_external;

    container_type container;

    size_t size;
    mutable size_t buckets;
    mutable size_t min_load;
    mutable size_t max_load;
    mutable float min_load_factor;
    mutable float max_load_factor;


public:


	explicit hashmap_oa_lp_do_tuple(size_t const & _capacity = 128, float const & _min_load_factor = 0.2, float const & _max_load_factor = 0.6) : size(0) {
		// get the nearest power of 2 above specified capacity.
		buckets = next_power_of_2(_capacity);

		container.resize(buckets);
		// set the min load and max load thresholds.  there should be a good separation so that when resizing, we don't encounter a resize immediately.
		set_min_load_factor(_min_load_factor);
		set_max_load_factor(_max_load_factor);
	};

	inline void set_min_load_factor(float const & _min_load_factor) {
		min_load_factor = _min_load_factor;
		min_load = static_cast<size_t>(static_cast<float>(container.size()) * min_load_factor);

	}

	inline void set_max_load_factor(float const & _max_load_factor) {
		max_load_factor = _max_load_factor;
		max_load = static_cast<size_t>(static_cast<float>(container.size()) * max_load_factor);
	}


	inline float get_load_factor() {
		return static_cast<float>(size) / static_cast<float>(container.size());
	}

	inline float get_min_load_factor() {
		return min_load_factor;
	}

	inline float get_max_load_factor() {
		return max_load_factor;
	}




	iterator begin() {
		return iterator(filter_iter(filter, container.begin(), container.end()), internal_to_external);
	}

	iterator end() {
		return iterator(filter_iter(filter, container.end()), internal_to_external);
	}

	const_iterator cbegin() {
		return const_iterator(filter_const_iter(filter, container.cbegin(), container.cend()), internal_to_external);
	}

	const_iterator cend() {
		return const_iterator(filter_const_iter(filter, container.cend()), internal_to_external);
	}


	/**
	 * @brief reserve space for specified entries.
	 */
	void reserve(size_type n ) {
		if (n > this->max_load) {   // if requested more than current max load, then we need to resize up.
			rehash(next_power_of_2(n));   // rehash to the new size.    current bucket count should be less than next_power_of_2(n).
		}  // do not resize down.  do so only when erase.
	}

	/**
	 * @brief reserve space for specified buckets.
	 * @details	note that buckets > entries.
	 */
	void rehash(size_type const & n) {
		// check it's power of 2
		assert(((n-1) & n) == 0);

		if (n > buckets) {
			buckets = n;
			container_type tmp(buckets);
			container.swap(tmp);

			min_load = static_cast<size_t>(static_cast<float>(container.size()) * min_load_factor);
			max_load = static_cast<size_t>(static_cast<float>(container.size()) * max_load_factor);

			insert(tmp.begin(), tmp.end());
		}
	}



protected:
	void insert(typename container_type::iterator begin, typename container_type::iterator end) {
		for (auto it = begin; it != end; ++it) {
			if ((*it).is_normal()) {
				insert(::std::move(*it));
			}
		}
	}


	/**
	 * @brief insert at the specified position.  does NOT check if already exists.  for internal use during rehash only.
	 */
	size_type insert(internal_value_type&& internal_value) {
		size_type pos = hash(internal_value.k) % buckets;
		size_type i = pos;

		while ((i < buckets) && container[i].is_normal()) ++i;  // find a place to insert.
		if (i == buckets) {
			// did not find one, so search from beginning
			i = 0;
			while ((i < pos) && container[i].is_normal()) ++i;  // find a place to insert.

			if (i == pos) // nothing was found.  this should not happen
				throw std::logic_error("ERROR: did not find any place to insert.  should not have happend");

			// else 0 <= i < pos
		} // else  pos <= i < buckets.

		// insert at i.
		container[i] = internal_value;
		container[i].info = 0;
		container[i].set_normal();

		return i;
	}

public:

// HERE:  TODO: change insert to remove reprobe distance and associated distance.
	/**
	 * @brief insert a single key, value pair.
	 */
	std::pair<iterator, bool> insert(key_type const & key, mapped_type const & val) {

		// first check if we need to resize.
		if (size >= max_load) rehash(buckets << 1);

		// first get the bucket id
		size_type pos = hash(key) % buckets;
		size_type i;
		size_type insert_pos = buckets;

		for (i = pos; i < buckets; ++i) {
			if (container[i].is_empty()) {
				insert_pos = i;
				break;
			}

			if (container[i].is_deleted() && (insert_pos == buckets))
				insert_pos = i;
			else if (container[i].is_normal() && (equal(key, container[i].k)))
				// found a match
				return std::make_pair(iterator(filter_iterator(filter, container.begin() + i, container.end()), internal_to_external), false);
		}
		if (i == buckets) {
			// now repeat for first part
			for (i = 0; i < pos; ++i) {
				if (container[i].is_empty()) {
					insert_pos = i;
					break;
				}

				if (container[i].is_deleted() && (insert_pos == buckets))
					insert_pos = i;
				else if (container[i].is_normal() && (equal(key, container[i].k)))
					// found a match
					return std::make_pair(iterator(filter_iterator(filter, container.begin() + i, container.end()), internal_to_external), false);
			}
		}

		// finally check if we need to insert.
		if (insert_pos == buckets) {  // went through the whole container.  must be completely full.
			throw std::logic_error("ERROR: did not find a slot to insert into.  container must be full.  should not happen.")l
		}

		container[insert_pos].k = key;
		container[insert_pos].v = val;
		container[insert_pos].info = 0;
		container[insert_pos].set_normal();
		++size;

		return std::make_pair(iterator(filter_iterator(filter, container.begin() + insert_pos, container.end()), internal_to_external), true);

	}

	size_type count( key_type const & k ) const {

		// first get the bucket id
		size_t pos = hash(k) % buckets;

		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (container[i].is_empty())  // done
				break;

			if (container[i].is_normal() && (equal(k, container[i].k)))
				return 1;
		}  // ends when i == buckets or empty node.
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (container[i].is_empty())  // done
					break;

				if (container[i].is_normal() && (equal(k, container[i].k)))
					return 1;
			}  // ends when i == buckets or empty node.
		}
		// if we are here, then we did not find it.  return 0.
		return 0;
	}

	iterator find(key_type const & k) {

		// first get the bucket id
		size_t pos = hash(k) % buckets;

		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (container[i].is_empty())  // done
				break;

			if (container[i].is_normal() && (equal(k, container[i].k)))
				return iterator(filter_iterator(filter, container.begin() + i, container.end()), internal_to_external);;
		}  // ends when i == buckets or empty node.
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (container[i].is_empty())  // done
					break;

				if (container[i].is_normal() && (equal(k, container[i].k)))
					return iterator(filter_iterator(filter, container.begin() + i, container.end()), internal_to_external);;
			}  // ends when i == buckets or empty node.
		}
		// if we are here, then we did not find it.  return 0.
		return iterator(filter_iterator(filter, container.end()), internal_to_external);  // end iterator.;

	}

	const_iterator find(key_type const & k) const {

		// first get the bucket id
		size_t pos = hash(k) % buckets;

		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (container[i].is_empty())  // done
				break;

			if (container[i].is_normal() && (equal(k, container[i].k)))
				return const_iterator(filter_iterator(filter, container.cbegin() + i, container.cend()), internal_to_external);;
		}  // ends when i == buckets or empty node.
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (container[i].is_empty())  // done
					break;

				if (container[i].is_normal() && (equal(k, container[i].k)))
					return const_iterator(filter_iterator(filter, container.cbegin() + i, container.cend()), internal_to_external);;
			}  // ends when i == buckets or empty node.
		}
		// if we are here, then we did not find it.  return 0.
		return const_iterator(filter_iterator(filter, container.cend()), internal_to_external);  // end iterator.;
	}

	void update() {
		// TODO: can't do right now, since we have a transform iterator so value cannot be updated.
	}

	size_type erase(key_type const & k) {

		// first get the bucket id
		size_t pos = hash(k) % buckets;

		size_t i;
		for (i = pos; i < buckets; ++i) {
			// first from here to end.
			if (container[i].is_empty())  // done
				break;

			if (container[i].is_normal() && (equal(k, container[i].k))) {
				container[i].set_deleted();

				--size;

				if (size < min_load) rehash(buckets >> 1);

				return 1;
			}
		}  // ends when i == buckets or empty node.
		if (i == buckets) {
			// search the rest of list.

			for (i = 0; i < pos; ++i) {
				// first from here to end.
				if (container[i].is_empty())  // done
					break;

				if (container[i].is_normal() && (equal(k, container[i].k))) {
					container[i].set_deleted();

					--size;

					if (size < min_load) rehash(buckets >> 1);
					return 1;
				}
			}  // ends when i == buckets or empty node.
		}
		// if we are here, then we did not find it.  return 0.
		return 0;

	}

};


#endif /* KMERHASH_HASHTABLE_OA_LP_DOUBLING_HPP_ */
