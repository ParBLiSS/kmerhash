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
 * robinhood_offset_hashmap.hpp
 *
 * current testing version.  has the following featuers
 * 1. batch mode interface
 * 2. prefetching for batch modes
 * 3. using offsets
 * 4. separate key and value arrays did not work - prob more mem usage...
 *
 * attempts to use _mm_stream_load_si128 will not succeed on x86 with write back memory.  see
 * https://software.intel.com/en-us/forums/intel-isa-extensions/topic/597075
 * https://stackoverflow.com/questions/40140728/why-intel-compiler-ignores-the-non-temporal-prefetch-pragma-directive-for-intel
 * https://stackoverflow.com/questions/40096894/do-current-x86-architectures-support-non-temporal-loads-from-normal-memory
 * on KNL there is some hope.
 *
 *  Created on: July 13, 2017
 *      Author: tpan
 *
 *  for robin hood hashing
 */

#ifndef KMERHASH_ROBINHOOD_OFFSET_HASHMAP_PAD_HPP_
#define KMERHASH_ROBINHOOD_OFFSET_HASHMAP_PAD_HPP_

#include <vector>   // for vector.
#include <array>
#include <type_traits> // for is_constructible
#include <iterator>  // for iterator traits
#include <algorithm> // for transform

#include "kmerhash/aux_filter_iterator.hpp"   // for join iteration of 2 iterators and filtering based on 1, while returning the other.
#include "kmerhash/math_utils.hpp"
//#include "mmintrin.h"  // emm: _mm_stream_si64

#include <x86intrin.h>
#include "immintrin.h"

#include "containers/fsc_container_utils.hpp"
#include "iterators/transform_iterator.hpp"

#include "kmerhash/hash_new.hpp"

#include <stdlib.h> // posix_memalign
#include <stdexcept>

#include "kmerhash/hyperloglog64.hpp"  // for size estimation.

#include "utils/benchmark_utils.hpp"
#include "kmerhash/mem_utils.hpp"

#include <memory>

#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif


// should be easier for prefetching
#if ENABLE_PREFETCH
#include "xmmintrin.h" // prefetch related.
#define KH_PREFETCH(ptr, level)  _mm_prefetch(reinterpret_cast<const char*>(ptr), level)
#else
#define KH_PREFETCH(ptr, level)
#endif


namespace fsc {
/// when inserting, does NOT replace existing.
struct DiscardReducer {
	template <typename T>
	inline T operator()(T const & x, T const & y) {
		return x;
	}
};
/// when inserting, REPALCES the existing
struct ReplaceReducer {
	template <typename T>
	inline T operator()(T const & x, T const & y) {
		return y;
	}
};
/// other reducer types include plus, max, etc.


/**
 * @brief Open Addressing hashmap that uses Robin Hood hashing, with doubling for resizing, circular internal array.  modified from standard robin hood hashmap to use bucket offsets, in attempt to improve speed.
 * @details  at the moment, nothing special for k-mers yet.
 * 		This class has the following implementation characteristics
 * 			vector of structs
 * 			open addressing with robin hood hashing.
 * 			doubling for reallocation
 * 			circular internal array
 *
 * 	@note	hash function "batch_size":  this class only supports hash function with a public static constexpr member of unsigned integral type, named "batch_size".
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
 *  [x] estimate distinct element counts in input.
 *
 *  [ ] verify that iterator returned has correct data offset and info offset (which are different)
 *
 *  [x] insert_with_hint (iterator version) needs to increase the size when lsize is greater than max load
 *  [x] insert_with_hint needs to increase size when any has 127 for offset.
 *  [ ] insert_integrated needs to increase size when any has 127 for offset. (worth doing?)
 *  [x] downsize needs to prevent offset of 127.
 *
 *  [ ] update erase
 *  [ ] update insert.
 *  [x] use array instead of vector.
 *  [ ] padding with prefetch dist or hash batch size (this is currently hardcoded to 32).
 */
template <typename Key, typename T,
		template <typename> class Hash = ::std::hash,
		template <typename> class Equal = ::std::equal_to,
		typename Allocator = ::std::allocator<std::pair<const Key, T> >,
		typename Reducer = ::fsc::DiscardReducer
		>
class hashmap_robinhood_offsets_reduction {


public:

	using key_type              = Key;
	using mapped_type           = T;
	using value_type            = ::std::pair<Key, T>;
	using hasher                = Hash<Key>;
	using key_equal             = Equal<Key>;
	using reducer               = Reducer;

protected:

	mutable uint8_t INSERT_LOOKAHEAD;
	mutable uint8_t QUERY_LOOKAHEAD;
	mutable uint8_t INSERT_LOOKAHEAD_MASK;
	mutable uint8_t QUERY_LOOKAHEAD_MASK;

	mutable uint8_t PADDING;

	template <typename S>
	struct modulus2 {

		static_assert(((sizeof(S) & (sizeof(S) - 1)) == 0) && (sizeof(S) <= 8), "only support 4- and 8-byte elements up to 8 bytes right now.");
		//==== AVX and SSE code commmented out because they are not correct and are causing problems with even copy_upsize.
//#if defined(__AVX2__)
//		static constexpr size_t batch_size = 32 / sizeof(S);
//#elif defined(__SSE2__)
//		static constexpr size_t batch_size = 16 / sizeof(S);
//#else
		static constexpr size_t batch_size = 1;
//#endif
//
//#if defined(__AVX2__)
//		__m256i vmask;
//#elif defined(__SSE2__)
//		__m128i vmask;
//#endif
		S mask;

		modulus2(S const & _mask) :
//#if defined(__AVX2__)
//				vmask(sizeof(S) == 4 ?  _mm256_set1_epi32(_mask) : _mm256_set1_epi64x(_mask)),
//#elif defined(__SSE2__)
//				vmask(sizeof(S) == 4 ?  _mm_set1_epi32(_mask) : _mm_set1_epi64x(_mask)),
//#endif
						mask(_mask)
				{}

		template <typename IN>
		inline IN operator()(IN const & x) const { return (x & mask); }

		// for in and out being different types.
		template <typename IN, typename OUT>
		inline void operator()(IN const * x, size_t const & _count, OUT * y) const {
			// TODO: [ ] do SSE version here
			for (size_t i = 0; i < _count; ++i)  y[i] = x[i] & mask;
		}

//#if defined(__AVX2__)
//		// when input nad output are the same types
//		template <typename IN>
//		inline void operator()(IN const * x, size_t const & _count, IN * y) const {
//			// 32 bytes at a time.  input should be
//			int i = 0;
//			int imax;
//
//			__m256i xx;
//			for (i = 0, imax = _count - batch_size; i < imax; i += batch_size)  {
//				xx = _mm256_lddqu_si256(reinterpret_cast<__m256i const *>(x + i));
//				xx = _mm256_and_si256(xx, vmask);
//				_mm256_storeu_si256(reinterpret_cast<__m256i *>(y + i), xx);
//			}
//			for (imax = _count; i < imax; ++i)
//				y[i] = x[i] & mask;
//		}
//#elif defined(__SSE2__)
//		// when input nad output are the same types
//		template <typename IN>
//		inline void operator()(IN const * x, size_t const & _count, IN * y) const {
//			// 32 bytes at a time.  input should be
//			int i = 0;
//			int imax;
//
//			__m128i xx;
//			for (i = 0, imax = _count - batch_size; i < imax; i += batch_size)  {
//				xx = _mm_lddqu_si128(reinterpret_cast<__m128i const *>(x + i));
//				xx = _mm_and_si128(xx, vmask);
//				_mm_storeu_si128(reinterpret_cast<__m128i *>(y + i), xx);
//			}
//			for (imax = _count; i < imax; ++i)
//				y[i] = x[i] & mask;
//		}
//
//#endif
	};

	// mod 2 okay since hashtable size is always power of 2.
	using InternalHash = ::fsc::hash::TransformedHash<Key, Hash, ::bliss::transform::identity, modulus2>;
  using hash_val_type = typename InternalHash::HASH_VAL_TYPE;


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
		x |= info_empty;
	}
	inline void set_normal(info_type & x) {
		x &= info_mask;
	}
	inline info_type get_offset(info_type const & x) const {
		return x & info_mask;
	}
	// make above explicit by preventing automatic type conversion.
	template <typename TT> inline bool is_empty(TT const & x) const  = delete;
	template <typename TT> inline bool is_normal(TT const & x) const = delete;
	template <typename TT> inline void set_empty(TT & x) = delete;
	template <typename TT> inline void set_normal(TT & x) = delete;
	template <typename TT> inline info_type get_offset(TT const & x) const = delete;

	//=========  end INFO_TYPE definitions.
	// filter
	struct valid_entry_filter {
		inline bool operator()(info_type const & x) {   // a container entry is empty only if the corresponding info is empty (0x80), not just have empty flag set.
			return x != info_empty;   // (curr bucket is empty and position is also empty.  otherwise curr bucket is here or prev bucket is occupying this)
		};
	};



	using container_type		= value_type*;
	using info_container_type	= ::std::vector<info_type, Allocator>;
	hyperloglog64<key_type, hasher, 12> hll;  // precision of 12bits  error rate : 1.04/(2^6)


public:

	using allocator_type        = Allocator;
	using reference 			= value_type &;
	using const_reference	    = value_type const &;
	using pointer				= value_type *;
	using const_pointer		    = value_type const *;
	using iterator              = ::bliss::iterator::aux_filter_iterator<pointer, typename info_container_type::iterator, valid_entry_filter>;
	using const_iterator        = ::bliss::iterator::aux_filter_iterator<const_pointer, typename info_container_type::const_iterator, valid_entry_filter>;
	using size_type             = typename info_container_type::size_type;
	using difference_type       = typename info_container_type::difference_type;


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
	static constexpr bucket_id_type bid_pos_mask = ~(static_cast<bucket_id_type>(0)) >> 1;   // lower 63 bits set.
	static constexpr bucket_id_type bid_pos_exists = 1ULL << 63;  // 64th bit set.
	// failed is speial, correspond to all bits set (max distnace failed).  not using 0x800000... because that indicates failed inserting due to occupied.
	static constexpr bucket_id_type insert_failed = bid_pos_mask;  // unrealistic value that also indicates it's missing.
	static constexpr bucket_id_type find_failed = bid_pos_mask;  // unrealistic value that also indicates it's missing.


	inline bucket_id_type make_missing_bucket_id(size_t const & pos) const { //, info_type const & info) const {
		assert(pos < bid_pos_exists);
		return static_cast<bucket_id_type>(pos);
	}
	inline bucket_id_type make_existing_bucket_id(size_t & pos) const { //, info_type const & info) const {
		return static_cast<bucket_id_type>(pos) | bid_pos_exists;
	}

	inline bool present(bucket_id_type const & x) const {
		//return (x & bid_pos_exists) > 0;
		return x > bid_pos_mask;
	}
	inline bool missing(bucket_id_type const & x) const {
		// return (x & bid_pos_exists) == 0;
		return x < bid_pos_exists;
	}

	inline size_t get_pos(bucket_id_type const & x) const {
		return x & bid_pos_mask;
	}

	// make above explicit by preventing automatic type conversion.
	template <typename TT> inline bool present(TT const & x) const = delete;
	template <typename TT> inline bool missing(TT const & x) const = delete;
	template <typename TT> inline size_t get_pos(TT const & x) const = delete;


	//=========  end BUCKET_ID_TYPE definitions.


	// =========  prefetch constants.
	static constexpr bucket_id_type cache_align_mask = ~(64ULL - 1ULL);


	static constexpr uint32_t info_per_cacheline = 64 / sizeof(info_type);
	static constexpr uint32_t value_per_cacheline = 64 / sizeof(value_type);
	// =========  END prefetech constants.


	size_t lsize;
	mutable size_t buckets;
	mutable size_t mask;
	mutable size_t min_load;
	mutable size_t max_load;
	mutable double min_load_factor;
	mutable double max_load_factor;

#if defined(REPROBE_STAT)
	// some stats.
	mutable size_type upsize_count;
	mutable size_type downsize_count;
	mutable size_type reprobes;   // for use as temp variable
	mutable info_type max_reprobes;
	mutable size_type moves;
	mutable size_type max_moves;
	mutable size_type shifts;
	mutable size_type max_shifts;
#endif


	valid_entry_filter filter;
	hasher hash;
	InternalHash hash_mod2;
	key_equal eq;
	reducer reduc;

	container_type container;
	info_container_type info_container;

public:

	/**
	 * _capacity is the number of usable entries, not the capacity of the underlying container.
	 */
	explicit hashmap_robinhood_offsets_reduction(size_t const & _capacity = 128,
			double const & _min_load_factor = 0.4,
			double const & _max_load_factor = 0.9,
			uint8_t const & _insert_lookahead = 8,
			uint8_t const & _query_lookahead = 16) :
			INSERT_LOOKAHEAD(_insert_lookahead), QUERY_LOOKAHEAD(_query_lookahead),
			INSERT_LOOKAHEAD_MASK(_insert_lookahead * 2 - 1), QUERY_LOOKAHEAD_MASK(_query_lookahead * 2 - 1),
			PADDING(::std::max(static_cast<uint8_t>(32), ::std::max(_insert_lookahead, _query_lookahead))),
			lsize(0), buckets(next_power_of_2(_capacity)), mask(buckets - 1),
#if defined (REPROBE_STAT)
			upsize_count(0), downsize_count(0),
#endif
			// hash(123457),   // not all hash functions have constructors that takes seeds.  e.g. std::hash.  goal of this hashmap is to be general.
			hash_mod2(hash, ::bliss::transform::identity<Key>(), modulus2<hash_val_type>(mask)),
			container(nullptr), info_container(buckets + info_empty + PADDING, info_empty)
	{
		container = ::utils::mem::aligned_alloc<value_type>(buckets + info_empty + PADDING);

		// set the min load and max load thresholds.  there should be a good separation so that when resizing, we don't encounter a resize immediately.
		set_min_load_factor(_min_load_factor);
		set_max_load_factor(_max_load_factor);
	};

	/**
	 * initialize and insert, allocate about 1/4 of input, and resize at the end to bound the usage.
	 */
	template <typename Iter, typename = typename std::enable_if<
			::std::is_constructible<value_type, typename ::std::iterator_traits<Iter>::value_type>::value  ,int>::type >
	hashmap_robinhood_offsets_reduction(Iter begin, Iter end,
			double const & _min_load_factor = 0.4,
			double const & _max_load_factor = 0.9,
			uint8_t const & _insert_lookahead = 8,
			uint8_t const & _query_lookahead = 16) :
			hashmap_robinhood_offsets_reduction(::std::distance(begin, end) / 4,
					_min_load_factor, _max_load_factor, _insert_lookahead, _query_lookahead) {

		insert(begin, end);
	}

	~hashmap_robinhood_offsets_reduction() {
		free(container);

#if defined(REPROBE_STAT)
		::std::cout << "RESIZE SUMMARY:\tupsize\t= " << upsize_count << "\tdownsize\t= " << downsize_count << std::endl;
#endif
	}


	hashmap_robinhood_offsets_reduction(hashmap_robinhood_offsets_reduction const & other) :
		INSERT_LOOKAHEAD(other.INSERT_LOOKAHEAD),
		QUERY_LOOKAHEAD(other.QUERY_LOOKAHEAD),
		INSERT_LOOKAHEAD_MASK(other.INSERT_LOOKAHEAD_MASK),
		QUERY_LOOKAHEAD_MASK(other.QUERY_LOOKAHEAD_MASK),
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
		hash_mod2(other.hash_mod2),
		eq(other.eq),
		reduc(other.reduc),
		info_container(other.info_container) {

		if (container != nullptr) { free(container);  container = nullptr; }
		container = ::utils::mem::aligned_alloc<value_type>(buckets + info_empty + PADDING);
		memcpy(container, other.container, (buckets + info_empty + PADDING) * sizeof(value_type));
	};

	hashmap_robinhood_offsets_reduction & operator=(hashmap_robinhood_offsets_reduction const & other) {
		INSERT_LOOKAHEAD = other.INSERT_LOOKAHEAD;
		QUERY_LOOKAHEAD = other.QUERY_LOOKAHEAD;
		INSERT_LOOKAHEAD_MASK = other.INSERT_LOOKAHEAD_MASK;
		QUERY_LOOKAHEAD_MASK = other.QUERY_LOOKAHEAD_MASK;
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
    hash_mod2 = other.hash_mod2;
		eq = other.eq;
		reduc = other.reduc;
		info_container = other.info_container;

		if (container != nullptr) { free(container);  container = nullptr; }
		container = ::utils::mem::aligned_alloc<value_type>(buckets + info_empty + PADDING);
		memcpy(container, other.container, (buckets + info_empty + PADDING) * sizeof(value_type));
	}

	hashmap_robinhood_offsets_reduction(hashmap_robinhood_offsets_reduction && other) :
		INSERT_LOOKAHEAD(std::move(other.INSERT_LOOKAHEAD)),
		QUERY_LOOKAHEAD(std::move(other.QUERY_LOOKAHEAD)),
		INSERT_LOOKAHEAD_MASK(std::move(other.INSERT_LOOKAHEAD_MASK)),
		QUERY_LOOKAHEAD_MASK(std::move(other.QUERY_LOOKAHEAD_MASK)),

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
		hash_mod2(std::move(other.hash_mod2)),
		eq(std::move(other.eq)),
		reduc(std::move(other.reduc)) {

		std::swap(container, other.container);  // swap the two...
		info_container.swap(other.info_container);
	}

	hashmap_robinhood_offsets_reduction & operator=(hashmap_robinhood_offsets_reduction && other) {
		INSERT_LOOKAHEAD = std::move(other.INSERT_LOOKAHEAD);
		QUERY_LOOKAHEAD = std::move(other.QUERY_LOOKAHEAD);
		INSERT_LOOKAHEAD_MASK = std::move(other.INSERT_LOOKAHEAD_MASK);
		QUERY_LOOKAHEAD_MASK = std::move(other.QUERY_LOOKAHEAD_MASK);

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
		reduc = std::move(other.reduc);

		std::swap(container, other.container);  // swap the two...
		info_container.swap(other.info_container);

	}

	void swap(hashmap_robinhood_offsets_reduction && other) {
		std::swap(INSERT_LOOKAHEAD, std::move(other.INSERT_LOOKAHEAD));
		std::swap(QUERY_LOOKAHEAD, std::move(other.QUERY_LOOKAHEAD));
		std::swap(INSERT_LOOKAHEAD_MASK, std::move(other.INSERT_LOOKAHEAD_MASK));
		std::swap(QUERY_LOOKAHEAD_MASK, std::move(other.QUERY_LOOKAHEAD_MASK));
		std::swap(hll, std::move(other.hll));
		std::swap(lsize, std::move(other.lsize));
		std::swap(buckets, std::move(other.buckets));
		std::swap(mask, std::move(other.mask));
		std::swap(min_load, std::move(other.min_load));
		std::swap(max_load, std::move(other.max_load));
		std::swap(min_load_factor, std::move(other.min_load_factor));
		std::swap(max_load_factor, std::move(other.max_load_factor));
#if defined(REPROBE_STAT)
		// some stats.
		std::swap(upsize_count, std::move(other.upsize_count));
		std::swap(downsize_count, std::move(other.downsize_count));
		std::swap(reprobes, std::move(other.reprobes));
		std::swap(max_reprobes, std::move(other.max_reprobes));
		std::swap(moves, std::move(other.moves));
		std::swap(max_moves, std::move(other.max_moves));
		std::swap(shifts, std::move(other.shifts));
		std::swap(max_shifts, std::move(other.max_shifts));
#endif
		std::swap(filter, std::move(other.filter));
		std::swap(hash, std::move(other.hash));
		std::swap(hash_mod2, other.hash_mod2);

		std::swap(eq, std::move(other.eq));
		std::swap(reduc, std::move(other.reduc));
		std::swap(container, std::move(other.container));
		std::swap(info_container, std::move(other.info_container));
	}


	/**
	 * @brief set the load factors.
	 */
	inline void set_min_load_factor(double const & _min_load_factor) {
		min_load_factor = _min_load_factor;
		min_load = static_cast<size_t>(static_cast<double>(buckets) * min_load_factor);

	}

	inline void set_max_load_factor(double const & _max_load_factor) {
		max_load_factor = _max_load_factor;
		max_load = static_cast<size_t>(::std::ceil(static_cast<double>(buckets) * max_load_factor));
	}


	/**
	 * @brief set the lookahead values.
	 */
	inline void set_insert_lookahead(uint8_t const & _insert_lookahead) {
		INSERT_LOOKAHEAD = _insert_lookahead;
		INSERT_LOOKAHEAD_MASK = (INSERT_LOOKAHEAD << 1) - 1;
	}
	inline void set_query_lookahead(uint8_t const & _query_lookahead) {
		QUERY_LOOKAHEAD = _query_lookahead;
		QUERY_LOOKAHEAD_MASK = (QUERY_LOOKAHEAD << 1) - 1;
	}

	inline void set_ignored_msb(uint8_t const & ignore_msb) {
		this->hll.set_ignored_msb(ignore_msb);
	}

	inline hyperloglog64<key_type, hasher, 12>& get_hll() {
		return this->hll;
	}


	/**
	 * @brief get the load factors.
	 */
	inline double get_load_factor() {
		return static_cast<double>(lsize) / static_cast<double>(buckets);
	}

	inline double get_min_load_factor() {
		return min_load_factor;
	}

	inline double get_max_load_factor() {
		return max_load_factor;
	}

	size_t capacity() {
		return buckets;
	}


	/**
	 * @brief iterators
	 */
	iterator begin() {
		return iterator(container, info_container.begin(), info_container.begin() + buckets + info_empty, filter);
	}

	iterator end() {
		return iterator(container + buckets + info_empty, info_container.begin() + buckets + info_empty, filter);
	}

	const_iterator cbegin() const {
		return const_iterator(container, info_container.cbegin(), info_container.cbegin() + buckets + info_empty, filter);
	}

	const_iterator cend() const {
		return const_iterator(container + buckets + info_empty, info_container.cbegin() + buckets + info_empty, filter);
	}



	void print() const {
		std::cout << "lsize " << lsize << "\tbuckets " << buckets << "\tmax load factor " << max_load_factor <<
				"\tinsert_lookahead " << INSERT_LOOKAHEAD << "\tquery_lookahead " << QUERY_LOOKAHEAD <<
				std::endl;
		size_type i = 0, j = 0;

		container_type tmp = ::utils::mem::aligned_alloc<value_type>(::std::numeric_limits<info_type>::max());
		container_type it;
		size_t offset = 0, len = 0;
		for (; i < buckets; ++i) {
			std::cout << "buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
					std::endl;


			if (! is_empty(info_container[i])) {
				offset = i + get_offset(info_container[i]);
				len = 1 + get_offset(info_container[i + 1]) - get_offset(info_container[i]);
				memcpy(tmp, container + offset, len);
				std::sort(tmp, tmp + len, [](value_type const & x,
						value_type const & y){
					return x.first < y.first;
				});
				for (j = 0; j < len; ++j) {
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
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}
	}

	void print_raw() const {
		std::cout << "lsize " << lsize << "\tbuckets " << buckets << "\tmax load factor " << max_load_factor <<
				"\tinsert_lookahead " << INSERT_LOOKAHEAD << "\tquery_lookahead " << QUERY_LOOKAHEAD <<
				std::endl;
		size_type i = 0;

		for (i = 0; i < buckets; ++i) {
			std::cout << "buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
					"\n" << std::setw(72) << i <<
					", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
					", key: " << container[i].first <<
					", val: " << container[i].second <<
					std::endl;
		}

		for (i = buckets; i < info_container.size(); ++i) {
			std::cout << "PAD: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
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
				"\t printing [" << first << " .. " << last << "]" <<
				"\tinsert_lookahead " << INSERT_LOOKAHEAD << "\tquery_lookahead " << QUERY_LOOKAHEAD <<
				std::endl;
		size_type i = 0;

		for (i = first; i <= last; ++i) {
			std::cout << prefix <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
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
				"\t printing [" << first << " .. " << last << "]" <<
				"\tinsert_lookahead " << INSERT_LOOKAHEAD << "\tquery_lookahead " << QUERY_LOOKAHEAD <<
				std::endl;
		size_type i = 0, j = 0;

		size_t offset = 0, len = 0;
		for (i = first; i <= last; ++i) {
			len = std::max(len,  1UL + get_offset(info_container[i+1]) - get_offset(info_container[i]));
		}

		container_type tmp = ::utils::mem::aligned_alloc<value_type>(len);

		for (i = first; i <= last; ++i) {
			std::cout << prefix <<
					" buc: " << std::setw(10) << i <<
					", inf: " << std::setw(3) << static_cast<size_t>(info_container[i]) <<
					", off: " << std::setw(3) << static_cast<size_t>(get_offset(info_container[i])) <<
					", pos: " << std::setw(10) << (i + get_offset(info_container[i])) <<
					", cnt: " << std::setw(3) << (is_empty(info_container[i]) ? 0UL : (get_offset(info_container[i+1]) - get_offset(info_container[i]) + 1)) <<
					std::endl;


			if (! is_empty(info_container[i])) {
				offset = i + get_offset(info_container[i]);
				len = (i + 1 + get_offset(info_container[i + 1]) - offset);
				memcpy(tmp, container+offset, sizeof(value_type) * len);
				std::sort(tmp, tmp + len, [](value_type const & x,
						value_type const & y){
					return x.first < y.first;
				});
				for (j = 0; j < len; ++j) {
					std::cout << prefix <<
							" " << std::setw(72) << (offset + j) <<
							", hash: " << std::setw(16) << std::hex << (hash(container[i].first) & mask) << std::dec <<
							", key: " << std::setw(22) << tmp[j].first <<
							", val: " << std::setw(22) << tmp[j].second <<
							std::endl;
				}
			}
		}
		free(tmp);
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
		rehash(static_cast<size_t>(::std::ceil(static_cast<double>(n) / this->max_load_factor)));
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

//#if defined(REPROBE_STAT)
		std::cout << "REHASH current " << buckets << " request " << b << " nears 2^x " << n << " lsize " << lsize << std::endl;
//#endif

		//		// early termination
		//		if (lsize == 0) {
		//		  container.resize(n + std::numeric_limits<info_type>::max() + 1);
		//		  info_container.resize(n + std::numeric_limits<info_type>::max() + 1, info_empty);
		//
		//      buckets = n;
		//      mask = n - 1;
		//
		//      min_load = static_cast<size_type>(static_cast<double>(n) * min_load_factor);
		//      max_load = static_cast<size_type>(static_cast<double>(n) * max_load_factor);
		//
		//		  return;
		//		}

		size_t max_offset;
		if ((n != buckets) && (lsize < static_cast<size_t>(::std::ceil(max_load_factor * static_cast<double>(n))))) {
			// don't resize if lsize is larger than the new max load.

			if ((lsize > 0) && (n < buckets)) {  // down sizing. check if we overflow info
				while ((max_offset = this->copy_downsize_max_offset(n)) > 127)  { // if downsizing creates offset > 127, then increase n and check again.
					n <<= 1;
					//          std::cout << "REHASH DOWN INFO FIELD OVERFLOW. " <<  max_offset << " INCREASE n to " << n << std::endl;
				}
			}
#if defined(REPROBE_STAT)
			std::cout << "REHASH_final current" << buckets << " request " << b << " nears 2^x " << n << " lsize " << lsize << std::endl;
#endif

			// if after checking we cannot downsize, then we stop and return.
			if ( n == buckets )  return;



			// this MAY cause infocontainer to be evicted from cache...
			container_type tmp = ::utils::mem::aligned_alloc<value_type>(n + info_empty + PADDING);
			info_container_type tmp_info(n + info_empty + PADDING, info_empty);

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
			this->hash_mod2.posttrans.mask = mask;  // increase mask..

			min_load = static_cast<size_t>(::std::ceil(static_cast<double>(n) * min_load_factor));
			max_load = static_cast<size_t>(::std::ceil(static_cast<double>(n) * max_load_factor));

			// swap in.
			free(container);
			container = tmp;
			info_container.swap(tmp_info);
		}
	}


protected:
	// checks and makes sure that we don't have offsets greater than 127.
	// return max_offset have to just try and see, no prediction right now.
	size_t copy_downsize_max_offset(size_type const & target_buckets) {
		assert((target_buckets & (target_buckets - 1)) == 0);   // assert this is a power of 2.


		if (target_buckets >= buckets) return 0;

		size_t id = 0, bid = 0;

		//    std::cout << "RESIZE DOWN " << target_buckets << std::endl;

		size_t new_end = 0;

		size_t blocks = buckets / target_buckets;

		size_t bl;

		// strategies:
		// 1. fill in target, then throw away if fails.  read 1x, write 1x normal, failure read 2x, write 2x
		// 2. fill in target_info first, then target.  read 2x (second time faster in L3?) write 1x normal.  write 1.5 x failure
		// 3. compute a max offset value.  read 2x (second time faster in L3?) write 1x normal or failure
		// choose 3.

		// calculate the max offset.
		size_t max_offset = 0;

		// iterate over all matching blocks.  fill one target bucket at a time and immediately fill the target info.
		for (bid = 0; bid < target_buckets; ++bid) {
			// if end of last bucket offset is higher than current target bucket id, calc the new offset for curr bucket.  else it's a no op.

			if (new_end > bid) {
				max_offset = std::max(max_offset, new_end - bid);
			} else {
				new_end = bid;
			}

			// early termination
			if (max_offset > 127) {
#if defined(REPROBE_STAT)
				std::cout << "MAX OFFSET early. " <<  max_offset << " for bucket " << bid << " new end " << new_end << std::endl;
#endif
				return max_offset;
			}

			for (bl = 0; bl < blocks; ++bl) {
				id = bid + bl * target_buckets;  // id within each block.

				if (is_normal(info_container[id])) {
					// get the range
					new_end += (1 + get_offset(info_container[id + 1]) - get_offset(info_container[id]));
				}
			}

		}

#if defined(REPROBE_STAT)
		std::cout << "MAX OFFSET full. " <<  max_offset << " for bucket " << bid << " new end " << new_end << " target buckets " << target_buckets << std::endl;
#endif

		//  std::cout << " info: " << (target_buckets - 1) << " info " << static_cast<size_t>(target_info[target_buckets - 1]) << " entry " << target[target_buckets - 1].first << std::endl;
		// adjust the target_info at the end, in the padding region.
		// new_end is end of all occupied entries.  target_bucket is the last bid.
		if (new_end > target_buckets)
			max_offset = std::max(max_offset, new_end - target_buckets);

		return max_offset;
	}


	void copy_downsize(container_type & target, info_container_type & target_info,
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
					pos = id + get_offset(info_container[id]);
					endd = id + 1 + get_offset(info_container[id + 1]);

					// copy the range.
					//        std::cout << id << " infos " << static_cast<size_t>(info_container[id]) << "," << static_cast<size_t>(info_container[id + 1]) << ", " <<
					//        		" copy from " << pos << " to " << new_end << " length " << (endd - pos) << std::endl;
					memmove((target + new_end), (container + pos), sizeof(value_type) * (endd - pos));

					new_end += (endd - pos);

				}
			}

			// offset - current bucket id.
			target_info[bid] = ((new_end - new_start) == 0 ? info_empty : info_normal) + new_start - bid;
		}
		//  std::cout << " info: " << (target_buckets - 1) << " info " << static_cast<size_t>(target_info[target_buckets - 1]) << " entry " << target[target_buckets - 1].first << std::endl;
		// adjust the target_info at the end, in the padding region.
		//		for (bid = target_buckets; bid < new_end; ++bid) {
		//			new_start = std::max(bid, new_end);  // fixed new_end.  get new start.
		//			// if last one is not empty, then first padding position is same distance with
		//			target_info[bid] = info_empty + new_start - bid;
		//			//    std::cout << " info: " << bid << " info " << static_cast<size_t>(target_info[bid]) << " entry " << target[bid].first << std::endl;
		//		}
		for (bid = target_buckets; bid < new_end; ++bid) {
			// if last one is not empty, then first padding position is same distance with
			target_info[bid] = info_empty + new_end - bid;
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
		assert((target_buckets & (target_buckets - 1)) == 0);   // assert this is a power of 2.

		uint8_t log_buckets = std::log2(buckets);  // should always be power of 2
		std::cout << "buckets " << buckets << " log of it " << static_cast<size_t>(log_buckets) << std::endl;

		size_t id, bid, p;
		size_t pos;
		size_t endd;
		value_type v;

		size_t bl;
		size_t blocks = target_buckets / buckets;
		std::vector<size_t> offsets(blocks + 1, 0);
		std::vector<size_t> len(blocks, 0);

//		std::cout << "offsets: " << offsets[0];
		// prefill with ideal offsets.
		for (bl = 0; bl < blocks; ++bl) {
		  offsets[bl + 1] = bl * buckets;
//			std::cout << ", " << offsets[bl + 1];
		}
//		std::cout << std::endl;


    // std::cout << "RESIZE UP from " << buckets << " to " << target_buckets << ", with blocks " << blocks << std::endl;

		// let's store the hash in order to avoid redoing hash.  This is needed only because we need to first count the number in a block,
		//  so that at block boundaries we have the right offsets.
		hash_val_type * hashes = ::utils::mem::aligned_alloc<hash_val_type>(info_container.size());

		// compute and store all hashes,
		InternalHash h2(hash, ::bliss::transform::identity<Key>(), modulus2<hash_val_type>(target_buckets - 1));
		h2(container, buckets + info_empty, hashes);  // compute even for empty positions.
		// load should be high so there should not be too much waste.  also, SSE and AVX.

		// try to compute the offsets, so that we can figure out exactly where to store the higher block data.
		// PROBLEM STATEMENT:  we want to find the index q_i of the last entry of a block i,
		//      block position p_i may be empty or shifted by some distance <= (q_(i-1) - i * buckets).  empty positions can be used to absorb overflow of prev block
		//    NOTE THAT WE ASSUME TRAVERSAL OF ORIGINAL HASH TABLE IN ORDER, SO IN-BLOCK TRAVERSAL IS ALSO IN ORDER in new table
		//		thus the HASH BUCKET ID should be monotonically increasing.
		// CHALLENGES:
		//      start of block is shifted by (q_(i-1) - i*buckets), but the shift may be absorbed by empty space in block i.
		//      The shift may be added to by entries in block i, such that for block (i+1), the starting position may not be shifted by q_i.
		//      we want to calculate this offset exactly while going over the input (hash) just once.
		// Observations:
		//  1. a block can only absorb as many as there are empty slots between [i *buckets, (i+1) * buckets), beyond which the overflow must extend.
		//  2. overflow in region [(i+1)*buckets, q_i) does not contain empty slots.
		//  3. absorption:  absorb o_(i-1) = q_(i-1) - i*buckets
		//        if number of empty is greater or equal to o_(i-1), complete absorption within the block boundary, and o_i is handled anew.
		//        if number of empty is less than o_(i-1), we have partial absorption within block boundary, and the remainder is added to o_i,  note that o_i region have no empty to absorb from o_(i-1)
		//        if no overflow, no absorption, and o_i is handled anew.
		//  4. the exact position assignment does not matter - these will be computed later.
		//
		// Algorithm:
		//   we need to track the number of empty entries, and the amount of overflow.
		//    compute the overflow prefix scan while doing absorption.
		//  1. iterate over all new bucket ids (i.e. check all entries.), compute per block
		//    a. non-empty count within block range -> empty count within block range
		//    b. max offset assuming starting from i*buckets.
		//  2. prefix scan of overflow:
		//      o_i += max(o_(i-1), empty_i) - empty_i.
		//REQUIRE: hash values in original array be in increasing order - this SHOULD BE TRUE.

		// step 1.  compute the POSITIONS and COUNTS.
		size_t cnt = 0;
		for (bid = 0; bid < buckets; ++bid) {
			if (is_normal(info_container[bid])) {

				pos = bid + get_offset(info_container[bid]);
				endd = bid + 1 + get_offset(info_container[bid + 1]);

				for (p = pos; p < endd; ++p, ++cnt) {
					id = hashes[p];

					// figure out which block it is in.
					bl = id >> log_buckets;
//					if (bl == 1) {
//					  std::cout << " orig bucket " << std::hex << bid  << std::dec  << " pos " << p << " new bucket " << std::hex << id  << " mask " << mask << std::dec << std::flush;
//					  std::cout << " block " << bl << " curr len " << len[bl] << " curr offset " << offsets[bl+1] <<  std::flush;
//					}

					// count.  at least the bucket id + 1, or last insert target position + 1.
					// increment by at least 1, or by the target bucket id (counting empty) within the current block

					// offsets store the maximum NEXT offset of the entries in the block.
					// note that id should increase within each block, but really should not JUMP BACK.
					offsets[bl+1] = std::max(offsets[bl+1], id) + 1; // where the current entry would go, +1 (for next empty entry).

					len[bl] += 1;
//					if (bl == 1) std::cout << " new offset " << offsets[bl+1] << " new len " << len[bl] << std::endl;
				}
			}
		}

//		std::cout << "after update1 [offsets, len]: " << offsets[0];
//		// prefill with ideal offsets.
//		for (bl = 0; bl < blocks; ++bl) {
//			std::cout << ":" << len[bl] << ", " << offsets[bl + 1];
//		}
//		std::cout << std::endl;

		// now compute the overflows.  at this point, we have count in each block, and offsets starting from block boundaries
		// and overflow in previous block is not yet considered.
		for (bl = 1; bl <= blocks ; ++bl) {
//			std::cout << "FINAL block offset " << offsets[bl] << " len in ideal region " << len[bl - 1];

			// compute the actual overflows.
			offsets[bl] = (offsets[bl] > (bl * buckets)) ? (offsets[bl] - (bl * buckets)) : 0;
			// recall that overflow region has no empty slots.  to (len - offsets[bl]) is the number in the ideal region, and buckets - that is the empty count.
			len[bl - 1] = buckets - (len[bl - 1] - offsets[bl]);  //
//			std::cout << " OVERFLOW = " << offsets[bl] << " empties " << len[bl-1] << std::endl;
		}

//		std::cout << "after update2 [offsets, len]: " << offsets[0];
//		// prefill with ideal offsets.
//		for (bl = 0; bl < blocks; ++bl) {
//			std::cout << ":" << len[bl] << ", " << offsets[bl + 1];
//		}
//		std::cout << std::endl;


		// now compute the true overflows.
		for (bl = 2; bl <= blocks ; ++bl) {
			// increase actual overflow if the block could not absorb all of it.
			offsets[bl] += std::max(len[bl-1], offsets[bl-1]) - len[bl-1];

//			std::cout << " FINAL OVERFLOW " << offsets[bl] << std::endl;
		}

//		std::cout << "after update3 [offsets]: ";
//		// prefill with ideal offsets.
//		for (bl = 0; bl <= blocks; ++bl) {
//			std::cout << ", " << offsets[bl];
//		}
//		std::cout << std::endl;


		// and convert back to offsets
    for (bl = 0; bl <= blocks ; ++bl) {
      // increase actual overflow if the block could not absorb all of it.
      offsets[bl] += bl * buckets;
//      std::cout << " FINAL Offsets " << offsets[bl] << std::endl;
    }
//		std::cout << "total cnt is " << cnt << " actual entries " << lsize << std::endl;

//	std::cout << "after final update: ";
//	// prefill with ideal offsets.
//	for (bl = 0; bl <= blocks; ++bl) {
//		std::cout << ", " << offsets[bl];
//	}
//	std::cout << std::endl;


		// now that we have the right offsets,  start moving things.
		size_t pp;
		for (bid = 0; bid < buckets; ++bid) {
		  std::fill(len.begin(), len.end(), 0);

		  if (is_normal(info_container[bid])) {

				pos = bid + get_offset(info_container[bid]);
				endd = bid + 1 + get_offset(info_container[bid + 1]);


				for (p = pos; p < endd; ++p) {
					// eval the target id.
					id = hashes[p];
//	    			std::cout << " moved bid " << bid << " from " << p << " id " << id << std::flush;

					// figure out which block it is in.
					bl = id >> log_buckets;
//					std::cout << " block " << bl << " curr offset " << offsets[bl] << std::flush;
					
					// now copy
					pp = std::max(offsets[bl], id);
//					std::cout << " to pp " << pp << std::flush;
					
					target[pp] = container[p];
					// TODO: POTENTIAL SAVINGS: no construction cost.
					//memcpy((target.data() + pp), (container.data() + p), sizeof(value_type));

//	    			std::cout << " moved from " << p << " to " << pp << " block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;

					// count.
					offsets[bl] = pp + 1;
					++len[bl];
//					std::cout << " new offset " << offsets[bl] << " len " << len[bl] << std::endl;
				}

				// update all positive ones.
				for (bl = 0; bl < blocks; ++bl) {
					id = bid + bl * buckets;
					target_info[id] = (len[bl] == 0 ? info_empty : info_normal) + static_cast<info_type>(std::max(offsets[bl] - len[bl], id) - id);
//    			std::cout << " updated info at " << id << " to " << static_cast<size_t>(target_info[id]) << ". block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;
				}
			} else {
//			  std::cout << " empty at " << bid << std::endl;

				for (bl = 0; bl < blocks; ++bl) {
					id = bid + bl * buckets;
					target_info[id] = info_empty + static_cast<info_type>(std::max(offsets[bl], id) - id);
//    			std::cout << " updated empty info at " << id << " to " << static_cast<size_t>(target_info[id]) << ". block " << bl << " with offset " << offsets[bl] << " len " << len[bl] << std::endl;
				}

			}
		}
//    for (bl = 0; bl <= blocks ; ++bl) {
//      // increase actual overflow if the block could not absorb all of it.
//      std::cout << " ACTUAL Offsets " << offsets[bl] << std::endl;
//    }

		// clean up the last part.
		size_t new_start;
		for (bid = target_buckets; bid < offsets[blocks]; ++bid) {
			new_start = std::max(bid, offsets[blocks]);  // fixed new_end.  get new start.
			// if last one is not empty, then first padding position is same distance with
			target_info[bid] = info_empty + new_start - bid;
			//		std::cout << " info: " << bid << " info " << static_cast<size_t>(target_info[bid]) << " entry " << target[bid].first << std::endl;
		}

		free(hashes);

	}


	/**
	 * return the position in container where the current key is found.  if not found, max is returned.
	 */
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate >
	bucket_id_type find_pos_with_hint(key_type const & k, size_t const & bid,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		assert(bid < buckets);

		if (! std::is_same<InPredicate, ::bliss::filter::TruePredicate>::value)
			if (!in_pred(k)) return find_failed;

		info_type offset = info_container[bid];
		size_t start = bid + get_offset(offset);  // distance is at least 0, and definitely not empty

		// no need to check for empty?  if i is empty, then this one should be.
		// otherwise, we are checking distance so doesn't matter if empty.

		// first get the bucket id
		if (is_empty(offset) ) {
			// std::cout << "Empty entry at " << i << " val " << static_cast<size_t>(info_container[i]) << std::endl;
			//return make_missing_bucket_id(start, offset);
			return make_missing_bucket_id(start);
		}

		// get the next bucket id
		size_t end = bid + 1 + get_offset(info_container[bid + 1]);   // distance is at least 0, and can be empty.

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
				if (!std::is_same<InPredicate, ::bliss::filter::TruePredicate>::value)
					if (!out_pred(container[start])) return find_failed;

				// else found one.
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
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate >
	inline bucket_id_type find_pos(key_type const & k,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {
		size_t i = hash(k) & mask;
		return find_pos_with_hint(k, i, out_pred, in_pred);
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
			end += std::max(get_offset(target_info[end]), static_cast<info_type>(1));  // move forward at least 1 (when info_normal)
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
			dist = get_offset(target_info[end]);
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
			end += std::max(get_offset(target_info[end]), static_cast<info_type>(1));  // 1 is when info is info_empty
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
	// insert with hint, while checking to see if any offset is almost overflowing.
	bucket_id_type insert_with_hint(container_type & target,
			info_container_type & target_info,
			size_t const & id,
			value_type const & v) {

		assert(id < buckets);

		// get the starting position
		info_type info = target_info[id];
		//    std::cout << "info " << static_cast<size_t>(info) << std::endl;


		// if this is empty and no shift, then insert and be done.
		if (info == info_empty) {
			set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.
			target[id] = v;
			return make_missing_bucket_id(id);
			//      return make_missing_bucket_id(id, target_info[id]);
		}

		// the bucket is either non-empty, or empty but offset by some amount

		// get the range for this bucket.
		size_t start = id + get_offset(info);
		size_t next = id + 1 + get_offset(target_info[id + 1]);

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

					// reduction if needed.  should optimize out if not needed.
					if (! std::is_same<reducer, ::fsc::DiscardReducer>::value)
						target[i].second = reduc(target[i].second, v.second);

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

		// scan for the next empty position AND update all info entries.
		size_t end = id + 1;
		for (; (end < target_info.size()) && (target_info[end] != info_empty); ++end ) {
			// loop until finding an empty spot
			if (get_offset(target_info[end]) == 127)
				return insert_failed;   // for upsizing.
		}


		if (end < next) {
			std::cout << "val " << v.first << " id " << id <<
					" info " << static_cast<size_t>(info) <<
					" start info " << static_cast<size_t>(target_info[id]) <<
					" next info " << static_cast<size_t>(target_info[id+1]) <<
					" start " << static_cast<size_t>(start) <<
					" next " << static_cast<size_t>(next) <<
					" end " << static_cast<size_t>(end) <<
					" buckets " << buckets <<
					" actual " << target_info.size() << std::endl;

			std::cout << "INFOs from start " << (id - get_offset(info)) << " to id " << id << ": " << std::endl;
			print(0, id, "\t");

			//			std::cout << "INFOs from prev " << (id - get_offset(info)) << " to id " << id << ": " << std::endl;
			//			print((id - get_offset(info)), id, "\t");
			//
			std::cout << "INFOs from id " << id << " to end " << end << ": " << std::endl;
			print(id, end, "\t");

			std::cout << "INFOs from end " << end << " to next " << next << ": " << std::endl;
			print(end, next, "\t");

			//print();
			throw std::logic_error("end should not be before next");
		}

		// now update, move, and insert.
		set_normal(target_info[id]);   // if empty, change it.  if normal, same anyways.
		for (size_t i = id + 1; i <= end; ++i) {
			++(target_info[i]);
			assert(get_offset(target_info[i]) > 0);
		}

		// now compact backwards.  first do the container via MEMMOVE
		// can potentially be optimized to use only swap, if distance is long enough.
		memmove((target + next + 1), (target + next), sizeof(value_type) * (end - next));

		// that's it.
		target[next] = v;

#if defined(REPROBE_STAT)
		this->shifts += (end - id);
		this->max_shifts = std::max(this->max_shifts, (end - id));
		this->moves += (end - next);
		this->max_moves = std::max(this->max_moves, (end - next));
#endif

		//    return make_missing_bucket_id(next, target_info[id]);
		return make_missing_bucket_id(next);

	}






	// batch insert, minimizing number of loop conditionals and rehash checks.
	// provide a set of precomputed hash values.  Also allows resize in case any estimation is not accurate.
	// RETURNs THE NUMBER REMAINING to be processed..
	template <typename IT>
	size_t insert_batch_by_hash(IT input, hash_val_type const * hashes, size_t input_size) {


#if defined(REPROBE_STAT)
//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		size_type before = lsize;
#endif
		if (input_size == 0) return 0;

		bucket_id_type id, bid1, bid;

		size_t ii;

//		std::cout << " mask " << mask << " first " << hashes[0] << " buckets " << buckets
//				<< " input_size " << input_size << " hash " << hashes[input_size -1 ] << std::endl;

//		for (size_t ii = 0; ii < input_size; ++ii) {
//			std::cout << (hashes[ii] & mask) << ", ";
//		}
//		std::cout << std::endl;

		//prefetch only if target_buckets is larger than INSERT_LOOKAHEAD
		size_t max_prefetch = std::min(input_size, static_cast<size_t>(2 * INSERT_LOOKAHEAD));
		// prefetch 2*INSERT_LOOKAHEAD of the info_container.
		//		for (ii = 0; ii < max_prefetch; ++ii) {
		//			KH_PREFETCH(reinterpret_cast<const char *>(&(*(hashes + ii))), _MM_HINT_T0);
		//
		//			// prefetch input
		//			KH_PREFETCH(reinterpret_cast<const char *>(&(*(input + ii))), _MM_HINT_T0);
		//		}

		for (ii = 0; ii < max_prefetch; ++ii) {

			id = *(hashes + ii) & mask;
			// prefetch the info_container entry for ii.
			KH_PREFETCH(reinterpret_cast<const char *>(info_container.data() + id), _MM_HINT_T0);

			//			KH_PREFETCH(reinterpret_cast<const char *>(reinterpret_cast<bucket_id_type>(info_container.data() + id) & cache_align_mask), _MM_HINT_T0);
			//			KH_PREFETCH(reinterpret_cast<const char *>(reinterpret_cast<bucket_id_type>(info_container.data() + id + 1) & cache_align_mask), _MM_HINT_T0);
			//			if ((reinterpret_cast<bucket_id_type>(info_container.data() + id + 1) & cache_align_mask) == 0)
			//			  KH_PREFETCH((const char *)(info_container.data() + id + 1), _MM_HINT_T1);

			// prefetch container as well - would be NEAR but may not be exact.
			KH_PREFETCH((const char *)(container + id), _MM_HINT_T0);

		}

		value_type val;
		IT it = input;

		// iterate based on size between rehashes
		size_t max2 = (input_size > (2*INSERT_LOOKAHEAD)) ? input_size - (2*INSERT_LOOKAHEAD) : 0;
		size_t max1 = (input_size > INSERT_LOOKAHEAD) ? input_size - INSERT_LOOKAHEAD : 0;
		size_t i = 0; //, i1 = INSERT_LOOKAHEAD, i2 = 2*INSERT_LOOKAHEAD;

		size_t lmax;
		size_t insert_bid;

		while (max2 > i) {

#if defined(REPROBE_STAT)
			std::cout << "hint: checking if rehash needed.  i = " << i << std::endl;
#endif

			// first check if we need to resize.  within 1% of
			if (static_cast<size_t>(static_cast<double>(lsize) * 1.01) >= max_load) {

//#if defined(REPROBE_STAT)
//				std::cout << "rehashing.  size = " << buckets << std::endl;
//				if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type))  > 0) {
//					std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//				} else {
//					std::cout << "STATUS: container alignment on value boundary" << std::endl;
//				}
//#endif

				return i;  // return with what's completed.
			}

			lmax = i + std::min(max_load - lsize, max2 - i);


			for (; i < lmax; ++i, ++it) {

				//				KH_PREFETCH(reinterpret_cast<const char *>(hashes + i + 2 * INSERT_LOOKAHEAD), _MM_HINT_T0);
				//				// prefetch input
				//				KH_PREFETCH(reinterpret_cast<const char *>(input + i + 2 * INSERT_LOOKAHEAD), _MM_HINT_T0);


				// prefetch container
				bid = *(hashes + i + INSERT_LOOKAHEAD) & mask;
				// intention is to write, so should prefetch...
				//				if (is_normal(info_container[bid])) {
				bid1 = bid + 1;
				bid += get_offset(info_container[bid]);
				bid1 += get_offset(info_container[bid1]);

				//					for (size_t j = bid; j < bid1; j += value_per_cacheline) {
				//						KH_PREFETCH((const char *)(container.data() + j), _MM_HINT_T0);
				//					}
				KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);

				// NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH, bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
				if (bid1 > (bid + value_per_cacheline))
					KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);

				//				}

				val = *it;
				// first get the bucket id
				insert_bid = insert_with_hint(container, info_container, *(hashes + i) & mask, val);
				if (insert_bid == insert_failed) {
				   return i;   // need to resize.
				}
				if (missing(insert_bid))
					++lsize;

				//      std::cout << "insert vec lsize " << lsize << std::endl;
				// prefetch info_container.
				bid = *(hashes + i + 2 * INSERT_LOOKAHEAD) & mask;
				KH_PREFETCH(reinterpret_cast<const char *>((info_container.data() + bid)), _MM_HINT_T0);
				//	      KH_PREFETCH(reinterpret_cast<const char *>(reinterpret_cast<bucket_id_type>((info_container.data() + bid)) & cache_align_mask), _MM_HINT_T0);
				//	      KH_PREFETCH(reinterpret_cast<const char *>(reinterpret_cast<bucket_id_type>((info_container.data() + bid + 1)) & cache_align_mask), _MM_HINT_T0);
				//	      if ((reinterpret_cast<bucket_id_type>(info_container.data() + bid + 1) & cache_align_mask) == 0)
				//	        KH_PREFETCH((const char *)(info_container.data() + bid + 1), _MM_HINT_T1);

				//            if (((bid + 1) % 64) == info_align)
				//              KH_PREFETCH((const char *)(info_container.data() + bid + 1), _MM_HINT_T0);
			}
		}


		if ((lsize + 2 * INSERT_LOOKAHEAD) >= max_load)
		  return i; // need to resize/


		// second to last INSERT_LOOKAHEAD
		for (; i < max1; ++i, ++it) {


			// === same code as in insert(1)..

			bid = *(hashes + i + INSERT_LOOKAHEAD) & mask;


			// prefetch container.  intention is to write. so should alway prefetch.
			//			if (is_normal(info_container[bid])) {
			bid1 = bid + 1;
			bid += get_offset(info_container[bid]);
			bid1 += get_offset(info_container[bid1]);

			//				for (size_t j = bid; j < bid1; j += value_per_cacheline) {
			//					KH_PREFETCH((const char *)(container.data() + j), _MM_HINT_T0);
			//				}
			KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);

			// NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH, bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
			if (bid1 > (bid + value_per_cacheline))
				KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
			//			}
			val = *it;
			insert_bid = insert_with_hint(container, info_container, *(hashes + i) & mask, val);
			if (insert_bid == insert_failed) {
			  return i;  // need to resize
			}
			if (missing(insert_bid))
				++lsize;

			//      std::cout << "insert vec lsize " << lsize << std::endl;
		}


		// last INSERT_LOOKAHEAD
		for (; i < input_size; ++i, ++it) {

			// === same code as in insert(1)..
			val = *it;
			insert_bid = insert_with_hint(container, info_container, *(hashes + i) & mask, val);
			if (insert_bid == insert_failed) {
			  return i; // need to resize;
			}
			if (missing(insert_bid))
				++lsize;

			//      std::cout << "insert vec lsize " << lsize << std::endl;

		}

		return input_size;

	}



  // batch insert, minimizing number of loop conditionals and rehash checks.
  // provide a set of precomputed hash values.  Also allows resize in case any estimation is not accurate.
	template <typename KV>
	size_t insert_batch(KV const * input, size_t input_size, mapped_type const & default_val) {


#if defined(REPROBE_STAT)
//    if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//      std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//    } else {
//      std::cout << "STATUS: container alignment on value boundary" << std::endl;
//    }

    size_type before = lsize;
#endif
	if (input_size == 0) return 0;

    // buffer size is larger of 2x insert_lookahead and 2x hash batch_size.
    const size_t lookahead2 = static_cast<size_t>(INSERT_LOOKAHEAD) << 1;

    // up to 512 x 4 (4 bytes for offsets.  note we may need to copy the keys too...
    const size_t batch_size =
    		(::std::is_same<key_type, KV>::value) ? 1024 : 512; //InternalHash::batch_size * lookahead2;

    // intermediate storage for hash values.
    using HT = decltype(hash_mod2.operator()(::std::declval<KV>()));
    HT* hashes = ::utils::mem::aligned_alloc<HT>(batch_size);

    if (max_load < batch_size)  {
    	reserve(batch_size);

#if defined(REPROBE_STAT)
      std::cout << "rehashed.  size = " << buckets << std::endl;
//      if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type))  > 0) {
//        std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//      } else {
//        std::cout << "STATUS: container alignment on value boundary" << std::endl;
//      }
#endif
    }

    size_t j;


#if defined(DEBUG_HASH_MAPPING)
    // DEBUGGING:  make histogram
    uint32_t* histo = ::utils::mem::aligned_alloc<uint32_t>(1024);  // 1024 bins.
    memset(histo, 0, 1024 * sizeof(uint32_t));
    uint32_t bin_size = std::max(1UL, (buckets >> 10));
    uint8_t histo_shift = std::log2(bin_size);
    uint32_t* profile = ::utils::mem::aligned_alloc<uint32_t>(bin_size);  // 1024 bins.
    uint32_t profile_mask = ~((~0) << histo_shift);
    memset(profile, 0, bin_size * sizeof(uint32_t));
#endif

    // compute the first part of the hashes
    size_t max_prefetch = std::min(input_size, lookahead2);
    //compute hash and prefetch a little.
    hash_mod2(input, hash_mod2.batch_size, hashes);
    for (j = 0; j < max_prefetch; ++j) {
      // prefetch the info_container entry for ii.
      KH_PREFETCH(reinterpret_cast<const char *>(info_container.data() + hashes[j]), _MM_HINT_T0);
      // prefetch container as well - would be NEAR but may not be exact.
      KH_PREFETCH((const char *)(container + hashes[j]), _MM_HINT_T0);
    }

    size_t i = 0, k, kmax;   // j is index fo hashes array
    size_t max = input_size - (input_size & (batch_size - 1));
    size_t insert_bid;
    size_t bid, bid1;

    value_type val;

    // do blocks of batch_size
    for (; i < max; ) {
    	// now hash a bunch
    	//hash_mod2(input + i + j, batch_size - j, hashes + j);
    	hash_mod2(input + i + hash_mod2.batch_size, batch_size - hash_mod2.batch_size, hashes + hash_mod2.batch_size);

    	// and loop and insert.
    	for (k = 0, kmax = batch_size - lookahead2; k < kmax; ++k, ++i) {

    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

            insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;
//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;

            // intention is to write, so should prefetch for empty entries too.
            // prefetch container
            bid = hashes[k + INSERT_LOOKAHEAD];
            bid1 = bid + 1;
            bid += get_offset(info_container[bid]);
            bid1 += get_offset(info_container[bid1]);

            KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
            // NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH,
            // bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
            if (bid1 > (bid + value_per_cacheline))
              KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);

            // prefetch info container.
            bid = hashes[k + lookahead2];
            KH_PREFETCH(reinterpret_cast<const char *>(info_container.data() + bid), _MM_HINT_T0);
    	}

    	// exhasuted indices for prefetching.  fetch some more.
    	if (input_size > (i+lookahead2)) {
    		max_prefetch = std::min(input_size - (i + lookahead2), lookahead2);
    		hash_mod2(input + i + lookahead2, hash_mod2.batch_size, hashes);
			for (j = 0; j < max_prefetch; ++j) {
          // prefetch the info_container entry for ii.
          KH_PREFETCH(reinterpret_cast<const char *>(info_container.data() + hashes[j]), _MM_HINT_T0);
          // prefetch container as well - would be NEAR but may not be exact.
          KH_PREFETCH((const char *)(container + hashes[j]), _MM_HINT_T0);
        }
    	}
    	// now finish the current section with limited prefetching.
    	// and loop and insert.
    	for (kmax = batch_size - INSERT_LOOKAHEAD; k < kmax; ++k, ++i) {

    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

            insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;
//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;

            // intention is to write, so should prefetch for empty entries too.
            // prefetch container
            bid = hashes[k + INSERT_LOOKAHEAD];
            bid1 = bid + 1;
            bid += get_offset(info_container[bid]);
            bid1 += get_offset(info_container[bid1]);

            KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
            // NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH,
            // bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
            if (bid1 > (bid + value_per_cacheline))
              KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
    	}

    	for (; k < batch_size; ++k, ++i) {
    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

    		insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;
//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;
    	}


        // first check if we need to resize.  within 1% of
        if (static_cast<size_t>(static_cast<double>(lsize) * 1.01) >= max_load) {
        	// start over from current position
        	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

        	return i;
        }

    }

    // do the remainder (less than blocks of batch_size.
    	// has the remainder
    	kmax = std::max((input_size - max), j) - j;
    	if (kmax > 0)
    		hash_mod2(input + i + j, kmax, hashes + j);

    	// and loop and insert.
    	for (k = 0; k < kmax; ++k, ++i) {

    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

            insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;

//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;

            // intention is to write, so should prefetch for empty entries too.
            // prefetch container
            if ((k + INSERT_LOOKAHEAD) < (input_size - max)) {
            bid = hashes[k + INSERT_LOOKAHEAD];
            bid1 = bid + 1;
            bid += get_offset(info_container[bid]);
            bid1 += get_offset(info_container[bid1]);

            KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
            // NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH,
            // bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
            if (bid1 > (bid + value_per_cacheline))
              KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
			}
            // prefetch info container.
            if ((k + lookahead2) < (input_size - max)) {
            bid = hashes[k + lookahead2];
            KH_PREFETCH(reinterpret_cast<const char *>(info_container.data() + bid), _MM_HINT_T0);
    	}
    	}

    	// now finish the current section with limited prefetching.
    	// and loop and insert.
    	kmax = std::max((input_size - max), static_cast<size_t>(INSERT_LOOKAHEAD)) - INSERT_LOOKAHEAD;
    	for (; k < kmax; ++k, ++i) {

    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

            insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;
//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;

            // intention is to write, so should prefetch for empty entries too.
            // prefetch container
            if ((k + INSERT_LOOKAHEAD) < (input_size - max)) {

            bid = hashes[k + INSERT_LOOKAHEAD];
            bid1 = bid + 1;
            bid += get_offset(info_container[bid]);
            bid1 += get_offset(info_container[bid1]);

            KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
            // NOTE!!!  IF WE WERE TO ALWAYS PREFETCH RATHER THAN CONDITIONALLY PREFETCH,
            // bandwidth is eaten up and on i7-4770 the overall time was 2x slower FROM THIS LINE ALONE
            if (bid1 > (bid + value_per_cacheline))
              KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
    	}
    	}

    	kmax = (input_size - max);
    	for (; k < kmax; ++k, ++i) {
    		// process current
    		val = get_tuple(input[i], default_val);

#if defined(DEBUG_HASH_MAPPING)
    		// DEBUG
        	++histo[(hashes[k] >> histo_shift)];
        	++profile[(hashes[k] & profile_mask)];
#endif

            insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            if (insert_bid == insert_failed) {
            	// start over from current position
            	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
#endif

            	return i;

//              rehash(buckets << 1);  // resize.
//              insert_bid = insert_with_hint(container, info_container, hashes[k], val);
            }
            if (missing(insert_bid))
              ++lsize;
    	}

    	free(hashes);

#if defined(DEBUG_HASH_MAPPING)
    	// DEBUG: printout the histogram and profile
    	{
    		::std::stringstream ss;
			ss << "HISTOGRAM, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < 1024; ++ii) {
				ss << histo[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}
    	{
    		::std::stringstream ss;
			ss << "PROFILE, " << buckets << " buckets: ";
			for (size_t ii = 0; ii < bin_size; ++ii) {
				ss << profile[ii] << ", ";
			}
			std::cout << ss.str() << std::endl;
    	}

    	free(histo);
    	free(profile);
#endif
    	return input_size;
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
		if (lsize >= max_load)
		  rehash(buckets << 1);

		// first get the bucket id
		bucket_id_type id = hash(vv.first) & mask;  // target bucket id.

		id = insert_with_hint(container, info_container, id, vv);
		while (id == insert_failed) {
			rehash(buckets << 1);  // resize.
			id = insert_with_hint(container, info_container, id, vv);
		}
		bool success = missing(id);
		size_t bid = get_pos(id);

		if (success) ++lsize;

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT 1", 1, (success ? 1 : 0));
#endif

		//		std::cout << "insert 1 lsize " << lsize << std::endl;
		return std::make_pair(iterator(container + bid, info_container.begin()+ bid, info_container.begin() + buckets + info_empty, filter), success);

	}

	std::pair<iterator, bool> insert(key_type const & key, mapped_type const & val) {
		return insert(std::make_pair(key, val));
	}

protected:

	// insert with iterator.  uses size estimate.
	template <bool estimate = true, typename H=hasher, typename K=Key, typename HVT = hash_val_type,
			typename IT,
			typename std::enable_if<::std::is_constructible<value_type,
			typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	auto insert_impl(IT begin, IT end, int)
		-> decltype(::std::declval<H>()(::std::declval<K*>(), ::std::declval<size_t>(), ::std::declval<HVT*>()), void()) {
#if defined(REPROBE_STAT)
		std::cout << "INSERT PAIR ITER B" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

    // compute hash value array, and estimate the number of unique entries in current.  then merge with current and get the after count.
		using HT = decltype(hash.operator()(::std::declval<key_type>()));
		HT* hash_vals = ::utils::mem::aligned_alloc<HT>(input_size);
		Key* keys = ::utils::mem::aligned_alloc<Key>(hash.batch_size);

		size_t max = input_size - (input_size & (hash.batch_size - 1));
		size_t i = 0, j = 0;

		IT it = begin;
		if (estimate) {
			for (; i < max; i += hash.batch_size) {
				for (j = 0; j < hash.batch_size; ++j, ++it) {
					keys[j] = (*it).first;
				}

				// compute hash
				hash(keys, hash.batch_size, hash_vals + i);
				this->hll.update_via_hashval(hash_vals + i, hash.batch_size);
			}

			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash((*it).first);
				this->hll.update_via_hashval(hash_vals[i]);
			}

			// estimate the number of unique entries in input.
			//#if defined(REPROBE_STAT)
#ifndef NDEBUG
			std::cout << " batch estimate cardinality as " << this->hll.estimate() << std::endl;
#endif
			//#endif
			// assume one element per bucket as ideal, resize now.  should not resize if don't need to.
			this->reserve(static_cast<size_t>(static_cast<double>(this->hll.estimate()) * (1.0 + this->hll.est_error_rate)));   // this updates the bucket counts also.  overestimate by 10 percent just to be sure.
		} else {
			for (; i < max; i += hash.batch_size) {
				for (j = 0; j < hash.batch_size; ++j, ++it) {
					keys[j] = (*it).first;
				}
				// compute hash
				hash(keys, hash.batch_size, hash_vals + i);
			}

			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash((*it).first);
			}
		}
		free(keys);
		// now try to insert.  hashing done already.  also, no need to hash vals again after rehash().
		size_t finished = 0;
		do {
		  finished += insert_batch_by_hash(begin + finished, hash_vals + finished, input_size - finished);
	        if (finished < input_size)  {
	        	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
	        	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
	        }
		} while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
		free(hash_vals);

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER B", input_size, (lsize - before));
#endif
	}

	// insert with iterator.  uses size estimate.
	template <bool estimate = true, typename H=hasher, typename K=Key, typename HVT = hash_val_type,
			typename IT,
			typename std::enable_if<::std::is_constructible<key_type,
			typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	auto insert_impl(IT begin, IT end, mapped_type const & default_val, int)
		-> decltype(::std::declval<H>()(::std::declval<K*>(), ::std::declval<size_t>(), ::std::declval<HVT*>()), void()) {
#if defined(REPROBE_STAT)
		std::cout << "INSERT KEY ITER B" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

    // compute hash value array, and estimate the number of unique entries in current.  then merge with current and get the after count.
		using HT = decltype(hash.operator()(::std::declval<key_type>()));
		HT* hash_vals = ::utils::mem::aligned_alloc<HT>(input_size);

		hash(begin, input_size, hash_vals);

		if (estimate) {
		  for (size_t i = 0; i < input_size; ++i) {
		    this->hll.update_via_hashval(hash_vals[i]);
		  }
  // estimate the number of unique entries in input.
//#if defined(REPROBE_STAT)
#ifndef NDEBUG
		  std::cout << " estimate cardinality as " << this->hll.estimate() << std::endl;
#endif
//#endif
		  // assume one element per bucket as ideal, resize now.  should not resize if don't need to.
		  this->reserve(static_cast<size_t>(static_cast<double>(this->hll.estimate()) * (1.0 + this->hll.est_error_rate)));
		}
  // this updates the bucket counts also.  overestimate by 10 percent just to be sure.

		// now try to insert.  hashing done already.
	      auto converter = [&default_val](key_type const & x) {
	        return ::std::make_pair(x, default_val);
	      };

	      using trans_iter_type = ::bliss::iterator::transform_iterator<IT, decltype(converter)>;
	      trans_iter_type local_start(begin, converter);

	      size_t finished = 0;
	      do {
	        finished += insert_batch_by_hash(local_start + finished, hash_vals + finished, input_size - finished);
	        if (finished < input_size)  {
	        	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
	        	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
	        }
	      } while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
		free(hash_vals);

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER B", input_size, (lsize - before));
#endif
	}


	template <bool estimate = true, typename H=hasher, typename K=Key, typename IT,
			typename std::enable_if<::std::is_constructible<value_type,
			typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	auto insert_impl(IT begin, IT end, long)
		-> decltype(::std::declval<H>()(::std::declval<K>()), void()) {
#if defined(REPROBE_STAT)
		std::cout << "INSERT PAIR ITER" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

    // compute hash value array, and estimate the number of unique entries in current.  then merge with current and get the after count.
		using HVT = decltype(hash.operator()(::std::declval<key_type>()));
		HVT* hash_vals = ::utils::mem::aligned_alloc<HVT>(input_size);

		size_t i = 0;
		IT it = begin;
		if (estimate) {
			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash((*it).first);
				this->hll.update_via_hashval(hash_vals[i]);
			}

			// estimate the number of unique entries in input.
			//#if defined(REPROBE_STAT)
#ifndef NDEBUG
			std::cout << " estimate cardinality as " << this->hll.estimate() << std::endl;
#endif
			//#endif
			// assume one element per bucket as ideal, resize now.  should not resize if don't need to.
			this->reserve(static_cast<size_t>(static_cast<double>(this->hll.estimate()) * (1.0 + this->hll.est_error_rate)));
			// this updates the bucket counts also.  overestimate by 10 percent just to be sure.
		} else {
			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash((*it).first);
			}
		}
		// now try to insert.  hashing done already.
    size_t finished = 0;
    do {
      finished += insert_batch_by_hash(begin + finished, hash_vals + finished, input_size - finished);
      if (finished < input_size)  {
      	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
      	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
      }
    } while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
		free(hash_vals);

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER", input_size, (lsize - before));
#endif
	}

	template <bool estimate = true, typename H=hasher, typename K=Key, typename IT,
			typename std::enable_if<::std::is_constructible<key_type,
			typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	auto insert_impl(IT begin, IT end, mapped_type const & default_val, long)
		-> decltype(::std::declval<H>()(::std::declval<K>()), void()) {
#if defined(REPROBE_STAT)
		std::cout << "INSERT KEY ITER" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

    // compute hash value array, and estimate the number of unique entries in current.  then merge with current and get the after count.
		using HVT = decltype(hash.operator()(::std::declval<key_type>()));
		HVT* hash_vals = ::utils::mem::aligned_alloc<HVT>(input_size);

		size_t i = 0;
		IT it = begin;

		if (estimate) {
			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash(*it);
				this->hll.update_via_hashval(hash_vals[i]);
			}

			// estimate the number of unique entries in input.
	//#if defined(REPROBE_STAT)
#ifndef NDEBUG
			std::cout << " estimate cardinality as " << this->hll.estimate() << std::endl;
#endif

			// assume one element per bucket as ideal, resize now.  should not resize if don't need to.
			this->reserve(static_cast<size_t>(static_cast<double>(this->hll.estimate()) * (1.0 + this->hll.est_error_rate)));
			// this updates the bucket counts also.  overestimate by 10 percent just to be sure.
		} else {
			for (; i < input_size; ++i, ++it) {
				hash_vals[i] = hash(*it);
			}
		}
      auto converter = [&default_val](key_type const & x) {
        return ::std::make_pair(x, default_val);
      };

      using trans_iter_type = ::bliss::iterator::transform_iterator<IT, decltype(converter)>;
      trans_iter_type local_start(begin, converter);

      size_t finished = 0;
      do {
        finished += insert_batch_by_hash(local_start + finished, hash_vals + finished, input_size - finished);
        if (finished < input_size)  {
        	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
        	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
        }

      } while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
		free(hash_vals);

#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER", input_size, (lsize - before));
#endif
	}
public:

	template <typename IT,
		typename std::enable_if<::std::is_constructible<value_type,
		typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	void insert(IT begin, IT end) {
		insert_impl<true>(begin, end, 0);
	}
	template <typename IT,
		typename std::enable_if<::std::is_constructible<key_type,
		typename ::std::iterator_traits<IT>::value_type>::value,
		int>::type = 1>
	void insert(IT begin, IT end, mapped_type const & default_val) {
		insert_impl<true>(begin, end, default_val, 0);
	}
	// insert without estimates.
	// when rehash is needed, the insertion is restarted.
	// uses insert_batch and calculate the hashes internally using hash_mod2.
	//   insert_batch is faster because the hashes can fit in cache..
	//   restart because the cache is invalidated anyways.
  void insert_no_estimate(key_type const * begin, key_type const * end, mapped_type const & default_val) {
    //insert_impl<false>(begin, end, default_val, 0);

#if defined(REPROBE_STAT)
		std::cout << "INSERT Key NO EST" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

      size_t finished = 0;
      do {
        finished += insert_batch(begin + finished, input_size - finished, default_val);
        if (finished < input_size)  {
        	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
        	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
        }
      } while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER", input_size, (lsize - before));
#endif
  }
  void insert_no_estimate(value_type const * begin, value_type const * end) {
    //insert_impl<false>(begin, end, default_val, 0);

#if defined(REPROBE_STAT)
		std::cout << "INSERT Pair No Est" << std::endl;

//		if ((reinterpret_cast<size_t>(container.data()) % sizeof(value_type)) > 0) {
//			std::cout << "WARNING: container alignment not on value boundary" << std::endl;
//		} else {
//			std::cout << "STATUS: container alignment on value boundary" << std::endl;
//		}

		reset_reprobe_stats();
		size_type before = lsize;
#endif

		size_t input_size = std::distance(begin, end);

		if (input_size == 0) return;

      size_t finished = 0;
      do {
        finished += insert_batch(begin + finished, input_size - finished, mapped_type());
        if (finished < input_size)  {
        	std::cout << "rehashing to "  << (buckets <<1) << std::endl;
        	rehash(buckets << 1);  // failed to completely insert (overflow, or max_load).  need to rehash.
        }
      } while (finished < input_size);

		// finally, update the hyperloglog estimator.  just swap.
#if defined(REPROBE_STAT)
		print_reprobe_stats("INSERT ITER", input_size, (lsize - before));
#endif
  }

	/// insert with estimated size to avoid resizing.  uses more memory because of the hash?
	// similar to insert with iterator in structure.  prefetch stuff is delegated to insert_with_hint_no_resize.
	void insert(::std::vector<value_type> const & input) {
		insert(input.data(), input.data() + input.size());
	}
	void insert(::std::vector<key_type> const & input, mapped_type const & default_val) {
		insert(input.data(), input.data() + input.size(), default_val);
	}

	void insert_no_estimate(::std::vector<value_type> const & input) {
		insert_no_estimate(input.data(), input.data() + input.size());
	}
	void insert_no_estimate(::std::vector<key_type> const & input, mapped_type const & default_val) {
		insert_no_estimate(input.data(), input.data() + input.size(), default_val);
	}

protected:
	inline value_type get_tuple(key_type const & key, mapped_type const & default_val = mapped_type()) const {
		return ::std::make_pair(key, default_val);
	}
	inline value_type const & get_tuple(value_type const & val, mapped_type const & default_val = mapped_type()) const {
		return val;
	}

	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	value_type,
	typename std::iterator_traits<Iter>::value_type >::value,
	int >::type = 1>
	inline key_type const & get_key(Iter it) const {
		return it->first;
	}
	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	key_type,
	typename std::iterator_traits<Iter>::value_type >::value,
	int >::type = 1>
	inline key_type const & get_key(Iter it) const {
		return *it;
	}
	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	typename std::iterator_traits<Iter>::value_type,
	value_type
	>::value,
	int >::type = 1>
	inline void copy_value(value_type const & val, Iter it) const {
		*it = val;
	}
	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	typename std::iterator_traits<Iter>::value_type,
	value_type
	>::value,
	int >::type = 1>
	inline void copy_value(key_type const & key, mapped_type const & val, Iter it) const {
		*it = std::make_pair(key, val);
	}
	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	typename std::iterator_traits<Iter>::value_type,
	mapped_type
	>::value,
	int >::type = 1>
	inline void copy_value(mapped_type const & val, Iter it) const {
		*it = val;
	}
	template <typename Iter, typename std::enable_if<
	std::is_constructible<
	typename std::iterator_traits<Iter>::value_type,
	mapped_type
	>::value,
	int >::type = 1>
	inline void copy_value(value_type const & val, Iter it) const {
		*it = val.second;
	}


	struct eval_exists {
		hashmap_robinhood_offsets_reduction const & self;
		eval_exists(hashmap_robinhood_offsets_reduction const & _self,
				container_type const & _cont) : self(_self) {}

		// return value only
		template <typename OutIter, typename std::enable_if<
		std::is_constructible<
		typename std::iterator_traits<OutIter>::value_type,
		uint8_t
		>::value,
		int >::type = 1>
		inline uint8_t operator()(OutIter & it, key_type const & k, bucket_id_type const & bid) const {
			uint8_t rs = self.present(bid);
			self.copy_value(rs, it);
			++it;
			return rs;
		}
		// return key, value pair.
		template <typename OutIter, typename std::enable_if<
		std::is_constructible<
		typename std::iterator_traits<OutIter>::value_type,
		std::pair<Key, uint8_t>
		>::value,
		int >::type = 1>
		inline uint8_t operator()(OutIter & it, key_type const & k, bucket_id_type const & bid) const {
			uint8_t rs = self.present(bid);
			self.copy_value(k, rs, it);
			++it;
			return rs;
		}

	};
	struct eval_find {
		hashmap_robinhood_offsets_reduction const & self;
		container_type const & cont;

		mapped_type unused;

		eval_find(hashmap_robinhood_offsets_reduction const & _self,
				container_type const & _cont,
				mapped_type _unused = mapped_type()) : self(_self), cont(_cont), unused(_unused) {}

		// populate with key-val pair
		template <typename OutIter, typename std::enable_if<
		std::is_constructible<
		typename std::iterator_traits<OutIter>::value_type,
		value_type
		>::value,
		int >::type = 1>
		inline uint8_t operator()(OutIter & it, key_type const & k, bucket_id_type const & bid) const {
			if (self.present(bid)) {
				self.copy_value(cont[self.get_pos(bid)], it);
				++it;
				return 1;
			} else {
				self.copy_value(k, unused, it);  // copy in unused value.
				++it;
				return 0;
			}
		}

		// populate with just mapped-type value.
		template <typename OutIter, typename std::enable_if<
		std::is_constructible<
		typename std::iterator_traits<OutIter>::value_type,
		mapped_type
		>::value,
		int >::type = 1>
		inline uint8_t operator()(OutIter & it, key_type const & k, bucket_id_type const & bid) const {
			if (self.present(bid)) {
				self.copy_value(cont[self.get_pos(bid)].second, it);
				++it;
				return 1;
			} else {
				self.copy_value(unused, it);
				++it;
				return 0;
			}
		}
	};


	/// returns only existing elements.
	struct eval_find_existing {
		hashmap_robinhood_offsets_reduction const & self;
		container_type const & cont;

		eval_find_existing(hashmap_robinhood_offsets_reduction const & _self,
				container_type const & _cont) : self(_self), cont(_cont) {}

		template <typename OutIter, typename std::enable_if<
		std::is_constructible<
		typename std::iterator_traits<OutIter>::value_type,
		value_type
		>::value,
		int >::type = 1>
		inline uint8_t operator()(OutIter & it, key_type const & k, bucket_id_type const & bid) const {
			if (self.present(bid)) {
				self.copy_value(cont[self.get_pos(bid)], it);
				++it;
				return 1;
			}
			return 0;
		}

	};


	template <typename Reduc>
	struct eval_update {
		hashmap_robinhood_offsets_reduction const & self;
		container_type const & cont;

		eval_update(hashmap_robinhood_offsets_reduction const & _self,
				container_type const & _cont) : self(_self), cont(_cont) {}

		template <typename Iter, typename R = Reduc,
				typename ::std::enable_if<
				!std::is_same<R, ::fsc::DiscardReducer>::value, int>::type = 1>
		inline uint8_t operator()(Iter & it, key_type const & k, bucket_id_type const & bid) {
			if (self.present(bid)) {
				cont[self.get_pos(bid)].second =
						self.reduc(cont[self.get_pos(bid)].second, it->second);
				++it;
				return 1;
			}
			++it;
			return 0;
		}
		template <typename Iter, typename R = Reduc,
				typename ::std::enable_if<
				std::is_same<R, ::fsc::DiscardReducer>::value, int>::type = 1>
		inline uint8_t operator()(Iter & it, key_type const & k, bucket_id_type const & bid) {
			++it;
			return self.present(bid);
		}
	};


	template <typename OutIter, typename Eval,
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate>
	size_t internal_find(key_type* begin, key_type* end, OutIter out,
			Eval const & eval,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const  {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif

		size_t cnt = 0;

		size_t total = std::distance(begin, end);

		//prefetch only if target_buckets is larger than QUERY_LOOKAHEAD
#if defined(ENABLE_PREFETCH)
		size_t h;
#endif

		size_t batch_size = InternalHash::batch_size; // static_cast<size_t>(QUERY_LOOKAHEAD));

//		std::cout << "ROBINHOOD internal_find. batch size: " << InternalHash::batch_size << " query lookahead: " << static_cast<size_t>(QUERY_LOOKAHEAD) <<
//		    " total count: " << total << std::endl;


		assert(((batch_size & (batch_size - 1) ) == 0) && "batch_size should be a power of 2.");
		assert(((QUERY_LOOKAHEAD & (QUERY_LOOKAHEAD - 1) ) == 0) && "QUERY_LOOKAHEAD should be a power of 2.");

		size_t lookahead = std::max(static_cast<size_t>(QUERY_LOOKAHEAD), batch_size);
		size_t lookahead2 = lookahead << 1;
		size_t lookahead2_mask = lookahead2 - 1;

		// allocate space here.
		typename InternalHash::result_type* bids =
				::utils::mem::aligned_alloc<typename InternalHash::result_type>(lookahead2);

		size_t max = std::min(lookahead2, total);

		// kick start prefetching.
		size_t i = 0;
		key_type * it = begin;
		hash_mod2(it, max, bids);
#if defined(ENABLE_PREFETCH)
		for (i = 0; i < max; ++it, ++i) {
			h =  bids[i];
			// prefetch the info_container entry for ii.
			KH_PREFETCH((const char *)(info_container.data() + h), _MM_HINT_T0);

			// prefetch container as well - would be NEAR but may not be exact.
			KH_PREFETCH((const char *)(container + h), _MM_HINT_T0);
		}
#endif

		size_t bid, bid1;
		max = total - (total & lookahead2_mask);
		bucket_id_type found;
		size_t j, k, jmax;
		size_t rem = 0;

		for (it = begin, i = 0; i < max; i += lookahead) {

			// first get the bucket id
			// note that we should be accessing the lower and upper halves of the lookahead.
			for (j = (i & lookahead2_mask), k = ((i+lookahead) & lookahead2_mask),
					jmax = ((i + lookahead - 1) & lookahead2_mask);
					j <= jmax; ++j, ++k, ++it ) {

				found = find_pos_with_hint(*it, bids[j], out_pred, in_pred);
				cnt += eval(out, *it, found);  // out is incremented here

				// prefetch the container in this loop too.
				bid = bids[k];
				if (is_normal(info_container[bid])) {
					bid1 = bid + 1 + get_offset(info_container[bid + 1]);
					bid += get_offset(info_container[bid]);

					KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
					if (bid1 > (bid + value_per_cacheline))
						KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
				}
			}

			// prefetch info container, for entries just vacated.
			// it already advanced by lookahead.
			if (total > (i + lookahead2) ) {
				rem = std::min(lookahead, total - (i + lookahead2));
				
				hash_mod2(it + lookahead, rem, bids + (i & lookahead2_mask));
				for (j = (i & lookahead2_mask), jmax = ((i + rem - 1) & lookahead2_mask);
						j <= jmax; ++j) {
					KH_PREFETCH((const char *)(info_container.data() + bids[j]), _MM_HINT_T0);
				}
			}
		}

		// i is at max. now do another iteration
		max = total - (total & (lookahead-1));
		for (j = (i & lookahead2_mask), k = ((i+lookahead) & lookahead2_mask);
				i < max; ++i, ++j, ++k, ++it) {

			found = find_pos_with_hint(*it, bids[j], out_pred, in_pred);
			cnt += eval(out, *it, found);  // out is incremented here

			// prefetch the container in this loop too.
			if (total > (i + lookahead) ) {
				bid = bids[k];
				if (is_normal(info_container[bid])) {
					bid1 = bid + 1 + get_offset(info_container[bid + 1]);
					bid += get_offset(info_container[bid]);

					KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
					if (bid1 > (bid + value_per_cacheline))
						KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
				}
			}
		}

		for (j = (i & lookahead2_mask);
				i < total; ++i, ++j, ++it) {

			found = find_pos_with_hint(*it, bids[j], out_pred, in_pred);
			cnt += eval(out, *it, found);  // out is incremented here
		}

		free(bids);

#if defined(REPROBE_STAT)
		print_reprobe_stats("INTERNAL_FIND ITER PAIR", std::distance(begin, end), total);
#endif

		return cnt;
	}




public:


	/**
	 * @brief count the presence of a key
	 */
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate >
	inline bool exists( key_type const & k,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()  ) const {

		return present(find_pos(k, out_pred, in_pred));
	}


	template <
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	std::vector<uint8_t> exists(key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_exists ev(*this, container);

		std::vector<uint8_t> results;
		results.reserve(std::distance(begin, end));

		::fsc::back_emplace_iterator<::std::vector<uint8_t> > count_emplace_iter(results);

		internal_find(begin, end, count_emplace_iter, ev, out_pred, in_pred);

		return results;
	}


	template <typename OITER,
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	size_t exists(OITER out, key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_exists ev(*this, container);

		return internal_find(begin, end, out, ev, out_pred, in_pred);
	}


	/**
	 * @brief count the presence of a key
	 */
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate >
	inline uint8_t count( key_type const & k,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()  ) const {

		return exists(k, out_pred, in_pred);
	}


	template <
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	std::vector<uint8_t> count(key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		return exists(begin, end, out_pred, in_pred);
	}


	template <typename OITER,
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	size_t count(OITER out, key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		return exists(out, begin, end, out_pred, in_pred);
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
		print_reprobe_stats("FIND 1 KEY", 1, ( present(idx) ? 1: 0));
#endif

		if (present(idx))
			return iterator(container + get_pos(idx), info_container.begin()+ get_pos(idx),
					info_container.begin() + buckets + info_empty, filter);
		else
			return this->end();

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
		print_reprobe_stats("FIND 1 KEY", 1, ( present(idx) ? 1: 0));
#endif

		if (present(idx))
			return const_iterator(container + get_pos(idx), info_container.cbegin()+ get_pos(idx),
					info_container.cbegin() + buckets + info_empty, filter);
		else
			return this->cend();

	}


	template <typename OT,
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate,
	typename std::enable_if<std::is_constructible<OT, mapped_type>::value ||
	::std::is_constructible<OT, value_type>::value,
	int>::type = 1
	>
	std::vector<OT> find(key_type* begin, key_type* end,
			mapped_type const & nonexistent = mapped_type(),
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_find ev(*this, container, nonexistent);

		std::vector<OT> results;
		results.reserve(std::distance(begin, end));

		::fsc::back_emplace_iterator<::std::vector<OT> > find_emplace_iter(results);

		internal_find(begin, end, find_emplace_iter, ev, out_pred, in_pred);

		return results;
	}

	// returns value for all, even if have to fill in missing.
	template <typename OIter,
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate,
	typename std::enable_if<
		::std::is_constructible<typename ::std::iterator_traits<OIter>::value_type, value_type>::value ||
		::std::is_constructible<typename ::std::iterator_traits<OIter>::value_type, mapped_type>::value,
		int>::type = 1
	>
	size_t find(OIter out, key_type* begin, key_type* end,
			mapped_type const & nonexistent = mapped_type(),
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_find ev(*this, container, nonexistent);

		return internal_find(begin, end, out, ev, out_pred, in_pred);
	}



	template <
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	std::vector<value_type> find_existing(key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_find_existing ev(*this, container);

		std::vector<value_type> results;
		results.reserve(std::distance(begin, end));

		::fsc::back_emplace_iterator<::std::vector<value_type> >
		find_emplace_iter(results);

		internal_find(begin, end, find_emplace_iter, ev, out_pred, in_pred);

		return results;
	}

	// find existing - has to use value_type since the presence of the keys indicates existence.
	template <
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	size_t find_existing(value_type* out, key_type* begin, key_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) const {

		eval_find_existing ev(*this, container);

		return internal_find(begin, end, out, ev, out_pred, in_pred);
	}



	/* ========================================================
	 *  update.  should only update existing entries
	 *   need separate insert with reducer.
	 */


	/**
	 * @brief.  updates current value.  does NOT insert new entries.
	 */
	void update(key_type const & k, mapped_type const & val) {
		// find and update.  if not present, insert.
		bucket_id_type bid = find_pos(k);

		if (present(bid)) {  // not inserted and no exception, so an equal entry has been found.

			if (! std::is_same<Reducer, ::fsc::DiscardReducer>::value)  container[get_pos(bid)].second =
					reduc(container[get_pos(bid)].second, val);   // so update.

		}
	}

	void update(value_type const & vv) {
		update(vv.first, vv.second);
	}


	template <
	typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate
	>
	size_t update(value_type* begin, value_type* end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate() ) {

		eval_update<Reducer> ev(*this, container);

		return internal_find(begin, end, begin, ev, out_pred, in_pred);
	}


protected:
	/**
	 * @brief erases a key.  performs backward shift using memmove
	 */
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate >
	size_type erase_and_compact(key_type const & k, bucket_id_type const & bid,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()) {

		bucket_id_type found = find_pos_with_hint(k, bid, out_pred, in_pred);  // get the matching position

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

		//		if (end < pos1)
		//		  std::cout << "erasing " << k << " hash " << bid << " at " << found << " pos " << pos << " end is " << end << std::endl;

		// move to backward shift.  move [found+1 ... end-1] to [found ... end - 2].  end is excluded because it has 0 dist.
		memmove((container + pos), (container + pos1), (end - pos1) * sizeof(value_type));



		// debug		print();

		// now change the offsets.
		// if that was the last entry for the bucket, then need to change this to say empty.
		if (get_offset(info_container[bid]) == get_offset(info_container[bid1])) {  // both have same distance, so bid has only 1 entry
			set_empty(info_container[bid]);
		}


		// start from bid+1, end at end - 1.
		for (size_t i = bid1; i < end; ++i ) {
			--(info_container[i]);
			if (get_offset(info_container[i]) == 127) throw std::logic_error("ERROR: should not get 127, indicates an underflow situation.");
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
	//  ERASE should do it in batch.  within each bucket, erase and compact, track end points.
	//  then one pass front to back compact across buckets.


public:

	/// single element erase with key.
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
	typename InPredicate = ::bliss::filter::TruePredicate >
	size_type erase_no_resize(key_type const & k,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()) {
#if defined(REPROBE_STAT)
				reset_reprobe_stats();
#endif
				size_t bid = hash(k) & mask;

				size_t erased = erase_and_compact(k, bid, out_pred, in_pred);

#if defined(REPROBE_STAT)
				print_reprobe_stats("ERASE 1", 1, erased);
#endif
				return erased;
	}

	/// batch erase with iterator of value pairs.
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate>
	size_type erase_no_resize(key_type const * begin, key_type const * end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()) {

#if defined(REPROBE_STAT)
		reset_reprobe_stats();
#endif

		size_type before = lsize;

		size_t total = std::distance(begin, end);

		//prefetch only if target_buckets is larger than QUERY_LOOKAHEAD
#if defined(ENABLE_PREFETCH)
		size_t h;
#endif


		size_t batch_size = InternalHash::batch_size; // static_cast<size_t>(QUERY_LOOKAHEAD));

//		std::cout << "ROBINHOOD erase_no_resize. batch size: " << static_cast<size_t>(InternalHash::batch_size) <<
//				" query lookahead: " << static_cast<size_t>(QUERY_LOOKAHEAD) <<
//		    " total count: " << total << std::endl;


		assert(((batch_size & (batch_size - 1) ) == 0) && "batch_size should be a power of 2.");
		assert(((QUERY_LOOKAHEAD & (QUERY_LOOKAHEAD - 1) ) == 0) && "QUERY_LOOKAHEAD should be a power of 2.");

		size_t lookahead = std::max(static_cast<size_t>(QUERY_LOOKAHEAD), batch_size);
		size_t lookahead2 = lookahead << 1;
		size_t lookahead2_mask = lookahead2 - 1;


		// allocate space here.
		typename InternalHash::result_type* bids =
				::utils::mem::aligned_alloc<typename InternalHash::result_type>(lookahead2);

		size_t max = std::min(lookahead2, total);

		// kick start prefetching.
		size_t i = 0;
		key_type const * it = begin;
		hash_mod2(it, max, bids);
#if defined(ENABLE_PREFETCH)
		for (i = 0; i < max; ++it, ++i) {

			h =  bids[i];
			// prefetch the info_container entry for ii.
			KH_PREFETCH((const char *)(info_container.data() + h), _MM_HINT_T0);

			// prefetch container as well - would be NEAR but may not be exact.
			KH_PREFETCH((const char *)(container + h), _MM_HINT_T0);
		}
#endif
//		std::cout << "hashed and prefetched [0, " << i << ")" << std::endl;

		size_t bid, bid1;
		max = total - (total & lookahead2_mask);
		//max = total - lookahead2_mask;
		size_t j, k, jmax;

		size_t rem = 0;

		for (it = begin, i = 0; i < max; i += lookahead) {

//			std::cout <<
//					" erased [ <" << std::distance(begin, it) << "," << (i & lookahead2_mask) << "," << ((i+lookahead) & lookahead2_mask) <<
//					"> .. ";


			// first get the bucket id
			// note that we should be accessing the lower and upper halves of the lookahead.
			for (j = (i & lookahead2_mask), k = ((i+lookahead) & lookahead2_mask),
					jmax = ((i + lookahead - 1) & lookahead2_mask);
					j <= jmax; ++j, ++k, ++it ) {

				erase_and_compact(*it, bids[j], out_pred, in_pred);

				// prefetch the container in this loop too.
				bid = bids[k];
				if (is_normal(info_container[bid])) {
					bid1 = bid + 1 + get_offset(info_container[bid + 1]);
					bid += get_offset(info_container[bid]);

					KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
					if (bid1 > (bid + value_per_cacheline))
						KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
				}
			}

//			std::cout << "< " << std::distance(begin, it) << ", " << j << ", " << k << "> )";

			// prefetch info container, for entries just vacated.
			if (total > (i + lookahead2) ) {
				rem = std::min(lookahead, total - (i + lookahead2));

//				std::cout << ", hashed [ <" << (std::distance(begin, it) + lookahead) << ", " <<
//						(i & lookahead2_mask) << "> .. ";

				// it already advanced by lookahead.
				hash_mod2(it + lookahead, rem, bids + (i & lookahead2_mask));
				for (j = (i & lookahead2_mask), jmax = ((i + rem - 1) & lookahead2_mask);
						j <= jmax; ++j) {
					KH_PREFETCH((const char *)(info_container.data() + bids[j]), _MM_HINT_T0);
				}

//				std::cout << "<" << (std::distance(begin, it) + lookahead + rem) << ", " << j << "> )" <<
//						std::endl;
			} else {
//				std::cout << std::endl;
			}
		}

//		std::cout <<
//				"LAST erased [ <" << std::distance(begin, it) << "," << (i & lookahead2_mask) << "," << ((i+lookahead) & lookahead2_mask) <<
//				"> .. ";

		// i is at max. now do another iteration
		max = total - (total & (lookahead-1));
		for (j = (i & lookahead2_mask), k = ((i+lookahead) & lookahead2_mask);
				i < max; ++i, ++j, ++k, ++it) {

			erase_and_compact(*it, bids[j], out_pred, in_pred);

			// prefetch the container in this loop too.
			if (total > (i + lookahead) ) {
				bid = bids[k];
				if (is_normal(info_container[bid])) {
					bid1 = bid + 1 + get_offset(info_container[bid + 1]);
					bid += get_offset(info_container[bid]);

					KH_PREFETCH((const char *)(container + bid), _MM_HINT_T0);
					if (bid1 > (bid + value_per_cacheline))
						KH_PREFETCH((const char *)(container + bid + value_per_cacheline), _MM_HINT_T1);
				}
			}
		}
//		std::cout << "< " << std::distance(begin, it) << ", " << j << ", " << k << "> )" << std::endl;
//
//
//		std::cout <<
//				"FINAL erased [ <" << std::distance(begin, it) << "," << (i & lookahead2_mask) <<
//				"> .. ";

		for (j = (i & lookahead2_mask); i < total; ++i, ++j, ++it) {

			erase_and_compact(*it, bids[j], out_pred, in_pred);
		}
//		std::cout << "< " << std::distance(begin, it) << ", " << j << "> )"  << std::endl;


		free(bids);

#if defined(REPROBE_STAT)
		print_reprobe_stats("ERASE ITER PAIR", std::distance(begin, end), before - lsize);
#endif
		return before - lsize;
	}


	/**
	 * @brief erases a key.
	 */
	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate >
	size_type erase(key_type const & k,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()) {

		size_type res = erase_no_resize(k, out_pred, in_pred);

		if (lsize < min_load) rehash(buckets >> 1);

		return res;
	}

	template <typename OutPredicate = ::bliss::filter::TruePredicate,
			typename InPredicate = ::bliss::filter::TruePredicate >
	size_type erase(key_type const * begin, key_type const * end,
			OutPredicate const & out_pred = OutPredicate(),
			InPredicate const & in_pred = InPredicate()) {

		size_type erased = erase_no_resize(begin, end, out_pred, in_pred);

		if (lsize < min_load) reserve(lsize);

		return erased;
	}



	//	/// batch erase with a flag array.  backward shift after all deletions
	//	template <typename Iter,
	//		typename OutPredicate = ::bliss::filter::TruePredicate,
	//		typename InPredicate = ::bliss::filter::TruePredicate>
	//	size_type erase2(Iter begin, Iter end,
	//	                          OutPredicate const & out_pred = OutPredicate(),
	//	                          InPredicate const & in_pred = InPredicate()) {
	//
	//#if defined(REPROBE_STAT)
	//		reset_reprobe_stats();
	//#endif
	//
	//		// set up a mask to track which are deleted.
	//		std::vector<info_type> deleted(info_container.size(), info_normal);   // "normal" means deleted at that exact location (correspond to container position).
	//		// the "offset" at a bucket is how many have been deleted from that bucket,
	//
	//		size_type before = lsize;
	//
	//		//prefetch only if target_buckets is larger than QUERY_LOOKAHEAD
	//		size_t i = 0;
	//		size_t h;
	//
	//		size_t min_pos = std::numeric_limits<size_t>::max(), max_pos = 0;
	//
	//		// prefetch 2*QUERY_LOOKAHEAD of the info_container.
	//    for (Iter it = begin; (i < (2* QUERY_LOOKAHEAD)) && (it != end); ++it, ++i) {
	//      h =  hash(get_key(it));
	//      lookahead_hashes[i] = h;
	//      // prefetch the info_container entry for ii.
	//      KH_PREFETCH((const char *)(info_container.data() + (h & mask)), _MM_HINT_T0);
	//
	//      // prefetch container as well - would be NEAR but may not be exact.
	//      KH_PREFETCH((const char *)(container.data() + (h & mask)), _MM_HINT_T0);
	//
	//      KH_PREFETCH((const char *)(deleted.data() + (h & mask)), _MM_HINT_T0);
	//    }
	//
	//		size_t total = std::distance(begin, end);
	//
	//		size_t id, bid, bid1;
	//		Iter it, new_end = begin;
	//		std::advance(new_end, (total > (2 * QUERY_LOOKAHEAD)) ? (total - (2 * QUERY_LOOKAHEAD)) : 0);
	//    i = 0;
	//    bucket_id_type found;
	//
	//
	//		for (it = begin; it != new_end; ++it) {
	//
	//			// first get the bucket id
	//			id = lookahead_hashes[i] & mask;  // target bucket id.
	//
	//			// prefetch info_container.
	//      h = hash(get_key(it + 2 * QUERY_LOOKAHEAD));
	//      lookahead_hashes[i] = h;
	//      KH_PREFETCH((const char *)(info_container.data() + (h & mask)), _MM_HINT_T0);
	//
	//      // prefetch container
	//      bid = lookahead_hashes[(i + QUERY_LOOKAHEAD) & QUERY_LOOKAHEAD_MASK] & mask;
	//      if (is_normal(info_container[bid])) {
	//
	//    	  // prefetch at bucket location
	//          KH_PREFETCH((const char *)(deleted.data() + bid), _MM_HINT_T0);  // 64  of these in a cacheline
	//
	//    	  bid1 = bid + 1 + get_offset(info_container[bid + 1]);
	//        bid += get_offset(info_container[bid]);
	//
	////        for (size_t j = bid; j < bid1; j += value_per_cacheline) {
	////          KH_PREFETCH((const char *)(container.data() + j), _MM_HINT_T0);
	////        }
	//		KH_PREFETCH((const char *)(container.data() + bid), _MM_HINT_T0);
	//		if (bid1 > (bid + value_per_cacheline))
	//			KH_PREFETCH((const char *)(container.data() + bid + value_per_cacheline), _MM_HINT_T1);
	//
	//        // prefetch at first element of bucket.
	//        KH_PREFETCH((const char *)(deleted.data() + bid), _MM_HINT_T0);  // 64  of these in a cacheline
	//      }
	//
	//		found = find_pos_with_hint(get_key(it), id, out_pred, in_pred);  // get the matching position
	//
	//		if (present(found) && is_normal(deleted[get_pos(found)])) {
	//				++deleted[id];  // mark to indicate that an entry in the bucket is to be deleted.
	//				set_empty(deleted[get_pos(found)]);  // and mark the specific entry's position.
	//				--lsize;
	//				min_pos = std::min(min_pos, id);
	//				max_pos = std::max(max_pos, id);
	//		}
	//
	//      ++i;
	//      i &= QUERY_LOOKAHEAD_MASK;
	//		}
	//
	//
	//    new_end = begin;
	//    std::advance(new_end, (total > QUERY_LOOKAHEAD) ? (total - QUERY_LOOKAHEAD) : 0);
	//    for (; it != new_end; ++it) {
	//
	//      // first get the bucket id
	//      id = lookahead_hashes[i] & mask;  // target bucket id.
	//
	//      // prefetch container
	//      bid = lookahead_hashes[(i + QUERY_LOOKAHEAD) & QUERY_LOOKAHEAD_MASK] & mask;
	//      if (is_normal(info_container[bid])) {
	//          KH_PREFETCH((const char *)(deleted.data() + bid), _MM_HINT_T0);  // 64 * 8 of these in a cacheline.
	//        bid1 = bid + 1 + get_offset(info_container[bid + 1]);
	//        bid += get_offset(info_container[bid]);
	//
	////        for (size_t j = bid; j < bid1; j += value_per_cacheline) {
	////          KH_PREFETCH((const char *)(container.data() + j), _MM_HINT_T0);
	////        }
	//		KH_PREFETCH((const char *)(container.data() + bid), _MM_HINT_T0);
	//		if (bid1 > (bid + value_per_cacheline))
	//			KH_PREFETCH((const char *)(container.data() + bid + value_per_cacheline), _MM_HINT_T1);
	//        KH_PREFETCH((const char *)(deleted.data() + bid), _MM_HINT_T0);  // 64 * 8 of these in a cacheline.
	//      }
	//
	//		found = find_pos_with_hint(get_key(it), id, out_pred, in_pred);  // get the matching position
	//
	//		if (present(found) && is_normal(deleted[get_pos(found)])) {
	//				++deleted[id];  // mark to indicate that an entry in the bucket is to be deleted.
	//				set_empty(deleted[get_pos(found)]);  // and mark the specific entry's position.
	//				--lsize;
	//				min_pos = std::min(min_pos, id);
	//				max_pos = std::max(max_pos, id);
	//		}
	//
	//
	//      ++i;
	//      i &= QUERY_LOOKAHEAD_MASK;
	//    }
	//
	//    for (; it != end; ++it) {
	//
	//      // first get the bucket id
	//      id = lookahead_hashes[i] & mask;  // target bucket id.
	//
	//		found = find_pos_with_hint(get_key(it), id, out_pred, in_pred);  // get the matching position
	//
	//		if (present(found) && is_normal(deleted[get_pos(found)])) {
	//				++deleted[id];  // mark to indicate that an entry in the bucket is to be deleted.
	//				set_empty(deleted[get_pos(found)]);  // and mark the specific entry's position.
	//				--lsize;
	//				min_pos = std::min(min_pos, id);
	//				max_pos = std::max(max_pos, id);
	//		}
	//
	//      ++i;
	//      i &= QUERY_LOOKAHEAD_MASK;
	//    }
	//
	//
	//
	//	// now compact.
	//	// note that original info has to "normal"
	//	// advance max_pos to next zero.
	//    max_pos = find_next_zero_offset_pos(info_container, max_pos + 1);  // then get the next zero offset pos from max_pos + 1
	//
	//    for (i = min_pos; i < std::min(max_pos, min_pos + QUERY_LOOKAHEAD); ++i) {
	//        KH_PREFETCH((const char *)(info_container.data() + i), _MM_HINT_T0);
	//
	//        // prefetch container as well - would be NEAR but may not be exact.
	//        KH_PREFETCH((const char *)(container.data() + i), _MM_HINT_T0);
	//
	//        KH_PREFETCH((const char *)(deleted.data() + i), _MM_HINT_T0);
	//
	//    }
	//
	//
	//    // operate between min_pos and max_pos
	//    // first compact and move the data, bucket by bucket.  do a simple scan
	//    size_t insert_pos, insert_start, read_pos, read_end;
	//
	//    info_type info = info_container[min_pos];
	//    info_type next_info;
	//    insert_start = min_pos + get_offset(info);  // track the start of the bucket.
	//    insert_pos = insert_start;  // need to initialize so we can use it in loop
	//
	//    // iterate between min pos and max pos (which is the last position past max_pos that had a non info_empty or info_normal entry
	//    // for each occupied bucket, iteratively walk though, compacting compacting each bucket.  the insert position is upshifted to i
	//    // when there is a stretch of empty buckets.
	//    for (i = min_pos; i < std::max(std::max(max_pos, static_cast<size_t>(QUERY_LOOKAHEAD)) - static_cast<size_t>(QUERY_LOOKAHEAD), min_pos); ++i) {
	//    	// NOTE:  not a complete compaction.   offset cannot go below zero.
	//
	//        KH_PREFETCH((const char *)(info_container.data() + i + QUERY_LOOKAHEAD), _MM_HINT_T0);
	//
	//        // prefetch container as well - would be NEAR but may not be exact.
	//        KH_PREFETCH((const char *)(container.data() + i + QUERY_LOOKAHEAD), _MM_HINT_T0);
	//
	//        KH_PREFETCH((const char *)(deleted.data() + i + QUERY_LOOKAHEAD), _MM_HINT_T0);
	//
	//
	//    	next_info = info_container[i + 1];
	//
	//    	insert_start = std::max(insert_pos, i);  // allow skipping over any entries that are empty.
	//    	info_container[i] = info_empty + (insert_start - i);
	//
	//    	if (is_normal(info)) {  // has content.
	//
	//    		insert_pos = insert_start;
	//
	//			read_pos = i + get_offset(info);
	//			read_end = i + 1 + get_offset(next_info);
	//
	////			if (get_offset(deleted[i]) == 0) {
	////				// nothing was deleted from thsi bucket, so do simple copy.
	////				memmove((container.data() + insert_pos), (container.data() + read_pos), sizeof(value_type) * (read_end - read_pos));
	////				insert_pos += (read_end - read_pos);
	////			} else {
	//				for (; read_pos < read_end; ++read_pos) {
	//					if (is_normal(deleted[read_pos])) {  // read_pos entry not deleted.
	//						if (insert_pos < read_pos) container[insert_pos] = std::move(container[read_pos]);  // move every one?
	//						++insert_pos;
	//					}
	//				}
	////			}
	//			if (insert_start < insert_pos) set_normal(info_container[i]);  // some inserted entries.
	//			// else already set as empty.
	//
	//    	} // else empty bucket, info already set.
	//
	//    	info = next_info;
	//
	//    }
	//
	//
	//    for (; i < max_pos; ++i) {
	//    	// NOTE:  not a complete compaction.   offset cannot go below zero.
	//
	//    	next_info = info_container[i + 1];
	//
	//    	insert_start = std::max(insert_pos, i);  // allow skipping over any entries that are empty.
	//    	info_container[i] = info_empty + (insert_start - i);
	//
	//    	if (is_normal(info)) {  // has content.
	//
	//    		insert_pos = insert_start;
	//
	//			read_pos = i + get_offset(info);
	//			read_end = i + 1 + get_offset(next_info);
	//
	////			if (get_offset(deleted[i]) == 0) {
	////				// nothing was deleted from thsi bucket, so do simple copy.
	////				memmove((container.data() + insert_pos), (container.data() + read_pos), sizeof(value_type) * (read_end - read_pos));
	////				insert_pos += (read_end - read_pos);
	////			} else {
	//				for (; read_pos < read_end; ++read_pos) {
	//					if (is_normal(deleted[read_pos])) {  // read_pos entry not deleted.
	//						if (insert_pos < read_pos) container[insert_pos] = std::move(container[read_pos]);  // move every one?
	//						++insert_pos;
	//					}
	//				}
	////			}
	//			if (insert_start < insert_pos) set_normal(info_container[i]);  // some inserted entries.
	//			// else already set as empty.
	//
	//    	} // else empty bucket, info already set.
	//
	//    	info = next_info;
	//
	//    }
	//
	//
	//
	//#if defined(REPROBE_STAT)
	//		print_reprobe_stats("ERASE ITER PAIR", std::distance(begin, end), before - lsize);
	//#endif
	//		return before - lsize;
	//	}
	//


};

template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_empty;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_mask;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_normal;

template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bucket_id_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bid_pos_mask;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bucket_id_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bid_pos_exists;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bucket_id_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::insert_failed;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bucket_id_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::find_failed;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr typename hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::bucket_id_type hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::cache_align_mask;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr uint32_t hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::info_per_cacheline;
template <typename Key, typename T, template <typename> class Hash, template <typename> class Equal, typename Allocator, typename Reducer >
constexpr uint32_t hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, Reducer>::value_per_cacheline;


//========== ALIASED TYPES

template <typename Key, typename T, template <typename> class Hash = ::std::hash,
		template <typename> class Equal = ::std::equal_to,
		typename Allocator = ::std::allocator<std::pair<const Key, T> > >
using hashmap_robinhood_offsets = hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, ::fsc::DiscardReducer>;

template <typename Key, typename T, template <typename> class Hash = ::std::hash,
		template <typename> class Equal = ::std::equal_to,
		typename Allocator = ::std::allocator<std::pair<const Key, T> > >
using hashmap_robinhood_offsets_count = hashmap_robinhood_offsets_reduction<Key, T, Hash, Equal, Allocator, ::std::plus<T> >;

}  // namespace fsc
#endif /* KMERHASH_ROBINHOOD_OFFSET_HASHMAP_HPP_ */
