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

/**
 * test_hyperloglog64.cpp
 * Test hyperloglog64 class
 *  Created on: Feb 28, 2017
 *      Author: Tony Pan <tpan7@gatech.edu>
 */

// #include "common/kmer.hpp"
// #include "index/kmer_hash.hpp"

#include "kmerhash/hash_new.hpp"

#include "kmerhash/hyperloglog64.hpp"

#include <gtest/gtest.h>
#include <cstdint>  // for uint64_t, etc.
#include <unordered_set>
#include <random>   // rand, srand



template <typename TT, typename HH, uint8_t p = 12, uint8_t i = 0>
struct HyperLogLog64TestParam {
	using Type = TT;
	using Hash = HH;
	static constexpr uint8_t precision = p;
	static constexpr uint8_t ignore = i;
};
template <typename TT, typename HH, uint8_t p, uint8_t i>
constexpr uint8_t HyperLogLog64TestParam<TT, HH, p, i>::ignore;
template <typename TT, typename HH, uint8_t p, uint8_t i>
constexpr uint8_t HyperLogLog64TestParam<TT, HH, p, i>::precision;

/*
 * test class holding some information.  Also, needed for the typed tests
 */
template<typename PARAMS>
class HyperLogLog64Test : public ::testing::Test
{
public:
	using TT = typename PARAMS::Type;
	using HH = typename PARAMS::Hash;

    using HLL = hyperloglog64<TT, HH, PARAMS::precision>;

protected:
    // values chosen for speed...
    static constexpr size_t iterations = 100ULL;
	static constexpr size_t step = 1001ULL;

    std::default_random_engine generator;
    std::uniform_int_distribution<TT> distribution;

    std::unordered_set<TT> uniq;

    HLL hll;

    virtual void SetUp()
    {
    	hll = std::move(HLL(PARAMS::ignore));
    }



    void update(HLL & hll, std::unordered_set<TT> & uniq) {
    	TT val;

    	for (size_t s = 0; s < step; ++s) {

			val = this->distribution(this->generator);

			hll.update(val);
			uniq.insert(val);
		}
    }


    void update_batch(HLL & hll, std::unordered_set<TT> & uniq) {
    	TT val;

    	std::vector<TT> vals;
    	vals.reserve(step);

    	for (size_t s = 0; s < step; ++s) {
			val = this->distribution(this->generator);

			vals.emplace_back(val);
			uniq.insert(val);
		}

    	hll.update(vals.data(), step);
    }

//    void update_via_hash(HLL & _hll, std::unordered_set<TT> & _uniq) {
//    	TT val;
//    	HH hash;
//    	typename HLL::HVT hval;
//    	::std::vector<TT> vals;
//    	vals.reserve(step);
//
//    	std::cout << "init " << std::endl;
//
//
//    	for (size_t s = 0; s < step; ++s) {
//			val = distribution(generator);
//			vals.emplace_back(val);
//    	}
//    	std::cout << "generated input " << std::endl;
//
//
//    	for (size_t s = 0; s < step; ++s) {
//			_hll.update_via_hashval(hash(vals[s]));
//    	}
//    	std::cout << "hashed and updated " << std::endl;
//
//    	for (size_t s = 0; s < step; ++s) {
//			_uniq.insert(vals[s]);
//		}
//    	std::cout << "unique inserted " << std::endl;
//    }
	void update_via_hash(HLL & _hll, std::unordered_set<TT> & _uniq) {
		TT val;
		HH hash;

		for (size_t s = 0; s < step; ++s) {
			val = distribution(generator);

			_hll.update_via_hashval(hash(val));
			_uniq.insert(val);
		}
	}


    void update_via_hash_batch(HLL & _hll, std::unordered_set<TT> & _uniq)
    {
    	TT val;
    	HH hash;
    	std::vector<typename HLL::HVT> hashes;
    	hashes.reserve(step);

    	for (size_t s = 0; s < step; ++s) {
			val = this->distribution(this->generator);

			hashes.emplace_back(hash.operator()(val));
			_uniq.insert(val);
		}

    	_hll.update_via_hashval(hashes.data(), step);
    }


    void report(size_t const & i, HLL const & _hll, std::unordered_set<TT> const & _uniq) const {
		double est = _hll.estimate();
		double act = static_cast<double>(_uniq.size());

		double err = fabs((est - act) / act);


		// arbitrary bound of 2.5 times est error rate.
		if ((err > 2.5 * HLL::est_error_rate) && (est < act))
      std::cout << "iteration " << std::setw(5) << i <<
          " total count " << std::setw(10) << ((i+1) * step) <<
          " estimate " << std::setw(10) << est <<
          " actual " << std::setw(10) << act <<
          " percent " << std::setw(10) << (100.0 * err) << "%" <<
          " expected " << std::setw(10) << (100.0 * HLL::est_error_rate) << "%" <<
          std::endl;


	EXPECT_FALSE((err > 2.5 * HLL::est_error_rate) && (est < act));

    }

};

template<typename PARAMS>
constexpr size_t HyperLogLog64Test<PARAMS>::iterations;
template<typename PARAMS>
constexpr size_t HyperLogLog64Test<PARAMS>::step;

// indicate this is a typed test
TYPED_TEST_CASE_P(HyperLogLog64Test);



// testing the copy constructor
TYPED_TEST_P(HyperLogLog64Test, estimate){

//	std::cout << "bit set for 1: " << static_cast<size_t>(leftmost_set_bit(0x1ULL)) << std::endl;
//	std::cout << "bit set for D: " << static_cast<size_t>(leftmost_set_bit(0xdULL)) << std::endl;

	this->hll.clear();
	this->uniq.clear();

	for (size_t i = 0; i < this->iterations; ++i) {

		this->update(this->hll, this->uniq);

//		this->report(i, this->hll, this->uniq);

	}
	this->report(this->iterations, this->hll, this->uniq);

}

TYPED_TEST_P(HyperLogLog64Test, estimate_batch){

	this->hll.clear();
	this->uniq.clear();

	for (size_t i = 0; i < this->iterations; ++i) {

		this->update_batch(this->hll, this->uniq);

//		this->report(i, this->hll, this->uniq);

	}
	this->report(this->iterations, this->hll, this->uniq);

}

// testing the copy constructor
TYPED_TEST_P(HyperLogLog64Test, estimate_by_hash){

//	std::cout << "bit set for 0x1000000000000000: " << static_cast<size_t>(leftmost_set_bit(0x1000000000000000ULL)) << std::endl;
//	std::cout << "bit set for 0x0000000100000000: " << static_cast<size_t>(leftmost_set_bit(0x0000000100000000ULL)) << std::endl;

	this->hll.clear();
	this->uniq.clear();

	for (size_t i = 0; i < this->iterations; ++i) {

		this->update_via_hash(this->hll, this->uniq);

//		this->report(i, this->hll, this->uniq);

	}


	this->report(this->iterations, this->hll, this->uniq);

}

// testing the copy constructor
TYPED_TEST_P(HyperLogLog64Test, estimate_by_hash_batch){

	this->hll.clear();
	this->uniq.clear();

	for (size_t i = 0; i < this->iterations; ++i) {

		this->update_via_hash_batch(this->hll, this->uniq);

//		this->report(i, this->hll, this->uniq);
	}
	this->report(this->iterations, this->hll, this->uniq);

}



// testing the copy constructor
TYPED_TEST_P(HyperLogLog64Test, merge){

	this->hll.clear();
	this->uniq.clear();

    using HLL = hyperloglog64<
    		typename TypeParam::Type,
    		typename TypeParam::Hash,
			TypeParam::precision>;
	HLL lhll(TypeParam::ignore);

	for (size_t i = 0; i < this->iterations; ++i) {

		lhll.clear();

		this->update(lhll, this->uniq);

		this->hll.merge(lhll);

//		this->report(i, this->hll, this->uniq);

	}
	this->report(this->iterations, this->hll, this->uniq);

}

// testing the copy constructor
TYPED_TEST_P(HyperLogLog64Test, swap){


	this->hll.clear();
	this->uniq.clear();

    using HLL = hyperloglog64<
    		typename TypeParam::Type,
    		typename TypeParam::Hash,
			TypeParam::precision>;
	HLL lhll(TypeParam::ignore);
	lhll.clear();

	for (size_t i = 0; i < this->iterations; ++i) {

		this->update(lhll, this->uniq);

		this->hll.swap(std::move(lhll));

//		this->report(i, this->hll, this->uniq);

		lhll.swap(std::move(this->hll));
		this->hll.clear();
	}
	this->hll.swap(std::move(lhll));

	this->report(this->iterations, this->hll, this->uniq);

}




// now register the test cases
REGISTER_TYPED_TEST_CASE_P(HyperLogLog64Test,
		estimate,
		estimate_by_hash,
		merge, swap,
		estimate_batch,
		estimate_by_hash_batch);

//////////////////// RUN the tests with different types.

typedef ::testing::Types<
//		HyperLogLog64TestParam< uint8_t, ::std::hash<uint8_t>         , 4 >,
//		HyperLogLog64TestParam<uint16_t, ::std::hash<uint16_t>        , 4 >,
//		HyperLogLog64TestParam<uint32_t, ::std::hash<uint32_t>        , 4 >,
//		HyperLogLog64TestParam<uint64_t, ::std::hash<uint64_t>        , 4 >,
		// HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 4 >,
		// HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 4 >,
		// HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 4 >,
		// HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 4 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur<uint8_t> , 4 >,
		HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur<uint16_t>, 4 >,
		HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur<uint32_t>, 4 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur<uint64_t>, 4 >,
//		HyperLogLog64TestParam< uint8_t, ::std::hash<uint8_t>         , 6 >,
//		HyperLogLog64TestParam<uint16_t, ::std::hash<uint16_t>        , 6 >,
//		HyperLogLog64TestParam<uint32_t, ::std::hash<uint32_t>        , 6 >,
//		HyperLogLog64TestParam<uint64_t, ::std::hash<uint64_t>        , 6 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 6 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 6 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 6 >,
//		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 6 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur<uint8_t> , 6 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur<uint16_t>, 6 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur<uint32_t>, 6 >,
//		HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur<uint64_t>, 6 >,
//		HyperLogLog64TestParam< uint8_t, ::std::hash<uint8_t>         , 10 >,
//		HyperLogLog64TestParam<uint16_t, ::std::hash<uint16_t>        , 10 >,
//		HyperLogLog64TestParam<uint32_t, ::std::hash<uint32_t>        , 10 >,
//		HyperLogLog64TestParam<uint64_t, ::std::hash<uint64_t>        , 10 >,
////		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 10 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 10 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 10 >,
//		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 10 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur<uint8_t> , 10 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur<uint16_t>, 10 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur<uint32_t>, 10 >,
//		HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur<uint64_t>, 10 >,
////		HyperLogLog64TestParam< uint8_t, ::std::hash<uint8_t>         , 12 >,
////		HyperLogLog64TestParam<uint16_t, ::std::hash<uint16_t>        , 12 >,  // bad case - only 4 bits left...
		HyperLogLog64TestParam<uint32_t, ::std::hash<uint32_t>        , 12 >,
		HyperLogLog64TestParam<uint64_t, ::std::hash<uint64_t>        , 12 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12 >,
		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12 >,
		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur<uint8_t> , 12 >,
		HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur<uint16_t>, 12 >,
		HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur<uint32_t>, 12 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur<uint64_t>, 12 >,
//		HyperLogLog64TestParam< uint8_t, ::std::hash<uint8_t>         , 14 >,
//		HyperLogLog64TestParam<uint16_t, ::std::hash<uint16_t>        , 14 >,
//		HyperLogLog64TestParam<uint32_t, ::std::hash<uint32_t>        , 14 >,
//		HyperLogLog64TestParam<uint64_t, ::std::hash<uint64_t>        , 14 >,
		// HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 14 >,
		// HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 14 >,
		// HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 14 >,
		// HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 14 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur<uint8_t> , 14 >,
		HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur<uint16_t>, 14 >,
		HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur<uint32_t>, 14 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur<uint64_t>, 14 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12, 1 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12, 1 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12, 1 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12, 1 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12, 2 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12, 2 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12, 2 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12, 2 >,
//		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12, 4 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12, 4 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12, 4 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12, 4 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12, 8 >,
//		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12, 8 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12, 8 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12, 8 >,
		HyperLogLog64TestParam< uint8_t, ::fsc::hash::farm<uint8_t>   , 12, 16 >,
		HyperLogLog64TestParam<uint16_t, ::fsc::hash::farm<uint16_t>  , 12, 16 >,
//		HyperLogLog64TestParam<uint32_t, ::fsc::hash::farm<uint32_t>  , 12, 16 >,
		HyperLogLog64TestParam<uint64_t, ::fsc::hash::farm<uint64_t>  , 12, 16 >,
	    HyperLogLog64TestParam< uint8_t, ::fsc::hash::crc32c<uint8_t> , 12 >,
	    HyperLogLog64TestParam<uint16_t, ::fsc::hash::crc32c<uint16_t>, 12 >,
	    HyperLogLog64TestParam<uint32_t, ::fsc::hash::crc32c<uint32_t>, 12 >,
	    HyperLogLog64TestParam<uint64_t, ::fsc::hash::crc32c<uint64_t>, 12 >,
	    HyperLogLog64TestParam< uint8_t, ::fsc::hash::clhash<uint8_t> , 12 >,
	    HyperLogLog64TestParam<uint16_t, ::fsc::hash::clhash<uint16_t>, 12 >,
	    HyperLogLog64TestParam<uint32_t, ::fsc::hash::clhash<uint32_t>, 12 >,
	    HyperLogLog64TestParam<uint64_t, ::fsc::hash::clhash<uint64_t>, 12 >,
    HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur32<uint8_t> , 12 >,
    HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur32<uint16_t>, 12 >,
    HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur32<uint32_t>, 12 >,
    HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur32<uint64_t>, 12 >,
    HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur3sse32<uint8_t> , 12 >,
    HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur3sse32<uint16_t>, 12 >,
    HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur3sse32<uint32_t>, 12 >,
    HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur3sse32<uint64_t>, 12 >,
// segv during construction of the test.  not sure why.
    HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur3avx32<uint8_t> , 12 >,
    HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur3avx32<uint16_t>, 12 >,
    HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur3avx32<uint32_t>, 12 >,
	HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur3avx32<uint64_t>, 12 >,
	HyperLogLog64TestParam< uint8_t, ::fsc::hash::murmur3avx64<uint8_t> , 12 >,
    HyperLogLog64TestParam<uint16_t, ::fsc::hash::murmur3avx64<uint16_t>, 12 >,
    HyperLogLog64TestParam<uint32_t, ::fsc::hash::murmur3avx64<uint32_t>, 12 >,
    HyperLogLog64TestParam<uint64_t, ::fsc::hash::murmur3avx64<uint64_t>, 12 >


> HyperLogLog64TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Bliss, HyperLogLog64Test, HyperLogLog64TestTypes);
