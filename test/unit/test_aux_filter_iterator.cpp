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
 * test_aux_filter_iterator.cpp
 * Test aux_filter_iterator class
 *  Created on: Feb 28, 2017
 *      Author: Tony Pan <tpan7@gatech.edu>
 */

#include "kmerhash/aux_filter_iterator.hpp"
#include "iterators/transform_iterator.hpp"
#include "iterators/zip_iterator.hpp"
#include "iterators/filter_iterator.hpp"
#include "iterators/counting_iterator.hpp"

#include <gtest/gtest.h>
#include <cstdint>  // for uint64_t, etc.
#include <limits>

using namespace bliss::iterator;

/*
 * test class holding some information.  Also, needed for the typed tests
 */
template<typename T>
class AuxFilterIteratorTest : public ::testing::Test
{
    static_assert(std::is_integral<T>::value, "data type has to be integral.");

  protected:

    struct Even {
      bool operator()(T const & v) {
        return (v % 2) == 0;
      }
    };

    struct Odd {
      bool operator()(T const & v) {
        return (v % 2) == 1;
      }
    };

    struct EvenZip {
    	bool operator()(std::pair<T, T> const & v) {
    		return (v.second % 2) == 0;
    	}
    };

    struct OddZip {
    	bool operator()(std::pair<T, T> const & v) {
    		return (v.second % 2) == 1;
    	}
    };

    struct unzip {
    	T operator()(std::pair<T, T> const & v) {
    		return v.first;
    	}
    };

    using Iter = CountingIterator<T>;
    using EvenAuxFilterIter = aux_filter_iterator<Iter, Iter, Even>;
    using OddAuxFilterIter = aux_filter_iterator<Iter, Iter, Odd>;

    using ZipIter = ZipIterator<Iter, Iter>;
    using EvenZipFilterIter = filter_iterator<EvenZip, ZipIter>;
    using OddZipFilterIter = filter_iterator<OddZip, ZipIter>;

    using EvenGoldIter = transform_iterator<
    		EvenZipFilterIter,
    		unzip>;
    using OddGoldIter = transform_iterator<
    		OddZipFilterIter,
    		unzip>;


	//setup counters.
	Iter begin;
    Iter end;

	Iter aux_begin;
    Iter aux_end;


	// set up odd aux_filter
    OddAuxFilterIter test_odd_begin;
    OddAuxFilterIter test_odd_end;

	// set up even aux_filter
    EvenAuxFilterIter test_even_begin;
    EvenAuxFilterIter test_even_end;


    OddGoldIter gold_odd_begin;
    OddGoldIter gold_odd_end;

	// set up even aux_filter
    EvenGoldIter gold_even_begin;
    EvenGoldIter gold_even_end;




    virtual void SetUp()
    {

    	//setup counters.
    	begin = Iter(23, 1);
        end = Iter(10023, 1);

    	aux_begin = Iter(0, 1);
        aux_end = Iter(10000, 1);


    	// set up odd aux_filter
        test_odd_begin  = OddAuxFilterIter(begin, aux_begin, aux_end, Odd());
        test_odd_end    = OddAuxFilterIter(end, aux_end, Odd());

    	// set up even aux_filter
        test_even_begin = EvenAuxFilterIter(begin, aux_begin, aux_end, Even());
        test_even_end   = EvenAuxFilterIter(end, aux_end, Even());

    	// set up zip iterators
        ZipIter zip_begin(begin, aux_begin);
        ZipIter zip_end(end, aux_end);

        // set up filter zp iterator
        OddZipFilterIter ofilter_begin(OddZip(), zip_begin, zip_end);
        OddZipFilterIter ofilter_end(OddZip(), zip_end);

        EvenZipFilterIter efilter_begin(EvenZip(), zip_begin, zip_end);
        EvenZipFilterIter efilter_end(EvenZip(), zip_end);


        gold_odd_begin = OddGoldIter(ofilter_begin, unzip());
        gold_odd_end   = OddGoldIter(ofilter_end, unzip());

    	// set up even aux_filter
        gold_even_begin = EvenGoldIter(efilter_begin, unzip());
        gold_even_end   = EvenGoldIter(efilter_end, unzip());

    }

};

// indicate this is a typed test
TYPED_TEST_CASE_P(AuxFilterIteratorTest);



// testing the copy constructor
TYPED_TEST_P(AuxFilterIteratorTest, increment){

double sum1 = 0;
double sum2 = 0;


std::cout << "base: [" << *(this->begin) << ", " << *(this->end) << "], aux [" << *(this->aux_begin) << "," << *(this->aux_end) << "]" << std::endl;


size_t count = 0;
    {
	  TypeParam i = 23;  // 23 matches to 0 for aux
	  auto it = this->test_even_begin;
	  auto it2 = this->gold_even_begin;

	  std::cout << "iterations = " << count <<
			  " test iterators same? " <<
			  (it == this->test_even_end ? "true" : "false") <<
			  " gold iterators same? " <<
			  (it2 == this->gold_even_end ? "true" : "false") <<
			  std::endl;

	  for (; it != this->test_even_end && it2 != this->gold_even_end; ++it, ++it2) {
		EXPECT_TRUE(*it % 2 == 1);
		EXPECT_TRUE(*it2 % 2 == 1);

		EXPECT_EQ(*it2, i);
		EXPECT_EQ(*it, i);
		EXPECT_EQ(*it, *it2);

		sum1 += *it;
		sum2 += *it2;

		i += 2;
		++count;
	  }

	  EXPECT_EQ(sum1, sum2);
	  std::cout << "iterations = " << count <<
			  " test iterators same? " <<
			  (it == this->test_even_end ? "true" : "false") <<
			  " gold iterators same? " <<
			  (it2 == this->gold_even_end ? "true" : "false") <<
			  std::endl;

	  // decrement
	  --it;
	  --it2;
	  i -= 2;
	  for (; it != this->test_even_begin && it2 != this->gold_even_begin; --it, --it2) {
			EXPECT_TRUE(*it % 2 == 1);
			EXPECT_TRUE(*it2 % 2 == 1);

			EXPECT_EQ(*it2, i);
			EXPECT_EQ(*it, i);
			EXPECT_EQ(*it, *it2);

			sum1 -= *it;
			sum2 -= *it2;

			--count;
			i -= 2;
	  }

	  EXPECT_EQ(sum1, sum2);
	  EXPECT_EQ(sum1, 23);

	  std::cout << "iterations = " << count <<
			  " test iterators same? " <<
			  (it == this->test_even_end ? "true" : "false") <<
			  " gold iterators same? " <<
			  (it2 == this->gold_even_end ? "true" : "false") <<
			  std::endl;


    }


    sum1 = 0.0;
    sum2 = 0.0;

    {
	  TypeParam i = 24;  // 24 matches to 1 aux
	  auto it = this->test_odd_begin;
	  auto it2 = this->gold_odd_begin;
	  for (; it != this->test_odd_end && it2 != this->gold_odd_end; ++it, ++it2) {
		EXPECT_TRUE(*it % 2 == 0);
		EXPECT_TRUE(*it2 % 2 == 0);

		EXPECT_EQ(*it2, i);
		EXPECT_EQ(*it, i);
		EXPECT_EQ(*it, *it2);

		sum1 += *it;
		sum2 += *it2;

		i += 2;

	  }
	  EXPECT_EQ(sum1, sum2);

	  // decrement
	  --it;
	  --it2;
	  i -= 2;
	  for (; it != this->test_odd_begin && it2 != this->gold_odd_begin; --it, --it2) {
			EXPECT_TRUE(*it % 2 == 0);
			EXPECT_TRUE(*it2 % 2 == 0);

			EXPECT_EQ(*it2, i);
			EXPECT_EQ(*it, i);
			EXPECT_EQ(*it, *it2);

			sum1 -= *it;
			sum2 -= *it2;

			i -= 2;
	  }

    }

    EXPECT_EQ(sum1, sum2);
    EXPECT_EQ(sum1, 24);


}






// now register the test cases
REGISTER_TYPED_TEST_CASE_P(AuxFilterIteratorTest, increment);


//////////////////// RUN the tests with different types.

typedef ::testing::Types<int32_t, uint32_t> AuxFilterIteratorTestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Bliss, AuxFilterIteratorTest, AuxFilterIteratorTestTypes);
