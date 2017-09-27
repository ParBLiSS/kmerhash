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
 * math_utils.hpp
 *
 *  Created on: Mar 1, 2017
 *      Author: tpan
 */

#ifndef KMERHASH_MATH_UTILS_HPP_
#define KMERHASH_MATH_UTILS_HPP_


/*
 * get the next power of 2 for unsigned integer type.  based on http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
// USE THE SIZEOF SPECIALIZATIONS - no compiler warning about shifting more than size of element.
//template <typename T>
//inline T next_power_of_2(T x) {
//	static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value, "ERROR: can only find power of 2 for unsigned integers.");
//
//	--x;
//	switch (sizeof(T)) {
//	case 8:
//		x |= x >> 32;
//	case 4:
//		x |= x >> 16;
//	case 2:
//		x |= x >> 8;
//	default:
//		x |= x >> 4;
//		x |= x >> 2;
//		x |= x >> 1;
//		break;
//	}
//	++x;
//	return x;
//}
//
#if defined(__LZCNT__)

#include <x86intrin.h>

template <typename T>
inline constexpr T next_power_of_2(T x) {

  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");
  return  0x1ULL << (64 - _lzcnt_u64(x-1));
}

#else

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 1), int>::type = 1>
inline constexpr T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	++x;
	return x;
}


template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 2), int>::type = 1>
inline constexpr T next_power_of_2(T x) {

  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");
	--x;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	++x;
	return x;
}

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 4), int>::type = 1>
inline constexpr T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
	  x |= x >> 16;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	++x;
	return x;
}

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 8), int>::type = 1 >
inline constexpr T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
	  x |= x >> 32;
  	x |= x >> 16;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	++x;
	return x;
}

#endif


#endif /* KMERHASH_MATH_UTILS_HPP_ */
