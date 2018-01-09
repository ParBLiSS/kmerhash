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

#include <numeric>

#if defined(__INTEL_COMPILER)
#define CONSTEXPR
#else
#define CONSTEXPR constexpr
#endif
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

inline uint64_t next_power_of_2(uint64_t x) {

//  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
//                "ERROR: can only find power of 2 for unsigned integers.");
  return  0x1ULL << (64 - __lzcnt64(x-1));
}

#elif defined(__GNUG__) && !defined(__INTEL_COMPILER)


inline uint64_t next_power_of_2(uint64_t x) {

//  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
//                "ERROR: can only find power of 2 for unsigned integers.");
  return  0x1ULL << (64 - __builtin_lzcll(x-1));
}

#else

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 1), int>::type = 1>
inline CONSTEXPR T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	return x + 1;
}


template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 2), int>::type = 1>
inline CONSTEXPR T next_power_of_2(T x) {

  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");
	--x;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	return x + 1;
}

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 4), int>::type = 1>
inline CONSTEXPR T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
	  x |= x >> 16;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	return x + 1;
}

template <typename T, typename ::std::enable_if<
	 (sizeof(T) == 8), int>::type = 1 >
inline CONSTEXPR T next_power_of_2(T x) {
  static_assert(::std::is_integral<T>::value && !::std::is_signed<T>::value,
                "ERROR: can only find power of 2 for unsigned integers.");

	--x;
	  x |= x >> 32;
  	x |= x >> 16;
		x |= x >> 8;
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
	return x + 1;
}

#endif



// compute gcd
#include <numeric>

#if defined(__BMI__)

// use built in.  from  https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
template <typename T,
 	 typename ::std::enable_if<(sizeof(T) > 4), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __tzcnt_u64(a|b);
  a >>= __tzcnt_u64(a);

  do
   {
    b>>=__tzcnt_u64(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}

template <typename T,
 	 typename ::std::enable_if<(sizeof(T) <= 4) && (sizeof(T) > 2), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __tzcnt_u32(a|b);
  a >>= __tzcnt_u32(a);

  do
   {
    b>>=__tzcnt_u32(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}

template <typename T,
 	 typename ::std::enable_if<(sizeof(T) <= 2), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __tzcnt_u16(a|b);
  a >>= __tzcnt_u16(a);

  do
   {
    b>>=__tzcnt_u16(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}

#elif defined(__GNUG__) && !defined(__INTEL_COMPILER)
// use built in.  from  https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
template <typename T,
 	 typename ::std::enable_if<(sizeof(T) > 4), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __builtin_ctzll(a|b);
  a >>= __builtin_ctzll(a);

  do
   {
    b>>=__builtin_ctzll(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}

template <typename T,
 typename ::std::enable_if<(sizeof(T) <= 4) && (sizeof(T) > 2), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __builtin_ctz(a|b);
  a >>= __builtin_ctz(a);

  do
   {
    b>>=__builtin_ctz(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}

template <typename T,
  typename ::std::enable_if<(sizeof(T) <= 2), int>::type = 1>
T gcd(T a, T b)
{
  if (a == 0) return b;
  if (b == 0) return a;

  int shift = __builtin_ctzs(a|b);
  a >>= __builtin_ctzs(a);

  do
   {
    b>>=__builtin_ctzs(b);

    if (a>b) std::swap(a,b);
    b-=a;
   } while (b);

  return a << shift;
}


#else

// no built in, so use iterative + mod.  from https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
template <typename T>
T gcd(T a, T b)
{
	T t;
	while (b)
	{
		t=b;
		b=a % b;
		a=t;
	}
	return a;
}


#endif

template <typename T>
T lcm(T a, T b)
{
    T temp = gcd(a, b);

    return temp ? ((a / temp) * b) : 0;
}


template <typename T>
constexpr T constexpr_gcd(T a, T b)
{
	return b ? constexpr_gcd(b, a % b) : a;
}

template <typename T>
constexpr T constexpr_lcm(T a, T b)
{
	return constexpr_gcd(a, b) ? (( a / constexpr_gcd(a, b) ) * b ) : 0;
}



#endif /* KMERHASH_MATH_UTILS_HPP_ */
