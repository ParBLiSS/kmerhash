/*
 * Copyright 2016 Georgia Institute of Technology
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
 * @file    mem_utils.hpp
 * @ingroup
 * @author  tpan
 * @brief   memory utilities, such as aligned allocation.
 * @details
 *
 *
 */

#ifndef KMERHASH_MEM_UTILS_HPP
#define KMERHASH_MEM_UTILS_HPP

#include <cstdlib>	// posix_memalign
#include <algorithm>  //std::fill
#include <stdexcept>  //logic_error

namespace utils {

	namespace mem {


		/// allocate aligned memory
		template <typename T>
		inline T* aligned_alloc(size_t const & cnt, size_t const & align = 64) {
			unsigned char * ptr = nullptr;
			if (posix_memalign(reinterpret_cast<void **>(&ptr), align, cnt * sizeof(T))) {
			  free(ptr);
			  throw std::length_error("failed to allocate aligned memory");
			}
			return reinterpret_cast<T *>(ptr);
		}

		template <typename T>
		inline void init(T* ptr, size_t const & cnt) {
			::std::fill(ptr, ptr+cnt, T());
		}

		template <typename T>
		inline void aligned_free(T* ptr) {
			free(ptr);
		}

	}  // mem ns
}  // utils ns


#endif // MEM_UTILS_HPP
