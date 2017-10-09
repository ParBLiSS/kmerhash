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
 * @file    io_utils.hpp
 * @ingroup
 * @author  tpan
 * @brief   extends mxx to send/recv incrementally, so as to minimize memory use and allocation
 * @details
 *
 */

#ifndef KMERHASH_IO_UTILS_HPP
#define KMERHASH_IO_UTILS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>  //logic_error
#include <iterator>

template <typename IT>
void serialize(IT _begin, IT _end, std::string const & filename) {
	// open the file
	 std::ofstream fout(filename, std::ios::out | std::ios::binary);

	 // first write the element size
	 size_t el_size = sizeof(typename ::std::iterator_traits<IT>::value_type);
	 fout.write(reinterpret_cast<char *>(&el_size), sizeof(size_t));

	 // then write the number of elements
	 size_t count = std::distance(_begin, _end);
	 fout.write(reinterpret_cast<char *>(&count), sizeof(size_t));

	 // finally write the elements.
	 for (IT it = _begin; it != _end; ++it) {
		 fout.write(reinterpret_cast<const char *>(&(*it)), el_size);
	 }
	 // close the file now.
	 fout.close();
}

template <typename T>
void serialize_vector(std::vector<T> const & input, std::string const & filename) {
	// open the file
	 std::ofstream fout(filename, std::ios::out | std::ios::binary);

	 // first write the element size
	 size_t size = sizeof(T);
	 fout.write(reinterpret_cast<char *>(&size), sizeof(size_t));

	 // then write the number of elements
	 size = input.size();
	 fout.write(reinterpret_cast<char *>(&size), sizeof(size_t));

	 // finally write the elements.
	 fout.write(reinterpret_cast<const char *>(input.data()), size * sizeof(T));

	 // close the file now.
	 fout.close();
}

template <typename T>
std::vector<T> deserialize_vector(std::string const & filename) {

	// open the file
	std::ifstream fin(filename, std::ios::in | std::ios::binary);

	 // first read the element size
	 size_t el_size;
	 fin.read(reinterpret_cast<char *>(&el_size), sizeof(size_t));

	 if (el_size != sizeof(T)) {
		 throw std::logic_error("input element size not as specified ");
	 }

	 // then write the number of elements
	 size_t n_el;
	 fin.read(reinterpret_cast<char *>(&n_el), sizeof(size_t));

	 std::vector<T> output(n_el);

	 // finally write the elements.
	 fin.read(reinterpret_cast<char *>(output.data()), n_el * sizeof(T));

	 // close the file now.
	 fin.close();

	 return output;
}

#endif // IO_UTILS_HPP
