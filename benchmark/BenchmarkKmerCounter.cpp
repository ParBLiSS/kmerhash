/*
 * Copyright 2015 Georgia Institute of Technology
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
 * @file    test_threads.cpp
 * @ingroup
 * @author  Tony Pan <tpan7@gatech.edu>
 * @brief
 * @details
 *

 */
#define _FILE_OFFSET_BITS 64
#define _LARGEFILE64_SOURCE

#ifndef _GNU_SOURCE
#define _GNU_SOURCE     // get O_DIRECT from fnctl
#endif

#include "bliss-config.hpp"


#ifdef VTUNE_ANALYSIS
#define MEASURE_DISABLED 0

#define MEASURE_RESERVE 10
#define MEASURE_TRANSFORM 11
#define MEASURE_UNIQUE 12
#define MEASURE_BUCKET 13
#define MEASURE_PERMUTE 14
#define MEASURE_A2A 15
#define MEASURE_COMPRESS 16

#define MEASURE_INSERT 1
#define MEASURE_FIND 2
#define MEASURE_COUNT 3
#define MEASURE_ERASE 4

static int measure_mode = MEASURE_DISABLED;

#include <ittnotify.h>

#endif

#include <fcntl.h>      // for O_DIRECT
#include <cstdlib>
#include <errno.h>      // posix calls generate errno.  also for perror()
#include <sys/ioctl.h>  // for ioctl, to get disk block size.
#include <linux/fs.h>  // for BLKSSZGET

#include <sys/mman.h> // mmap
#include <sys/stat.h>  // for mmap sysconf
#include <sys/types.h>
#include <sys/sysinfo.h>

// for RLIMITS_FSIZE
#include <sys/time.h>
#include <sys/resource.h>  

#include <unistd.h> // for sysconf

#include <functional>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>  // for system("pause");

#include "utils/logging.h"

#include "common/alphabets.hpp"
#include "common/kmer.hpp"
#include "common/base_types.hpp"
#include "utils/kmer_utils.hpp"
#include "utils/transform_utils.hpp"

#include "io/mxx_support.hpp"

#include "io/sequence_iterator.hpp"
#include "io/sequence_id_iterator.hpp"
#include "io/filtered_sequence_iterator.hpp"

#include "iterators/transform_iterator.hpp"

#include "common/kmer_iterators.hpp"

#include "iterators/zip_iterator.hpp"

#include "index/quality_score_iterator.hpp"
#include "index/kmer_hash.hpp"   // workaround for distributed_map_base requiring farm hash.

#include "kmerhash/hash_new.hpp"
#include "kmerhash/distributed_robinhood_map.hpp"
#include "kmerhash/distributed_batched_robinhood_map.hpp"
#include "kmerhash/distributed_batched_radixsort_map.hpp"
#include "kmerhash/hybrid_batched_robinhood_map.hpp"
#include "kmerhash/hybrid_batched_radixsort_map.hpp"
#include "kmerhash/math_utils.hpp"  // lcm

#include "index/kmer_index.hpp"

#include "utils/benchmark_utils.hpp"
#include "utils/exception_handling.hpp"

#include "tclap/CmdLine.h"
#include "utils/tclap_utils.hpp"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

// ================ define preproc macro constants
// needed as #if can only calculate constant int expressions
#define FASTA 1
#define FASTQ 0

#define IDEN 10
#define LEX 11
#define XOR 12

#define STD 21
#define FARM 22
#define FARM32 23
#define MURMUR 24
#define MURMUR32 25
#define MURMUR32sse 26
#define MURMUR32avx 27
#define MURMUR64avx 28
#define CRC32C 29
#define CLHASH 30

#define POS 31
#define POSQUAL 32
#define COUNT 33

#define SORTED 41
#define ORDERED 42
//#define VEC 43
//#define COMPACTVEC 44
//#define HASHEDVEC 45
#define UNORDERED 46
#define DENSEHASH 47
#define ROBINHOOD 48
#define BROBINHOOD 49
#define RADIXSORT 50
#define MTROBINHOOD 51
#define MTRADIXSORT 52

#define SINGLE 61
#define CANONICAL 62
#define BIMOLECULE 63




//================= define types - changeable here...

//============== index input file format
#if (pPARSER == FASTA)
	using IdType = bliss::common::LongSequenceKmerId;
#define PARSER_TYPE ::bliss::io::FASTAParser
#elif (pPARSER == FASTQ)
	using IdType = bliss::common::ShortSequenceKmerId;
#define PARSER_TYPE ::bliss::io::FASTQParser
#endif


// ============  index value type
using QualType = float;
using KmerInfoType = std::pair<IdType, QualType>;
using CountType = uint16_t;


//========   Kmer parameters
#if (pDNA == 16)
using Alphabet = bliss::common::DNA16;
#elif (pDNA == 5)
using Alphabet = bliss::common::DNA5;
#elif (pDNA == 4)
using Alphabet = bliss::common::DNA;
#endif

// USE the same type for Count and Kmer Word - better memory usage?

#if defined(pK)
#if (pK == 15)
	using KmerType = bliss::common::Kmer<pK, Alphabet, uint32_t>;
#else
	using KmerType = bliss::common::Kmer<pK, Alphabet, uint64_t>;
#endif
#else
using KmerType = bliss::common::Kmer<31, Alphabet, uint64_t>;
#endif





#if (pINDEX == POS)
	using ValType = IdType;
#elif (pINDEX == POSQUAL)
	using ValType = KmerInfoType;
#elif (pINDEX == COUNT)
	using ValType = CountType;
#endif




//============== MAP properties


//----- get them all. may not use subsequently.

// distribution transforms
#if (pDistTrans == LEX)
	template <typename KM>
	using DistTrans = bliss::kmer::transform::lex_less<KM>;
#elif (pDistTrans == XOR)
	template <typename KM>
	using DistTrans = bliss::kmer::transform::xor_rev_comp<KM>;
#else //if (pDistTrans == IDEN)
	template <typename KM>
	using DistTrans = bliss::transform::identity<KM>;
#endif


	// distribution hash
	#if (pMAP == MTRADIXSORT) || (pMAP == RADIXSORT) || (pMAP == MTROBINHOOD) || (pMAP == BROBINHOOD)
	#if (pDistHash == IDEN)
		template <typename KM>
	//	using DistHash = bliss::kmer::hash::identity<KM, true>;
		using DistHash = ::fsc::hash::identity<KM>;
	#elif (pDistHash == MURMUR)
		template <typename KM>
	//	using DistHash = bliss::kmer::hash::murmur<KM, true>;
		using DistHash = ::fsc::hash::murmur<KM>;
	#elif (pDistHash == MURMUR32)
		template <typename KM>
		using DistHash = ::fsc::hash::murmur32<KM>;
	#elif (pDistHash == MURMUR32sse)
	  template <typename KM>
	  using DistHash = ::fsc::hash::murmur3sse32<KM>;
	#elif (pDistHash == MURMUR32avx)
	  template <typename KM>
	  using DistHash = ::fsc::hash::murmur3avx32<KM>;
	#elif (pDistHash == MURMUR64avx)
	  template <typename KM>
	  using DistHash = ::fsc::hash::murmur3avx64<KM>;
	#elif (pDistHash == CRC32C)
	  template <typename KM>
	  using DistHash = ::fsc::hash::crc32c<KM>;
	#elif (pDistHash == CLHASH)
	  template <typename KM>
	  using DistHash = ::fsc::hash::clhash<KM>;
	#elif (pDistHash == FARM32)
		template <typename KM>
		using DistHash = ::fsc::hash::farm32<KM>;
	#elif (pDistHash == FARM)
	  template <typename KM>
	  //using DistHash = bliss::kmer::hash::farm<KM, true>;
	  using DistHash = ::fsc::hash::farm<KM>;
	#else
	  static_assert(false, "RADIXSORT and BROBINHOOD do not support the specified distr hash function");
	#endif
	#else
	#if (pDistHash == STD)
		template <typename KM>
		using DistHash = bliss::kmer::hash::cpp_std<KM, true>;
	#elif (pDistHash == IDEN)
		template <typename KM>
		using DistHash = bliss::kmer::hash::identity<KM, true>;
	#elif (pDistHash == MURMUR)
		template <typename KM>
		using DistHash = bliss::kmer::hash::murmur<KM, true>;
	#elif (pDistHash == FARM)
	  template <typename KM>
	  	using DistHash = bliss::kmer::hash::farm<KM, true>;
	#else
	  static_assert(false, "DENSEHASH, unordered map, sorted, ordered, and ROBINHOOD do not support the specified distr hash function");
	#endif

	#endif

	// storage hash type
	#if (pMAP == MTRADIXSORT) || (pMAP == RADIXSORT) || (pMAP == MTROBINHOOD) || (pMAP == BROBINHOOD)
	#if (pStoreHash == IDEN)
		template <typename KM>
	//	using StoreHash = bliss::kmer::hash::identity<KM, false>;
		using StoreHash = ::fsc::hash::identity<KM>;
	#elif (pStoreHash == MURMUR)
		template <typename KM>
	//	using StoreHash = bliss::kmer::hash::murmur<KM, false>;
		using StoreHash = ::fsc::hash::murmur<KM>;
	#elif (pStoreHash == MURMUR32)
		template <typename KM>
		using StoreHash = ::fsc::hash::murmur32<KM>;
	#elif (pStoreHash == MURMUR32sse)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::murmur3sse32<KM>;
	#elif (pStoreHash == MURMUR32avx)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::murmur3avx32<KM>;
	#elif (pStoreHash == MURMUR64avx)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::murmur3avx64<KM>;
	#elif (pStoreHash == CRC32C)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::crc32c<KM>;
	#elif (pStoreHash == CLHASH)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::clhash<KM>;
	#elif (pStoreHash == FARM32)
		template <typename KM>
		using StoreHash = ::fsc::hash::farm32<KM>;
	#elif (pStoreHash == FARM)
	  template <typename KM>
	//  using StoreHash = bliss::kmer::hash::farm<KM, false>;
	  using StoreHash = ::fsc::hash::farm<KM>;
	#else
	  static_assert(false, "RADIXSORT and BROBINHOOD do not support the specified store hash function");
	#endif
	#else
	#if (pStoreHash == STD)
		template <typename KM>
		using StoreHash = bliss::kmer::hash::cpp_std<KM, false>;
	#elif (pStoreHash == IDEN)
		template <typename KM>
		using StoreHash = bliss::kmer::hash::identity<KM, false>;
	#elif (pStoreHash == MURMUR)
		template <typename KM>
		using StoreHash = bliss::kmer::hash::murmur<KM, false>;
	#elif (pStoreHash == FARM)
	  template <typename KM>
	  using StoreHash = bliss::kmer::hash::farm<KM, false>;
	#elif (pStoreHash == CRC32C)
	  template <typename KM>
	  using StoreHash = ::fsc::hash::crc32c<KM>;
	#else
	  static_assert(false, "DENSEHASH, unordered map, sorted, ordered, and ROBINHOOD do not support the specified store hash function");
	#endif
	#endif



// ==== define Map parameter
#if (pMAP == SORTED)
	// choose a MapParam based on type of map and kmer model (canonical, original, bimolecule)
	#if (pKmerStore == SINGLE)  // single stranded
		template <typename Key>
		using MapParams = ::bliss::index::kmer::SingleStrandSortedMapParams<Key>;
	#elif (pKmerStore == CANONICAL)
		template <typename Key>
		using MapParams = ::bliss::index::kmer::CanonicalSortedMapParams<Key>;
	#elif (pKmerStore == BIMOLECULE)  // bimolecule
		template <typename Key>
		using MapParams = ::bliss::index::kmer::BimoleculeSortedMapParams<Key>;
	#endif

	// DEFINE THE MAP TYPE base on the type of data to be stored.
	#if (pINDEX == POS) || (pINDEX == POSQUAL)  // multimap
		using MapType = ::dsc::sorted_multimap<
				KmerType, ValType, MapParams>;
	#elif (pINDEX == COUNT)  // map
		using MapType = ::dsc::counting_sorted_map<
				KmerType, ValType, MapParams>;
	#endif


#elif (pMAP == ORDERED)
	// choose a MapParam based on type of map and kmer model (canonical, original, bimolecule)
	#if (pKmerStore == SINGLE)  // single stranded
		template <typename Key>
		using MapParams = ::bliss::index::kmer::SingleStrandOrderedMapParams<Key, DistHash, ::std::less, DistTrans>;
	#elif (pKmerStore == CANONICAL)
		template <typename Key>
		using MapParams = ::bliss::index::kmer::CanonicalOrderedMapParams<Key, DistHash>;
	#elif (pKmerStore == BIMOLECULE)  // bimolecule
		template <typename Key>
		using MapParams = ::bliss::index::kmer::BimoleculeOrderedMapParams<Key, DistHash>;
	#endif

	// DEFINE THE MAP TYPE base on the type of data to be stored.
	#if (pINDEX == POS) || (pINDEX == POSQUAL)  // multimap
		using MapType = ::dsc::multimap<
				KmerType, ValType, MapParams>;
	#elif (pINDEX == COUNT)  // map
		using MapType = ::dsc::counting_map<
				KmerType, ValType, MapParams>;
	#endif


#else  // hashmap
//		struct MSBSplitter {
//		    template <typename Kmer>
//		    bool operator()(Kmer const & kmer) const {
//		      return (kmer.getData()[Kmer::nWords - 1] & ~(~(static_cast<typename Kmer::KmerWordType>(0)) >> 2)) > 0;
//		    }
//
//		    template <typename Kmer, typename V>
//		    bool operator()(std::pair<Kmer, V> const & x) const {
//		      return (x.first.getData()[Kmer::nWords - 1] & ~(~(static_cast<typename Kmer::KmerWordType>(0)) >> 2)) > 0;
//		    }
//		};
		//	#if (pDNA == 5) || (pKmerStore == CANONICAL)
		//	using Splitter = ::bliss::filter::TruePredicate;
		//	#else
		//	using Splitter = typename ::std::conditional<(KmerType::nBits == (KmerType::nWords * sizeof(typename KmerType::KmerWordType) * 8)),
		//			MSBSplitter, ::bliss::filter::TruePredicate>::type;
		//	#endif


  // choose a MapParam based on type of map and kmer model (canonical, original, bimolecule)
  #if (pKmerStore == SINGLE)  // single stranded
    template <typename Key>
    using MapParams = ::bliss::index::kmer::SingleStrandHashMapParams<Key, DistHash, StoreHash, DistTrans>;
  using SpecialKeys = ::bliss::kmer::hash::sparsehash::special_keys<KmerType, false>;
  #elif (pKmerStore == CANONICAL)
    template <typename Key>
    using MapParams = ::bliss::index::kmer::CanonicalHashMapParams<Key, DistHash, StoreHash>;
  using SpecialKeys = ::bliss::kmer::hash::sparsehash::special_keys<KmerType, true>;
  #elif (pKmerStore == BIMOLECULE)  // bimolecule
    template <typename Key>
    using MapParams = ::bliss::index::kmer::BimoleculeHashMapParams<Key, DistHash, StoreHash>;
  using SpecialKeys = ::bliss::kmer::hash::sparsehash::special_keys<KmerType, false>;
  #endif


  // DEFINE THE MAP TYPE base on the type of data to be stored.
  #if (pINDEX == POS) || (pINDEX == POSQUAL)  // multimap
	//    #if (pMAP == VEC)
	//      using MapType = ::dsc::unordered_multimap_vec<
	//          KmerType, ValType, MapParams>;
	//
	//   #elif (pMAP == UNORDERED)
	    #if (pMAP == UNORDERED)
	      using MapType = ::dsc::unordered_multimap<
	          KmerType, ValType, MapParams>;
	//    #elif (pMAP == COMPACTVEC)
	//      using MapType = ::dsc::unordered_multimap_compact_vec<
	//          KmerType, ValType, MapParams>;
	//    #elif (pMAP == HASHEDVEC)
	//      using MapType = ::dsc::unordered_multimap_hashvec<
	//          KmerType, ValType, MapParams>;
    #elif (pMAP == DENSEHASH)
      using MapType = ::dsc::densehash_multimap<
          KmerType, ValType, MapParams, SpecialKeys>;
    #endif
  #elif (pINDEX == COUNT)  // map
    #if (pMAP == DENSEHASH)
      using MapType = ::dsc::counting_densehash_map<
        KmerType, ValType, MapParams, SpecialKeys>;
#elif (pMAP == ROBINHOOD)
  using MapType = ::dsc::counting_robinhood_map<
      KmerType, ValType, MapParams>;
#elif (pMAP == MTROBINHOOD)
  using MapType = ::hsc::counting_batched_robinhood_map<
      KmerType, ValType, MapParams>;
#elif (pMAP == MTRADIXSORT)
  using MapType = ::hsc::counting_batched_radixsort_map<
      KmerType, ValType, MapParams>;
#elif (pMAP == BROBINHOOD)
  using MapType = ::dsc::counting_batched_robinhood_map<
      KmerType, ValType, MapParams>;
#elif (pMAP == RADIXSORT)
  using MapType = ::dsc::counting_batched_radixsort_map<
      KmerType, ValType, MapParams>;
#elif (pMAP == UNORDERED)
  using MapType = ::dsc::counting_unordered_map<
    KmerType, ValType, MapParams>;
#endif

  #endif


#endif




//================ FINALLY, the actual index type.

#if (pINDEX == POS)
	using IndexType = bliss::index::kmer::PositionIndex<MapType>;

#elif (pINDEX == POSQUAL)
  using IndexType = bliss::index::kmer::PositionQualityIndex<MapType>;

#elif (pINDEX == COUNT)  // map
#if (pMAP == SORTED)
	// SORTED ARRAY count index should parse kmer and value pair.
	using IndexType = bliss::index::kmer::CountIndex<MapType>;
#else
	// HASHTABLE based indices should parse kmer only.
	using IndexType = bliss::index::kmer::CountIndex2<MapType>;
#endif
#endif


	size_t get_file_size(std::string const & filename) {
		int fd = open64(filename.c_str(), O_RDONLY | O_LARGEFILE );
		if (fd < 0) {
	          std::cout << "ERROR: " << errno << std::endl;
	          perror("get_file_size open");
	          throw std::logic_error("ERROR: get_file_size open failed");
		}

		off64_t fsize = lseek64(fd, 0, SEEK_END);

		if (fsize < 0) {
	          std::cout << "ERROR: " << errno << std::endl;
	          perror("get_file_size");
	          throw std::logic_error("ERROR: get_file_size failed");
		}

		return fsize;
	}


	std::string get_error_string(std::string const & filename, std::string const & op_name, int const & return_val, mxx::comm const & comm) {
		char error_string[BUFSIZ];
		int length_of_error_string, error_class;
		std::stringstream ss;

		MPI_Error_class(return_val, &error_class);
		MPI_Error_string(error_class, error_string, &length_of_error_string);

		ss << "ERROR in mpiio: rank " << comm.rank() << " " << op_name << " " << filename << " error: " << error_string << std::endl;
		return ss.str();
	}

	std::string get_error_string(std::string const & filename, std::string const & op_name, int const & return_val, MPI_Status const & stat, mxx::comm const & comm) {
		char error_string[BUFSIZ];
		int length_of_error_string, error_class;
		std::stringstream ss;

		MPI_Error_class(return_val, &error_class);
		MPI_Error_string(error_class, error_string, &length_of_error_string);

		ss << "ERROR in mpiio: rank " << comm.rank() << " " << op_name << " " << filename << " error: " << return_val << " [" << error_string << "]";

		//		// status.MPI_ERROR does not appear to be decodable by error_class.  google search did not find how to decode it.
		//		MPI_Error_class(stat.MPI_ERROR, &error_class);
		//		MPI_Error_string(error_class, error_string, &length_of_error_string);

		ss << " MPI_Status error: [" << stat.MPI_ERROR << "]" << std::endl;

		return ss.str();
	}


	void write_mpiio(std::string const & filename, const unsigned char* data, size_t len, mxx::comm const & comm ) {
		// TODO: subcommunicator to work with only nodes that have data.

		/// MPI file handle
		MPI_File fh;

		int res = MPI_File_open(comm, const_cast<char *>(filename.c_str()), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
		if (res != MPI_SUCCESS) {
			throw ::bliss::utils::make_exception<::bliss::io::IOException>(get_error_string(filename, "open", res, comm));
		}

		res = MPI_File_set_size(fh, 0);
		if (res != MPI_SUCCESS) {
			throw ::bliss::utils::make_exception<::bliss::io::IOException>(get_error_string(filename, "truncate", res, comm));
		}

		// ensure atomicity is turned off
		MPI_File_set_atomicity(fh, 1);

		// get the global offset.
		size_t global_offset = ::mxx::exscan(len, comm);

		size_t step = (0x1 << 30);
		size_t iterations = (len + step - 1) / step;

		//std::cout << "rank " << comm.rank() << " mpiio write offset is " << global_offset << " len " << len << " iterations " << iterations << ::std::endl;

		// get the maximum number of iterations
		iterations = ::mxx::allreduce(iterations, [](size_t const & x, size_t const & y){
			return (x >= y) ? x : y;
		}, comm);

		size_t remainder = len;
		size_t curr_step = step;
		MPI_Status stat;
		int count = 0;
		for (size_t i = 0; i < iterations; ++i) {
			curr_step = std::min(remainder, step);

			res = MPI_File_write_at_all( fh, global_offset, const_cast<unsigned char*>(data), curr_step, MPI_BYTE, &stat);

			if (res != MPI_SUCCESS)
				throw ::bliss::utils::make_exception<::bliss::io::IOException>(get_error_string(filename, "write", res, stat, comm));

			res = MPI_Get_count(&stat, MPI_BYTE, &count);
			if (res != MPI_SUCCESS)
				throw ::bliss::utils::make_exception<::bliss::io::IOException>(get_error_string(filename, "write count", res, stat, comm));

			if (static_cast<size_t>(count) != curr_step) {
				std::stringstream ss;
				ss << "ERROR in mpiio: rank " << comm.rank() << " write error. request " << curr_step << " bytes got " << count << " bytes" << std::endl;

				throw ::bliss::utils::make_exception<::bliss::io::IOException>(ss.str());
			}

			global_offset += curr_step;
			data += curr_step;
			remainder -= curr_step;
		}

		// close the file when done.
		MPI_File_close(&fh);
	}


	  /// opens a file. side effect computes size of the file.
	  int open_out_file(std::string const & filename, bool direct = true) {
	//printf("open seq file\n");

	    if (filename.length() == 0) {
	    	throw std::invalid_argument("ERROR: bad file name for open");
	    }

	    int fd;
	    if (direct) {

	    // open the file and get a handle.
	      fd = open64(filename.c_str(), O_WRONLY | O_LARGEFILE | O_CREAT | O_TRUNC | O_DIRECT , S_IRWXU | S_IRWXG );
	    } else {
	      fd = open64(filename.c_str(), O_WRONLY | O_LARGEFILE | O_CREAT | O_TRUNC, S_IRWXU | S_IRWXG );
	    }

	    if (fd == -1)
	    {
	      // if open failed, throw exception.
	      ::std::stringstream ss;
	      int myerr = errno;
	      ss << "ERROR in base_file open: ["  << filename << "] error " << myerr << ": " << strerror(myerr);
	      throw ::bliss::utils::make_exception<::bliss::io::IOException>(ss.str());
	    }

	    return fd;
	  }

	  /// funciton for closing a file
	  void close_out_file(int & fd) {
	    if (fd >= 0) {
	      close(fd);
	      fd = -1;
	    }
	  }


	void write_posix(std::string const & filename, const void* data, size_t len ) {
		if (len == 0) return;

	    // first close file.
	    int fd = open_out_file(filename, false);

	    if (fd == -1) {
	      throw std::logic_error("ERROR: read_range: file pointer is null");
	    }

	    // write has limit at 0x7ffff000 (2,147,479,552) bytes at a time.
			// SSIZE_MAX does not seem to be the right check, so leave it at 
			// 1 GB for now.
	    const size_t step = (0x1 << 30);
	    size_t iterations = len >> 30;
	    size_t remainder = len & (step - 1);

	    const unsigned char* ptr = (const unsigned char*)data;
	    size_t written = 0;
	    ssize_t iter_written;

	    for ( size_t i = 0; i < iterations; ++i) {
	      iter_written = write(fd, ptr, step);


        if (iter_written < 0) {
          std::cout << "ERROR: " << errno << std::endl;
          perror("write");
          throw std::logic_error("ERROR: write failed");
        } else if (iter_written < static_cast<ssize_t>(step)) {
          std::cout << "ERROR: write " << iter_written << " specified " << step << std::endl;
          throw std::logic_error("ERROR: written less than specified");
        }

        ptr += step;
        written += iter_written;
	    }

	    if (remainder > 0) {

        // handle remainder
        iter_written = write(fd, ptr, remainder);

        if (iter_written < 0) {
          std::cout << "ERROR: " << errno << std::endl;
          perror("write remainder ");
          throw std::logic_error("ERROR: write remainder  failed");
        } else if (iter_written < static_cast<ssize_t>(remainder)) {
          std::cout << "ERROR: remainder write " << iter_written << " specified " << remainder << std::endl;
          throw std::logic_error("ERROR: remainder written less than specified");
        }

        written += iter_written;
	    }

		// close the file when done.
		close_out_file(fd);

	}


	// data must be aligned to 512 boundaries.
	void write_posix_direct(std::string const & filename, const void* data, size_t bytes, long block_size) {
		if (bytes == 0) return;

		// size_t last = len % 512;

	    // first close file.
	    int fd = open_out_file(filename, true);

	    //std::cout << "write " << bytes << " bytes to " << filename << std::endl;

	    if (fd == -1) {
	      throw std::logic_error("ERROR: direct write: file pointer is null");
	    }

	    // get block size.
	    if (block_size <= 0) {
	      throw std::invalid_argument("ERROR: bad block size.");
	    }

	    // part of block that can be written in 512 blocks.
	    size_t remainder = bytes % block_size;
	    size_t blocked = bytes - remainder;

	    // write has limit at 0x7ffff000 (2,147,479,552) bytes at a time.
			// SSIZE_MAX does not seem to be the right check, so leave it at 
			// 1 GB for now.
      const size_t step = (0x1 << 30);
      size_t iterations = blocked >> 30;
      size_t rem = blocked & (step - 1);

      const unsigned char* ptr = (const unsigned char*)data;
      size_t written = 0;
      ssize_t iter_written;

			//std::cout << "ready to write to " << reinterpret_cast<size_t>(ptr) << std::endl;

      for ( size_t i = 0; i < iterations; ++i) {
        iter_written = write(fd, ptr, step);

        if (iter_written < 0) {
          std::cout << "ERROR: " << errno << std::endl;
          perror("write direct");
          throw std::logic_error("ERROR: write direct failed");
        } else if (iter_written < static_cast<ssize_t>(step)) {
          std::cout << "ERROR: direct write " << iter_written << " specified " << step << std::endl;
          throw std::logic_error("ERROR: direct written less than specified");
        }

        ptr += step;
        written += iter_written;
				//std::cout << "written " << written << " iter " << i << std::endl;
      }
			//std::cout << "wrote main part " << std::endl;
      if (rem > 0) {

        // handle 1GB remainders.  still aligned to 512
        iter_written = write(fd, ptr, rem);

        if (iter_written < 0) {
          std::cout << "ERROR: " << errno << std::endl;
          perror("direct write medium remainder ");
          throw std::logic_error("ERROR: direct write medium remainder  failed");
        } else if (iter_written < static_cast<ssize_t>(rem)) {
          std::cout << "ERROR: direct medium remainder write " << iter_written << " specified " << rem << std::endl;
          throw std::logic_error("ERROR: direct medium remainder written less than specified");
        }

        ptr += rem;
        written += iter_written;
      }
			//std::cout << "wrote rem1 part " << std::endl;

      if (remainder > 0) {

        // write the remainder in a 512 block
        unsigned char *x = ::utils::mem::aligned_alloc<unsigned char>(block_size, 512);
        memcpy(x, ptr, remainder);
        memset(x + remainder, 0, block_size - remainder);
        iter_written = write(fd, x, block_size);

        if (iter_written < 0) {
          std::cout << "ERROR: " << errno << std::endl;
          perror("write direct small remainder");
          throw std::logic_error("ERROR: direct write small remainder failed");
        } else if (iter_written < static_cast<ssize_t>(remainder)) {
          std::cout << "ERROR: wrote " << iter_written << " specified " << remainder << std::endl;

          throw std::logic_error("ERROR: direct write small remainder: written less than specified");
        }

        written += iter_written;

        ::utils::mem::aligned_free(x);
      }

			//std::cout << "wrote rem2 part " << std::endl;

		// close the file when done.
		close_out_file(fd);

	}


	unsigned char * map_out_file(std::string const & filename, size_t len, size_t offset, mxx::comm const & comm) {

    if (filename.length() == 0) {
      throw std::invalid_argument("ERROR: bad file name for open");
    }


	  size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

	  size_t total = mxx::reduce(len, comm.size() - 1, comm);

	  int fd;

	  //    if (comm.rank() == (comm.size() - 1)) {
	  //      std::cout << "total size is " << (offset + target_size) << " by reduce " << total << std::endl;
	  //    }

    // highest rank make sure file is big enough.
    if (comm.rank() == (comm.size()-1)) {
      fd = open_out_file(filename, false); // can't use direct - 512 boundary would come in...

      // see to end
      if (lseek64(fd, total -1, SEEK_SET) == -1)
      {
          close(fd);
          perror("Error calling lseek() to 'stretch' the file");
          throw std::logic_error("ERROR: failed to stretch file, seek");
      }

      // write something at last position
      if (write(fd, "", 1) == -1) {
        close(fd);
        perror("Error calling write() to 'stretch' the file");
        throw std::logic_error("ERROR: failed to stretch file by write");

      }

      close_out_file(fd);

      std::cout << "completed stretching." << std::endl;
    }
	  comm.barrier();

    // cannot be write only.
    //fd = open64(filename.c_str(), O_RDWR | O_LARGEFILE | O_DIRECT, S_IRWXU | S_IRWXG );
    fd = open64(filename.c_str(), O_RDWR | O_LARGEFILE , S_IRWXU | S_IRWXG );

    if (fd == -1)
    {
      // if open failed, throw exception.
      ::std::stringstream ss;
      int myerr = errno;
      ss << "ERROR in file map: ["  << filename << "] error " << myerr << ": " << strerror(myerr);
      throw std::logic_error("ERROR: faile to open file for mapping");
    }

	  unsigned char * addr = (unsigned char*)(mmap64(NULL, len + offset - pa_offset, PROT_WRITE, MAP_SHARED, fd, pa_offset ));

	  if (addr == MAP_FAILED) {
	    close_out_file(fd);
	    std::cout << "ERROR: for file id " << fd << " len " << len << " offset " << offset << std::endl;
	    perror("mmap");
	    throw std::logic_error("ERROR: faile to map file");
	  }

    close_out_file(fd);  // and close it...

	  return addr;

	}
	int unmap_out_file(unsigned char * addr, size_t len, size_t offset, mxx::comm const & comm) {
	  if (msync(addr, len, MS_ASYNC) == -1) {
	    perror("ERROR: failed to sync before unmap");
	  }

    size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

    comm.barrier();

	  return munmap(addr, len + offset - pa_offset);
	}


	unsigned char * map_out_file(std::string const & filename, size_t len, size_t offset) {

    if (filename.length() == 0) {
      throw std::invalid_argument("ERROR: bad file name for open");
    }


	  size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);
		size_t total = len;

	  int fd;

	  //    if (comm.rank() == (comm.size() - 1)) {
	  //      std::cout << "total size is " << (offset + target_size) << " by reduce " << total << std::endl;
	  //    }

    // highest rank make sure file is big enough.
		fd = open_out_file(filename, false); // can't use direct - 512 boundary would come in...

		// see to end
		if (lseek64(fd, total -1, SEEK_SET) == -1)
		{
				close(fd);
				perror("Error calling lseek() to 'stretch' the file");
				throw std::logic_error("ERROR: failed to stretch file, seek");
		}

		// write something at last position
		if (write(fd, "", 1) == -1) {
			close(fd);
			perror("Error calling write() to 'stretch' the file");
			throw std::logic_error("ERROR: failed to stretch file by write");

		}

		close_out_file(fd);

		//std::cout << "completed stretching." << std::endl;

	// cannot be write only
    //fd = open64(filename.c_str(), O_RDWR | O_LARGEFILE | O_DIRECT, S_IRWXU | S_IRWXG );
    fd = open64(filename.c_str(), O_RDWR | O_LARGEFILE, S_IRWXU | S_IRWXG );

    if (fd == -1)
    {
      // if open failed, throw exception.
      ::std::stringstream ss;
      int myerr = errno;
      ss << "ERROR in file map: ["  << filename << "] error " << myerr << ": " << strerror(myerr);
      throw std::logic_error("ERROR: faile to open file for mapping");
    }

	  unsigned char * addr = (unsigned char*)(mmap64(NULL, len + offset - pa_offset, PROT_WRITE, MAP_SHARED, fd, pa_offset ));

	  if (addr == MAP_FAILED) {
	    close_out_file(fd);
	    std::cout << "ERROR: for file id " << fd << " len " << len << " offset " << offset << std::endl;
	    perror("mmap");
	    throw std::logic_error("ERROR: faile to map file");
	  }

    close_out_file(fd);  // and close it...

	  return addr;

	}
	int unmap_out_file(unsigned char * addr, size_t len, size_t offset) {
	  if (msync(addr, len, MS_ASYNC) == -1) {
	    perror("ERROR: failed to sync before unmap");
	  }

    size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

	  return munmap(addr, len + offset - pa_offset);
	}


// SERIALIZER for kmer and count.
struct KVSerializer {
	template <typename KT = KmerType, typename VT = CountType>
	inline unsigned char* operator()(KT const & k, VT const & c, unsigned char* out)  const {
		memcpy(out, k.getData(), sizeof(KT));  
		memcpy(out + sizeof(KT), &c, sizeof(VT));  

		return out + sizeof(KT) + sizeof(VT);
	}

	template <typename KT = KmerType, typename VT = CountType,
		typename std::enable_if<((sizeof(KT) + sizeof(VT)) == sizeof(std::pair<KT, VT>)), int>::type = 1>
	inline unsigned char* operator()(::std::pair<KT, VT> const & kv, unsigned char* out) const {
		memcpy(out, &kv, sizeof(std::pair<KT, VT>));

		return out + sizeof(std::pair<KT, VT>);
	}

	template <typename KT = KmerType, typename VT = CountType,
		typename std::enable_if<((sizeof(KT) + sizeof(VT)) != sizeof(std::pair<KT, VT>)), int>::type = 1>
	inline unsigned char* operator()(::std::pair<KT, VT> const & kv, unsigned char* out) const {
		return this->operator()(kv.first, kv.second, out);
	}

};



// return bytes copied.
template <typename C>
size_t copyToByteArray(C const & map, unsigned char * dest) {
// assume that dest is large enough...
	KVSerializer kvs;

	#if (pMAP == RADIXSORT)  // serialize.  since some entries are empty, and we don't have a usable iterator.
		return map.get_local_container().serialize(dest, kvs);
	#elif (pMAP == MTRADIXSORT) || (pMAP == MTROBINHOOD)
		return map.serialize(dest, kvs);
	#elif (pMAP == SORTED)

		size_t out_elem_size = sizeof(KmerType) + sizeof(CountType);
		if (out_elem_size == sizeof(typename IndexType::TupleType)) {
			// same size, so just copy directly.
			memcpy(dest, map.get_local_container().data(), map.get_local_container().size() * out_elem_type);

		} else {
			unsigned char * cdest = dest;
			// copy into temporary storage.
			auto it_end = map.get_local_container().cend();
			
			for (auto it = map.get_local_container().cbegin(); it != it_end; ++it) {
				cdest = kvs(it->first, it->second, cdest);
			}
			
		}
		return map.get_local_container().size() * out_elem_size;

	#else  // copy one by one because iterator may be filtering or concatenating.

	
		size_t out_elem_size = sizeof(KmerType) + sizeof(CountType);
		unsigned char * cdest = dest;
		// copy into temporary storage.
		auto it_end = map.get_local_container().cend();
		
		for (auto it = map.get_local_container().cbegin(); it != it_end; ++it) {
			cdest = kvs(*it, cdest);
		}
		return map.get_local_container().size() * out_elem_size;
	#endif
}


template <bool direct = false, size_t CHUNK_SIZE = (1UL << 30U), typename C>
size_t writeToPOSIX(C const & map, std::string const & out_filename) {

	static_assert((CHUNK_SIZE <= (1 << 30)), "CHUNK_SIZE is limited to 1GB due to linux limitations.");

	size_t out_elem_size = sizeof(KmerType) + sizeof(typename IndexType::ValueType);
	KVSerializer kvs;

	size_t written = 0; 
    // std::cout << "rank " << comm.rank() << " out file name " << out_filename << " target size " << target_size << std::endl;
#if (pMAP == SORTED)   // COPY and compact first to reduce space.
	if ((sizeof(KmerType) + sizeof(CountType)) == sizeof(typename IndexType::TupleType)) {
		written = map.get_local_container().size() * sizeof(typename IndexType::TupleType);
		// may not be able to write directly because of lack of 512 byte alignment.
		write_posix(out_filename, reinterpret_cast<unsigned char*>(map.get_local_container().data()), written);
	} else {  // not same length data types, so copy first.
		unsigned char * values = 
			::utils::mem::aligned_alloc<unsigned char>(map.get_local_container().size() * out_elem_size, 512);
		written = copyToByteArray(map.get_local_container(), values);
		if (direct)
			write_posix_direct(out_filename, values, written, 512);
		else 
			write_posix(out_filename, values, written);
	  ::utils::mem::aligned_free(values);
	}

#elif (pMAP == RADIXSORT)   // copy the entire dataset and then write.
	unsigned char* values = 
		::utils::mem::aligned_alloc<unsigned char>(map.get_local_container().size() * out_elem_size, 512);
  written = map.get_local_container().serialize(values, kvs);
	if (direct)
	  write_posix_direct(out_filename, values, written , 512);
	else
	  write_posix(out_filename, values, written);
  ::utils::mem::aligned_free(values);

#elif (pMAP == MTRADIXSORT) || (pMAP == MTROBINHOOD)  // copy the entire dataset and then write.
	unsigned char* values = 
		::utils::mem::aligned_alloc<unsigned char>(map.local_size() * out_elem_size, 512);
  written = map.serialize(values, kvs);
	if (direct)
	  write_posix_direct(out_filename, values, written , 512);
	else
	  write_posix(out_filename, values, written);
  ::utils::mem::aligned_free(values);

#else // write 1 batch at a time

	// assume 32KB, 8 way set-associative cache.
	// there are 64 sets of 8x64byte cachelines.
	// use 1/4 of it. = 2x64x64 bytes = 8KB.
	typename MapType::local_container_type const & container = map.get_local_container();
  //auto container = map.get_local_container();

	// compute some step sizes.  then 8192 - (8192 % lcm)
		// get lowest common multiple of 512 and tuple type size
		size_t step512 = lcm(512UL, out_elem_size);
		size_t step_bytes = std::max(CHUNK_SIZE, step512);
		step_bytes = step_bytes - (step_bytes % step512);  // user specified.
		size_t container_bytes = container.size() * out_elem_size;
		container_bytes = ((container_bytes + step512 - 1) / step512) * step512;  // next multiple of step512.
		step_bytes = std::min(step_bytes, container_bytes); // at most CHUNK_SIZE, and exact multiple of 512.
    size_t step = step_bytes / out_elem_size;  // number of elements - always divisible by sizeof(TupleType)
		
		std::cout << "ELEM SIZE " << out_elem_size << " CHUNK_SIZE " << CHUNK_SIZE << " lcm " << step512 << " total bytes " << step_bytes << " step " << step << " container size " << container.size() <<  std::endl;

	// open file.
		int fd = open_out_file(out_filename, direct);
		if (fd == -1) {
			throw std::logic_error("ERROR: read_range: file pointer is null");
		}

	// allocate temp buffer , allocate a multiple of 512.
		unsigned char *p = 
			::utils::mem::aligned_alloc<unsigned char>(step_bytes , 512);  // align at 512

		std::cout << "posix aligned allocation at 512 boundary, address " << std::hex << reinterpret_cast<size_t>(p) << std::dec << std::endl;
		// typename IndexType::TupleType *q = reinterpret_cast<typename IndexType::TupleType *>(p);
		unsigned char *q = p;
	// set up loop
		written = 0;
	  ssize_t iter_written;

		size_t i = 0, imax = (container.size() > step) ? (container.size() - step) : 0;
		size_t j = 0, jmax;
		auto it = container.cbegin();

		// iterate over loop
	  for ( i = 0; i < imax; i += step) {
			// load data into temp buffer.
			for ( j = i, jmax = i + step, q = p; j < jmax; ++j, ++it) {
        // memcopy the kmer
				q = kvs(*it, q);
      }

			// next write to file
			iter_written = write(fd, p, step_bytes);

			if (iter_written < 0) {
				std::cout << "ERROR: " << errno << std::endl;
				perror("write");
				throw std::logic_error("ERROR: write failed");
			} else if (iter_written < static_cast<ssize_t>(step_bytes)) {
				std::cout << "ERROR: write " << iter_written << " specified " << step_bytes << std::endl;
				throw std::logic_error("ERROR: written less than specified");
			}

			written += iter_written;
		}

	  if (i < container.size()) {
			// load data into temp buffer.
			for ( j = i, jmax = container.size(), q = p; j < jmax; ++j, ++it) {
        // memcopy the kmer
				q = kvs(*it, q);
      }
			size_t remainder = (container.size() - i) * out_elem_size;
			std::cout << "remainder bytes " << remainder << " i " << i << " container " << container.size() << std::endl;

			if (direct) {  // if direct, then need to zero some entries.
				remainder = ((remainder + 511) >> 9) << 9;

				memset(q, 0, remainder - (container.size() - i) * out_elem_size);  // from q to end.
			}

			// next write to file, only write as much as needed.
			iter_written = write(fd, p, remainder );

			if (iter_written < 0) {
				std::cout << "ERROR: " << errno << std::endl;
				perror("write");
				throw std::logic_error("ERROR: write failed");
			} else if (iter_written < static_cast<ssize_t>(remainder)) {
				std::cout << "ERROR: write " << iter_written << " specified " << remainder << std::endl;
				throw std::logic_error("ERROR: written less than specified");
			}

			written += iter_written;

		}

		// close the file when done.
		close_out_file(fd);
		::utils::mem::aligned_free(p);

#endif
		return written;
}


	inline unsigned long get_free_mem() {
		struct sysinfo memInfo;
		sysinfo(&memInfo);

		return memInfo.freeram * memInfo.mem_unit;
	}


	inline unsigned long get_free_mem_per_proc(const mxx::comm & comm) {
		unsigned long free_mem = get_free_mem();
		int local_procs = 1;
		{
			mxx::comm shared_comm = comm.split_shared();
			local_procs = shared_comm.size();
		}
		if (local_procs < 1) throw std::logic_error("evaluated number of local processes on same node to be 0 or less.");
		if (free_mem < 1) throw std::logic_error("evaluated free memory on the node to be 0 or less.");

		size_t mem_per_p = (free_mem / local_procs);  // number of elements that can be held in freemem
		// find the minimum across all processors.

    size_t mem_per_p_mean = mxx::allreduce(mem_per_p, std::plus<size_t>(), comm);
    mem_per_p_mean /= comm.size();
		mem_per_p = mxx::allreduce(mem_per_p, mxx::min<size_t>(), comm);
		if (comm.rank() == 0) std::cout << "estimate available mem on node=" << free_mem << " bytes, p=" << local_procs <<
				", per proc min=" << mem_per_p << " bytes, mean=" << mem_per_p_mean << " bytes." << std::endl;

		return mem_per_p;
	}

	// overestimates the number of kmers in files.
	size_t estimate_kmers_in_file_per_proc(std::vector<std::string> const & filenames, float const & chars_per_kmer, mxx::comm const & comm) {
	  size_t total_file_size = 0;

	  for (size_t i = comm.rank(); i < filenames.size(); i+= comm.size()) {
	    total_file_size += (get_file_size(filenames[i]) + comm.size() - 1) / comm.size();
	  }
	  total_file_size = mxx::allreduce(total_file_size, comm);

	  std::cout << "ESTIMATE: per proc file size = " << total_file_size << " kmers " << (static_cast<float>(total_file_size) / chars_per_kmer) << std::endl;

	  return static_cast<size_t>(static_cast<float>(total_file_size) / chars_per_kmer);
	}


/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv) {

  //////////////// init logging
  LOG_INIT();


  //////////////// initialize MPI and openMP

  mxx::env e(argc, argv);
  mxx::comm comm;

  if (comm.rank() == 0) printf("EXECUTING %s\n", argv[0]);

  comm.barrier();




	//////////////// parse parameters
	std::vector<std::string> filenames;
	std::string filename;

	filename.assign(PROJ_SRC_DIR);
	#if (pPARSER == FASTA)
	      filename.append("/test/data/test.fasta");
	#elif (pPARSER == FASTQ)
	      filename.append("/test/data/test.fastq");
	#endif

	std::string out_filename;
	out_filename.assign("./counts.bin");

//#if (pPARSER == FASTQ)
	// CountType lower;
	// CountType upper;
//#endif

//	bool thresholding = false;
//	bool benchmark = false;

	int reader_algo = -1;
  int writer_algo = -1;

	//  std::string queryname(filename);
	//  int sample_ratio = 100;

	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {

		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.
		TCLAP::CmdLine cmd("Parallel Kmer Counting", ' ', "0.4");

		// MPI friendly commandline output.
		::bliss::utils::tclap::MPIOutput cmd_output(comm);
		cmd.setOutput(&cmd_output);

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<std::string> outputArg("O", "out_filename", "output filename", false, out_filename, "string", cmd);

//		TCLAP::SwitchArg threshArg("T", "thresholding", "on/off for thresholding", cmd, false);
// 		TCLAP::ValueArg<CountType> lowerThreshArg("L", "lower_thresh", "Lower Threshold for Kmer and Edge frequency", false, 0, "uint16", cmd);
//		TCLAP::ValueArg<CountType> upperThreshArg("U", "upper_thresh", "Upper Threshold for Kmer and Edge frequency", false,
//				std::numeric_limits<CountType>::max(), "uint16", cmd);

//		TCLAP::SwitchArg benchmarkArg("B", "benchmark", "on/off for benchmarking (no file output)", cmd, false);

	    TCLAP::ValueArg<int> algoArg("A",
	                                 "algo", "Reader Algorithm id. mmap = 5, posix=7, mpiio = 10. default is 7.",
	                                 false, 7, "int", cmd);

	    // output algo 7 and 8 are not working.
      TCLAP::ValueArg<int> outAlgoArg("B",
                                   "output_algo", "Writer Algorithm id. mmap_1file_1=1, mmap_1=2, mmap_all=3, mmap_1file_all=4, posix_1=5, posix_all=6, posix_direct_1=7, posix_direct_all=8, mpiio=10. no_output=0.  default is 0.",
                                   false, 0, "int", cmd);

		TCLAP::UnlabeledMultiArg<std::string> fileArg("filenames", "FASTA or FASTQ file names", false, "string", cmd);


		// Parse the argv array.
		cmd.parse( argc, argv );

		filenames = fileArg.getValue();
		out_filename = outputArg.getValue();

		if (filenames.size() == 0) {
			filenames.push_back(filename);
		}

//#if (pPARSER == FASTQ)
//   lower = lowerThreshArg.getValue();
//  upper = upperThreshArg.getValue();
//#endif

//		thresholding = threshArg.getValue();
//		benchmark = benchmarkArg.getValue();
  reader_algo = algoArg.getValue();
  writer_algo = outAlgoArg.getValue();


	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		exit(-1);
	}


  // ================  set up index.

  IndexType idx(comm);

#if (pMAP == DENSEHASH)
//  KmerType empty_key = ::bliss::kmer::hash::sparsehash::empty_key<KmerType>::generate();
//  KmerType deleted_key = ::bliss::kmer::hash::sparsehash::deleted_key<KmerType>::generate();
//
//  	idx.get_map().reserve_keys(empty_key, deleted_key);
//
//  	// upper key is negation of lower keys
//  	KmerType upper_empty_key = empty_key;
//  	KmerType upper_deleted_key = deleted_key;
//  	for (size_t i = 0; i < KmerType::nWords; ++i) {
//  		upper_empty_key.getDataRef()[i] = ~(upper_empty_key.getDataRef()[i]);
//  		upper_deleted_key.getDataRef()[i] = ~(upper_deleted_key.getDataRef()[i]);
//  	}
//
//  	idx.get_map().reserve_upper_keys(upper_empty_key, upper_deleted_key, Splitter());
//

#endif

  BL_BENCH_INIT(test);

  BL_BENCH_START(test);

  // =======  set up to estimate and determine, up to free mem, the total size that can be processed.
  //   note that we can use TupleType size, which is always greater or equal to parser's output type.
  // get estimated number of k-mers in file, using a crude chars_per_kmer estimate.  refine later.
  float chars_per_kmer = 1.0;
#if (pPARSER == FASTQ)
      chars_per_kmer *= 2.0;
#endif
#if (pKmerStore == CANONICAL)
      chars_per_kmer *= 2.0;
#endif

  float distinct_ratio = 0.125; //1.0;
  size_t delta_distinct = 0;
  size_t max_load = 0;
  size_t buckets = 0;
#if (pMAP == SORTED)
#else
  size_t orig_buckets = 0;
#endif

  // -----  for all iterations
  size_t total_file_size = 0;
  size_t kmer_total = 0;   // total number of kmers generated.
  size_t iters = 0;

  // -----  for last iteration
  size_t avg_distinct_count = 0;   // last global index size.
  size_t avg_distinct_before = 0;
  size_t kmer_iter_total = 0;   // total number of kmers generated in between inserts.
  size_t kmer_iter_est = 0;   // total number of kmers generated in between inserts.


  // ========= get the initial estimate of maximum number kmers for free memory.  refine later.
  size_t free_mem = 0;
  size_t usable_mem = 0;
  size_t mem_use_est = 0;

  // ====== collect some statistics.


  BL_BENCH_END(test, "init", filenames.size());



  { // scoped to clear temp

    BL_BENCH_LOOP_START(test, 0);  // estimate
    BL_BENCH_LOOP_START(test, 1);  // reserve
    BL_BENCH_LOOP_START(test, 2);  // read
    BL_BENCH_LOOP_START(test, 3);  // grow
    BL_BENCH_LOOP_START(test, 4);  // insert
    BL_BENCH_LOOP_START(test, 5);  // measure


	  // storage.
	  using kmer_vec_type = ::std::vector<typename IndexType::KmerParserType::value_type>;
	  kmer_vec_type temp;

	  size_t file_size = 0;
	  size_t iter_file_size = 0;
	  size_t i = 0, j = 0;
	  bool iter_done = false;
	  bool too_much = false;
	  bool iter_done_all = false;
	  size_t kmer_est = 0;

	  if (comm.rank() == 0) std::cout << "filename count " << filenames.size() << std::endl;

	  for (; i < filenames.size();) {

      BL_BENCH_LOOP_RESUME(test, 0);
      iter_file_size += file_size;


      // =========== update the parameters
#if (pMAP == SORTED) || (pMAP == RADIXSORT) || (pMAP == MTRADIXSORT) 
    buckets = idx.get_map().local_capacity();
	orig_buckets = buckets;
      max_load = buckets;
#elif (pMAP == MTROBINHOOD)
	    buckets = idx.get_map().local_capacity();
	    orig_buckets = buckets;
      max_load = idx.get_map().get_max_load_factor() * buckets;
#else 
	    buckets = idx.get_map().local_capacity();
	    orig_buckets = buckets;
      max_load = idx.get_map().get_local_container().get_max_load_factor() * buckets;
#endif
      free_mem = get_free_mem_per_proc(comm); // - total_file_size;  // free memory usage after last insert.
      usable_mem = free_mem - std::min((1UL << 33U) / comm.size(), free_mem / 4UL);   // reserve either 1/4 or 8GB of GLOBAL TOTAL.  (to deal with 10 GB files?)

      file_size = 0;
      iter_file_size = 0;
      iter_done = false;
      too_much = false;
      kmer_iter_est = 0;
      mem_use_est = 0;
      delta_distinct = 0;

	  // check files to see how much space we can use.
	  for (j = i; j < filenames.size(); ++j) {

        // rank 0 get file size info and broadcast.
        if (comm.rank() == 0) {
          file_size = (get_file_size(filenames[j]) + comm.size() - 1) / comm.size();
        }
        MPI_Bcast(&file_size, 1, MPI_UNSIGNED_LONG, 0, comm);

        // estimate kmer_iter_total before actually reading.
        kmer_est = static_cast<size_t>(static_cast<float>(file_size) / chars_per_kmer);

        // estimate free memory usage.
#if (pMAP == BROBINHOOD) || (pMAP == MTROBINHOOD) || (pMAP == RADIXSORT) || (pMAP == MTRADIXSORT)
        mem_use_est += kmer_est * 5UL * sizeof(typename IndexType::KmerParserType::value_type) +
            file_size * 2UL;
#else
        mem_use_est += kmer_est * 3UL * sizeof(typename IndexType::KmerParserType::value_type) +
            file_size * 2UL;
#endif
        // expected change in hash table size
        delta_distinct += static_cast<size_t>(distinct_ratio * static_cast<float>(kmer_est));
#if (pMAP == SORTED)
#else
        while ((avg_distinct_before + delta_distinct) >= max_load) {
          // will need to resize.
          max_load *= 2;
          buckets *= 2;
        }
#endif

	    if (comm.rank() == 0) {
	        std::cout <<
	            " SYSTEM STATS free_mem " << free_mem <<
				" usable mem " << usable_mem <<
	            " mem_use_est " << mem_use_est <<
	            std::endl;
	        std::cout <<
	            " FILE STAT file_size " << file_size <<
	            " iter_file_size " << iter_file_size <<
	            " kmer_est " << kmer_est <<
				" kmer_iter_est " << kmer_iter_est <<
				" delta_distinct " << delta_distinct <<
				" avg_distinct_before " << avg_distinct_before <<
	            " max_load " << max_load <<
	            " buckets " << buckets <<
	            std::endl;
	        std::cout <<
	            " INDEX CONFIG " <<
	            " kmer size " << sizeof(KmerType) <<
	            " input tuple size " << sizeof(typename IndexType::KmerParserType::value_type) <<
	            " tuple size " << sizeof(typename IndexType::TupleType) <<
	            std::endl;
	      }

        kmer_iter_est += kmer_est;
        iter_file_size += file_size;
#if (pMAP == SORTED)
        mem_use_est += idx.local_size();
#else
	    if (buckets > orig_buckets) {
	    	if (comm.rank() == 0) std::cout << " SCALING UP from " << orig_buckets << " to " << buckets << ", mem from " << mem_use_est << " to ";
	    	mem_use_est += buckets * sizeof(typename IndexType::TupleType);
	    	if (comm.rank() == 0) std::cout << mem_use_est << std::endl;
	    }
#endif
        // if too much memory, back off it - save j for next run.
        iter_done = mem_use_est > usable_mem;
        too_much = mem_use_est > free_mem;
//        if (iter_done) std::cout << " iter done on rank " << comm.rank() << std::endl;

        iter_done_all = mxx::any_of(too_much, comm);
        if (iter_done_all) {
          break;
        }

        iter_done_all = mxx::any_of(iter_done, comm);
        if (iter_done_all) {
        	++j;
        	break;
        }

        // update the file size for this iteration.
	  }

	  // check if 1 file is too much.  without this check, we have an infinite loop.
	  if (i == j) {
	    std::cerr << "WARNING: estimates indicate that there is not enough room to insert and grow table.  continuing but may fail or be very slow: i = j = " << i << std::endl;
	    j = std::min(j + 1, filenames.size());
	  }
      BL_BENCH_LOOP_PAUSE(test, 0);

	    // now prepare to insert.  first reserve (using the estimate distinct count
    BL_BENCH_LOOP_RESUME(test, 3);
    if (comm.rank() == 0) std::cout << " resizing. iter " << iters << " from " << avg_distinct_count << " to ";
    // estimate the number of distinct new entries using previous distinct ratio
    delta_distinct = static_cast<size_t>(distinct_ratio * static_cast<float>(kmer_iter_est));
    if (comm.rank() == 0) std::cout << (delta_distinct + avg_distinct_count);

#if (pMAP == SORTED)
	    idx.get_map().local_reserve(idx.local_size() + kmer_est);
	    if (comm.rank() == 0) std::cout << " buckets " << idx.get_map().local_capacity() << std::endl;
#elif (pMAP == RADIXSORT) || (pMAP == MTRADIXSORT)
//	    if (idx.get_map().local_capacity() < (avg_distinct_count + delta_distinct))
//	    	idx.get_map().local_reserve(avg_distinct_count + delta_distinct);
	    // do nothing, since on insertion we reserve.
	    if (comm.rank() == 0) std::cout << " buckets " << idx.get_map().local_capacity() << std::endl;
#elif (pMAP == BROBINHOOD) || (pMAP == MTROBINHOOD)

#else
	    if ((idx.get_map().local_capacity() * idx.get_map().get_local_container().get_max_load_factor()) < (avg_distinct_count + delta_distinct))
	    	idx.get_map().local_reserve(avg_distinct_count + delta_distinct);
	    if (comm.rank() == 0) std::cout << " buckets " << idx.get_map().local_capacity() << std::endl;
#endif
    BL_BENCH_LOOP_PAUSE(test, 3);



      BL_BENCH_LOOP_RESUME(test, 1);

	    // now reserve space.
	    {
	      kmer_vec_type().swap(temp);   // clear and destroy, so when we resize we are starting from scratch.
	    }
	    temp.reserve( static_cast<size_t>(static_cast<float>(iter_file_size) / chars_per_kmer) );   // initially, (empty index), assume (3 * size(kmer) + 2 + size(count)) * N

	    BL_BENCH_LOOP_PAUSE(test, 1);

	    if (comm.rank() == 0) {
	        std::cout <<
	            " FILE STAT chars_per_kmer " << chars_per_kmer <<
	            " avg iter file size " << iter_file_size <<
				" temp reserved " << temp.capacity() <<
	            std::endl;
	        std::cout <<
	            " INDEX CONFIG " <<
	            " kmer size " << sizeof(KmerType) <<
	            " input tuple size " << sizeof(typename IndexType::KmerParserType::value_type) <<
	            " tuple size " << sizeof(typename IndexType::TupleType) <<
	            std::endl;
	      }

      BL_BENCH_LOOP_RESUME(test, 2);

	    // ========== read from files i to j.
	    for (; i < j; ++i) {

	      // ---- now read each file
  	  if (reader_algo == 5) {
  		if (comm.rank() == 0) printf("reading %s via mmap\n", filenames[i].c_str());
  		::bliss::io::KmerFileHelper::read_file_mmap<typename IndexType::KmerParserType, PARSER_TYPE, bliss::io::NSplitSequencesIterator>(filenames[i], temp, comm);

  	  } else if (reader_algo == 7) {
  		if (comm.rank() == 0) printf("reading %s via posix\n", filenames[i].c_str());
  		::bliss::io::KmerFileHelper::read_file_posix<typename IndexType::KmerParserType, PARSER_TYPE, bliss::io::NSplitSequencesIterator>(filenames[i], temp, comm);

  	  } else if (reader_algo == 10){
  		if (comm.rank() == 0) printf("reading %s via mpiio\n", filenames[i].c_str());
  		::bliss::io::KmerFileHelper::read_file_mpiio<typename IndexType::KmerParserType, PARSER_TYPE, bliss::io::NSplitSequencesIterator>(filenames[i], temp, comm);
  	  } else {
  		throw std::invalid_argument("missing file reader type");
  	  }

	    }
      ::mxx::distribute_inplace(temp, comm);

      BL_BENCH_LOOP_PAUSE(test, 2);

      if (comm.rank() == 0) std::cout << " DONE READING iter " << iters << std::endl;

      BL_BENCH_LOOP_RESUME(test, 5);

	    // get the actual total
	    kmer_iter_total = temp.size();
	    total_file_size += iter_file_size;
	    kmer_total += kmer_iter_total;

	    // get some averages.
	    kmer_iter_total = mxx::allreduce(kmer_iter_total, comm) / comm.size();
	    iter_file_size = mxx::allreduce(iter_file_size, comm) / comm.size();

	    size_t global_kmer_total = mxx::allreduce(kmer_total, comm);
	    size_t global_file_total = mxx::allreduce(total_file_size, comm);

	    // and compute number of characters per kmer as an iteration average.
	    chars_per_kmer = static_cast<float>(global_file_total) / static_cast<float>(global_kmer_total);

      BL_BENCH_LOOP_PAUSE(test, 5);


	    // now insert.
      BL_BENCH_LOOP_RESUME(test, 4);
#if (pMAP == RADIXSORT) || (pMAP == MTRADIXSORT)
      // don't estimate...
      //idx.get_map().insert_no_finalize<false>(temp);  // should be insert_no_finalize but just to be safe don't do it right now...
      idx.get_map().insert_no_finalize<true>(temp);  // should be insert_no_finalize but just to be safe don't do it right now...
#elif (pMAP == BROBINHOOD)  || (pMAP == MTROBINHOOD)
      // don't estimate...
	    //idx.get_map().insert<false>(temp);
	    idx.get_map().insert<true>(temp);
#else
	    idx.insert(temp);
#endif
	    BL_BENCH_LOOP_PAUSE(test, 4);


      BL_BENCH_LOOP_RESUME(test, 5);

      ++iters;

      // global distinct kmer count after last insert.
      avg_distinct_before = avg_distinct_count;
      avg_distinct_count = idx.size() / comm.size();   // last global index size.

      // save this for next iteration
      distinct_ratio = static_cast<float>(avg_distinct_count - avg_distinct_before) / static_cast<float>(kmer_iter_total);

		  BL_BENCH_LOOP_PAUSE(test, 5);


      if (comm.rank() == 0) {
        std::cout <<
            " SYSTEM STATS free_mem " << free_mem <<
			" usable mem " << usable_mem <<
            " mem_use_est " << mem_use_est <<
            std::endl;
        std::cout <<
            " FILE STAT chars_per_kmer " << chars_per_kmer <<
			" avg distinct " << avg_distinct_count <<
			" avg distinct b4 " << avg_distinct_before <<
			" kmer iter total " << kmer_iter_total <<
            " distinct_ratio " << distinct_ratio <<
            " delta_distinct " << delta_distinct <<
            " avg iter file size " << iter_file_size <<
            " total file size " << total_file_size <<
            std::endl;
        std::cout <<
            " INDEX CONFIG " <<
            " kmer size " << sizeof(KmerType) <<
            " input tuple size " << sizeof(typename IndexType::KmerParserType::value_type) <<
            " tuple size " << sizeof(typename IndexType::TupleType) <<
            std::endl;
        std::cout <<
            " INSERT STATS kmer total " << kmer_total <<
            " iters " << iters <<
            " pre-insert kmer count est " << kmer_iter_est <<
            " avg kmer count " << kmer_iter_total <<
            " prev distinct " << avg_distinct_before <<
            " distinct " << avg_distinct_count <<
            " max load before " << max_load << " after " <<
#if (pMAP == SORTED) || (pMAP == RADIXSORT) || (pMAP == MTRADIXSORT)
            static_cast<size_t>(idx.get_map().local_capacity()) <<
            " buckets " << buckets << " after " << idx.get_map().local_capacity() <<
#elif (pMAP == MTROBINHOOD)
            static_cast<size_t>(idx.get_map().local_capacity() * idx.get_map().get_max_load_factor()) <<
            " buckets " << buckets << " after " << idx.get_map().local_capacity() <<
#else
            static_cast<size_t>(idx.get_map().local_capacity() * idx.get_map().get_local_container().get_max_load_factor()) <<
            " buckets " << buckets << " after " << idx.get_map().local_capacity() <<
#endif
            std::endl;
      }

    } // filename loop.

#if (pMAP == RADIXSORT) 
	idx.get_map().get_local_container().finalize_insert();
#elif (pMAP == MTRADIXSORT)
	idx.get_map().finalize_insert();
#endif
  } // scoped to clear temp.



  comm.barrier();

  BL_BENCH_LOOP_END(test, 0, "estimate", total_file_size);
  BL_BENCH_LOOP_END(test, 1, "reserve", iters );
  BL_BENCH_LOOP_END(test, 2, "read", kmer_total);
  BL_BENCH_LOOP_END(test, 3, "resize", idx.get_map().local_capacity() );
  BL_BENCH_LOOP_END(test, 4, "insert", idx.get_map().local_size() );
  BL_BENCH_LOOP_END(test, 5, "measure", chars_per_kmer);



  // erase if needed.
  // TODO: [ ] CURRENTLY NOT SUPPORTED BY ROBINHOOD OR RADIXSORT
//  if (lower > 1)	{
//	  BL_BENCH_START(test);
//	  idx.erase_if([&lower](typename IndexType::TupleType const & v){
//		  return v.second < lower;
//	  });
//	  BL_BENCH_COLLECTIVE_END(test, "threshold", idx.local_size(), comm);
//  }

//===== IO!
		size_t out_elem_size = sizeof(KmerType) + sizeof(typename IndexType::ValueType);
		KVSerializer kvs;

		size_t target_size = idx.local_size() * out_elem_size; 


	// total size.
	size_t total = mxx::allreduce(target_size, comm);

	// get the maximum file size.
	rlimit rl; 
	getrlimit(RLIMIT_FSIZE, &rl);
	size_t max_fsize = rl.rlim_cur;
	



  if (writer_algo == 10) {  // mpiio
		if (comm.rank() == 0) std::cout << "write using MPI-IO, 1 file, all at the same time" << std::endl;
		if (total > max_fsize)  throw std::logic_error("target file size larger than supported");


	  BL_BENCH_START(test);


		unsigned char* values;
			// std::cout << "rank " << comm.rank() << " out file name " << out_filename << " target size " << target_size << std::endl;
		#if (pMAP == SORTED)   // COPY and compact first to reduce space.
			
			if (out_elem_size == sizeof(typename IndexType::TupleType)) {
				values = reinterpret_cast<unsigned char*>(idx.get_map().get_local_container().data());
				write_mpiio(out_filename, values, target_size, comm);
			} else {  // not same length data types, so copy first.
				values = 
					::utils::mem::aligned_alloc<unsigned char>(target_size, 512);
				target_size = copyToByteArray(idx.get_map(), values);
				write_mpiio(out_filename, values, target_size, comm);
				::utils::mem::aligned_free(values);
			}

		#else

			values = 
				::utils::mem::aligned_alloc<unsigned char>(target_size, 512);

			#if (pMAP == RADIXSORT) 
				target_size = idx.get_map().get_local_container().serialize(values, kvs);
			#elif (pMAP == MTRADIXSORT) || (pMAP == MTROBINHOOD)
				target_size = idx.get_map().serialize(values, kvs);

			#else
				unsigned char* q = values;
				auto it_end = idx.get_map().get_local_container().cend();
				
				for (auto it = idx.get_map().get_local_container().cbegin(); it != it_end; ++it) {
					q = kvs(*it, q);
				}
			#endif

			write_mpiio(out_filename, values, target_size, comm);
			::utils::mem::aligned_free(values);
		#endif

		BL_BENCH_COLLECTIVE_END(test, "write_10", target_size, comm);


  } else if ((writer_algo == 5) || (writer_algo == 7)) {
		if (writer_algo == 5) {
			if (comm.rank() == 0) std::cout << "write using posix with buffering, 1 file per core, 1 core per node at once.  DO NOT USE." << std::endl;
		} else {
			if (comm.rank() == 0) std::cout << "write using posix direct, 1 file per core, 1 core per node at once.  DO NOT USE." << std::endl;
		}

		if (target_size > max_fsize)  throw std::logic_error("target file size larger than supported");


    BL_BENCH_START(test);

    out_filename.append(std::to_string(comm.rank()));
    // std::cout << "rank " << comm.rank() << " out file name " << out_filename << " target size " << target_size << std::endl;


    mxx::comm group = comm.split_shared();  // split comm into groups that share memory, i.e. by node.
    int max_group_size = group.size();  // size of each group.
    //max_group_size = mxx::allreduce(max_group_size, mxx::max<int>(), comm);  // get the biggest group size.


		size_t copied = 0;
		
    for (int i = 0; i < max_group_size; ++i) {

      if (i == group.rank()) {
				// writing
				if (writer_algo == 5)
					copied = writeToPOSIX<false, (1UL << 30U)>(idx.get_map(), out_filename);
				else
					copied = writeToPOSIX<true, (1UL << 30U)>(idx.get_map(), out_filename);
      }
			group.barrier();  // use barrier to force one from a node at a time.

    }


    BL_BENCH_COLLECTIVE_END(test, ((writer_algo == 5) ? "write_5" : "write_7"), copied, comm);

  } else if ((writer_algo == 6) || (writer_algo == 8)) {
		if (writer_algo == 6) {
			if (comm.rank() == 0) std::cout << "write using posix with buffering, 1 file per core, all cores at once. For MODERATE CORE COUNT and sufficient MEM." << std::endl;
		} else {
			if (comm.rank() == 0) std::cout << "write using posix direct, 1 file per core, all cores at once. For MODERATE CORE COUNT and sufficient MEM." << std::endl;
		}
		if (target_size > max_fsize)  throw std::logic_error("target file size larger than supported");
    
    BL_BENCH_START(test);

    out_filename.append(std::to_string(comm.rank()));

		size_t copied = 0;
		if (writer_algo == 6)
			copied = writeToPOSIX<false, (1UL << 30U)>(idx.get_map(), out_filename);
		else {
			copied = writeToPOSIX<true, (1UL << 30U)>(idx.get_map(), out_filename);
		}
    BL_BENCH_COLLECTIVE_END(test, ((writer_algo == 6) ? "write_6" : "write_8"), copied, comm);


  } else if (writer_algo == 2) {  // mmap_1.  one core from each node writes at a time..

		if (comm.rank() == 0) std::cout << "write using mmap: NOT FOR GPFS! 1 file, 1 core/node at a time. DO NOT USE." << std::endl;
		if (total > max_fsize)  throw std::logic_error("target file size larger than supported");

    BL_BENCH_START(test);

    mxx::comm group = comm.split_shared();  // split comm into groups that share memory, i.e. by node.
    int max_group_size = group.size();  // size of each group.
    //max_group_size = mxx::allreduce(max_group_size, mxx::max<int>(), comm);  // get the biggest group size.

		// target offset in file
    size_t offset = mxx::exscan(target_size, comm);
		// page aligned offset in file. (smaller than offset)
    size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

		// p is aligned to pa_offset.
		unsigned char *p = map_out_file(out_filename, target_size, offset, comm);

		// q is aligned to offset.
		unsigned char *q = p + (offset - pa_offset);


		size_t copied = 0;
    for (int i = 0; i < max_group_size; ++i) {

      if (i == group.rank()) {
				// copy data into array (writing)
				copied = copyToByteArray(idx.get_map(), q);
      }
			group.barrier();  // use barrier to force one from a node at a time.
    }

		// unmap the file
		unmap_out_file(p, target_size, offset, comm);

    BL_BENCH_COLLECTIVE_END(test, "write_2", copied, comm);

  } else if (writer_algo == 3) {  // mmap_all.  all writing at once.
		if (comm.rank() == 0) std::cout << "write using mmap: NOT FOR GPFS! 1 file, all cores concurrently: for SHARED MEM with local file system." << std::endl;
		if (total > max_fsize)  throw std::logic_error("target file size larger than supported");

    BL_BENCH_START(test);

		// target offset in file
    size_t offset = mxx::exscan(target_size, comm);
		// page aligned offset in file. (smaller than offset)
    size_t pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);

		// p is aligned to pa_offset.
    unsigned char *p = map_out_file(out_filename, target_size, offset, comm);

		// q is aligned to offset.
    unsigned char *q = p + (offset - pa_offset);

		// copy data into array
		size_t copied = copyToByteArray(idx.get_map(), q);

		// unmap the file
    unmap_out_file(p, target_size, offset, comm);

		// barrier for all.
    BL_BENCH_COLLECTIVE_END(test, "write_3", copied, comm);

  } else if (writer_algo == 4) {  // mmap_all.  all writing at once.
		if (comm.rank() == 0) std::cout << "write using mmap: NOT FOR GPFS! 1 file per core, all at once: for smaller COMM SIZE" << std::endl;
		if (target_size > max_fsize)  throw std::logic_error("target file size larger than supported");


    BL_BENCH_START(test);

    out_filename.append(std::to_string(comm.rank()));

		// q starts at offset 0.
    unsigned char *q = map_out_file(out_filename, target_size, 0);

		// copy data into array
		size_t copied = copyToByteArray(idx.get_map(), q);

		// unmap the file
    unmap_out_file(q, target_size, 0);

		// barrier for all.
    BL_BENCH_COLLECTIVE_END(test, "write_4", copied, comm);

  } else if (writer_algo == 1) {  // mmap_all.  all writing at once.

		if (comm.rank() == 0) std::cout << "write using MMAP: NOT FOR GPFS! 1 file per core, 1core/node at a time: for LARGE NODE COUNT.  DO NOT USE." << std::endl;
		if (target_size > max_fsize)  throw std::logic_error("target file size larger than supported");


    BL_BENCH_START(test);

    out_filename.append(std::to_string(comm.rank()));

    mxx::comm group = comm.split_shared();  // split comm into groups that share memory, i.e. by node.
    int max_group_size = group.size();  // size of each group.
    //max_group_size = mxx::allreduce(max_group_size, mxx::max<int>(), comm);  // get the biggest group size.
				// q starts at offset 0.
		unsigned char *q = map_out_file(out_filename, target_size, 0);


		size_t copied = 0;
    for (int i = 0; i < max_group_size; ++i) {

      if (i == group.rank()) {
				// copy data into array (writing)
				copied = copyToByteArray(idx.get_map(), q);

		}
			group.barrier();
		}
				// unmap the file
				unmap_out_file(q, target_size, 0);
	
		// barrier for all.
    BL_BENCH_COLLECTIVE_END(test, "write_1", copied, comm);


  } else {
    if (comm.rank() == 0) std::cout << "WRITE DISABLED." << std::endl;
  }




  BL_BENCH_REPORT_MPI_NAMED(test, "app", comm);

  // mpi cleanup is automatic

  return 0;
}
