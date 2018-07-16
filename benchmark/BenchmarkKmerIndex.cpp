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


#include "index/kmer_index.hpp"

#include "utils/benchmark_utils.hpp"
#include "utils/exception_handling.hpp"

#include "tclap/CmdLine.h"

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

//========   Kmer parameters
#if (pDNA == 16)
using Alphabet = bliss::common::DNA16;
#elif (pDNA == 5)
using Alphabet = bliss::common::DNA5;
#elif (pDNA == 4)
using Alphabet = bliss::common::DNA;
#endif

#if defined(pK)
using KmerType = bliss::common::Kmer<pK, Alphabet, WordType>;
#else
using KmerType = bliss::common::Kmer<21, Alphabet, WordType>;
#endif

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
using CountType = uint32_t;

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
#elif (pDistTrans == IDEN) //if (pDistTrans == IDEN)
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
    #elif (pMAP == BROBINHOOD)
      using MapType = ::dsc::counting_batched_robinhood_map<
          KmerType, ValType, MapParams>;
    #elif (pMAP == MTROBINHOOD)  // hybrid version
      using MapType = ::hsc::counting_batched_robinhood_map<
          KmerType, ValType, MapParams>;
    #elif (pMAP == MTRADIXSORT)  // hybrid version
      using MapType = ::hsc::counting_batched_radixsort_map<
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

#ifdef INSERT_KMER_COUNT_PAIR
	using IndexType = bliss::index::kmer::CountIndex<MapType>;
#else
	using IndexType = bliss::index::kmer::CountIndex2<MapType>;
#endif

#endif



/*
 * BENCHMARK Kmer index building.
 *
 * variables:   hash function   (std, murmur, farm)
 *              store canonical or not (canonicalize on the fly, canonicalize on build/query, no op on build, query doubled)
 *              k (15, 21, 31, 63)
 *              backing container type (hashmap, unordered vecmap, sorted array)
 *
 *              file reader type - mpi-io, mmap, fileloader without prefetch.  mpi-io performs really well when file's cached.
 *
 */

//template <typename IndexType, typename KmerType = typename IndexType::KmerType>
//std::vector<KmerType> readForQuery(const std::string & filename, MPI_Comm comm) {
//
//  ::std::vector<KmerType> query;
//
//  IndexType::template read_file<PARSER_TYPE, ::bliss::index::kmer::KmerParser<KmerType> >(filename, query, comm);
//
//  return query;
//}


template <typename IndexType, typename KmerType = typename IndexType::KmerType>
std::vector<KmerType> readForQuery_mpiio(const std::string & filename, MPI_Comm comm) {

  ::std::vector<KmerType> query;

  IndexType idx(comm);

  ::bliss::io::KmerFileHelper::template read_file_mpiio<::bliss::index::kmer::KmerParser<KmerType>, PARSER_TYPE, bliss::io::NSplitSequencesIterator >(filename, query, comm);

  return query;
}

template <typename IndexType, typename KmerType = typename IndexType::KmerType>
std::vector<KmerType> readForQuery_mmap(const std::string & filename, MPI_Comm comm) {

  ::std::vector<KmerType> query;
  IndexType idx(comm);

  // default to including quality score iterators.
  ::bliss::io::KmerFileHelper::template read_file_mmap<::bliss::index::kmer::KmerParser<KmerType>, PARSER_TYPE, bliss::io::NSplitSequencesIterator >(filename, query, comm);

  return query;
}


template <typename IndexType, typename KmerType = typename IndexType::KmerType>
std::vector<KmerType> readForQuery_posix(const std::string & filename, MPI_Comm comm) {

  ::std::vector<KmerType> query;
  IndexType idx(comm);

  // default to including quality score iterators.
  ::bliss::io::KmerFileHelper::template read_file_posix<::bliss::index::kmer::KmerParser<KmerType>, PARSER_TYPE, bliss::io::NSplitSequencesIterator  >(filename, query, comm);

  return query;
}


template<typename KmerType>
void sample(std::vector<KmerType> &query, size_t n, unsigned int seed, mxx::comm const & comm) {
  //std::shuffle(query.begin(), query.end(), std::default_random_engine(seed));

  size_t n_p = (n / comm.size());
  std::vector<size_t> send_counts(comm.size(), n_p);

  if (n < static_cast<size_t>(comm.size())) {
    n_p = 1;

    for (size_t i = 0; i < n; ++i) {
      send_counts[(i + comm.rank()) % comm.size()] = 1;
    }
    for (int i = n; i < comm.size(); ++i) {
      send_counts[(i + comm.rank()) % comm.size()] = 0;
    }
  }

  std::vector<KmerType> out = ::mxx::all2allv(query, send_counts, comm);
  query.swap(out);

  if (comm.rank() == 0) std::cout << "shuffled query input." << std::endl;
}



/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv) {
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif

  //////////////// init logging
  LOG_INIT();


  //////////////// initialize MPI and openMP

  mxx::env e(argc, argv);
  mxx::comm comm;

  if (comm.rank() == 0) printf("EXECUTING %s\n", argv[0]);

  comm.barrier();


  //////////////// parse parameters

  std::string filename;
  filename.assign(PROJ_SRC_DIR);
#if (pPARSER == FASTA)
      filename.append("/test/data/test2.fasta");
#elif (pPARSER == FASTQ)
      filename.append("/test/data/test.small.fastq");
#endif
  std::string queryname(filename);

  int sample_ratio = 2;

  double max_load = 0.8;
  double min_load = 0.35;
  uint8_t insert_prefetch = 8;
  uint8_t query_prefetch = 8;

  bool balance_input = false;

  int reader_algo = -1;
  // Wrap everything in a try block.  Do this every time,
  // because exceptions will be thrown for problems.
  try {

    // Define the command line object, and insert a message
    // that describes the program. The "Command description message"
    // is printed last in the help text. The second argument is the
    // delimiter (usually space) and the last one is the version number.
    // The CmdLine object parses the argv array based on the Arg objects
    // that it contains.
    TCLAP::CmdLine cmd("Benchmark parallel kmer index building", ' ', "0.1");

    // Define a value argument and add it to the command line.
    // A value arg defines a flag and a type of value that it expects,
    // such as "-n Bishop".
    TCLAP::ValueArg<std::string> fileArg("F", "file", "FASTQ file path", false, filename, "string", cmd);
    TCLAP::ValueArg<std::string> queryArg("Q", "query", "FASTQ file path for query. default to same file as index file", false, "", "string", cmd);

	TCLAP::SwitchArg balanceArg("b", "balance-input", "balance the input", cmd, balance_input);

    TCLAP::ValueArg<int> algoArg("A",
                                 "algo", "Reader Algorithm id. Fileloader w/o preload = 2, mmap = 5, posix=7, mpiio = 10. default is 7.",
                                 false, 7, "int", cmd);

    TCLAP::ValueArg<int> sampleArg("q",
                                 "query-sample", "sampling ratio for the query kmers. default=2 (half)",
                                 false, sample_ratio, "int", cmd);

	  TCLAP::ValueArg<double> maxLoadArg("","max_load","maximum load factor", false, max_load, "double", cmd);
	  TCLAP::ValueArg<double> minLoadArg("","min_load","minimum load factor", false, min_load, "double", cmd);
	  TCLAP::ValueArg<uint32_t> insertPrefetchArg("","insert_prefetch","number of elements to prefetch during insert", false, insert_prefetch, "uint32_t", cmd);
	  TCLAP::ValueArg<uint32_t> queryPrefetchArg("","query_prefetch","number of elements to prefetch during queries", false, query_prefetch, "uint32_t", cmd);

#ifdef VTUNE_ANALYSIS
    std::vector<std::string> measure_modes;
    measure_modes.push_back("insert");
    measure_modes.push_back("find");
    measure_modes.push_back("count");
    measure_modes.push_back("erase");
    measure_modes.push_back("reserve");
    measure_modes.push_back("transform");
    measure_modes.push_back("unique");
    measure_modes.push_back("bucket");
    measure_modes.push_back("permute");
    measure_modes.push_back("a2a");
    measure_modes.push_back("disabled");
    TCLAP::ValuesConstraint<std::string> measureModeVals( measure_modes );
    TCLAP::ValueArg<std::string> measureModeArg("","measured_op","function to measure (default insert)",false,"insert",&measureModeVals, cmd);
#endif



    // Parse the argv array.
    cmd.parse( argc, argv );

    // Get the value parsed by each arg.
    queryname = queryArg.getValue();   // get this first
    if (queryname.empty()) // at default  set to same as input.
    	queryname = fileArg.getValue();

    balance_input = balanceArg.getValue();

    filename = fileArg.getValue();
    reader_algo = algoArg.getValue();
    sample_ratio = sampleArg.getValue();

	  min_load = minLoadArg.getValue();
	  max_load = maxLoadArg.getValue();
	  insert_prefetch = insertPrefetchArg.getValue();
	  query_prefetch = queryPrefetchArg.getValue();


#ifdef VTUNE_ANALYSIS
    // set the default for query to filename, and reparse
    std::string measure_mode_str = measureModeArg.getValue();
    if (comm.rank() == 0) std::cout << "Measuring " << measure_mode_str << std::endl;

    if (measure_mode_str == "insert") {
      measure_mode = MEASURE_INSERT;
    } else if (measure_mode_str == "find") {
      measure_mode = MEASURE_FIND;
    } else if (measure_mode_str == "count") {
      measure_mode = MEASURE_COUNT;
    } else if (measure_mode_str == "erase") {
      measure_mode = MEASURE_ERASE;
    } else if (measure_mode_str == "reserve") {
      measure_mode = MEASURE_RESERVE;
    } else if (measure_mode_str == "transform") {
      measure_mode = MEASURE_TRANSFORM;
    } else if (measure_mode_str == "unique") {
      measure_mode = MEASURE_UNIQUE;
    } else if (measure_mode_str == "bucket") {
      measure_mode = MEASURE_BUCKET;
    } else if (measure_mode_str == "permute") {
      measure_mode = MEASURE_PERMUTE;
    } else if (measure_mode_str == "a2a") {
      measure_mode = MEASURE_A2A;
    } else {
      measure_mode = MEASURE_DISABLED;
    }
#endif


    // Do what you intend.

  } catch (TCLAP::ArgException &e)  // catch any exceptions
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    exit(-1);
  }



  if (comm.rank() == 0) {
    std::cout << "USING:\tKmerType=" << typeid(KmerType).name() << std::endl;
    std::cout << "      \tValType=" << typeid(ValType).name() << std::endl;
    std::cout << "      \tDistHash=" << typeid(DistHash<KmerType>).name() << std::endl;
    std::cout << "      \tDistTrans=" << typeid(DistTrans<KmerType>).name() << std::endl;
    std::cout << "      \tStoreHash=" << typeid(StoreHash<KmerType>).name() << std::endl;
    std::cout << "      \tMapType=" << typeid(MapType).name() << std::endl;
  }

  // ================  read and get file
  IndexType idx(comm);


#if (pINDEX == COUNT)  // map
  #if (pMAP == UNORDERED)
	  // std unordered.
	  idx.get_map().get_local_container().max_load_factor(max_load);
	#elif (pMAP == ORDERED)
	  // std unordered.
	  idx.get_map().get_local_container().max_load_factor(max_load);
  #elif (pMAP == ROBINHOOD)
    idx.get_map().get_local_container().set_max_load_factor(max_load);
    idx.get_map().get_local_container().set_min_load_factor(min_load);
  #elif (pMAP == BROBINHOOD)
    idx.get_map().get_local_container().set_max_load_factor(max_load);
    idx.get_map().get_local_container().set_min_load_factor(min_load);
    idx.get_map().get_local_container().set_insert_lookahead(insert_prefetch);
    idx.get_map().get_local_container().set_query_lookahead(query_prefetch);
  #elif (pMAP == MTROBINHOOD)
    idx.get_map().set_max_load_factor(max_load);
    idx.get_map().set_min_load_factor(min_load);
    idx.get_map().set_insert_lookahead(insert_prefetch);
    idx.get_map().set_query_lookahead(query_prefetch);

  #endif
#endif

  BL_BENCH_INIT(test);
  {
	  ::std::vector<typename IndexType::KmerParserType::value_type> temp;

	  BL_BENCH_COLLECTIVE_START(test, "read", comm);
//	  if (reader_algo == 2)
//	  {
//		if (comm.rank() == 0) printf("reading %s via fileloader\n", filename.c_str());
//
//		idx.read_file<PARSER_TYPE, typename IndexType::KmerParserType>(filename, temp, comm);
//
//	  } else
	  if (reader_algo == 5) {
		if (comm.rank() == 0) printf("reading %s via mmap\n", filename.c_str());
		::bliss::io::KmerFileHelper::read_file_mmap<typename IndexType::KmerParserType, PARSER_TYPE,
		 bliss::io::NSplitSequencesIterator
		 >(filename, temp, comm);

	  } else if (reader_algo == 7) {
		if (comm.rank() == 0) printf("reading %s via posix\n", filename.c_str());
		::bliss::io::KmerFileHelper::read_file_posix<typename IndexType::KmerParserType, PARSER_TYPE, bliss::io::NSplitSequencesIterator>(filename, temp, comm);

	  } else if (reader_algo == 10){
		if (comm.rank() == 0) printf("reading %s via mpiio\n", filename.c_str());
		::bliss::io::KmerFileHelper::read_file_mpiio<typename IndexType::KmerParserType, PARSER_TYPE, bliss::io::NSplitSequencesIterator>(filename, temp, comm);
	  } else {
		throw std::invalid_argument("missing file reader type");
	  }
	  BL_BENCH_END(test, "read", temp.size());

	  size_t total = mxx::allreduce(temp.size(), comm);
	  if (comm.rank() == 0) printf("total size is %lu\n", total);


	  if (balance_input) {
		  BL_BENCH_COLLECTIVE_START(test, "balance", comm);
		  ::mxx::distribute_inplace(temp, comm);
		  BL_BENCH_END(test, "balance", temp.size());
	  }


	  BL_BENCH_COLLECTIVE_START(test, "insert", comm);
	  idx.insert(temp);
	  BL_BENCH_END(test, "insert", idx.local_size());

    total = idx.size();
    if (comm.rank() == 0) printf("total size after insert/rehash is %lu\n", total);
  }

  {

  if (comm.rank() == 0) printf("reading query %s via posix\n", queryname.c_str());
  BL_BENCH_COLLECTIVE_START(test, "read_query", comm);
  auto query = readForQuery_posix<IndexType>(queryname, comm);
  BL_BENCH_END(test, "read_query", query.size());

  BL_BENCH_COLLECTIVE_START(test, "sample", comm);
  sample(query, query.size() / sample_ratio, comm.rank(), comm);
  BL_BENCH_END(test, "sample", query.size());



	  {
		  auto lquery = query;
#if (pMAP == MTROBINHOOD) || (pMAP == MTRADIXSORT) || (pMAP == BROBINHOOD) || (pMAP == RADIXSORT)
		  BL_BENCH_COLLECTIVE_START(test, "count", comm);
		  uint8_t * count_res = ::utils::mem::aligned_alloc<uint8_t>(lquery.size(), 64);
		  size_t count_res_size = idx.get_map().count(lquery, count_res);

		  ::utils::mem::aligned_free(count_res);
		  BL_BENCH_END(test, "count", count_res_size);
#else
		  BL_BENCH_COLLECTIVE_START(test, "count", comm);
		  auto counts = idx.count(lquery);
		  BL_BENCH_END(test, "count", counts.size());
#endif
		  }

	  {
		  auto lquery = query;
#if (pMAP == MTROBINHOOD) || (pMAP == MTRADIXSORT) || (pMAP == BROBINHOOD) || (pMAP == RADIXSORT)
		  BL_BENCH_COLLECTIVE_START(test, "find", comm);
		  CountType * find_res = ::utils::mem::aligned_alloc<CountType>(lquery.size(), 64);
		  size_t find_res_size = idx.get_map().find(lquery, find_res);

		  ::utils::mem::aligned_free(find_res);
		  BL_BENCH_END(test, "find", find_res_size);
#else
		  BL_BENCH_COLLECTIVE_START(test, "find", comm);
		  auto found = idx.find(lquery);
		  BL_BENCH_END(test, "find", found.size());
#endif
	  }


	  BL_BENCH_COLLECTIVE_START(test, "erase", comm);
	  idx.erase(query);
	  BL_BENCH_END(test, "erase", idx.local_size());

  }

  
  BL_BENCH_REPORT_MPI_NAMED(test, "app", comm);


  // mpi cleanup is automatic
  comm.barrier();

  return 0;
}
