

#define ITER_MODE 1
#define INDEX_MODE 2
#define INTEGRATED_MODE 3
#define SORT_MODE 4
#define SHUFFLE_MODE 5

#include "bliss-config.hpp"



#define MEASURE_ESTIMATE 6
#define MEASURE_INSERT 1
#define MEASURE_FIND 2
#define MEASURE_COUNT 3
#define MEASURE_ERASE 4
#define MEASURE_COUNT2 5
#define MEASURE_DISABLED 0


#ifdef VTUNE_ANALYSIS

#include <ittnotify.h>

#endif

#include <unordered_map>
#include <vector>
#include <random>
#include <cstdint>

#include <tuple>
#include <string>
#include <exception>
#include <functional>

#include "robinhood.h"   // from martinus

#if 0
#include <tommyds/tommyalloc.h>
#include <tommyds/tommyalloc.c>
#include <tommyds/tommyhashdyn.h>
#include <tommyds/tommyhashdyn.c>
#include <tommyds/tommyhashlin.h>
#include <tommyds/tommyhashlin.c>
#include <tommyds/tommytrie.h>
#include <tommyds/tommytrie.c>

#endif

#include "containers/unordered_vecmap.hpp"
//#include "containers/hashed_vecmap.hpp"
#include "containers/densehash_map.hpp"

#include "kmerhash/hashmap_linearprobe.hpp"
#include "kmerhash/hashmap_robinhood.hpp"
// experimental
#include "kmerhash/experimental/hashmap_robinhood_doubling_noncircular3.hpp"
//#include "kmerhash/experimental/hashmap_robinhood_doubling_memmove.hpp"
//#include "kmerhash/experimental/hashmap_robinhood_doubling_offsets2.hpp"
#include "kmerhash/experimental/hashmap_robinhood_offsets_prefetch.hpp"
#include "kmerhash/robinhood_offset_hashmap_ptr.hpp"
#include "kmerhash/hashmap_radixsort.hpp"

#include "kmerhash/hashmap_robinhood_prefetch.hpp"

#include "common/kmer.hpp"
#include "common/kmer_transform.hpp"
#include "index/kmer_hash.hpp"

#include "tclap/CmdLine.h"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

#include "utils/benchmark_utils.hpp"
#include "utils/transform_utils.hpp"

#include "kmerhash/hyperloglog64.hpp"

#include <smmintrin.h>  // for _mm_stream_load_si128
#include <stdlib.h>  // aligned_alloc
#include <cstring>  // memcpy

#include "kmerhash/io_utils.hpp"
#include "kmerhash/hash_new.hpp"


// comparison of some hash tables.  note that this is not exhaustive and includes only the well tested ones and my own.  not so much
// the one-off ones people wrote.
// see http://preshing.com/20110603/hash-table-performance-tests/  - suggests google sparsehash dense, and Judy array
//      http://incise.org/hash-table-benchmarks.html  - suggests google dense hash map and glib ghashtable
//      https://attractivechaos.wordpress.com/2008/08/28/comparison-of-hash-table-libraries/  - suggests google sparsehash dense and khash (distant second)
//      http://preshing.com/20130107/this-hash-table-is-faster-than-a-judy-array/  - suggests judy array
//      http://www.tommyds.it/doc/index.html  - suggets Tommy_hashtable and google dense.  at the range we operate, Tommy and Google Densehash are competitive.
//      http://www.nothings.org/computer/judy/ - shows that judy performs poorly with random data insertion. sequential is better (so sorted kmers would be better)

// same with unordered vecmap
// unordered multimap is very slow relative to google dense

// google dense requires an empty key and a deleted key. it is definitely fast.
//      it does not support multimap...

// results:  google dense hash is fastest.
// question is how to make google dense hash support multimap style operations?  vector is expensive...

//TODO: [ ] refactor main function.
//      [ ] estimate for all cases.
//      [ ] use only insert(iter, iter)

#define STD 21
#define IDEN 20
#define FARM 22
#define FARM32 23
#define MURMUR 24
#define MURMUR32 25
#define MURMUR32sse 26
#define MURMUR32avx 27
#define MURMUR64avx 28
#define CRC32C 29
#define CLHASH 30


#define LOOK_AHEAD 16


using CountType = uint32_t;

// storage hash type
#if (pStoreHash == STD)
  template <typename KM>
  using StoreHash = bliss::kmer::hash::cpp_std<KM, false>;
#elif (pStoreHash == IDEN)
  template <typename KM>
  using StoreHash = fsc::hash::identity<KM>;
#elif (pStoreHash == MURMUR)
  template <typename KM>
  using StoreHash = fsc::hash::murmur<KM>;
#elif (pStoreHash == MURMUR32)
  template <typename KM>
  using StoreHash = fsc::hash::murmur32<KM>;
#elif (pStoreHash == MURMUR32sse)
  template <typename KM>
  using StoreHash = fsc::hash::murmur3sse32<KM>;
#elif (pStoreHash == MURMUR32avx)
  template <typename KM>
  using StoreHash = fsc::hash::murmur3avx32<KM>;
  #elif (pStoreHash == MURMUR64avx)
  template <typename KM>
  using StoreHash = fsc::hash::murmur3avx64<KM>;
#elif (pStoreHash == CRC32C)
  template <typename KM>
  using StoreHash = fsc::hash::crc32c<KM>;
#elif (pStoreHash == CLHASH)
  template <typename KM>
  using StoreHash = fsc::hash::clhash<KM>;
#elif (pStoreHash == FARM32)
  template <typename KM>
  using StoreHash = fsc::hash::farm32<KM>;
#elif (pStoreHash == FARM)
  template <typename KM>
  using StoreHash = fsc::hash::farm<KM>;
//  template <typename KM>
//  using StoreHash = bliss::kmer::hash::cpp_std<KM, false>;
#endif


template <class T>
struct equal_to {
    using result_type = bool;
    using first_argument_type = T;
    using second_argument_type = T;

    inline constexpr bool operator()(T const & lhs, T const & rhs) const {
      return lhs == rhs;
    }
};


template <typename Kmer, typename Value>
void generate_input(std::vector<::std::pair<Kmer, Value> > & output,
		size_t const count,
		size_t const repeats = 10, bool canonical = false) {
  output.reserve(count);

  size_t freq;
  //typename Kmer::KmerWordType val;
  Kmer k, kr;

  srand(23);
  for (size_t i = 0; i < count; ++i) {

//    if ((i + LOOK_AHEAD) < count) {
//      _mm_prefetch(&(output[i]), _MM_HINT_T0);
//    }

    for (size_t j = 0; j < Kmer::nWords; ++j) {
      //val = static_cast<typename Kmer::KmerWordType>(static_cast<long>(rand()) << 32) ^ static_cast<long>(rand());
      k.getDataRef()[j] = static_cast<typename Kmer::KmerWordType>(static_cast<long>(rand()) << 32) ^ static_cast<long>(rand());
    }
    k.sanitize();

    // do the reverse complement if desired.
    if (canonical) {
      kr = k.reverse_complement();
      k = (k < kr) ? k : kr;
    }

    output.emplace_back(k, i);

    // average repeat/2 times inserted.
    freq = rand() % repeats;

    for (size_t j = 0; j < freq; ++j) {
    	if (i+1 < count) {
    		++i;

    		output.emplace_back(k, i);
    	}
    }

  }

  // do random shuffling to avoid consecutively identical items.
  std::random_shuffle(output.begin(), output.end());
}


template <typename Kmer, typename Value>
std::vector<::std::pair<Kmer, Value> > generate_input( size_t const count, size_t const repeats = 10,
		bool canonical = false) {
	std::vector<::std::pair<Kmer, Value> > output;

	generate_input(output, count, repeats, canonical);

	return output;
}

template <typename K, typename T>
std::vector<::std::pair<K, T> > get_input(::std::string const & fname,
		size_t const & count, size_t const & repeat_rate = 10, bool canonical = false) {
	  if (fname.compare("") == 0) {
		  return generate_input<K, T>(count, repeat_rate, canonical);
	  } else {
		  return deserialize_vector<::std::pair<K, T> >(fname);
	  }
}




template <typename Kmer, typename Value>
void benchmark_unordered_map(std::string name, std::vector<::std::pair<Kmer, Value> > const & input,
		size_t const query_frac,
		double const max_load, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved
  ::std::unordered_map<Kmer, Value, StoreHash<Kmer> > map;//(count * 2 / repeat_rate);
  map.max_load_factor(max_load);


  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }


  BL_BENCH_START(map);
  size_t result = 0;
  for (auto q : query) {
    auto iters = map.equal_range(q);
    if (iters.first != iters.second) ++result;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count", result);


  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
     result += map.erase(q);
  }
  BL_BENCH_END(map, "erase", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}


// the number of bits should be 64
template <typename Kmer, typename Value>
void benchmark_classicRH(std::string name, std::vector<::std::pair<Kmer, Value> > const & input,
		size_t const query_frac,
		::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
#if (pStoreHash == STD)
  RobinHoodHashMap<Value, std::hash<size_t> > map;
#else
  RobinHoodHashMap<Value, StoreHash<size_t> > map;
#endif
  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());

    BL_BENCH_START(map);
    for (size_t i = 0; i < input.size(); ++i) {
    	map.insert(*(reinterpret_cast<size_t const *>(input[i].first.getData())), input[i].second);
    }
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  bool success = false;
  for (auto q : query) {
	  map.find(*(reinterpret_cast<size_t const *>(q.getData())), success);
    if (success)
      ++result;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  BL_BENCH_END(map, "count", 0);

  BL_BENCH_START(map);
  BL_BENCH_END(map, "erase", 0);

  BL_BENCH_START(map);
  BL_BENCH_END(map, "count2", 0);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}


template <typename Kmer, typename Value>
void benchmark_densehash_map(std::string name, std::vector<::std::pair<Kmer, Value> > const & input,
		size_t const query_frac,
		::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::fsc::densehash_map<Kmer, Value, 
	::bliss::kmer::hash::sparsehash::special_keys<Kmer, false>,
	::bliss::transform::identity,
	StoreHash<Kmer> > map;//(count * 2 / repeat_rate);

  BL_BENCH_END(map, "reserve", input.size());



  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (auto q : query) {
    auto iters = map.equal_range(q);
    if (iters.first != iters.second)
      ++result;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  result = map.erase(query.begin(), query.end());

//  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
//    result += map.erase(query[i]);
//  }
  map.resize(0);
  BL_BENCH_END(map, "erase", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count2", result);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}


template <bool canonical = false, typename Kmer, typename Value>
void benchmark_densehash_full_map(std::string name, std::vector<::std::pair<Kmer, Value> > const & input,
		size_t const query_frac,
		::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::fsc::densehash_map<Kmer, Value, 
	::bliss::kmer::hash::sparsehash::special_keys<Kmer, canonical>,
	::bliss::transform::identity,
	StoreHash<Kmer> > map;//(count * 2 / repeat_rate);

  BL_BENCH_END(map, "reserve", input.size());



  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (auto q : query) {
    auto iters = map.equal_range(q);
    if (iters.first != iters.second) ++result;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  result = map.erase(query.begin(), query.end());

//  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
//    result += map.erase(query[i]);
//  }
  map.resize(0);
  BL_BENCH_END(map, "erase", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count2", result);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}





template <typename Kmer, typename Value>
void benchmark_google_densehash_map(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, double const max_load, double const min_load, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::google::dense_hash_map<Kmer, Value,
	StoreHash<Kmer> > map; //(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", input.size());

  map.max_load_factor(max_load);
  map.min_load_factor(min_load);

  ::bliss::kmer::hash::sparsehash::special_keys<Kmer, false> special;

  map.set_empty_key(special.generate(0));
  map.set_deleted_key(special.generate(1));

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (auto q : query) {
    auto iters = map.equal_range(q);
    if (iters.first != iters.second) ++result;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  //result = map.erase(query.begin(), query.end());
  result = 0;
  for (auto q : query) {
    result += map.erase(q);
  }
  map.resize(0);
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
  result = 0;
  for (auto q : query) {
    result += map.count(q);
  }
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}



template <template <typename, typename, typename, typename, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap_insert_mode(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, int vector_mode, int measure_mode,
		double const max_load, double const min_load, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, StoreHash<Kmer>, ::equal_to<Kmer>, ::std::allocator<std::pair<Kmer, Value> > >;

  std::cout << " tuple size " << sizeof(typename MAP_TYPE::value_type) << std::endl;

  //MAP_TYPE map(count * 2 / repeat_rate);
  MAP_TYPE map;
  map.set_max_load_factor(max_load);
  map.set_min_load_factor(min_load);


  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());


    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_resume();
#endif

	// compute hyperloglog estimate for reference.
    Kmer * keys = (Kmer*)_mm_malloc(input.size() * sizeof(Kmer), 64);
	hyperloglog64<Kmer, StoreHash<Kmer>, 12> hll;
	for (size_t i = 0; i < input.size(); ++i) {
		keys[i] = input[i].first;
	}
	hll.update(keys, input.size());

	double est = hll.estimate();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_pause();
#endif
    _mm_free(keys);
    BL_BENCH_END(map, "estimate", static_cast<size_t>(est));

	std::cout << "insert testing: estimated distinct = " << est << " in " << input.size() << std::endl;


    std::string insert_type;
    if (vector_mode == ITER_MODE) {
    	insert_type = "insert";
    } else if (vector_mode == INDEX_MODE) {
    	insert_type = "v_insert";
    } else if (vector_mode == INTEGRATED_MODE) {
        insert_type = "insert_integrated";
    } else if (vector_mode == SORT_MODE) {
    	std::cout << "WARNING: SORTING ONLY, NO INSERTION.  4x slower on i5-4300U hashwell with 10M DNA 31-mers even without insertion." << std::endl;
    	insert_type = "insert_sorted";
    } else if (vector_mode == SHUFFLE_MODE) {
    	std::cout << "WARNING: SHUFFLING ONLY, NO INSERTION.  2x slower on i5-4300U hashwell with 10M DNA 31-mers, even without insertion." << std::endl;
    	insert_type =  "insert_shuffled";
    } else {
    	insert_type = "insert";
    }

    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    if (vector_mode == ITER_MODE) {
    	map.insert(input.begin(), input.end());
    } else if (vector_mode == INDEX_MODE) {
    	map.insert(input);
    } else if (vector_mode == INTEGRATED_MODE) {
    	map.insert_integrated(input);
    } else if (vector_mode == SORT_MODE) {
    	map.insert_sort(input);
    } else if (vector_mode == SHUFFLE_MODE) {
    	map.insert_shuffled(input);
    } else {
    	map.insert(input.begin(), input.end());
    }
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(map, insert_type, map.size());
  }

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
  auto finds = map.find(query.begin(), query.end());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find", finds.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
  auto counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count", counts.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
  size_t result = map.erase(query.begin(), query.end());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_resume();
#endif
  counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count2", counts.size());


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}




template <template <typename, typename, typename, typename, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap_insert_mode(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, int vector_mode, int measure_mode,
		double const max_load, double const min_load, unsigned char const insert_prefetch, unsigned char const query_prefetch, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);
  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, StoreHash<Kmer>, ::equal_to<Kmer>, ::std::allocator<std::pair<Kmer, Value> > >;

  std::cout << " tuple size " << sizeof(typename MAP_TYPE::value_type) << std::endl;

  //MAP_TYPE map(count * 2 / repeat_rate);
  MAP_TYPE map;
  map.set_max_load_factor(max_load);
  map.set_min_load_factor(min_load);

  map.set_insert_lookahead(insert_prefetch);
  map.set_query_lookahead(query_prefetch);


  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());


    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_resume();
#endif

	// compute hyperloglog estimate for reference.
    Kmer * keys = (Kmer*)_mm_malloc(input.size() * sizeof(Kmer), 64);
	hyperloglog64<Kmer, StoreHash<Kmer>, 12> hll;
	for (size_t i = 0; i < input.size(); ++i) {
		keys[i] = input[i].first;
	}
	hll.update(keys, input.size());

	double est = hll.estimate();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_pause();
#endif
    _mm_free(keys);
    BL_BENCH_END(map, "estimate", static_cast<size_t>(est));

	std::cout << "insert testing: estimated distinct = " << est << " in " << input.size() << std::endl;


    std::string insert_type;
    if (vector_mode == ITER_MODE) {
    	insert_type = "insert";
    } else if (vector_mode == INDEX_MODE) {
    	insert_type = "v_insert";
    } else if (vector_mode == INTEGRATED_MODE) {
        insert_type = "insert_integrated";
    } else if (vector_mode == SORT_MODE) {
    	std::cout << "WARNING: SORTING ONLY, NO INSERTION.  4x slower on i5-4300U hashwell with 10M DNA 31-mers even without insertion." << std::endl;
    	insert_type = "insert_sorted";
    } else if (vector_mode == SHUFFLE_MODE) {
    	std::cout << "WARNING: SHUFFLING ONLY, NO INSERTION.  2x slower on i5-4300U hashwell with 10M DNA 31-mers, even without insertion." << std::endl;
    	insert_type =  "insert_shuffled";
    } else {
    	insert_type = "insert";
    }

    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    if (vector_mode == ITER_MODE) {
    	map.insert(input.begin(), input.end());
    } else if (vector_mode == INDEX_MODE) {
    	map.insert(input);
    } else if (vector_mode == INTEGRATED_MODE) {
    	map.insert_integrated(input);
    } else if (vector_mode == SORT_MODE) {
    	map.insert_sort(input);
    } else if (vector_mode == SHUFFLE_MODE) {
    	map.insert_shuffled(input);
    } else {
    	map.insert(input.begin(), input.end());
    }
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(map, insert_type, map.size());
  }

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
  auto finds = map.find(query.begin(), query.end());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find", finds.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
  auto counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count", counts.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
  size_t result = map.erase(query.begin(), query.end());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_resume();
#endif
  counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count2", counts.size());


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}

template <typename Kmer, typename Value>
void benchmark_hashmap_radixsort(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, int measure_mode, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);
  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  

  BL_BENCH_END(map, "reserve", input.size());

    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());


    ::fsc::hashmap_radixsort<Kmer, int32_t, StoreHash, ::equal_to > map;
    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_resume();
#endif

	// compute hyperloglog estimate for reference.
    Kmer *keyArray = (Kmer *)_mm_malloc(input.size() * sizeof(Kmer), 64);
    size_t i;
    for(i = 0; i < input.size(); i++)
    {
        keyArray[i] = input[i].first;
    }
	map.get_hll().update(keyArray, input.size());

	double est = map.get_hll().estimate();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_pause();
#endif
    BL_BENCH_END(map, "estimate", static_cast<size_t>(est));
	std::cout << "insert testing: estimated distinct = " << est << " in " << input.size() << std::endl;
	map.resize(static_cast<size_t>(est));

    std::string insert_type;
    insert_type = "batch_insert";

    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    int64_t startTick = __rdtsc();
    map.insert(keyArray, input.size());
    map.finalize_insert();
    int64_t endTick = __rdtsc();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(map, insert_type, map.size());
    _mm_free(keyArray);
    cout << "insert ticks = " << (endTick - startTick) << endl;

  map.sanity_check();
  Kmer *queryArray = (Kmer *)_mm_malloc(query.size() * sizeof(Kmer), 64);
  for(i = 0; i < query.size(); i++)
  {
      queryArray[i] = query[i];
  }
  uint32_t *findResult = (uint32_t *)_mm_malloc(query.size() * sizeof(uint32_t), 64);
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
  size_t foundCount = map.find(queryArray, query.size(), findResult);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find", foundCount);
  _mm_free(findResult);

  uint8_t *countResult = (uint8_t *)_mm_malloc(query.size() * sizeof(uint8_t), 64);
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
  foundCount = map.count(queryArray, query.size(), countResult);
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count", foundCount);

#if 1
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
  map.erase(queryArray, query.size());
  size_t eraseCount = map.finalize_erase();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
  BL_BENCH_END(map, "erase", eraseCount);


  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_resume();
#endif
  foundCount = map.count(queryArray, query.size(), countResult);
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count2", foundCount);
#endif
  _mm_free(countResult);

  _mm_free(queryArray);
  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}




template <template <typename, typename, typename, typename, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, int vector_mode, int measure_mode,
		double const max_load, double const min_load, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;


  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, StoreHash<Kmer>, ::equal_to<Kmer>, ::std::allocator<std::pair<Kmer, Value> > >;

  std::cout << " tuple size " << sizeof(typename MAP_TYPE::value_type) << std::endl;

  //MAP_TYPE map(count * 2 / repeat_rate);
  MAP_TYPE map;

  map.set_max_load_factor(max_load);
  map.set_min_load_factor(min_load);

  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());


    ::std::pair<Kmer, Value>* input_ptr;
    int ret = posix_memalign(reinterpret_cast<void **>(&input_ptr), 16, sizeof(::std::pair<Kmer, Value>) * input.size());
	if (ret)
		throw std::length_error("failed to allocate aligned memory");

	memcpy(input_ptr, input.data(), input.size() * sizeof(::std::pair<Kmer, Value>));

    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_resume();
#endif

	// compute hyperloglog estimate for reference.
    Kmer * keys = (Kmer*)_mm_malloc(input.size() * sizeof(Kmer), 64);
	hyperloglog64<Kmer, StoreHash<Kmer>, 12> hll;
	for (size_t i = 0; i < input.size(); ++i) {
		keys[i] = input[i].first;
	}
	hll.update(keys, input.size());

	double est = hll.estimate();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_pause();
#endif
    _mm_free(keys);
    BL_BENCH_END(map, "estimate_w_mmstreamload", static_cast<size_t>(est));

    free(input_ptr);
//	std::cout << "insert testing: estimated using _mm_stream_load_si128 (slower), distinct = " << est << " in " << input.size() << std::endl;
	std::cout << "insert testing: estimated using [] operator, distinct = " << est << " in " << input.size() << std::endl;


    std::string insert_type;
    if (vector_mode == INDEX_MODE) {
    	insert_type = "v_insert";
    } else {
    	insert_type = "insert";
    }


    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    if (vector_mode == INDEX_MODE) {
    	map.insert(input);
    } else {
    	map.insert(input.begin(), input.end());
    }
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(map, insert_type, map.size());
  }

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
  size_t result = 0;
  for (auto q : query) {
    auto iter = map.find(q);
    if (iter != map.end()) ++result;
  }
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
  auto counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count", counts.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
  result = map.erase(query.begin(), query.end());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_resume();
#endif
  counts = map.count(query.begin(), query.end());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count2", counts.size());


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}



// for robinhood offset with prefetch.
template <template <typename, typename, template <typename> class, template <typename> class, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap(std::string name, std::vector<::std::pair<Kmer, Value> > const & input, size_t const query_frac, int vector_mode, int measure_mode,
    double const max_load, double const min_load, unsigned char const insert_prefetch, unsigned char const query_prefetch, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, StoreHash, ::equal_to, ::std::allocator<std::pair<Kmer, Value> > >;

  std::cout << " tuple size " << sizeof(typename MAP_TYPE::value_type) << std::endl;

  //MAP_TYPE map(count * 2 / repeat_rate);
  MAP_TYPE map;
  map.set_max_load_factor(max_load);
  map.set_min_load_factor(min_load);

  map.set_insert_lookahead(insert_prefetch);
  map.set_query_lookahead(query_prefetch);


  BL_BENCH_END(map, "reserve", input.size());

  {
    BL_BENCH_START(map);
    query.resize(input.size() / query_frac);
    std::transform(input.begin(), input.begin() + query.size(), query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate query", input.size());


    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_resume();
#endif

  // compute hyperloglog estimate for reference.
    Kmer * keys = (Kmer*)_mm_malloc(input.size() * sizeof(Kmer), 64);
	for (size_t i = 0; i < input.size(); ++i) {
		keys[i] = input[i].first;
	}
	map.get_hll().update(keys, input.size());

	double est = map.get_hll().estimate();
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ESTIMATE)
        __itt_pause();
#endif
    _mm_free(keys);
    BL_BENCH_END(map, "estimate", static_cast<size_t>(est));

  std::cout << "insert testing: estimated distinct = " << est << " in " << input.size() << std::endl;
  map.reserve(est);

    std::string insert_type;
    if (vector_mode == ITER_MODE) {
      insert_type = "insert";
    } else if (vector_mode == INDEX_MODE) {
      insert_type = "v_insert";
    } else {
      insert_type = "insert";
    }

    BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_resume();
#endif
    if (vector_mode == ITER_MODE) {
      map.insert_no_estimate(input.data(), input.data() + input.size() );
    } else if (vector_mode == INDEX_MODE) {
      map.insert_no_estimate(input);
    } else {
      map.insert_no_estimate(input.data(), input.data() + input.size());
    }
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_INSERT)
        __itt_pause();
#endif
    BL_BENCH_END(map, insert_type, map.size());
  }


  {
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
      auto finds = map.template find<typename MAP_TYPE::value_type>(query.data(), query.data() + query.size());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find_pair", finds.size());
    }

  {
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
      auto finds = map.template find<Value>(query.data(), query.data() + query.size());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find_value", finds.size());
    }



  {
  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_resume();
#endif
      auto finds = map.find_existing(query.data(), query.data() + query.size());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_FIND)
        __itt_pause();
#endif
  BL_BENCH_END(map, "find_existing", finds.size());
  }

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_resume();
#endif
  auto counts = map.count(query.data(), query.data() + query.size());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count", counts.size());

  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_resume();
#endif
  size_t result = map.erase(query.data(), query.data() + query.size());
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_ERASE)
        __itt_pause();
#endif
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_resume();
#endif
  counts = map.count(query.data(), query.data() + query.size());
  //result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
#ifdef VTUNE_ANALYSIS
    if (measure_mode == MEASURE_COUNT2)
        __itt_pause();
#endif
  BL_BENCH_END(map, "count2", counts.size());


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}




#define STD_UNORDERED_TYPE 1
#define GOOGLE_TYPE 2
#define KMERIND_TYPE 3
#define LINEARPROBE_TYPE 4
#define ROBINHOOD_TYPE 5
#define ROBINHOOD_NONCIRC_TYPE 6
#define ROBINHOOD_OFFSET_TYPE 7
#define ROBINHOOD_OFFSET2_TYPE 9
#define ROBINHOOD_PREFETCH_TYPE 8
#define RADIXSORT_TYPE 10
#define CLASSIC_ROBINHOOD_TYPE 11
#define ALL_TYPE 0

#define DNA_TYPE 1
#define DNA5_TYPE 2
#define DNA16_TYPE 3




/// parse the parameters.  return int map type, int DNA type, bool full, and bool canonical
/// size, query frac, repeat rate.  vector mode (input is vector, not iterator.), then measure_func
std::tuple<int, int, bool, bool, size_t, size_t, size_t, int, int, double, double, unsigned char, unsigned char, std::string>
parse_cmdline(int argc, char** argv) {

	int map = ALL_TYPE;
	int dna = DNA_TYPE;
	bool canonical = false;
	bool full = false;

	  size_t count = 100000000;
	  size_t query_frac = 2;
	  size_t repeat_rate = 10;

	  int insert_mode = INDEX_MODE;
	  int measure_mode = MEASURE_DISABLED;

	  double max_load = 0.8;
	  double min_load = 0.35;
	  uint8_t insert_prefetch = 8;
	  uint8_t query_prefetch = 8;

	  std::string filename;

	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {

	  // Define the command line object, and insert a message
	  // that describes the program. The "Command description message"
	  // is printed last in the help text. The second argument is the
	  // delimiter (usually space) and the last one is the version number.
	  // The CmdLine object parses the argv array based on the Arg objects
	  // that it contains.
	  TCLAP::CmdLine cmd("Benchmark parallel kmer hash table", ' ', "0.1");


	  std::vector<std::string> allowed;
	  allowed.push_back("std_unordered");
	  allowed.push_back("google_densehash");
	  allowed.push_back("kmerind");
	  allowed.push_back("linearprobe");
	  allowed.push_back("classic_robinhood");
	  allowed.push_back("robinhood");
    allowed.push_back("robinhood_prefetch");
//	  allowed.push_back("robinhood_noncirc");
	  allowed.push_back("robinhood_offset");
	  allowed.push_back("robinhood_offset_overflow");
	  allowed.push_back("radixsort");
    allowed.push_back("all");
	  TCLAP::ValuesConstraint<std::string> allowedVals( allowed );

	  TCLAP::ValueArg<std::string> mapArg("m","map_type","type of map to use (default ALL)",false,"robinhood_offset_overflow",&allowedVals, cmd);

	  std::vector<std::string> allowed_alphabet;
	  allowed_alphabet.push_back("dna");
	  allowed_alphabet.push_back("dna5");
	  allowed_alphabet.push_back("dna16");
	  TCLAP::ValuesConstraint<std::string> allowedAlphabetVals( allowed_alphabet );
	  TCLAP::ValueArg<std::string> alphabetArg("A","alphabet","alphabet to use (default dna)",false,"dna",&allowedAlphabetVals, cmd);

	  std::vector<std::string> insert_modes;
	  insert_modes.push_back("iter");
	  insert_modes.push_back("index");
	  insert_modes.push_back("integrated");
	  insert_modes.push_back("sort");
	  insert_modes.push_back("shuffle");
	  TCLAP::ValuesConstraint<std::string> insertModeVals( insert_modes );
	  TCLAP::ValueArg<std::string> insertModeArg("I","insert_mode","insert mode (default index)",false,"index",&insertModeVals, cmd);


	  // Define a value argument and add it to the command line.
	  // A value arg defines a flag and a type of value that it expects,
	  // such as "-n Bishop".
	  TCLAP::SwitchArg fullArg("f", "full", "set k-mer to fully occupy machine word", cmd, full);
	  TCLAP::SwitchArg canonicalArg("c", "canonical", "use canonical k-mers", cmd, canonical);

	  TCLAP::ValueArg<std::string> fileArg("F", "file", "<kmer, count> binary file path", false, filename, "string", cmd);

	  TCLAP::ValueArg<size_t> countArg("N","num_elements","number of elements", false, count, "size_t", cmd);
	  TCLAP::ValueArg<size_t> queryArg("Q","query_fraction","percent of count to use for query", false, query_frac, "size_t", cmd);
	  TCLAP::ValueArg<size_t> repeatArg("R","repeate_rate","maximum number of repeats in data", false, repeat_rate, "size_t", cmd);

	  TCLAP::ValueArg<double> maxLoadArg("","max_load","maximum load factor", false, max_load, "double", cmd);
	  TCLAP::ValueArg<double> minLoadArg("","min_load","minimum load factor", false, min_load, "double", cmd);
	  TCLAP::ValueArg<uint32_t> insertPrefetchArg("","insert_prefetch","number of elements to prefetch during insert", false, insert_prefetch, "uint32_t", cmd);
	  TCLAP::ValueArg<uint32_t> queryPrefetchArg("","query_prefetch","number of elements to prefetch during queries", false, query_prefetch, "uint32_t", cmd);

	  std::vector<std::string> measure_modes;
	  measure_modes.push_back("estimate");
	  measure_modes.push_back("insert");
	  measure_modes.push_back("find");
	  measure_modes.push_back("count");
	  measure_modes.push_back("erase");
	  measure_modes.push_back("count2");
	  measure_modes.push_back("disabled");
	  TCLAP::ValuesConstraint<std::string> measureModeVals( measure_modes );
	  TCLAP::ValueArg<std::string> measureModeArg("","measured_op","function to measure (default insert)",false,"insert",&measureModeVals, cmd);


	  // Parse the argv array.
	  cmd.parse( argc, argv );


	  std::string map_type = mapArg.getValue();
	  if (map_type == "std_unordered") {
		  map = STD_UNORDERED_TYPE;
	  } else if (map_type == "google_densehash") {
		  map = GOOGLE_TYPE;
	  } else if (map_type == "kmerind") {
		  map = KMERIND_TYPE;
	  } else if (map_type == "linearprobe") {
		  map = LINEARPROBE_TYPE;
	  } else if (map_type == "classic_robinhood") {
		  map = CLASSIC_ROBINHOOD_TYPE;
	  } else if (map_type == "robinhood") {
		  map = ROBINHOOD_TYPE;
	  } else if (map_type == "robinhood_noncirc") {
		  map = ROBINHOOD_NONCIRC_TYPE;
	  } else if (map_type == "robinhood_offset") {
		  map = ROBINHOOD_OFFSET_TYPE;
	  } else if (map_type == "robinhood_offset_overflow") {
		  map = ROBINHOOD_OFFSET2_TYPE;
	  } else if (map_type == "robinhood_prefetch") {
		  map = ROBINHOOD_PREFETCH_TYPE;
	  } else if (map_type == "radixsort") {
		  map = RADIXSORT_TYPE;
	  } else if (map_type == "all") {
	    map = ALL_TYPE;
	  }

	  std::string alpha = alphabetArg.getValue();
	  if (alpha == "DNA") {
		  dna = DNA_TYPE;
	  } else if (alpha == "DNA5") {
		  dna = DNA5_TYPE;
	  } else if (alpha == "DNA16") {
		  dna = DNA16_TYPE;
	  }

	  full = fullArg.getValue();
	  canonical = canonicalArg.getValue();

	  count = countArg.getValue();
	  query_frac = queryArg.getValue();
	  repeat_rate = repeatArg.getValue();

	  filename = fileArg.getValue();

	  min_load = minLoadArg.getValue();
	  max_load = maxLoadArg.getValue();
	  insert_prefetch = insertPrefetchArg.getValue();
	  query_prefetch = queryPrefetchArg.getValue();


	  std::string insert_mode_str = insertModeArg.getValue();
	  if (insert_mode_str == "iter") {
		  insert_mode = ITER_MODE;
	  } else if (insert_mode_str == "index") {
		  insert_mode = INDEX_MODE;
	  } else if (insert_mode_str == "integrated") {
		  insert_mode = INTEGRATED_MODE;
	  } else if (insert_mode_str == "sort") {
		  insert_mode = SORT_MODE;
	  } else if (insert_mode_str == "shuffle") {
		  insert_mode = SHUFFLE_MODE;
	  }

	  std::string measure_mode_str = measureModeArg.getValue();
	  std::cout << "Measuring " << measure_mode_str << std::endl;
	  if (measure_mode_str == "insert") {
		  measure_mode = MEASURE_INSERT;
	  } else if (measure_mode_str == "estimate") {
		  measure_mode = MEASURE_ESTIMATE;
	  } else if (measure_mode_str == "find") {
		  measure_mode = MEASURE_FIND;
	  } else if (measure_mode_str == "count") {
		  measure_mode = MEASURE_COUNT;
	  } else if (measure_mode_str == "erase") {
		  measure_mode = MEASURE_ERASE;
	  } else if (measure_mode_str == "count2") {
		  measure_mode = MEASURE_COUNT2;
	  } else
		  measure_mode = MEASURE_DISABLED;


	  // Do what you intend.

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
	  std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	  exit(-1);
	}


	return std::make_tuple(map, dna, full, canonical, count, query_frac, repeat_rate, insert_mode, measure_mode, max_load, min_load, insert_prefetch, query_prefetch, filename);
}





int main(int argc, char** argv) {
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
	int map = ALL_TYPE;
	int dna = DNA_TYPE;
	bool canonical = false;
	bool full = false;

	  size_t count = 100000000;
	  size_t query_frac = 2;
	  size_t repeat_rate = 10;

	  int batch_mode = ITER_MODE;
	  int measure = MEASURE_INSERT;

	  double max_load = 0.8;
	  double min_load = 0.35;
	  uint8_t insert_prefetch = 8;
	  uint8_t query_prefetch = 8;

	  std::string fname;

	  std::tie(map, dna, full, canonical, count, query_frac, repeat_rate, batch_mode, measure, max_load, min_load, insert_prefetch, query_prefetch, fname) =
			  parse_cmdline(argc, argv);

  mxx::env e(argc, argv);
  mxx::comm comm;

  if (comm.rank() == 0) printf("EXECUTING %s\n", argv[0]);

  comm.barrier();


  using Kmer = ::bliss::common::Kmer<31, ::bliss::common::DNA, uint64_t>;
  using DNA5Kmer = ::bliss::common::Kmer<21, ::bliss::common::DNA5, uint64_t>;
  using FullKmer = ::bliss::common::Kmer<32, ::bliss::common::DNA, uint64_t>;
  using DNA16Kmer = ::bliss::common::Kmer<15, ::bliss::common::DNA16, uint64_t>;

  BL_BENCH_INIT(test);

  comm.barrier();

//  BL_BENCH_START(test);
//  benchmark_densehash_map<Kmer, size_t>("densehash_map_warmup", count, repeat_rate, query_frac, comm);
//  BL_BENCH_COLLECTIVE_END(test, "densehash_map_warmup", count, comm);

  // make the input

  std::cout << "USING:\tStoreHash=" << typeid(StoreHash<Kmer>).name() << std::endl;
  std::cout << "      \tStoreHash=" << typeid(StoreHash<DNA5Kmer>).name() << std::endl;
  std::cout << "      \tStoreHash=" << typeid(StoreHash<FullKmer>).name() << std::endl;
  std::cout << "      \tStoreHash=" << typeid(StoreHash<DNA16Kmer>).name() << std::endl;

#if (pStoreHash != MURMUR32sse) && (pStoreHash != MURMUR32avx) && (pStoreHash != MURMUR64avx)

  if ((map == STD_UNORDERED_TYPE) || (map == ALL_TYPE)) {
    BL_BENCH_INIT(test);

	  // ============ unordered map
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
				  benchmark_unordered_map("unordered_map_full",
					  get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, max_load, comm);

			  BL_BENCH_COLLECTIVE_END(test, "unordered_map_full", count, comm);
		  } else {
			  BL_BENCH_START(test);
				  benchmark_unordered_map("unordered_map_DNA",
					  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, max_load, comm);

			  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
			  benchmark_unordered_map("unordered_map_DNA5",
				  get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, max_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
			  benchmark_unordered_map("unordered_map_DNA16",
				  get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, max_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

  // ------------- unordered map
  }

  if (map == KMERIND_TYPE)   {
  // =============== dense hash map wrapped
    BL_BENCH_INIT(test);

	  if (dna == DNA_TYPE) {
		  if (full) {
			  if (canonical) {
				  BL_BENCH_START(test);
					  benchmark_densehash_full_map<true>("densehash_full_map_canonical",
						  get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, comm);
				  BL_BENCH_COLLECTIVE_END(test, "densehash_full_map_canonical", count, comm);
			  } else {
				  BL_BENCH_START(test);
				  benchmark_densehash_full_map<false>("densehash_full_map_noncanonical",
					  get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, comm);
				  BL_BENCH_COLLECTIVE_END(test, "densehash_full_map", count, comm);
			  }
		  } else {
			  BL_BENCH_START(test);
			  benchmark_densehash_map("densehash_map_DNA",
				  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_densehash_map("densehash_map_DNA5",
			  get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
			  query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_densehash_map("densehash_map_DNA16",
			  get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
			  query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

// ------------------- end dense hash map wrapped.
  }

  if ((map == GOOGLE_TYPE) || (map == ALL_TYPE))  {
    BL_BENCH_INIT(test);

  // =============== google dense hash map
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_google_densehash_map("benchmark_google_densehash_map_Full",
					  get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_google_densehash_map("benchmark_google_densehash_map_DNA",
					  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_google_densehash_map("benchmark_google_densehash_map_DNA5",
				  get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_google_densehash_map("benchmark_google_densehash_map_DNA16",
				  get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

  // --------------- end google
  }

  if ((map == LINEARPROBE_TYPE) || (map == ALL_TYPE))  {
  //================ my new hashmap
    BL_BENCH_INIT(test);

	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling>("hashmap_linearprobe_Full",
					  get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling>("hashmap_linearprobe_DNA",
					  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling>("hashmap_linearprobe_DNA5",
				  get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling>("hashmap_linearprobe_DNA16",
				  get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);


  // --------------- my new hashmap.
  }

  if ((map == CLASSIC_ROBINHOOD_TYPE) || (map == ALL_TYPE))  {
  //================ my new hashmap Robin hood
    BL_BENCH_INIT(test);

	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_classicRH("classic_robinhood_Full",
				  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "classic_robinhood_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_classicRH("classic_robinhood_DNA",
				  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "classic_robinhood_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_classicRH("classic_robinhood_DNA5",
			  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
			  query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "classic_robinhood_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_classicRH("classic_robinhood_DNA16",
			  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
			  query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "classic_robinhood_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

  }


  if ((map == ROBINHOOD_TYPE) || (map == ALL_TYPE))  {
  //================ my new hashmap Robin hood
    BL_BENCH_INIT(test);

	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap<::fsc::hashmap_robinhood_doubling>("hashmap_robinhood_Full",
				get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap<::fsc::hashmap_robinhood_doubling>("hashmap_robinhood_DNA",
					  get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap<::fsc::hashmap_robinhood_doubling>("hashmap_robinhood_DNA5",
				get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap<::fsc::hashmap_robinhood_doubling>("hashmap_robinhood_DNA16",
				get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

  }

  if ((map == ROBINHOOD_PREFETCH_TYPE) || (map == ALL_TYPE))  {
    BL_BENCH_INIT(test);


     //================ my new hashmap offsets
     if (dna == DNA_TYPE) {
       if (full) {
         BL_BENCH_START(test);
         benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_prefetch>("hashmap_robinhood_prefetch_Full",
         get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
             query_frac, batch_mode, measure,
             max_load, min_load, comm);
         BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_prefetch_Full", count, comm);
       } else {
         BL_BENCH_START(test);
         benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_prefetch>("hashmap_robinhood_prefetch_DNA",
         get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
             query_frac, batch_mode, measure,
             max_load, min_load, comm);
         BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_prefetch_DNA", count, comm);
       }
     } else if (dna == DNA5_TYPE) {
       BL_BENCH_START(test);
       benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_prefetch>("hashmap_robinhood_prefetch_DNA5",
         get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
           query_frac, batch_mode, measure,
           max_load, min_load, comm);
       BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_prefetch_DNA5", count, comm);
     } else if (dna == DNA16_TYPE) {
       BL_BENCH_START(test);
       benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_prefetch>("hashmap_robinhood_prefetch_DNA16",
         get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
           query_frac, batch_mode, measure,
           max_load, min_load, comm);
       BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_prefetch_DNA16", count, comm);
     } else {

       throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
     }
     BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

   }


  if (map == ROBINHOOD_OFFSET_TYPE)  {
    BL_BENCH_INIT(test);

	  //================ my new hashmap offsets
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_doubling_offsets>("hashmap_robinhood_offsets_Full",
				get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_doubling_offsets>("hashmap_robinhood_offsets_DNA",
				get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure,
					  max_load, min_load, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_doubling_offsets>("hashmap_robinhood_offsets_DNA5",
				get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap_insert_mode<::fsc::hashmap_robinhood_doubling_offsets>("hashmap_robinhood_offsets_DNA16",
				get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
				  query_frac, batch_mode, measure,
				  max_load, min_load, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
	  // --------------- my new hashmap.
	  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

	}



#endif

  if ((map == ROBINHOOD_OFFSET2_TYPE) || (map == ALL_TYPE))  {
    BL_BENCH_INIT(test);

	  //================ my new hashmap Robin hood
		  if (dna == DNA_TYPE) {
			  if (full) {
				  BL_BENCH_START(test);
				  benchmark_hashmap<::fsc::hashmap_robinhood_offsets>("hashmap_robinhood_offsets_nooverflow_Full",
				get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, batch_mode, measure, max_load, min_load, insert_prefetch, query_prefetch, comm);
				  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_nooverflow_Full", count, comm);
			  } else {
				  BL_BENCH_START(test);
				  benchmark_hashmap<::fsc::hashmap_robinhood_offsets>("hashmap_robinhood_offsets_nooverflow_DNA",
				get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, batch_mode, measure, max_load, min_load, insert_prefetch, query_prefetch, comm);
				  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_nooverflow_DNA", count, comm);
			  }
		  } else if (dna == DNA5_TYPE) {
			  BL_BENCH_START(test);
			  benchmark_hashmap<::fsc::hashmap_robinhood_offsets>("hashmap_robinhood_offsets_nooverflow_DNA5",
				get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure, max_load, min_load, insert_prefetch, query_prefetch, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_nooverflow_DNA5", count, comm);
		  } else if (dna == DNA16_TYPE) {
			  BL_BENCH_START(test);
			  benchmark_hashmap<::fsc::hashmap_robinhood_offsets>("hashmap_robinhood_offsets_nooverflow_DNA16",
				get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
					  query_frac, batch_mode, measure, max_load, min_load, insert_prefetch, query_prefetch, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_offsets_nooverflow_DNA16", count, comm);
		  } else {

			  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
		  }
		  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

	}
  if ((map == RADIXSORT_TYPE) || (map == ALL_TYPE))  {
    BL_BENCH_INIT(test);

	  //================ new hashmap Radixsort
		  if (dna == DNA_TYPE) {
			  if (full) {
				  BL_BENCH_START(test);
				  benchmark_hashmap_radixsort("hashmap_radixsort",
				get_input<FullKmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, measure, comm);
				  BL_BENCH_COLLECTIVE_END(test, "hashmap_radixsort", count, comm);
			  } else {
				  BL_BENCH_START(test);
				  benchmark_hashmap_radixsort("hashmap_radixsort",
				get_input<Kmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, measure, comm);
				  BL_BENCH_COLLECTIVE_END(test, "hashmap_radixsort", count, comm);
			  }
		  } else if (dna == DNA5_TYPE) {
			  BL_BENCH_START(test);
			  benchmark_hashmap_radixsort("hashmap_radixsort",
				get_input<DNA5Kmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, measure, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_radixsort", count, comm);
		  } else if (dna == DNA16_TYPE) {
			  BL_BENCH_START(test);
			  benchmark_hashmap_radixsort("hashmap_radixsort",
				get_input<DNA16Kmer, CountType>(fname, count, repeat_rate, canonical),
						  query_frac, measure, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_radixsort", count, comm);
		  } else {

			  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
		  }
		  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

  }



}

