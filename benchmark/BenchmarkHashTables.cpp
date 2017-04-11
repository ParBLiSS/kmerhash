#include <unordered_map>
#include <vector>
#include <random>
#include <cstdint>

#include <tuple>
#include <string>
#include <exception>
#include <functional>

#if 0
#include <tommyds/tommyalloc.h>
#include <tommyds/tommyalloc.c>
#include <tommyds/tommyhashdyn.h>
#include <tommyds/tommyhashdyn.c>
#include <tommyds/tommyhashlin.h>
#include <tommyds/tommyhashlin.c>
#include <tommyds/tommytrie.h>
#include <tommyds/tommytrie.c>

#include "flat_hash_map/flat_hash_map.hpp"

#endif

#include "containers/unordered_vecmap.hpp"
//#include "containers/hashed_vecmap.hpp"
#include "containers/densehash_map.hpp"

#include "kmerhash/hashmap_linearprobe_doubling.hpp"
#include "kmerhash/hashmap_robinhood_doubling.hpp"
// experimental
#include "kmerhash/experimental/hashmap_robinhood_doubling_noncircular.hpp"
//#include "kmerhash/experimental/hashmap_robinhood_doubling_memmove.hpp"
#include "kmerhash/experimental/hashmap_robinhood_doubling_offsets2.hpp"

#include "common/kmer.hpp"
#include "common/kmer_transform.hpp"
#include "index/kmer_hash.hpp"

#include "tclap/CmdLine.h"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

#include "utils/benchmark_utils.hpp"
#include "utils/transform_utils.hpp"

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



template <typename Kmer, typename Value>
void generate_input(std::vector<::std::pair<Kmer, Value> > & output, size_t const count, size_t const repeats = 10, bool canonical = false) {
  output.resize(count);

  size_t freq;

  srand(23);
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < Kmer::nWords; ++j) {
      output[i].first.getDataRef()[j] = static_cast<typename Kmer::KmerWordType>(static_cast<long>(rand()) << 32) ^ static_cast<long>(rand());
    }
    output[i].first.sanitize();
    //output[i].second = static_cast<Value>(static_cast<long>(rand()) << 32) ^ static_cast<long>(rand());
    output[i].second = i;

    // average repeat/2 times inserted.
    freq = rand() % repeats;

    for (size_t j = 0; j < freq; ++j) {
    	if (i+1 < count) {
    		++i;

            output[i] = output[i-1];
            //output[i].second = static_cast<Value>(static_cast<long>(rand()) << 32) ^ static_cast<long>(rand());
            output[i].second = i;
    	}
    }

  }

  if (canonical) {
	  for (size_t i = 0; i < output.size(); ++i) {
		  Kmer revcomp = output[i].first.reverse_complement();
		  if (revcomp < output[i].first) output[i].first = revcomp;
	  }
  }

  // do random shuffling to avoid consecutively identical items.
  std::random_shuffle(output.begin(), output.end());
}

template <typename Kmer, typename Value>
void benchmark_unordered_map(std::string name, size_t const count, size_t const repeat_rate, size_t const query_frac, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved
  ::std::unordered_map<Kmer, Value, ::bliss::kmer::hash::farm<Kmer, false> > map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);

  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }


  BL_BENCH_START(map);
  size_t result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    auto iters = map.equal_range(query[i]);
    for (auto it = iters.first; it != iters.second; ++it)
      result ^= it->second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count", result);


  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.erase(query[i]);
  }
  BL_BENCH_END(map, "erase", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}


template <typename Kmer, typename Value>
void benchmark_densehash_map(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::fsc::densehash_map<Kmer, Value, 
	::bliss::kmer::hash::sparsehash::special_keys<Kmer, false>,
	::bliss::transform::identity,
	::bliss::kmer::hash::farm<Kmer, false> > map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);



  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    auto iters = map.equal_range(query[i]);
    for (auto it = iters.first; it != iters.second; ++it)
      result ^= it->second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
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
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count2", result);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}


template <typename Kmer, typename Value, bool canonical = false>
void benchmark_densehash_full_map(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::fsc::densehash_map<Kmer, Value, 
	::bliss::kmer::hash::sparsehash::special_keys<Kmer, canonical>,
	::bliss::transform::identity,
	::bliss::kmer::hash::farm<Kmer, false> > map(count * 2 / repeat_rate);

  BL_BENCH_END(map, "reserve", count);



  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate, canonical);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    auto iters = map.equal_range(query[i]);
    for (auto it = iters.first; it != iters.second; ++it)
      result ^= (*it).second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
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
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count2", result);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}



#if 0
// cannot get it to compile
template <typename Kmer, typename Value>
void benchmark_flat_hash_map(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::ska::flat_hash_map<Kmer, Value,
	::bliss::kmer::hash::farm<Kmer, false> > map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);

  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  size_t i = 0;
  size_t max = count / query_frac;
  for (; i < max; ++i) {
    auto iter = map.find(query[i]);
    result ^= (*iter).second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
//  result = map.erase(query.begin(), query.end());

  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.erase(query[i]);
  }
//  map.resize(0);
  BL_BENCH_END(map, "erase", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count2", result);

  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}
#endif


template <typename Kmer, typename Value>
void benchmark_google_densehash_map(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  ::google::dense_hash_map<Kmer, Value,
	::bliss::kmer::hash::farm<Kmer, false> > map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);


  map.max_load_factor(0.6);
  map.min_load_factor(0.2);

  ::bliss::kmer::hash::sparsehash::special_keys<Kmer, false> special;

  map.set_empty_key(special.generate(0));
  map.set_deleted_key(special.generate(1));

  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    BL_BENCH_START(map);
    map.insert(input.begin(), input.end());
    BL_BENCH_END(map, "insert", map.size());
  }

  BL_BENCH_START(map);
  size_t result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    auto iters = map.equal_range(query[i]);
    for (auto it = iters.first; it != iters.second; ++it)
      result ^= it->second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  //result = map.erase(query.begin(), query.end());
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.erase(query[i]);
  }
  map.resize(0);
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
  result = 0;
  for (size_t i = 0, max = count / query_frac; i < max; ++i) {
    result += map.count(query[i]);
  }
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}

#define ITER_MODE 1
#define INDEX_MODE 2
#define INTEGRATED_MODE 3
#define SORT_MODE 4
#define SHUFFLE_MODE 5


template <template <typename, typename, typename, typename, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap_insert_mode(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, int vector_mode, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, ::bliss::kmer::hash::farm<Kmer, false>, ::std::equal_to<Kmer>, ::std::allocator<std::pair<Kmer, Value> > >;
  MAP_TYPE map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);

  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    if (vector_mode == ITER_MODE) {
        BL_BENCH_START(map);
    	map.insert(input.begin(), input.end());
        BL_BENCH_END(map, "insert", map.size());
    } else if (vector_mode == INDEX_MODE) {
        BL_BENCH_START(map);
    	map.insert(std::move(input));
        BL_BENCH_END(map, "v_insert", map.size());

    } else if (vector_mode == INTEGRATED_MODE) {
        BL_BENCH_START(map);
    	map.insert_integrated(std::move(input));
        BL_BENCH_END(map, "insert_integrated", map.size());

    } else if (vector_mode == SORT_MODE) {
        BL_BENCH_START(map);
    	map.insert_sort(std::move(input));
        BL_BENCH_END(map, "insert_integrated", map.size());
    } else {
        BL_BENCH_START(map);
    	map.insert(input.begin(), input.end());
        BL_BENCH_END(map, "insert", map.size());
    }
  }

  BL_BENCH_START(map);
  size_t result = 0;
  size_t i = 0;
  size_t max = count / query_frac;
  for (; i < max; ++i) {
    auto iter = map.find(query[i]);
    result ^= (*iter).second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  auto counts = map.count(query.begin(), query.end());
  result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  result = map.erase(query.begin(), query.end());
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
  counts = map.count(query.begin(), query.end());
  result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}



template <template <typename, typename, typename, typename, typename> class MAP,
typename Kmer, typename Value>
void benchmark_hashmap(std::string name, size_t const count,  size_t const repeat_rate, size_t const query_frac, int vector_mode, ::mxx::comm const & comm) {
  BL_BENCH_INIT(map);

  std::vector<Kmer> query;

  BL_BENCH_START(map);
  // no transform involved.
  using MAP_TYPE = MAP<Kmer, Value, ::bliss::kmer::hash::farm<Kmer, false>, ::std::equal_to<Kmer>, ::std::allocator<std::pair<Kmer, Value> > >;
  MAP_TYPE map(count * 2 / repeat_rate);
  BL_BENCH_END(map, "reserve", count);

  {
    BL_BENCH_START(map);
    std::vector<::std::pair<Kmer, Value> > input(count);
    BL_BENCH_END(map, "reserve input", count);

    BL_BENCH_START(map);
    generate_input(input, count, repeat_rate);
    query.resize(count / query_frac);
    std::transform(input.begin(), input.begin() + input.size() / query_frac, query.begin(),
                   [](::std::pair<Kmer, Value> const & x){
      return x.first;
    });
    BL_BENCH_END(map, "generate input", input.size());

    if (vector_mode == INDEX_MODE) {
        BL_BENCH_START(map);
    	map.insert(std::move(input));
        BL_BENCH_END(map, "v_insert", map.size());

    } else {
        BL_BENCH_START(map);
    	map.insert(input.begin(), input.end());
        BL_BENCH_END(map, "insert", map.size());
    }
  }

  BL_BENCH_START(map);
  size_t result = 0;
  size_t i = 0;
  size_t max = count / query_frac;
  for (; i < max; ++i) {
    auto iter = map.find(query[i]);
    result ^= (*iter).second;
  }
  BL_BENCH_END(map, "find", result);

  BL_BENCH_START(map);
  auto counts = map.count(query.begin(), query.end());
  result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
  BL_BENCH_END(map, "count", result);

  BL_BENCH_START(map);
  result = map.erase(query.begin(), query.end());
  BL_BENCH_END(map, "erase", result);


  BL_BENCH_START(map);
  counts = map.count(query.begin(), query.end());
  result = std::accumulate(counts.begin(), counts.end(), static_cast<size_t>(0));
  BL_BENCH_END(map, "count2", result);


  BL_BENCH_REPORT_MPI_NAMED(map, name, comm);
}







#define STD_UNORDERED_TYPE 1
#define GOOGLE_TYPE 2
#define KMERIND_TYPE 3
#define LINEARPROBE_TYPE 4
#define ROBINHOOD_TYPE 5
#define ROBINHOOD_NONCIRC_TYPE 6
#define ROBINHOOD_OFFSET_TYPE 7

#define DNA_TYPE 1
#define DNA5_TYPE 2
#define DNA16_TYPE 3




/// parse the parameters.  return int map type, int DNA type, bool full, and bool canonical
/// size_t, query frac, repeat rate.  last bool is vector mode (input is vector, not iterator.)
std::tuple<int, int, bool, bool, size_t, size_t, size_t, int> parse_cmdline(int argc, char** argv) {

	int map = ROBINHOOD_TYPE;
	int dna = DNA_TYPE;
	bool canonical = false;
	bool full = false;

	  size_t count = 100000000;
	//  size_t count = 100;
	  size_t query_frac = 10;
	  size_t repeat_rate = 10;

	  int insert_mode = INDEX_MODE;

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
	  allowed.push_back("robinhood");
	  allowed.push_back("robinhood_noncirc");
	  allowed.push_back("robinhood_offset");
	  TCLAP::ValuesConstraint<std::string> allowedVals( allowed );

	  TCLAP::ValueArg<std::string> mapArg("m","map_type","type of map to use (default robinhood)",false,"robinhood",&allowedVals, cmd);

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
	  TCLAP::SwitchArg fullArg("F", "full", "set k-mer to fully occupy machine word", cmd, full);
	  TCLAP::SwitchArg canonicalArg("C", "canonical", "use canonical k-mers", cmd, canonical);


	  TCLAP::ValueArg<size_t> countArg("N","num_elements","number of elements", false, count, "size_t", cmd);
	  TCLAP::ValueArg<size_t> queryArg("q","query_fraction","percent of count to use for query", false, query_frac, "size_t", cmd);
	  TCLAP::ValueArg<size_t> repeatArg("R","repeate_rate","maximum number of repeats in data", false, repeat_rate, "size_t", cmd);


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
	  } else if (map_type == "robinhood") {
		  map = ROBINHOOD_TYPE;
	  } else if (map_type == "robinhood_noncirc") {
		  map = ROBINHOOD_NONCIRC_TYPE;
	  } else if (map_type == "robinhood_offset") {
		  map = ROBINHOOD_OFFSET_TYPE;
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

	  // Do what you intend.

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
	  std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	  exit(-1);
	}

	return std::make_tuple(map, dna, full, canonical, count, query_frac, repeat_rate, insert_mode);
}



int main(int argc, char** argv) {

	int map = ROBINHOOD_TYPE;
	int dna = DNA_TYPE;
	bool canonical = false;
	bool full = false;

	  size_t count = 100000000;
	  size_t query_frac = 10;
	  size_t repeat_rate = 10;

	  int batch_mode = INDEX_MODE;

	  std::tie(map, dna, full, canonical, count, query_frac, repeat_rate, batch_mode) = parse_cmdline(argc, argv);

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


  if (map == STD_UNORDERED_TYPE) {

	  // ============ unordered map
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_unordered_map<FullKmer, size_t>("unordered_map_full", count, repeat_rate, query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "unordered_map_full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_unordered_map<Kmer, size_t>("unordered_map_DNA", count, repeat_rate, query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_unordered_map<DNA5Kmer, size_t>("unordered_map_DNA5", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_unordered_map<DNA16Kmer, size_t>("unordered_map_DNA16", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "unordered_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
  // ------------- unordered map
  } else if (map == KMERIND_TYPE) {
  // =============== dense hash map wrapped

	  if (dna == DNA_TYPE) {
		  if (full) {
			  if (canonical) {
				  BL_BENCH_START(test);
				  benchmark_densehash_full_map<FullKmer, size_t, true>("densehash_full_map_canonical", count, repeat_rate, query_frac, comm);
				  BL_BENCH_COLLECTIVE_END(test, "densehash_full_map_canonical", count, comm);
			  } else {
				  BL_BENCH_START(test);
				  benchmark_densehash_full_map<FullKmer, size_t, false>("densehash_full_map", count, repeat_rate, query_frac, comm);
				  BL_BENCH_COLLECTIVE_END(test, "densehash_full_map", count, comm);
			  }
		  } else {
			  BL_BENCH_START(test);
			  benchmark_densehash_map<Kmer, size_t>("densehash_map_DNA", count, repeat_rate, query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_densehash_map<DNA5Kmer, size_t>("densehash_map_DNA5", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_densehash_map<DNA16Kmer, size_t>("densehash_map_DNA16", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "densehash_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }

// ------------------- end dense hash map wrapped.
  } else if (map == GOOGLE_TYPE) {

  // =============== google dense hash map
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_google_densehash_map<FullKmer, size_t>("benchmark_google_densehash_map_Full", count, repeat_rate, query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_google_densehash_map<Kmer, size_t>("benchmark_google_densehash_map_DNA", count, repeat_rate, query_frac, comm);
			  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_google_densehash_map<DNA5Kmer, size_t>("benchmark_google_densehash_map_DNA5", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_google_densehash_map<DNA16Kmer, size_t>("benchmark_google_densehash_map_DNA16", count, repeat_rate, query_frac, comm);
		  BL_BENCH_COLLECTIVE_END(test, "benchmark_google_densehash_map_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }

  // --------------- end google
  } else if (map == LINEARPROBE_TYPE) {
  //================ my new hashmap
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling, FullKmer, size_t>("hashmap_linearprobe_doubling_Full", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_doubling_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling, Kmer, size_t>("hashmap_linearprobe_doubling_DNA", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_doubling_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling, DNA5Kmer, size_t>("hashmap_linearprobe_doubling_DNA5", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_doubling_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_linearprobe_doubling, DNA16Kmer, size_t>("hashmap_linearprobe_doubling_DNA16", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_linearprobe_doubling_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }


  // --------------- my new hashmap.
  } else if (map == ROBINHOOD_TYPE) {
  //================ my new hashmap Robin hood
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap_insert_mode< ::fsc::hashmap_robinhood_doubling, FullKmer, size_t>("hashmap_robinhood_doubling_Full", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap_insert_mode< ::fsc::hashmap_robinhood_doubling, Kmer, size_t>("hashmap_robinhood_doubling_DNA", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap_insert_mode< ::fsc::hashmap_robinhood_doubling, DNA5Kmer, size_t>("hashmap_robinhood_doubling_DNA5", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap_insert_mode< ::fsc::hashmap_robinhood_doubling, DNA16Kmer, size_t>("hashmap_robinhood_doubling_DNA16", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }

  } else if (map == ROBINHOOD_NONCIRC_TYPE) {

	    // experimental...
	    //================ my new hashmap non_circular
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_noncircular, FullKmer, size_t>("hashmap_robinhood_doubling_noncirc_Full", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_noncirc_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_noncircular, Kmer, size_t>("hashmap_robinhood_doubling_noncirc_DNA", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_noncirc_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_noncircular, DNA5Kmer, size_t>("hashmap_robinhood_doubling_noncirc_DNA5", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_noncirc_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_noncircular, DNA16Kmer, size_t>("hashmap_robinhood_doubling_noncirc_DNA16", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_noncirc_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }


  } else if (map == ROBINHOOD_OFFSET_TYPE) {

	  //================ my new hashmap offsets
	  if (dna == DNA_TYPE) {
		  if (full) {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_offsets, FullKmer, size_t>("hashmap_robinhood_doubling_offsets_Full", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_offsets_Full", count, comm);
		  } else {
			  BL_BENCH_START(test);
			  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_offsets, Kmer, size_t>("hashmap_robinhood_doubling_offsets_DNA", count, repeat_rate, query_frac, batch_mode, comm);
			  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_offsets_DNA", count, comm);
		  }
	  } else if (dna == DNA5_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_offsets, DNA5Kmer, size_t>("hashmap_robinhood_doubling_offsets_DNA5", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_offsets_DNA5", count, comm);
	  } else if (dna == DNA16_TYPE) {
		  BL_BENCH_START(test);
		  benchmark_hashmap< ::fsc::hashmap_robinhood_doubling_offsets, DNA16Kmer, size_t>("hashmap_robinhood_doubling_offsets_DNA16", count, repeat_rate, query_frac, batch_mode, comm);
		  BL_BENCH_COLLECTIVE_END(test, "hashmap_robinhood_doubling_offsets_DNA16", count, comm);
	  } else {

		  throw std::invalid_argument("UNSUPPORTED ALPHABET TYPE");
	  }
  } else {
	  throw std::invalid_argument("UNSUPPORTED MAP TYPE");
  }



#if 0
  // ============ flat_hash_map  not compiling.
  // doesn't compile.  it's using initializer lists extensively, and the templated_iterator is having trouble constructing from initializer list.
  BL_BENCH_START(test);
  benchmark_flat_hash_map<Kmer, size_t>("flat_hash_map_DNA", count, repeat_rate, query_frac, comm);
  BL_BENCH_COLLECTIVE_END(test, "flat_hash_map_DNA", count, comm);

  BL_BENCH_START(test);
  benchmark_flat_hash_map<DNA5Kmer, size_t>("flat_hash_map_DNA5", count, repeat_rate, query_frac, comm);
  BL_BENCH_COLLECTIVE_END(test, "flat_hash_map_DNA5", count, comm);

  BL_BENCH_START(test);
  benchmark_flat_hash_map<DNA16Kmer, size_t>("flat_hash_map_DNA16", count, repeat_rate, query_frac, comm);
  BL_BENCH_COLLECTIVE_END(test, "flat_hash_map_DNA16", count, comm);

  BL_BENCH_START(test);
  benchmark_flat_hash_map<FullKmer, size_t>("flat_hash_map_Full", count, repeat_rate, query_frac, comm);
  BL_BENCH_COLLECTIVE_END(test, "flat_hash_map_Full", count, comm);

  // -------------flat hash map end
#endif


  BL_BENCH_REPORT_MPI_NAMED(test, "hashmaps", comm);

}

