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

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <tuple>      // pair, tuple
#include <functional> // std::hash
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream> // for system("pause");
#include <cstring>  // memcpy
#include <stdlib.h> // aligned_alloc

#include "utils/logging.h"

#include "iterators/transform_iterator.hpp"
#include "iterators/zip_iterator.hpp"

#include "index/kmer_hash.hpp" // needed by distributed_densehash_map.hpp

#include "io/file.hpp"
#include "kmerhash/hash_new.hpp"
#include "kmerhash/io_utils.hpp"

#include <unordered_map>
#include "containers/distributed_densehash_map.hpp"
#include "containers/distributed_unordered_map.hpp"
#include "kmerhash/distributed_robinhood_map.hpp"
#include "kmerhash/distributed_batched_robinhood_map.hpp"
#include "kmerhash/distributed_batched_radixsort_map.hpp"
#include "kmerhash/hybrid_batched_robinhood_map.hpp"
#include "kmerhash/hybrid_batched_radixsort_map.hpp"

#include "utils/benchmark_utils.hpp"

#include "tclap/CmdLine.h"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"
#include "mxx/bcast.hpp"

// ================ define preproc macro constants
// needed as #if can only calculate constant int expressions

#define IDEN 10
#define STD 21
#define FARM 22
#define FARM32 23
#define MURMUR 24
#define MURMUR32 25
#define MURMUR32sse 26
#define MURMUR32avx 27
#define MURMUR32FINALIZERavx 20
#define MURMUR64avx 28
#define CRC32C 29
#define CLHASH 30

#define COUNT 33
#define FIRST 34
#define LAST 35

#define UNORDERED 46
#define DENSEHASH 47
// #define ROBINHOOD 48
#define BROBINHOOD 49
#define RADIXSORT 50
#define MTROBINHOOD 51
#define MTRADIXSORT 52

//================= define types - changeable here...

#if (pBits == 64)
    using KeyType = uint64_t;
    using ValType = uint32_t;
#else
    using KeyType = uint32_t;
    using ValType = uint32_t;
#endif

//============== MAP properties
template <typename KM>
using DistTrans = bliss::transform::identity<KM>;

//----- get them all. may not use subsequently.

// distribution hash
#if (pDistHash == STD)
template <typename KM>
using DistHash = std::hash<KM>;
#elif (pDistHash == IDEN)
template <typename KM>
using DistHash = ::fsc::hash::identity<KM>;
#elif (pDistHash == MURMUR)
template <typename KM>
using DistHash = ::fsc::hash::murmur<KM>;
#elif (pDistHash == MURMUR32)
template <typename KM>
using DistHash = ::fsc::hash::murmur32<KM>;
#elif (pDistHash == CRC32C)
template <typename KM>
using DistHash = ::fsc::hash::crc32c<KM>;
#elif (pDistHash == CLHASH)
template <typename KM>
using DistHash = ::fsc::hash::clhash<KM>;
#elif (pDistHash == FARM)
template <typename KM>
using DistHash = ::fsc::hash::farm<KM>;
#elif (pDistHash == FARM32)
template <typename KM>
using DistHash = ::fsc::hash::farm32<KM>;
#elif (pDistHash == MURMUR32sse)
template <typename KM>
using DistHash = ::fsc::hash::murmur3sse32<KM>;
#elif (pDistHash == MURMUR32avx)
template <typename KM>
using DistHash = ::fsc::hash::murmur3avx32<KM>;
#elif (pDistHash == MURMUR32FINALIZERavx)
template <typename KM>
using DistHash = ::fsc::hash::murmur3finalizer_avx32<KM>;
#elif (pDistHash == MURMUR64avx)
template <typename KM>
using DistHash = ::fsc::hash::murmur3avx64<KM>;
#else
static_assert(false, "unsupported distr hash function");
#endif

// storage hash type
#if (pStoreHash == STD)
template <typename KM>
using StoreHash = std::hash<KM>;
#elif (pStoreHash == IDEN)
template <typename KM>
using StoreHash = ::fsc::hash::identity<KM>;
#elif (pStoreHash == MURMUR)
template <typename KM>
using StoreHash = ::fsc::hash::murmur<KM>;
#elif (pStoreHash == MURMUR32)
template <typename KM>
using StoreHash = ::fsc::hash::murmur32<KM>;
#elif (pStoreHash == CRC32C)
template <typename KM>
using StoreHash = ::fsc::hash::crc32c<KM>;
#elif (pStoreHash == CLHASH)
template <typename KM>
using StoreHash = ::fsc::hash::clhash<KM>;
#elif (pStoreHash == FARM)
template <typename KM>
using StoreHash = ::fsc::hash::farm<KM>;
#elif (pStoreHash == FARM32)
template <typename KM>
using StoreHash = ::fsc::hash::farm32<KM>;
#elif (pStoreHash == MURMUR32sse)
template <typename KM>
using StoreHash = ::fsc::hash::murmur3sse32<KM>;
#elif (pStoreHash == MURMUR32avx)
template <typename KM>
using StoreHash = ::fsc::hash::murmur3avx32<KM>;
#elif (pStoreHash == MURMUR32FINALIZERavx)
template <typename KM>
using StoreHash = ::fsc::hash::murmur3finalizer_avx32<KM>;
#elif (pStoreHash == MURMUR64avx)
template <typename KM>
using StoreHash = ::fsc::hash::murmur3avx64<KM>;
#else
static_assert(false, "Unsupported store hash function");
#endif

// ==== define Map parameter

template <typename Key>
using MapParams = ::dsc::HashMapParams<Key,
                                       bliss::transform::identity,
                                       bliss::transform::identity,
                                       DistHash,
                                       std::equal_to,
                                       bliss::transform::identity,
                                       StoreHash,
                                       std::equal_to>;
//using SpecialKeys = ::bliss::kmer::hash::sparsehash::special_keys<KeyType, false>;

// special keys definition
/// base class to store some convenience functions
template <typename T>
class special_keys
{

    static_assert(::std::is_fundamental<T>::value, "only support fundamental key types");

  protected:
    static constexpr T max = ~(static_cast<T>(0));

  public:
    // used to generate new empty and deleted keys when splitting.
    inline constexpr T invert(T const &x)
    {
        return ~x; /// 1111111111111 -> 000000000000,  11111111110 -> 000000000001
        // works with splitter of 100000000000
    }

    /// kmer empty key for DNA5  000000000010  or 0000000000101  - for use as empty and deleted.
    inline constexpr T generate(uint8_t id = 0)
    {
        return (id == 0) ? max : (max - 1);
    }

    inline constexpr T get_splitter()
    { // 10000000000000
        return ~(max >> 1);
    }

    // primitive type, all values possible, so need to split
    static constexpr bool need_to_split = true;
};

// // DEFINE THE MAP TYPE base on the type of data to be stored.
// #if (pINDEX == COUNT) // map
// #if (pMAP == UNORDERED)
// using MapType = ::dsc::counting_unordered_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == DENSEHASH)
// using MapType = ::dsc::counting_densehash_map<
//     KeyType, ValType, MapParams, special_keys<KeyType>>;
// #elif (pMAP == ROBINHOOD)
// using MapType = ::dsc::counting_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == BROBINHOOD)
// using MapType = ::dsc::counting_batched_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == RADIXSORT)
// using MapType = ::dsc::counting_batched_radixsort_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == MTROBINHOOD) // hybrid version
// using MapType = ::hsc::counting_batched_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == MTRADIXSORT) // hybrid version
// using MapType = ::hsc::counting_batched_radixsort_map<
//     KeyType, ValType, MapParams>;
// #endif
// #else
// #if (pINDEX == FIRST)
// using REDUC = ::fsc::DiscardReducer;
// #elif (pINDEX == LAST)
// using REDUC = ::fsc::ReplaceReducer;
// #else
// static_assert(false, "UNSUPPORTED REDUCTION TYPE");
// #endif

// #if (pMAP == UNORDERED)
// using MapType = ::dsc::reduction_unordered_map<
//     KeyType, ValType, MapParams, REDUC>;
// #elif (pMAP == DENSEHASH)
// using MapType = ::dsc::reduction_densehash_map<
//     KeyType, ValType, MapParams, special_keys<KeyType>, REDUC>;
// #elif (pMAP == ROBINHOOD)
// using MapType = ::dsc::reduction_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == BROBINHOOD)
// using MapType = ::dsc::reduction_batched_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == RADIXSORT)
// using MapType = ::dsc::reduction_batched_radixsort_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == MTROBINHOOD) // hybrid version
// using MapType = ::hsc::reduction_batched_robinhood_map<
//     KeyType, ValType, MapParams>;
// #elif (pMAP == MTRADIXSORT) // hybrid version
// using MapType = ::hsc::reduction_batched_radixsort_map<
//     KeyType, ValType, MapParams>;
// #endif
// #endif

// count is meant to be 2x as many as we need.  count is global
template <typename K>
void read_unique_keys(std::unordered_set<K> &output, size_t const &global_count, std::string filename, mxx::comm const &comm)
{

    // load the file
    size_t pSize = 0;
    std::pair<K, ValType> *input = nullptr;
    if (comm.rank() == 0)
    {
        // map the file.
        ::bliss::io::mmap_file fobj(filename);
        ::bliss::io::mapped_data partition = fobj.map(::bliss::io::base_file::range_type(0, fobj.size()));

        // get the size and compute the split.
        pSize = partition.size() / sizeof(std::pair<K, ValType>);
        input = reinterpret_cast<std::pair<K, ValType> *>(partition.get_data());

        // get unique by inserting into a map
        ::std::unordered_set<K> temp;
        temp.reserve(pSize);
        for (size_t i = 0; i < pSize; ++i)
        {
            temp.emplace(input[i].first);
        }

        // copy into a vector for communication
        std::vector<K> temp2(temp.begin(), temp.end());
        temp.clear();

        // broadcast size
        pSize = temp2.size() / comm.size();
        pSize = std::min(pSize, global_count / comm.size());
        if (pSize < (global_count / comm.size()))
            printf("WARN: input file has too few unique elements to produce %ld per process.  actual %ld\n", (global_count / comm.size()), pSize);

        mxx::bcast(pSize, 0, comm);

        // allocate temp storage.
        std::vector<K> tout(pSize);

        mxx::scatter(temp2.data(), pSize, tout.data(), 0, comm);

        output.clear();
        output.insert(tout.begin(), tout.end());
    }
    else
    {
        std::vector<K> temp2;
        mxx::bcast(pSize, 0, comm);
        std::vector<K> tout(pSize);

        mxx::scatter(temp2.data(), pSize, tout.data(), 0, comm);

        output.clear();
        output.insert(tout.begin(), tout.end());
    }
}

// count is meant to be 2x of what we need.,  count is global
template <typename K>
void generate_unique_keys(std::unordered_set<K> &output,
                          size_t const &global_count, mxx::comm const &comm, typename std::mt19937_64::result_type seed = 0)
{

    // obtain a seed from the system clock:
    if (seed == 0)
        seed = static_cast<typename std::mt19937_64::result_type>(std::chrono::system_clock::now().time_since_epoch().count());

    // seeds the random number engine, the mersenne_twister_engine
    std::mt19937 generator(seed);

    generator.discard(10000); // warm up.

    // set a distribution range (1 - 100)
    std::uniform_int_distribution<K> distribution;

    size_t pCount = global_count / comm.size();

    output.reserve(pCount);
    while (output.size() < pCount)
    {
        output.emplace(distribution(generator));
    }
}

// count here is GLOBAL
template <typename K>
void get_unique_keys(std::unordered_set<K> &uniques, ::std::string const &fname,
                     size_t const &global_count, mxx::comm const &comm)
{

    uniques.reserve((global_count << 1) / comm.size());

    // first get unique source
    if (fname.compare("") == 0)
    {
        generate_unique_keys(uniques, global_count << 1, comm);
    }
    else
    {
        read_unique_keys(uniques, global_count << 1, fname, comm);
    }
}

template <typename K, typename V>
void read_key_val_pairs(std::vector<std::pair<K, V>> &output, size_t const &global_count, std::string filename, mxx::comm const &comm)
{

    // load the file
    size_t pSize = 0;
    std::pair<K, V> *input = nullptr;
    if (comm.rank() == 0)
    {

        // map the file.
        ::bliss::io::mmap_file fobj(filename);
        ::bliss::io::mapped_data partition = fobj.map(fobj.file_range_bytes);

        // get the size and compute the split.
        pSize = partition.size() / sizeof(std::pair<K, V>) / comm.size();
        pSize = std::min(pSize, global_count / comm.size());
        if (pSize < (global_count / comm.size()))
            printf("input file has to few elements to produce %ld per process.  actual %ld\n", global_count, pSize);

        input = reinterpret_cast<std::pair<K, V> *>(partition.get_data());

        mxx::bcast(pSize, 0, comm);
        output.resize(pSize);
        mxx::scatter(input, pSize, output.data(), 0, comm);
    }
    else
    {
        mxx::bcast(pSize, 0, comm);
        output.resize(pSize);
        mxx::scatter(input, pSize, output.data(), 0, comm);
    }
}

template <typename K, typename V>
void generate_unique_key_val_pairs(std::vector<std::pair<K, V>> &output,
                                   std::unordered_set<K> const &uniques,
                                   size_t const &count, bool second = false)
{

    // check the counts
    size_t lcount = ::std::min(count, static_cast<size_t>(static_cast<float>(uniques.size() >> 1)));

    if (lcount < count)
        printf("too few elements in unique list to produce %ld per process.  actual %ld\n", count, lcount);

    // copy over, make copied as we go.
    output.clear();
    output.reserve(lcount);
    auto start = uniques.begin();
    if (second)
        std::advance(start, uniques.size() >> 1);

    for (size_t i = 0; i < lcount; ++i, ++start)
    {
        output.emplace_back(*start, i + 1);
    }
}

template <typename K, typename V>
void generate_key_val_pairs(std::vector<std::pair<K, V>> &output,
                            std::unordered_set<K> const &uniques,
                            size_t const &count,
                            float const &repeat_rate = 8.0, bool second = false,
                            typename std::mt19937_64::result_type seed = 0)
{

    // check the counts
    size_t lcount = ::std::min(count, static_cast<size_t>(static_cast<float>(uniques.size() >> 1) * repeat_rate));

    if (lcount < count)
        printf("too few elements in unique list to produce %ld per process.  actual %ld\n", count, lcount);

    // set up  the random number generator.
    if (seed == 0)
        seed = static_cast<typename std::mt19937_64::result_type>(std::chrono::system_clock::now().time_since_epoch().count());

    // seeds the random number engine, the mersenne_twister_engine
    std::mt19937 generator(seed);
    generator.discard(10000); // warm up.

    // set a distribution range (1 - 100)
    std::uniform_real_distribution<float> distribution(0.0, repeat_rate * 2.0);

    // copy over, make copied as we go.
    output.reserve(lcount + repeat_rate * 2.0);
    auto iter = uniques.begin();
    if (second)
        std::advance(iter, uniques.size() >> 1);
    ValType val = 1;
    size_t el_cnt = 0;
    while (output.size() < lcount)
    {
        el_cnt = static_cast<size_t>(distribution(generator)) + 1;

        // insert integer number of elements
        for (size_t j = 0; j < el_cnt; ++j, ++val)
        {
            output.emplace_back(*iter, val);
        }

        ++iter;
    }
    std::random_shuffle(output.begin(), output.end());
}

template <typename K>
void generate_unique_keys(std::vector<K> &output,
                          std::unordered_set<K> const &uniques,
                          size_t const &count, bool second = false)
{

    // check the counts
    size_t lcount = ::std::min(count, static_cast<size_t>(static_cast<float>(uniques.size() >> 1)));

    if (lcount < count)
        printf("too few elements in unique list to produce %ld per process.  actual %ld\n", count, lcount);

    // copy over, make copied as we go.
    output.clear();
    output.reserve(lcount);
    auto start = uniques.begin();
    if (second)
        std::advance(start, uniques.size() >> 1);

    for (size_t i = 0; i < lcount; ++i, ++start)
    {
        output.emplace_back(*start);
    }
}

template <typename K>
void generate_keys(std::vector<K> &output,
                   std::unordered_set<K> const &uniques,
                   size_t const &count,
                   float const &repeat_rate = 8.0, bool second = false,
                   typename std::mt19937_64::result_type seed = 0)
{

    // check the counts
    size_t lcount = ::std::min(count, static_cast<size_t>(static_cast<float>(uniques.size() >> 1) * repeat_rate));

    if (lcount < count)
        printf("too few elements in unique list to produce %ld per process.  actual %ld\n", count, lcount);

    // set up  the random number generator.
    if (seed == 0)
        seed = static_cast<typename std::mt19937_64::result_type>(std::chrono::system_clock::now().time_since_epoch().count());

    // seeds the random number engine, the mersenne_twister_engine
    std::mt19937 generator(seed);
    generator.discard(10000); // warm up.

    // set a distribution range (1 - 100)
    std::uniform_real_distribution<float> distribution(0.0, repeat_rate * 2.0);

    // copy over, make copied as we go.
    output.reserve(lcount + repeat_rate * 2.0);
    auto iter = uniques.begin();
    if (second)
        std::advance(iter, uniques.size() >> 1);
    size_t el_cnt = 0;
    while (output.size() < lcount)
    {
        el_cnt = static_cast<size_t>(distribution(generator)) + 1;

        // insert integer number of elements
        for (size_t j = 0; j < el_cnt; ++j)
        {
            output.emplace_back(*iter);
        }

        ++iter;
    }
    std::random_shuffle(output.begin(), output.end());
}

// assume count number for each of inserted and not_inserted.
// not_inserted should be generated te same way as inserted, but with "second" flag on.
template <typename K>
void get_query(
    std::vector<K> &query,
    std::vector<K> const &inserted,
    std::vector<K> const &not_inserted,
    float const &missing_frac = 0.0)
{

    size_t cnt = inserted.size();

    query.reserve(cnt);
    query.assign(not_inserted.begin(), not_inserted.begin() + cnt * missing_frac);

    query.insert(query.end(), inserted.begin(), inserted.begin() + (cnt - query.size()));

    std::random_shuffle(query.begin(), query.end());
}

template <typename MapType, int pMAP, typename TT>
void benchmark(std::vector<TT> const & input,
               std::vector<KeyType> const & query,
               std::string const &name,
               double const &max_load,
               double const &min_load,
               uint8_t const &insert_prefetch,
               uint8_t const &query_prefetch,
               mxx::comm const &comm)
{
    BL_BENCH_INIT(test);

    BL_BENCH_COLLECTIVE_START(test, "init", comm);
    // initialize the map.
    MapType map(comm);
#if (pMAP == UNORDERED)
    // std unordered.
    map.get_local_container().max_load_factor(max_load);
#elif (pMAP == DENSEHASH)
    // std unordered.
    map.get_local_container().max_load_factor(max_load);
#elif (pMAP == BROBINHOOD)
    map.get_local_container().set_max_load_factor(max_load);
    map.get_local_container().set_min_load_factor(min_load);
    map.get_local_container().set_insert_lookahead(insert_prefetch);
    map.get_local_container().set_query_lookahead(query_prefetch);
#elif (pMAP == MTROBINHOOD)
    map.set_max_load_factor(max_load);
    map.set_min_load_factor(min_load);
    map.set_insert_lookahead(insert_prefetch);
    map.set_query_lookahead(query_prefetch);
#endif
    BL_BENCH_END(test, "init", map.local_size());

    // debug print total input size.
    size_t total = mxx::allreduce(input.size(), comm);
    if (comm.rank() == 0)
        printf("total input size is %lu\n", total);

    // balance
    // if (balance_input)
    // {
    //     BL_BENCH_COLLECTIVE_START(test, "balance", comm);
    //     ::mxx::distribute_inplace(input, comm);
    //     BL_BENCH_END(test, "balance", input.size());
    // }

    BL_BENCH_COLLECTIVE_START(test, "copy_insert", comm);
    std::vector<TT> in(input.begin(), input.end());
    BL_BENCH_END(test, "copy_insert", in.size());

    // insert
    BL_BENCH_COLLECTIVE_START(test, "insert", comm);
    map.insert(in);
    BL_BENCH_END(test, "insert", map.local_size());

    // debug print total map size.
    total = map.size();
    if (comm.rank() == 0)
        printf("total map size after insert/rehash is %lu\n", total);

    // debug print total query size.
    total = mxx::allreduce(query.size(), comm);
    if (comm.rank() == 0)
        printf("total query size is %lu\n", total);

    { // count
        auto lquery = query;
#if (pMAP == MTROBINHOOD) || (pMAP == MTRADIXSORT) || (pMAP == BROBINHOOD) || (pMAP == RADIXSORT)
        BL_BENCH_COLLECTIVE_START(test, "count", comm);
        uint8_t *count_res = ::utils::mem::aligned_alloc<uint8_t>(lquery.size(), 64);
        size_t count_res_size = map.count(lquery, count_res);

        ::utils::mem::aligned_free(count_res);
        BL_BENCH_END(test, "count", count_res_size);
#else
        BL_BENCH_COLLECTIVE_START(test, "count", comm);
        auto counts = map.count(lquery);
        BL_BENCH_END(test, "count", counts.size());
#endif
    }

    { // find
        auto lquery = query;
#if (pMAP == MTROBINHOOD) || (pMAP == MTRADIXSORT) || (pMAP == BROBINHOOD) || (pMAP == RADIXSORT)
        BL_BENCH_COLLECTIVE_START(test, "find", comm);
        ValType *find_res = ::utils::mem::aligned_alloc<ValType>(lquery.size(), 64);
        size_t find_res_size = map.find(lquery, find_res);

        ::utils::mem::aligned_free(find_res);
        BL_BENCH_END(test, "find", find_res_size);
#else
        BL_BENCH_COLLECTIVE_START(test, "find", comm);
        auto found = map.find(lquery);
        BL_BENCH_END(test, "find", found.size());
#endif
    }

    { // find
        auto lquery = query;
        BL_BENCH_COLLECTIVE_START(test, "erase", comm);
        map.erase(lquery);
        BL_BENCH_END(test, "erase", map.local_size());
    }
    BL_BENCH_REPORT_MPI_NAMED(test, name, comm);
}

/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif

    //////////////// init logging
    LOG_INIT();

    //////////////// initialize MPI and openMP
    mxx::env e(argc, argv);
    mxx::comm comm;

    if (comm.rank() == 0)
        printf("EXECUTING %s\n", argv[0]);

    comm.barrier();

    //////////////// parse parameters

    std::string filename;
    size_t global_count = 10000000;
    double max_load = 0.8;
    double min_load = 0.35;
    uint8_t insert_prefetch = 8;
    uint8_t query_prefetch = 8;

    // bool balance_input = false;

    float repeats = 8.0;
    float missing_frac = 0.0;

    bool hybrid = false;

    // Wrap everything in a try block.  Do this every time,
    // because exceptions will be thrown for problems.
    try
    {

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
        TCLAP::ValueArg<std::string> fileArg("F", "file", "Key-Val binary file path", false, filename, "string", cmd);

        TCLAP::ValueArg<size_t> countArg("C", "count", "total insert/query element count", false, global_count, "size_t", cmd);
        TCLAP::ValueArg<float> repeatArg("R", "repeat-rate", "key-val pair repeat rate", false, repeats, "float", cmd);
        //    TCLAP::ValueArg<std::string> queryArg("Q", "query", "FASTQ file path for query. default to same file as index file", false, "", "string", cmd);

        // TCLAP::SwitchArg balanceArg("b", "balance-input", "balance the input", cmd, balance_input);
        TCLAP::SwitchArg hybridArg("", "hybrid", "OMP MPI hybrid hash tables", cmd, hybrid);
        TCLAP::ValueArg<float> missingArg("",
                                          "missing-frac", "fraction of query keys not in table. default=0.0 (all in)",
                                          false, missing_frac, "float", cmd);

        TCLAP::ValueArg<double> maxLoadArg("", "max_load", "maximum load factor", false, max_load, "double", cmd);
        TCLAP::ValueArg<double> minLoadArg("", "min_load", "minimum load factor", false, min_load, "double", cmd);
        TCLAP::ValueArg<uint32_t> insertPrefetchArg("", "insert_prefetch", "number of elements to prefetch during insert", false, insert_prefetch, "uint32_t", cmd);
        TCLAP::ValueArg<uint32_t> queryPrefetchArg("", "query_prefetch", "number of elements to prefetch during queries", false, query_prefetch, "uint32_t", cmd);

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
        TCLAP::ValuesConstraint<std::string> measureModeVals(measure_modes);
        TCLAP::ValueArg<std::string> measureModeArg("", "measured_op", "function to measure (default insert)", false, "insert", &measureModeVals, cmd);
#endif

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        filename = fileArg.getValue();
        global_count = countArg.getValue();
        repeats = repeatArg.getValue();

        // balance_input = balanceArg.getValue();
        hybrid = hybridArg.getValue();

        missing_frac = missingArg.getValue();

        min_load = minLoadArg.getValue();
        max_load = maxLoadArg.getValue();
        insert_prefetch = insertPrefetchArg.getValue();
        query_prefetch = queryPrefetchArg.getValue();

#ifdef VTUNE_ANALYSIS
        // set the default for query to filename, and reparse
        std::string measure_mode_str = measureModeArg.getValue();
        if (comm.rank() == 0)
            std::cout << "Measuring " << measure_mode_str << std::endl;

        if (measure_mode_str == "insert")
        {
            measure_mode = MEASURE_INSERT;
        }
        else if (measure_mode_str == "find")
        {
            measure_mode = MEASURE_FIND;
        }
        else if (measure_mode_str == "count")
        {
            measure_mode = MEASURE_COUNT;
        }
        else if (measure_mode_str == "erase")
        {
            measure_mode = MEASURE_ERASE;
        }
        else if (measure_mode_str == "reserve")
        {
            measure_mode = MEASURE_RESERVE;
        }
        else if (measure_mode_str == "transform")
        {
            measure_mode = MEASURE_TRANSFORM;
        }
        else if (measure_mode_str == "unique")
        {
            measure_mode = MEASURE_UNIQUE;
        }
        else if (measure_mode_str == "bucket")
        {
            measure_mode = MEASURE_BUCKET;
        }
        else if (measure_mode_str == "permute")
        {
            measure_mode = MEASURE_PERMUTE;
        }
        else if (measure_mode_str == "a2a")
        {
            measure_mode = MEASURE_A2A;
        }
        else
        {
            measure_mode = MEASURE_DISABLED;
        }
#endif
    }
    catch (TCLAP::ArgException &e) // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        exit(-1);
    }

    if (comm.rank() == 0)
    {
        std::cout << "USING:\tKeyType=" << typeid(KeyType).name() << std::endl;
        std::cout << "      \tValType=" << typeid(ValType).name() << std::endl;
        std::cout << "      \tDistHash=" << typeid(DistHash<KeyType>).name() << std::endl;
        std::cout << "      \tStoreHash=" << typeid(StoreHash<KeyType>).name() << std::endl;
    }

    // ================  make input
    BL_BENCH_INIT(data);

    size_t count = global_count / comm.size();
#if (pINDEX == COUNT)
    ::std::vector<KeyType> input;
#else
    ::std::vector<std::pair<KeyType, ValType>> input;
#endif
    ::std::vector<KeyType> query;
    std::unordered_set<KeyType> unique_keys;

    {
        BL_BENCH_COLLECTIVE_START(data, "get_keys", comm);
        get_unique_keys(unique_keys, filename, global_count, comm);
        BL_BENCH_END(data, "get_keys", unique_keys.size());

#if (pINDEX == COUNT)
        BL_BENCH_COLLECTIVE_START(data, "get_input", comm);
        if (repeats == 1.0)
            generate_unique_keys(input, unique_keys, count, false);
        else
            generate_keys(input, unique_keys, count, repeats, false);
        BL_BENCH_END(data, "get_input", input.size());
#else
        BL_BENCH_COLLECTIVE_START(data, "get_keyvals", comm);
        if (repeats == 1.0)
            generate_unique_key_val_pairs(input, unique_keys, count, false);
        else
            generate_key_val_pairs(input, unique_keys, count, repeats, false);
        BL_BENCH_END(data, "get_keyvals", input.size());
#endif
    }

    {
#if (pINDEX != COUNT)
        // copy the keys if needed.
        BL_BENCH_COLLECTIVE_START(data, "get_inserted", comm);
        std::vector<KeyType> inserted;
        inserted.reserve(input.size());
        for (auto x : input)
        {
            inserted.emplace_back(x.first);
        }
        BL_BENCH_END(data, "get_inserted", inserted.size());
#endif
        // generate the elements that are "missing"
        BL_BENCH_COLLECTIVE_START(data, "get_missing", comm);
        ::std::vector<KeyType> missing;
        if (repeats == 1.0)
            generate_unique_keys(missing, unique_keys, count * missing_frac + 1, true);
        else
            generate_keys(missing, unique_keys, count * missing_frac + 1, repeats, true);
        BL_BENCH_END(data, "get_missing", missing.size());

        // combine to get the query vector.
        BL_BENCH_COLLECTIVE_START(data, "get_query", comm);
#if (pINDEX == COUNT)
        get_query(query, input, missing, missing_frac);
#else
        get_query(query, inserted, missing, missing_frac);
#endif
        BL_BENCH_END(data, "get_query", query.size());
    }

    BL_BENCH_REPORT_MPI_NAMED(data, "gen_data", comm);

if (hybrid) {
#if (pINDEX == COUNT) // map
    benchmark<::hsc::counting_batched_robinhood_map<KeyType, ValType, MapParams>, MTROBINHOOD>(input, query, "hybrid_robinhood_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::hsc::counting_batched_radixsort_map<KeyType, ValType, MapParams>, MTRADIXSORT>(input, query, "hybrid_radixsort_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == FIRST)
    using REDUC = ::fsc::DiscardReducer;
    benchmark<::hsc::reduction_batched_robinhood_map<KeyType, ValType, MapParams, REDUC>, MTROBINHOOD>(input, query, "hybrid_robinhood_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::hsc::reduction_batched_radixsort_map<KeyType, ValType, MapParams, REDUC>, MTRADIXSORT>(input, query, "hybrid_radixsort_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == LAST)
    using REDUC = ::fsc::ReplaceReducer;
    benchmark<::hsc::reduction_batched_robinhood_map<KeyType, ValType, MapParams, REDUC>, MTROBINHOOD>(input, query, "hybrid_robinhood_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::hsc::reduction_batched_radixsort_map<KeyType, ValType, MapParams, REDUC>, MTRADIXSORT>(input, query, "hybrid_radixsort_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
#else
    static_assert(false, "UNSUPPORTED REDUCTION TYPE");
#endif // pINDEX

} else {  // not hybrid
    // now run the experiments.
#if (pDistHash != MURMUR32sse) && (pDistHash != MURMUR32avx) && (pDistHash != MURMUR32FINALIZERavx) && (pDistHash != MURMUR64avx) && (pStoreHash != MURMUR32sse) && (pStoreHash != MURMUR32avx) && (pStoreHash != MURMUR32FINALIZERavx) && (pStoreHash != MURMUR64avx)
#if (pINDEX == COUNT) // map
    benchmark<::dsc::counting_unordered_map<KeyType, ValType, MapParams>, UNORDERED>(input, query, "std::unordered_map_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::counting_densehash_map<KeyType, ValType, MapParams, special_keys<KeyType>>, DENSEHASH>(input, query, "google::densehash_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == FIRST)
    using REDUC = ::fsc::DiscardReducer;
    benchmark<::dsc::reduction_unordered_map<KeyType, ValType, MapParams, REDUC>, UNORDERED>(input, query, "std::unordered_map_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::reduction_densehash_map<KeyType, ValType, MapParams, special_keys<KeyType>, REDUC>, DENSEHASH>(input, query, "google::densehash_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == LAST)
    using REDUC = ::fsc::ReplaceReducer;
    benchmark<::dsc::reduction_unordered_map<KeyType, ValType, MapParams, REDUC>, UNORDERED>(input, query, "std::unordered_map_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::reduction_densehash_map<KeyType, ValType, MapParams, special_keys<KeyType>, REDUC>, DENSEHASH>(input, query, "google::densehash_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
#else
    static_assert(false, "UNSUPPORTED REDUCTION TYPE");
#endif // pINDEX
#endif // pDistHash and pStoreHash

#if (pINDEX == COUNT) // map
    benchmark<::dsc::counting_batched_robinhood_map<KeyType, ValType, MapParams>, BROBINHOOD>(input, query, "robinhood_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::counting_batched_radixsort_map<KeyType, ValType, MapParams>, RADIXSORT>(input, query, "radixsort_count", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == FIRST)
    using REDUC = ::fsc::DiscardReducer;
    benchmark<::dsc::reduction_batched_robinhood_map<KeyType, ValType, MapParams, REDUC>, BROBINHOOD>(input, query, "robinhood_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::reduction_batched_radixsort_map<KeyType, ValType, MapParams, REDUC>, RADIXSORT>(input, query, "radixsort_first", max_load, min_load, insert_prefetch, query_prefetch, comm);
#elif (pINDEX == LAST)
    using REDUC = ::fsc::ReplaceReducer;
    benchmark<::dsc::reduction_batched_robinhood_map<KeyType, ValType, MapParams, REDUC>, BROBINHOOD>(input, query, "robinhood_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
    benchmark<::dsc::reduction_batched_radixsort_map<KeyType, ValType, MapParams, REDUC>, RADIXSORT>(input, query, "radixsort_last", max_load, min_load, insert_prefetch, query_prefetch, comm);
#else
    static_assert(false, "UNSUPPORTED REDUCTION TYPE");
#endif // pINDEX

} // done hybrid

    // mpi cleanup is automatic
    comm.barrier();

    return 0;
}
