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

#include <cstdio>
#include <stdint.h>

#include <tuple>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>

#include "common/kmer.hpp"
#include "common/kmer_transform.hpp"

#include "index/kmer_hash.hpp"

#include "containers/densehash_map.hpp"

#include "utils/benchmark_utils.hpp"


using Kmer = bliss::common::Kmer<31, bliss::common::DNA, uint64_t>;
using KmerCount = ::std::pair<Kmer, uint64_t>;

typedef bliss::kmer::transform::lex_less<Kmer> KeyTransform;
typedef bliss::kmer::hash::farm<Kmer, false> KmerHash;

struct TransformedHash {
    KmerHash h;
    KeyTransform trans;

    inline uint64_t operator()(Kmer const& k) const {
      return h(trans(k));
    }
    template<typename V>
    inline uint64_t operator()(::std::pair<Kmer, V> const& x) const {
      return this->operator()(x.first);
    }
    template<typename V>
    inline uint64_t operator()(::std::pair<const Kmer, V> const& x) const {
      return this->operator()(x.first);
    }
};

template <typename Comparator>
struct TransformedComp {
    Comparator comp;
    KeyTransform trans;

    inline bool operator()(Kmer const & x, Kmer const & y) const {
      return comp(trans(x), trans(y));
    }
    template<typename V>
    inline bool operator()(::std::pair<Kmer, V> const & x, Kmer const & y) const {
      return this->operator()(x.first, y);
    }
    template<typename V>
    inline bool operator()(::std::pair<const Kmer, V> const & x, Kmer const & y) const {
      return this->operator()(x.first, y);
    }
    template<typename V>
    inline bool operator()(Kmer const & x, ::std::pair<Kmer, V> const & y) const {
      return this->operator()(x, y.first);
    }
    template<typename V>
    inline bool operator()(Kmer const & x, ::std::pair<const Kmer, V> const & y) const {
      return this->operator()(x, y.first);
    }
    template<typename V>
    inline bool operator()(::std::pair<Kmer, V> const & x, ::std::pair<Kmer, V> const & y) const {
      return this->operator()(x.first, y.first);
    }
    template<typename V>
    inline bool operator()(::std::pair<const Kmer, V> const & x, ::std::pair<const Kmer, V> const & y) const {
      return this->operator()(x.first, y.first);
    }
};

typedef TransformedComp<::std::less<Kmer> > TransformedLess;
typedef TransformedComp<::std::equal_to<Kmer> > TransformedEqual;

template <typename T>
using vector_stl_alloc = ::std::vector<T, ::std::allocator<T > >;

template <typename K, typename T>
using umap_stl_alloc = ::std::unordered_map<K, T, TransformedHash, TransformedEqual, ::std::allocator<::std::pair<const K, T> > >;


template <typename K, typename T>
using dmap_stl_alloc = ::fsc::densehash_map<K, T, ::bliss::kmer::hash::sparsehash::special_keys<K, false>,
    ::bliss::kmer::transform::lex_less, TransformedHash >;



class KmerHelper
{
public:

    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution;
    uint64_t v[1];

    Kmer random_kmer() {
        *v = distribution(generator);
        return Kmer(v);
    }

};




int main(int argc, char** argv) {


	int iterations = 10000000;
	if (argc > 1) {
		iterations = atoi(argv[1]);
	}


	printf("Benchmarking preallocation effects\n\n");

	printf("SIZES:\n");
	printf("kmer size: %lu\n", sizeof(Kmer));
	printf("Kmer Pos size: %lu\n", sizeof(KmerCount));

	printf("Vector of Kmer, base size: %lu\n", sizeof(::std::vector<Kmer>));
	printf("Vector of KmerCount, base size: %lu\n", sizeof(::std::vector<KmerCount >));

	printf("Kmer Vector1 pair, base size: %lu\n", sizeof(::std::pair<Kmer, ::std::vector<Kmer> >));
	printf("Kmer Vector2 pair, base size: %lu\n", sizeof(::std::pair<Kmer, ::std::vector<KmerCount> >));



	KmerHelper helper;


	Kmer result;

  std::default_random_engine generator;
  std::uniform_int_distribution<uint64_t> distribution;

  {
    vector_stl_alloc<KmerCount> stlumap;
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.capacity();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.emplace_back(helper.random_kmer(), 1UL);

      if (curr_buckets < stlumap.capacity()) {
        curr_buckets = stlumap.capacity();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }




	{
    BL_BENCH_INIT(vector);

    BL_BENCH_START(vector);
	  vector_stl_alloc<KmerCount> stlvec;
    result = helper.random_kmer();
    BL_BENCH_END(vector, "stl_init", stlvec.capacity());

	  BL_BENCH_START(vector);
		for (int i = 0; i < iterations; ++i) {
			stlvec.emplace_back(helper.random_kmer(), 1UL);
		}
	  BL_BENCH_END(vector, "stl_emplace", stlvec.size());

	  BL_BENCH_START(vector);
		for (size_t i = 0; i < stlvec.size(); ++i) {
			result ^= stlvec[i].first.reverse_complement();
		}
    BL_BENCH_END(vector, "stl_seq_access", stlvec.size());
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    size_t id;
    for (int i = 0; i < iterations; ++i) {
      id = distribution(generator);
      result ^= stlvec[id % stlvec.size()].first.reverse_complement();
    }
    BL_BENCH_END(vector, "stl_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    std::sort(stlvec.begin(), stlvec.end(), TransformedLess());
    BL_BENCH_END(vector, "stl_sort", stlvec.size());

    BL_BENCH_START(vector);
    auto new_end = std::unique(stlvec.begin(), stlvec.end(), TransformedEqual());
    stlvec.erase(new_end);
    BL_BENCH_END(vector, "stl_unique", stlvec.size());


    BL_BENCH_START(vector);
    for (size_t i = 0; i < stlvec.size(); ++i) {
      result ^= stlvec[i].first.reverse_complement();
    }
    BL_BENCH_END(vector, "stl_seq_access", stlvec.size());
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    for (int i = 0; i < iterations; ++i) {
      id = distribution(generator);
      result ^= stlvec[id % stlvec.size()].first.reverse_complement();
    }
    BL_BENCH_END(vector, "stl_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());


    BL_BENCH_REPORT_NAMED(vector, "stl_vector");

	}


  {
    vector_stl_alloc<KmerCount> stlumap;
    stlumap.reserve(iterations);
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.capacity();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.emplace_back(helper.random_kmer(), 1UL);

      if (curr_buckets < stlumap.capacity()) {
        curr_buckets = stlumap.capacity();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }


  {
    BL_BENCH_INIT(vector);

    BL_BENCH_START(vector);
    vector_stl_alloc<KmerCount> stlvec;
    stlvec.reserve(iterations);
    result = helper.random_kmer();
    BL_BENCH_END(vector, "stl_init", stlvec.capacity());

    BL_BENCH_START(vector);
    for (int i = 0; i < iterations; ++i) {
      stlvec.emplace_back(helper.random_kmer(), 1UL);
    }
    BL_BENCH_END(vector, "stl_emplace", stlvec.size());

    BL_BENCH_START(vector);
    for (size_t i = 0; i < stlvec.size(); ++i) {
      result ^= stlvec[i].first;
    }
    BL_BENCH_END(vector, "stl_seq_access", stlvec.size());
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    size_t id;
    for (int i = 0; i < iterations; ++i) {
      id = distribution(generator);
      result ^= stlvec[id % stlvec.size()].first;
    }
    BL_BENCH_END(vector, "stl_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    std::sort(stlvec.begin(), stlvec.end(), TransformedLess());
    BL_BENCH_END(vector, "stl_sort", stlvec.size());

    BL_BENCH_START(vector);
    auto new_end = std::unique(stlvec.begin(), stlvec.end(), TransformedEqual());
    stlvec.erase(new_end);
    BL_BENCH_END(vector, "stl_unique", stlvec.size());


    BL_BENCH_START(vector);
    for (size_t i = 0; i < stlvec.size(); ++i) {
      result ^= stlvec[i].first;
    }
    BL_BENCH_END(vector, "stl_seq_access", stlvec.size());
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_START(vector);
    for (int i = 0; i < iterations; ++i) {
      id = distribution(generator);
      result ^= stlvec[id % stlvec.size()].first;
    }
    BL_BENCH_END(vector, "stl_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_REPORT_NAMED(vector, "stl_vector_prealloc");

  }

  {
    umap_stl_alloc<Kmer, size_t> stlumap;
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.bucket_count();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));

      if (curr_buckets < stlumap.bucket_count()) {
        curr_buckets = stlumap.bucket_count();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }





  {
    BL_BENCH_INIT(umap);

    BL_BENCH_START(umap);
    umap_stl_alloc<Kmer, size_t> stlumap;
    result = helper.random_kmer();
    BL_BENCH_END(umap, "umap_init", stlumap.bucket_count());

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      stlumap.emplace(helper.random_kmer(), 1UL);
    }
    BL_BENCH_END(umap, "umap_emplace", stlumap.size());

    std::vector<Kmer> kmers;
    kmers.reserve(stlumap.size());

    BL_BENCH_START(umap);
    for (auto x : stlumap) {
      kmers.emplace_back(x.first);
    }
    BL_BENCH_END(umap, "umap_seq_access", kmers.size());

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(kmers.begin(), kmers.end(), g);

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      result ^= (*stlumap.find(kmers[i % kmers.size()])).first;
    }
    BL_BENCH_END(umap, "umap_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_REPORT_NAMED(umap, "unordered_map");

  }

  {
    umap_stl_alloc<Kmer, size_t> stlumap;
    stlumap.reserve(iterations);
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.bucket_count();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));

      if (curr_buckets < stlumap.bucket_count()) {
        curr_buckets = stlumap.bucket_count();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }

  {
    BL_BENCH_INIT(umap);

    BL_BENCH_START(umap);
    umap_stl_alloc<Kmer, size_t> stlumap;
    stlumap.reserve(iterations);
    result = helper.random_kmer();
    BL_BENCH_END(umap, "umap_init", stlumap.bucket_count());

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      stlumap.emplace(helper.random_kmer(), 1UL);
    }
    BL_BENCH_END(umap, "umap_emplace", stlumap.size());

    std::vector<Kmer> kmers;
    kmers.reserve(stlumap.size());

    BL_BENCH_START(umap);
    for (auto x : stlumap) {
      kmers.emplace_back(x.first);
    }
    BL_BENCH_END(umap, "umap_seq_access", kmers.size());

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(kmers.begin(), kmers.end(), g);

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      result ^= (*stlumap.find(kmers[i % stlumap.size()])).first;
    }
    BL_BENCH_END(umap, "umap_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_REPORT_NAMED(umap, "unordered_map_prealloc");

  }



  {
    dmap_stl_alloc<Kmer, size_t> stlumap;
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.bucket_count();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));

      if (curr_buckets < stlumap.bucket_count()) {
        curr_buckets = stlumap.bucket_count();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }






  {
    BL_BENCH_INIT(umap);

    BL_BENCH_START(umap);
    dmap_stl_alloc<Kmer, size_t> stlumap;
    result = helper.random_kmer();
    BL_BENCH_END(umap, "umap_init", stlumap.bucket_count());

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));
    }
    BL_BENCH_END(umap, "umap_emplace", stlumap.size());

    std::vector<Kmer> kmers;
    kmers.reserve(stlumap.size());

    BL_BENCH_START(umap);
    for (auto x : stlumap) {
      kmers.emplace_back(x.first);
    }
    BL_BENCH_END(umap, "umap_seq_access", kmers.size());

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(kmers.begin(), kmers.end(), g);

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      result ^= (*stlumap.find(kmers[i % stlumap.size()])).first;
    }
    BL_BENCH_END(umap, "umap_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_REPORT_NAMED(umap, "densehash_map");

  }

  {
    dmap_stl_alloc<Kmer, size_t> stlumap;
    stlumap.resize(iterations);
    result = helper.random_kmer();

    size_t curr_buckets = stlumap.bucket_count();
    printf("curr_buckets = %ld at iteration 0\n", curr_buckets);

    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));

      if (curr_buckets < stlumap.bucket_count()) {
        curr_buckets = stlumap.bucket_count();
        printf("curr_buckets = %ld at iteration %d\n", curr_buckets, i);
      }
    }

  }

  {
    BL_BENCH_INIT(umap);

    BL_BENCH_START(umap);
    dmap_stl_alloc<Kmer, size_t> stlumap;
    stlumap.resize(iterations);
    result = helper.random_kmer();
    BL_BENCH_END(umap, "umap_init", stlumap.bucket_count());

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      stlumap.insert(std::make_pair(helper.random_kmer(), 1UL));
    }
    BL_BENCH_END(umap, "umap_emplace", stlumap.size());

    std::vector<Kmer> kmers;
    kmers.reserve(stlumap.size());

    BL_BENCH_START(umap);
    for (auto x : stlumap) {
      kmers.emplace_back(x.first);
    }
    BL_BENCH_END(umap, "umap_seq_access", kmers.size());

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(kmers.begin(), kmers.end(), g);

    BL_BENCH_START(umap);
    for (int i = 0; i < iterations; ++i) {
      result ^= (*stlumap.find(kmers[i % stlumap.size()])).first;
    }
    BL_BENCH_END(umap, "umap_rand_access", iterations);
    printf("result : %s\n", result.toAlphabetString().c_str());

    BL_BENCH_REPORT_NAMED(umap, "densehash_map_prealloc");

  }


}
