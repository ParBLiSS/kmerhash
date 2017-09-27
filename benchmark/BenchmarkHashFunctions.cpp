#include <vector>
#include <algorithm>  // std::generate
#include <cstdlib>    // std::rand
#include <sstream>

#include "kmerhash/hash.hpp"
#include "utils/benchmark_utils.hpp"

#include "tclap/CmdLine.h"


// comparison of some hash functions.

#ifdef VTUNE_ANALYSIS
#define MEASURE_DISABLED 0

#define MEASURE_MURMURSSE 10
#define MEASURE_MURMURAVX 11
#define MEASURE_CRC32C 12

static int measure_mode = MEASURE_DISABLED;

#include <ittnotify.h>

#endif


template <size_t N>
struct DataStruct {
    unsigned char data[N];
};


template <typename H, size_t N>
void benchmark_hash(H const & hasher, DataStruct<N> const * data, unsigned int * hashes, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    hashes[i] = hasher(data[i]);
  }
}

template <typename H, size_t N>
void benchmark_hash_batch(H const & hasher, DataStruct<N> const * data, unsigned int * hashes, size_t count) {
  hasher.hash(data, count, hashes);
}


template <size_t N>
void benchmarks(size_t count, unsigned char* in, unsigned int* out) {
  BL_BENCH_INIT(benchmark);

  DataStruct<N>* data = reinterpret_cast<DataStruct<N>*>(in);

  // ============ flat_hash_map  not compiling.
  // doesn't compile.  it's using initializer lists extensively, and the templated_iterator is having trouble constructing from initializer list.
  BL_BENCH_START(benchmark);
  {
    ::fsc::hash::identity<DataStruct<N> > h;
     benchmark_hash(h, data, out, count);
  }
  BL_BENCH_END(benchmark, "iden", count);


  BL_BENCH_START(benchmark);
  {
    ::fsc::hash::farm<DataStruct<N> > h;
     benchmark_hash(h, data, out, count);
  }
  BL_BENCH_END(benchmark, "farm", count);

  BL_BENCH_START(benchmark);
  {
    ::fsc::hash::murmur<DataStruct<N> > h;
     benchmark_hash(h, data, out, count);
  }
  BL_BENCH_END(benchmark, "murmur", count);

  BL_BENCH_START(benchmark);
  {
    ::fsc::hash::murmur32<DataStruct<N> > h;
     benchmark_hash(h, data, out, count);
  }
  BL_BENCH_END(benchmark, "murmur32", count);

#if defined(__SSE4_1__)
  BL_BENCH_START(benchmark);
  {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_MURMURSSE)
      __itt_resume();
#endif
    ::fsc::hash::murmur3sse32<DataStruct<N> > h;
     benchmark_hash_batch(h, data, out, count);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_MURMURSSE)
      __itt_pause();
#endif
  }
  BL_BENCH_END(benchmark, "murmur32sse", count);
#endif

#if defined(__AVX2__)
  BL_BENCH_START(benchmark);
  {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_MURMURAVX)
      __itt_resume();
#endif
    ::fsc::hash::murmur3avx32<DataStruct<N> > h;
     benchmark_hash_batch(h, data, out, count);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_MURMURAVX)
      __itt_pause();
#endif
  }
  BL_BENCH_END(benchmark, "murmur32avx", count);
#endif

  BL_BENCH_START(benchmark);
  {
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_CRC32C)
      __itt_resume();
#endif
    ::fsc::hash::crc32c<DataStruct<N> > h;
     benchmark_hash_batch(h, data, out, count);
#ifdef VTUNE_ANALYSIS
  if (measure_mode == MEASURE_CRC32C)
      __itt_pause();
#endif
  }
  BL_BENCH_END(benchmark, "CRC32C", count);


  std::stringstream ss;
  ss << "hash " << count << " " << size_t(N) << "-byte elements";

  BL_BENCH_REPORT_NAMED(benchmark, ss.str().c_str());

}


int main(int argc, char** argv) {

#ifdef VTUNE_ANALYSIS
      __itt_pause();
#endif

      size_t count = 100000000;


      try {

        // Define the command line object, and insert a message
        // that describes the program. The "Command description message"
        // is printed last in the help text. The second argument is the
        // delimiter (usually space) and the last one is the version number.
        // The CmdLine object parses the argv array based on the Arg objects
        // that it contains.
        TCLAP::CmdLine cmd("Benchmark hash function", ' ', "0.1");

        // Define a value argument and add it to the command line.
        // A value arg defines a flag and a type of value that it expects,
        // such as "-n Bishop".

        TCLAP::ValueArg<size_t> countArg("c","count","number of elements to hash", false, count, "size_t", cmd);

    #ifdef VTUNE_ANALYSIS
        std::vector<std::string> measure_modes;
        measure_modes.push_back("murmur_sse");
        measure_modes.push_back("murmur_avx");
        measure_modes.push_back("crc32c");
        measure_modes.push_back("disabled");
        TCLAP::ValuesConstraint<std::string> measureModeVals( measure_modes );
        TCLAP::ValueArg<std::string> measureModeArg("","measured_op","hash function to measure (default insert)",false,"disabled",&measureModeVals, cmd);
    #endif

        // Parse the argv array.
        cmd.parse( argc, argv );

        count = countArg.getValue();


    #ifdef VTUNE_ANALYSIS
        // set the default for query to filename, and reparse
        std::string measure_mode_str = measureModeArg.getValue();
        if (comm.rank() == 0) std::cout << "Measuring " << measure_mode_str << std::endl;

        if (measure_mode_str == "murmur_sse") {
          measure_mode = MEASURE_MURMURSSE;
        } else if (measure_mode_str == "murmur_avx") {
          measure_mode = MEASURE_MURMURAVX;
        } else if (measure_mode_str == "crc32c") {
          measure_mode = MEASURE_CRC32C;
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






  unsigned char* data = (unsigned char*) malloc(count * 256);
  unsigned int* hashes = (unsigned int*) malloc(count * sizeof(unsigned int));

  benchmarks<  1>(count, data, hashes);
  benchmarks<  2>(count, data, hashes);
  benchmarks<  3>(count, data, hashes);
  benchmarks<  4>(count, data, hashes);
  benchmarks<  5>(count, data, hashes);
  benchmarks<  7>(count, data, hashes);
  benchmarks<  8>(count, data, hashes);
  benchmarks<  9>(count, data, hashes);
  benchmarks< 15>(count, data, hashes);
  benchmarks< 16>(count, data, hashes);
  benchmarks< 17>(count, data, hashes);
  benchmarks< 31>(count, data, hashes);
  benchmarks< 32>(count, data, hashes);
  benchmarks< 33>(count, data, hashes);
  benchmarks< 63>(count, data, hashes);
  benchmarks< 64>(count, data, hashes);
  benchmarks< 65>(count, data, hashes);
  benchmarks<127>(count, data, hashes);
  benchmarks<128>(count, data, hashes);
  benchmarks<129>(count, data, hashes);
  benchmarks<255>(count, data, hashes);
  benchmarks<256>(count, data, hashes);


  free(data);
  free(hashes);
#ifdef VTUNE_ANALYSIS
      __itt_resume();
#endif

}

