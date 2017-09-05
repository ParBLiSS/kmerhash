set(TEST_NAME kmerhash_test_apps)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")

if(BUILD_TESTS)
endif(BUILD_TESTS)




if (BUILD_BENCHMARKS)


# EXECUTABLES
	add_executable(benchmark_hashtables_FARM BenchmarkHashTables.cpp)
	SET_TARGET_PROPERTIES(benchmark_hashtables_FARM
     PROPERTIES COMPILE_FLAGS 
     "-DpStoreHash=FARM")
	target_link_libraries(benchmark_hashtables_FARM ${EXTRA_LIBS})
	
	add_executable(benchmark_hashtables_MURMUR BenchmarkHashTables.cpp)
	SET_TARGET_PROPERTIES(benchmark_hashtables_MURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpStoreHash=MURMUR")
	target_link_libraries(benchmark_hashtables_MURMUR ${EXTRA_LIBS})


  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=FARM -DpStoreHash=FARM -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM
   ${EXTRA_LIBS})

  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhFARM-shFARM BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhFARM-shFARM
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=DENSEHASH -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=FARM -DpStoreHash=FARM -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhFARM-shFARM
   ${EXTRA_LIBS})

  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=FARM -DpStoreHash=FARM -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM
   ${EXTRA_LIBS})


  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=DENSEHASH -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare")
  target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare -DINSERT_KMER_COUNT_PAIR")
  target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(testLZ4KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testLZ4KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare -DENABLE_LZ4_COMM")
  target_link_libraries(testLZ4KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(dumpKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare -DDUMP_DISTRIBUTED_INPUT")
  target_link_libraries(dumpKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(overlap-KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(overlap-KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DOVERLAPPED_COMM")
  target_link_libraries(overlap-KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})

  add_executable(dumpKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=ROBINHOOD_BATCHED -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DpCompare -DINSERT_KMER_COUNT_PAIR -DDUMP_DISTRIBUTED_INPUT")
  target_link_libraries(dumpKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})


message(STATUS "VTUNE_ANALYSIS set to ${VTUNE_ANALYSIS}")
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(benchmark_hashtables_FARM ittnotify dl)
	target_link_libraries(benchmark_hashtables_MURMUR ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhFARM-shFARM ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhFARM-shFARM ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-ROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(testKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(overlap-KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(dumpKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(dumpKmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
	target_link_libraries(testLZ4KmerIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)

# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/martinus/src")
include_directories("${EXT_PROJECTS_DIR}/martinus/src/3rdparty/google")
add_executable(benchmark_hashtables_martinus martinus_test_robinhood.cpp)
target_link_libraries(benchmark_hashtables_martinus ${EXTRA_LIBS})






add_executable(hash_vs_sort hash_vs_sort.cpp)
target_link_libraries(hash_vs_sort ${EXTRA_LIBS})

add_executable(pointer_vs_ref pointer_vs_reference.cpp)
target_link_libraries(pointer_vs_ref ${EXTRA_LIBS})


if (Boost_FOUND)
	add_executable(cust_alloc test_custom_allocator.cpp)
	target_link_libraries(cust_alloc ${EXTRA_LIBS})
endif(Boost_FOUND)

add_executable(test_prealloc test_prealloc.cpp)
target_link_libraries(test_prealloc ${EXTRA_LIBS})

endif(BUILD_BENCHMARKS)

if (BUILD_APPLICATIONS)

endif(BUILD_APPLICATIONS)

