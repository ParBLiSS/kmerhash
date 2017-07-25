set(TEST_NAME kmerhash_test_apps)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")

if(BUILD_TESTS)
endif(BUILD_TESTS)




if (BUILD_BENCHMARKS)


# EXECUTABLES
#include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/tommyds")
add_executable(benchmark_hashtables BenchmarkHashTables.cpp)
target_link_libraries(benchmark_hashtables ${EXTRA_LIBS})

message(STATUS "VTUNE_ANALYSIS set to ${VTUNE_ANALYSIS}")
if (ENABLE_VTUNE_PROFILING)
	ADD_LIBRARY(ittnotify STATIC IMPORTED)
	SET_TARGET_PROPERTIES(ittnotify PROPERTIES
	    IMPORTED_LOCATION ${VTUNE_LIB})
	target_link_libraries(benchmark_hashtables ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)

# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/martinus/src")
include_directories("${EXT_PROJECTS_DIR}/martinus/src/3rdparty/google")
add_executable(benchmark_hashtables_martinus martinus_test_robinhood.cpp)
target_link_libraries(benchmark_hashtables_martinus ${EXTRA_LIBS})



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

