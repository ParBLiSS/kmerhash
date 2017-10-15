set(TEST_NAME kmerhash_test_apps)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")

if(BUILD_TESTS)
endif(BUILD_TESTS)


message(STATUS "VTUNE_ANALYSIS set to ${VTUNE_ANALYSIS}")


function(add_hashmap_target storehash)

      add_executable(benchmark_hashtables_${storehash} BenchmarkHashTables.cpp)
      SET_TARGET_PROPERTIES(benchmark_hashtables_${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpStoreHash=${storehash}")
      target_link_libraries(benchmark_hashtables_${storehash} ${EXTRA_LIBS})

	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(benchmark_hashtables_${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)

endfunction(add_hashmap_target)


function(add_dist_hashmap_target prefix map disthash storehash extra)

      add_executable(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerIndex.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra}")
      target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)       
endfunction(add_dist_hashmap_target)



if (BUILD_BENCHMARKS)


# EXECUTABLES
	# benchmark executable for hash functions
	add_executable(benchmark_hashes BenchmarkHashFunctions.cpp)
	target_link_libraries(benchmark_hashes ${EXTRA_LIBS})
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(benchmark_hashes ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)

	# benchmark executable, FARM and MURMUR
	foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
		add_hashmap_target(${hash})
	endforeach(hash)
	
	# distributed benchmarks.
	foreach(map ROBINHOOD DENSEHASH)
		foreach(hash FARM MURMUR)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY)
		endforeach(hash)
	endforeach(map)
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY)
		endforeach(hash)
	endforeach(map)
	
	
	# lz4 distributed benchmarks.
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(testLZ4KmerIndex ${map} ${hash} ${hash} ENABLE_LZ4_RESULT)
		endforeach(hash)
	endforeach(map)

	#overlapped comm and insert/find/count/erase.
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(overlap-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM)
		endforeach(hash)
	endforeach(map)
	
	#overlapped comm and insert/find/count/erase.
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(overlapFull-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_FULLBUFFER)
		endforeach(hash)
	endforeach(map)

	#overlapped comm and insert/find/count/erase.
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(overlap2P-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_2P)
		endforeach(hash)
	endforeach(map)


# kmer-count pair.
  add_executable(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DINSERT_KMER_COUNT_PAIR")
  target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)

# dump k-mers
  add_executable(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DDUMP_DISTRIBUTED_INPUT")
  target_link_libraries(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)

# dump kmer-count pair.
  add_executable(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DINSERT_KMER_COUNT_PAIR -DDUMP_DISTRIBUTED_INPUT")
  target_link_libraries(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)



# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/martinus/src")
include_directories("${EXT_PROJECTS_DIR}/martinus/src/3rdparty/google")
add_executable(benchmark_hashtables_martinus martinus_test_robinhood.cpp)
target_link_libraries(benchmark_hashtables_martinus ${EXTRA_LIBS})



#add_executable(hash_vs_sort hash_vs_sort.cpp)
#target_link_libraries(hash_vs_sort ${EXTRA_LIBS})

#add_executable(pointer_vs_ref pointer_vs_reference.cpp)
#target_link_libraries(pointer_vs_ref ${EXTRA_LIBS})


if (Boost_FOUND)
	add_executable(cust_alloc test_custom_allocator.cpp)
	target_link_libraries(cust_alloc ${EXTRA_LIBS})
endif(Boost_FOUND)

add_executable(test_prealloc test_prealloc.cpp)
target_link_libraries(test_prealloc ${EXTRA_LIBS})

endif(BUILD_BENCHMARKS)

if (BUILD_APPLICATIONS)

endif(BUILD_APPLICATIONS)

