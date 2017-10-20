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
         "-DpStoreHash=${storehash} -DENABLE_PREFETCH")
      target_link_libraries(benchmark_hashtables_${storehash} ${EXTRA_LIBS})

	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(benchmark_hashtables_${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)
	#cotire(benchmark_hashtables_${storehash})

endfunction(add_hashmap_target)


function(add_dist_hashmap_target prefix map disthash storehash extra extra2)

      add_executable(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerIndex.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)
	
	#cotire(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
endfunction(add_dist_hashmap_target)


function(add_dist_hashmap_kmer_target prefix k dna map disthash storehash extra extra2)

      add_executable(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerIndex.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=FASTQ -DpDNA=${dna} -DpK=${k} -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)
	
	#cotire(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
endfunction(add_dist_hashmap_kmer_target)


function(add_dist_counter_target prefix format k map disthash storehash extra extra2)

      add_executable(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerCounter.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=${format} -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
	endif (ENABLE_VTUNE_PROFILING)
	
	#cotire(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
endfunction(add_dist_counter_target)


if (BUILD_BENCHMARKS)


# EXECUTABLES
	# benchmark executable for hash functions
	add_executable(benchmark_hashes BenchmarkHashFunctions.cpp)
	target_link_libraries(benchmark_hashes ${EXTRA_LIBS})
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(benchmark_hashes ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
	#cotire(benchmark_hashes)

	# benchmark executable, FARM and MURMUR
	foreach(hash STD IDEN FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C CLHASH)
		add_hashmap_target(${hash})
	endforeach(hash)
	
	# distributed benchmarks.
	foreach(map DENSEHASH)
		foreach(hash FARM MURMUR CLHASH)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY KH_DUMMY2)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} CRC32C KH_DUMMY KH_DUMMY2)

			foreach(k 15 21 63)
				add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} CRC32C KH_DUMMY KH_DUMMY2)
				add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} ${hash} KH_DUMMY KH_DUMMY2)
			endforeach(k)
			foreach(dna 5 16)
				add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} CRC32C KH_DUMMY KH_DUMMY2)
				add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} ${hash} KH_DUMMY KH_DUMMY2)
			endforeach(dna)
		endforeach(hash)
	endforeach(map)
	
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash MURMUR MURMUR32avx CLHASH)
			# with prefetch
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH)

			# without prefetch		
			add_dist_hashmap_target(noPref_KmerIndex ${map} ${hash} ${hash} KH_DUMMY1 KH_DUMMY2)
			add_dist_hashmap_target(noPref_KmerIndex ${map} ${hash} CRC32C KH_DUMMY1 KH_DUMMY2)

			# with overlap and no prefetch
			add_dist_hashmap_target(overlap_noPref-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM KH_DUMMY2)
			add_dist_hashmap_target(overlap_noPref-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM KH_DUMMY2)

			#with overlap and prefetch
			add_dist_hashmap_target(overlap-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH)
			add_dist_hashmap_target(overlap-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH)

			foreach(k 15 21 63)
					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(overlap-KmerIndex ${k} 4 ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(overlap-KmerIndex ${k} 4 ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH)
			endforeach(k)
			foreach(dna 5 16)
					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(overlap-KmerIndex 31 ${dna} ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH)
					add_dist_hashmap_kmer_target(overlap-KmerIndex 31 ${dna} ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH)
			endforeach(dna)

		endforeach(hash)
		add_dist_hashmap_target(testKmerIndex ${map} FARM FARM KH_DUMMY1 ENABLE_PREFETCH)
		add_dist_hashmap_target(noPref_KmerIndex ${map} FARM FARM KH_DUMMY1 KH_DUMMY2)
		add_dist_hashmap_target(overlap_noPref-KmerIndex ${map} FARM FARM OVERLAPPED_COMM KH_DUMMY2)
		add_dist_hashmap_target(overlap-KmerIndex ${map} FARM FARM OVERLAPPED_COMM ENABLE_PREFETCH)
					
	endforeach(map)

	foreach(map BROBINHOOD RADIXSORT)
		# lz4 distributed benchmarks.
		foreach(hash MURMUR32avx CLHASH) 
		 # FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C
			add_dist_hashmap_target(testLZ4KmerIndex ${map} ${hash} ${hash} ENABLE_LZ4_RESULT ENABLE_PREFETCH)
		endforeach(hash)
	
		#overlapped comm using full array
		foreach(hash MURMUR32avx CLHASH) 
		# FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(overlapFull-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_FULLBUFFER ENABLE_PREFETCH)
		endforeach(hash)
	
		#overlapped comm 2 phase.
		foreach(hash MURMUR32avx CLHASH) 
		#FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx CRC32C)
			add_dist_hashmap_target(overlap2P-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_2P ENABLE_PREFETCH)
		endforeach(hash)
	endforeach(map)


# kmer-count pair.
  add_executable(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DINSERT_KMER_COUNT_PAIR -DENABLE_PREFETCH")
  target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)
	#cotire(testKmerPairIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR)

# dump k-mers
  add_executable(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DDUMP_DISTRIBUTED_INPUT -DENABLE_PREFETCH")
  target_link_libraries(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)
	#cotire(dumpKIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR)

# dump kmer-count pair.
  add_executable(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR BenchmarkKmerIndex.cpp)
  SET_TARGET_PROPERTIES(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
     PROPERTIES COMPILE_FLAGS 
     "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=BROBINHOOD -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=MURMUR -DpStoreHash=MURMUR -DINSERT_KMER_COUNT_PAIR -DDUMP_DISTRIBUTED_INPUT -DENABLE_PREFETCH")
  target_link_libraries(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR
   ${EXTRA_LIBS})
if (ENABLE_VTUNE_PROFILING)
	target_link_libraries(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR ittnotify dl)
endif(ENABLE_VTUNE_PROFILING)
	#cotire(dumpKVIndex-FASTQ-a4-k31-CANONICAL-BROBINHOOD-COUNT-dtIDEN-dhMURMUR-shMURMUR)



# kmer counter builds
foreach(map BROBINHOOD RADIXSORT)
	foreach(hash MURMUR32avx MURMUR CLHASH) 
		add_dist_counter_target(testKmerCounter FASTA 31 ${map} ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH)
		add_dist_counter_target(testKmerCounter FASTA 31 ${map} ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH)
	
		foreach(k 15 21 23 27 31 39 47 55 63 65 95 127 159 191 223 255)
			add_dist_counter_target(testKmerCounter FASTQ ${k} ${map} ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH)
			add_dist_counter_target(testKmerCounter FASTQ ${k} ${map} ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH)
		endforeach(k)
	endforeach(hash)

endforeach(map)




# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/martinus/src")
include_directories("${EXT_PROJECTS_DIR}/martinus/src/3rdparty/google")
add_executable(benchmark_hashtables_martinus martinus_test_robinhood.cpp)
SET_TARGET_PROPERTIES(benchmark_hashtables_martinus
	PROPERTIES COMPILE_FLAGS "-DENABLE_PREFETCH")
target_link_libraries(benchmark_hashtables_martinus ${EXTRA_LIBS})
#cotire(benchmark_hashtables_martinus)



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

