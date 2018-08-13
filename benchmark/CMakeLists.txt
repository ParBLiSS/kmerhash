set(TEST_NAME kmerhash_test_apps)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")

if(BUILD_TESTS)
endif(BUILD_TESTS)


message(STATUS "VTUNE_ANALYSIS set to ${VTUNE_ANALYSIS}")


if(NOT TARGET serial_benchmarks)
	add_custom_target(serial_benchmarks)
endif(NOT TARGET serial_benchmarks)

if(NOT TARGET shmem_benchmarks)
	add_custom_target(shmem_benchmarks)
endif(NOT TARGET shmem_benchmarks)
	
if(NOT TARGET dist_benchmarks)
	add_custom_target(dist_benchmarks)
endif(NOT TARGET dist_benchmarks)

if(NOT TARGET distht_benchmarks)
	add_custom_target(distht_benchmarks)
endif(NOT TARGET distht_benchmarks)
	
if(NOT TARGET prefetch_benchmarks)
	add_custom_target(prefetch_benchmarks)
endif(NOT TARGET prefetch_benchmarks)
	
if(NOT TARGET overlap_benchmarks)
	add_custom_target(overlap_benchmarks)
endif(NOT TARGET overlap_benchmarks)

if(NOT TARGET lz4_benchmarks)
	add_custom_target(lz4_benchmarks)
endif(NOT TARGET lz4_benchmarks)
	
if(NOT TARGET k_benchmarks)
	add_custom_target(k_benchmarks)
endif(NOT TARGET k_benchmarks)
	

function(add_hashmap_target storehash group)

	if (NOT TARGET benchmark_hashtables_${storehash})
      add_executable(benchmark_hashtables_${storehash} BenchmarkHashTables.cpp)
      SET_TARGET_PROPERTIES(benchmark_hashtables_${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpStoreHash=${storehash} -DENABLE_PREFETCH")
      target_link_libraries(benchmark_hashtables_${storehash} ${EXTRA_LIBS})

		if (ENABLE_VTUNE_PROFILING)
			target_link_libraries(benchmark_hashtables_${storehash} ittnotify dl)
		endif (ENABLE_VTUNE_PROFILING)
	
	endif(NOT TARGET benchmark_hashtables_${storehash})
	
	add_dependencies(${group} benchmark_hashtables_${storehash})

endfunction(add_hashmap_target)


function(add_dist_hashmap_target prefix map disthash storehash extra extra2 group)

	if (NOT TARGET ${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
      add_executable(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerIndex.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=FASTQ -DpDNA=4 -DpK=31 -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
		if (ENABLE_VTUNE_PROFILING)
			target_link_libraries(${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
		endif (ENABLE_VTUNE_PROFILING)
		
	endif(NOT TARGET ${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
	
	add_dependencies(${group} ${prefix}-FASTQ-a4-k31-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
endfunction(add_dist_hashmap_target)


function(add_dist_hashmap_kmer_target prefix k dna map disthash storehash extra extra2 group)

	if (NOT TARGET ${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
      add_executable(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerIndex.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=FASTQ -DpDNA=${dna} -DpK=${k} -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
       
		if (ENABLE_VTUNE_PROFILING)
			target_link_libraries(${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
		endif (ENABLE_VTUNE_PROFILING)
	
	endif(NOT TARGET ${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
	add_dependencies(${group} ${prefix}-FASTQ-a${dna}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
		
endfunction(add_dist_hashmap_kmer_target)


function(add_dist_counter_target prefix format k map disthash storehash extra extra2 group)
	if (NOT TARGET ${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})

      add_executable(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} BenchmarkKmerCounter.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
         PROPERTIES COMPILE_FLAGS 
         "-DpPARSER=${format} -DpDNA=4 -DpK=${k} -DpKmerStore=CANONICAL -DpMAP=${map} -DpINDEX=COUNT -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -D${extra} -D${extra2}")
      target_link_libraries(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash}
       ${EXTRA_LIBS})
	       
		if (ENABLE_VTUNE_PROFILING)
			target_link_libraries(${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash} ittnotify dl)
		endif (ENABLE_VTUNE_PROFILING)
		
	endif(NOT TARGET ${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
		
	add_dependencies(${group} ${prefix}-${format}-a4-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${disthash}-sh${storehash})
	
	
endfunction(add_dist_counter_target)


function(add_distht_target prefix reducer disthash storehash bits overlap prefetch group)

	if (NOT TARGET ${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits})
      add_executable(${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits} BenchmarkDistHashTables.cpp)      
      SET_TARGET_PROPERTIES(${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits}
         PROPERTIES COMPILE_FLAGS 
         "-DpINDEX=${reducer} -DpDistTrans=IDEN -DpDistHash=${disthash} -DpStoreHash=${storehash} -DpBits=${bits} -D${overlap} -D${prefetch}")
      target_link_libraries(${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits}
       ${EXTRA_LIBS})
       
		if (ENABLE_VTUNE_PROFILING)
			target_link_libraries(${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits} ittnotify dl)
		endif (ENABLE_VTUNE_PROFILING)
		
	endif(NOT TARGET ${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits})
	
	add_dependencies(${group} ${prefix}-${reducer}-dtIDEN-dh${disthash}-sh${storehash}-${bits})
	
endfunction(add_distht_target)


if (BUILD_BENCHMARKS)


# EXECUTABLES
	# benchmark executable for hash functions
	add_executable(benchmark_hashes BenchmarkHashFunctions.cpp)
	target_link_libraries(benchmark_hashes ${EXTRA_LIBS})
	if (ENABLE_VTUNE_PROFILING)
		target_link_libraries(benchmark_hashes ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
	
	add_dependencies(serial_benchmarks benchmark_hashes)
	

	# benchmark executable, FARM and MURMUR
	foreach(hash STD IDEN FARM FARM32 MURMUR MURMUR32 MURMUR32sse MURMUR32avx MURMUR64avx CRC32C CLHASH)
		add_hashmap_target(${hash} serial_benchmarks)
	endforeach(hash)
	
	# distributed benchmarks.
	foreach(map DENSEHASH)
		foreach(hash FARM MURMUR ) # MURMUR32 CLHASH) # no FARM because it was affected by prefetching.

			foreach(k 15 21 31 63)
				add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} ${hash} KH_DUMMY KH_DUMMY2 dist_benchmarks)
				add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} CRC32C KH_DUMMY KH_DUMMY2 dist_benchmarks)
			endforeach(k)
			foreach(dna 4 5 16)
				add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} ${hash} KH_DUMMY KH_DUMMY2 dist_benchmarks)
				add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} CRC32C KH_DUMMY KH_DUMMY2 dist_benchmarks)
			endforeach(dna)
		endforeach(hash)
	endforeach(map)
	
	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash MURMUR MURMUR32 MURMUR32avx MURMUR64avx CLHASH)  # for non-overlapped io, put the fastest hash function with the local hash table (CRC32C)
			# with prefetch
			foreach(k 15 21 31 63)
					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)
					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)
			endforeach(k)
			foreach(dna 4 5 16)
					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)
					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)
			endforeach(dna)

		endforeach(hash)
		add_dist_hashmap_target(testKmerIndex ${map} FARM FARM KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)					
	endforeach(map)

	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash MURMUR MURMUR32 MURMUR32avx MURMUR64avx) # CLHASH)  // for non-overlapped io, put the fastest hash function with the local hash table (CRC32C)
			# with prefetch
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH prefetch_benchmarks)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH prefetch_benchmarks)

			# without prefetch		
			add_dist_hashmap_target(noPref_KmerIndex ${map} ${hash} ${hash} KH_DUMMY1 KH_DUMMY2 prefetch_benchmarks)
			add_dist_hashmap_target(noPref_KmerIndex ${map} ${hash} CRC32C KH_DUMMY1 KH_DUMMY2 prefetch_benchmarks)

		endforeach(hash)
		add_dist_hashmap_target(testKmerIndex ${map} FARM FARM KH_DUMMY1 ENABLE_PREFETCH dist_benchmarks)					
	endforeach(map)


	foreach(map BROBINHOOD RADIXSORT)
		foreach(hash MURMUR MURMUR64avx CLHASH)  # For overlapped comm, need good global estimate, so need 64 bit.
	
			#with overlap and prefetch
#			foreach(k 15 21 31 63)
#					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(testKmerIndex ${k} 4 ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(overlap-KmerIndex ${k} 4 ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(overlap-KmerIndex ${k} 4 ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
#			endforeach(k)
#			foreach(dna 4 5 16)
#					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(testKmerIndex 31 ${dna} ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(overlap-KmerIndex 31 ${dna} ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
#					add_dist_hashmap_kmer_target(overlap-KmerIndex 31 ${dna} ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
#			endforeach(dna)

			# no overlap
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
			
			#overlap comm 1 rank-pair at a time.
			add_dist_hashmap_target(overlap-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
			add_dist_hashmap_target(overlap-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
	
			#overlapped comm using full array
			add_dist_hashmap_target(overlapFull-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_FULLBUFFER ENABLE_PREFETCH overlap_benchmarks)
			add_dist_hashmap_target(overlapFull-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM_FULLBUFFER ENABLE_PREFETCH overlap_benchmarks)
		
			#overlapped comm 2 phase.
			add_dist_hashmap_target(overlap2P-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_2P ENABLE_PREFETCH overlap_benchmarks)
			add_dist_hashmap_target(overlap2P-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM_2P ENABLE_PREFETCH overlap_benchmarks)
	
			#overlapped comm in batch.
			add_dist_hashmap_target(overlapBatch-KmerIndex ${map} ${hash} ${hash} OVERLAPPED_COMM_BATCH ENABLE_PREFETCH overlap_benchmarks)
			add_dist_hashmap_target(overlapBatch-KmerIndex ${map} ${hash} CRC32C OVERLAPPED_COMM_BATCH ENABLE_PREFETCH overlap_benchmarks)
	
	
		endforeach(hash)
	endforeach(map)

	# no overlap
	add_dist_hashmap_target(testKmerIndex MTROBINHOOD MURMUR64avx MURMUR64avx KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
	add_dist_hashmap_target(testKmerIndex MTROBINHOOD MURMUR64avx CRC32C KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
	
	#overlap comm 1 rank-pair at a time.
	add_dist_hashmap_target(overlap-KmerIndex MTROBINHOOD MURMUR64avx MURMUR64avx OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
	add_dist_hashmap_target(overlap-KmerIndex MTROBINHOOD MURMUR64avx CRC32C OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)

	# no overlap
	add_dist_hashmap_target(testKmerIndex MTRADIXSORT MURMUR64avx MURMUR64avx KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
	add_dist_hashmap_target(testKmerIndex MTRADIXSORT MURMUR64avx CRC32C KH_DUMMY1 ENABLE_PREFETCH overlap_benchmarks)
	
	#overlap comm 1 rank-pair at a time.
	add_dist_hashmap_target(overlap-KmerIndex MTRADIXSORT MURMUR64avx MURMUR64avx OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)
	add_dist_hashmap_target(overlap-KmerIndex MTRADIXSORT MURMUR64avx CRC32C OVERLAPPED_COMM ENABLE_PREFETCH overlap_benchmarks)


	foreach(map BROBINHOOD RADIXSORT) # just Batched Robinhood should get the point across.
		# lz4 distributed benchmarks.
		foreach(hash MURMUR32avx MURMUR64avx)
					# no lz4
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} ${hash} KH_DUMMY1 ENABLE_PREFETCH lz4_benchmarks)
			add_dist_hashmap_target(testKmerIndex ${map} ${hash} CRC32C KH_DUMMY1 ENABLE_PREFETCH lz4_benchmarks)
		
		 add_dist_hashmap_target(testLZ4KmerIndex ${map} ${hash} ${hash} ENABLE_LZ4_RESULT ENABLE_PREFETCH lz4_benchmarks)
		 add_dist_hashmap_target(testLZ4KmerIndex ${map} ${hash} CRC32C ENABLE_LZ4_RESULT ENABLE_PREFETCH lz4_benchmarks)
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



# kmer counter builds for shmem benchmark
foreach(map BROBINHOOD RADIXSORT MTROBINHOOD MTRADIXSORT)
	foreach(hash MURMUR32avx MURMUR64avx CLHASH) # MURMUR)  #  this is not using overlapped IO, so can use MURMUR32avx.
		add_dist_counter_target(testKmerCounter FASTA 31 ${map} ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
		add_dist_counter_target(testKmerCounter FASTA 31 ${map} ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
		add_dist_counter_target(testKmerCounter FASTQ 31 ${map} ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
		add_dist_counter_target(testKmerCounter FASTQ 31 ${map} ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
	endforeach(hash)
endforeach(map)
foreach(hash FARM MURMUR)  #  MURMUR32 CLHASH this is not using overlapped IO, so can use MURMUR32avx.
	add_dist_counter_target(testKmerCounter FASTA 31 DENSEHASH ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
	add_dist_counter_target(testKmerCounter FASTA 31 DENSEHASH ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
	add_dist_counter_target(testKmerCounter FASTQ 31 DENSEHASH ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
	add_dist_counter_target(testKmerCounter FASTQ 31 DENSEHASH ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH shmem_benchmarks)
endforeach(hash)
	

#k scalability
foreach(map BROBINHOOD RADIXSORT)
	foreach(hash MURMUR32avx MURMUR64avx) # MURMUR CLHASH)  #  this is not using overlapped IO, so can use MURMUR32avx.
		foreach(k 15 21 23 27 31 39 47 55 63 65 95 127 159 191 223 255)
			add_dist_counter_target(testKmerCounter FASTQ ${k} ${map} ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH k_benchmarks)
			add_dist_counter_target(testKmerCounter FASTQ ${k} ${map} ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH k_benchmarks)
		endforeach(k)
	endforeach(hash)
endforeach(map)
foreach(hash FARM MURMUR) # MURMUR32 CLHASH)  #  this is not using overlapped IO, so can use MURMUR32avx.
	foreach(k 15 21 23 27 31 39 47 55 63 65 95 127 159 191 223 255)
		add_dist_counter_target(testKmerCounter FASTQ ${k} DENSEHASH ${hash} ${hash} KH_DUMMY ENABLE_PREFETCH k_benchmarks)
		add_dist_counter_target(testKmerCounter FASTQ ${k} DENSEHASH ${hash} CRC32C KH_DUMMY ENABLE_PREFETCH k_benchmarks)
	endforeach(k)
endforeach(hash)


# primitive type performance
#k scalability
foreach(index COUNT FIRST LAST)
	foreach(hash IDEN MURMUR32 CRC32C MURMUR32avx MURMUR32FINALIZERavx)  #  this is not using overlapped IO, so can use MURMUR32avx.
		add_distht_target(benchmarkHT ${index} ${hash} ${hash} 32 KH_DUMMY ENABLE_PREFETCH distht_benchmarks)
		add_distht_target(overlapHT ${index} ${hash} ${hash} 32 OVERLAPPED_COMM ENABLE_PREFETCH distht_benchmarks)
	endforeach(hash)
	# foreach(hash IDEN MURMUR CRC32C MURMUR64avx)  #  this is not using overlapped IO, so can use MURMUR32avx.
	# 	add_distht_target(benchmarkHT ${index} ${hash} ${hash} 64 KH_DUMMY ENABLE_PREFETCH distht_benchmarks)
	# 	add_distht_target(overlapHT ${index} ${hash} ${hash} 64 OVERLAPPED_COMM ENABLE_PREFETCH distht_benchmarks)
	# endforeach(hash)
endforeach(index)

# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/martinus/src")
include_directories("${EXT_PROJECTS_DIR}/martinus/src/3rdparty/google")
add_executable(martinus_benchmark_hashtables martinus_test_robinhood.cpp)
SET_TARGET_PROPERTIES(martinus_benchmark_hashtables
	PROPERTIES COMPILE_FLAGS "-DENABLE_PREFETCH")
target_link_libraries(martinus_benchmark_hashtables ${EXTRA_LIBS})

add_dependencies(serial_benchmarks martinus_benchmark_hashtables)

add_executable(benchmark_a2av benchmark_mpi_a2av.cpp)
target_link_libraries(benchmark_a2av ${EXTRA_LIBS})
add_dependencies(overlap_benchmarks benchmark_a2av)

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


if(NOT TARGET microbenchmarks)
	add_custom_target(microbenchmarks)
	add_dependencies(microbenchmarks benchmark_a2av)
	if (Boost_FOUND)
		add_dependencies(microbenchmarks cust_alloc)
	endif(Boost_FOUND)
endif()


endif(BUILD_BENCHMARKS)

if (BUILD_APPLICATIONS)

endif(BUILD_APPLICATIONS)

