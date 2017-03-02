set(TEST_NAME kmerhash_test_apps)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")

if(ENABLE_TESTING)


endif(ENABLE_TESTING)




if (BL_BENCHMARK)


# EXECUTABLES
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/tommyds")
add_executable(benchmark_hashtables BenchmarkHashTables.cpp)
target_link_libraries(benchmark_hashtables ${EXTRA_LIBS})


endif(BL_BENCHMARK)

if (BUILD_TEST_APPLICATIONS)





if (USE_OPENMP)


endif(USE_OPENMP)



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




endif(BUILD_TEST_APPLICATIONS)

