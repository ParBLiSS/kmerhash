set(TEST_NAME kmerhash_tests)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")



if (BUILD_TESTS)


if(NOT TARGET test_targets)
add_custom_target(test_targets)
endif(NOT TARGET test_targets)



# load the testing:
if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)

#message(status, "TESTING IS ENABLED.")

    # get all files from ./unit
#    FILE(GLOB TEST_FILES unit/test_*.cpp)
#    kmerhash_add_test(${TEST_NAME} FALSE ${TEST_FILES})

    kmerhash_add_test(kmerhash_aux_filter_iter FALSE unit/test_aux_filter_iterator.cpp)
    add_dependencies(test_targets test-kmerhash_aux_filter_iter)

    kmerhash_add_test(hyperloglog64 FALSE unit/test_hyperloglog64.cpp)
    add_dependencies(test_targets test-hyperloglog64)
    
    kmerhash_add_test(hash FALSE unit/test_kmer_hash.cpp)
    add_dependencies(test_targets test-hash)
    

    kmerhash_add_test(kmerhash_LP FALSE unit/test_hashmap_linearprobe_doubling.cpp)
    add_dependencies(test_targets test-kmerhash_LP)
    
    kmerhash_add_test(kmerhash_RH FALSE unit/test_hashmap_robinhood_doubling.cpp)
    add_dependencies(test_targets test-kmerhash_RH)
#    kmerhash_add_test(kmerhash_RH_NonCirc FALSE unit/test_hashmap_robinhood_doubling_noncirc.cpp)
#    kmerhash_add_test(kmerhash_RH_Offsets FALSE unit/test_hashmap_robinhood_doubling_offsets.cpp)
	kmerhash_add_test(kmerhash_RH_Offsets2 FALSE unit/test_hashmap_robinhood_offsets.cpp)
    add_dependencies(test_targets test-kmerhash_RH_Offsets2)
    kmerhash_add_test(kmerhash_RH_Prefetch FALSE unit/test_hashmap_robinhood_prefetch.cpp)
    add_dependencies(test_targets test-kmerhash_RH_Prefetch)
    
    # get all mpi test files from ./test
#    FILE(GLOB MPI_TEST_FILES unit/mpi_test_*.cpp)
#    kmerhash_add_mpi_test(${TEST_NAME} FALSE ${MPI_TEST_FILES})
endif(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)


# EXECUTABLES
#add_executable(projection projection_lut.cpp)
#target_link_libraries(projection ${EXTRA_LIBS})


endif(BUILD_TESTS)


