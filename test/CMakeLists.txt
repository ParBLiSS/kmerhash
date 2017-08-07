set(TEST_NAME kmerhash_tests)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")



if (BUILD_TESTS)



# load the testing:
if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)

#message(status, "TESTING IS ENABLED.")

    # get all files from ./unit
#    FILE(GLOB TEST_FILES unit/test_*.cpp)
#    kmerhash_add_test(${TEST_NAME} FALSE ${TEST_FILES})

    kmerhash_add_test(kmerhash_aux_filter_iter FALSE unit/test_aux_filter_iterator.cpp)
    kmerhash_add_test(hyperloglog64 FALSE unit/test_hyperloglog64.cpp)


    kmerhash_add_test(kmerhash_LP FALSE unit/test_hashmap_linearprobe_doubling.cpp)
    kmerhash_add_test(kmerhash_RH FALSE unit/test_hashmap_robinhood_doubling.cpp)
#    kmerhash_add_test(kmerhash_RH_NonCirc FALSE unit/test_hashmap_robinhood_doubling_noncirc.cpp)
    kmerhash_add_test(kmerhash_RH_Offsets FALSE unit/test_hashmap_robinhood_doubling_offsets.cpp)
	kmerhash_add_test(kmerhash_RH_Offsets2 FALSE unit/test_hashmap_robinhood_offsets.cpp)
    kmerhash_add_test(kmerhash_RH_Prefetch FALSE unit/test_hashmap_robinhood_prefetch.cpp)

    # get all mpi test files from ./test
#    FILE(GLOB MPI_TEST_FILES unit/mpi_test_*.cpp)
#    kmerhash_add_mpi_test(${TEST_NAME} FALSE ${MPI_TEST_FILES})
endif(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)


# EXECUTABLES
#add_executable(projection projection_lut.cpp)
#target_link_libraries(projection ${EXTRA_LIBS})


endif(BUILD_TESTS)


