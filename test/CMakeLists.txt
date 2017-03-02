set(TEST_NAME kmerhash_tests)
include("${PROJECT_SOURCE_DIR}/cmake/Sanitizer.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/ExtraWarnings.cmake")



if (ENABLE_TESTING)




# load the testing:
if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)

message(status, "TESTING IS ENABLED.")

    # get all files from ./unit
    FILE(GLOB TEST_FILES unit/test_*.cpp)
    kmerhash_add_test(${TEST_NAME} FALSE ${TEST_FILES})
    # get all mpi test files from ./test
#    FILE(GLOB MPI_TEST_FILES unit/mpi_test_*.cpp)
#    kmerhash_add_mpi_test(${TEST_NAME} FALSE ${MPI_TEST_FILES})
endif(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/test/unit)

endif(ENABLE_TESTING)
