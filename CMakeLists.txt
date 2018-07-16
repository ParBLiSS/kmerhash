# CMakeLists.txt for the KmerHash

cmake_minimum_required(VERSION 2.8)


# project settings
project(kmerhash)


set (kmerhash_VERSION_MAJOR 0)
set (kmerhash_VERSION_MINOR 1)

INCLUDE(CMakeDependentOption)

##### General Compilation Settings

### REQUIRE C++11
# Initialize CXXFLAGS.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG") # -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g") # -funroll-loops -g")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wuninitialized --std=c11 -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG") # -funroll-loops")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -g") # -funroll-loops -g")


set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Compiler-specific C++11 activation.
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
# below 2 lines are only for cmake < 2.8.10
   execute_process(
     COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE BLISS_CXX_COMPILER_VERSION)
  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "16.0")
    message(FATAL_ERROR "Intel (icpc) compiler version 16.0 or later is required.")
  endif()

  #add_definitions(-Wabi)

  SET(SUPPORTS_SANITIZER OFF CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE OFF CACHE INTERNAL "compiler supports coverage?")

# intel dependency on gcc.  not compatible with gcc 5.  should search.
#  set(GCC_BINARY "gcc" CACHE PATH "Path to gcc binary. Requires version 4.9 or earlier.  Intel compiler uses GCC headers.")
#  EXECUTE_PROCESS( COMMAND ${GCC_BINARY} --version OUTPUT_VARIABLE gcc_full_version_string )
#  string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" GCC_VERSION ${gcc_full_version_string})
  
#  if (GCC_VERSION VERSION_GREATER 5.0.0 OR GCC_VERSION VERSION_EQUAL 5.0.0)
#    message(FATAL_ERROR "${PROJECT_NAME} is using Intel compiler, which requires gcc headers from version 4.9 or earlier.")
#  endif ()
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gcc-name=${GCC_BINARY}")

  SET(EXTRA_WARNING_FLAGS "-Wextra -Wno-unused-parameter -Wcheck")
  SET(TYPE_CONVERSION_WARNING_FLAGS "-Wconversion")
  SET(SUGGESTION_WARNING_FLAGS "-Wremarks")

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
# next 2 lines needed for cmake < 2.8.10
   EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE
   gcc_full_version_string )
   string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
BLISS_CXX_COMPILER_VERSION
   ${gcc_full_version_string})

  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "4.8.1")
    message(FATAL_ERROR "GNU CXX compiler version 4.8.1 or later is required.
Found ${CMAKE_CXX_COMPILER_VERSION}")
  endif()

  SET(SUPPORTS_SANITIZER ON CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE ON CACHE INTERNAL "compiler supports coverage?")
  
  # -Wzero-as-null-pointer-constant : causes a lot of errors in system headers.
  # -Wfloat-equal : most of comparisons are not with float.
  SET(EXTRA_WARNING_FLAGS "-Wextra -Wno-unused-parameter") #-Wundef 
  SET(TYPE_CONVERSION_WARNING_FLAGS "-Wdouble-promotion -Wconversion -Wsign-conversion -Wcast-qual -Wuseless-cast")
  SET(SUGGESTION_WARNING_FLAGS "-Wsuggest-override -Wsuggest-final-types -Wsuggest-final-methods")
  # gcc 4.9 colorize via: -fdiagnostics-color=always
  
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  # these 3 lines are needed because cmake < 2.8.10
  EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE
     clang_full_version_string )
  string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1"
   BLISS_CXX_COMPILER_VERSION ${clang_full_version_string})

  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "3.5")
    message(FATAL_ERROR "Clang++ compiler version 3.5 or later is required.")
  endif()
    
  #add_definitions(-Wabi)
 
  # intel dependency on gcc.  not compatible with gcc 5.  should search.
#  set(GCC_BINARY "/usr/bin/gcc" CACHE PATH "Path to gcc binary. Requires version 4.9 or earlier due to ABI change.  Clang uses GCC headers.")
#  EXECUTE_PROCESS( COMMAND ${GCC_BINARY} --version OUTPUT_VARIABLE gcc_full_version_string )
#  string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" GCC_VERSION ${gcc_full_version_string})
#  
#  if (GCC_VERSION VERSION_GREATER 5.0.0 OR GCC_VERSION VERSION_EQUAL 5.0.0)
#    message(FATAL_ERROR "${PROJECT_NAME} is using Clang compiler, which requires gcc headers from version 4.9 or earlier.")
#  endif ()
    
    # message(FATAL_ERROR "${PROJECT_NAME} currently does not support clang due to include path issues")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")   # do not use libc++ - may not be there . libstdc++ (from gnu) is a safer bet.
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -E")		      # for debugging to see how clang finds header files.
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp")  # not sure if needed for openmp. 
#  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fopenmp=libomp")  # not sure if needed for openmp. 
#  set(_CMAKE_TOOLCHAIN_PREFIX "llvm-")
  
  get_filename_component(CLANG_COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)
  # not needed
  # include_directories("${CLANG_COMPILER_DIR}/../lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")		      # this is here for SSE headers.
  # not needed  
  # include_directories("${CLANG_COMPILER_DIR}/../include")  # this is here for omp header, if one is installed.
  link_directories(${CLANG_COMPILER_DIR}/../lib)
  
  set(CMAKE_EXE_LINK_FLAGS "${CMAKE_EXE_LINK_FLAGS} -lstdc++")
  set(CMAKE_MODULE_LINK_FLAGS "${CMAKE_MODULE_LINK_FLAGS} -lstdc++")
  set(CMAKE_SHARED_LINK_FLAGS "${CMAKE_SHARED_LINK_FLAGS} -lstdc++")
  set(CMAKE_STATIC_LINK_FLAGS "${CMAKE_STATIC_LINK_FLAGS} -lstdc++")
  
  
  
  # clang uses gcc headers, and can find most by itself.
  SET(SUPPORTS_SANITIZER ON CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE ON CACHE INTERNAL "compiler supports coverage?")

  SET(EXTRA_WARNING_FLAGS "-Wextra -Wno-unused-parameter" CACHE INTERNAL "extra compiler warning flags")
  SET(TYPE_CONVERSION_WARNING_FLAGS "-W" CACHE INTERNAL "compiler flags to check type conversions")
  SET(SUGGESTION_WARNING_FLAGS CACHE INTERNAL "compiler flags that suggest keywords for better type resolutions")
  
else ()
  message(FATAL_ERROR "Your C++ compiler is not supported.")
endif ()
### from http://stackoverflow.com/questions/10984442/how-to-detect-c11-support-of-a-compiler-with-cmake


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
  set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)

  #for recognizing numbers and boolean constants in if()
  if(CMAKE_VERSION VERSION_GREATER 2.6.4)
    cmake_policy(SET CMP0012 NEW)
  endif()
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# Default: dynamic libraries
# ----------------------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)


###### Extra compiler flags and prettifying.
OPTION(COMPILER_EXTRA_WARNINGS "EXTRA COMPILATION OPTIONS" OFF)
CMAKE_DEPENDENT_OPTION(COMPILER_WARNINGS_ENABLE_EXTRA "Enable compiler to generate extra warnings?" OFF "COMPILER_EXTRA_WARNINGS" OFF)
CMAKE_DEPENDENT_OPTION(COMPILER_WARNINGS_ENABLE_CONVERSIONS "Enable compiler to generate extra warnings for type conversions?" OFF "COMPILER_WARNINGS" OFF)
CMAKE_DEPENDENT_OPTION(COMPILER_WARNINGS_ENABLE_SUGGESTIONS "Enable compiler to generate suggestions for polymorphic type analysis?" OFF "COMPILER_WARNINGS" OFF)
           
include("${PROJECT_SOURCE_DIR}/cmake/GCCFilter.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/CompilerProfiling.cmake")

###### LOGGING ENGINE configuration
SET(LOGGER_VERBOSITY "WARNING" CACHE STRING "Logger verbosity.  Each level includes PREVIOUS levels. ERROR is most terse.  TRACE is most verbose")  
set_property(CACHE LOGGER_VERBOSITY PROPERTY STRINGS ERROR WARNING INFO DEBUG TRACE)

set(LOG_ENGINE "PRINTF" CACHE STRING "choose a logging engine.")
set_property(CACHE LOG_ENGINE PROPERTY STRINGS NO_LOG PRINTF)

function(update_logging_engine varname cacheaccess varval) 
##       message(STATUS "${varname} changed to ${varval} with unhandled access: ${cacheaccess}.")
#
#
#    if (${cacheaccess} STREQUAL READ_ACCESS AND
#        ${varname} STREQUAL USE_OPENMP)
#
#            
#        if (USE_OPENMP)
#
#            # restrict log engine to no_log or printf
#            if (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "PRINTF")
#                
#                set(LOG_ENGINE ${LOG_ENGINE} CACHE STRING
#                "choose a logging engine.  options are NO_LOG PRINTF." FORCE)
#                
#            else (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "PRINTF")
#
##                message(STATUS "OMP ENABLED.  Default Log Engine set to NO_LOG")
#                
#                set(LOG_ENGINE "PRINTF" CACHE STRING
#                "choose a logging engine.  options are NO_LOG PRINTF." FORCE)
#                
#            endif (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "PRINTF")
#            
#        else (USE_OPENMP)
#            # OMP debugging is not on.  so log engine choice depends on whether boost logging is enabled.
#        
#          if (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "CERR" OR
#                LOG_ENGINE STREQUAL "PRINTF" OR
#                LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")
#                
#                
#            set(LOG_ENGINE ${LOG_ENGINE} CACHE STRING
#            "choose a logging engine.  options are NO_LOG PRINTF CERR BOOST_TRIVIAL BOOST_CUSTOM." FORCE)
#          else (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "CERR" OR
#                LOG_ENGINE STREQUAL "PRINTF" OR
#                LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")
#                
##                message(STATUS "OMP DISABLED.  Default Log Engine set to NO_LOG")
#                
#            set(LOG_ENGINE "NO_LOG" CACHE STRING
#            "choose a logging engine.  options are NO_LOG PRINTF CERR BOOST_TRIVIAL BOOST_CUSTOM." FORCE)
#          endif (LOG_ENGINE STREQUAL "NO_LOG" OR
#                LOG_ENGINE STREQUAL "CERR" OR
#                LOG_ENGINE STREQUAL "PRINTF" OR
#                LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")
#
#        endif(USE_OPENMP)
#
#    else(${cacheaccess} STREQUAL READ_ACCESS AND
#
#
#
#
#    
#        ${varname} STREQUAL USE_OPENMP)
#        
#       message(WARNING "${varname} changed to ${varval} with unhandled access: ${cacheaccess}.")
#        
#    endif(${cacheaccess} STREQUAL READ_ACCESS AND
#        ${varname} STREQUAL USE_OPENMP)
#        
        
    # set the log engine in the logger_config.hpp file.
    set(LOGGER_DEFINE "#define USE_LOGGER BLISS_LOGGING_${LOG_ENGINE}")

endfunction(update_logging_engine varname cacheaccess varval)
variable_watch(LOG_ENGINE update_logging_engine)


# if OMP_DEBUGGING is enabled, default(none) is set in the omp pragmas.   this means that only NO_LOG and PRINTF logging are compatible.

CMAKE_DEPENDENT_OPTION(OPENMP_STRICT_SCOPING "Enable OpenMP debugging (turns on default(none) in omp pragma)" ON
                        "OMP_DEBUGGING" OFF)
mark_as_advanced(OPENMP_STRICT_SCOPING)
if (OPENMP_STRICT_SCOPING)
    add_definitions(-DOMP_DEBUG)
endif(OPENMP_STRICT_SCOPING)
    




###### External Library Settings
#### Boost Library
#set(EXTRA_LIBS pthread)
#if (LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")
#    set(Boost_USE_MULTITHREADED ON)
#    set(Boost_ADDITIONAL_VERSIONS "1.54" "1.54.0")
#    set(Boost_DETAILED_FAILURE_MSG ON)
#    find_path(BOOST_ROOT NAMES include/boost/circular_buffer.hpp
#      HINTS /usr /usr/local /usr/local/share /opt/local ENV BOOST_HOME
#      DOC "path to Boost root directory")
#    # must specify the components otherwise it's a header only usage
#    find_package(Boost REQUIRED COMPONENTS system program_options log)
#    if (Boost_FOUND)
#      include_directories( ${Boost_INCLUDE_DIRS} )
#      link_directories( ${Boost_LIBRARY_DIRS} )
#      set(EXTRA_LIBS ${EXTRA_LIBS} ${Boost_LIBRARIES})
#      add_definitions(-DBOOST_LOG_DYN_LINK)
#    else(Boost_FOUND)
#      unset(Boost_USE_MULTITHREADED)
#      unset(Boost_ADDITIONAL_VERSIONS)
#      unset(Boost_DETAILED_FAILURE_MSG)
#      unset(BOOST_ROOT CACHE)
#
#        set(LOG_ENGINE "PRINTF" CACHE STRING
#            "choose a logging engine.  options are NO_LOG PRINTF CERR BOOST_TRIVIAL BOOST_CUSTOM." FORCE)
#        message(WARNING "Did not find boost.  Default Log Engine set to PRINTF")
#        
#        set(LOGGER_DEFINE "#define USE_LOGGER BLISS_LOGGING_${LOG_ENGINE}")
#
#    endif(Boost_FOUND)
#    
#    # need this flag for boost log
#else(LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")
#    unset(Boost_USE_MULTITHREADED)
#    unset(Boost_ADDITIONAL_VERSIONS)
#    unset(Boost_DETAILED_FAILURE_MSG)
#    unset(Boost_FOUND)
#    unset(BOOST_ROOT CACHE)
#    unset(Boost_DIR CACHE)
#    unset(Boost_INCLUDE_DIR CACHE)
#    unset(Boost_LIBRARY_DIR CACHE)
#    unset(Boost_LOG_LIBRARY_DEBUG CACHE)
#    unset(Boost_LOG_LIBRARY_RELEASE CACHE)
#    unset(Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG CACHE)
#    unset(Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE CACHE)
#    unset(Boost_SYSTEM_LIBRARY_DEBUG CACHE)
#    unset(Boost_SYSTEM_LIBRARY_RELEASE CACHE)
#
#endif(LOG_ENGINE STREQUAL "BOOST_TRIVIAL" OR
#                LOG_ENGINE STREQUAL "BOOST_CUSTOM")


#### MPI
OPTION(USE_MPI "Build with MPI support" ON)
if (USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  message(STATUS "Found MPI:")
  message(STATUS "    headers: ${MPI_INCLUDE_PATH}")
  message(STATUS "    libs:    ${MPI_LIBRARIES}")
  add_definitions(${MPI_COMPILE_FLAGS})
  set(CMAKE_EXE_LINK_FLAGS "${CMAKE_EXE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
  set(CMAKE_MODULE_LINK_FLAGS "${CMAKE_MODULE_LINK_FLAGS} ${MPI_LINK_FLAGS}")
  set(CMAKE_SHARED_LINK_FLAGS "${CMAKE_SHARED_LINK_FLAGS} ${MPI_LINK_FLAGS}")
  set(CMAKE_STATIC_LINK_FLAGS "${CMAKE_STATIC_LINK_FLAGS} ${MPI_LINK_FLAGS}")
else(USE_MPI)
  set(MPI_FOUND 0)
endif(USE_MPI)

if (MPI_FOUND)
  set(MPI_DEFINE "#define USE_MPI")
  set(EXTRA_LIBS ${EXTRA_LIBS} ${MPI_LIBRARIES})
  set(ENV{OMPI_CC} ${CMAKE_C_COMPILER})
  set(ENV{OMPI_CXX} ${CMAKE_CXX_COMPILER})
  mark_as_advanced(MPI_EXTRA_LIBRARY MPI_LIBRARY)
  
else (MPI_FOUND)
  set(MPI_DEFINE "")
  message(WARNING "Not using MPI")
endif (MPI_FOUND)

#### OpenMP
include(FindOpenMP)

if (OPENMP_FOUND)
    message(STATUS "Found OpenMP")
else(OPENMP_FOUND)
	message(STATUS "NO OpenMP.  check compiler version now.")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
    string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION ${clang_full_version_string})
  
   	if (CLANG_VERSION VERSION_GREATER 3.7 OR CLANG_VERSION VERSION_EQUAL 3.7)
	  	message(STATUS "Found OpenMP for CLANG ${CLANG_VERSION}")
	  	set(OPENMP_FOUND 1)
	  	set(OpenMP_C_FLAGS "-fopenmp=libomp")
	  	set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
	  	
   	    set(CLANG_OPENMP_HOME "${CLANG_COMPILER_DIR}/.." CACHE PATH "Path to Clang OpenMP root directory.  Ideally, same as clang home.")
   	    include_directories(${CLANG_OPENMP_HOME}/include)  # clang uses gcc headers, but can't find all gcc headers by itself.  this is for omp.h
   	    link_directories(${CLANG_OPENMP_HOME}/lib)
   	
	  else()
		set(OPNEMP_FOUND 0)
#	    message(FATAL_ERROR "${PROJECT_NAME} requires clang 3.7 or greater for OpenMP support.")
  	endif()
  endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang") 
endif(OPENMP_FOUND)

CMAKE_DEPENDENT_OPTION(USE_OPENMP "Build with OpenMP support" ON
                        "OPENMP_FOUND" OFF)
#variable_watch(USE_OPENMP update_logging_engine)

if (USE_OPENMP)
    # add OpenMP flags to compiler flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
else (USE_OPENMP)
    set(OPENMP_DEFINE "")
endif (USE_OPENMP)

#### native hardware architecture
OPTION(USE_SIMD_IF_AVAILABLE "Enable SIMD instructions, if available on hardware. (-march=native)" ON)
if (USE_SIMD_IF_AVAILABLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fabi-version=0")
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=0")    
    endif()
    
    add_definitions(-DUSE_SIMD)
endif(USE_SIMD_IF_AVAILABLE)



###### Doxygen documentation

find_package(Doxygen)

if(DOXYGEN_FOUND)
  message(STATUS "Found Doxygen")
  message(STATUS "    use `make kmerind-doc` to build documentation")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile @ONLY)
  add_custom_target(kmerind-doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)


############### TESTING
#  organization:  3 levels:  benchmark (coarse grain), profile (fine grain), debug (include coverage and sanitizer).  profile and debug are mutually exclusive.
### executable option organization:
# build_applications - useful applications
# build_tests -  both can be on.
# BUILD_BENCHMARKS - enables benchmark reporting at application level
### code analysis
# enable_code_analysis - mutually exclusive options
#	enable_profiling - mutually exclusive options
#		enable_vtune_profiling
#	enable_coverage
#	enable_sanitizer

###### EXECUTABLE CATEGORIES

# Check if the user want to build sample applications
OPTION(BUILD_APPLICATIONS "whether sample applications should be built" ON)
OPTION(BUILD_TESTS "whether unit tests should be built" ON)
OPTION(BUILD_BENCHMARKS "whether benchmarking tests should be built" ON)


###### CODE ANALYSIS FLAGS.  TODO: sanitizer flags for icc?  profiling flags for icc?
set(CODE_ANALYSIS_TYPE "disabled" CACHE STRING "Enable different testing mechanisms. options are disabled, coverage, sanitizer, profiling")
set_property(CACHE CODE_ANALYSIS_TYPE PROPERTY STRINGS disabled profiling coverage sanitizer)

set(ENABLE_CODE_ANALYSIS OFF CACHE INTERNAL "enabling code analysis")
set(ENABLE_PROFILING OFF CACHE INTERNAL "enabling profiling")	
set(ENABLE_COVERAGE OFF CACHE INTERNAL "enabling coverage")
set(ENABLE_SANITIZER OFF CACHE INTERNAL "enabling sanitizer")
set(SANITIZER_STYLE "${${SANITIZER_STYLE}}" CACHE INTERNAL "Disabled Sanitizer.")


find_library(VTUNE_LIB NAMES libittnotify.a HINTS /opt/intel/parallel_studio_xe_2017.3.053/vtune_amplifier_xe_2017/lib64)
if (NOT ${VTUNE_LIB} STREQUAL "VTUNE_LIB-NOTFOUND")
	get_filename_component(VTUNE_DIR ${VTUNE_LIB} PATH) # lib path
	get_filename_component(VTUNE_DIR ${VTUNE_DIR} PATH) # root path.
else()
	set(VTUNE_DIR "")
endif()
message(STATUS "vtune found at ${VTUNE_DIR}, ${VTUNE_LIB}")

function(update_analysis_options varname cacheaccess varval) 

	message(STATUS "changing ${varname} with value ${varval}")

    if (${cacheaccess} STREQUAL READ_ACCESS AND
        ${varname} STREQUAL CODE_ANALYSIS_TYPE)
            
        set(ENABLE_CODE_ANALYSIS ON CACHE INTERNAL "enabling code analysis")
        set(ENABLE_PROFILING OFF CACHE INTERNAL "enabling profiling")	
        set(ENABLE_COVERAGE OFF CACHE INTERNAL "enabling coverage")
        set(ENABLE_SANITIZER OFF CACHE INTERNAL "enabling sanitizer")
        set(SANITIZER_STYLE "${${SANITIZER_STYLE}}" CACHE INTERNAL "Disabled Sanitizer.")
        
        if (${varval} STREQUAL "disabled")
	        set(ENABLE_CODE_ANALYSIS OFF CACHE INTERNAL "enabling code analysis")
        elseif (${varval} STREQUAL "profiling")
	        set(ENABLE_PROFILING ON CACHE INTERNAL "enabling profiling")			
			OPTION(ENABLE_VTUNE_PROFILING "Turn on vtune specific profiling flags" OFF)
        elseif (${varval} STREQUAL "coverage")
	        if (SUPPORTS_COVERAGE)
		        set(ENABLE_COVERAGE ON CACHE INTERNAL "enabling coverage")
	        endif(SUPPORTS_COVERAGE)
        elseif (${varval} STREQUAL "sanitizer")
	        if (SUPPORTS_SANITIZER)
	            set(ENABLE_SANITIZER ON CACHE INTERNAL "enabling sanitizer")
	            set(SANITIZER_STYLE "address" CACHE STRING "Any Compiler supported Sanitizer style.  e.g. thread, address" FORCE)
				set_property(CACHE SANITIZER_STYLE PROPERTY STRINGS thread address)
	            
	        endif(SUPPORTS_SANITIZER)
        else()
            set(ENABLE_CODE_ANALYSIS OFF CACHE INTERNAL "enabling code analysis")	
        
            message(SEND_ERROR "Unknown code analysis type: ${varval}")
        endif()
                    
    endif(${cacheaccess} STREQUAL READ_ACCESS AND
        ${varname} STREQUAL CODE_ANALYSIS_TYPE)
                
endfunction(update_analysis_options varname cacheaccess varval)
variable_watch(CODE_ANALYSIS_TYPE update_analysis_options)

# NOT SURE WHY, BUT REMOVING THESE CAUSES THE VARIABLES NOT TO BE SET CORRECTLY.
message(STATUS "Code Analysis type = ${CODE_ANALYSIS_TYPE}")
message(STATUS "Code Analysis status = ${ENABLE_CODE_ANALYSIS}")
message(STATUS "sanitizer status = ${ENABLE_SANITIZER}")
message(STATUS "profiler status = ${ENABLE_PROFILING}")
message(STATUS "coverage status = ${ENABLE_COVERAGE}")

###### SUPPORTING FUNCTIONS.

# code analysis configuration

              
#### COMPILER based Profiling?
if(ENABLE_PROFILING)
#  add_definitions(-pg -g)
#else()
#  # Remove unreferenced functions: function level linking
#  if(NOT APPLE)
#    add_definitions(-ffunction-sections)
#  endif()

# VTUNE profiling
if (ENABLE_VTUNE_PROFILING)
	add_definitions(-DVTUNE_ANALYSIS)
	include_directories("${VTUNE_DIR}/include")
else(ENABLE_VTUNE_PROFILING)
	add_definitions(-pg -g)
endif(ENABLE_VTUNE_PROFILING)

endif()


## Prefetch configuration.
#OPTION(ENABLE_PREFETCH "enable prefetching in hashmap" ON)
#if (ENABLE_PREFETCH)
#    add_definitions(-DENABLE_PREFETCH)
#endif(ENABLE_PREFETCH)

## LZ4 configuration.
#OPTION(ENABLE_LZ4 "enable LZ4 compression during communication" ON)
#if (ENABLE_LZ4)
#    add_definitions(-DENABLE_LZ4_COMM)
#endif(ENABLE_LZ4)
#

# Reprobe configuration.
CMAKE_DEPENDENT_OPTION(REPORT_REPROBES "Report reprobe counts.  Only when ENABLE_PROFILING is set to off" OFF "NOT ENABLE_PROFILING" OFF)
if (REPORT_REPROBES)
    add_definitions(-DREPROBE_STAT)
endif(REPORT_REPROBES)



    if (ENABLE_COVERAGE)
      # set flags for coverage test
      message(STATUS "Code coverage reporting enabled")
#      add_definitions(-fprofile-arcs -ftest-coverage -g)
	  add_definitions(--coverage -g)
#      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -g")
#      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -g")
#      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -g")
#      set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -g")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage -g")
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --coverage -g")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage -g")
      set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} --coverage -g")
    endif()



###### Google test library
# require "git submodule init" and "git submodule update"
if (BUILD_TESTS)

  # GTEST use of pthreads does not appear to mix well with openmp code.
#  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    #clang compile with gtest have trouble with pthreads.
#    add_definitions(-DGTEST_HAS_PTHREAD=0)
#  endif()
  
  message(STATUS "Testing enabled")
  # include/load the Google Test framework
#  set_property(DIRECTORY ${EXT_PROJECTS_DIR}/gtest APPEND PROPERTY CMAKE_CXX_FLAGS -fPIC)
#  set_property(DIRECTORY ${EXT_PROJECTS_DIR}/gtest APPEND PROPERTY CMAKE_C_FLAGS -fPIC)
  add_subdirectory(${EXT_PROJECTS_DIR}/kmerind/ext/gtest)
  include_directories(${EXT_PROJECTS_DIR}/kmerind/ext/gtest/include)

  # enable the CMAKE testing framework
  enable_testing()

  # Set paths outside of the function so the function can
  # be called from anywhere in the source tree
  #   set path for output of testing binaries
  set(TEST_BINARY_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test)
  #   set path for output of test reports (XML format)
  set(TEST_XML_OUTPUT_DIR ${CMAKE_BINARY_DIR}/Testing)
endif(BUILD_TESTS)



# set here since timer needs to have this defined allways.
if (BUILD_BENCHMARKS)
  SET(BL_BENCHMARK 1)
else(BUILD_BENCHMARKS)
  SET(BL_BENCHMARK 0)
endif(BUILD_BENCHMARKS)

CMAKE_DEPENDENT_OPTION(ENABLE_TIME_BENCHMARK "Enable Time Benchmarking" ON
                        "BUILD_BENCHMARKS" OFF)
if (ENABLE_TIME_BENCHMARK)
  SET(BL_BENCHMARK_TIME 1)
else(ENABLE_TIME_BENCHMARK)
  SET(BL_BENCHMARK_TIME 0)
endif(ENABLE_TIME_BENCHMARK)
CMAKE_DEPENDENT_OPTION(ENABLE_MEMUSE_BENCHMARK "Enable Memory Usage Benchmarking" ON
                        "BUILD_BENCHMARKS" OFF)
if (ENABLE_MEMUSE_BENCHMARK)
  SET(BL_BENCHMARK_MEM 1)
else(ENABLE_MEMUSE_BENCHMARK)
  SET(BL_BENCHMARK_MEM 0)
endif(ENABLE_MEMUSE_BENCHMARK)
  


message(STATUS "VTUNE_ANALYSIS set to ${VTUNE_ANALYSIS}")
if (ENABLE_VTUNE_PROFILING)
	ADD_LIBRARY(ittnotify STATIC IMPORTED)
	SET_TARGET_PROPERTIES(ittnotify PROPERTIES
	    IMPORTED_LOCATION ${VTUNE_LIB})
endif(ENABLE_VTUNE_PROFILING)


                        
if (BUILD_TESTS)
  function(kmerhash_add_test module_name module_link)
    # set the name for the test target and executable
    set(test_target_name test-${module_name})

    # add all give files
    message(STATUS "adding gtest ${test_target_name} with files ${ARGN}") 

    add_executable(${test_target_name} ${ARGN})
    # add the googletest dependency
    # commented out because there is no googletest dependency    add_dependencies(${test_target_name} googletest)
    # set binary output path for tests
    set_target_properties(${test_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_OUTPUT_DIR})
    # link test with gtest_main
    target_link_libraries(${test_target_name} gtest_main)

    # all the extra libs
    target_link_libraries(${test_target_name} ${EXTRA_LIBS})

    # link to the tested module (but only if that module produces a linkable
    # library)
    if (module_link)
        target_link_libraries(${test_target_name} ${module_name})
    endif (module_link)
    
    # if code coverage is to be determined: link with gcov
    if (ENABLE_COVERAGE)
      target_link_libraries(${test_target_name} gcov)
    endif(ENABLE_COVERAGE)

	if (ENABLE_VTUNE_PROFILING)
	  target_link_libraries(${test_target_name} ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
	
    # generate google test XML results, to be parsed by Jenkins
    # additionally to the CTest generated xml, the gtest xml is much more detailed (listing all single tests)
    add_test(NAME ${test_target_name} WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND ${test_target_name} ARGS "--gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}.xml")
  endfunction(kmerhash_add_test)


  function(kmerhash_add_mpi_test module_name module_link)
      message(STATUS "adding gtest mpi tests ${module_name} with files ${ARGN}") 

    # set the name for the test target and executable
    foreach(CPP_FILE ${ARGN})
      # assume filename is of the format mpi_test_NAME.cpp -> extract NAME as the name for the executable
      get_filename_component(CPP_FILE_NAME ${CPP_FILE} NAME) 
      string(REPLACE "mpi_test_" "" CPP_FILE_SUFF ${CPP_FILE_NAME})
      string(REPLACE ".cpp" "" MPI_TEST_NAME ${CPP_FILE_SUFF})
      message(STATUS "MPI test module found: ${MPI_TEST_NAME}")
      set(test_target_name test-mpi-${module_name}-${MPI_TEST_NAME})
      
      
      # add all give files
      add_executable(${test_target_name} ${CPP_FILE})

      # add the googletest dependency
      # commented out because there is no googletest dependency    add_dependencies(${test_target_name} googletest)
      # set binary output path for tests
      set_target_properties(${test_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_OUTPUT_DIR})

	  set_target_properties(${test_target_name} PROPERTIES COMPILE_FLAGS -DGTEST_HAS_PTHREAD=0)

      # link test with gtest (no main, since it provides it's own main function)
      target_link_libraries(${test_target_name} gtest)

      # all the extra libs
      target_link_libraries(${test_target_name} ${EXTRA_LIBS})

      # link to the tested module (but only if that module produces a linkable
      # library)
      if (module_link)
          target_link_libraries(${test_target_name} ${module_name})
      endif (module_link)
      
      # if code coverage is to be determined: link with gcov
      if (ENABLE_COVERAGE)
        target_link_libraries(${test_target_name} gcov)
      endif(ENABLE_COVERAGE)

	if (ENABLE_VTUNE_PROFILING)
	  target_link_libraries(${test_target_name} ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)

      # generate google test XML results, to be parsed by Jenkins
      # additionally to the CTest generated xml, the gtest xml is much more detailed (listing all single tests)
      
      set (mpi_params_1 -np 1 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_1.xml)
      set (mpi_params_2 -np 2 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_2.xml)
      set (mpi_params_3 -np 3 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_3.xml)
      set (mpi_params_4 -np 4 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_4.xml)
#      set (mpi_params_7 -np 7 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_7.xml)
#      set (mpi_params_8 -np 8 ${TEST_BINARY_OUTPUT_DIR}/${test_target_name} --gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}_8.xml)
            
      add_test(NAME ${test_target_name}   WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND ${test_target_name} ARGS "--gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}.xml")
      add_test(NAME ${test_target_name}_1 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_1}  )
      add_test(NAME ${test_target_name}_2 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_2}  )
      add_test(NAME ${test_target_name}_3 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_3}  )
      add_test(NAME ${test_target_name}_4 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_4}  )
# these 2 take too long on travis.
#      add_test(NAME ${test_target_name}_7 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_7}  )
#      add_test(NAME ${test_target_name}_8 WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND mpirun ${mpi_params_8}  )
    endforeach()
  endfunction(kmerhash_add_mpi_test)

  function(kmerhash_add_non_google_test module_names module_link)
        message(STATUS "adding non google test ${module_names} with files ${ARGN}") 

      foreach(f ${ARGN})
        get_filename_component(test_name ${f} NAME_WE)
  
        # set the name for the test target and executable
        set(test_target_name ${test_name})
        
        # add all give files
        
        add_executable(${test_target_name} ${f})
        # set binary output path for tests
        set_target_properties(${test_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_OUTPUT_DIR})
    
        # all the extra libs
        target_link_libraries(${test_target_name} ${EXTRA_LIBS})
    
        # link to the tested module (but only if that module produces a linkable
        # library)
        if (module_link)
            target_link_libraries(${test_target_name} ${module_names})
        endif (module_link)
        
        # if code coverage is to be determined: link with gcov
        if (ENABLE_COVERAGE)
          target_link_libraries(${test_target_name} gcov)
        endif(ENABLE_COVERAGE)
    
    	if (ENABLE_VTUNE_PROFILING)
	  target_link_libraries(${test_target_name} ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
    
        # generate google test XML results, to be parsed by Jenkins
        # additionally to the CTest generated xml, the gtest xml is much more detailed (listing all single tests)
        add_test(NAME ${test_target_name} WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND ${test_target_name})
      endforeach(f)
  
  endfunction(kmerhash_add_non_google_test)

    function(kmerhash_add_metaprog_test module_names module_link filename)
      get_filename_component(test_name ${filename} NAME_WE)
      
      message(STATUS "adding metaprog gtest ${test_name} for file ${filename} and definitions ${ARGN}") 
      
      
      foreach(metadef ${ARGN})
  
        # set the name for the test target and executable
        set(test_target_name ${test_name}_${metadef})
        # add all give files
        add_executable(${test_target_name} ${filename})

        # set binary output path for tests
        set_target_properties(${test_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_OUTPUT_DIR})
        
        #get_target_property(targetCompDefs ${test_target_name} COMPILE_DEFINITIONS )
        set_target_properties(${test_target_name} PROPERTIES COMPILE_DEFINITIONS ${metadef} )

        # all the extra libs
        target_link_libraries(${test_target_name} ${EXTRA_LIBS})
    
        # link to the tested module (but only if that module produces a linkable
        # library)
        if (module_link)
            target_link_libraries(${test_target_name} ${module_names})
        endif (module_link)
        
        # if code coverage is to be determined: link with gcov
        if (ENABLE_COVERAGE)
          target_link_libraries(${test_target_name} gcov)
        endif(ENABLE_COVERAGE)
        
    	if (ENABLE_VTUNE_PROFILING)
	  target_link_libraries(${test_target_name} ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
    
        # generate google test XML results, to be parsed by Jenkins
        # additionally to the CTest generated xml, the gtest xml is much more detailed (listing all single tests)
        add_test(NAME ${test_target_name} WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND ${test_target_name})
      endforeach(metadef)
  
  endfunction(kmerhash_add_metaprog_test)
  

endif (BUILD_TESTS)


if (BUILD_BENCHMARKS)

  function(kmerhash_add_benchmark module_name module_link)
    # set the name for the test target and executable
    set(test_target_name benchmark-${module_name})
    # add all give files
    message(STATUS "adding gtest benchmark ${test_target_name} with files ${ARGN}") 

    add_executable(${test_target_name} ${ARGN})
    
    # add the googletest dependency
    # commented out because there is no googletest dependency    add_dependencies(${test_target_name} googletest)
    # set binary output path for tests
    set_target_properties(${test_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_OUTPUT_DIR})

    # disable pthreads for benchmark applications
	set_target_properties(${test_target_name} PROPERTIES COMPILE_FLAGS -DGTEST_HAS_PTHREAD=0)

    # link test with gtest_main
    target_link_libraries(${test_target_name} gtest_main)

    # all the extra libs
    target_link_libraries(${test_target_name} ${EXTRA_LIBS})

    # link to the tested module (but only if that module produces a linkable
    # library)
    if (module_link)
        target_link_libraries(${test_target_name} ${module_name})
    endif (module_link)
    
    # if code coverage is to be determined: link with gcov
    if (ENABLE_COVERAGE)
      target_link_libraries(${test_target_name} gcov)
    endif(ENABLE_COVERAGE)

	if (ENABLE_VTUNE_PROFILING)
	  target_link_libraries(${test_target_name} ittnotify dl)
	endif(ENABLE_VTUNE_PROFILING)
	
    # generate google test XML results, to be parsed by Jenkins
    # additionally to the CTest generated xml, the gtest xml is much more detailed (listing all single tests)
    add_test(NAME ${test_target_name} WORKING_DIRECTORY ${TEST_BINARY_OUTPUT_DIR} COMMAND ${test_target_name} ARGS "--gtest_output=xml:${TEST_XML_OUTPUT_DIR}/gtest/${test_target_name}.xml")
  endfunction(kmerhash_add_benchmark)

endif(BUILD_BENCHMARKS)


###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )


# configure a header file to pass some of the CMake settings to the source code
configure_file(
    "${PROJECT_SOURCE_DIR}/ext/kmerind/src/config/config.hpp.in"
    "${CMAKE_BINARY_DIR}/bliss-config.hpp"
    @ONLY
    )
configure_file(
    "${PROJECT_SOURCE_DIR}/ext/kmerind/src/config/logger_config.hpp.in"
    "${CMAKE_BINARY_DIR}/bliss-logger_config.hpp"
    @ONLY
    )


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
# this below gets the smhasher (murmur hash)
include_directories("${EXT_PROJECTS_DIR}") 
include_directories("${EXT_PROJECTS_DIR}/kmerind/src")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/Nadeau")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/mxx/include")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/mxx/ext")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/tclap-1.2.1/include")
include_directories("${EXT_PROJECTS_DIR}/kmerind/ext/sparsehash-c11")
include_directories("${EXT_PROJECTS_DIR}/lz4/lib")
include_directories("${EXT_PROJECTS_DIR}/clhash/include")


#for when the linker (ld) has problem finding librt, e.g. error: undefined referecne clock_gettime@@GLIBC_2.2.5.  this needs to be close to the end of the ld list.
#SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lrt" )
#SET( CMAKE_MODULE_LINKER_FLAGS  "${CMAKE_MODULE_LINKER_FLAGS} -lrt" )
#SET( CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} -lrt" )
#SET( CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -lrt" )

add_subdirectory(test)
add_subdirectory(benchmark)
add_subdirectory("${EXT_PROJECTS_DIR}/kmerind/ext/mxx")

# Install Settings


# Installer Settings


# Test Settings
