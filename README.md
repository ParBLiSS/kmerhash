# README #


### What is this repository for? ###

This repository contains implementation of hash tables that are aimed to better support kmer indexing.  Specifically, the hash tables are based on open addressing and uses either linear probing or Robin Hood hashing.

The implemention is c++ 11 compliant and header only.


### How do I get set up? ###

* Summary of set up

The repository is organized as follows:
** "include/kmerhash" contains the implementations, in particular hashmap\_linearprobe\_doubling.hpp and hashmap\_robinhood\_doubling.hpp.
** "include/kmerhash/experimental" contains some experimental implementations.
** "test/unit" contains the unit tests for the clases.
** "benchmark" contains benchmarks, in particular BenchmarkHashTables.cpp is the primary benchmark tool.
** "ext" contains dependencies as git submodules. 

* Dependencies
This project depends on "kmerind".  It also requires a c++11 compliant compiler (4.8.4 or later) and cmake ver 2.8 or later.

To initialize the git submodules, invoke the following:

```
#!sh
cd {src}
git submodule init
git submodule update
cd ext/kmerind
git submodule init
git submodule update

```


* Configuration

To compile, first create a build directory, preferably outside of the source directory.

```
#!sh
mkdir {build}
cd {build}
cmake {src} -DENABLE_TESTING-ON -DENABLE_BENCHMARKING-ON -DBUILD_EXAMPLE_APPLICATIONS=ON
make

```

Alternatively, you can use ccmake. 

```
#!sh
mkdir {build}
cd {build}
ccmake {src}
make

```

* How to run tests

The unit test are located in the "test" subdirectory inside the build directory.  To run individual tests, the executable can be invoked directly.  To run all tests, use

```
#!sh
cd {build}
make test

```

* How to run tests

There is currently only 1 benchmark and it is hard coded to insert 100M elements with average 5x repeats, and query with 10M elements.  The following tests are run in sequence:  insert, find, count, erase, count.

To run the benchmark, invoke the executable below from the build directory root. 

```
#!sh
cd {build}
bin/benchmark_hashtables

```



### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

