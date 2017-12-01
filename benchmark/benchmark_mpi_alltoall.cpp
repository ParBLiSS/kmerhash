/*
 * Copyright 2015 Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <stdint.h>
#include <vector>

#include "utils/benchmark_utils.hpp"
#include "kmerhash/mem_utils.hpp"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

int main(int argc, char** argv) {

  BL_BENCH_INIT(bm);

  BL_BENCH_START(bm);

  mxx::env e(argc, argv);
  mxx::comm comm;

  int comm_size = comm.size();
  int comm_rank = comm.rank();

  comm.barrier();
  mxx::datatype dt = mxx::get_datatype<size_t>();
  BL_BENCH_END(bm, "mpi_init", comm_size);

  if (comm_rank == 0) std::cout << "BENCHMARKING MPI_Alltoall with " << argv[0] << " comm size = " << comm_size << std::endl << std::flush;

  {
    BL_BENCH_COLLECTIVE_START(bm, "alloc", comm);

    size_t * src = ::utils::mem::aligned_alloc<size_t>(comm_size);
    size_t * dest = ::utils::mem::aligned_alloc<size_t>(comm_size);

    BL_BENCH_END(bm, "alloc", comm_size);


    BL_BENCH_COLLECTIVE_START(bm, "alltoall", comm);
    MPI_Alltoall(src, 1, dt.type(), dest, 1, dt.type(), comm);
    BL_BENCH_END(bm, "alltoall", comm_size);

    BL_BENCH_COLLECTIVE_START(bm, "alltoall_inplace", comm);
    MPI_Alltoall(MPI_IN_PLACE, 1, dt.type(), dest, 1, dt.type(), comm);
    BL_BENCH_END(bm, "alltoall_inplace", comm_size);


    BL_BENCH_COLLECTIVE_START(bm, "mxx::alltoall", comm);
    mxx::all2all(src, 1, dest, comm);
    BL_BENCH_END(bm, "mxx::alltoall", comm_size);


  }

  {

    BL_BENCH_COLLECTIVE_START(bm, "vec alloc", comm);

    ::std::vector<size_t> src(comm_size);
    ::std::vector<size_t> dest(comm_size);

    BL_BENCH_END(bm, "vec alloc", comm_size);

    BL_BENCH_COLLECTIVE_START(bm, "vec_alltoall", comm);
    MPI_Alltoall(src.data(), 1, dt.type(), dest.data(), 1, dt.type(), comm);
    BL_BENCH_END(bm, "vec_alltoall", comm_size);


  }


  BL_BENCH_REPORT_MPI_NAMED(bm, "benchmark", comm);

  comm.barrier();
}
