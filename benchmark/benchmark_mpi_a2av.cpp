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
#include <ctime>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h> // usleep

#include "utils/benchmark_utils.hpp"
#include "kmerhash/mem_utils.hpp"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

// generate about 10000 elements, with +/- 5% variability
std::vector<size_t> generateCounts(size_t const & per_pair_elem_count, mxx::comm const & comm){
  srand(comm.rank());

  size_t var = per_pair_elem_count / 10;


  std::vector<size_t> counts;
  counts.reserve(comm.size());

  for (int i = 0; i < comm.size(); ++i) {
    counts.emplace_back(per_pair_elem_count - (var / 2) + (rand() % var));
  }

  return counts;
}


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

  if (comm_rank == 0) std::cout << "BENCHMARKING MPI_Alltoallv with " << argv[0] << " comm size = " << comm_size << std::endl << std::flush;

  {

    BL_BENCH_COLLECTIVE_START(bm, "alloc_src", comm);
    std::vector<size_t> send_counts = generateCounts(20000, comm);
    std::vector<size_t> send_displs(comm_size);

    size_t src_size = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));

    // ex scan
    std::partial_sum(send_counts.begin(), send_counts.begin() + send_counts.size() - 1, send_displs.begin() + 1);
    size_t * src = ::utils::mem::aligned_alloc<size_t>(src_size);
    BL_BENCH_END(bm, "alloc_src", src_size);

    BL_BENCH_COLLECTIVE_START(bm, "init_src", comm);
    srand(comm_rank);
    for (size_t i = 0; i < src_size; ++i) {
      src[i] = static_cast<size_t>(rand()) << 32 | static_cast<size_t>(rand());
    }
    BL_BENCH_END(bm, "init_src", src_size);

    BL_BENCH_COLLECTIVE_START(bm, "alloc_dest", comm);
    std::vector<size_t> recv_counts = mxx::all2all(send_counts.data(), 1, comm);
    std::vector<size_t> recv_displs(comm_size);

    size_t dest_size = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // ex scan
    std::partial_sum(recv_counts.begin(), recv_counts.begin() + recv_counts.size() - 1, recv_displs.begin() + 1);
    size_t * gold = ::utils::mem::aligned_alloc<size_t>(dest_size);
    size_t * dest = ::utils::mem::aligned_alloc<size_t>(dest_size);
    BL_BENCH_END(bm, "alloc_dest", dest_size);

    ///////////////// -----------


    BL_BENCH_COLLECTIVE_START(bm, "mxx::a2av", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "mxx::alltoallv", comm);
      mxx::all2allv(src, send_counts, gold, recv_counts, comm);
      usleep(static_cast<unsigned int>(static_cast<double>(dest_size) * 0.025d));
      BL_BENCH_END(bm1, "mxx::alltoallv", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "mxx::a2av", comm);
    }
    BL_BENCH_END(bm, "mxx::a2av", src_size);

    ///////////////// -----------


    BL_BENCH_COLLECTIVE_START(bm, "a2av_isend", comm);
    {
      BL_BENCH_INIT(bm2);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_alloc", comm);
      // find the max recv size.
      size_t max_recv_n = *(std::max_element(recv_counts.begin(), recv_counts.end()));

      // allocate
      size_t * buff = ::utils::mem::aligned_alloc<size_t>(max_recv_n * 2);
      size_t * recv = buff;
      size_t * proc = buff + max_recv_n;
      BL_BENCH_END(bm2, "a2av_isend_alloc", max_recv_n * 2);


      // process self.
      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_self", comm);

      std::copy(src + send_displs[comm_rank], src + send_displs[comm_rank] + send_counts[comm_rank],
                dest + recv_displs[comm_rank]);

      BL_BENCH_END(bm2, "a2av_isend_self", send_counts[comm_rank]);

      // set up all the sends.
      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_setup", comm);
      mxx::datatype dt = mxx::get_datatype<size_t>();
      std::vector<MPI_Request> reqs(comm_size - 1);

      int step;
      int curr_peer = comm_rank;

      for (step = 1; step < comm_size; ++step) {
        // target rank
        curr_peer = (comm_rank + comm_size - step) % comm_size;

        // issend all, avoids buffering.
        MPI_Issend(src + send_displs[curr_peer], send_counts[curr_peer], dt.type(),
                   curr_peer, 1234, comm, &reqs[step - 1] );
      }

      int completed;
      MPI_Testall(comm_size - 1, reqs.data(), &completed, MPI_STATUSES_IGNORE);

      BL_BENCH_END(bm2, "a2av_isend_setup", comm_size - 1);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_proc", comm);

      MPI_Request req;
      size_t total = 0;
      int flag;

      for (step = 1; step < comm_size; ++step) {

        curr_peer = (comm_rank + step) % comm_size;

        MPI_Irecv(recv, recv_counts[curr_peer], dt.type(),
                  curr_peer, 1234, comm, &req );
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);


        if (step > 1) {
          curr_peer = (comm_rank + step - 1) % comm_size;
          std::copy(proc, proc + recv_counts[curr_peer],
                    dest + recv_displs[curr_peer]);
          total += recv_counts[curr_peer];
        }

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        ::std::swap(recv, proc);

      }
      if (comm_size > 1) {
		  // final piece.
		  curr_peer = (comm_rank + comm_size - 1) % comm_size;
		  usleep(static_cast<unsigned int>(static_cast<double>(recv_counts[curr_peer]) * 0.025d));
		  std::copy(proc, proc + recv_counts[curr_peer],
					dest + recv_displs[curr_peer]);
		  total += recv_counts[curr_peer];
      }
      BL_BENCH_END(bm2, "a2av_isend_proc", total);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_cleanup", comm);
      ::utils::mem::aligned_free(buff);
      BL_BENCH_END(bm2, "a2av_isend_cleanup", max_recv_n * 2);

      BL_BENCH_REPORT_MPI_NAMED(bm2, "a2av_isend", comm);
    }
    BL_BENCH_END(bm, "a2av_isend", dest_size);


    BL_BENCH_COLLECTIVE_START(bm, "compare", comm);
    bool eq = std::equal(dest, dest+dest_size, gold);
    //std::cout << "EQUAL ? " << (eq ? "YES" : "NO") << std::endl;
    BL_BENCH_END(bm, "compare", eq ? 1 : 0);


    ///////////////// -----------


    BL_BENCH_COLLECTIVE_START(bm, "a2av_isend2", comm);
    {
      BL_BENCH_INIT(bm2);

      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_order", comm);

      // split by node.
      mxx::comm node_comm = comm.split_shared();

      // assert to check that all nodes are the same size.
      int node_size = node_comm.size();
      int node_rank = node_comm.rank();
      assert(mxx::all_same(node_size, comm) && "Different size nodes");

      // === do these 2 steps to handle heterogeneous entries
      // for procs in each node, use the minimum rank as node id.
      int node_id = comm_rank;
      // note that the split assigns rank within node_comm, subcomm rank 0 has the lowest global rank.
      // so a broadcast within the node_comm is enough.
      mxx::bcast(node_id, 0, node_comm);
//      // allreduce within node_comm to get node id to all ranks in a node.
//      mxx::allreduce(node_id, mxx::min<int>(), node_comm);

      // get all the rank's node ids.  this will be used to order the global rank for traversal.
      std::tuple<int, int, int> node_coord = std::make_tuple(node_id, node_rank, comm_rank);
      // gathered results are in rank order.
      std::vector<std::tuple<int, int, int> > node_coords = mxx::allgather(node_coord, comm);
      // all ranks will have the array in the same order.

      // now shift by the current node_id and node rank, so that
      //     on each node, the array has the current node first.
      //     and within each node, the array has the targets ordered s.t. the current proc is first and relativel orders are preserved.
      ::std::for_each(node_coords.begin(), node_coords.end(),
    		  [&node_id, &comm_size, &node_size, &node_rank](std::tuple<int, int, int> & x){
    	  // circular shift around the global comm, shift by node_id
    	  std::get<0>(x) = (std::get<0>(x) + comm_size - node_id) % comm_size;
    	  // circular shift within a node comm, shift by node_rank
    	  std::get<1>(x) = (std::get<1>(x) + node_size - node_rank) % node_size;
      });

      // generate the forward order (for receive)
      // now sort, first by node then by rank.  no need for stable sort.
      // this is needed because rank may be scattered when mapped to nodes, so interleaved.
      // the second field should already be increasing order for each node.  so no sorting on
      //   by second field necessary, as long as stable sorting is used.
      ::std::sort(node_coords.begin(), node_coords.end(),
                         [](std::tuple<int, int, int> const & x, std::tuple<int, int, int> const & y){
        return (std::get<0>(x) == std::get<0>(y)) ? (std::get<1>(x) < std::get<1>(y)) : (std::get<0>(x) < std::get<0>(y));
      });
      // all ranks will have the array in the same order.  grouped by node id, and shifted by total size and node size

      // copy over the the forward ranks
      std::vector<int> forward_ranks;
      forward_ranks.reserve(comm_size);

      // so now all we have to do is traverse the reordered ranks in sequence
      if ((comm_rank == 1) ) std::cout << "rank " << comm_rank << ", forward: ";
      for (int i = 0; i < comm_size; ++i) {
    	  forward_ranks.emplace_back(std::get<2>(node_coords[i]));
    	  if ((comm_rank == 1) ) std::cout << std::get<2>(node_coords[i]) << ",";
      }
      if ((comm_rank == 1) ) std::cout << std::endl;


      // now generate the reverse order (for send).  first negate the node_id and node_rank
      ::std::for_each(node_coords.begin(), node_coords.end(),
    		  [&node_size, &comm_size](std::tuple<int, int, int> & x){
    	  // circular shift around the global comm, shift by node_id
    	  std::get<0>(x) = (comm_size - std::get<0>(x)) % comm_size;
    	  // circular shift within a node comm, shift by node_rank
    	  std::get<1>(x) = (node_size - std::get<1>(x)) % node_size;
      });
      ::std::sort(node_coords.begin(), node_coords.end(),
              [](std::tuple<int, int, int> const & x, std::tuple<int, int, int> const & y){
    	  return (std::get<0>(x) == std::get<0>(y)) ? (std::get<1>(x) < std::get<1>(y)) : (std::get<0>(x) < std::get<0>(y));
      });

      // copy over the the forward ranks
      std::vector<int> reverse_ranks;
      reverse_ranks.reserve(comm_size);

      // so now all we have to do is traverse the reordered ranks in sequence
      if ((comm_rank == 1)) std::cout << "rank " << comm_rank << ", reverse: ";
      for (int i = 0; i < comm_size; ++i) {
    	  reverse_ranks.emplace_back(std::get<2>(node_coords[i]));
    	  if ( (comm_rank == 1)) std::cout << std::get<2>(node_coords[i]) << ",";
      }
      if ( (comm_rank == 1)) std::cout << std::endl;

      BL_BENCH_END(bm2, "a2av_isend2_order", comm_size);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_alloc", comm);
      // find the max recv size.
      size_t max_recv_n = *(std::max_element(recv_counts.begin(), recv_counts.end()));

      // allocate
      size_t * buff = ::utils::mem::aligned_alloc<size_t>(max_recv_n * 2);
      size_t * recv = buff;
      size_t * proc = buff + max_recv_n;
      BL_BENCH_END(bm2, "a2av_isend2_alloc", max_recv_n * 2);


      size_t curr_peer = comm_rank;


      // process self.
      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_self", comm);
      curr_peer = forward_ranks[0];
      std::copy(src + send_displs[curr_peer], src + send_displs[curr_peer] + send_counts[curr_peer],
                dest + recv_displs[curr_peer]);
      BL_BENCH_END(bm2, "a2av_isend2_self", send_counts[comm_rank]);


      // set up all the sends.
      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_setup", comm);
      mxx::datatype dt = mxx::get_datatype<size_t>();
      std::vector<MPI_Request> reqs(comm_size - 1);

      int step;


      for (step = 1; step < comm_size; ++step) {
        // target rank
        curr_peer = reverse_ranks[step];

        // issend all, avoids buffering.
        MPI_Issend(src + send_displs[curr_peer], send_counts[curr_peer], dt.type(),
                   curr_peer, 1234, comm, &reqs[step - 1] );

      }

      int completed;
      MPI_Testall(comm_size - 1, reqs.data(), &completed, MPI_STATUSES_IGNORE);

      BL_BENCH_END(bm2, "a2av_isend2_setup", comm_size - 1);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_proc", comm);

      MPI_Request req;
      size_t total = 0;
      int flag;

      for (step = 1; step < comm_size; ++step) {

        curr_peer = forward_ranks[step];

        MPI_Irecv(recv, recv_counts[curr_peer], dt.type(),
                  curr_peer, 1234, comm, &req );
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);

        if (step > 1) {
          curr_peer = forward_ranks[step - 1];
          std::copy(proc, proc + recv_counts[curr_peer],
                    dest + recv_displs[curr_peer]);
          total += recv_counts[curr_peer];
        }

        MPI_Wait(&req, MPI_STATUS_IGNORE);


        ::std::swap(recv, proc);

      }
      // final piece.
      if (comm_size > 1) {
		  curr_peer = forward_ranks.back();
		  usleep(static_cast<unsigned int>(static_cast<double>(recv_counts[curr_peer]) * 0.025d));
		  std::copy(proc, proc + recv_counts[curr_peer],
					dest + recv_displs[curr_peer]);
		  total += recv_counts[curr_peer];
      }
      BL_BENCH_END(bm2, "a2av_isend2_proc", total);


      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_cleanup", comm);
      ::utils::mem::aligned_free(buff);
      BL_BENCH_END(bm2, "a2av_isend2_cleanup", max_recv_n * 2);



      BL_BENCH_REPORT_MPI_NAMED(bm2, "a2av_isend2", comm);
    }
    BL_BENCH_END(bm, "a2av_isend2", dest_size);



    BL_BENCH_COLLECTIVE_START(bm, "compare2", comm);
    eq = std::equal(dest, dest+dest_size, gold);
    //std::cout << "EQUAL2 ? " << (eq ? "YES" : "NO") << std::endl;
    BL_BENCH_END(bm, "compare2", eq ? 1 : 0);

    //////////////////////----------------

    BL_BENCH_COLLECTIVE_START(bm, "cleanup", comm);
    ::utils::mem::aligned_free(src);
    ::utils::mem::aligned_free(dest);
    ::utils::mem::aligned_free(gold);
    BL_BENCH_END(bm, "cleanup", src_size + dest_size);

    BL_BENCH_REPORT_MPI_NAMED(bm, "benchmark", comm);

  }



  comm.barrier();
}
