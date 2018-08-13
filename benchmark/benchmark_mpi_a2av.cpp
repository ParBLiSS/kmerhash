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

#include "bliss-config.hpp"

#include <cstdio>
#include <stdint.h>
#include <vector>
#include <ctime>
#include <stdlib.h>
#include <algorithm>
#include <unistd.h> // usleep
#include <future>  // future, async, etc.

#include "tclap/CmdLine.h"
#include "utils/tclap_utils.hpp"

#include "utils/benchmark_utils.hpp"
#include "kmerhash/mem_utils.hpp"

#include "kmerhash/incremental_mxx.hpp"

#include "mxx/env.hpp"
#include "mxx/comm.hpp"

// generate about 10000 elements, with +/- 5% variability
std::vector<size_t> generateCounts(size_t const & per_pair_elem_count, float const & window, mxx::comm const & comm){
  srand(comm.rank());

  size_t var = per_pair_elem_count * window;


  std::vector<size_t> counts;
  counts.reserve(comm.size());

  for (int i = 0; i < comm.size(); ++i) {
    counts.emplace_back(per_pair_elem_count - (var / 2) + (rand() % var));
  }

  return counts;
}

struct sim_work {
    size_t cycles;

    sim_work(size_t const & c = 20) : cycles(c) {}

    template <typename T>
    void operator()(T & x) {
       for (size_t i = 0; i < cycles; ++i) {
         x += i;
       }
    }
};

struct sim_work2 {
    size_t cycles;
    volatile size_t result;

    sim_work2(size_t const & c = 20) : cycles(c) {}

    template <typename T>
    void operator()(int rank, T const * start, T const * end) {
      for (T const * it = start; it != end; ++it) {
        T x = *it;
        for (size_t i = 0; i < cycles; ++i) {
          x += i;
        }

       result ^= x;
      }
    }
};


//template <typename T, typename OP>
//T * compute(T* buf, size_t const & cnt,
//		OP const & sw, T * dest, size_t * total) {
//
//	  // first get the proc
//    std::copy(buf, buf + cnt, dest);
//    ::std::for_each(dest, dest + cnt, sw);
//    *total += cnt;
//
//    return buf;  // return src buffer pointer for reuse.
//};

template <typename T, typename OP>
void compute(std::vector<std::promise<T *> > & proc_promises,
		std::vector<std::promise<T *> > & recv_promises,
		std::vector<int> const & forward_ranks,
		std::vector<size_t> const & counts,
		std::vector<size_t> const & displs,
		OP const & sw, T* dest, size_t & total) {

	size_t i = 0;
	size_t imax = (forward_ranks.size() > 2) ? (forward_ranks.size() - 2) : 0;
	T *d, *s;
	size_t cnt;
	int curr_peer;

    for (i = 0; i < imax; ++i) {  // note: iterator over all, 0 to comm_size - 3

      curr_peer = forward_ranks[i];
      cnt = counts[curr_peer];
      d = dest + displs[curr_peer];
//      if (forward_ranks[0] == 0) printf("compute reading proc_promises %ld\n", i);
      s = proc_promises[i].get_future().get();  // blocking until ready.

      // run it.
      std::copy(s, s + cnt, d);
      ::std::for_each(d, d + cnt, sw);
      total += cnt;

//      if (forward_ranks[0] == 0) printf("compute writing recv_promises %ld\n", i+1);
      recv_promises[i+1].set_value(s);   // set the first, so can start receiving
    }

    for (; i < forward_ranks.size(); ++i) {  // note: iterator over all, 0 to comm_size - 3

        curr_peer = forward_ranks[i];
        cnt = counts[curr_peer];
        d = dest + displs[curr_peer];
        //      if (forward_ranks[0] == 0) printf("compute reading proc_promises %ld\n", i);
        s = proc_promises[i].get_future().get();  // blocking until ready.

        // run it.
        std::copy(s, s + cnt, d);
        ::std::for_each(d, d + cnt, sw);
        total += cnt;

    }
}

int main(int argc, char** argv) {


  BL_BENCH_INIT(bm);

  BL_BENCH_START(bm);

  mxx::env e(argc, argv);
  mxx::comm comm;

  int comm_size = comm.size();
  int comm_rank = comm.rank();

  if (comm_size <= 1) {
	std::cout << "ERROR: this benchmark is meant to be run with more than 1 MPI processes." << std::endl;
	return 1;
	}

  comm.barrier();
  mxx::datatype dt = mxx::get_datatype<size_t>();
  BL_BENCH_END(bm, "mpi_init", comm_size);


  int work_cpe = 100;
  int message_size = 8192;
  float variance = 0.03;


  // Wrap everything in a try block.  Do this every time,
  // because exceptions will be thrown for problems.
  try {

    // Define the command line object, and insert a message
    // that describes the program. The "Command description message"
    // is printed last in the help text. The second argument is the
    // delimiter (usually space) and the last one is the version number.
    // The CmdLine object parses the argv array based on the Arg objects
    // that it contains.
    TCLAP::CmdLine cmd("MPI_AllToallv microbenchmark", ' ', "0.1");

    // MPI friendly commandline output.
    ::bliss::utils::tclap::MPIOutput cmd_output(comm);
    cmd.setOutput(&cmd_output);

    // Define a value argument and add it to the command line.
    // A value arg defines a flag and a type of value that it expects,
    // such as "-n Bishop".

    // output algo 7 and 8 are not working.
    TCLAP::ValueArg<int> msizeArg("m",
                                 "message-size", "total message size Default=8KB",
                                 false, message_size, "int", cmd);
    TCLAP::ValueArg<int> cpeArg("c",
                                 "work-cpe", "Cycle Per Element for computation, for simulating comm-compute overlap.  Default=20",
                                 false, work_cpe, "int", cmd);
    TCLAP::ValueArg<float> varArg("v",
                                     "variance", "range in which the number of elements for each rank-pair can vary, as percent of everage.  Default=0.03",
                                     false, variance, "float", cmd);

    // Parse the argv array.
    cmd.parse( argc, argv );

    message_size = msizeArg.getValue();
    work_cpe = cpeArg.getValue();
    variance = varArg.getValue();


  } catch (TCLAP::ArgException &e)  // catch any exceptions
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    exit(-1);
  }


  sim_work sw(work_cpe);
  sim_work2 sw2(work_cpe);

  if (comm_rank == 0) std::cout << "BENCHMARKING MPI_Alltoallv with " << argv[0] << " comm size = " << comm_size << std::endl << std::flush;

  {

    BL_BENCH_COLLECTIVE_START(bm, "alloc_src", comm);
    std::vector<size_t> send_counts = generateCounts(message_size / sizeof(size_t), variance, comm);
    std::vector<size_t> send_displs(comm_size);

    size_t src_size = std::accumulate(send_counts.begin(), send_counts.end(), static_cast<size_t>(0));

    // ex scan
    std::partial_sum(send_counts.begin(), send_counts.begin() + send_counts.size() - 1, send_displs.begin() + 1);
    size_t * src = ::utils::mem::aligned_alloc<size_t>(src_size);
    BL_BENCH_END(bm, "alloc_src", src_size);

    BL_BENCH_START(bm);
    srand(comm_rank);
    for (size_t i = 0; i < src_size; ++i) {
      src[i] = static_cast<size_t>(rand()) << 32 | static_cast<size_t>(rand());
    }
    BL_BENCH_END(bm, "init_src", src_size);

    BL_BENCH_START(bm);
    std::vector<size_t> recv_counts = mxx::all2all(send_counts.data(), 1, comm);
    std::vector<size_t> recv_displs(comm_size);

    size_t dest_size = std::accumulate(recv_counts.begin(), recv_counts.end(), static_cast<size_t>(0));
    // ex scan
    std::partial_sum(recv_counts.begin(), recv_counts.begin() + recv_counts.size() - 1, recv_displs.begin() + 1);
    size_t * gold = ::utils::mem::aligned_alloc<size_t>(dest_size);
    BL_BENCH_END(bm, "alloc_dest", dest_size);

    ///////////////// -----------


    BL_BENCH_COLLECTIVE_START(bm, "mxx::a2av", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "mxx::alltoallv", comm);
      mxx::all2allv(src, send_counts, gold, recv_counts, comm);
      BL_BENCH_END(bm1, "mxx::alltoallv", src_size);

      BL_BENCH_START(bm1);
      ::std::for_each(gold, gold + dest_size, sw);
      BL_BENCH_END(bm1, "compute", dest_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "mxx::a2av", comm);
    }
    BL_BENCH_END(bm, "mxx::a2av", src_size);

    ///////////////// -----------

    size_t * dest = ::utils::mem::aligned_alloc<size_t>(dest_size);


    BL_BENCH_COLLECTIVE_START(bm, "a2av_isend", comm);
    {
//      BL_TIMER_INIT(bm_loop);
      BL_BENCH_INIT(bm2);

      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend_alloc", comm);
      // find the max recv size.
      size_t max_recv_n = *(std::max_element(recv_counts.begin(), recv_counts.end()));

      // allocate
      size_t * buff = ::utils::mem::aligned_alloc<size_t>(max_recv_n * 2);
      size_t * recv = buff;
      size_t * proc = buff + max_recv_n;
      BL_BENCH_END(bm2, "a2av_isend_alloc", max_recv_n * 2);


      // set up all the sends.
      BL_BENCH_START(bm2);
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

      // BL_TIMER_START(bm_loop);
      comm.barrier();
      // BL_TIMER_END(bm_loop, "bar", 0);


      BL_BENCH_LOOP_START(bm2, 0);
      BL_BENCH_LOOP_START(bm2, 1);
      BL_BENCH_LOOP_START(bm2, 2);


      MPI_Request req;
      size_t total = 0;
      int flag;

      for (step = 1; step < comm_size; ++step) {

        // BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 0);
        curr_peer = (comm_rank + step) % comm_size;

        MPI_Irecv(recv, recv_counts[curr_peer], dt.type(),
                  curr_peer, 1234, comm, &req );
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        BL_BENCH_LOOP_PAUSE(bm2, 0);
        // BL_TIMER_END(bm_loop, "0", recv_counts[curr_peer]);


        if (step == 1) {
            // process self.
            // BL_TIMER_START(bm_loop);
            BL_BENCH_LOOP_RESUME(bm2, 1);
            std::copy(src + send_displs[comm_rank],
                      src + send_displs[comm_rank] + send_counts[comm_rank],
                      dest + recv_displs[comm_rank]);
            ::std::for_each(dest + recv_displs[comm_rank],
                dest + recv_displs[comm_rank] + recv_counts[comm_rank], sw);
            BL_BENCH_LOOP_PAUSE(bm2, 1);
            // BL_TIMER_END(bm_loop, "1", recv_counts[comm_rank]);

        } else {
          // BL_TIMER_START(bm_loop);
          BL_BENCH_LOOP_RESUME(bm2, 1);
          curr_peer = (comm_rank + step - 1) % comm_size;
          std::copy(proc, proc + recv_counts[curr_peer],
                    dest + recv_displs[curr_peer]);
          ::std::for_each(dest + recv_displs[curr_peer],
              dest + recv_displs[curr_peer] + recv_counts[curr_peer], sw);
          total += recv_counts[curr_peer];
          BL_BENCH_LOOP_PAUSE(bm2, 1);
          // BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);

        }

        // BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 2);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        ::std::swap(recv, proc);

        BL_BENCH_LOOP_PAUSE(bm2, 2);
        // BL_TIMER_END(bm_loop, "2", 0);


      }
      if (comm_size > 1) {
//        BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 1);

		  // final piece.
		  curr_peer = (comm_rank + comm_size - 1) % comm_size;
		  std::copy(proc, proc + recv_counts[curr_peer],
					dest + recv_displs[curr_peer]);
      ::std::for_each(dest + recv_displs[curr_peer],
          dest + recv_displs[curr_peer] + recv_counts[curr_peer], sw);
          total += recv_counts[curr_peer];
          BL_BENCH_LOOP_PAUSE(bm2, 1);
//          BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);

      }
      //BL_BENCH_END(bm2, "a2av_isend_proc", total);

      BL_BENCH_LOOP_END(bm2, 0, "loop_irecv", total);
      BL_BENCH_LOOP_END(bm2, 1, "loop_compute", total);
      BL_BENCH_LOOP_END(bm2, 2, "loop_wait", total  );



      BL_BENCH_START(bm2);
      ::utils::mem::aligned_free(buff);
      BL_BENCH_END(bm2, "a2av_isend_cleanup", max_recv_n * 2);

      BL_BENCH_REPORT_MPI_NAMED(bm2, "a2av_isend", comm);
//      BL_TIMER_REPORT_MPI_NAMED(bm_loop, "a2av_isend_loop", comm);
    }
    BL_BENCH_END(bm, "a2av_isend", dest_size);


    BL_BENCH_START(bm);
    bool eq = std::equal(dest, dest+dest_size, gold);
    //std::cout << "EQUAL ? " << (eq ? "YES" : "NO") << std::endl;
    ::utils::mem::aligned_free(dest);
    BL_BENCH_END(bm, "compare", eq ? 1 : 0);


    ///////////////// ============  algo 1.  should be same as above.





    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av1", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp1", comm);
      khmxx::incremental::ialltoallv_and_modify(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp1", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp1", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av1", src_size);



    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av2", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp2", comm);
      khmxx::incremental::ialltoallv_and_modify_fullbuffer(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp2", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp2", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av2", src_size);


    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av3", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp3", comm);
      khmxx::incremental::ialltoallv_and_modify_2phase(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp3", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp3", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av3", src_size);



    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av1B_8", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp1B_8", comm);
      khmxx::incremental::ialltoallv_and_modify_batch<8>(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp1B_8", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp1B_8", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av1B_8", src_size);



    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av1B_16", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp1B_16", comm);
      khmxx::incremental::ialltoallv_and_modify_batch<16>(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp1B_16", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp1B_16", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av1B_16", src_size);


    BL_BENCH_COLLECTIVE_START(bm, "khmxx::a2av1B_32", comm);
    {
      BL_BENCH_INIT(bm1);

      BL_BENCH_COLLECTIVE_START(bm1, "khmxx::a2av_comp1B_32", comm);
      khmxx::incremental::ialltoallv_and_modify_batch<32>(src, src + src_size, send_counts, sw2, comm);
      BL_BENCH_END(bm1, "khmxx::a2av_comp1B_32", src_size);

      BL_BENCH_REPORT_MPI_NAMED(bm1, "khmxx::a2av_comp1B_32", comm);
    }
    BL_BENCH_END(bm, "khmxx::a2av1B_32", src_size);



    ///////////////// -----------

    size_t * dest2 = ::utils::mem::aligned_alloc<size_t>(dest_size);

    BL_BENCH_COLLECTIVE_START(bm, "a2av_isend2", comm);
    {
      // BL_TIMER_INIT(bm_loop);
      BL_BENCH_INIT(bm2);

      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend2_order", comm);

//      // split by node.
//      mxx::comm node_comm = comm.split_shared();
//
//      // assert to check that all nodes are the same size.
//      int node_size = node_comm.size();
//      int node_rank = node_comm.rank();
//      assert(mxx::all_same(node_size, comm) && "Different size nodes");
//
//      // === do these 2 steps to handle heterogeneous entries
//      // for procs in each node, use the minimum rank as node id.
//      int node_id = comm_rank;
//      // note that the split assigns rank within node_comm, subcomm rank 0 has the lowest global rank.
//      // so a broadcast within the node_comm is enough.
//      mxx::bcast(node_id, 0, node_comm);
////      // allreduce within node_comm to get node id to all ranks in a node.
////      mxx::allreduce(node_id, mxx::min<int>(), node_comm);
//
//      // get all the rank's node ids.  this will be used to order the global rank for traversal.
//      std::tuple<int, int, int> node_coord = std::make_tuple(node_id, node_rank, comm_rank);
//      // gathered results are in rank order.
//      std::vector<std::tuple<int, int, int> > node_coords = mxx::allgather(node_coord, comm);
//      // all ranks will have the array in the same order.
//
//      // now shift by the current node_id and node rank, so that
//      //     on each node, the array has the current node first.
//      //     and within each node, the array has the targets ordered s.t. the current proc is first and relativel orders are preserved.
//      ::std::for_each(node_coords.begin(), node_coords.end(),
//    		  [&node_id, &comm_size, &node_size, &node_rank](std::tuple<int, int, int> & x){
//    	  // circular shift around the global comm, shift by node_id
//    	  std::get<0>(x) = (std::get<0>(x) + comm_size - node_id) % comm_size;
//    	  // circular shift within a node comm, shift by node_rank
//    	  std::get<1>(x) = (std::get<1>(x) + node_size - node_rank) % node_size;
//      });
//
//      // generate the forward order (for receive)
//      // now sort, first by node then by rank.  no need for stable sort.
//      // this is needed because rank may be scattered when mapped to nodes, so interleaved.
//      // the second field should already be increasing order for each node.  so no sorting on
//      //   by second field necessary, as long as stable sorting is used.
//      ::std::sort(node_coords.begin(), node_coords.end(),
//                         [](std::tuple<int, int, int> const & x, std::tuple<int, int, int> const & y){
//        return (std::get<0>(x) == std::get<0>(y)) ? (std::get<1>(x) < std::get<1>(y)) : (std::get<0>(x) < std::get<0>(y));
//      });
//      // all ranks will have the array in the same order.  grouped by node id, and shifted by total size and node size
//
//      // copy over the the forward ranks
//      std::vector<int> forward_ranks;
//      forward_ranks.reserve(comm_size);
//
//      // so now all we have to do is traverse the reordered ranks in sequence
////      if ((comm_rank == 1) ) std::cout << "rank " << comm_rank << ", forward: ";
//      for (int i = 0; i < comm_size; ++i) {
//    	  forward_ranks.emplace_back(std::get<2>(node_coords[i]));
////    	  if ((comm_rank == 1) ) std::cout << std::get<2>(node_coords[i]) << ",";
//      }
////      if ((comm_rank == 1) ) std::cout << std::endl;
//
//
//      // now generate the reverse order (for send).  first negate the node_id and node_rank
//      ::std::for_each(node_coords.begin(), node_coords.end(),
//    		  [&node_size, &comm_size](std::tuple<int, int, int> & x){
//    	  // circular shift around the global comm, shift by node_id
//    	  std::get<0>(x) = (comm_size - std::get<0>(x)) % comm_size;
//    	  // circular shift within a node comm, shift by node_rank
//    	  std::get<1>(x) = (node_size - std::get<1>(x)) % node_size;
//      });
//      ::std::sort(node_coords.begin(), node_coords.end(),
//              [](std::tuple<int, int, int> const & x, std::tuple<int, int, int> const & y){
//    	  return (std::get<0>(x) == std::get<0>(y)) ? (std::get<1>(x) < std::get<1>(y)) : (std::get<0>(x) < std::get<0>(y));
//      });
//
//      // copy over the the forward ranks
//      std::vector<int> reverse_ranks;
//      reverse_ranks.reserve(comm_size);
//
//      // so now all we have to do is traverse the reordered ranks in sequence
////      if ((comm_rank == 1)) std::cout << "rank " << comm_rank << ", reverse: ";
//      for (int i = 0; i < comm_size; ++i) {
//    	  reverse_ranks.emplace_back(std::get<2>(node_coords[i]));
////    	  if ( (comm_rank == 1)) std::cout << std::get<2>(node_coords[i]) << ",";
//      }
////      if ( (comm_rank == 1)) std::cout << std::endl;

      std::vector<int> forward_ranks;
      std::vector<int> reverse_ranks;

      khmxx::group_ranks_by_node(forward_ranks, reverse_ranks, comm);

      BL_BENCH_END(bm2, "a2av_isend2_order", comm_size);


      BL_BENCH_START(bm2);
      // find the max recv size.
      size_t max_recv_n = *(std::max_element(recv_counts.begin(), recv_counts.end()));

      // allocate
      size_t * buff = ::utils::mem::aligned_alloc<size_t>(max_recv_n * 2);
      size_t * recv = buff;
      size_t * proc = buff + max_recv_n;
      BL_BENCH_END(bm2, "a2av_isend2_alloc", max_recv_n * 2);


      size_t curr_peer = comm_rank;



      // set up all the sends.
      BL_BENCH_START(bm2);
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

      // BL_TIMER_START(bm_loop);
      comm.barrier();
      // BL_TIMER_END(bm_loop, "bar", 0);


      BL_BENCH_LOOP_START(bm2, 0);
      BL_BENCH_LOOP_START(bm2, 1);
      BL_BENCH_LOOP_START(bm2, 2);


      MPI_Request req;
      size_t total = 0;
      int flag;

      for (step = 1; step < comm_size; ++step) {

        // BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 0);

        curr_peer = forward_ranks[step];

        MPI_Irecv(recv, recv_counts[curr_peer], dt.type(),
                  curr_peer, 1234, comm, &req );
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        BL_BENCH_LOOP_PAUSE(bm2, 0);
        // BL_TIMER_END(bm_loop, "0", recv_counts[curr_peer]);

        if (step == 1) {
            // process self.
            // BL_TIMER_START(bm_loop);
            BL_BENCH_LOOP_RESUME(bm2, 1);
            curr_peer = forward_ranks[0];
            std::copy(src + send_displs[curr_peer],
                      src + send_displs[curr_peer] + send_counts[curr_peer],
                      dest2 + recv_displs[curr_peer]);
            ::std::for_each(dest2 + recv_displs[curr_peer],
                dest2 + recv_displs[curr_peer] + recv_counts[curr_peer], sw);
            BL_BENCH_LOOP_PAUSE(bm2, 1);
            // BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);

        } else {
          // BL_TIMER_START(bm_loop);
          BL_BENCH_LOOP_RESUME(bm2, 1);

          curr_peer = forward_ranks[step - 1];
          std::copy(proc, proc + recv_counts[curr_peer],
                    dest2 + recv_displs[curr_peer]);
          ::std::for_each(dest2 + recv_displs[curr_peer],
              dest2 + recv_displs[curr_peer] + recv_counts[curr_peer], sw);
          total += recv_counts[curr_peer];
          BL_BENCH_LOOP_PAUSE(bm2, 1);
          // BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);
        }

        // BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 2);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        BL_BENCH_LOOP_PAUSE(bm2, 2);
        // BL_TIMER_END(bm_loop, "2", 0);


        ::std::swap(recv, proc);

      }
      // final piece.
      if (comm_size > 1) {
        // BL_TIMER_START(bm_loop);
        BL_BENCH_LOOP_RESUME(bm2, 1);

		  curr_peer = forward_ranks.back();
		  std::copy(proc, proc + recv_counts[curr_peer],
					dest2 + recv_displs[curr_peer]);
      ::std::for_each(dest2 + recv_displs[curr_peer],
          dest2 + recv_displs[curr_peer] + recv_counts[curr_peer], sw);
		  total += recv_counts[curr_peer];
      BL_BENCH_LOOP_PAUSE(bm2, 1);
      // BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);


      }
//      BL_BENCH_END(bm2, "a2av_isend2_proc", total);


      BL_BENCH_LOOP_END(bm2, 0, "loop_irecv", total);
      BL_BENCH_LOOP_END(bm2, 1, "loop_compute", total);
      BL_BENCH_LOOP_END(bm2, 2, "loop_wait", total  );

      BL_BENCH_START(bm2);
      ::utils::mem::aligned_free(buff);
      BL_BENCH_END(bm2, "a2av_isend2_cleanup", max_recv_n * 2);



      BL_BENCH_REPORT_MPI_NAMED(bm2, "a2av_isend2", comm);
      // BL_TIMER_REPORT_MPI_NAMED(bm_loop, "a2av_isend2_loop", comm);

    }
    BL_BENCH_END(bm, "a2av_isend2", dest_size);



    BL_BENCH_START(bm);
    eq = std::equal(dest2, dest2+dest_size, gold);
    //std::cout << "EQUAL2 ? " << (eq ? "YES" : "NO") << std::endl;
    ::utils::mem::aligned_free(dest2);
    BL_BENCH_END(bm, "compare2", eq ? 1 : 0);

    //////////////////////----------------


    // version using openmp.
    size_t * dest3 = ::utils::mem::aligned_alloc<size_t>(dest_size);

    BL_BENCH_COLLECTIVE_START(bm, "a2av_isend3", comm);
    {
//      BL_TIMER_INIT(bm_loop);
      BL_BENCH_INIT(bm2);

      BL_BENCH_COLLECTIVE_START(bm2, "a2av_isend3_order", comm);

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
//      if ((comm_rank == 1) ) std::cout << "rank " << comm_rank << ", forward: ";
      for (int i = 0; i < comm_size; ++i) {
    	  forward_ranks.emplace_back(std::get<2>(node_coords[i]));
//    	  if ((comm_rank == 1) ) std::cout << std::get<2>(node_coords[i]) << ",";
      }
//      if ((comm_rank == 1) ) std::cout << std::endl;


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
//      if ((comm_rank == 1)) std::cout << "rank " << comm_rank << ", reverse: ";
      for (int i = 0; i < comm_size; ++i) {
    	  reverse_ranks.emplace_back(std::get<2>(node_coords[i]));
//    	  if ( (comm_rank == 1)) std::cout << std::get<2>(node_coords[i]) << ",";
      }
//      if ( (comm_rank == 1)) std::cout << std::endl;

      BL_BENCH_END(bm2, "a2av_isend3_order", comm_size);


      BL_BENCH_START(bm2);
      // find the max recv size.
      size_t max_recv_n = *(std::max_element(recv_counts.begin(), recv_counts.end()));

      // allocate
      size_t * buff = ::utils::mem::aligned_alloc<size_t>(max_recv_n * 2);
      size_t * recv = buff;
      size_t * proc = buff + max_recv_n;
      BL_BENCH_END(bm2, "a2av_isend3_alloc", max_recv_n * 2);


      size_t curr_peer = comm_rank;



      // set up all the sends.
      BL_BENCH_START(bm2);
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

      BL_BENCH_END(bm2, "a2av_isend3_send", comm_size - 1);

//      BL_TIMER_START(bm_loop);
//      comm.barrier();
//      BL_TIMER_END(bm_loop, "bar", 0);

      BL_TIMER_START(bm2);
//      BL_BENCH_LOOP_START(bm2, 0);
//      BL_BENCH_LOOP_START(bm2, 1);
//      BL_BENCH_LOOP_START(bm2, 2);


      size_t total = 0;

      std::vector<std::promise<size_t*> > proc_promises(comm_size);
      std::vector<std::promise<size_t*> > recv_promises(comm_size - 1);

      // init the first future, set it as recv.
      recv_promises[0].set_value(recv);   // set the first, so can start receiving

      // make it process self.
//            BL_TIMER_START(bm_loop);
//      BL_BENCH_LOOP_RESUME(bm2, 1);

      // and get the proc ready for first async
      curr_peer = forward_ranks[0];
      std::copy(src + send_displs[curr_peer],
    		    src + send_displs[curr_peer] + send_counts[curr_peer],
			    proc);

      proc_promises[0].set_value(proc);   // set the first, so can start receiving

      // get the second future with the first compute call
//      recv_futures.emplace_back(std::async(::std::launch::async,
//    		  compute<size_t, sim_work>, proc, recv_counts[curr_peer],
//			  sw, dest3 + recv_displs[curr_peer], &total));

      std::thread compute_thread(compute<size_t, sim_work>,
    		  std::ref(proc_promises),
    		  std::ref(recv_promises),
			  forward_ranks,
			  recv_counts, recv_displs, sw, dest3, std::ref(total));

//      BL_BENCH_LOOP_PAUSE(bm2, 1);
//            BL_TIMER_END(bm_loop, "1", recv_counts[curr_peer]);


      for (step = 1; step < comm_size; ++step) {

//        BL_TIMER_START(bm_loop);
//        BL_BENCH_LOOP_RESUME(bm2, 0);

        curr_peer = forward_ranks[step];

//        if (comm_rank == 0) printf("comm reading recv_promises %d\n", step-1);
        recv = recv_promises[step - 1].get_future().get();  // blocking until ready.

        MPI_Recv(recv, recv_counts[curr_peer], dt.type(), curr_peer, 1234, comm, MPI_STATUS_IGNORE );
//        BL_BENCH_LOOP_PAUSE(bm2, 0);
//        BL_TIMER_END(bm_loop, "0", recv_counts[curr_peer]);

//        if (comm_rank == 0) printf("comm writing proc_promises %d\n", step);
        proc_promises[step].set_value(recv);   // set the first, so can start receiving


        // once recived, async process.
//        BL_BENCH_LOOP_RESUME(bm2, 1);
//        recv_futures.emplace_back(std::async(::std::launch::async,
//      		  compute<size_t, sim_work>, recv, recv_counts[curr_peer],
//  			  sw, dest3 + recv_displs[curr_peer], &total));
//        BL_BENCH_LOOP_PAUSE(bm2, 1);

      }
      BL_BENCH_END(bm2, "a2av_isend3_recv", comm_size - 1);

      // now wait for thread join.
      compute_thread.join();
      BL_BENCH_END(bm2, "a2av_isend3_compute", total);


      BL_BENCH_START(bm2);
      ::utils::mem::aligned_free(buff);
      BL_BENCH_END(bm2, "a2av_isend3_cleanup", max_recv_n * 2);



      BL_BENCH_REPORT_MPI_NAMED(bm2, "a2av_isend3", comm);
//      BL_TIMER_REPORT_MPI_NAMED(bm_loop, "a2av_isend3_loop", comm);

    }

    BL_BENCH_END(bm, "a2av_isend3", dest_size);



    BL_BENCH_START(bm);
    eq = std::equal(dest3, dest3+dest_size, gold);
    //std::cout << "EQUAL2 ? " << (eq ? "YES" : "NO") << std::endl;
    ::utils::mem::aligned_free(dest3);
    BL_BENCH_END(bm, "compare3", eq ? 1 : 0);

    //////////////////////----------------




    ::utils::mem::aligned_free(src);
    ::utils::mem::aligned_free(gold);

    BL_BENCH_REPORT_MPI_NAMED(bm, "benchmark", comm);

  }



  comm.barrier();
}
