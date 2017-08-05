#!/bin/sh

if [ $1 -le 4 ]; then
	cores_per_socket=1
else
	cores_per_socket=`expr $1 / 4`
fi 

# see https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/532405


# collect data for all MPI processes at the same time.
perf record -e \
instructions,\
cycles,\
cache-misses,\
cache-references,\
L1-dcache-loads,\
L1-dcache-stores,\
L1-dcache-load-misses,\
L1-icache-load-misses,\
LLC-loads,\
LLC-load-misses,\
LLC-stores,\
LLC-store-misses,\
dTLB-loads,\
dTLB-load-misses,\
dTLB-stores,\
dTLB-store-misses,\
iTLB-loads,\
iTLB-load-misses,\
mem-loads,\
mem-stores \
--call-graph lbr -o ${2}.perf \
mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.log \
${@:3}

