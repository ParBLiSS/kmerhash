#!/bin/sh

mpirun -np 1 perf record -e instructions,cycles,cache-misses --call-graph lbr -- $@ : -np 63 $@
