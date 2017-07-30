#!/bin/sh

perf record -e instructions,cycles,cache-misses --call-graph lbr -o $1 ${@:2} > ${1}.log
