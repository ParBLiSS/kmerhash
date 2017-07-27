#!/bin/sh

perf record -e instructions,cycles,cache-misses --call-graph lbr -- $@
