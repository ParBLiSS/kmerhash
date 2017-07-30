#!/bin/sh

if [ $1 -eq 1 ]; then
	echo "mpirun -np 1 --map-by core --bind-to core --rank-by core perf record -e instructions,cycles,cache-misses --call-graph lbr -o ${2} ${@:3} " > ${2}.log
	mpirun -np 1 --map-by ppr:1:socket --bind-to core --rank-by core perf record -e instructions,cycles,cache-misses --call-graph lbr -o ${2} ${@:3} >> ${2}.log
else 
	echo "mpirun -np 1 --map-by core --bind-to core --rank-by core perf record -e instructions,cycles,cache-misses --call-graph lbr -o ${2} ${@:3} : -np `expr $1 - 1` --map-by core --bind-to core --rank-by core ${@:3}" > ${2}.log
	mpirun -np 1 --map-by ppr:1:socket --bind-to core --rank-by core perf record -e instructions,cycles,cache-misses --call-graph lbr -o ${2} ${@:3} : -np `expr $1 - 1` --map-by ppr:1:socket --bind-to core --rank-by core ${@:3} >> ${2}.log
fi
