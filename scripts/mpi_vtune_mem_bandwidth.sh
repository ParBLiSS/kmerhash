#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

if [ $1 -eq 1 ]; then
        mpirun -np 1 --map-by ppr:1:socket --bind-to core --rank-by core amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2} ${@:3} > ${2}.log
else
        mpirun -np 1 --map-by ppr:1:socket --bind-to core --rank-by core amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2} ${@:3} : -np `expr $1 - 1` --map-by ppr:1:socket --bind-to core --rank-by core ${@:3} > ${2}.log
fi
amplxe-cl -report summary


