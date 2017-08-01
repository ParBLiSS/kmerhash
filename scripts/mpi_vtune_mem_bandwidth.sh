#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh
cores_per_socket=`expr $1 / 4` 

if [ $1 -le 4 ]; then
        mpirun -np $1 --map-by ppr:1:socket --bind-to core --rank-by core --output-filename ${2}.log amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2} ${@:3}
else
        mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.log amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2} ${@:3}
        #mpirun -np 1 --map-by ppr:1:socket --bind-to core --rank-by core amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2} ${@:3} : -np `expr $1 - 1` --map-by ppr:1:socket --bind-to core --rank-by core ${@:3} > ${2}.log
fi
#amplxe-cl -report summary


