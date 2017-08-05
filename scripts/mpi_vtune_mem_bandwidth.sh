#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

if [ $1 -le 4 ]; then
	cores_per_socket=1
else
	cores_per_socket=`expr $1 / 4`
fi 

# see https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/532405

amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2}.vtune \
mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.log \
${@:3}

amplxe-cl -report summary


