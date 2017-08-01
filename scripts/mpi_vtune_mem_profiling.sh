#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

if [ $1 -le 4 ]; then
	cores_per_socket=1
else
	cores_per_socket=`expr $1 / 4`
fi

mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.log \
amplxe-cl -collect-with runsa -knob event-config=\
INST_RETIRED.ANY,\
CPU_CLK_UNHALTED.THREAD,\
MEM_UOPS_RETIRED.ALL_LOADS,\
MEM_UOPS_RETIRED.ALL_STORES,\
MEM_LOAD_UOPS_RETIRED.L1_HIT,\
MEM_LOAD_UOPS_RETIRED.L2_HIT,\
MEM_LOAD_UOPS_RETIRED.L3_HIT,\
MEM_LOAD_UOPS_RETIRED.L1_MISS,\
MEM_LOAD_UOPS_RETIRED.L2_MISS,\
MEM_LOAD_UOPS_RETIRED.L3_MISS,\
OFFCORE_RESPONSE:request=ALL_REQUESTS:response=LLC_MISS.LOCAL_DRAM \
-r ${2}.vtune -- ${@:3}


#amplxe-cl -report summary

