#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

if [ $1 -le 4 ]; then
	cores_per_socket=1
else
	cores_per_socket=`expr $1 / 4`
fi

# see https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/532405

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
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_4,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_8,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_16,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_32,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_64,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_128,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_256,\
MEM_TRANS_RETIRED.LOAD_LATENCY_GT_512,\
OFFCORE_RESPONSE:request=ALL_REQUESTS:response=LLC_MISS.LOCAL_DRAM \
-r ${2}.vtune -- \
mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.core \
${@:3} > ${2}.raw.log

amplxe-cl -report summary -r ${2}.vtune -report-output ${2}.sum.log
amplxe-cl -report hw-events -r ${2}.vtune -report-output ${2}.hw.log
amplxe-cl -report top-down -r ${2}.vtune -report-output ${2}.top.log



amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${2}.bw.vtune -- \
mpirun -np $1 --map-by ppr:${cores_per_socket}:socket --bind-to core --rank-by core --output-filename ${2}.core \
${@:3} > ${2}.bw.raw.log

amplxe-cl -report summary -r ${2}.bw.vtune -report-output ${2}.bw.log

