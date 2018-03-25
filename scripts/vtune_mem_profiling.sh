#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

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
-r ${1}.vtune -- ${@:2} > ${1}.raw.log

amplxe-cl -report summary -r ${1}.vtune -report-output ${1}.sum.log
amplxe-cl -report hw-events -r ${1}.vtune -report-output ${1}.hw.log
amplxe-cl -report top-down -r ${1}.vtune -report-output ${1}.top.log

amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${1}.bw.vtune -- ${@:2} > ${1}.bw.raw.log

amplxe-cl -report summary -r ${1}.bw.vtune -report-output ${1}.bw.log
