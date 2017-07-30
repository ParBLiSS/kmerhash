#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

if [ $1 -eq 1 ]; then
        mpirun -np 1 --map-by core --bind-to core --rank-by core amplxe-cl -collect-with runsa -knob event-config=\
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
OFFCORE_RESPONSE:request=ALL_REQUESTS:response=LLC_MISS.LOCAL_DRAM\
-- ${@:2}

else
        mpirun -np 1 --map-by core --bind-to core --rank-by core amplxe-cl -collect-with runsa -knob even-config=\
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
OFFCORE_RESPONSE:request=ALL_REQUESTS:response=LLC_MISS.LOCAL_DRAM\
-- ${@:2} : -np `expr $1 - 1` --map-by core --bind-to core --rank-by core ${@:2}
fi
amplxe-cl -report summary

