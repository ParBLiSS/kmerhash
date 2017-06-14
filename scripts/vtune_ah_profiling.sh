#!/bin/bash
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

#echo "0" | sudo tee /proc/sys/kernel/yama/ptrace_scope
#echo "0" | sudo tee /proc/sys/kernel/nmi_watchdog
#echo "0" | sudo tee /proc/sys/kernel/kptr_restrict

amplxe-cl -collect advanced-hotspots -- $1
amplxe-cl -report summary
amplxe-cl -report hotspots

amplxe-cl -collect hpc-performance -- $1
amplxe-cl -report summary
amplxe-cl -report hotspots
