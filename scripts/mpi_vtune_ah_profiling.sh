#!/bin/bash
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

#echo "0" | sudo tee /proc/sys/kernel/yama/ptrace_scope
#echo "0" | sudo tee /proc/sys/kernel/nmi_watchdog
#echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
rm -rf vtunes-ah


if [ $1 -eq 1 ]; then
	mpirun -np 1 --map-by core --bind-to core --rank-by core amplxe-cl -collect advanced-hotspots -r vtunes-ah -- ${@:2} 
else
	mpirun -np 1 --map-by core --bind-to core --rank-by core amplxe-cl -collect advanced-hotspots -r vtunes-ah -- ${@:2} : -np `expr $1 - 1` --map-by core --bind-to core --rank-by core ${@:2}
fi
amplxe-cl -report summary
amplxe-cl -report hotspots

#amplxe-cl -collect hpc-performance -- $@
#amplxe-cl -report summary
#amplxe-cl -report hotspots
