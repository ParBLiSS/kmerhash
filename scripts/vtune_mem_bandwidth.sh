#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r ${1}.bw.vtune -- ${@:2} > ${1}.bw.raw.log
 
amplxe-cl -report summary -r ${1}.bw.vtune -report-output ${1}.bw.log
