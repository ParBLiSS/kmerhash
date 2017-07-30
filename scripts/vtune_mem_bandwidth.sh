#!/bin/sh
source /opt/intel/vtune_amplifier_xe_2017/amplxe-vars.sh

amplxe-cl -collect general-exploration -knob collect-memory-bandwidth=true -r $1 ${@:2} > ${1}.log

