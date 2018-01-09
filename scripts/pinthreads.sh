#!/bin/bash

# creates a core map list for pinning OpenMP threads to CPU cores in the 
# GNU's GOMP_CPU_AFFINITY or PGI's MPI_BLIST format 
# more info at https://gcc.gnu.org/onlinedocs/libgomp/index.html#toc_Environment-Variables

# please address any feedback to m.cuma at utah.edu

# There are the following presumptions:
# - we have 2 sockets per node
# - only task counts which will evenly distribute tasks and threads over sockets 
#   are supported (e.g. on 20 core dual socket node, 4 tasks 5 threads/tasks is OK
#   but 5 tasks 4 threads/task is not OK

# determine procs/threads per node
TPN=`echo $SLURM_STEP_TASKS_PER_NODE | cut -f 1 -d \(` # tasks per node
HTTPN=`cat /proc/cpuinfo | grep processor | wc -l`      # total number of (hyper)cores
NTPC=`lscpu | grep Thread | cut -d : -f 2 | tr -d " "` # number of threads per core
PPN=$((HTTPN/NTPC)) # physical cores per node 
NTHR=$((PPN/TPN)) # number of threads per task
CPT=$((NTHR))   # cores per task 
CPS=$((PPN/2))    # cores per socket - assume 2 procs/node

#echo NTPC $NTPC PPN $PPN $HTTPN HTTPN

if (( $PPN == $HTTPN )); then HTON="off"; else HTON="on"; fi
#echo CPS $CPS CPT $CPT
#echo $HTON

# find MPI process rank, first MPICH style
if [ -n "${PMI_RANK+1}" ]; then
  MY_RANK=$PMI_RANK
fi
# now OpenMPI style
if [ -n "${OMPI_COMM_WORLD_RANK+1}" ]; then
  MY_RANK=$OMPI_COMM_WORLD_RANK
fi
# else we don't know this MPI
if [ -n "${MY_RANK+1}" ]; then
  echo "Using unknown MPI"
  exit
fi

#determine NUMA mapping
MAP=(`numactl -H | grep "node 0 cpus" | cut -d : -f 2`)
if (( ${MAP[0]} == "0" && ${MAP[1]} == "1" )); then
 if (( $PMI_RANK == 0 )); then
   echo NUMA core mapping is sequential and hyperthreading is $HTON
 fi
 NUMAMAP=0
else
 if (( $PMI_RANK == 0 )); then
   echo NUMA core mapping is round-robin and hyperthreading is $HTON
 fi
 NUMAMAP=1
fi

# some convenience variables
SPACE=","
CTR=1
LOCRANK=$((PMI_RANK%TPN))    # local MPI rank on a node
LOCOFFSET=$((LOCRANK/2))  
LOCOFFSET=$((LOCOFFSET*CPT)) # core offset of local rank 2*N from rank N
#echo Local rank $LOCRANK offset $LOCOFFSET 
# create CORELIST mapping, first for round-robin NUMA core mapping
if (( $NUMAMAP == 1 )); then
  LOCOFFSET=$((LOCOFFSET*2))
  if (( $PMI_RANK % 2 == 0 )); then
    CORE=$((LOCOFFSET))
  else
    CORE=$((1+LOCOFFSET))
  fi
  CORELIST=$CORE
  let CORE=CORE+2
  while [ $CTR -lt $CPT ]; do
    CORELIST=$CORELIST$SPACE$CORE
    let CTR=CTR+1
    let CORE=CORE+2
  done
  # hypercores
  if [[ "$HTON" == "on" ]]; then
    CORE=$((CORE+(CPS-CPT)*2))
    CTR=0
    while [ $CTR -lt $CPT ]; do
      CORELIST=$CORELIST$SPACE$CORE
      let CTR=CTR+1
      let CORE=CORE+2
    done
  fi
else # here's CORELIST for sequential NUMA core mapping
  if (( $PMI_RANK % 2 == 0 )); then
    CORE=$((LOCOFFSET))
  else
    CORE=$((CPS+LOCOFFSET))
  fi
#  echo Local rank $LOCRANK offset $LOCOFFSET st. core $CORE 
  CORELIST=$CORE
  let CORE=CORE+1
  while [ $CTR -lt $CPT ]; do  # first fill the first hypercore set
    CORELIST=$CORELIST$SPACE$CORE
    let CTR=CTR+1
    let CORE=CORE+1
  done
  # hypercores
  if [[ "$HTON" == "on" ]]; then
    let CORE=CORE+CPS+CPS-CPT
    CTR=0
    while [ $CTR -lt $CPT ]; do  # now fil the second hypercore set
      CORELIST=$CORELIST$SPACE$CORE
      let CTR=CTR+1
      let CORE=CORE+1
    done
  fi
fi

# set the environment variables and run whatever needs to be run
export OMP_PROC_BIND=true #export MP_BIND="yes" # MP_BIND is equivalent to OMP_PROC_BIND
# PGI core list
export MP_BLIST=$CORELIST 
# Intel's explicit core list, but, Intel OMP also supports GOMP_CPU_AFFINITY
#export KMP_AFFINITY="explicit,proclist=[$CORELIST],verbose"
# GNU core list
export GOMP_CPU_AFFINITY=$CORELIST
echo MPI rank $PMI_RANK maps to cores $CORELIST
$*
