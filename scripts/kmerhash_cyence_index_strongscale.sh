#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J compare_kmerhash_indices     # Job name
##SBATCH -o benchmark_fastq_load.o%j # Name of stdout output file(%j expands to jobId) 
##SBATCH -e benchmark_fastq_load.e%j # Name of stderr output file(%j expands to jobId)
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 16               # Total number of nodes requested (16 cores/node)
#SBATCH -n 256              # Total number of mpi tasks requested

#SBATCH -t 01:00:00       # Run time (hh:mm:ss) - 1.5 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

#PLEASE RUN AS ROOT

source /home/tpan/scripts/gcc-openmpi.sh

which mpirun

HOME_DIR=/home/tpan/work
DATA_DIR=${HOME_DIR}/data
LOCALTMP=${HOME_DIR}/tmp
OUT_DIR=${HOME_DIR}/tmp

DATE=`date +%Y%m%d-%H%M%S`
logdir=${HOME_DIR}/log/strongscale/cyence
mkdir -p ${logdir}
cd ${logdir}

TIME_CMD="/usr/bin/time -v"
MPIRUN_CMD="/opt/rit/app/openmpi/2.1.0/bin/mpirun"


##================= now execute.


BIN_DIR=/home/tpan/build/kmerhash-gcc-ompi/bin

dataset=bumblebee
datafile=${DATA_DIR}/${dataset}/${dataset}.fastq



#unset any OMP or numa stuff.
unset OMP_PROC_BIND
unset GOMP_CPU_AFFINITY
unset OMP_PLACES
export OMP_NUM_THREADS=1
K=31
F=1
HASH=MURMUR

cd $OUT_DIR

t=${SLURM_NTASKS}


#for iter in 1 2 3
#do


cpu_node_cores=8

for l in 0.9
do

for MAP in BROBINHOOD DENSEHASH # ROBINHOOD
do

	EXEC=${BIN_DIR}/testKmerIndex-FASTQ-a4-k31-CANONICAL-${MAP}-COUNT-dtIDEN-dh${HASH}-sh${HASH}

# kmerind
 
	exec_name=$(basename ${EXEC})

	OUT_PREFIX=${logdir}/kh-${MAP}-${HASH}-${dataset}-L${l}-P${t}.${iter}

	logfile=${OUT_PREFIX}.log

  # only execute if the file does not exist.
  if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
  then
    
    	# command to execute
   	cmd="$MPIRUN_CMD -np $t --map-by ppr:${cpu_node_cores}:socket --rank-by core --bind-to core $EXEC --max_load $l --min_load 0.2 -q 2 -F ${datafile}"
    	echo "COMMAND" >> $logfile
    	echo $cmd >> $logfile
  	  echo "COMMAND: ${cmd}" 
	  echo "LOGFILE: ${logfile}"
    			
    	# call the executable and save the results
    	echo "RESULTS" >> $logfile
	eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"
    
      echo "COMPLETED" >> $logfile
      echo "$exec_name COMPLETED."
      
  else
   
    echo "$logfile exists and COMPLETED.  skipping."
  fi

#kmerind

done
#MAP





done
#load



#done
#iter


