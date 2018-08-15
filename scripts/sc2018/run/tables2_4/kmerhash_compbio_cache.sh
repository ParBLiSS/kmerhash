#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J compare_kmerhash_indices     # Job name
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 1              # Total number of mpi tasks requested

#SBATCH -t 12:00:00       # Run time (hh:mm:ss) - 1.5 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

ROOTDIR=CHANGE_ME



module load binutils-2.26 gcc-5.3.0 openmpi-1.10.2

DATE=`date +%Y%m%d-%H%M%S`
logdir=${ROOTDIR}/cache
BIN_DIR=${ROOTDIR}/build/kmerhash/bin

mkdir -p ${logdir}

cd ${logdir}

TIME_CMD="/usr/bin/time -v"

/usr/bin/numactl -H

#drop cache
#eval "sudo /usr/local/crashplan/bin/CrashPlanEngine stop"
#eval "/usr/local/sbin/drop_caches"

##================= now execute.

#unset any OMP or numa stuff.
unset OMP_PROC_BIND
unset GOMP_CPU_AFFINITY
unset OMP_PLACES
export OMP_NUM_THREADS=1

K=31
dna=4
l=0.8
pfd=16
t=1
op=insert
hash=MURMUR
EXEC=${BIN_DIR}/benchmark_hashtables_$hash

for iter in 1 2 3
do


	for MAP in "std_unordered" "google_densehash" "kmerind" "linearprobe" "classic_robinhood" "robinhood_prefetch" "robinhood_offset_overflow" "radixsort"
	do

		for N in 11300000 22500000 45000000 90000000
		do 
			exec_name=$(basename ${EXEC})
			logprefix=${logdir}/${exec_name}.${MAP}.${op}.${hash}.p${t}.l${l}.d${pfd}.${N}.${iter}
			logfile=${logprefix}.log

			# only execute if the file does not exist.
			if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
			then

				# command to execute
				cmd="${ROOTDIR}/src/kmerhash/scripts/vtune_mem_profiling.sh ${logprefix} $EXEC --measured_op $op --map_type ${MAP} --max_load $l --min_load 0.2 --query_prefetch ${pfd} --insert_prefetch ${pfd} -c -Q 2 -N $N -R 8 "
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

		done
		#N

	done
	#MAP


done
#iter
