#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J seq_bench_hashtables    # Job name
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 72              # Total number of mpi tasks requested

#SBATCH -t 11:59:59       # Run time (hh:mm:ss) - 12 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

ROOTDIR=CHANGE_ME


module load binutils-2.26 gcc-5.3.0

DATE=`date +%Y%m%d-%H%M%S`
logdir=${ROOTDIR}/data/benchmark_hashtables
BINDIR=${ROOTDIR}/build/kmerhash

mkdir -p ${logdir}

cd ${logdir}

TIME_CMD="/usr/bin/time -v"

/usr/bin/numactl -H

#drop cache
#eval "sudo /usr/local/crashplan/bin/CrashPlanEngine stop"
#eval "/usr/local/sbin/drop_caches"

for iter in 1 2 3
do

	l=0.8
	pfd=8

	for hash in FARM MURMUR32 MURMUR32avx MURMUR MURMUR64avx CRC32C
	do 

		##================= now execute.

		EXEC=${BINDIR}/bin/benchmark_hashtables_${hash}
		exec_name=$(basename ${EXEC})

		## ================ params


		logfile=${logdir}/${exec_name}.l${l}.pfd${pfd}.${iter}.log


		# only execute if the file does not exist.
		if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
		then
			
			# command to execute
			cmd="$EXEC"
			echo "COMMAND" > $logfile
			echo $cmd >> $logfile
			echo "COMMAND: ${cmd}" 
			echo "LOGFILE: ${logfile}"
					
			# call the executable and save the results
			echo "RESULTS" >> $logfile
			eval "$cmd -c -m all --max_load ${l} --insert_prefetch ${pfd} --query_prefetch ${pfd} >> $logfile 2>&1"
		
			echo "COMPLETED" >> $logfile
			echo "$EXEC COMPLETED."
			
		else
		
			echo "$logfile exists and COMPLETED.  skipping."
		fi

	done
	#hash


	hash=MURMUR64avx
	l=0.8
	for pfd in 2 4 8 16
	do 

		##================= now execute.

		EXEC=${BINDIR}/bin/benchmark_hashtables_${hash}
		exec_name=$(basename ${EXEC})

		## ================ params


	    logfile=${logdir}/${exec_name}.l${l}.pfd${pfd}.${iter}.log


		# only execute if the file does not exist.
		if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
		then
			
			# command to execute
			cmd="$EXEC"
			echo "COMMAND" > $logfile
			echo $cmd >> $logfile
			echo "COMMAND: ${cmd}" 
			echo "LOGFILE: ${logfile}"
					
			# call the executable and save the results
			echo "RESULTS" >> $logfile
			eval "$cmd -c -m all --max_load ${l} --insert_prefetch ${pfd} --query_prefetch ${pfd} >> $logfile 2>&1"
		
			echo "COMPLETED" >> $logfile
			echo "$EXEC COMPLETED."
      
		else
		
			echo "$logfile exists and COMPLETED.  skipping."
		fi

	done
	#pfd



	hash=MURMUR64avx
	pfd=8
	for l in 0.5 0.6 0.7 0.8 0.9
	do 

		##================= now execute.

		EXEC=${BINDIR}/bin/benchmark_hashtables_${hash}
		exec_name=$(basename ${EXEC})

		## ================ params


		logfile=${logdir}/${exec_name}.l${l}.pfd${pfd}.${iter}.log


		# only execute if the file does not exist.
		if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
		then
		
			# command to execute
			cmd="$EXEC"
			echo "COMMAND" > $logfile
			echo $cmd >> $logfile
			echo "COMMAND: ${cmd}" 
			echo "LOGFILE: ${logfile}"
					
			# call the executable and save the results
			echo "RESULTS" >> $logfile
			eval "$cmd -c -m all --max_load ${l} --insert_prefetch ${pfd} --query_prefetch ${pfd} >> $logfile 2>&1"
		
			echo "COMPLETED" >> $logfile
			echo "$EXEC COMPLETED."
		
		else
		
			echo "$logfile exists and COMPLETED.  skipping."
		fi

	done
	#l


done
#iter
