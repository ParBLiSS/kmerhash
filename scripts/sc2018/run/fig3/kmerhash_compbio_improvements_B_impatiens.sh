#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J kmerhash_improvements    # Job name
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 72              # Total number of mpi tasks requested

#SBATCH -t 24:00:00       # Run time (hh:mm:ss) - 1.5 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

ROOTDIR=CHANGE_ME


module load binutils-2.26 gcc-5.3.0 openmpi-1.10.2

which mpirun

DATA_DIR=${ROOTDIR}/data
LOCALTMP=${ROOTDIR}/tmp
OUT_DIR=${ROOTDIR}/tmp

DATE=`date +%Y%m%d-%H%M%S`
logdir=${ROOTDIR}/data/improvement/compbio_B_impatiens
mkdir -p ${logdir}
cd ${logdir}

TIME_CMD="/usr/bin/time -v"
MPIRUN_CMD="/usr/local/modules/openmpi/1.10.2/bin/mpirun"

/usr/bin/numactl -H

#drop cache
#eval "sudo /usr/local/crashplan/bin/CrashPlanEngine stop"
#eval "/usr/local/sbin/drop_caches"

##================= now execute.


BIN_DIR=${ROOTDIR}/build/kmerhash/bin

dataset=B_impatiens
datafile=${DATA_DIR}/${dataset}/all_1.fastq


#unset any OMP or numa stuff.
unset OMP_PROC_BIND
unset GOMP_CPU_AFFINITY
unset OMP_PLACES
export OMP_NUM_THREADS=1

K=31
dna=4
l=0.8

for pfd in 8 16
do 


	BIN_TYPE="Index"
	PARAMS="--max_load ${l} --min_load 0.2 --query_prefetch ${pfd} --insert_prefetch ${pfd} -b -q 2 -F ${datafile}"


	OUT_PREFIX=${OUTDIR}/kmerind

	cd $OUT_DIR


	for iter in 1 2 3
	do

		for t in 64 
		do

			if [ $t -le 4 ]; then
				cpu_node_cores=1
			else
				cpu_node_cores=$((t / 4))
			fi

			MPI_CMD="$MPIRUN_CMD -np $t --map-by ppr:${cpu_node_cores}:socket --rank-by core --bind-to core"


			MAP=DENSEHASH
			for hash in MURMUR FARM 
			do

				EXEC=${BIN_DIR}/testKmer${BIN_TYPE}-FASTQ-a${dna}-k${K}-CANONICAL-${MAP}-COUNT-dtIDEN-dh${hash}-sh${hash}

				exec_name=$(basename ${EXEC})
				logfile=${logdir}/${exec_name}.p${t}.l${l}.pfd${pfd}.${iter}.log

				# only execute if the file does not exist.
				if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
				then
				
					# command to execute
					cmd="$MPI_CMD $EXEC $PARAMS"
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



			for MAP in BROBINHOOD RADIXSORT
			do


				EXEC=${BIN_DIR}/noPref_Kmer${BIN_TYPE}-FASTQ-a${dna}-k${K}-CANONICAL-${MAP}-COUNT-dtIDEN-dhMURMUR-shMURMUR

				exec_name=$(basename ${EXEC})
				logfile=${logdir}/${exec_name}.p${t}.l${l}.pfd${pfd}.${iter}.log

				# only execute if the file does not exist.
				if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
				then
				
					# command to execute
					cmd="$MPI_CMD $EXEC $PARAMS"
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


				EXEC=${BIN_DIR}/testKmer${BIN_TYPE}-FASTQ-a${dna}-k${K}-CANONICAL-${MAP}-COUNT-dtIDEN-dhMURMUR-shMURMUR

				exec_name=$(basename ${EXEC})
				logfile=${logdir}/${exec_name}.p${t}.l${l}.pfd${pfd}.${iter}.log

				# only execute if the file does not exist.
				if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
				then
				
					# command to execute
					cmd="$MPI_CMD $EXEC $PARAMS"
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


				EXEC=${BIN_DIR}/testKmer${BIN_TYPE}-FASTQ-a${dna}-k${K}-CANONICAL-${MAP}-COUNT-dtIDEN-dhMURMUR64avx-shMURMUR64avx

				exec_name=$(basename ${EXEC})
				logfile=${logdir}/${exec_name}.p${t}.l${l}.pfd${pfd}.${iter}.log

				# only execute if the file does not exist.
				if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
				then
				
					# command to execute
					cmd="$MPI_CMD $EXEC $PARAMS"
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


				EXEC=${BIN_DIR}/testKmer${BIN_TYPE}-FASTQ-a${dna}-k${K}-CANONICAL-${MAP}-COUNT-dtIDEN-dhMURMUR64avx-shCRC32C
			
				exec_name=$(basename ${EXEC})
				logfile=${logdir}/${exec_name}.p${t}.l${l}.pfd${pfd}.${iter}.log

				# only execute if the file does not exist.
				if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
				then
				
					# command to execute
					cmd="$MPI_CMD $EXEC $PARAMS"
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
			#MAP

		done
		#t

	done
	#iter

done
#pfd
