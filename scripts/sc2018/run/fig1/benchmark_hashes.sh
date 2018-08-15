#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J benchmark_hashes     # Job name
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 1               # Total number of mpi tasks requested

#SBATCH -t 11:59:59       # Run time (hh:mm:ss) - 12 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

ROOTDIR=CHANGE_ME



module load binutils-2.26 gcc-5.3.0

DATE=`date +%Y%m%d-%H%M%S`
logdir=${ROOTDIR}/data/hashes
BINDIR=${ROOTDIR}/build/kmerhash

mkdir -p ${logdir}

cd ${logdir}

TIME_CMD="/usr/bin/time -v"

/usr/bin/numactl -H

#drop linux file system cache
#eval "sudo /usr/local/crashplan/bin/CrashPlanEngine stop"
#eval "/usr/local/sbin/drop_caches"

##================= now execute.

EXEC=${BINDIR}/bin/benchmark_hashes
exec_name=$(basename ${EXEC})

## ================ params
n=100000000
e=0 #all kmer sizes to test

for iter in 1 2 3
do

	logfile=${logdir}/${exec_name}.e${e}.c${n}.$iter.log


	# only execute if the file does not exist.
	if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
	then
    
    	# command to execute
    	cmd="$EXEC -e ${e} -c ${n}"
    	echo "COMMAND" > $logfile
    	echo $cmd >> $logfile
		echo "COMMAND: ${cmd}" 
	  	echo "LOGFILE: ${logfile}"
    			
    	# call the executable and save the results
    	echo "RESULTS" >> $logfile
    	eval "$cmd >> $logfile 2>&1"
    
	    echo "COMPLETED" >> $logfile
      	echo "$EXEC COMPLETED."
    
	else
	
		echo "$logfile exists and COMPLETED.  skipping."
	fi

done