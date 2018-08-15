#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run hybrid applications 
# (MPI/OpenMP or MPI/pthreads) on TACC's Stampede 
# system.
#----------------------------------------------------
#SBATCH -J datascale_fasta     # Job name
##SBATCH -p normal    # Queue name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 72              # Total number of mpi tasks requested

#SBATCH -t 03:59:59       # Run time (hh:mm:ss) - 1.5 hours
# The next line is required if the user has more than one project
# #SBATCH -A A-yourproject  # <-- Allocation name to charge job against

# Set the number of threads per task(Default=1)

ROOTDIR=CHANGE_ME




module load binutils-2.26 gcc-5.3.0 openmpi-1.10.2 boost-1.61.0

which mpirun


DATA_DIR=${ROOTDIR}/data
LOCALTMP=${ROOTDIR}/tmp
OUT_DIR=${ROOTDIR}/tmp

DATE=`date +%Y%m%d-%H%M%S`
logdir=${ROOTDIR}/data/compbio-gerbil-datascale
mkdir -p ${logdir}/gerbil
mkdir -p ${logdir}/kmc3
mkdir -p ${logdir}/kmerind
mkdir -p ${logdir}/jf

cd ${logdir}

TIME_CMD="/usr/bin/time -v"
CACHE_CLEAR_CMD="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
MPIRUN_CMD="/usr/local/modules/openmpi/1.10.2/bin/mpirun"


/usr/bin/numactl -H

##================= now execute.

echo "CACHE CLEARING VERSION"
echo "Nodes:  ORIG ${SLURM_NODELIST}"
echo "NNodes:  ORIG ${SLURM_NNODES}"
echo "T/Nodes:  ORIG ${SLURM_TASKS_PER_NODE}"
echo "NT/Node:  ORIG ${SLURM_NTASKS_PER_NODE}"
echo "TASKS:  ORIG ${SLURM_NTASKS}"


JELLYFISH_EXEC=${ROOTDIR}/build/jellyfish/bin/jellyfish
KMC_EXEC=${ROOTDIR}/build/kmc3/kmc
GERBIL_EXEC=${ROOTDIR}/build/gerbil/gerbil
KMERIND_BIN_DIR=${ROOTDIR}/build/kmerhash/bin


#unset any OMP or numa stuff.
unset OMP_PROC_BIND
unset GOMP_CPU_AFFINITY
unset OMP_PLACES
unset OMP_NUM_THREADS

#UNSET some OMP envars. this is absolutely necessary to spread the computation out to cores.
export OMP_PROC_BIND=false
unset OMP_PLACES
# as soon as OMP_PROC_BIND is enabled, we either get all threads on 1 core, or all threads on 1 socket.
# specifying OMP_PLACES as can allow all threads on 1 socket, but unable to get to all sockets.
#export OMP_PLACES="{0:$cpu_node_cores},{18:$cpu_node_cores},{36:$cpu_node_cores},{54:$cpu_node_cores}"
# specifying cores gets all threads to 1 core, not even to 1 socket.
#export OMP_PLACES="cores($t)"


t=64
cpu_max_1=$(((t / 4) - 1))
cpu_max_2=$((18 + (t / 4) - 1))
cpu_max_3=$((36 + (t / 4) - 1))
cpu_max_4=$((54 + (t / 4) - 1))
cpu_node_cores=$((t / 4))

export GOMP_CPU_AFFINITY=0-${cpu_max_1},18-${cpu_max_2},36-${cpu_max_3},54-${cpu_max_4}
NUMA_CMD="/usr/bin/numactl -C ${GOMP_CPU_AFFINITY}"
echo $NUMA_CMD

#IF get error about numa affinity, then make sure the number of tasks (-n) for slurm is set to sufficiently large number.
echo "TEST NUMA"
/usr/bin/numactl --physcpubind=${GOMP_CPU_AFFINITY} hostname
echo "TEST NUMA DONE"

# don't believe that gerbil is using omp.
#export OMP_PROC_BIND=false
# seems to be needed for kmc.
#export OMP_PLACES="{0:$cpu_node_cores}:4:18"
#export OMP_NUM_THREADS=${t}


#Q4: human genome
#Q16: lob lolly pine genome
declare -a datasets2=("Q4" "Q16")

## now loop through the above array
for dataset in "${datasets2[@]}"
do

  datafile=${DATA_DIR}/${dataset}.fasta

  #drop cache
  eval "sudo /usr/local/crashplan/bin/CrashPlanEngine stop"
  eval "/usr/local/sbin/drop_caches"


  #=========== kmerind

  #cache populate
  EXEC=${KMERIND_BIN_DIR}/testKmerCounter-FASTA-a4-k31-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhFARM-shFARM
  echo "$MPIRUN_CMD -np 64 --map-by ppr:16:socket --rank-by core --bind-to core $EXEC -O ${LOCALTMP}/test.out ${datafile}" > ${logdir}/kmerind_uncached.log
  eval "$MPIRUN_CMD -np 64 --map-by ppr:16:socket --rank-by core --bind-to core $EXEC -O ${LOCALTMP}/test.out ${datafile} >> ${logdir}/kmerind_uncached.log 2>&1"


  rm ${LOCALTMP}/test.out*


  t=64
  K=31


  for iter in 1 2 3
  do


    for map in BROBINHOOD RADIXSORT
    do
      hash=MURMUR64avx



      # kmerhash
      for EXEC in ${KMERIND_BIN_DIR}/testKmerCounter-FASTA-a4-k${K}-CANONICAL-${map}-COUNT-dtIDEN-dh${hash}-shCRC32C
      do 

        exec_name=$(basename ${EXEC})

        logfile=${logdir}/kmerind/${exec_name}-n1-p${t}-${dataset}.$iter.log
        outfile=${OUT_DIR}/${exec_name}-n1-p${t}-${dataset}.$iter.bin

        # only execute if the file does not exist.
        if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
        then
          
          # command to execute
          cmd="$MPIRUN_CMD -np ${t} --map-by ppr:${cpu_node_cores}:socket --rank-by core --bind-to core $EXEC -O $outfile -B 6 ${datafile}"
          echo "COMMAND" > $logfile
          echo $cmd >> $logfile
          echo "COMMAND: ${cmd}" 
          echo "LOGFILE: ${logfile}"
                
          # call the executable and save the results
          echo "RESULTS" >> $logfile
          eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"
        
          echo "COMPLETED" >> $logfile
          echo "$exec_name COMPLETED."
          rm ${outfile}*
            
        else
        
          echo "$logfile exists and COMPLETED.  skipping."
        fi

      done
      #kmerind


    done
    #map


    map=DENSEHASH
    hash=FARM

    # kmerind
    for EXEC in ${KMERIND_BIN_DIR}/testKmerCounter-FASTA-a4-k${K}-CANONICAL-${map}-COUNT-dtIDEN-dh${hash}-sh${hash}
    do 

      exec_name=$(basename ${EXEC})

      logfile=${logdir}/kmerind/${exec_name}-n1-p${t}-${dataset}.$iter.log
      outfile=${OUT_DIR}/${exec_name}-n1-p${t}-${dataset}.$iter.bin

      # only execute if the file does not exist.
      if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
      then
        
        # command to execute
        cmd="$MPIRUN_CMD -np ${t} --map-by ppr:${cpu_node_cores}:socket --rank-by core --bind-to core $EXEC -O $outfile -B 6 ${datafile}"
        echo "COMMAND" > $logfile
        echo $cmd >> $logfile
        echo "COMMAND: ${cmd}" 
        echo "LOGFILE: ${logfile}"
              
        # call the executable and save the results
        echo "RESULTS" >> $logfile
        eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"
      
        echo "COMPLETED" >> $logfile
        echo "$exec_name COMPLETED."
        rm ${outfile}*
          
      else
      
        echo "$logfile exists and COMPLETED.  skipping."
      fi

    done
    #EXEC




  done
  #iter



  #=========== gerbil


  for iter in 1 2 3
  do


    # gerbil

    outfile=${OUT_DIR}/gerbil-CANONICAL-k${K}-t${t}-${dataset}.$iter.out
    # canonical
    logfile=${logdir}/gerbil/gerbil-CANONICAL-k${K}-t${t}-${dataset}.$iter.log
    if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
    then
  
      cmd="${NUMA_CMD} ${GERBIL_EXEC} -i -k ${K} -e 512GB -t ${t} -l 1 ${datafile} $LOCALTMP $outfile"
      echo "COMMAND (CANONICAL): ${cmd}" 
      echo "LOGFILE: ${logfile}"
      echo "$cmd" > $logfile
      
      eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"		  
      echo "COMPLETED" >> $logfile 
    
      rm ${outfile}*
    else
      echo "$logfile exists and COMPLETED.  skipping."
    fi

  done
  #iter




  #========= KMC3


  for iter in 1 2 3
  do

    outfile=${OUT_DIR}/kmc3.out

    # canonical
    logfile=${logdir}/kmc3/kmc3-CANONICAL-k${K}-t${t}-${dataset}.$iter.log
    if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
    then

      # -r mem only
      # -ci1 include kmers of all frequencies
      # -m16 16GB max (each? no.)
      # -b single strand form
      cmd="${NUMA_CMD} ${KMC_EXEC} -v -k${K} -m512 -fm -ci1 -cs1000000000 -r -t${t} ${datafile} $outfile ${OUT_DIR}"
      echo "COMMAND (CANONICAL): ${cmd}"
      echo "LOGFILE: ${logfile}"
      echo "$cmd" > $logfile
        
      eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"
      echo "COMPLETED" >> $logfile
      
      rm ${outfile}*
    else
      echo "$logfile exists and COMPLETED.  skipping."
    fi

  done
  #iter


  #============ JellyFish
  
  for iter in 1 2 3
  do

    # jellyfish
    outfile=${OUT_DIR}/jf_count.jf	

    # canonical
    timelog=${logdir}/jf/jf-time-CANONICAL-k${K}-t${t}-${dataset}.$iter.log
    logfile=${logdir}/jf/jellyfish-CANONICAL-k${K}-t${t}-${dataset}.$iter.log
    if [ ! -f $logfile ] || [ "$(tail -1 $logfile)" != "COMPLETED" ]
    then
  
      cmd="${NUMA_CMD} ${JELLYFISH_EXEC} count -m ${K} -s 16G -t ${t} -C -o $outfile --timing=${timelog} ${datafile}"
      echo "COMMAND (CANONICAL): ${cmd}" 
      echo "LOGFILE: ${logfile}"
      
      eval "($TIME_CMD $cmd >> $logfile 2>&1) >> $logfile 2>&1"		  
      echo "$timelog" >> $logfile
      cat $timelog >> $logfile
      
      rm $timelog
      rm $outfile
    else
      echo "$logfile exists and COMPLETED.  skipping."
    fi

  done
  #iter

done
#dataset


