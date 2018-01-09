#!/bin/bash
#PBS -l nodes=1:ppn=28
#PBS -l walltime=01:00:00
#PBS -q swarm
#PBS -n

module load gcc/4.9.4
module load hwloc
module load mvapich2/2.3b

cat $PBS_NODEFILE > nodes.log

cd ~/build/kmerhash-test-gcc494-mvapich23b
mkdir -p ~/build/kmerhash-test-gcc494-mvapich23b/log

if [ 1 -eq 1 ]
then

for EXEC in test/test-*
do
	echo "running ${EXEC}, log in ${EXEC/#test/log}.p1.t1.log"
	${EXEC} > ${EXEC/#test/log}.p1.t1.log 2>&1
done

for EXEC in bin/benchmark_hash*
do
	echo "running ${EXEC}, log in ${EXEC/test\//log\/}.p1.t1.log"
	${EXEC} > ${EXEC/#bin/log}.p1.t1.log 2>&1
done

for EXEC in bin/benchmark_a*
do
	echo "running ${EXEC}, log in ${EXEC/test\//log\/}.p${PBS_NP}.t1.log"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} > ${EXEC/#bin/log}.p${PBS_NP}.t1.log 2>&1
done

fi

df=~/data/data/gage-chr14/gage_human_chr14_frag_1.fastq
df2=~/data/data/gage-chr14/gage_human_chr14_frag_2.fastq

format=FASTQ
k=31
a=4
dhash=MURMUR64avx
shash=CRC32C

export MV2_SHOW_CPU_BINDING=1
export MV2_ENABLE_AFFINITY=1

if [ 1 -eq 1 ]
then


for map in RADIXSORT BROBINHOOD
do

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
do
	echo "running ${EXEC}"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1
done

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dhMURMUR32avx-sh${shash}
do
	echo "running ${EXEC}"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1
done

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	echo "running ${EXEC}"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} ${df} ${df2} > ${EXEC/#bin/log}.p28.t1.log 2>&1
done

fi

export OMP_DISPLAY_ENV=true
export OMP_PLACES=cores
export OMP_PROC_BIND=true

if [ 1 -eq 1 ]
then


for map in MTRADIXSORT MTROBINHOOD 
do

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
do
	echo "running ${EXEC} 1x28"
	export OMP_NUM_THREADS=$PBS_NP
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 1 OMP_NUM_THREADS=28 MV2_CPU_MAPPING=0-27 ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p1.t28.log 2>&1

	echo "running ${EXEC} 2x14"
	export OMP_NUM_THREADS=14
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 2 OMP_NUM_THREADS=14 MV2_CPU_MAPPING=0-13:14-27 ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p2.t14.log 2>&1

	echo "running ${EXEC} 28x1"
	export OMP_NUM_THREADS=1	
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP OMP_NUM_THREADS=1 MV2_CPU_BINDING_POLICY=scatter MV2_CPU_BINDING_LEVEL=core ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1
	

done

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	echo "running ${EXEC} 1x64"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 1 OMP_NUM_THREADS=$PBS_NP MV2_CPU_BINDING_POLICY=scatter MV2_CPU_BINDING_LEVEL=socket ${EXEC} ${df} ${df2} > ${EXEC/#bin/log}.p1.t28.log 2>&1

done

fi

if [ 1 -eq 1 ]
then

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
	echo "running ${EXEC}"
mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
	echo "running ${EXEC}"
mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
	echo "running ${EXEC}"
mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} ${df} ${df2} > ${EXEC/#bin/log}.p28.t1.log 2>&1

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
	echo "running ${EXEC}"
mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP ${EXEC} ${df} ${df2} > ${EXEC/#bin/log}.p28.t1.log 2>&1

fi
