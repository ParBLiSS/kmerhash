#!/bin/bash
#PBS -l nodes=1:ppn=28
#PBS -l walltime=01:00:00
#PBS -q swarm

module load gcc/4.9.4
module load mvapich2/2.3b

cat $PBS_NODEFILE > nodes.log

cd ~/build/kmerhash-test-gcc494-mvapich23b
mkdir -p ~/build/kmerhash-test-gcc494-mvapich23b/log

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

df=~/data/data/gage-chr14/gage_human_chr14_frag_1.fastq
df2=~/data/data/gage-chr14/gage_human_chr14_frag_2.fastq

format=FASTQ
k=31
a=4
dhash=MURMUR64avx
shash=CRC32C

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

for map in MTRADIXSORT MTROBINHOOD
do

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
do
	echo "running ${EXEC} 1x28"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 1 OMP_NUM_THREADS=$PBS_NP ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p1.t28.log 2>&1

	echo "running ${EXEC} 2x14"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 2 OMP_NUM_THREADS=14 ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p2.t14.log 2>&1

	echo "running ${EXEC} 28x1"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np $PBS_NP OMP_NUM_THREADS=1 ${EXEC} -b -F ${df} > ${EXEC/#bin/log}.p28.t1.log 2>&1
done

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	echo "running ${EXEC} 1x64"
	mpirun_rsh -hostfile=$PBS_NODEFILE -np 1 OMP_NUM_THREADS=$PBS_NP ${EXEC} ${df} ${df2} > ${EXEC/#bin/log}.p1.t28.log 2>&1

done

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

