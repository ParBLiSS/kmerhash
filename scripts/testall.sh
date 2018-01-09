#!/bin/sh

for EXEC in test/test-*
do
	${EXEC} > ${EXEC}.p1.t1.log
done

for EXEC in bin/benchmark_*
do
	${EXEC} > ${EXEC}.p1.t1.log
done


df=~/data/1000genome/SRR077487_1.filt.0_015625.fastq
df2=~/data/1000genome/SRR077487_1.filt.0_03125.fastq

format=FASTQ
k=31
a=4
dhash=MURMUR64avx
shash=CRC32C

for map in RADIXSORT BROBINHOOD
do
	EXEC=bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	mpirun -np 4 ${EXEC} -b -F ${df} > ${EXEC}.p4.t1.log 

	EXEC=bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dhMURMUR32avx-sh${shash}
	mpirun -np 4 ${EXEC} -b -F ${df} > ${EXEC}.p4.t1.log 

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	mpirun -np 4 ${EXEC} ${df} ${df2} > ${EXEC}.p4.t1.log
done

for map in MTRADIXSORT MTROBINHOOD
do
	EXEC=bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	OMP_NUM_THREADS=4; mpirun -np 1 ${EXEC} -b -F ${df} > ${EXEC}.p1.t4.log

	OMP_NUM_THREADS=2; mpirun -np 2 ${EXEC} -b -F ${df} > ${EXEC}.p2.t2.log

	OMP_NUM_THREADS=1; mpirun -np 4 ${EXEC} -b -F ${df} > ${EXEC}.p4.t1.log

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	OMP_NUM_THREADS=4; mpirun -np 1 ${EXEC} ${df} ${df2} > ${EXEC}.p1.t4.log

done

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
mpirun -np 4 ${EXEC} -b -F ${df} > ${EXEC}.p4.t1.log

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
mpirun -np 4 ${EXEC} -b -F ${df} > ${EXEC}.p4.t1.log

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
mpirun -np 4 ${EXEC} ${df} ${df2} > ${EXEC}.p4.t1.log

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
mpirun -np 4 ${EXEC} ${df} ${df2} > ${EXEC}.p4.t1.log
