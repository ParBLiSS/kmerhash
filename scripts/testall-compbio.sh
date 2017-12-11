#!/bin/sh
#SBATCH -J test-kmerhash-execs     # Job name

## OVERRIDE THESE ON COMMANDLINE.
#SBATCH -N 1               # Total number of nodes requested (16 cores/node)
#SBATCH -n 72              # Total number of mpi tasks requested

cd ~/build/kmerhash-test
mkdir -p ~/build/kmerhash-test/log

for EXEC in test/test-*
do
	echo "running ${EXEC}"
	${EXEC} > ${EXEC/test\//log\/}.p1.t1.log 2>&1
done

for EXEC in bin/benchmark_hash*
do
	echo "running ${EXEC}"
	${EXEC} > ${EXEC/bin\//log\/}.p1.t1.log 2>&1
done

for EXEC in bin/benchmark_a*
do
	echo "running ${EXEC}"
	mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1
done


df=/project/tpan7/data/1000genome/HG00096/SRR077487_1.filt.fastq
df2=/project/tpan7/data/1000genome/HG00096/SRR077487_2.filt.fastq

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
	mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1
done

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dhMURMUR32avx-sh${shash}
do
	echo "running ${EXEC}"
	mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1
done

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	echo "running ${EXEC}"
	mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} ${df} ${df2} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1
done

for map in MTRADIXSORT MTROBINHOOD
do

for EXEC in bin/*KmerIndex-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
do
	echo "running ${EXEC} 1x64"
	OMP_NUM_THREADS=64; mpirun -np 1 --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p1.t64.log 2>&1

	echo "running ${EXEC} 4x16"
	OMP_NUM_THREADS=16; mpirun -np 4 --map-by ppr:1:socket --bind-to core  ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p4.t16.log 2>&1

	echo "running ${EXEC} 64x1"
	OMP_NUM_THREADS=1; mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1
done

	EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-${map}-COUNT-dtIDEN-dh${dhash}-sh${shash}
	echo "running ${EXEC} 1x64"
	OMP_NUM_THREADS=64; mpirun -np 1 --bind-to core ${EXEC} ${df} ${df2} > ${EXEC/bin\//log\/}.p1.t64.log 2>&1

done

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
	echo "running ${EXEC}"
mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1

EXEC=bin/testKmerIndex-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
	echo "running ${EXEC}"
mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} -b -F ${df} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR-shCRC32C
	echo "running ${EXEC}"
mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} ${df} ${df2} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1

EXEC=bin/testKmerCounter-${format}-a${a}-k${k}-CANONICAL-DENSEHASH-COUNT-dtIDEN-dhMURMUR32-shCRC32C
	echo "running ${EXEC}"
mpirun -np 64 --map-by ppr:16:socket --bind-to core ${EXEC} ${df} ${df2} > ${EXEC/bin\//log\/}.p64.t1.log 2>&1

