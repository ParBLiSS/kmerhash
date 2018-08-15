#!/bin/sh


# usage: this_script wildcard_of_all_log_files
# example:  this_script *.log

summarize_times() {
	## input file
	INPUT=$1

	#time parsing from
	#http://stackoverflow.com/questions/14309032/bash-script-difference-in-minutes-between-two-times
	
	
	stage1_time=$(grep "1st stage:" $INPUT | sed 's/1st stage: //' | sed 's/s//')
	stage2_time=$(grep "2nd stage:" $INPUT | sed 's/2nd stage: //' | sed 's/s//')
	
	
	walltime=$(grep -m1 "Elapsed (wall clock) time (h:mm:ss or m:ss):" $INPUT | sed 's/Elapsed (wall clock) time (h:mm:ss or m:ss)://' | sed 's/^[ \t]*//')
	
	maxmem=$(grep -m1 "Maximum resident set size (kbytes):" $INPUT | sed 's/Maximum resident set size (kbytes)://' | sed 's/^[ \t]*//')
	
	
	echo "${stage1_time},${stage2_time},${walltime},${maxmem}"
	
#	walltime=$(grep "Elapsed (wall clock) time (h:mm:ss or m:ss)" $INPUT | sed 's/^[ \t]*//')
#	maxmem=$(grep "Maximum resident set size (kbytes)" $INPUT | sed 's/^[ \t]*//')
#	
#	echo "$walltime"
#	echo "$maxmem"	
}


#read_and_split(s),sort(s),walltime(m:ss),maxmem(kb)
echo "experiment,format,dna,k,strand,nodes,ppn,dataset,iter,read_and_split,sort,walltime,maxmem"

for f in "$@"
do
	if grep -e "\.fastq" $f > /dev/null;
	then 
		format=FASTQ
	else
		format=FASTA
	fi

	strand=$(echo $f | sed 's/kmc3-\(.*\)-k[0-9]*-t[0-9]*-[^\.^-]*\.[0-9]*\.log/\1/')
	k=$(echo $f | sed 's/kmc3-.*-k\([0-9]*\)-t[0-9]*-[^\.^-]*\.[0-9]*\.log/\1/')
	threads=$(echo $f | sed 's/kmc3-.*-k[0-9]*-t\([0-9]*\)-[^\.^-]*\.[0-9]*\.log/\1/')
	iter=$(echo $f | sed 's/kmc3-.*-k[0-9]*-t[0-9]*-[^\.^-]*\.\([0-9]*\)\.log/\1/')

	if grep -e "1000genome" $f > /dev/null;
	then 
		dataset=K1
	else
		dataset=$(echo $f | sed 's/kmc3-.*-k[0-9]*-t[0-9]*-\([^\.^-]*\)\.[0-9]*\.log/\1/')
	fi


	echo "kmc3,${format},4,${k},${strand},1,${threads},${dataset},${iter},$(summarize_times $f)"
done
