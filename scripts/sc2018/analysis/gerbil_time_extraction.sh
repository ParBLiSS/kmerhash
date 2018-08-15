#!/bin/sh


# usage: this_script wildcard_of_all_log_files
# example:  this_script *.log

summarize_times() {
	## input file
	INPUT=$1

	#time parsing from
	#http://stackoverflow.com/questions/14309032/bash-script-difference-in-minutes-between-two-times
	
	
	stage1_time=$(grep "time (real) for stage1" $INPUT | sed 's/^time.*for stage1[^0-9]*\([0-9\.]*\) s$/\1/')
	stage2_time=$(grep "time (real) for stage2" $INPUT | sed 's/^time.*for stage2[^0-9]*\([0-9\.]*\) s$/\1/')
	
	
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

	strand=$(echo $f | sed 's/gerbil-\(.*\)-k[0-9]*-t[0-9]*-[^\.^-]*\.[0-9]*\.log/\1/')
	k=$(echo $f | sed 's/gerbil-.*-k\([0-9]*\)-t[0-9]*-[^\.^-]*\.[0-9]*\.log/\1/')
	threads=$(echo $f | sed 's/gerbil-.*-k[0-9]*-t\([0-9]*\)-[^\.^-]*\.[0-9]*\.log/\1/')
	iter=$(echo $f | sed 's/gerbil-.*-k[0-9]*-t[0-9]*-[^\.^-]*\.\([0-9]*\)\.log/\1/')

	if grep -e "1000genome" $f > /dev/null;
	then 
		dataset=K1
	else
		dataset=$(echo $f | sed 's/gerbil-.*-k[0-9]*-t[0-9]*-\([^\.^-]*\)\.[0-9]*\.log/\1/')
	fi


	echo "gerbil,${format},4,${k},${strand},1,${threads},${dataset},${iter},$(summarize_times $f)"
done



