#!/bin/sh


# usage: this_script wildcard_of_all_log_files
# example:  this_script *.log


summarize_times() {
	## input file
	INPUT=$1
	
	walltime=$(grep -m1 "Elapsed (wall clock) time (h:mm:ss or m:ss):" $INPUT | sed 's/Elapsed (wall clock) time (h:mm:ss or m:ss)://' | sed 's/^[ \t]*//')
	
	maxmem=$(grep -m1 "Maximum resident set size (kbytes):" $INPUT | sed 's/Maximum resident set size (kbytes)://' | sed 's/^[ \t]*//')
	
	echo "[,${walltime},${maxmem},]"
}

get_iter() {
	if echo $1 | grep "[123]/testKmer" > /dev/null;
	then 
		iter=$(echo $1 | sed 's/^\([123]\)\/.*\.log/\1/')
	else
		iter=$(echo $1 | sed 's/.*\.\([0-9]\)\.log/\1/')
	fi
	
	echo $iter
}


get_format() {

	if grep -e "\.fastq" $1 > /dev/null;
	then 
		format=FASTQ
	else
		format=FASTA
	fi
	
	echo $format
}


get_dataset() {
	if grep -e "1000genome" $1 > /dev/null;
	then 
		dataset=K4
	else
		dataset=$(echo $1 | sed 's/.*-\([PDKQ][0-9]*\)[\.]*[0-9]*\.log/\1/')
	fi
	
	echo $dataset
}

get_params() {

	format=$(get_format $1)
	
	#convert -a[1]- 
	indexparams=$(echo $1 | sed 's/^.*-a\([0-9]*\)-k\([0-9]*\)-\([^-]*\)-\([^-]*\)-\([^-]*\)-.*\.log$/\1,\2,\3,\4,\5/')
		
	hashes=$(echo $1 | sed 's/^.*-dt\([A-Z]*\)-dh\([A-Z]*\)-sh\([A-Z]*\)-.*\.log$/\1,\2,\3/')

	procs=$(echo $1 | sed 's/^.*-n\([0-9]*\)-[pt]\([0-9]*\)[-\.]*.*\.log$/\1,\2/')

	iter=$(get_iter $1)
	dataset=$(get_dataset $1)
	
	echo "kmerind,$format,$indexparams,$hashes,$procs,$dataset,$iter"
}


#echo "CUMULATIVE: $(grep app $1 | grep header | grep TIME)"
#for f in "$@"
#do
#	echo "$f: $(grep app $f | grep cum_max)"
#done

#app duration
#get the header.  within a directory, they should all be the same.
header="read,insert,read_query,sample,count,find,find_overlap,erase"
echo "experiment,format,dna,k,strand,map,index,dist_trans,dist_hash,store_hash,nodes,ppn,dataset,iter,meas_cat,meas,${header}" > SUMMARY_TIME_1

header="read,insert,read_query,sample,count,find_overlap,erase"
echo "experiment,format,dna,k,strand,map,index,dist_trans,dist_hash,store_hash,nodes,ppn,dataset,iter,meas_cat,meas,${header}" > SUMMARY_TIME_2		

header="read_query,sample,read,insert,count,find,find_overlap,erase"
echo "experiment,format,dna,k,strand,map,index,dist_trans,dist_hash,store_hash,nodes,ppn,dataset,iter,meas_cat,meas,${header}" > SUMMARY_TIME_3

header="read_query,sample,read,insert,count,find_overlap,erase"
echo "experiment,format,dna,k,strand,map,index,dist_trans,dist_hash,store_hash,nodes,ppn,dataset,iter,meas_cat,meas,${header}" > SUMMARY_TIME_4

for f in "$@"
do
	if grep -q app "$f"; then

		params=$(get_params $f)$@
			content1=$(grep app $f | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\]$/\1,\2,\3/')
			content2=$(grep app $f | grep cnt_min | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\]$/\1,\2,\3/')
			content3=$(grep app $f | grep cnt_mean | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\]$/\1,\2,\3/')
			content4=$(grep app $f | grep cnt_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\]$/\1,\2,\3/')
			content5=$(grep app $f | grep cnt_stdev | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\]$/\1,\2,\3/')

		if grep -q "read,insert,read_query,sample,count,find,find_overlap,erase" $f
		then
			echo "1 $f"

			echo "$params,$content1" >> SUMMARY_TIME_1
			echo "$params,$content2" >> SUMMARY_TIME_1
			echo "$params,$content3" >> SUMMARY_TIME_1
			echo "$params,$content4" >> SUMMARY_TIME_1
			echo "$params,$content5" >> SUMMARY_TIME_1

		elif grep -q "read,insert,read_query,sample,count,find_overlap,erase" $f
		then
			echo "2 $f"

			echo "$params,$content1" >> SUMMARY_TIME_2
			echo "$params,$content2" >> SUMMARY_TIME_2
			echo "$params,$content3" >> SUMMARY_TIME_2
			echo "$params,$content4" >> SUMMARY_TIME_2
                        echo "$params,$content5" >> SUMMARY_TIME_2

		elif grep -q "read_query,sample,read,insert,count,find,find_overlap,erase" $f
		then
			echo "3 $f"

			echo "$params,$content1" >> SUMMARY_TIME_3
			echo "$params,$content2" >> SUMMARY_TIME_3
			echo "$params,$content3" >> SUMMARY_TIME_3
			echo "$params,$content4" >> SUMMARY_TIME_3
                        echo "$params,$content5" >> SUMMARY_TIME_3

		elif grep -q "read_query,sample,read,insert,count,find_overlap,erase" $f
		then
			echo "4 $f"
			echo "$params,$content1" >> SUMMARY_TIME_4
			echo "$params,$content2" >> SUMMARY_TIME_4
			echo "$params,$content3" >> SUMMARY_TIME_4
			echo "$params,$content4" >> SUMMARY_TIME_4
                        echo "$params,$content5" >> SUMMARY_TIME_4

		else
			echo "NOT CATEGORIZED: $f"

		fi
	fi

done


