#!/bin/bash



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

get_procs() {
	procs=$(echo $1 | sed 's/^.*-n\([0-9]*\)-[pt]\([0-9]*\)[-\.]*.*\.log$/\1,\2/')

	echo $procs
}

get_index() {
	index=$(echo $1 | sed 's/^.*-a\([0-9]*\)-k\([0-9]*\)-\([^-]*\)-\([^-]*\)-\([^-]*\)-.*\.log$/\5/')

	echo $index
}


get_params() {

	format=$(get_format $1)
	
	#convert -a[1]- 
	indexparams=$(echo $1 | sed 's/^.*-a\([0-9]*\)-k\([0-9]*\)-\([^-]*\)-\([^-]*\)-\([^-]*\)-.*\.log$/\1,\2,\3,\4,\5/')
		
	hashes=$(echo $1 | sed 's/^.*-dt\([A-Z]*\)-dh\([A-Z]*\)-sh\([A-Z]*\)-.*\.log$/\1,\2,\3/')

	#procs=$(echo $1 | sed 's/^.*-n\([0-9]*\)-[pt]\([0-9]*\)[-\.]*.*\.log$/\1,\2/')
	procs=$(get_procs $1)

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
header="transform,barrier,empty,alloc_mapping,bucket,to_pos,alloc_permute,permute,a2a_count,realloc_out,a2a,reserve,insert"
echo "experiment,format,dna,k,strand,map,index,dist_trans,dist_hash,store_hash,nodes,ppn,dataset,iter,meas_cat,meas,${header}" > SUMMARY_TIME_HASH_INSERT


for f in "$@"
do
	if grep -q ".*hash.*\:insert" "$f"; then

		# get experimental param
		params=$(get_params $f)

		procs=$(get_procs $f)
		index=$(get_index $f)

		if [ "${procs}" == "1,1" ]; then

			content1=""

			if [ "${index}" == "POS" ]; then

				# get the rehash stuff
				content2=$(grep -m 12 -A 12 "hash_multimap\:insert" $f | grep -A 9 transform_ | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\(.*\),\]$/\1,\2,\3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\4,\5/')
				#echo "$f 2: $content2"

			else

				# get transform_input
				content2=$(grep -A 12 ".*_densehash\:insert" $f | grep -A 9 transform_input | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\]$/\1,\2,\3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\4/')
				#echo "$f 2: $content2"

			fi

			content3=""

		else 
			if [ "${index}" == "POS" ]; then

				# get transform_input
				content1=$(grep -A 12 "hash_multimap\:insert" $f | grep -A 9 transform_ | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\(.*\),\(.*\),\]$/\1,\2,\3/')
				#echo "$f 1: $content1"


				# get the rehash stuff
				content2=$(grep -m 12 -A 12 "imxx\:distribute" $f | grep -A 9 permute | grep dur_max | sed 's/^\[.*\].*\t\[,\(.*\),\]$/\1/')
				#echo "$f 2: $content2"


				# get transform_input
				content3=$(grep -A 12 "hash_multimap\:insert" $f | grep -A 9   transform_ | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\(.*\),\(.*\),\]$/\5,\6/')
				#echo "$f 3: $content3"

			else 
				# get transform_input
				content1=$(grep -A 12 ".*_densehash\:insert" $f | grep -A 9 transform_ | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\(.*\),\]$/\1,\2,\3/')
				#echo "$f 1: $content1"


				# get the rehash stuff
				content2=$(grep -m 12 -A 12 "imxx\:distribute" $f | grep -A 9 permute | grep dur_max | sed 's/^\[.*\].*\t\[,\(.*\),\]$/\1/')
				#echo "$f 2: $content2"


				# get transform_input
				content3=$(grep -A 12 ".*_densehash\:insert" $f | grep -A 9   transform_ | grep dur_max | sed 's/^\[\(.*\)\].*\t\(.*\)\t\[,\(.*\),\(.*\),\(.*\),\]$/0.0,\5/')
				#echo "$f 3: $content3"
			fi

		fi


		echo "$params,$content1,$content2,$content3" >> SUMMARY_TIME_HASH_INSERT

	fi

done


