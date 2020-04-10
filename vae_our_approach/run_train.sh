#!/bin/bash

# check number of parameters
if [ $# == 0 ]; then
	echo "Please, give pfam sequence ID as parameter of this script"
	exit 1
else
	# Run qsub with desired param
	echo "Running qsub on GPU node with sequence ${1}"
	qsub -v seqID=${1} train_vae.sh
fi


