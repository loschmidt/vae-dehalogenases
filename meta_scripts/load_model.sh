#!/bin/bash

# check number of parameters
if [ $# == 0 ]; then
	echo "Please, name the directory to loaded to executable destination"
	exit 1
else
	# Run qsub with desired param
	echo "Loading files to exe directory ${1}"
	cd pfam_msa/output/.
	rm -rf *.pkl 
	rm -rf model
	rm *.png
	cp -r ../results/${1}
	cd ../../.
fi
