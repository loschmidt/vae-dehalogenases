#!/bin/bash

## This git repository is allocate in DATADIR_BASE/AI-dir/vae-for-hlds
## Please change for your location
DATADIR_BASE=/storage/brno2/home/xkohou15
ROOT_SCR=AI_dir/vae-for-hlds/vae_our_approach/devel_LL/run_scripts
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/

# check number of parameters
if [ $# == 0 ]; then
	echo "Please, give sequence/s for reference in run "
	exit 1
else
  mkdir -p $LOGDIR
  cd $ROOT_SCR
  # Default value
  pfam_id="PF00561"
  # Run in parallel for multiple nodes, for each query sequence
  for que in "$*"; do
    # Run qsub with desired param
    echo "Running qsub on GPU node with sequence ${1} and query sequence ${que}"
    qsub -v query=${que} seqID=${pfam_id} pipe_train.sh &
  done
fi


