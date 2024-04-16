#!/bin/bash

## This git repository is allocate in DATADIR_BASE/AI-dir/vae-for-hlds
## Please change for your location
DATADIR_BASE=/storage/brno2/home/xkohou15
SCRIPT_RUN=AI_dir/vae-for-hlds/vae_our_approach/pfam_msa
ROOT_SCR=AI_dir/vae-for-hlds/vae_our_approach

# check number of parameters
if [ $# == 0 ]; then
	echo "Please, give pfam sequence ID as parameter of this script"
	exit 1
else
  cd $DATADIR_BASE
  # Activate venv for download script
  source .bashrc
  conda activate #vae-env

  # Download desire sequence and its rps and seed seq
  cd $SCRIPT_RUN
  python3 ./script/download_MSA.py --Pfam_id ${1}
  cd $DATADIR_BASE
  # Deactivate venv
  conda deactivate
  cd $ROOT_SCR
  # Run in parallel for multiple nodes, each sub sequence
  for rp_i in "full" "rp75" "rp55" "rp35" "rp15" "seed"; do
    # Run qsub with desired param
    echo "Running qsub on GPU node with sequence ${1} and ${rp_i}"
    qsub -v seqID=${1},rp=${rp_i} train_vae.sh &
  done
fi


