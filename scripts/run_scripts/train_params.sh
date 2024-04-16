#!/bin/bash

## This git repository is allocate in DATADIR_BASE/AI-dir/vae-for-hlds
## Please change for your location
DATADIR_BASE=/storage/brno2/home/xkohou15
ROOT_SCR=AI_dir/vae-for-hlds/vae_our_approach/devel_LL/run_scripts
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/
PYTHONSCRIPTS=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL

# check number of parameters
if [ $# == 0 ]; then
	echo "Please, give values of C you wish to run with. "
	exit 1
else
  mkdir -p $LOGDIR

  # move into data directory
  cd $DATADIR_BASE
  #source $DATADIR_BASE/.bashrc
  #conda activate #vae-env
  echo "Sourcing anaconda succesful" >> $LOGDIR/jobs_info.txt

  cd $PYTHONSCRIPTS
  #python3 msa_preprocessor.py --Pfam_id PF00561 --ref P59336_S14 --output_dir C20dim --stats --in_file results/PF00561/MSA/identified_targets_msa.fa

  cd run_scripts

  # Run in parallel for multiple nodes, for each query sequence
  for c; do
    # Run qsub with desired param
    echo "Running qsub on GPU node with Decay value ${c}"
    qsub -v c=${c} params_pipe.sh &
  done
fi


