#!/bin/bash
PBS -N GroundTruth
PBS -q gpu -l select=1:ngpus=1:mem=32gb:scratch_local=500mb
PBS -l walltime=24:00:00
PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
DATADIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/ # substitute for your log directory

# move into data directory
cd $DATADIR_BASE
source $DATADIR_BASE/.bashrc
conda activate #vae-env
echo "Sourcing anaconda succesful" >> $LOGDIR/jobs_info.txt

cd $DATADIR

python3 runner.py msa_preprocessor.py
python3 runner.py train.py
python3 runner.py run_task.py --run_package_stats_order
#python3 runner.py run_task.py --run_package_stats_tree
#python3 runner.py run_task.py --run_package_stats_mapper
#python3 runner.py run_task.py --run_package_stats_reconstruction
#python3 runner.py benchmark.py