#!/bin/bash
#PBS -N trainerHTS3
#PBS -q gpu -l select=1:ngpus=1:mem=16gb:scratch_local=10gb
#PBS -l walltime=24:00:00
#PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
DATASET=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/datasets/datasets.tar.gz
RESULTS=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/results/.
DATADIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/ # substitute for your log directory

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

# move into data directory
cd $DATADIR_BASE
source $DATADIR_BASE/.bashrc
conda activate #vae-env
echo "Sourcing anaconda succesful" >> $LOGDIR/jobs_info.txt

## copy project to scratch directory
LOCALPROJECT=${SCRATCHDIR}/project/.

cp $DATADIR $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

## transfer datasets
cp $DATASET $SCRATCHDIR

## Run pipeline
cd $SCRATCHDIR

## untar datasets
tar -zxvf datasets.tar.gz

## run project
cd $LOCALPROJECT

echo "========================================================================"
echo "Training model with current configuration on"
python3 runner.py msa_handlers/msa_preprocessor.py
python3 runner.py train.py

## Order statistics
python3 runner.py benchmark.py
python3 runner.py run_task.py --run_generative_evaluation
python3 runner.py run_task.py --run_generative_evaluation_plot

## transport results
LOCALRESULTS=${SCRATCHDIR}/results/.
cd $LOCALRESULTS
tar -zcvf results-$PBS_JOBID.tar.gz .
cp results-$PBS_JOBID.tar.gz $RESULTS || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

echo "Training process is finished" >> $LOGDIR/jobs_info.txt
# clean the SCRATCH directory
clean_scratch
