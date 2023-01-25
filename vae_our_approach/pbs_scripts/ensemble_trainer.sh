#!/bin/bash
#PBS -q gpu -l select=1:ngpus=1:mem=16gb:scratch_local=10gb
#PBS -l walltime=24:00:00
#PBS -m ae

# Parse passed sequence code throught param
#CONFFILE=${conf_path}
#PICKLEFLD=${pkl_fld}
#MODEL=${model}

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
RESULTS=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/results/.
PICKLES=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/${PICKLEFLD}
DATADIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/scripts # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/scripts/devel_log/ # substitute for your log directory

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

# move into data directory
cd $DATADIR_BASE
source $DATADIR_BASE/.bashrc
conda activate #vae-env
echo "Sourcing anaconda successful" >> $LOGDIR/jobs_info.txt

## copy project to scratch directory
LOCALPROJECT=${SCRATCHDIR}/scripts/.

echo "scratch directory is " $SCRATCHDIR

cp -r $DATADIR $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

## transfer clustalo
cp -r $CLUSTAL $SCRATCHDIR

## transfer preprocessed MSA data for ensembles here
## it will create also results directory for this model
cp -r $PICKLES $SCRATCHDIR

## run project
cd $LOCALPROJECT

echo "========================================================================"
echo "Training model $MODEL with current configuration $CONFFILE"
python3 runner.py ensemble_train.py --json ${CONFFILE} --ensemble_num ${MODEL}

## transport results
LOCALRESULTS=${SCRATCHDIR}/results/.
cd $LOCALRESULTS

tar cvzf results-$PBS_JOBID.tar.gz .
cp results-$PBS_JOBID.tar.gz $RESULTS || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

echo "Training process is finished" >> $LOGDIR/jobs_info.txt
# clean the SCRATCH directory
clean_scratch
