#!/bin/bash
#PBS -N latentSpaceJob
#PBS -q gpu -l select=1:ngpus=1:mem=20gb:scratch_local=500mb
#PBS -l walltime=24:00:00 
#PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
DATADIR=/storage/brno2/xkohou15/AI_dir/vae_our_aproach/pfam_msa # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae_our_aproach # substitute for your log directory

# Parse passed sequence code throught param
PFAMSEQ=${seqID}

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $LOGDIR/jobs_info.txt

# move into data directory
cd $DATADIR_BASE
source $DATADIR_BASE/.bashrc
conda activate vae-env
echo "Sourcing anaconda succesful" >> $LOGDIR/jobs_info.txt

cd $DATADIR
python3 ./script/download_MSA.py --Pfam_id $PFAMSEQ
echo "Dowload of PFAM succesfull" >> $LOGDIR/jobs_info.txt

python3 ./script/proc_msa.py
echo "proc_msa succesful" >> $LOGDIR/jobs_info.txt

python3 ./script/train.py --num_epoch 10000 --weight_decay 0.01
echo "Train completed " >> $LOGDIR/jobs_info.txt

#python3 ./script/analyze_model.py
#echo "Analyze clean" >> $LOGDIR/jobs_info.txt
# clean the SCRATCH directory
clean_scratch
