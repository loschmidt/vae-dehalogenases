#!/bin/bash
#PBS -N latentSpaceJob
#PBS -q gpu -l select=1:ngpus=1:mem=8gb:scratch_local=500mb
#PBS -l walltime=24:00:00
#PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
DATADIR=/storage/brno2/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/ # substitute for your log directory

# Parse passed sequence code throught param
QUERY=${query}
PFAMSEQ=${seqID}

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

# move into data directory
cd $DATADIR_BASE
source $DATADIR_BASE/.bashrc
conda activate #vae-env
echo "Sourcing anaconda succesful" >> $LOGDIR/jobs_info.txt
## Run pipeline
cd $DATADIR
python3 pipeline.py --Pfam_id $PFAMSEQ --ref ${QUERY} --output_dir ${QUERY} --stats --highlight_seqs ${QUERY}

##python3 ./script/analyze_model1.py --Pfam_id $PFAMSEQ --RPgroup ${RPgroup}
echo "Training process is finished" >> $LOGDIR/jobs_info.txt
# clean the SCRATCH directory
clean_scratch
