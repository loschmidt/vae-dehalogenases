#!/bin/bash
#PBS -N FstTree
#PBS -q default -l select=1:ncpus=1:mem=5gb:scratch_local=500mb
#PBS -l walltime=24:00:00
#PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
FASTTREE=/storage/brno2/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/pfam_msa/FastTree # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/pfam_msa # substitute for your log directory

# Parse passed file name
F_NAME=${file}

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $LOGDIR/fast_jobs_info.txt

outfile=${F_NAME//.fasta/.newick}

# move into data directory
cd $FASTTREE
echo "Starting FastTreeMP" >> $LOGDIR/fast_jobs_info.txt
./FastTreeMP -lg -gamma < ../MSA/${F_NAME} > ${outfile}
echo "Fast tree process was successful" >> $LOGDIR/fast_jobs_info.txt