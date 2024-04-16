#!/bin/bash
## Script for running fast tree on whole msa directory on individual nodes

## This git repository is allocate in DATADIR_BASE/AI-dir/vae-for-hlds
## Please change for your location
MSAs=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/pfam_msa/MSA
ROOT_SCR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach
FAST_SRC=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/pfam_msa/FastTree

cd ${MSAs}
pwd
for filename in *.fasta
do
    echo "Running fastTree for ${filename}"
    qsub -v file=${filename} ../../fast_tree_batch.sh &
done

cd ${ROOT_SCR}
