#!/bin/bash
#PBS -N DHAA
#PBS -q gpu -l select=1:ngpus=1:mem=32gb:scratch_local=500mb
# -l select=1:ncpus=1:mem=16gb:scratch_local=1gb
#PBS -l walltime=24:00:00
#PBS -m ae

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR_BASE=/storage/brno2/home/xkohou15 # substitute to your home directory
DATADIR=/storage/brno2/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL # substitute username and path to to your real username and path
LOGDIR=/storage/brno2/home/xkohou15/AI_dir/vae-for-hlds/vae_our_approach/devel_LL/devel_log/ # substitute for your log directory

# Parse passed sequence code throught param
C=${c}

TARGET=second_report #trainValidation #sequenceGen
UPDIR=simple
L=299
D=2
NAME=dhaa #Reference #SimpleVAE
QUERY=P59336_S14 #P27652.1 #P59336_S14

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
## python3 parser_handler.py --Pfam_id $PFAMSEQ --ref ${QUERY} --output_dir ${QUERY} --stats --highlight_seqs ${QUERY}

echo "========================================================================"
echo "Training model for given decay factor = 0.0, dimensionality = ${D}, C = ${C}"
python3 train.py --Pfam_id ${UPDIR} --output_dir ${TARGET} --stats --in_file ./results/PF00561/MSA/identified_targets_msa.fa --num_epoch 16700 --ref ${QUERY} --weight_decay 0.0 --C ${C} --layers ${L} --dimensionality ${D} --model_name ${NAME}

#echo "========================================================================"
#echo "Measurement of generative process has begun with C = ${C} dim = ${D}"
#python3 benchmark.py --Pfam_id ${UPDIR} --output_dir ${TARGET} --ref ${QUERY} --weight_decay 0.0 --dimensionality ${D} --C ${C} --layers ${L} --in_file ./results/PF00561/MSA/identified_targets_msa.fa --model_name ${NAME}

#echo "========================================================================"
#echo "Highlighting reference sequence and Babkovas ancestrals"
#python3 analyzer.py --Pfam_id ${UPDIR} --output_dir ${TARGET} --ref ${QUERY} --highlight_seqs ${QUERY} --highlight_files ./results/PF00561/MSA/ancestors.fasta --stats --align --weight_decay 0.0 --layers ${L} --dimensionality ${D} --in_file ./results/PF00561/MSA/identified_targets_msa.fa --model_name ${NAME}

#echo "========================================================================"
#echo "Creation of new ancestral candidates has begun with D = ${D}, C = ${C}, L = ${L}"
#python3 mutagenesis.py --Pfam_id ${UPDIR} --ref ${QUERY} --align --highlight_files ./results/PF00561/MSA/ancestors.fasta --in_file ./results/PF00561/MSA/identified_targets_msa.fa --stats --output_dir ${TARGET} --weight_decay 0.0 --C ${C} --layers ${L} --dimensionality ${D} #--model_name ${NAME}

#echo "========================================================================"
#echo "Benchmarking model D = 16, C = ${C}, L = 320"
#python3 benchmark.py --Pfam_id PF00561 --ref ${QUERY} --output_dir ${TARGET} --stats --weight_decay 0.0 --layers 320 --C ${C} --dimensionality 16 --in_file ./results/PF00561/MSA/identified_targets_msa.fa

#echo "========================================================================"
#echo "Validation of model D = ${D}, C = ${C}, L = ${L}"
#python3 validation_train.py --Pfam_id ${UPDIR} --ref ${QUERY} --output_dir ${TARGET} --stats --weight_decay 0.0 --layers ${L} --C ${C} --dimensionality ${D} --in_file ./results/PF00561/MSA/identified_targets_msa.fa --model_name SimpelVal


##python3 ./script/analyze_model1.py --Pfam_id $PFAMSEQ --RPgroup ${RPgroup}
echo "Training process is finished" >> $LOGDIR/jobs_info.txt
# clean the SCRATCH directory
clean_scratch
