# VAE - modified pipeline
**authors:**  Pavel Kohout <xkohou15@stud.fit.vutbr.cz>, Stas Mazurenko

Describing how to run our modified pipeline for whole family (including RPXX)

`conda install`:  
```
conda install pytorch torchvision -c pytorch
conda install numpy
conda install ete3
```

### Running
For running in metacentrum is prepared script `run_train.sh` with one parameter. It is name of sequence to procced. 
Script downloads , rpXX and seed sequences. Then run train process in parallel on many nodes for individual 
sequences. 

For creating .newick files with phylogenetic trees run this script `fast_tree_run.sh`

All backbone scripts have param parse interface set to cooperate with `--Pfam_id` and `--RPgroup` options.

##### Backbone scripts
```
proc_msa.py
train.py
analyze_model1.py
highlight.py
```

##### Supplementary scripts
Scripts mentioned above are for running multiple jobs for selected files. `fast_tree_batch.sh` a `train_vae.sh` are
batch scripts per specific file run by scripts above.
  
### Scripts

* download_MSA.py - it downloads full, rpXX and seed of given Pfam_id and creates .fasta
* proc_msa.py -
 
        run python3 ./script/prac_msa.py --Pfam_id id [--RPgroup rp default full]
        creates output directory ./output/{id}_{rp}/  where results are stored 
        All following scripts with some options will work with some directory
* train.py - training of model, `--num_epoch 10000 --weight_decay 0.01` options usage required out 
of standard interface `--Pfam_id` and `--RPgroup`.
* analyze_model1.py - creates latent space representation
* highlight.py - shows subfamily mapping to latent space learned by selected `--Pfam_id` and `--RPgroup`
family. `--High` option for file (multiple files) to be highlighted in selected latent space