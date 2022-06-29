# VAEs for ancestral sequence reconstruction

**authors:**  Pavel Kohout <xkohou15@stud.fit.vutbr.cz>

This repository contains scripts for diploma thesis work by Pavel Kohout at Brno University of Technology 

## Installation

For the installation and running purposes, I strongly recommend installing [Anaconda](https://www.anaconda.com/products/distribution) and adding additional packages using conda installation via commands:

```bash
conda create -n vae_env python=3.6
conda activate vae_env
conda install pytorch torchvision -c pytorch
```
Then python packages can be installed into the environment from ``requrements.txt`` using standard pip commands:
```bash
pip install -r requirements.txt
``` 
For alignment management scripts are using clustalOmega tool. Its binary can be found in ``clustal/`` directory. If there is no binary the clustalOmega binary can be downloaded from [here](http://www.clustal.org/omega/)

## Usage

Use script ``setup_datasets.sh`` to setup datasets directory

For running the scripts user has to be in the ``scripts/`` directory. 

```bash
cd scripts
```
Users can specify the desired configuration of the model set ``status on`` in

```bash
vim model_configurations/models.json
```
Then user can run pipeline preprocessing selected MSA from dataset directory via config file, train model, and generate desired statics by running:

```bash
python3 runner.py msa_handlers/msa_preprocessor.py
python3 runner.py train.py
python3 runner.py benchmark.py
python3 runner.py run_task.py --run_generative_evaluation
```
After that result can be found in ``../results/dir_name/experimet_name`` directory, where ``dir`` and ``experiment`` name are specified in configuration file. 
Following ancestors' generation can be done simply via running one/every command corresponding to strategies described in the diploma thesis:

```bash
python3 runner.py --run_task.py --run_random_mutanesis
python3 runner.py --run_task.py --run_straight_evolution
python3 runner.py --run_task.py --run_evolution
``` 
Results can be found in the same directory in the subdirectory ``Highlights/``

### Support scripts

The latent space figure for the current model can be created by running

```bash 
python3 runner.py analyzer.py
```

The statistic plot depicted in the appendix section can be generated via command

```bash
python3 supportScripts/dual_axis.py --csv ../results/path/to/experiment/higlight_dir/generated.csv
```

## Description of repository

- **scripts**  : program source files
- **results**  : directory for results
- **pbs_scripts** : scripts for remote job runs. trainer.sh training script
- **datasets.tar.gz** : compressed datasets directory, 
  - **datasets** : datasets for experiments create from compressed datasets.tar.gz via using script setup_datasets.sh
- **clustal**  : clustal omega binary file directory, clustalo binary
- **meta_scripts** : storage of pbs scripts variants 
- **requirements.txt** : python pip dependencies
- **text** : latex source files 
- **diploma_thesis_xkohou15.pdf** : pdf file of diploma thesis
