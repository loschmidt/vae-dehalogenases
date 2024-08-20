# Variational Autoencoders for Protein Ancestral Sequence Reconstruction

This repository contains scripts for the paper:<br>
Kohout P, Vasina M, Majerova M, Novakova V, Damborsky J, Bednar D, Marek M, Prokop Z, Mazurenko S. Engineering Dehalogenase Enzymes using Variational Autoencoder-Generated Latent Spaces and Microfluidics. (DOI 10.26434/chemrxiv-2023-jcds7)

The work is based on the code published in nature communications paper <em>Deciphering protein evolution and fitness
landscapes with latent space models</em> (https://doi.org/10.1038/s41467-019-13633-0)

## Installation

For the installation and running purposes, we recommend installing [Anaconda](https://www.anaconda.com/products/distribution) and adding additional packages using conda installation via commands:

```bash
conda create -n vae_env python=3.6
conda activate vae_env
conda install pytorch torchvision -c pytorch
```
Then python packages can be installed into the environment from ``requrements.txt`` using standard pip commands:
```bash
pip install -r requirements.txt
``` 
For alignment management our, scripts use the Clustal Omega tool. Its binary can be found in ``clustal/`` directory. If there is no binary the Clustal Omega binary can be downloaded from [here](http://www.clustal.org/omega/)

### Dataset

Dataset are provided in the tar archive and can be extracted using `setup_datasets.sh` script. 
Simply run `bash setup_datasets.sh` in the root directory of this repository.
The above command will create the directory **datasets** with a description of extracted files and further information
about preparing your custom data for phylogenetic mapping into the latent space in `README_datasets.md`.

## Usage

Use script ``setup_datasets.sh`` to setup the datasets directory

For running the scripts, the user has to be in the ``scripts/`` directory. 

```bash
cd scripts
```
Users can specify the desired configuration of the model set ``status on``.
The demonstration configuration file includes setups for Model1 and Model2 used in our study.

```bash
vim model_configurations/runner-conf.json
```
<em>runner-conf.json</em> is the default configuration file with examples.
If you want to run your custom analysis, you can add their configuration entry and set it to **on**
or modify a new configuration file with your models as needed. Then you need to use an additional
parameter while running commands below `--json path_to_conf.json`

Then, the user can run pipeline preprocessing the selected MSA from the dataset directory via config file, train the model, and generate desired statics by running:


```bash
python3 runner.py msa_handlers/msa_preprocessor.py --json model_configurations/runner-conf.json
python3 runner.py train.py --json model_configurations/runner-conf.json
python3 runner.py benchmark.py --json model_configurations/runner-conf.json
python3 runner.py run_task.py --run_generative_evaluation --json model_configurations/runner-conf.json
```
After that, the result can be found in the ``../results/dir_name/experiment_name`` directory, where ``dir`` and ``experiment``names are specified in the configuration file. 
The commands above generate Figure 2A-C in the paper.

#### Ancestors
Ancestors generation can be done simply via running the command corresponding to the straight line evolution strategy:

```bash
python3 runner.py run_task.py --run_straight_evolution --json model_configurations/runner-conf.json
``` 

We also examined more strategies: 

<em>Random mutagenesis</em> modifies the input query sequence, maps it into the latent space, and picks the one with 
the closest variant to the latent space origin (run_random_mutagenesis).

<em>CMA-ES evolution</em> multi-objective optimization in the latent space.
We focused on the simple straight evolutionary strategy in our paper.
```bash
python3 runner.py run_task.py --run_random_mutanesis --json model_configurations/runner-conf.json
python3 runner.py run_task.py --run_evolution --json model_configurations/runner-conf.json
``` 
Results can be found in the same directory in the subdirectory ``Highlights/``

### Support scripts

To get the latent space, run:
```bash
python3 runner.py run_task.py --run_plot_latent_space --json model_configurations/runner-conf.json
```

The statistics plot for the evolutionary profile in the latent space (Figure 3C in the paper) can be produced:

```bash
python3 supportscripts/dual_axis.py --csv ../results/path/to/experiment/higlight_dir/selected_strategy_profile.csv --pos "" --o path_to_profile.jpg
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
