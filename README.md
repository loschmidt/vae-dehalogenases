# VAEs for ancestral sequence reconstruction of HLDS

This repository contains scripts for the paper 
<em>Engineering Dehalogenase Enzymes using Variational Autoencoder-Generated Latent Spaces and Microfluidics</em>.

The work is based on the code published in nature communications paper <em>Deciphering protein evolution and fitness
landscapes with latent space models</em> (https://doi.org/10.1038/s41467-019-13633-0)

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

### Dataset

Dataset are provided in tar archive and can be extracted using `setup_datasets.sh` script. 
Simply run `bash setup_datasets.sh` in root directory of this repository.
The above command will create directory **datasets** with description and further information
about preparing your custom data for phylogenetic mapping into the latent space in `README_datasets.md`.
In the provided README user can find instruction 

## Usage

Use script ``setup_datasets.sh`` to setup datasets directory

For running the scripts user has to be in the ``scripts/`` directory. 

```bash
cd scripts
```
Users can specify the desired configuration of the model set ``status on``.
The demonstration configuration file include setups for Model1 and Model2 used in our study.

```bash
vim model_configurations/runner-conf.json
```
<em>runner-conf.json</em> is the default configuration file.
If you want to run your custom analysis you can add there configuration entry on set it to **on**
or create a new configuration file with you models and modify it as needed. Then you need to use additional
parameter while running commands below `--json path_to_conf.json`

Then user can run pipeline preprocessing selected MSA from dataset directory via config file, train model, and generate desired statics by running:



```bash
python3 runner.py msa_handlers/msa_preprocessor.py
python3 runner.py train.py
python3 runner.py benchmark.py
python3 runner.py run_task.py --run_generative_evaluation
```
After that result can be found in ``../results/dir_name/experimet_name`` directory, where ``dir`` and ``experiment`` name are specified in configuration file. 
Commands above generate Figure 2a-c.

#### Ancestors
Following ancestors' generation can be done simply via running command corresponding to straight line evolution strategy:

```bash
python3 runner.py --run_task.py --run_straight_evolution
``` 

We did examine more strategies: 

<em>Random mutagenesis</em> modifies input query sequence and maps it into the latent space and picks the one with 
the highest proximity to the latent space origin (run_random_mutagenesis).

<em>CMA-ES evolution</em> multicriterio optimization in the latent space.
We focused on simple straight evolutionary strategy in our paper.
```bash
python3 runner.py --run_task.py --run_random_mutanesis
python3 runner.py --run_task.py --run_evolution
``` 
Results can be found in the same directory in the subdirectory ``Highlights/``

### Support scripts

To get latent space run:
```bash
python3 runner.py --run_task.py --run_plot_latent_space
```

The statistic plot for the profile of evolution in the latent space (Figure 3c) can be produced:

```bash
python3 supportScripts/dual_axis.py --csv ../results/path/to/experiment/higlight_dir/generated.csv --pos ""
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
