__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/04/30 00:33:12"

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp   ## command line hangling

import torch
from torch.utils.data import Dataset, DataLoader
from supportClasses.MSA_VAE_loader import *
from VAE_model import *

parser = argparse.ArgumentParser(description='Parameters for training the model')
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
parser.add_argument("--RPgroup", help = "RP specifier of given Pfam_id family, e.g. RP15")
parser.add_argument("--High", help = "Sequencies to highlight, array of files in stockholm to be depict here", type=str)

args = parser.parse_args()

# Prepare input name and create directory name
gen_dir_id = ""
out_dir = "./output"
## get RP subgroup if it is specified
if args.RPgroup is not None and args.Pfam_id is not None:
    rp_id = args.RPgroup
    pfam_id = args.Pfam_id
    gen_dir_id = "{0}_{1}".format(pfam_id, rp_id)
    out_dir = "./output/{0}".format(gen_dir_id)

if args.High is None:
    print("Nothing to highlight!! Use --High file_name in Stockholm format")
    exit(0)

in_files = [item for item in args.High.split(',')]
## Not array of files just one file
if len(in_files) == 0:
    in_files = [args.High]

## create directory for highlight
sp.run("mkdir -p {0}/highlight".format(out_dir), shell=True)

## prepare model to mapping from highlighting files
with open(out_dir + "/seq_msa_binary.pkl", 'rb') as file_handle:
    msa_original_binary = pickle.load(file_handle)
len_protein = msa_original_binary.shape[1]
num_res_type = msa_original_binary.shape[2]
vae = VAE(20, 2, len_protein * num_res_type, [100])
vae.cuda()
vae.load_state_dict(torch.load(out_dir + "/model/vae_0.01_fold_0.model"))

## read data from highlighting files, transform to binary and run through VAE
dict_lat_sps = {}
for rps in in_files:
    ## Prepare key to dictionary
    dict_key = rps.split(".")[-2].split("/")[-1]
    ## Load data and prepare them for VAE
    msa_vae = MSA_VAE_loader(rps, gen_dir_id)
    ret = msa_vae.binary_seq(len_protein)
    msa_binary = ret[0]
    msa_keys = ret[1]

    num_seq = msa_binary.shape[0]
    msa_weight = np.ones(num_seq) / num_seq
    msa_weight = msa_weight.astype(np.float32)

    batch_size = num_seq

    msa_binary = msa_binary.reshape((num_seq, -1))
    msa_binary = msa_binary.astype(np.float32)

    train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
    train_data_loader = DataLoader(train_data, batch_size = batch_size)

    mu_list = []
    sigma_list = []
    for idx, data in enumerate(train_data_loader):
        msa, weight, key = data
        with torch.no_grad():
            msa = msa.cuda()
            mu, sigma = vae.encoder(msa)
            mu_list.append(mu.cpu().data.numpy())
            sigma_list.append(sigma.cpu().data.numpy())

    mu = np.vstack(mu_list)
    sigma = np.vstack(sigma_list)

    rp_latent_space = {}
    rp_latent_space['mu'] = mu
    rp_latent_space['sigma'] = sigma
    dict_lat_sps[dict_key] = rp_latent_space

## Get referent latent space (it'll be background in plots)
with open(out_dir + "/latent_space.pkl", 'rb') as file_handle:
    latent_space = pickle.load(file_handle)
mu = latent_space['mu']
sigma = latent_space['sigma']
key = latent_space['key']

## ---------------
## Plotting part
## ---------------
colors = ["red", "green", "brown", "black", "yellow", "magenta"]

## Get keys and show them
labels = ["latent space"]
for file_name in in_files:
    label_name = file_name.split(".")[-2].split("/")[-1]
    labels.append(label_name)

cnt_of_subplots = len(labels) + 1 ## plus everything
str_plot = str(cnt_of_subplots) + "2"

plt.figure(0)
plt.clf()
## Initial plot with latent space
cur_sub = str_plot + str(1)
plt.subplot(int(cur_sub))
plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, label=labels[0])
plt.title = labels[0]
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()

## plot individual subplots
color_i = 0
for k in dict_lat_sps.keys():
    sub_mu = dict_lat_sps[k]['mu']
    col = colors[color_i]
    color_i += 1
    cur_sub = str_plot + str(color_i+1)
    plt.subplot(int(cur_sub))
    plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, label=labels[0]) ## Original latent space
    plt.plot(sub_mu[:, 0], sub_mu[:, 1], '.', color=col, alpha=1, markersize=3, label=labels[color_i]) ## Overlap original with subfamily
    plt.title = labels[color_i].split("_")[-1]

    plt.legend(loc='upper right')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.tight_layout()

## Print everything at one last plot
cur_sub = str_plot + str(cnt_of_subplots)
plt.subplot(int(cur_sub))
plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, label=labels[0]) ## Original latent space
color_i = 0
for k in dict_lat_sps.keys():
    sub_mu = dict_lat_sps[k]['mu']
    col = colors[color_i]
    color_i += 1
    plt.plot(sub_mu[:, 0], sub_mu[:, 1], '.', color=col, alpha=1, markersize=3, label=labels[color_i]) ## Overlap original with subfamily
graph_str = "All RPs"
plt.legend(loc='upper right')
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()

save_name = out_dir + "/highlight/"
for i in range(1,len(labels)):
    save_name += labels[i].split("_")[-1] + "_"
save_name += "highlight.png"
print(" Saving plot to : {0}".format(save_name))
plt.savefig(save_name)
