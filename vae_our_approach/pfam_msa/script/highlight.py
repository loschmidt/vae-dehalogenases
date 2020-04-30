__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/04/30 00:33:12"

import argparse
import pickle
import matplotlib.pyplot as plt
import subprocess as sp   ## command line hangling

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
    print("Nothing to highlight!! Use --High file_name in Stackholm format")
    exit(0)

in_files = [int(item) for item in args.High.split(',')]
## read all the sequences into a dictionary
high_seq = {}
for file_name in in_files:
    seq_dict = {}
    with open(file_name, 'r') as file_handle:
        for line in file_handle:
            if line[0] == "#" or line[0] == "/" or line[0] == "":
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            seq_id, seq = line.split()
            seq_dict[seq_id] = seq.upper()
    high_seq[file_name] = seq_dict.copy()

## create directory for highlight
sp.run("mkdir -p {0}/highlight".format(out_dir), shell=True)

## Get keys and show them
labels = ["latent space"]
for file_name in in_files:
    label_name = file_name.split(".")[0].split("/")[-1]
    labels.append(label_name)

colors = ["red", "brown", "black", "green", "yellow", "magenta"]

with open(out_dir + "/latent_space.pkl", 'rb') as file_handle:
    latent_space = pickle.load(file_handle)
mu = latent_space['mu']
sigma = latent_space['sigma']
key = latent_space['key']

## Key to representation of index
key2idx = {}
for i in range(len(key)):
    key2idx[key[i]] = i

## Name to highlight and gets its indices
high_idx = {}
for file_name in in_files:
    names = high_seq[file_name].keys()
    idx = []
    succ = 0
    fail = 0
    for n in names:
        try:
            idx.append(int(key2idx[n]))
            succ += 1
        except KeyError as e:
            fail += 1
    print(file_name, " Success: ", succ, " Fails: ", fail)
    high_idx[file_name] = idx.copy()

plt.figure(0)
plt.clf()
plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, label=labels[0])

## Plot selected
color_i = 0
for file_name in in_files:
    idx = high_idx[file_name]
    col = colors[color_i]
    color_i += 1
    plt.plot(mu[idx, 0], mu[idx, 1], '.', color=col, alpha=1, markersize=3, label=labels[color_i])

plt.legend(loc='upper right')
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig(out_dir + "/highlight/{0}_{1}_highlight.png".format(gen_dir_id, labels[0]))
