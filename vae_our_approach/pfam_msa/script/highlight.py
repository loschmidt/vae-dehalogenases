__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/04/30 00:33:12"

import argparse
import pickle
import matplotlib.pyplot as plt
import subprocess as sp   ## command line hangling

parser = argparse.ArgumentParser(description='Parameters for training the model')
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
parser.add_argument("--RPgroup", help = "RP specifier of given Pfam_id family, e.g. RP15")
parser.add_argument("--High", help = "Sequencies to highlight")

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

## read all the sequences into a dictionary
file_name = args.High
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

## create directory for highlight
sp.run("mkdir -p {0}/highlight".format(out_dir))

## Get keys and show them


group_name = file_name.split(".")[0].split("/")[-1]
colors = ("red", "blue")
groups = ("latent space", group_name)

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
names = seq_dict.keys()
idx = []
succ = 0
fail = 0
for n in names:
    try:
        idx.append(key2idx[n])
        succ += 1
    except KeyError as e:
        fail += 1

print("Success: ", succ, " Fails: ", fail)

plt.figure(0)
plt.clf()
plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3)
## Plot selected
plt.plot(mu[idx:, 0], mu[idx:, 1], '.', color='red', alpha=0.1, markersize=5)
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig(out_dir + "/{0}_{1}_highlight.png".format(gen_dir_id, group_name))