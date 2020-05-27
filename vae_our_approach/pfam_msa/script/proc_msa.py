__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/10 19:51:12"

import pickle
import numpy as np
import argparse

import subprocess as sp   ## command line hangling

PRAGMA_REFERENCE = False

parser = argparse.ArgumentParser(description = "Process given MSA")
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
parser.add_argument("--Length", help = "the length of final sequences (to fit width fit trained model); e.g. 97")

## To get different input from MSA directory and to save result to seperate directory
## for parallel run on multiple computation nodes
parser.add_argument("--RPgroup", help = "RP specifier of given Pfam_id family, e.g. RP15")

if PRAGMA_REFERENCE:
    parser.add_argument("--Ref_seq", help = "the reference sequence; e.g. TENA_HUMAN/804-884")

args = parser.parse_args()
pfam_id = args.Pfam_id

target_length = args.Length

if PRAGMA_REFERENCE:
    ref_seq = args.Ref_seq
## get RP subgroup if it is specified
rp_id = ""
if args.RPgroup is not None:
    rp_id = args.RPgroup
else:
    ## using full sequence
    rp_id = "full"

## Prepare input name and create directory name
gen_dir_id = "{0}_{1}".format(pfam_id, rp_id)
out_dir = "./output/{0}".format(gen_dir_id)
file_name = "./MSA/{0}_{1}.txt".format(pfam_id, rp_id)

## Create output directory in ../output/out_dir/
sp.run("mkdir -p {0}".format(out_dir), shell=True)

if PRAGMA_REFERENCE:
    query_seq_id =  ref_seq #"TENA_HUMAN/804-884"

## read all the sequences into a dictionary
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

## remove gaps in the query sequences
if PRAGMA_REFERENCE:
    query_seq = seq_dict[query_seq_id] ## with gaps
    idx = [ s == "-" or s == "." for s in query_seq]
    for k in seq_dict.keys():
        seq_dict[k] = [seq_dict[k][i] for i in range(len(seq_dict[k])) if idx[i] == False]
    query_seq = seq_dict[query_seq_id] ## without gaps

## remove sequences with too many gaps
seq_id = list(seq_dict.keys())
for k in seq_id:
    if seq_dict[k].count("-") + seq_dict[k].count(".") > len(seq_dict[k]) * 0.8:
        seq_dict.pop(k)

with open(out_dir + "/seq_dict.pkl", 'wb') as file_handle:
    pickle.dump(seq_dict, file_handle)
        
## convert aa type into num 0-20
aa = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W',
      'O']
aa_index = {}
aa_index['-'] = 0
aa_index['.'] = 0
i = 1
for a in aa:
    aa_index[a] = i
    i += 1
with open(out_dir + "/aa_index.pkl", 'wb') as file_handle:
    pickle.dump(aa_index, file_handle)

## Remove everything with unexplored residues from dictionary
seq_msa = []
keys_list = []
for k in seq_dict.keys():
    if seq_dict[k].count('X') > 0 or seq_dict[k].count('Z') > 0:
        continue    
    seq_msa.append([aa_index[s] for s in seq_dict[k]])
    keys_list.append(k)
seq_msa = np.array(seq_msa)

with open(out_dir + "/keys_list.pkl", 'wb') as file_handle:
    pickle.dump(keys_list, file_handle)

## remove positions where too many sequences have gaps
pos_idx = []
for i in range(seq_msa.shape[1]):
    if np.sum(seq_msa[:,i] == 0) <= seq_msa.shape[0]*0.2:
        pos_idx.append(i)

if args.Length is not None:
    ## Length is set this have to have same length as target length
    if len(pos_idx) > target_length:
        ## Length is bigger, remove colums with most gaps
        seq_msa = seq_msa[:, np.array(pos_idx)]
        gaps_idx = {} ## count of gaps, index
        for i in range(len(pos_idx)):
            gaps_idx[i] = np.sum(seq_msa[:,i] == 0)
        ## Sort by length
        gaps_idx = {k: v for k, v in sorted(gaps_idx.items(), key=lambda item: item[1])}
        while seq_msa.shape[1] > target_length:
            ## Remove with most gaps
            k, v = gaps_idx.popitem()
            pos_idx.remove(k)

with open(out_dir + "/seq_pos_idx.pkl", 'wb') as file_handle:
    pickle.dump(pos_idx, file_handle)

seq_msa = seq_msa[:, np.array(pos_idx)]

## Fasta file names and sequencies in inner representation
fasta_keys = keys_list
fasta_seq_num = seq_msa

with open(out_dir + "/seq_msa.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa, file_handle)

## reweighting sequences
seq_weight = np.zeros(seq_msa.shape)
for j in range(seq_msa.shape[1]):
    aa_type, aa_counts = np.unique(seq_msa[:,j], return_counts = True)
    num_type = len(aa_type)
    aa_dict = {}
    for a in aa_type:
        aa_dict[a] = aa_counts[list(aa_type).index(a)]
    for i in range(seq_msa.shape[0]):
        seq_weight[i,j] = (1.0/num_type) * (1.0/aa_dict[seq_msa[i,j]])
tot_weight = np.sum(seq_weight)
seq_weight = seq_weight.sum(1) / tot_weight 
with open(out_dir + "/seq_weight.pkl", 'wb') as file_handle:
    pickle.dump(seq_weight, file_handle)

## Detect how many sequences was used and its average and max lenght
lengths = [len(i) for i in seq_msa]
av_len = 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))

print("="*60)
print("Pfam ID : {0} - {2}, Reference sequence {1}".format(pfam_id, PRAGMA_REFERENCE, rp_id))
print("Sequences used: {0}".format(seq_msa.shape[0]))
print("Average lenght: {0}".format(av_len))
print("Max lenght of sequence: {0}".format(max(lengths)))
print("="*60)
print()

## change aa numbering into binary
K = 21 ## num of classes of aa
D = np.identity(K)
num_seq = seq_msa.shape[0]
len_seq_msa = seq_msa.shape[1]
seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
for i in range(num_seq):
    seq_msa_binary[i,:,:] = D[seq_msa[i]]

with open(out_dir + "/seq_msa_binary.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa_binary, file_handle)

###################################################################
## Prepare fasta structure fo FastTree phylogenetic tree generation
fasta_dict = {}

## Reverse transformation
reverse_index = {}
reverse_index[0] = '-'

i = 1
for a in aa:
    reverse_index[i] = a
    i += 1

print ("Count of keys is : {0}".format(len(fasta_keys)))

## Sequencies back to aminoacid representation
for i in range(len(fasta_keys)):
    to_amino = fasta_seq_num[i]
    amino_seq = [reverse_index[s] for s in to_amino]
    fasta_dict[fasta_keys[i]] = ''.join([str(elem) for elem in amino_seq])

## Now transform sequences back to fasta
with open("./FastTree/{0}_for_tree_gen.fasta".format(gen_dir_id), 'w') as file_handle:
    for seq_name, seq in fasta_dict.items():
        file_handle.write(">" + seq_name + "\n" + seq + "\n")
