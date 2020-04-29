__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/17 21:29:29"

import pickle
from ete3 import Tree
from sys import exit
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
from sklearn import linear_model
import time
import argparse
import subprocess as sp   ## command line hangling

parser = argparse.ArgumentParser(description='Parameters for clustering')
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
parser.add_argument("--RPgroup", help = "RP specifier of given Pfam_id family, e.g. RP15")

args = parser.parse_args()

# Prepare input name and create directory name
gen_dir_id = ""
src_dir = "./output"
out_dir = "./output"
## get RP subgroup if it is specified
if args.RPgroup is not None and args.Pfam_id is not None:
    rp_id = args.RPgroup
    pfam_id = args.Pfam_id
    gen_dir_id = "{0}_{1}".format(pfam_id, rp_id)
    src_dir = "./output/{0}".format(gen_dir_id)
    out_dir = "./output/{0}/clustering".format(gen_dir_id)

## Prepare cluster directory if not exists
sp.run("mkdir {0}".format(out_dir), shell=True)

## read latent space representation
with open(src_dir + "/latent_space.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
key = data['key']
mu = data['mu']
sigma = data['sigma']

key2idx = {}
for i in range(len(key)):
    key2idx[key[i]] = i

## read tree
t = Tree("./FastTree/{0}.newick".format(gen_dir_id), format = 1)
num_leaf = len(t)
t.name = str(num_leaf)
##leaf_idx = []
##ancestral_idx = []
##for i in range(len(key)):
  ##  if int(key[i]) < num_leaf:
    ##    leaf_idx.append(i)
    ##else:
    ##    ancestral_idx.append(i)
        
for node in t.traverse('preorder'):
    if node.is_root():
        node.add_feature('anc', [])
        node.add_feature('sumdist', 0)
    else:
        node.add_feature('anc', node.up.anc + [node.up.name])
        node.add_feature('sumdist', node.up.sumdist + node.dist)

color_names = list(mpl.colors.cnames.keys())
dist_cutoff = 0.5
head_node_names = []
for node in t.traverse('preorder'):
    if node.is_leaf() and node.sumdist < dist_cutoff:
        head_node_names.append(node.name)
    if (not node.is_leaf()) and node.sumdist > dist_cutoff and node.up.sumdist < dist_cutoff:
        head_node_names.append(node.name)

cluster_node_names = {}
for name in head_node_names:
    cluster_node_names[name] = []
    for node in (t&name).traverse('preorder'):
        cluster_node_names[name].append(node.name)

##print()
##print("Head printing:")
##print (head_node_names[0:20])
##
##print()
##print()
##print("Cluster printing:")
##ip = 0
##for k, n in cluster_node_names.items():
##    if ip < 1:
##        print(k, " : ", n)
##    ip += 1
##print()
##print()
##
##print("Key2idx printing:")
##ip = 0
##for k, n in key2idx.items():
##    if ip < 1:
##        print(k, " : ", n)
##    ip += 1
##print()
##print()
##print("Taht one")
##print(key2idx['A0A2A5NU41_9MICO/6-128'])

fig = plt.figure(0)
fig.clf()

fail = 0
for i in range(len(head_node_names)):
    names = cluster_node_names[head_node_names[i]]
    ##idx =  [ key2idx[n] for n in names[1::2]] ##[ key2idx[n] for n in names]
    idx = []
    for n in names[1::2]:
        try:
            idx.append(key2idx[n])
         ##   print ("Success")
        except KeyError as e:
            fail += 1
    ##print("Ploting ; ", mu[idx,0])  
    ##print("Ploting ; ", mu[idx,1]) 
    plt.plot(mu[idx,0], mu[idx,1], '.', markersize = 2, label = head_node_names[i])
# plt.xlim((-6.5,10))
# plt.ylim((-8,8))    
# plt.title("dist_cutoff: {:.2f}, num of cluster: {:}".format(dist_cutoff, len(head_node_names)))
# plt.legend(markerscale= 4, loc = 'upper left')
plt.xlabel(r'$Z_1$')
plt.ylabel(r'$Z_2$')
plt.tight_layout()
fig.savefig(out_dir + "/cluster.eps")

main_nodes = []

num_of_nodes = 11
proc_nodes = 0

for node in t.traverse('preorder'):
    if proc_nodes > num_of_nodes:
        break
    main_nodes.append(node.name)
    proc_nodes += 1

print("="*40)
print("Choose your branches")
print(main_nodes)

## zoom in branches
for branch in main_nodes[4:11]:
    print("Generating zoom branch : {0}".format(branch))
    sub_t = t&branch
    dist_cutoff = sub_t.sumdist + 0.3
    head_node_names = []
    for node in sub_t.traverse('preorder'):
        if node.is_leaf() and node.sumdist < dist_cutoff:
            head_node_names.append(node.name)
        if (not node.is_leaf()) and node.sumdist > dist_cutoff and node.up.sumdist < dist_cutoff:
            head_node_names.append(node.name)

    cluster_node_names = {}
    for name in head_node_names:
        cluster_node_names[name] = []
        for node in (t&name).traverse('preorder'):
            cluster_node_names[name].append(node.name)

    fig = plt.figure(1)
    fig.clf()
    for i in range(len(head_node_names)):
        names = cluster_node_names[head_node_names[i]]
        ##idx = [ key2idx[n] for n in names]
        idx = []
        for n in names[1::2]:
            try:
                idx.append(key2idx[n])
            ##   print ("Success")
            except KeyError as e:
                fail += 1
        plt.plot(mu[idx,0], mu[idx,1], '.', markersize = 2, label = head_node_names[i])
    # plt.xlim((-6.5,10))
    # plt.ylim((-8,8))    
    # plt.title("dist_cutoff: {:.2f}, num of cluster: {:}".format(dist_cutoff, len(head_node_names)))
    # plt.legend(markerscale= 4, loc = 'upper right')
    plt.xlabel(r'$Z_1$')
    plt.ylabel(r'$Z_2$')
    plt.tight_layout()
    fig.savefig(out_dir + "/branch_{}_cluster.eps".format(branch.replace("/","_")))


#plt.show()


