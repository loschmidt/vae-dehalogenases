"""
Map given tree to the latent space

Tree is supposed to be composed of sequences which are aligned and trimmed to the training MSA
"""
import os
import pickle

from Bio import Phylo
from matplotlib import pyplot as plt

from analyzer import AncestorsHandler
from sequence_transformer import Transformer
from parser_handler import CmdHandler
from VAE_accessor import VAEAccessor
from vae_utils.my_utils import acc_to_mu
from VAE_logger import Capturing
from msa_handlers.msa_preparation import MSA


def map_tree(cmd_handler: CmdHandler, embedding_dict: dict, tree_dir=None):
    """
    @cmdHandler: command line parser object
    @embedding_dict: dictionary with sequence accessions and mus
    @tree_dir: directory to store tree stats, default highlights/latent_space_trees/custom_tree directory
    """
    tree_path = cmd_handler.tree_nwk
    msa_tree_path = cmd_handler.tree_msa
    tree_msa_align = cmd_handler.tree_msa_align
    tree_leaves_to_show = cmd_handler.tree_leaves.split(",")
    check_leaves = tree_leaves_to_show[0] != ''

    if tree_dir is None:
        tree_dir = os.path.join(cmd_handler.high_fld, "latent_space_trees", "custom_tree")
    os.makedirs(tree_dir, exist_ok=True)

    if tree_path is None or msa_tree_path is None:
        print("   Please provide the newick tree file and the MSA tree sequences in the config file")
        exit(0)

    print(f" The tree figure and data will be generated into {tree_dir}")
    # Init all objects
    aligner = AncestorsHandler(cmd_handler)
    transformer = Transformer(cmd_handler)
    vae = VAEAccessor(cmd_handler, cmd_handler.get_model_to_load())

    # Align to the training MSA if required
    if tree_msa_align:
        print(" Aligning tree MSA to the  training dataset leaves...")
        msa = aligner.align_tree_msa_to_msa(msa_tree_path, tree_path, tree_dir)
    else:
        print(" Working with tree MSA as aligned to training, if not please rerun with --tree_msa_align flag")
        with Capturing():
            msa = MSA.load_msa(msa_tree_path)

    leaf_mus, leaf_keys, ancestor_nodes = acc_to_mu(embedding_dict, list(msa.keys()))

    # Encode tree ancestors to the latent space
    msa_ancestors = {k: msa[k] for k in ancestor_nodes}
    binaries, weights, keys = transformer.sequence_dict_to_binary(msa_ancestors)
    mus, _ = vae.propagate_through_VAE(binaries, weights, keys)

    # Make mapping key to mu and add the leaf sequences
    key_to_mu = {k: m for k, m in zip(keys, mus)}
    for leaf_key, leaf_mu in zip(embedding_dict['keys'], embedding_dict['mu']):
        key_to_mu[leaf_key] = leaf_mu
    # Get branches from tree
    tree = Phylo.read(tree_path, "newick")
    terminals = tree.get_terminals()

    branches = []
    for t in terminals:
        branch = []
        for node in tree.get_path(t):
            if node in terminals:
                branch.append(node.name.replace("\"", ""))
            else:
                branch.append(str(node.confidence))
        branches.append(branch)

    # Prepare edges for plotting
    edges = []
    for branch in branches:
        if check_leaves and (branch[-1] not in tree_leaves_to_show):
            continue
        branch_edges = [(key_to_mu[branch[i - 1]], key_to_mu[branch[i]]) for i in range(1, len(branch))]
        edges.extend(branch_edges)

    # Plot tree into latent space
    plt.plot(embedding_dict['mu'][:, 0], embedding_dict['mu'][:, 1], '.', alpha=0.5, markersize=3, )
    for p1, p2 in edges:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],  '-o', color='black', linewidth=.5)
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.show()
    # plt.savefig(os.path.join(plot_dir, f"tree_msa{i}.png"), dpi=600)
    # print(f"  Saving to tree_msa{i}.png")
    # plt.clf()


def run_map_tree():
    """
    Map tree to the latent space
    The tree is stored in the highlights/latent_space_trees/custom_tree directory
    """
    cmdline = CmdHandler()
    try:
        with open(os.path.join(cmdline.pickles_fld, 'embeddings.pkl'), 'rb') as embed_file:
            embeddings_pkl = pickle.load(embed_file)
    except FileNotFoundError:
        print("   Please prepare MSA embedding before you run this case!!\n"
              "   run python3 runner.py run_task.py --run_package_generative_plots --json config.file")
        exit(1)

    embeddings_dict = embeddings_pkl
    map_tree(cmdline, embeddings_dict)

