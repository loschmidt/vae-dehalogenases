__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import pickle
import sys
import inspect

from typing import List
from math import cos, sin

import numpy as np
import pandas as pd
from Bio import Phylo
from ete3 import Tree
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from scipy.spatial import distance
from sklearn.decomposition import PCA

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from analyzer import AncestorsHandler
from sequence_transformer import Transformer
from parser_handler import CmdHandler
from project_enums import VaePaths
from VAE_accessor import VAEAccessor


class MSASubsampler:
    """
    The class creates MSAs for Fireprot ASR with 100 sequences all the time including query
    """

    def __init__(self, setuper: CmdHandler):
        self.pickle = setuper.pickles_fld
        self.tree = os.path.join(setuper.high_fld, VaePaths.TREE_EVALUATION_DIR.value)
        self.target_dir = VaePaths.STATS_DATA_SOURCE.value + "fireprots_msas/"
        self.query_id = setuper.query_id
        self.setuper = setuper
        self.setup_output_folder()

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.tree, exist_ok=True)

    @staticmethod
    def fasta_record(file, key, seq):
        n = 80
        file.write(">" + key + "\n")
        for i in range(0, len(seq), n):
            file.write(seq[i:i + n] + "\n")

    def sample_msa(self, n: int, seq_cnt: int = 2000):
        """ Sample MSA from training input space """
        with open(self.pickle + "/training_alignment.pkl", "rb") as file_handle:
            msa = pickle.load(file_handle)
        msa_keys = np.array(list(msa.keys()))
        query = msa[self.query_id]
        file_templ = self.target_dir + "fireprot_msa{}.fasta"

        for msa_i in range(n):
            file_name = file_templ.format(msa_i)
            print("   ", file_name)

            with open(file_name, 'w') as file_handle:
                # fasta_record(file_handle, self.query_id, "".join(query))  # Store query
                selected_seqs = msa_keys[np.random.randint(1, msa_keys.shape[0], seq_cnt, dtype=np.int)]

                for key in selected_seqs:
                    MSASubsampler.fasta_record(file_handle, key, "".join(msa[key]))

    def sample_representative(self):
        """ Sample MSA from the circle with query like radius """

        def vector_rotation(vec, angle):
            theta = np.deg2rad(angle)
            rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            return np.dot(rot, vec)

        def closest_node(node, nodes):
            closest_index = distance.cdist(np.array([node]), nodes).argmin()
            return closest_index

        with open(self.pickle + "/training_alignment.pkl", "rb") as file_handle:
            msa = pickle.load(file_handle)
        query = msa[self.query_id]

        try:
            with open(os.path.join(self.pickle, 'embeddings.pkl'), 'rb') as embed_file:
                embeddings_pkl = pickle.load(embed_file)
            key_to_mu = {k: mu for k, mu in zip(embeddings_pkl['keys'], embeddings_pkl['mu'])}
            mu = embeddings_pkl['mu']
            keys = np.array(embeddings_pkl['keys'])
        except FileNotFoundError:
            print("   Please prepare MSA embedding before you run this case!!\n"
                  "   run python3 runner.py run_task.py --run_package_generative_plots --json config.file")
            exit(1)

        # Get sequences on the circle around latent space
        query_emb = key_to_mu[self.query_id]
        latent_space_points = [vector_rotation(query_emb, i * 2) for i in range(1, 181)]  # 180 points in circle
        mu_idx = [closest_node(p, mu) for p in latent_space_points]
        selected_seqs = [self.query_id]
        selected_seqs.extend(keys[np.sort(mu_idx)])
        # Only unique sequences
        unique_keys = list(set(selected_seqs))
        print(f" The total number of selected sequences is {len(selected_seqs)} out of are unique {len(unique_keys)}")

        # Store it in fasta
        file_path = os.path.join(self.tree, 'circularSample.fasta')
        with open(file_path, 'w') as file_handle:
            for key in unique_keys:
                MSASubsampler.fasta_record(file_handle, key, "".join(msa[key]))
        return file_path


class AncestralTree:
    """
    This class highlights the individual levels of Ancestral tree from Fireprot in the latent space
    """

    def __init__(self, setuper: CmdHandler):
        self.model_name = setuper.get_model_to_load()
        self.vae = VAEAccessor(setuper, self.model_name)
        self.aligner_obj = AncestorsHandler(setuper)
        self.transformer = Transformer(setuper)
        self.data_dir = VaePaths.STATS_DATA_SOURCE.value
        self.target_dir = setuper.high_fld + "/" + VaePaths.TREE_EVALUATION_DIR.value + "/"
        self.coord_dir = os.path.join(self.target_dir, "coordinates_data/")

        dataset = (setuper.in_file.split("/")[-1]).split(".")[0]
        self.tree_dir = VaePaths.STATS_DATA_TREE.value + dataset + "/"

        self.setup_output_folder()

        self.max_depth = 6

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.tree_dir, exist_ok=True)
        os.makedirs(self.coord_dir, exist_ok=True)

    def get_tree_levels(self, tree_nwk_file: str):
        """
        Get nodes by the levels of newick tree from root
        Return dictionary with key = depth and value as a list with sequence names in that depth
        Last level is allocated for msa sequence (list of the tree)
        """
        tree = Phylo.read(self.data_dir + tree_nwk_file, "newick")
        terminals = tree.get_terminals()

        depths_dict = tree.depths(unit_branch_lengths=True)
        # depths_dict = tree.depths()
        depths_levels = max(list(depths_dict.values()))

        levels = {}
        for level in range(depths_levels + 1):
            levels[level] = []

        tree_lists = []
        for clade, depth in depths_dict.items():
            if clade in terminals:
                tree_lists.append(clade.name)
            else:
                levels[depth].append("ancestral_" + str(clade.confidence))  # fireprot format

        # Now check on which level are not seqs, the last one allocate for tree lists
        ret_levels = {}
        last_level = 0
        for level in range(depths_levels + 1):
            if len(levels[level]) != 0:
                ret_levels[level] = levels[level]
                last_level = level
        ret_levels[last_level + 1] = tree_lists
        return ret_levels

    def get_tree_depths(self, tree_nwk_file: str):
        """ Get tree depths with distance from root """
        tree = Phylo.read(self.tree_dir + tree_nwk_file, "newick")
        terminals = tree.get_terminals()

        depths_dict = tree.depths()

        sequence_dephts = {}
        for clade, depth in depths_dict.items():
            if clade in terminals:
                sequence_dephts[clade.name.replace("\"", "")] = depth  # self.max_depth
            else:
                sequence_dephts["ancestral_" + str(clade.confidence)] = depth
                clade.name = "ancestral_" + str(clade.confidence)
        return sequence_dephts

    def get_tree_branches(self, tree_nwk_file: str) -> List[List[str]]:
        """
        Get paths from the root to all lists for tree depth correlation with latent space distance calculation.
        Returns sequence names which will be match with names from get_tree_depths method.
        """
        tree = Phylo.read(self.tree_dir + tree_nwk_file, "newick")
        terminals = tree.get_terminals()

        branch_node_names = []
        for t in terminals:
            branch = []
            for node in tree.get_path(t):
                if node in terminals:
                    branch.append(node.name.replace("\"", ""))
                else:
                    branch.append("ancestral_" + str(node.confidence))
            branch_node_names.append(branch)
        return branch_node_names

    def calculate_latent_branch_depth_correlation(self, branches: List[List[str]], depths: dict, msa: dict):
        """
        Calculate correlation between branch and depth for every branch in the tree
        """
        binaries, weights, keys = self.transformer.sequence_dict_to_binary(msa)
        mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)

        correlations = []
        for branch in branches:
            latent_depth_pairs = np.zeros((len(branch), 2))
            for i, node in enumerate(branch):
                latent_depth_pairs[i][0] = np.linalg.norm(mus[keys.index(node)])  # get latent space distance to center
                latent_depth_pairs[i][1] = depths[node]  # get node depth in tree
            my_rho = np.corrcoef(latent_depth_pairs[:, 0], latent_depth_pairs[:, 1])
            correlations.append(my_rho[0, 1])
        return correlations

    def encode_levels(self, levels: dict, msa_file):
        """
        Get latent space coordinates.
        Returns same list with coords values instead of sequence names
        """
        msa = self.aligner_obj.align_fasta_to_original_msa(self.data_dir + msa_file, already_msa=True, verbose=True)
        ret_levels = []

        for level, names in levels.items():
            level_msa = {}
            for name in names:
                name = name.replace("\"", "")  # remove strange artefact for some sequences
                level_msa[name] = msa[name]

            binaries, weights, keys = self.transformer.sequence_dict_to_binary(level_msa)
            mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)
            ret_levels.append(mus)

        return ret_levels

    def encode_sequences_with_depths(self, seq_depths: dict, tree_msa_file, tree_path):
        """
        Get latent space coordinates.
        Returns latent space coordinates with depth value in 3rd dimension, loaded msa
        """
        msa = self.aligner_obj.align_tree_msa_to_msa(self.tree_dir + tree_msa_file, self.tree_dir + tree_path,
                                                     self.tree_dir)
        binaries, weights, keys = self.transformer.sequence_dict_to_binary(msa)
        mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)

        depths = np.array([])
        for name in msa.keys():
            depths = np.append(depths, seq_depths[name])

        coords_depth = np.zeros((mus.shape[0], mus.shape[1] + 1))
        coords_depth[:, :mus.shape[1]] = mus
        coords_depth[:, mus.shape[1]] = depths
        return coords_depth, msa

    def tree_pca_component_correlation(self, newick_tree: str, mu: np.ndarray, key, iter):
        """ Follow the used protocol in paper to check the correlations with 1st components """
        t = Tree(self.tree_dir + newick_tree, format=1)
        for node in t.traverse('preorder'):
            if node.is_root():
                node.add_feature('anc', [])
                node.add_feature('sumdist', 0)
                root_name = node.name
            else:
                node.add_feature('anc', node.up.anc + [node.up.name])
                node.add_feature('sumdist', node.up.sumdist + node.dist)

        reg = linear_model.LinearRegression()
        pca = PCA(n_components=2)
        leaf = [k for k in key if "ancestral" not in k]
        R2, PCC, PCA_direction = [], [], []
        root_projection_corr = []
        root_coords = np.zeros(mu.shape[0])

        branches = ["query", "OUT68545.1", "WP_012286701.1", "TFG98623.1", "PYV12062.1", "TDI51065.1", "TDJ42344.1"]
        print("PCA tree branch correlation proceeding...")
        for k in range(len(leaf)):
            print(k, flush=True, end=("," if (k + 1) % 40 != 0 else "\n"))

            leaf_name = "\"" + leaf[k] + "\""
            idx = key.index(leaf[k])
            data = pd.DataFrame(index=(t & leaf_name).anc + [leaf_name], columns=("mu1", 'mu2', 'depth'))
            data.loc[leaf_name, :] = (mu[idx, 0], mu[idx, 1], (t & leaf_name).sumdist)
            num_anc = len((t & leaf_name).anc)

            for i in range(num_anc):
                n = (t & leaf_name).anc[i]
                idx = key.index("ancestral_" + n)
                data.loc[n, :] = (mu[idx, 0], mu[idx, 1], (t & n).sumdist)
                if k == 0 and root_name == n:
                    root_coords = mu[idx]
            if iter == 0 and leaf[k] in branches:
                plt.scatter(data.loc[:, 'mu1'], data.loc[:, 'mu2'], c=data.loc[:, 'depth'],
                            cmap=plt.get_cmap('viridis'))
                plt.plot(data.loc[leaf_name, 'mu1'], data.loc[leaf_name, 'mu2'], '+r', markersize=16)

            data = np.array(data).astype(np.float64)
            res = reg.fit(data[:, 0:2], data[:, -1])
            yhat = res.predict(data[:, 0:2])

            SS_res = np.sum((data[:, -1] - yhat) ** 2)
            SS_tot = np.sum((data[:, -1] - np.mean(data[:, -1])) ** 2)
            r2 = 1 - SS_res / SS_tot
            R2.append(r2)

            pca.fit(data[:, 0:2])
            pca_coor = pca.transform(data[:, 0:2])

            if np.sum(pca.components_[0, :] * data[-1, 0:2]) < 0:
                main_coor = -pca_coor[:, 0]
                pca_vector = -pca.components_[0]
            else:
                main_coor = pca_coor[:, 0]
                pca_vector = pca.components_[0]
            PCC.append(stats.pearsonr(main_coor, data[:, 2])[0])

            # Compute PCA component pointing to origin
            unit_origin_vec = (mu[idx] / np.linalg.norm(mu[idx]))  # to point into center of latent space
            unit_pca0_comp = pca_vector / np.linalg.norm(pca_vector)
            root_vec_proc = root_coords * np.dot(mu[idx], root_coords) / np.dot(root_coords, root_coords)
            PCA_direction.append(np.dot(unit_origin_vec, unit_pca0_comp))

        if iter == 0:
            # Finish plot
            plt.xlim((-6.5, 6.5))
            plt.ylim((-6.5, 6.5))
            plt.xlabel("$Z_1$")
            plt.ylabel("$Z_2$")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(self.target_dir + "branch_evolution.png", dpi=600)
            print("  Storing tree evolution")
        print()
        return R2, PCC, PCA_direction, root_coords

    def plot_levels(self, levels: list):
        """ Plot levels into latent space """
        for level, mus in enumerate(levels):
            if level == 0:
                continue
            plt.scatter(mus[:, 0], mus[:, 1], s=0.5)

    def plot_depths(self, depths: list):
        """ Plot depth into latent space """
        plt.scatter(depths[:, 0], depths[:, 1], c=depths[:, 2], s=1.5, cmap='jet', vmin=0, vmax=self.max_depth)

    def plot_corr_histogram(self, correlations, file_name, title):
        """ Plot histogram of correlations in branches """
        fig, ax = plt.subplots()
        ax.hist(correlations, bins=25)
        ax.set_title(title)
        ax.set(xlabel=" Pearson correlation coefficient", ylabel="")
        plt.savefig(self.target_dir + file_name, dpi=400)

    def finalize_plot(self, file_name: str):
        """ Store and finalize plot """
        color_bar = plt.colorbar()
        color_bar.set_label('Sequence distance from root')
        plt.savefig(self.target_dir + file_name, dpi=600)

    def map_roots_into_latent_space(self, roots):
        """ maps root latent space coordinates """
        plt.figure()
        roots = roots.reshape(-1, 2)
        plt.plot(roots[:, 0], roots[:, 1], '+', markersize=8, color='blue')
        plt.title("Roots mapping into the latent space")
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.savefig(self.target_dir + "roots.png", dpi=600)


# Number of randomly created MSAs and then trees
n = 13


def run_sampler():
    """ Design the run setup for this package class MSASubsampler """
    cmdline = CmdHandler()
    sampler = MSASubsampler(cmdline)

    print("=" * 80)
    print("   Creating Fireprot subsampled {} MSAs".format(n))
    sampler.sample_msa(1)
    print("   MSAs created into ", sampler.target_dir)


def run_circular_sampler():
    """ Sample in the circle around center in same distance as query """
    cmdline = CmdHandler()
    sampler = MSASubsampler(cmdline)

    print("=" * 80)
    print("   Sample representatives")
    file_path = sampler.sample_representative()
    print("   MSAs created into ", file_path)


def run_tree_highlighter():
    """ Get levels of the phylo tree and highlight it in latent space """
    cmdline = CmdHandler()
    file_tree_templ = "msa_tree{}.nwk"
    file_bigmsa_templ = "bigMSA{}.fasta"

    print("=" * 80)
    print("   Mapping trees for {} MSAs".format(n))

    anc_tree_handler = AncestralTree(cmdline)
    over_tree_corr, R2_all, PCC_all, roots, directions = [], [], [], np.array([]), []
    for i in range(n):
        if not os.path.exists(os.path.join(anc_tree_handler.tree_dir, file_tree_templ.format(i))):
            continue  # These alignments do not have ancestors by Fireprot
        print("   Level parsing and plotting tree for ", file_tree_templ.format(i), " ", file_bigmsa_templ.format(i))
        # levels = anc_tree_handler.get_tree_levels(file_tree_templ.format(i))
        # levels = anc_tree_handler.encode_levels(levels, file_bigmsa_templ.format(i))
        # anc_tree_handler.plot_levels(levels)
        depths = anc_tree_handler.get_tree_depths(file_tree_templ.format(i))
        depths_coords, msa = anc_tree_handler.encode_sequences_with_depths(depths, file_bigmsa_templ.format(i),
                                                                           file_tree_templ.format(i))
        # anc_tree_handler.plot_depths(depths_coords)
        # branches correlations
        branches = anc_tree_handler.get_tree_branches(file_tree_templ.format(i))
        correlations = anc_tree_handler.calculate_latent_branch_depth_correlation(branches, depths, msa)
        r2, pcc, dire, root = anc_tree_handler.tree_pca_component_correlation(file_tree_templ.format(i),
                                                                              depths_coords[:, :2],
                                                                              list(msa.keys()), i)
        over_tree_corr.extend(correlations)
        R2_all.extend(r2)
        PCC_all.extend(pcc)
        directions.extend(dire)
        roots = np.append(root, roots)

        # store depths and coordinates to the file
        with open(os.path.join(anc_tree_handler.coord_dir, f"tree_coordinates_{i}.npy"), "wb") as np_file:
            np.save(np_file, depths_coords)

    # anc_tree_handler.finalize_plot("latent_tree.png")
    anc_tree_handler.plot_corr_histogram(over_tree_corr, "tree_depths_corr.png",
                                         "Correlation of latent origin distance and depth in the tree")
    anc_tree_handler.plot_corr_histogram(R2_all, "tree_r2.png", "R2 correlations with the origin distance")
    anc_tree_handler.plot_corr_histogram(PCC_all, "tree_pcc.png", "PCC correlations upon 1st component")
    anc_tree_handler.plot_corr_histogram(directions, "pca_origin_product.png",
                                         "Dot product of PCA 0th component and leaf origin vector")
    anc_tree_handler.map_roots_into_latent_space(roots)

    with open(anc_tree_handler.target_dir + "correlations.pkl", "wb") as file_handle:
        pickle.dump(over_tree_corr, file_handle)
    with open(anc_tree_handler.target_dir + "r2_correlations.pkl", "wb") as file_handle:
        pickle.dump(R2_all, file_handle)
    with open(anc_tree_handler.target_dir + "pcc_correlations.pkl", "wb") as file_handle:
        pickle.dump(PCC_all, file_handle)


def plot_tree_to_latent_space():
    """
    Plot tree structure into the latent space
    @WARNING please the tree newick file to highlight and
    @WARNING rename to distinguish from custom tree mapping
    """
    cmdline = CmdHandler()
    anc_tree_handler = AncestralTree(cmdline)
    plot_dir = os.path.join(anc_tree_handler.target_dir, "latent_space_trees")
    os.makedirs(plot_dir, exist_ok=True)

    try:
        with open(os.path.join(cmdline.pickles_fld, 'embeddings.pkl'), 'rb') as embed_file:
            embeddings_pkl = pickle.load(embed_file)
    except FileNotFoundError:
        print("   Please prepare MSA embedding before you run this case!!\n"
              "   run python3 runner.py run_task.py --run_package_generative_plots --json config.file")
        exit(1)

    embeddings = embeddings_pkl['mu']

    print("=" * 80)
    print("   Plotting tree into the latent space ")
    file_tree_templ = "msa_tree{}.nwk"
    file_bigmsa_templ = "bigMSA{}.fasta"

    # Prepare data one by one
    for i in range(n):
        if not os.path.exists(os.path.join(anc_tree_handler.tree_dir, file_tree_templ.format(i))):
            continue
        print("   Latent space tree ", file_tree_templ.format(i), " ", file_bigmsa_templ.format(i))
        depths = anc_tree_handler.get_tree_depths(file_tree_templ.format(i))
        depths_coords, msa = anc_tree_handler.encode_sequences_with_depths(depths, file_bigmsa_templ.format(i),
                                                                           file_tree_templ.format(i))
        branches = anc_tree_handler.get_tree_branches(file_tree_templ.format(i))

        key_to_mu = {k: d[:-1] for k, d in zip(msa.keys(), depths_coords)}
        edges = []
        # Prepare edges for branches and plot them
        for branch in branches:
            branch_edges = [(key_to_mu[branch[i - 1]], key_to_mu[branch[i]]) for i in range(1, len(branch))]
            edges.extend(branch_edges)

        # Plot tree into latent space
        plt.plot(embeddings[:, 0], embeddings[:, 1], '.', alpha=0.1, markersize=3, )
        for p1, p2 in edges:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.0)
        plt.savefig(os.path.join(plot_dir, f"tree_msa{i}.png"), dpi=600)
        print(f"  Saving to tree_msa{i}.png")
        plt.clf()
