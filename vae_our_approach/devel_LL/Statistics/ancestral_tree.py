__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import pickle
import sys
import inspect

import numpy as np
from Bio import Phylo
import matplotlib.pyplot as plt

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
        self.target_dir = setuper.high_fld + "/" + VaePaths.STATISTICS_DIR.value + "/"
        self.query_id = setuper.query_id
        self.setup_output_folder()

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)

    def sample_msa(self, n: int, seq_cnt: int = 100):
        """ Sample MSA from training input space """

        def fasta_record(file, key, seq):
            n = 80
            file.write(">" + key + "\n")
            for i in range(0, len(seq), n):
                file.write(seq[i:i + n] + "\n")

        with open(self.pickle + "/training_alignment.pkl", "rb") as file_handle:
            msa = pickle.load(file_handle)
        msa_keys = np.array(list(msa.keys()))
        query = msa[self.query_id]
        file_templ = self.target_dir + "fireprot_msa{}.fasta"

        for msa_i in range(n):
            file_name = file_templ.format(msa_i)
            print("   ", file_name)

            with open(file_name, 'w') as file_handle:
                fasta_record(file_handle, self.query_id, "".join(query))  # Store query
                selected_seqs = msa_keys[np.random.randint(1, msa_keys.shape[0], seq_cnt, dtype=np.int)]

                for key in selected_seqs:
                    fasta_record(file_handle, key, "".join(msa[key]))


class AncestralTree:
    """
    This class highlights the individual levels of Ancestral tree from Fireprot in the latent space
    """

    def __init__(self, setuper: CmdHandler):
        self.model_name = setuper.get_model_to_load()
        self.vae = VAEAccessor(setuper, self.model_name)
        self.aligner_obj = AncestorsHandler(setuper)
        self.transformer = Transformer(setuper)
        self.target_dir = setuper.high_fld + "/" + VaePaths.TREE_EVALUATION_DIR.value + "/"
        self.setup_output_folder()

        self.max_depth = 6

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)

    def get_tree_levels(self, tree_nwk_file: str):
        """
        Get nodes by the levels of newick tree from root
        Return dictionary with key = depth and value as a list with sequence names in that depth
        Last level is allocated for msa sequence (list of the tree)
        """
        tree = Phylo.read(self.target_dir + tree_nwk_file, "newick")
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
        tree = Phylo.read(self.target_dir + tree_nwk_file, "newick")
        terminals = tree.get_terminals()

        depths_dict = tree.depths()

        sequence_dephts = {}
        for clade, depth in depths_dict.items():
            if clade in terminals:
                sequence_dephts[clade.name.replace("\"", "")] = self.max_depth
            else:
                sequence_dephts["ancestral_" + str(clade.confidence)] = depth
        return sequence_dephts

    def encode_levels(self, levels: dict, msa_file):
        """
        Get latent space coordinates.
        Returns same list with coords values instead of sequence names
        """
        print(self.target_dir + msa_file)
        msa = self.aligner_obj.align_fasta_to_original_msa(self.target_dir + msa_file, verbose=True)
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

    def encode_sequences_with_depths(self, seq_depths: dict, msa_file):
        """
        Get latent space coordinates.
        Returns latent space coordinates with depth value in 3rd dimension
        """
        print(self.target_dir + msa_file)
        msa = self.aligner_obj.align_fasta_to_original_msa(self.target_dir + msa_file, verbose=True)
        binaries, weights, keys = self.transformer.sequence_dict_to_binary(msa)
        mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)

        depths = np.array([])
        for name in msa.keys():
            depths = np.append(depths, seq_depths[name])

        coords_depth = np.zeros((mus.shape[0], 3))
        coords_depth[:, :2] = mus
        coords_depth[:, 2] = depths
        return coords_depth

    def plot_levels(self, levels: list):
        """ Plot levels into latent space """
        depth = len(levels)
        for level, mus in enumerate(levels):
            if level == 0:
                continue
            plt.scatter(mus[:, 0], mus[:, 1], s=0.5)#, c=depth - level, cmap='viridis', vmin=0, vmax=150)

    def plot_depths(self, depths: list):
        """ Plot depth into latent space """
        plt.scatter(depths[:, 0], depths[:, 1], c=depths[:, 2], s=1.5, cmap='jet', vmin=0, vmax=self.max_depth)

    def finalize_plot(self, file_name: str):
        """ Store and finalize plot """
        color_bar = plt.colorbar()
        color_bar.set_label('Sequence distance from root')
        plt.savefig(self.target_dir + file_name, dpi=600)


# Number of randomly created MSAs and then trees
n = 2


def run_sampler():
    """ Design the run setup for this package class MSASubsampler """
    cmdline = CmdHandler()
    sampler = MSASubsampler(cmdline)

    print("=" * 80)
    print("   Creating Fireprot subsampled {} MSAs".format(n))
    sampler.sample_msa(n)


def run_tree_highlighter():
    """ Get levels of the phylo tree and highlight it in latent space """
    cmdline = CmdHandler()
    file_tree_templ = "msa_tree{}.nwk"
    file_bigmsa_templ = "bigMSA{}.fasta"

    print("=" * 80)
    print("   Mapping trees for {} MSAs".format(n))

    anc_tree_handler = AncestralTree(cmdline)
    for i in range(n):
        print("   Level parsing and plotting tree for ", file_tree_templ.format(i), " ", file_bigmsa_templ.format(i))
        # levels = anc_tree_handler.get_tree_levels(file_tree_templ.format(i))
        # levels = anc_tree_handler.encode_levels(levels, file_bigmsa_templ.format(i))
        # anc_tree_handler.plot_levels(levels)
        depths = anc_tree_handler.get_tree_depths(file_tree_templ.format(i))
        depths_coords = anc_tree_handler.encode_sequences_with_depths(depths, file_bigmsa_templ.format(i))
        anc_tree_handler.plot_depths(depths_coords)
    anc_tree_handler.finalize_plot("latent_tree.png")
