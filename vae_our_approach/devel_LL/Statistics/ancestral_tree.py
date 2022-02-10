__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import pickle
import sys
import inspect

import numpy as np
from Bio import Phylo

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

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
        self.target_dir = setuper.high_fld + "/" + VaePaths.STATISTICS_DIR.value + "/"
        self.setup_output_folder()

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

        depths_dict = tree.depths()
        depths_levels = max(list(depths_dict.values()))

        levels = {}
        for level in range(depths_levels + 1):
            levels[level] = []

        tree_lists = []
        for clade, depth in depths_dict.items():
            if clade in terminals:
                tree_lists.append(clade.name)
            levels[depth].append(clade.name)

        # Now check on which level are not seqs, the last one allocate for tree lists
        ret_levels = {}
        last_level = 0
        for level in range(depths_levels + 1):
            if len(levels[level]) != 0:
                ret_levels[level] = levels[level]
                last_level = level
        ret_levels[last_level + 1] = tree_lists
        return ret_levels


# Number of randomly created MSAs and then trees
n = 3


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
    file_name_templ = "msa_tree{}.nwk"

    print("=" * 80)
    print("   Mapping trees for {} MSAs".format(n))

    anc_tree_handler = AncestralTree(cmdline)
    for i in range(n):
        print("   Level parsing and plotting tree for ", file_name_templ.format(i))
        levels = anc_tree_handler.get_tree_levels(file_name_templ.format(i))
        print(levels)
