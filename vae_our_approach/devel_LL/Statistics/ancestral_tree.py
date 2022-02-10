__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import pickle
import sys
import inspect

import numpy as np

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from parser_handler import CmdHandler
from project_enums import VaePaths


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


def run_sampler():
    """ Design the run setup for this package class MSASubsampler """
    cmdline = CmdHandler()
    sampler = MSASubsampler(cmdline)
    n = 3

    print("=" * 80)
    print("   Creating Fireprot subsampled {} MSAs".format(n))
    sampler.sample_msa(n)
