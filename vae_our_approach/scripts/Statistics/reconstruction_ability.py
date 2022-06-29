__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/17 14:40:00"

import os
import pickle
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from parser_handler import CmdHandler
from project_enums import VaePaths
from sequence_transformer import Transformer
from analyzer import AncestorsHandler
from msa_handlers.msa_preparation import MSA
from VAE_accessor import VAEAccessor
from experiment_handler import ExperimentStatistics


class Reconstructor:
    """
    Create histogram for specific reconstruct watch statistics
    Stats:
        self_reconstruction: how well are sequences reconstructed
    """

    def __init__(self, setuper: CmdHandler):
        self.transformer = Transformer(setuper)
        self.vae = VAEAccessor(setuper, setuper.get_model_to_load())
        self.aligner_obj = AncestorsHandler(setuper)
        self.pickle = setuper.pickles_fld
        self.target_dir = setuper.high_fld + "/" + VaePaths.FREQUENCIES_STATS.value + "/" + VaePaths.RECONSTRUCTOR.value
        self.data_dir = self.target_dir + "data/"
        self.setup_output_folder()

        self.query_id = setuper.query_id

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def get_sequences_to_reconstruct(self, fasta_msa_file: str, realign: bool) -> Dict[str, str]:
        """ Get sequences which will be highlighted in the latent space """
        if realign:
            return self.aligner_obj.align_fasta_to_original_msa(fasta_msa_file, True, verbose=False)
        # else input is original msa, we need to slice it
        ret = self.transformer.slice_msa_by_pos_indexes(MSA.load_msa(fasta_msa_file))
        return ret

    def measure_reconstruct_ability(self, msa: Dict[str, str], c: List = None) -> Tuple[List[float], float]:
        """
        Get how well can VAE reconstruct sequences for N times random
        sampling around coordinates in the latent space
        """
        binaries, weights, keys = self.transformer.sequence_dict_to_binary(msa)
        z, _ = self.vae.propagate_through_VAE(binaries, weights, keys, c)
        reconstructed_msa = self.vae.decode_z_to_aa_dict(z, ref_name="ref")

        # now match identity
        query_identity = 0
        identities = []
        for key, reconstructed_sequence in zip(keys, reconstructed_msa.values()):
            source = msa[key]
            identities.append(ExperimentStatistics.sequence_identity(source, reconstructed_sequence))
            if key == self.query_id:
                query_identity = identities[-1]
        return identities, query_identity

    def plot_reconstructions(self, data: List[float], query_identity: float, title: str, file_name):
        """ Plots histogram distribution of identities of reconstructed and original sequences """
        fig, ax = plt.subplots()
        Reconstructor.created_subplot(ax, data, query_identity, title)
        plt.savefig(self.target_dir + file_name, dpi=400)

    @staticmethod
    def get_plot_data_file_names():
        """ Returns names of data files for plotting """
        return "identities.pkl", "query_identity.pkl"

    @staticmethod
    def created_subplot(ax, data: List[float], query_identity: float, title: str):
        """ Method plotting desired graph into given subplot """
        ax.hist(data, bins=100)
        ax.set_title(title)
        ax.set(xlabel="%", ylabel="Quantity")
        ax.axvline(x=query_identity, color='r', linestyle='dashed', linewidth=2)
        ax.text(query_identity + 0.2, 0, "Query({:.2f})".format(query_identity), rotation=90)
        return ax


def run_input_dataset_reconstruction():
    cmdline = CmdHandler()
    reconstructor = Reconstructor(cmdline)
    msa = reconstructor.get_sequences_to_reconstruct(cmdline.in_file, False)
    cond = MSA.parse_solubility_file_to_cat(cmdline.solubility_file)

    print("=" * 80)
    print("     Measuring input dataset reconstruction for {} model".format(cmdline.get_model_to_load()))

    identities, query_identity = reconstructor.measure_reconstruct_ability(msa, cond)
    reconstructor.plot_reconstructions(identities, query_identity, "Input dataset reconstruction",
                                       "input_training_reconstruction.png")
    print("     Saving plot into ", reconstructor.target_dir + "input_training_reconstruction.png")

    # Save plot data
    identities_file, query_identity_file = Reconstructor.get_plot_data_file_names()
    with open(reconstructor.data_dir + identities_file, 'wb') as f:
        pickle.dump(identities, f)
    with open(reconstructor.data_dir + query_identity_file, 'wb') as f:
        pickle.dump(query_identity, f)
