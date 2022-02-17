__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import sys
import inspect

import numpy as np
import matplotlib.pyplot as plt

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from sequence_transformer import Transformer
from msa_preparation import MSA
from parser_handler import CmdHandler
from project_enums import VaePaths
from VAE_accessor import VAEAccessor


class LatentSpaceMapper:
    """
    The purpose of this class is to highlight sequences in the latent space with value labels
    The class expect having these sequences (or its names having it in training dataset) and
    values in the file in the following format:
        ProteinName;ProteinIdentificator;InSilicoVal;WetLabVal
    example:
        DfxA;WP_071046332.1;73.4;24.2
    """

    def __init__(self, setuper: CmdHandler):
        self.model_name = setuper.get_model_to_load()
        self.vae = VAEAccessor(setuper, self.model_name)
        self.transformer = Transformer(setuper)
        self.in_file = setuper.in_file
        self.target_dir = setuper.high_fld + "/" + VaePaths.MAPPING_DIR.value + "/"
        self.setup_output_folder()

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)

    def load_training_sequence_labels(self, file_name: str):
        """ Loads training sequence binaries and label them from file """
        sequences = []

        with open(self.target_dir + file_name, "r") as lines:
            for line in lines:
                prot_id, seq_id, soluprot, experimental = line.split(";")
                experimental = float(experimental.split("\n")[0])
                sequences.append([seq_id, experimental])
        return sequences

    def map_sequences_with_labels_to_space(self, seq_labels: list):
        """ Assign labels to the sequences from the input set """
        whole_msa = MSA.load_msa(self.in_file)

        labels = []
        selected_sequences_dict = {}
        for seq_id, label in seq_labels:
            try:
                selected_sequences_dict[seq_id] = whole_msa[seq_id]
                labels.append(label)
            except KeyError:
                print("     Sequence name {} not found in input MSA {}".format(seq_id, self.in_file))

        sliced_selected = self.transformer.slice_msa_by_pos_indexes(selected_sequences_dict)
        binaries, weights, keys = self.transformer.sequence_dict_to_binary(sliced_selected)
        mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)
        return mus, labels

    @staticmethod
    def plot_labels(z: np.ndarray, labels: list):
        """ Plot depth into latent space """
        plt.scatter(z[:, 0], z[:, 1], c=labels, s=1.5, cmap='jet', vmin=0, vmax=max(labels))

    def finalize_plot(self, file_name: str, axis_label: str):
        """ Store and finalize plot """
        color_bar = plt.colorbar()
        color_bar.set_label(axis_label)
        plt.savefig(self.target_dir + file_name, dpi=600)


def run_latent_mapper():
    """ Map label to the sequences in the latent space """
    cmdline = CmdHandler()
    label_file = "hts_project.txt"
    plot_file = "mapped_hts_tm.png"

    print("=" * 80)
    print("   Mapping sequence with labels to latent space")

    latent_mapper = LatentSpaceMapper(cmdline)
    labels_with_sequences = latent_mapper.load_training_sequence_labels(label_file)
    z, labels = latent_mapper.map_sequences_with_labels_to_space(labels_with_sequences)
    LatentSpaceMapper.plot_labels(z, labels)
    latent_mapper.finalize_plot(plot_file, "Tm")
    print("Storing mapped plot in ", latent_mapper.target_dir + plot_file)
