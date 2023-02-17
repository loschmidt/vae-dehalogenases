__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/10 14:12:00"

import os
import pickle
import sys
import inspect
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from sequence_transformer import Transformer
from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler
from project_enums import VaePaths
from VAE_accessor import VAEAccessor
from analyzer import AncestorsHandler


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
        self.aligner_obj = AncestorsHandler(setuper)
        self.in_file = setuper.in_file
        self.pickle = setuper.pickles_fld
        self.input_dir = VaePaths.STATS_MAPPING_SOURCE.value
        self.target_dir = setuper.high_fld + "/" + VaePaths.MAPPING_DIR.value + "/"
        self.cmd_line = setuper
        self.setup_output_folder()

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)

    def load_training_sequence_labels(self, file_name: str):
        """ Loads training sequence binaries and label them from file """
        sequences = []

        with open(self.input_dir + file_name, "r") as lines:
            for line in lines:
                prot_id, seq_id, soluprot, experimental = line.split(";")
                experimental = float(experimental.split("\n")[0])
                sequences.append([seq_id, experimental, float(soluprot)])
        return sequences

    def map_sequences_with_labels_to_space(self, seq_labels: list, hts_project_fasta: str, align_to: int):
        """ Assign labels to the sequences from the input set """
        whole_msa = MSA.load_msa(self.in_file)
        # self.aligner_obj.align_to_closest_msa(self.input_dir + hts_project_fasta, "solubility")
        if align_to == 1:
            hts_msa = self.aligner_obj.align_fasta_to_original_msa(self.input_dir + hts_project_fasta,
                                                                   already_msa=False,
                                                                   verbose=True)
        else:
            seq_to_align = MSA.load_msa(self.input_dir + hts_project_fasta)
            hts_msa = self.aligner_obj.align_to_ref(seq_to_align)
        labels, soluprot_labels = [], []
        selected_sequences_dict = {}
        slicing_selected_sequences_dict = {}
        for seq_id, label, soluprot_label in seq_labels:
            try:
                slicing_selected_sequences_dict[seq_id] = whole_msa[seq_id]
                labels.append(label)
                soluprot_labels.append(soluprot_label)
            except KeyError:
                # not found in Input MSA, look for it in hts sequences
                try:
                    selected_sequences_dict[seq_id] = hts_msa[seq_id]
                    labels.append(label)
                    soluprot_labels.append(soluprot_label)
                except KeyError:
                    print("    Key {} not found in hts project".format(seq_id))

        sliced_selected = self.transformer.slice_msa_by_pos_indexes(slicing_selected_sequences_dict)
        # add the already sliced sequences
        for k, v in selected_sequences_dict.items():
            sliced_selected[k] = v

        binaries, weights, keys = self.transformer.sequence_dict_to_binary(sliced_selected)

        mus, _ = self.vae.propagate_through_VAE(binaries, weights, keys)
        return mus, labels, soluprot_labels

    def plot_labels(self, z: np.ndarray, labels: list, latent_space: bool):
        """ Plot depth into latent space """
        if latent_space:
            with open(self.cmd_line.pickles_fld + "/training_alignment.pkl", 'rb') as file_handle:
                train_dict = pickle.load(file_handle)
            msa_binary, weights, msa_keys = self.transformer.sequence_dict_to_binary(train_dict)
            msa_binary = torch.from_numpy(msa_binary)
            # In the case of CVAE
            solubility = self.cmd_line.get_solubility_data()
            # binaries = self.transformer.msa_binary
            # binaries, weights = self.transformer.shape_binary_for_vae(binaries)
            mus, _ = self.vae.propagate_through_VAE(msa_binary, weights, msa_keys)
            plt.plot(mus[:, 0], mus[:, 1], '.', alpha=0.3, markersize=3, color='blue')
            query_index = msa_keys.index(self.cmd_line.query_id)
            query_coords = mus[query_index]
            plt.plot(query_coords[0], query_coords[1], '+', alpha=1, markersize=5, color='red')

        plt.scatter(z[:, 0], z[:, 1], marker=mpath.Path.unit_regular_star(6),
                    c=labels, s=10, cmap='spring', vmin=min(labels), vmax=max(labels))

    def plot_focus(self, z: np.ndarray):
        """ Plot heatmap of the mapped sequence """
        plt.xlim([min(z[:, 0]) - 0.5, max(z[:, 0]) + 0.5])
        plt.ylim([min(z[:, 1]) - 0.5, max(z[:, 1]) + 0.5])

    def finalize_plot(self, file_name: str, axis_label: str):
        """ Store and finalize plot """
        color_bar = plt.colorbar()
        color_bar.set_label(axis_label)
        plt.savefig(self.target_dir + file_name, dpi=600)
        plt.clf()


def run_latent_mapper():
    """ Map label to the sequences in the latent space """
    cmdline = CmdHandler()
    label_file = "hts_project.txt"
    hts_fasta = "hts_project.fasta"
    plot_file = "mapped_hts_solubility.png"
    plot_soluprot_file = "mapped_soluprot_vals.png"

    print("=" * 80)
    print("   Mapping sequence with labels to latent space")

    latent_mapper = LatentSpaceMapper(cmdline)
    labels_with_sequences = latent_mapper.load_training_sequence_labels(label_file)
    z, labels, soluprot_labels = latent_mapper.map_sequences_with_labels_to_space(labels_with_sequences,
                                                                                  hts_fasta,
                                                                                  align_to=2)
    latent_mapper.plot_labels(z, labels, True)
    latent_mapper.finalize_plot(plot_file, "Solubility")
    print("  Storing mapped plot in ", latent_mapper.target_dir + plot_file)
    latent_mapper.plot_labels(z, soluprot_labels, True)
    latent_mapper.finalize_plot(plot_soluprot_file, "Solubility")
    print("  Storing mapped soluprot plot in ", latent_mapper.target_dir + plot_soluprot_file)


def run_latent_dhaa115_mapper():
    """ Map DhaA115 mutant to the latent space """
    cmdline = CmdHandler()
    label_file = "dhaa115_mut_values.txt"
    hts_fasta = "dhaa115_mut.fasta"
    plot_file = "mapped_dhaa115_dTm.png"
    plot_soluprot_file = "mapped_dhaa115_dTm_vals.png"

    print("=" * 80)
    print("   Mapping DhaA115 sequence variants with labels to latent space")

    latent_mapper = LatentSpaceMapper(cmdline)
    labels_with_sequences = latent_mapper.load_training_sequence_labels(label_file)
    z, labels, dTm_labels = latent_mapper.map_sequences_with_labels_to_space(labels_with_sequences, hts_fasta,
                                                                             align_to=2)
    # Store coordinates into specific files
    embedding_map_file = os.path.join(latent_mapper.target_dir, "tm_embeddings.pkl")
    with open(embedding_map_file, "wb") as file:
        pickle.dump({"z": z, "labels": labels, "dTm": dTm_labels}, file)

    # Corrected Tm value from wild-type not DhaA115 variant dTm DhaA115 > 23 degree
    labels = list(map(lambda x: x + 23, labels))
    dTm_labels = list(map(lambda x: x + 23, dTm_labels))

    latent_mapper.plot_labels(z, labels, True)
    latent_mapper.finalize_plot(plot_file, "dTm")
    print("  Storing mapped plot in ", latent_mapper.target_dir + plot_file)
    latent_mapper.plot_labels(z, dTm_labels, True)
    latent_mapper.finalize_plot(plot_soluprot_file, "dTm.1")
    print("  Storing mapped dTm.1 plot in ", latent_mapper.target_dir + plot_soluprot_file)

    # focused variants
    latent_mapper.plot_labels(z, labels, True)
    latent_mapper.plot_focus(z)
    latent_mapper.finalize_plot("focus_" + plot_file, "dTm")
    print("  Storing mapped plot in ", latent_mapper.target_dir + "focus_" + plot_file)
    latent_mapper.plot_labels(z, dTm_labels, True)
    latent_mapper.plot_focus(z)
    latent_mapper.finalize_plot("focus_" + plot_soluprot_file, "dTm.1")
    print("  Storing mapped dTm.1 plot in ", latent_mapper.target_dir + "focus_" + plot_soluprot_file)
