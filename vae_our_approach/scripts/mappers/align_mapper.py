__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2022/05/04 14:12:00"

import os
import pickle
import sys
import inspect

import matplotlib.pyplot as plt

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler
from sequence_transformer import Transformer
from VAE_accessor import VAEAccessor
from analyzer import AncestorsHandler


def align_bakovas_sequences(cmd_line: CmdHandler):
    """ Embed Babkova's sequences into the latent space """
    transformer = Transformer(cmd_line)
    vae = VAEAccessor(cmd_line, cmd_line.get_model_to_load())
    aligner = AncestorsHandler(cmd_line)

    with open(cmd_line.pickles_fld + "/embeddings.pkl", 'rb') as file_handle:
        emb = pickle.load(file_handle)

    # In the case of CVAE
    solubility = cmd_line.get_solubility_data()

    mu, msa_keys = emb['mu'], emb['keys']

    aligning_id = 'P59337_S15'

    anc_msa = MSA.load_msa(cmd_line.highlight_files)
    ancestor_aligned = aligner.align_to_key(anc_msa, aligning_id)  # Ancestors aligned to entire MSA
    ancestor_msa, ancestor_weight, ancestor_keys = transformer.sequence_dict_to_binary(ancestor_aligned)
    ancestor_mus, ancestor_sigma = vae.propagate_through_VAE(ancestor_msa, ancestor_weight, ancestor_keys)
    templ_emb = ancestor_mus[ancestor_keys.index('template')]

    # Store coordinates of Babkovas sequences in the file
    anc_embeddings = os.path.join(cmd_line.pickles_fld, "babkova_embeddings.pkl")
    with open(anc_embeddings, "wb") as anc_file:
        pickle.dump({"mus": ancestor_mus, "keys": ancestor_keys}, anc_file)

    fig_lat, ax_lat = plt.subplots(1, 1)

    # highlight template sequence
    plt.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, )
    plt.plot(templ_emb[0], templ_emb[1], '.', color='red')
    # Highlight Babkovas ancestors
    for ancestor_idx, anc_mu in enumerate(ancestor_mus):
        plt.plot(anc_mu[0], anc_mu[1], '.', color='black', alpha=1, markersize=3,
                 label=ancestor_keys[ancestor_idx] + '({})'.format(ancestor_idx+1))
        plt.annotate(str(ancestor_idx+1), (anc_mu[0], anc_mu[1]))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    fig_lat.savefig(cmd_line.high_fld + "/babkova_mapping.png", dpi=600)
    print(f"Plot stored in {cmd_line.high_fld}/babkova_mapping.png and the raw data are {cmd_line.pickles_fld}/babkova_embeddings.pkl")


def run_align_babkova():
    """ Map babkova sequences label to the sequences in the latent space """
    cmdline = CmdHandler()
    align_bakovas_sequences(cmdline)
