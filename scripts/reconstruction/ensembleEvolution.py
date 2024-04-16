__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/09/29 14:12:00"

import math
import os
import pickle

import numpy as np
import torch
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from VAE_accessor import VAEAccessor
from parser_handler import CmdHandler
from project_enums import VaePaths
from robustness import Robustness
from sequence_transformer import Transformer
from experiment_handler import ExperimentStatistics


class EnsembleEvo:
    """
    This class does not use generative power of VAE directly.
    It studies mutation suggestion over more models ancestors mapped into latent space representation.
    """

    def __init__(self, cmd_handler: CmdHandler, models_cnt: int = 5):
        self.vae = VAEAccessor(cmd_handler, model_name=cmd_handler.get_model_to_load())
        self.cmd = cmd_handler
        self.transformer = Transformer(cmd_handler)
        self.pickle = cmd_handler.pickles_fld

        self.target_dir = cmd_handler.high_fld + "/" + VaePaths.ENSEMBLE.value + "/"
        self.plot_dir = self.target_dir + "plots/"
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.models = []
        self.model_ancs = []
        self.models_cnt = models_cnt
        for fold in range(models_cnt):
            model_name = self.cmd.model_name + f"_fold_{fold}.model"
            self.models.append(VAEAccessor(cmd_handler, model_name=model_name))

        self.models_latent_embeddings, self.query_model_embeddings = self.get_models_training_embeddings(cmd_handler)

        self.active_model = None
        self.set_active_model(0)

        with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
            ref_dict = pickle.load(file_handle)
        self.query = ref_dict

    def set_active_model(self, model_i: int):
        """ Inference is done for chosen model """
        if model_i > len(self.models):
            print(f"  WARNING YOU are setting model {model_i}/{len(self.models) - 1}")
            model_i = 0
        self.active_model = self.models[model_i]

    def get_models_training_embeddings(self, cmd_line):
        """
        Create embeddings of training sequences for all models
        Return pair of latent space embeddings and query embeddings for all models
        """
        with open(cmd_line.pickles_fld + "/training_alignment.pkl", 'rb') as file_handle:
            train_dict = pickle.load(file_handle)
        msa_binary, weights, msa_keys = self.transformer.sequence_dict_to_binary(train_dict)
        msa_binary = torch.from_numpy(msa_binary)
        # In the case of CVAE
        solubility = cmd_line.get_solubility_data()

        models_latent, models_query = [], []
        for model in self.models:
            mu, sigma = model.propagate_through_VAE(msa_binary, weights, msa_keys, solubility)
            query_index = msa_keys.index(cmd_line.query_id)
            query_coords = mu[query_index]
            models_latent.append(mu)
            models_query.append(query_coords)
        return models_latent, models_query

    def mutants_positions(self, seqs):
        """ Get mutants position in the latent space from dictionary """
        binary, weights, keys = self.transformer.sequence_dict_to_binary(seqs)
        data, _ = self.active_model.propagate_through_VAE(binary, weights, keys)
        return data

    def get_straight_ancestors(self, model_i, cnt_of_anc=100):
        """
        Get straight ancestors of i-th model.
        Return dictionary of ancestors
        """
        self.set_active_model(model_i)

        ref_pos = self.query_model_embeddings[model_i]
        coor_tuple = tuple(ref_pos)
        latent_dim = range(len(ref_pos))
        step_list = [0.0 for _ in latent_dim]
        for i in latent_dim:
            step_list[i] = (coor_tuple[i] / cnt_of_anc)
        to_highlight = [coor_tuple]
        i = 0
        while i <= cnt_of_anc:
            cur_coor = [0.0 for _ in latent_dim]
            for s_i in latent_dim:
                cur_coor[s_i] = coor_tuple[s_i] - (step_list[s_i] * i)
            to_highlight.append(tuple(cur_coor))
            i += 1
        ancestors = self.active_model.decode_z_to_aa_dict(to_highlight, list(self.query.keys())[0], 0)
        return ancestors

    def get_ancestors_embeddings(self, model_i, ancestors):
        """ Get embeddings of generated ancestors for i-th model """
        self.set_active_model(model_i)
        embeddings = self.mutants_positions(ancestors)
        return embeddings

    def measure_models_robustness(self):
        """ Get ancestors of all models and see deviation from trajectory. """
        cross_embeddings = [[] for _ in range(self.models_cnt)]
        model_embedding_deviations = list()
        for model_i in range(self.models_cnt):
            model_ancestors = self.get_straight_ancestors(model_i)
            model_query_emb = self.query_model_embeddings[model_i]
            self.model_ancs.append(model_ancestors)
            # get embeddings of its ancestors in others models
            model_i_deviation = 0
            for cross_model_i in range(self.models_cnt):
                model_embeddings = self.get_ancestors_embeddings(cross_model_i, model_ancestors)
                cross_embeddings[cross_model_i].append(model_embeddings)
                # get their deviations
                average_dev, _ = Robustness.compute_deviation([model_query_emb], model_embeddings)
                model_i_deviation += average_dev
            model_embedding_deviations.append(model_i_deviation)
        return cross_embeddings, model_embedding_deviations

    def plot_one_model_embeddings(self, model_i, models_embeddings, ax):
        """ Create plot for just one subplot and store it separately """
        model_embeddings = models_embeddings[model_i]  # just for given models
        mu = self.models_latent_embeddings[model_i]
        query = self.query_model_embeddings[model_i]

        fig_lat, ax_lat = plt.subplots(1, 1)
        colors = cm.rainbow(np.linspace(0, 1, self.models_cnt))
        plot_i = 0
        for ax_cur in [ax, ax_lat]:
            ax_cur.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3)

            # Highlight different models embeddings in the latent space
            for i, (emb, c) in enumerate(zip(model_embeddings, colors)):
                ax_cur.plot(emb[:, 0], emb[:, 1], '.', color=c, alpha=1, markersize=3, label=f'Ancs model {i}')
                ax_cur.plot(emb[0, 0], emb[0, 1], '*', color='black', markersize=5)
            if plot_i == 1:
                ax_cur.legend(loc="upper right")

            ax_cur.plot(query[0], query[1], 'x', color='black', markersize=5)
            ax_cur.title.set_text(f'Model {model_i}')
            ax_cur.set_xlabel("$Z_1$")
            ax_cur.set_ylabel("$Z_2$")
            plot_i += 1
        fig_lat.savefig(self.plot_dir + f"/cross{model_i}.png", dpi=900)

    def plot_cross_embeddings(self, models_embeddings):
        """ Create plots with embeddings mapped to individual models """
        row_cnt = int(math.sqrt(self.models_cnt))
        col_cnt = math.ceil((math.sqrt(self.models_cnt)))
        fig, ax = plt.subplots(row_cnt, col_cnt)

        for model_i in range(self.models_cnt):
            print(model_i // col_cnt, model_i % col_cnt)
            self.plot_one_model_embeddings(model_i, models_embeddings, ax[model_i // col_cnt, model_i % col_cnt])
        # handles, labels = ax.get_legend_handles_labels()
        # [[c.get_legend().remove() for c in r] for r in ax]  # remove legends from subplots
        # fig.legend(handles, labels, loc='upper center')
        fig.savefig(self.plot_dir + "/cross_all.png")

    def ensemble_evolution(self):
        """
        Strategy to run ensemble evolution optimization, over more models
        """
        models_embeddings, deviations = self.measure_models_robustness()
        print(deviations)
        self.plot_cross_embeddings(models_embeddings)


def run_ensemble_evolution():
    """ Run class as task target """
    cmdline = CmdHandler()
    ensembler = EnsembleEvo(cmdline, cmdline.K)
    ensembler.ensemble_evolution()


if __name__ == "__main__":
    run_ensemble_evolution()
