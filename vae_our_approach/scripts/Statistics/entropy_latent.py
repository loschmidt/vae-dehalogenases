__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2022/07/01 10:06:00"

import inspect
import os
import sys
import torch
import os.path as path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributions as D

custom_style = {'grid.color': '0.5'}
sns.set_theme(style="ticks")

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from parser_handler import CmdHandler
from sequence_transformer import Transformer
from VAE_accessor import VAEAccessor
from project_enums import SolubilitySetting
from msa_handlers.msa_preparation import MSA
from vae_models.latent_priors.entropy_prior import DistNet


class EntropyLatent:
    """
    This class implements measurement of entropy by article Meaningful latent space representation
    """

    def __init__(self, setuper: CmdHandler, optimized_entropy):
        self.vae = VAEAccessor(setuper, setuper.get_model_to_load())
        self.transformer = Transformer(setuper)
        self.cmd_line = setuper
        self.optimized_entropy = optimized_entropy
        self.distnet = self.init_distnet()

    def init_distnet(self):
        """ Check whether there is distnet already initialized """
        distnet = DistNet(self.cmd_line.dimensionality, 2000)

        model_path = self.cmd_line.VAE_model_dir + "/distnet.model"
        if path.isfile(model_path):
            distnet.load_state_dict(torch.load(model_path))
            return distnet
        embeddings, _ = self.create_latent_space()
        distnet.kmeans_initializer(embeddings)

        torch.save(distnet.state_dict(), model_path)
        return distnet

    def create_latent_space(self):
        """ Creates latent space coordinates """
        with open(self.cmd_line.pickles_fld + "/training_alignment.pkl", 'rb') as file_handle:
            train_dict = pickle.load(file_handle)
        msa_binary, weights, msa_keys = self.transformer.sequence_dict_to_binary(train_dict)
        msa_binary = torch.from_numpy(msa_binary)

        # In the case of CVAE
        solubility = self.cmd_line.get_solubility_data()

        mu, sigma = self.vae.propagate_through_VAE(msa_binary, weights, msa_keys, solubility)

        query_index = msa_keys.index(self.cmd_line.query_id)
        query_coords = mu[query_index]
        return mu, query_coords

    def calculate_entropies(self, xy_min, xy_max):
        """
        Get entropies for each point in the grid of latent space by formula
            H(X|Z) = SUM p_theta(X|Z)_i * log(p_theta(X|Z)_i)
        """
        n_points = 100
        xy_min, xy_max = xy_min - 1, xy_max + 1
        z_grid = torch.stack([m.flatten() for m in torch.meshgrid(2 * [torch.linspace(xy_min, xy_max, n_points)])]).t()

        solubility = self.cmd_line.get_solubility_data()

        if solubility is not None:
            solubility = torch.tensor([SolubilitySetting.SOL_BIN_MEDIUM.value for _ in range(z_grid.shape[0])])

        final_shape = (z_grid.shape[0], -1, len(MSA.aa) + 1)
        log_p = self.vae.vae.decoder(z_grid, c=solubility)
        probs = torch.unsqueeze(log_p, -1)
        probs = probs.view(final_shape)
        probs = torch.exp(probs)

        if self.optimized_entropy:
            s = self.distnet(z_grid).view(*z_grid.shape[:-1], 1, 1)
            probs = (1 - s) * probs + s * probs.mean()

        d = D.Categorical(probs=probs)
        entropies = d.entropy().sum(dim=-1)
        entropies = entropies.detach().numpy()

        return entropies, z_grid

    def create_plot(self, ax_p=None):
        """ Create plot """
        if ax_p is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            ax = ax_p
        zs, query = self.create_latent_space()
        ent, z_grid = self.calculate_entropies(min([min(zs[:, 0]), min(zs[:, 0])]), max([max(zs[:, 0]), max(zs[:, 0])]))
        ax = EntropyLatent.fill_plot_entropy_latent(ax, ent, z_grid, zs, query)

        if ax_p is None:
            fig.savefig(self.cmd_line.high_fld + "/entropy_net.png", dpi=600)
        return ax

    @staticmethod
    def fill_plot_entropy_latent(ax, entropies, z_grid, embeddings, query_embedding, n_points=100):
        """ Plot contour map with information entropy in the latent space """
        m = ax.contourf(z_grid[:, 0].reshape(n_points, n_points),
                        z_grid[:, 1].reshape(n_points, n_points),
                        entropies.reshape(n_points, n_points), 40, levels=50, cmap='Greys_r',
                        zorder=0)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], s=1)
        ax.scatter(query_embedding[0], query_embedding[1], s=1, color='red')
        plt.colorbar(m)
        return ax


def run_entropy(optimized_entropy=False):
    """ Design the run setup for this package """
    cmdline = CmdHandler()
    entropy_gauge = EntropyLatent(cmdline, optimized_entropy)
    entropy_gauge.create_plot()


if __name__ == '__main__':
    run_entropy()
