__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/04/16 00:10:00"

from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from EVO.create_library import CommandHandler, Curator
from pipeline import StructChecker
from benchmark import Benchmarker as Vae_encoder
from analyzer import AncestorsHandler, VAEHandler
from mutagenesis import MutagenesisGenerator as FastaStore


class EvolutionSearch:
    """
        Implementation of Covariance Matrix Adaptation Evolution Strategy
        including preprocessing of data and preparation of heat map.

        run with additional options source_txt to specify path to library description file
    """

    def __init__(self, setuper, cmd_handler):
        self.curator = Curator(cmd_handler)
        self.setuper = setuper
        self.vae = Vae_encoder(None, None, self.setuper, generate_negative=False)
        self.out_dir = setuper.high_fld + '/'
        self.handler = VAEHandler(setuper)

    def fit_landscape(self):
        """
        Prepare fitness landscape using gaussian processes
        return: gp_regressor, mutants mapped to latent space
        """
        mutants, y = self.curator.get_data()
        # Store them in the file
        fasta = FastaStore(self.setuper)
        fasta.anc_seqs = list(mutants.values())
        file_name ='thermo_mutant_library.fasta'
        fasta.store_ancestors_in_fasta(names=list(mutants.keys()), file_name=file_name)
        print(" Evolutionary search : Mutant library saved to", self.out_dir + file_name)

        self.setuper.highlight_files = self.out_dir + file_name
        mutant_aligned = AncestorsHandler(setuper=self.setuper, seq_to_align=mutants).align_to_ref()
        binary, _, _ = self.vae.binaryConv.prepare_aligned_msa_for_Vae(mutant_aligned)
        X, _ = self.vae.prepareMusSigmas(binary)

        # Input space
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max())  # p
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max())  # q
        x = (np.array([x1, x2])).T

        # Prepare gaussian process class and fit data
        kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
        
        gp.fit(X, list(y.values()))

        x1x2 = np.array(list(product(x1, x2)))
        y_pred, MSE = gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)
        Zp = np.reshape(y_pred, (50, 50))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.pcolormesh(X0p, X1p, Zp)

        save_path = self.out_dir + 'fitness_landscape_{}.png'.format(self.setuper.model_name)
        print(" CMA_EV message : saving landscape fitness to", save_path)
        plt.savefig(save_path)

        return gp, X

    def init_plot_fitness_LS(self, gp):
        """ Plot latent space with fitness Tm distribution """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Plot Tm background
        area = np.linspace(-6, 6)  # p
        gp_points = np.array(list(product(area, area)))
        gp_bgr = gp_points[:, 0].reshape(500, 500)
        y_pred, MSE = gp.predict(gp_points, return_std=True)
        zp = np.reshape(y_pred, (500, 500))
        ax.pcolormesh(gp_bgr, gp_bgr, zp)

        # Highlight latent space
        mus, _, _ = self.handler.latent_space(check_exists=True)
        ax.plot(mus[:, 0], mus[:, 1], '.', alpha=0.1, markersize=3, label='sequences')
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xlabel("$Z_1$")
        ax.set_ylabel("$Z_2$")
        return ax

    def save_plot(self, name):
        save_path = self.out_dir + '{}.png'.format(name)
        print(" CMA_EV message : saving plot to", save_path)
        plt.savefig(save_path)

if __name__ == "__main__":
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    cmd_line = CommandHandler()

    evo = EvolutionSearch(tar_dir, cmd_line)
    evo.fit_landscape()
