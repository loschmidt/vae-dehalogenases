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

    def fitness_landscape(self):
        """ Prepare fitness landscape using gaussian processes """
        mutants, y = self.curator.get_data()
        binary = self.vae.binaryConv.prepare_aligned_msa_for_Vae(mutants)
        X, _ = self.vae.prepareMusSigmas(binary)

        # Input space
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max())  # p
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max())  # q
        x = (np.array([x1, x2])).T

        # Prepare gaussian process class and fit data
        kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

        gp.fit(X, y)

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

if __name__ == "__main__":
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    cmd_line = CommandHandler()

    evo = EvolutionSearch(tar_dir, cmd_line)
