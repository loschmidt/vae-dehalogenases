__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/04/16 00:10:00"

from itertools import product

import pickle
import os
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy.stats import multivariate_normal as norm

from EVO.create_library import CommandHandler, Curator
from pipeline import StructChecker
from benchmark import Benchmarker as Vae_encoder
from analyzer import AncestorsHandler, VAEHandler
from mutagenesis import MutagenesisGenerator as FastaStore

from cma_ev import update_mean, update_pc, update_Cov, update_ps, path_length_control


def highlight_coord(ax, mus, color='r'):
    ax.plot(mus[:, 0], mus[:, 1], '.', alpha=1, markersize=3, color=color)
    return ax


class EvolutionSearch:
    """
        Implementation of Covariance Matrix Adaptation Evolution Strategy
        including preprocessing of data and preparation of heat map.

        run with additional options source_txt to specify path to library description file
    """

    def __init__(self, setuper, cmd_handler):
        self.curator = Curator(cmd_handler)
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.vae = Vae_encoder(None, None, self.setuper, generate_negative=False)
        self.out_dir = setuper.high_fld + '/'
        self.handler = VAEHandler(setuper)
        self.gp = None

        with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
            self.query = pickle.load(file_handle)
            self.query_seq = list(self.query.values())
            self.seq_len = len(self.query_seq)
        with open(self.pickle + "/elbo_all.pkl", 'rb') as file_handle:
            self.mean_elbo = np.mean(pickle.load(file_handle))

    def fit_landscape(self):
        """
        Prepare fitness landscape using gaussian processes
        return: gp_regressor, mutants mapped to latent space
        """
        mutants, y = self.curator.get_data()
        # Store them in the file
        fasta = FastaStore(self.setuper)
        fasta.anc_seqs = list(mutants.values())
        file_name = 'thermo_mutant_library.fasta'
        fasta.store_ancestors_in_fasta(names=list(mutants.keys()), file_name=file_name)
        print(" Evolutionary search : Mutant library saved to", self.out_dir + file_name)

        self.setuper.highlight_files = self.out_dir + file_name
        mutant_aligned = AncestorsHandler(setuper=self.setuper, seq_to_align=mutants).align_to_ref()
        X = self.encode(mutant_aligned)

        # Input space
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max())  # p
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max())  # q

        # Prepare gaussian process class and fit data
        kernel = C(1.0, (1e-3, 1e3)) * RBF(5.3, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
        gp.fit(X, list(y.values()))
        print("Evo Search message : Gaussian processes likelihood {:.4f}".format(gp.score(X, list(y.values()))))

        x1x2 = np.array(list(product(x1, x2)))
        tm_pred, MSE = gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)
        Zp = np.reshape(tm_pred, (50, 50))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.pcolormesh(X0p, X1p, Zp)

        save_path = self.out_dir + 'fitness_landscape_{}.png'.format(self.setuper.model_name)
        print("Evo Search message : saving landscape fitness to", save_path)
        plt.savefig(save_path)

        self.gp = gp
        return X

    def init_plot_fitness_LS(self, plot_landscape=False):
        """ Plot latent space with fitness Tm distribution """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Plot Tm background
        area = np.linspace(-7, 7, 200)  # p
        gp_points = np.array(list(product(area, area)))
        gp_bgr_x, gp_bgr_y = gp_points[:, 0].reshape(200, 200), gp_points[:, 1].reshape(200, 200)
        y_pred, MSE = self.gp.predict(gp_points, return_std=True)

        zp = np.reshape(y_pred, (200, 200))
        if plot_landscape:
            ax.pcolormesh(gp_bgr_x, gp_bgr_y, zp)

        # Highlight latent space
        mus, _, _ = self.handler.latent_space(check_exists=True)
        ax.plot(mus[:, 0], mus[:, 1], '.', alpha=0.1, markersize=3, label='sequences')
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_xlabel("$Z_1$")
        ax.set_ylabel("$Z_2$")
        return ax

    def save_plot(self, name):
        save_path = self.out_dir + '{}.png'.format(name)
        print("Evo Search message : saving plot to", save_path)
        plt.savefig(save_path)

    def encode(self, seqs):
        """ Encode dictionary to the latent space """
        binary, _, _ = self.vae.binaryConv.prepare_aligned_msa_for_Vae(seqs)
        X, _ = self.vae.prepareMusSigmas(binary)
        return X

    def decode(self, coord):
        """ Decode sequence from the latent space, also return log likelihood """
        seq_dict = self.handler.decode_sequences_VAE(coord, "coded")
        seq = list(seq_dict["coded"])
        log_likelihood = self.handler.get_marginal_probability(coord, already_encoded=True)
        return seq, log_likelihood

    def search(self, generations, members, start_coords, identity, filename=None):
        """ Evolutionary search in the latent space. Main loop. """
        def rescale_fit(to_rescale):
            """ To secure fitness values are positive """
            min_fit = min([x[0] for x in to_rescale])
            xs_rescaled = [(x+min_fit, coord, stats) for x, coord, stats in to_rescale]
            return xs_rescaled

        n = start_coords.shape[0]
        m, sigma, cov,  p_s, p_c, mu = start_coords, 0.2, np.identity(n), np.zeros(n), np.zeros(n), members // 3
        best_identity = 0.0

        step = 0
        print("while {} {}".format(abs(best_identity - identity) > 2, step < generations))
        while abs(best_identity - identity) > 2 and step < generations:
            samples = norm.rvs(mean=m, cov=sigma*cov, size=members)
            xs = list(map(lambda x: self.fitness(x, target_identity=identity), samples))
            xs = rescale_fit(xs)
            xs = sorted(xs, key=lambda x: x[0], reverse=True) # Maximize fitness
            m_prev = m
            m = update_mean(m_prev, xs, sigma, n=mu)
            p_c = update_pc(m_prev, xs, mu, p_c, n)
            cov = update_Cov(cov, m_prev, xs, mu, p_c, n)
            p_s = update_ps(p_s, cov, m_prev, xs, mu, n)
            sigma = path_length_control(sigma, p_s, m_prev, xs, mu, n)

            mean_stats = self.fitness(m, target_identity=identity)
            self.log(step, xs, mean_stats, filename)
            step += 1
            print("Steppp")

    def fitness(self, coord, target_identity):
        """
        Fitness function for CMA-ES composed of:
            1) Gaussian processes Tm predicted value + its certainty
               - In the case of low information rate in latent space do not include it
            2) Percentage target identity with query sequence
               - 0.9 means 90 % identity sequence where 90 % of amino acid stay same as in the query
            3) Latent space distance to the center, smaller means slightly better
            4) Log likelihood of sequence observing by the model
               - The higher likelihood means that sequence is more likely from aligned protein family

        Fitness function is compose to be maximized!
        """
        gp_c, iden_c, dist_c, like_c = (0.25, 0.25, 0.25, 0.02)
        seq, log_likelihood = self.decode(coord)
        fitness_value = 0.0

        # Predicted temperature, enough confidence to include it?
        tm_pred, MSE = self.gp.predict(coord, return_std=True)
        if abs(MSE) < 2.5:
            fitness_value += gp_c * tm_pred * (3.5 - abs(MSE))
        else:
            tm_pred = None
        # Percentage identity
        query_identity = sum([a == b for a, b in zip(self.query_seq, seq)]) / self.seq_len
        fitness_value -= iden_c * abs(target_identity - query_identity) # Abs distance, try another
        # Distance to the center, support multidimensional space
        center_distance = sqrt(sum(list(map(lambda x: x**2, coord))))
        fitness_value -= dist_c * center_distance
        # Log likelihood influence, quite huge negative number (-236.05 ...)
        fitness_value += log_likelihood * like_c

        return fitness_value, coord, (tm_pred, query_identity, center_distance, log_likelihood, seq)

    def log(self, step, stats, mean, filename=None):
        """ Logging method """
        def insert_newlines(string, every=80):
            return '\n'.join(string[i:i + every] for i in range(0, len(string), every))

        if step == 0:
            log_str = "######################################################\n" \
                      "# Variational autoencoder CMA-ES approach starts      \n"
        best = stats[0]
        seq = insert_newlines(best[2][4])
        mean_seq = insert_newlines(mean[2][4])
        log_str = "======================================================\n" \
                  "Step : {}\n" \
                  "Best member:\n" \
                  "best_fitness: {:.4f}; coords: {}, tm_pred: {:.4f}, identity: {:.4f}, likelihood: {:.4f}\n" \
                  "sequence: {}\n" \
                  "New mean:\n" \
                  "best_fitness: {:.4f}; coords: {},{}, tm_pred: {:.4f}, identity: {:.4f}, likelihood: {:.4f}\n" \
                  "sequence: {}\n".format(step, best[0], best[1], best[2][0], best[2][1], best[2][3], seq,
                                                mean[0], mean[1], mean[2][0], mean[2][1], mean[2][3], mean_seq)
        if filename is not None:
            filename = self.out_dir + filename
            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not
            hs = open(filename, append_write)
            hs.write(log_str)
            hs.close()
        else:
            print(log_str)


if __name__ == "__main__":
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    cmd_line = CommandHandler()

    evo = EvolutionSearch(tar_dir, cmd_line)
    coords = evo.fit_landscape()
    # ax = evo.init_plot_fitness_LS()
    # ax = highlight_coord(ax, coords, color='g')
    # evo.save_plot(name="green")
    query_coords = evo.encode(evo.query)
    evo.search(10, 30, query_coords, 0.75)
    print("Hree")
