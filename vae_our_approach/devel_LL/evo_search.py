__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/04/16 00:10:00"

import pickle
from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from EVO.create_library import CommandHandler, Curator
from analyzer import AncestorsHandler
from VAE_accessor import VAEAccessor
from benchmark import Benchmarker as Vae_encoder
from cma_ev import update_mean, update_pc, update_Cov, update_ps, path_length_control
from experiment_handler import ExperimentStatistics
from parser_handler import CmdHandler


def highlight_coord(ax, mus, color='r'):
    ax.plot(mus[:, 0], mus[:, 1], '.', alpha=1, markersize=3, color=color)
    return ax


def load_means_from_file(filename):
    """ Helpful function due to quotes at Metacentrum """
    means = []
    with open(filename, 'r') as lines:
        for line in lines:
            if line[0] == "#" or line.split(";")[0] == "Step":
                continue
            items = line.split(";")
            x, y = float(items[5]), float(items[6])
            means.append([x, y])
    return np.array(means)


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
        self.handler = VAEAccessor(setuper)
        self.gp = None
        self.fitness_class_setting = (0.7, 0.25, 1.5, 0.5)
        self.log_str = ""
        self.exp_stats_handler = ExperimentStatistics(setuper, experiment_name="cma_search")

        with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
            self.query = pickle.load(file_handle)
            self.query_seq = list(self.query.values())[0]
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
        file_name = 'thermo_mutant_library.fasta'
        msg = "Evolutionary search : Mutant library saved to"
        self.exp_stats_handler.store_ancestor_dict_in_fasta(mutants, file_name, msg=msg)

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
        return ax, fig

    def save_plot(self, name):
        save_path = self.out_dir + '{}.png'.format(name)
        print("Evo Search message : saving plot to", save_path)
        plt.savefig(save_path)

    def encode(self, seqs):
        """ Encode dictionary to the latent space """
        binary, _, _ = self.vae.binaryConv.prepare_aligned_msa_for_Vae(seqs)
        X, _ = self.vae.prepareMusSigmas(binary)
        return X

    def decode(self, coords):
        """ Decode sequence from the latent space, also return log likelihood """
        vae_coord = [tuple(coord) for coord in coords]
        seq_dict = self.handler.decode_z_to_aa_dict(vae_coord, "coded")
        seqs = list(seq_dict.values())
        binary = self.get_binary(seq_dict)
        log_likelihood = self.handler.get_marginal_probability(binary, multiple_likelihoods=True)
        return seqs, log_likelihood.numpy()

    def get_binary(self, seqs):
        """ Encode sequence to one hot encoding and prepare for vae"""
        binary, _, _ = self.vae.binaryConv.prepare_aligned_msa_for_Vae(seqs)
        binary = binary.astype(np.float32)
        binary = binary.reshape((binary.shape[0], -1))
        binary = torch.from_numpy(binary)
        return binary

    def search(self, generations, members, start_coords, identity, step=0.5, filename=None, pareto=True):
        """ Evolutionary search in the latent space. Main loop. """

        def rescale_fit(to_rescale):
            """ To secure fitness values are positive """
            min_fit = min([x[0] for x in to_rescale])
            xs_rescaled = [(x + min_fit, coord, stats) for x, coord, stats in to_rescale]
            return xs_rescaled

        n = start_coords.shape[0]
        m, sigma, cov, p_s, p_c, mu = start_coords, step, np.identity(n), np.zeros(n), np.zeros(n), members // 3
        best_identity = 0.0

        step, means = 0, [m]
        while abs(best_identity - identity) > 0.04 and step < generations:
            samples = norm.rvs(mean=m, cov=sigma * cov, size=members)
            xs = self.fitness(samples, target_identity=identity, pareto=pareto)
            xs = rescale_fit(xs)
            xs = sorted(xs, key=lambda x: x[0], reverse=True)  # Maximize fitness
            mu = len(xs) if pareto else mu
            m_prev = m
            m = update_mean(m_prev, xs, sigma, n=mu)
            p_c = update_pc(m_prev, xs, mu, p_c, n)
            cov = update_Cov(cov, m_prev, xs, mu, p_c, n)
            p_s = update_ps(p_s, cov, m_prev, xs, mu, n)
            sigma = path_length_control(sigma, p_s, m_prev, xs, mu, n)

            mean_stats = self.fitness(np.array([m]), target_identity=identity)
            self.log(step, sigma, xs, mean_stats[0], filename)
            step += 1
            best_identity = xs[0][2][1]
            means.append(m)

            if filename is not None and step % 10 == 0:
                print("# Progress report : step {} / {}".format(step, generations))
        self.log_to_file(filename)
        return means

    def fitness(self, coords, target_identity, pareto=True):
        """
        Fitness function for CMA-ES composed of:
            1) Gaussian processes Tm predicted value + its certainty
               - In the case of low information rate in latent space do not include it
            2) Percentage target identity with query sequence
               - 0.9 means 90 % identity sequence where 90 % of amino acid stay same as in the query
            3) Latent space distance to the center, smaller means slightly better
            4) Log likelihood of sequence observing by the model
               - The higher likelihood means that sequence is more likely from aligned protein family

        Modes: Fitness function can run in different mode
                True - Pareto, False - Weight multicriterial fitness function

        Fitness function is compose to be maximized!
        """
        gp_c, iden_c, dist_c, like_c = self.fitness_class_setting  # (0.7, 0.25, 1.5, 0.5)
        seqs, log_likelihoods = self.decode(coords)

        # Predicted temperature
        tm_pred, MSE = self.gp.predict(coords, return_std=True)

        xs = []
        for i in range(coords.shape[0]):
            fitness_value, tm = 0.0, -6.666  # Magic constant
            # Enough confidence to include it?
            if abs(MSE[i]) < 2.5:
                fitness_value += gp_c * tm_pred[i] * (3.5 - abs(MSE))
                tm = tm_pred[i]
            # Percentage identity
            query_identity = sum([1 if a == b else 0 for a, b in zip(self.query_seq, seqs[i])]) / self.seq_len
            fitness_value -= iden_c * abs(target_identity - query_identity) * 100  # Abs distance, try another
            # Distance to the center, support multidimensional space
            center_distance = np.linalg.norm(coords[i])
            fitness_value -= dist_c * center_distance
            # Log likelihood influence, quite huge negative number (-236.05 ...)
            fitness_value += log_likelihoods[i] * like_c
            xs.append((fitness_value, coords[i], (tm, query_identity, center_distance, log_likelihoods[i], seqs[i])))
        # In the case of pareto optimization choose only not dominated fitness
        if pareto:
            xs_sorted = sorted(xs, key=lambda x: x[0], reverse=True)
            i, id = 0, target_identity
            while i < len(xs_sorted) - 1:
                x, j = xs_sorted[i][2], i + 1
                while j < len(xs_sorted):
                    y = xs_sorted[j][2]
                    # x >> y, y is dominated by x
                    if x[0] >= y[0] and abs(id - x[1]) > abs(id - y[1]) and x[2] < y[2] and x[3] > y[3]:
                        xs_sorted.pop(j)
                    else:
                        j += 1
                i += 1
            xs = xs_sorted
        return xs

    def log(self, step, sigma, stats, mean, filename=None):
        """ Logging method """

        def insert_newlines(string, every=80):
            string = "".join(string)
            return '\n'.join(string[i:i + every] for i in range(0, len(string), every))

        log_str = ""
        if step == 0:
            self.log_str = "#####################################################################################\n" \
                           "# Variational autoencoder CMA-ES approach starts      \n" \
                           "Step;Step_size;fitness;identity;likelihood;MEAN_x;MEAN_y;fitness;identity;likelihood;Sekvence best;mean\n"
        best = stats[0]
        seq = insert_newlines(best[2][4])
        mean_seq = insert_newlines(mean[2][4])
        self.log_str += "{};{};{:.4f};{:.4f};{:.4f};{};{};{};{};{};{}" \
                        "\n".format(step, sigma, best[0], best[2][1], best[2][3],
                                    mean[1][0], mean[1][1], mean[0], mean[2][1], mean[2][3], "".join(mean[2][4]))
        if filename is None:
            print(self.log_str)
            self.log_str = ""

    def log_to_file(self, filename):
        if filename is not None:
            filename = self.out_dir + filename
            hs = open(filename, 'w')
            hs.write(self.log_str)
            hs.close()

    @staticmethod
    def _trajectories(runs):
        """ Prepare trajectories for gif """

        def splitter(p1, p2, cnt=5):
            x1, y1 = p1
            x2, y2 = p2
            d_x, d_y = (x2 - x1) / cnt, (y2 - y1) / cnt
            return [[x1 + i * d_x, y1 + i * d_y] for i in range(1, cnt + 1)]

        # Make smooth movement
        trajectories = []
        for run in runs:
            trajectory = [run[0]]
            for i in range(1, len(run)):
                trajectory.extend(splitter(run[i - 1], run[i]))
            trajectories.append(trajectory)
        return np.array(trajectories)

    def animate(self, runs, generations, gif='search.gif'):
        """ Creates an demonstration movement in the latent space only for 2D"""
        from celluloid import Camera

        split_ratio = 5  # Intervals divided to 5 intervals
        trajectories = EvolutionSearch._trajectories(runs)

        ax, fig = self.init_plot_fitness_LS(plot_landscape=False)
        mus, _, _ = self.handler.latent_space(check_exists=True)
        camera = Camera(fig)
        for i in range(generations * split_ratio + 1):
            ax.plot(mus[:, 0], mus[:, 1], '.', alpha=0.1, markersize=3)
            for r in trajectories:
                ax.scatter(r[i][0], r[i][1], marker="o", s=13 ** 2, color="black", alpha=1.0)
                ax.plot(r[0:i, 0], r[0:i, 1], linestyle="dashed", linewidth=2, color="black", label=str(i))
            plt.tight_layout()
            camera.snap()
        animation = camera.animate(interval=110, repeat=True, repeat_delay=500)  # create animation
        animation.save(self.out_dir + gif, writer="imagemagick")  # save animation as gif
        print("Animation {} saved into {}".format(gif, self.out_dir))


if __name__ == "__main__":
    tar_dir = CmdHandler()
    cmd_line = CommandHandler()

    # Prepare evo class
    evo = EvolutionSearch(tar_dir, cmd_line)
    coords = evo.fit_landscape()

    ##########################
    # Experiment setup
    experiment_runs = 10
    experiment_generations = 50
    population = 64
    sigma_step = 0.5
    target_identity = 0.75
    query_coords = evo.encode(evo.query)[0]
    PARETO, WEIGHT = True, False
    # Fitness influence of: Tm, identity, distance to center, likelihood. Applies only in weight mode!!
    evo.fitness_class_setting = (0.7, 0.35, 1.1, 0.8)

    ##########################
    # Run experiments
    run_trajectories = []
    for run_i in range(experiment_runs):
        print("=" * 80)
        print("# Run {} out of {}".format(run_i + 1, experiment_runs))
        ret = evo.search(experiment_generations, population, query_coords, target_identity,
                         sigma_step, pareto=PARETO, filename="run_pokus{}.csv".format(run_i + 1))
        run_trajectories.append(ret)
    #evo.animate(run_trajectories, experiment_generations)
