__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/25 00:30:00"

import inspect
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from multiprocessing import Pool
from scipy.spatial import distance

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from parser_handler import CmdHandler
from project_enums import VaePaths, SolubilitySetting
from msa_handlers.msa_preparation import MSA
from sequence_transformer import Transformer
from VAE_accessor import VAEAccessor
from VAE_logger import Logger


def safe_log(x, eps=1e-10):
    """ Calculate numerically stable log """
    result = np.where(x > eps, x, -10)

    np.log(result, out=result, where=result > 0)
    return result


class OrderStatistics:
    """
    Class gather methods responsible for creating first and second order statistics in sequence analyses
    It creates frequencies bins of AA at positions in artificial(generated) and target(input) MSA
    1st order                                           2nd order
           target     artificial                   target     artificial
    1  A    1           5                    1 1 A A
    .                                           .
    .                                           .
    N  A                                     1 N A A
    |  K                                     1 1 A K
    .                                        .
    .                                        .
    N  k                                     1 N A K

    where N is length of sequence in MSA
    """

    def __init__(self, setuper: CmdHandler):
        self.model_name = setuper.get_model_to_load()
        self.vae = VAEAccessor(setuper, self.model_name)
        self.dimensions = self.vae.vae.dim_latent_vars
        self.pickle = setuper.pickles_fld
        self.target_dir = setuper.high_fld + "/" + VaePaths.FREQUENCIES_STATS.value + "/"
        self.data_dir = self.target_dir + "data/"
        self.cond = setuper.conditional
        self.setup_output_folder()

        # Keep calculations of amino appearances in columns, reshape later
        """
        Profile format:
                    |-------len(msa)----|
                    0.2 0.3 0.15 ... 0.4
            cnt AA  0.4 0.3 0.15 ... 0.4
                    0.4 0.4 0.7  ... 0.2
        """
        self.msa_profile = np.zeros(1)

    def setup_output_folder(self):
        """ Creates directory in Highlight results directory """
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def shannon_entropy(self, msa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        1st order statistic/frequencies
        Computes Shannon entropy of MSA given as:
            H_j = -SUM(p^a_j * log(p^a_j))
                p^a_j = n^a_j / N
        where:
            j is alignment column
            n^a_j is number of observations in the jth column of letter a
        """
        H = np.zeros(msa.shape[1])
        alphabet = list(MSA.amino_acid_dict(self.pickle).values())
        column_p = np.zeros(len(set(alphabet)))

        frequencies = np.zeros((column_p.shape[0], msa.shape[1]))

        for j in range(msa.shape[1]):
            column_j = msa[:, j]
            for a in alphabet:
                frequencies[a, j] = np.where(column_j == a)[0].shape[0]
            column_p = frequencies[:, j] / msa.shape[0]
            H[j] = -np.sum(column_p * safe_log(column_p))
        return H, frequencies / msa.shape[0]

    def mutual_information(self, msa: np.ndarray, shannon: np.ndarray, process_num=None):
        """
        2nd order statistics/frequencies/Covariance
        Computes mutual information of MSA given as:
            I_j,k = H_j + H_k - H_j,k
        where H_j,k is joint Shannon entropy
            H_j,k = - SUM_a( SUM_b( p^a,b_j,k * log(p^a,b_j,k)))
        Covariance is computed:
            C^j,k_a,b = f^j,k_a,b - f^j_a * f^k_b
            f^j,k_a,b    - bivariate marginals, meaning the frequency of amino
                           acid combination α, β at positions i, j in the MSA.
            f^j_a, f^k_b - univariate marginals, or individual amino acid frequencies at positions i and j.
        """
        assert msa.shape[1] == shannon.shape[0]
        n = shannon.shape[0]
        msa_rows = msa.shape[0]

        # Shape according number of unique column pairs in set
        I = np.zeros(n * (n - 1) // 2)

        alphabet = list(set(MSA.amino_acid_dict(self.pickle).values()))
        aa_cnt = len(alphabet)
        column_jk_p = np.zeros(aa_cnt ** 2)

        # prepare amino acids pairs
        pairs = []
        for a in alphabet:
            pairs.extend([(a, b) for b in alphabet])
        pairs_cnt = (aa_cnt ** 2)

        # Matrix, row number of MSA columns except last column which is included in previous frequencies count
        # Row for pairs of amino acids
        # Vector length given by number of column pairs and AA pairs for each pair of column
        covariances = np.zeros(I.shape[0] * pairs_cnt)
        frequencies = np.zeros(I.shape[0] * pairs_cnt)
        # print("Shapes of covariances: {} {}".format(covariances.shape, I.shape))
        # covariances = np.zeros((shannon.shape[0] - 1, column_jk_p.shape[0]))
        # frequencies = np.zeros((shannon.shape[0] - 1, column_jk_p.shape[0]))

        index_counter = 0
        for j in range(msa.shape[1] - 1):
            for k in range(j + 1, msa.shape[1]):
                # calculate H_j,k
                for a, b in pairs:
                    vec_a, vec_b = np.where(msa[:, j] == a)[0], np.where(msa[:, k] == b)[0]
                    pair_freq = np.intersect1d(vec_a, vec_b, assume_unique=True).shape[0] / msa_rows
                    tar_index = index_counter * pairs_cnt + a * aa_cnt + b
                    frequencies[tar_index] = pair_freq
                    # frequencies[j, a * aa_cnt + b] = sum([a in vec_b for a in vec_a])
                    # Compute covariance
                    single_marginals_product = (vec_a.shape[0]/msa_rows) * (vec_b.shape[0]/msa_rows)
                    covariances[tar_index] = pair_freq - single_marginals_product
                # column_jk_p[:] = frequencies[j, :] / msa.shape[0]
                column_jk_p = frequencies[-pairs_cnt:]
                H_j_k = -np.sum(column_jk_p)
                I[index_counter] = shannon[j] + shannon[k] - H_j_k
                index_counter += 1
                if process_num is not None:
                    Logger.process_update_out(process_num, "{:.1f} %".format((index_counter / I.shape[0]) * 100))
                else:
                    Logger.update_msg("{:.1f} %".format((index_counter / I.shape[0]) * 100))
        return I[:index_counter], frequencies, covariances
        # return I[:index_counter], frequencies / msa.shape[0], covariances

    def sample_dataset_from_normal(self, origin: np.ndarray, scale: float, N: int) -> np.ndarray:
        """ Sample from the model N data points and transform them to ndarray """
        z = np.random.multivariate_normal(origin, np.identity(self.dimensions) * scale, N)
        c = np.random.randint(low=0, high=SolubilitySetting.SOLUBILITY_BINS.value, size=N) if self.cond else None
        return self.vae.decode_z_to_number(z, c)

    def plot_order_statistics(self, train, sampled, label_x, label_y, file_name, show_gap,
                              frequencies: bool = False):
        """
        Plots the order statistics of data, returns Pearson correlation coefficient
        show_gap - True if you want highlight gap frequencies in red dots
        """
        print("Variance {} vs {}".format(np.var(train.flatten()), np.var(sampled.flatten())))

        fig, ax = plt.subplots()
        OrderStatistics.created_subplot(ax, train, sampled, label_x, label_y, show_gap, frequencies)

        plt.savefig(self.target_dir + file_name, dpi=400)

    @staticmethod
    def get_plot_data_file_names():
        """ Returns names of data files for plotting """
        return ["msa_shannon.pkl", "msa_frequencies.pkl",
                "sampled_shannon.pkl", "sampled_frequencies.pkl",
                "mutual_msa.pkl", "mutual_msa_frequencies.pkl",
                "mutual_sampled.pkl", "mutual_sampled_frequencies.pkl",
                "covariances_msa.pkl", "covariances_generated.pkl"
                ]

    @staticmethod
    def get_plot_query_data_file_names():
        """ Returns names of data files for plotting """
        return ["msa_shannon_query.pkl", "msa_frequencies_query.pkl",
                "sampled_shannon_query.pkl", "sampled_frequencies_query.pkl",
                "mutual_msa_query.pkl", "mutual_msa_frequencies_query.pkl",
                "mutual_sampled_query.pkl", "mutual_sampled_frequencies_query.pkl",
                "covariances_msa_query.pkl", "covariances_generated_query.pkl"
                ]

    @staticmethod
    def created_subplot(ax, train, sampled, label_x, label_y, show_gap, frequencies):
        """ Method plotting desired graph into given subplot """
        my_rho = np.corrcoef(train.flatten(), sampled.flatten())

        ax.scatter(train.flatten(), sampled.flatten(), s=0.5)
        ax.title.set_text('Correlation = ' + "{:.2f}".format(my_rho[0, 1]))
        ax.set(xlabel=label_x, ylabel=label_y)

        if frequencies:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        if show_gap:
            gaps_msa_frequencies = train[0, :]
            gaps_sampled_frequencies = sampled[0, :]
            ax.scatter(gaps_msa_frequencies, gaps_sampled_frequencies, s=0.5, c='red')
        return ax


def run_setup():
    """ Design the run setup for this package """
    cmdline = CmdHandler()

    with open(cmdline.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_original_binary = pickle.load(file_handle)

    sample_cnt = 3000
    random_idx = np.random.permutation(range(0, msa_original_binary.shape[0]))

    msa_dataset = Transformer.binaries_to_numbers_coding(msa_original_binary[random_idx[:sample_cnt]])
    # msa_dataset = np.array([[0, 1, 2, 3, 4, 5, 6, 7,0],
    #                             [0, 1, 2, 3, 4, 5, 6, 7,0],
    #                             [0, 1, 2, 3, 4, 5, 6, 7,0],
    #                             [1, 1, 1, 1, 1, 1, 1, 1,0],
    #                             [2, 2, 2, 2, 2, 2, 2, 2,0]])
    stat_obj = OrderStatistics(cmdline)
    sampled_dataset = stat_obj.sample_dataset_from_normal(np.zeros(stat_obj.dimensions), 2.5, sample_cnt)
    # sampled_dataset = np.array([[0,1,2,3,4,5,6,7,0],
    #                    [0,1,2,3,4,5,6,7,0],
    #                    [0,1,2,3,4,5,6,7,0],
    #                    [1,1,1,1,1,1,1,1,0],
    #                    [2,2,2,2,2,2,2,2,0]])
    # 1st order stats
    msa_shannon, msa_frequencies = stat_obj.shannon_entropy(msa_dataset)
    sampled_shannon, sampled_frequencies = stat_obj.shannon_entropy(sampled_dataset)
    print("   Shannon entropy and single frequencies calculation ............... DONE")

    # 2nd order statistics
    if stat_obj.vae.use_cuda:
        # On cuda device, there is a problem with subprocess initialization so run it in parallel
        Logger.print_for_update("   Mutual information, pair frequencies and covariance for MSA ...... {}", str(0) + "%")
        mutual_msa, mutual_msa_frequencies, cov_msa = stat_obj.mutual_information(msa_dataset, msa_shannon)
        Logger.update_msg("DONE", True)
        Logger.print_for_update("   Mutual information, pair frequencies and covariance for SAMPLE ... {}", str(0)+"%")
        mutual_sampled, mutual_sampled_frequencies, cov_gen = stat_obj.mutual_information(sampled_dataset, sampled_shannon)
        Logger.update_msg("DONE", True)
    else:
        # In parallel process aligning
        pool_values = [(msa_dataset, msa_shannon, 0), (sampled_dataset, sampled_shannon, 1)]
        Logger.init_multiprocess_msg(" 2nd order stats computation .... {}", processes=2, init_value=str(0) + "%")
        pool = Pool(processes=2)
        pool_results = pool.starmap(stat_obj.mutual_information, pool_values)
        mutual_msa, mutual_msa_frequencies, cov_msa = pool_results[0]
        mutual_sampled, mutual_sampled_frequencies, cov_gen = pool_results[1]
        pool.close()
        pool.join()
        print()

    # plot 1st order data statistics and frequencies statistics
    print("=" * 80)
    print(" Creating first and second order statistics to  {}".format(stat_obj.target_dir))
    stat_obj.plot_order_statistics(msa_shannon, sampled_shannon, 'Training Data Entropy', 'VAE Sampled Entropy',
                                   'first_order.png', show_gap=False)
    stat_obj.plot_order_statistics(mutual_msa, mutual_sampled, 'Training Mutual Information',
                                   'Generated Mutual Information', 'second_order.png', show_gap=False)
    stat_obj.plot_order_statistics(cov_msa, cov_gen, 'Target MSA covariances',
                                   'Generated MSA covariances', 'second_covariances.png', show_gap=False)
    stat_obj.plot_order_statistics(msa_frequencies, sampled_frequencies,
                                   'Training Data Frequencies', 'VAE Sampled Frequencies',
                                   'first_order_frequencies.png', show_gap=True, frequencies=True)
    stat_obj.plot_order_statistics(mutual_msa_frequencies, mutual_sampled_frequencies,
                                   'Training Mutual Frequencies', 'Generated Mutual Frequencies',
                                   'second_order_frequencies.png', show_gap=False, frequencies=True)

    data_to_store = [msa_shannon, msa_frequencies, sampled_shannon, sampled_frequencies,
                     mutual_msa, mutual_msa_frequencies, mutual_sampled, mutual_sampled_frequencies, cov_msa, cov_gen]
    data_files = OrderStatistics.get_plot_data_file_names()
    for f, data in zip(data_files, data_to_store):
        with open(stat_obj.data_dir + f, 'wb') as file_handle:
            pickle.dump(data, file_handle)


def run_sampling_query():
    """ Sample around the query embedding """
    def get_nodes_closer_from_node_than(node, limit, nodes):
        d = distance.cdist(np.array([node]), nodes)
        _, closer_indexes = (d < limit).nonzero()
        return closer_indexes

    cmdline = CmdHandler()
    limit = 2.5

    with open(cmdline.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_original_binary = pickle.load(file_handle)

    try:
        with open(os.path.join(cmdline.pickles_fld, 'embeddings.pkl'), 'rb') as embed_file:
            embeddings = pickle.load(embed_file)
    except FileNotFoundError:
        print("   Please prepare MSA embedding before you run this case!!\n"
              "   run python3 runner.py run_task.py --run_package_generative_plots --json config.file")
        exit(1)

    key_to_mu = {k: mu for k, mu in zip(embeddings['keys'], embeddings['mu'])}
    query_embedding = 2.*key_to_mu[cmdline.query_id]/3.
    indx = get_nodes_closer_from_node_than(query_embedding, limit, embeddings['mu'])
    sample_cnt = indx.shape[0]

    print(f"  Creating order statistics around query with limit {limit} and {sample_cnt} samples...")

    msa_dataset = Transformer.binaries_to_numbers_coding(msa_original_binary[indx])
    stat_obj = OrderStatistics(cmdline)
    sampled_dataset = stat_obj.sample_dataset_from_normal(query_embedding, limit, sample_cnt)
    # 1st order stats
    msa_shannon, msa_frequencies = stat_obj.shannon_entropy(msa_dataset)
    sampled_shannon, sampled_frequencies = stat_obj.shannon_entropy(sampled_dataset)
    print("   Shannon entropy and single frequencies for Query embedding calculation ............... DONE")

    # 2nd order statistics
    if stat_obj.vae.use_cuda:
        # On cuda device, there is a problem with subprocess initialization so run it in parallel
        Logger.print_for_update("   Mutual information, pair frequencies and covariance for query MSA ...... {}",
                                str(0) + "%")
        mutual_msa, mutual_msa_frequencies, cov_msa = stat_obj.mutual_information(msa_dataset, msa_shannon)
        Logger.update_msg("DONE", True)
        Logger.print_for_update("   Mutual information, pair frequencies and covariance for query SAMPLE ... {}",
                                str(0) + "%")
        mutual_sampled, mutual_sampled_frequencies, cov_gen = stat_obj.mutual_information(sampled_dataset,
                                                                                          sampled_shannon)
        Logger.update_msg("DONE", True)
    else:
        # In parallel process aligning
        pool_values = [(msa_dataset, msa_shannon, 0), (sampled_dataset, sampled_shannon, 1)]
        Logger.init_multiprocess_msg(" 2nd order query stats computation .... {}", processes=2, init_value=str(0) + "%")
        pool = Pool(processes=2)
        pool_results = pool.starmap(stat_obj.mutual_information, pool_values)
        mutual_msa, mutual_msa_frequencies, cov_msa = pool_results[0]
        mutual_sampled, mutual_sampled_frequencies, cov_gen = pool_results[1]
        pool.close()
        pool.join()
        print()

    # plot 1st order data statistics and frequencies statistics
    print("=" * 80)
    print(" Creating first and second order statistics  for query to  {}".format(stat_obj.target_dir))
    stat_obj.plot_order_statistics(msa_shannon, sampled_shannon, 'Training Data Entropy', 'VAE Sampled Entropy',
                                   'first_order_query.png', show_gap=False)
    stat_obj.plot_order_statistics(mutual_msa, mutual_sampled, 'Training Mutual Information',
                                   'Generated Mutual Information', 'second_order_query.png', show_gap=False)
    stat_obj.plot_order_statistics(cov_msa, cov_gen, 'Target MSA covariances',
                                   'Generated MSA covariances', 'second_covariances_query.png', show_gap=False)
    stat_obj.plot_order_statistics(msa_frequencies, sampled_frequencies,
                                   'Training Data Frequencies', 'VAE Sampled Frequencies',
                                   'first_order_frequencies_query.png', show_gap=True, frequencies=True)
    stat_obj.plot_order_statistics(mutual_msa_frequencies, mutual_sampled_frequencies,
                                   'Training Mutual Frequencies', 'Generated Mutual Frequencies',
                                   'second_order_frequencies_query.png', show_gap=False, frequencies=True)

    data_to_store = [msa_shannon, msa_frequencies, sampled_shannon, sampled_frequencies,
                     mutual_msa, mutual_msa_frequencies, mutual_sampled, mutual_sampled_frequencies, cov_msa, cov_gen]
    data_files = OrderStatistics.get_plot_query_data_file_names()
    for f, data in zip(data_files, data_to_store):
        with open(stat_obj.data_dir + f, 'wb') as file_handle:
            pickle.dump(data, file_handle)


if __name__ == "__main__":
    run_setup()
