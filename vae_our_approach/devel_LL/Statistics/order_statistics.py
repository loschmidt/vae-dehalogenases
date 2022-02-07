__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/25 00:30:00"

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ..parser_handler import CmdHandler
from ..project_enums import VaePaths
from ..msa_preparation import MSA
from ..sequence_transformer import Transformer
from ..VAE_accessor import VAEAccessor


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
        self.target_dir = setuper.high_fld + "/" + VaePaths.STATISTICS_DIR + "/"
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

    def shannon_entropy(self, msa: np.ndarray) -> np.ndarray:
        """
        1st order statistic
        Computes Shannon entropy of MSA given as:
            H_j = -SUM(p^a_j * log(p^a_j))
                p^a_j = n^a_j / N
        where:
            j is alignment column
            n^a_j is number of observations in the jth column of letter a
        """
        H = np.zeros(msa.shape[1])
        alphabet = list(MSA.amino_acid_dict(self.pickle).values())
        column_p = np.zeros(len(alphabet))

        self.msa_profile = np.zeros(column_p.shape[0], msa.shape[1])

        for j in range(msa.shape[0]):
            column_j = msa[:, j]
            for a in alphabet:
                column_p[a] = np.where(column_j == a)[0].shape[0] / msa.shape[0]
            H[j] = -np.sum(column_p * np.log(column_p))
            # Store MSA profile for the future
            self.msa_profile[:, j] = column_p
        return H

    def mutual_information(self, msa: np.ndarray, shannon: np.ndarray) -> np.ndarray:
        """
        2nd order statistics
        Computes mutual information of MSA given as:
            I_j,k = H_j + H_k - H_j,k
        where H_j,k is joint Shannon entropy
            H_j,k = - SUM_a( SUM_b( p^a,b_j,k * log(p^a,b_j,k)))
        """
        assert msa.shape[1] == shannon.shape[0]
        n = msa.shape[0]

        # Shape according number of unique pairs in set
        I = np.zeros(n * (n - 1) / 2)

        alphabet = list(MSA.amino_acid_dict(self.pickle).values())
        column_jk_p = np.zeros(len(alphabet)**2)

        # prepare amino acids pairs
        pairs = []
        for a in alphabet:
            pairs = [(a, b) for b in alphabet]

        for j in range(msa.shape[0]-1):
            for k in range(j+1, msa.shape[0]):
                # calculate H_j,k
                for a, b in pairs:
                    vec_a, vec_b = np.where(msa[:, j] == a)[0], np.where(msa[:, k] == b)[0]
                    column_jk_p[a*len(alphabet) + b] = sum([a in vec_b for a in vec_a]) / msa.shape[0]
                H_j_k = -np.sum(column_jk_p)
                I[j*msa.shape[1] + k] = shannon[j] + shannon[k] - H_j_k
        return I

    def sample_dataset_from_normal(self, origin: np.ndarray, scale: float, N: int) -> np.ndarray:
        """ Sample from the model N data points and transform them to ndarray """
        z = np.random.multivariate_normal(origin, np.identity(self.dimensions) * scale, N)
        return self.vae.decode_z_to_number(z)

    def plot_order_statistics(self, train, sampled, label_x, label_y, file_name) -> float:
        """ Plots the order statistics of data, returns Pearson correlation coefficient """
        my_rho = np.corrcoef(train, sampled)

        fig, ax = plt.subplots()
        ax[0, 0].scatter(train, sampled)
        ax[0, 0].title.set_text('Correlation = ' + "{:.2f}".format(my_rho[0, 1]))
        ax[0, 0].set(xlabel=label_x, ylabel=label_y)

        plt.savefig(self.target_dir + file_name, dpi=400)
        return my_rho[0, 1]


if __name__ == "__main__":
    cmdline = CmdHandler()

    with open(cmdline.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
        msa_original_binary = pickle.load(file_handle)

    msa_dataset = Transformer.binary_to_numbers_coding(msa_original_binary)

    stat_obj = OrderStatistics(cmdline)
    sampled_dataset = stat_obj.sample_dataset_from_normal(np.zeros(stat_obj.dimensions), 1.5, msa_dataset.shape[0])

    # 1st order stats
    msa_shannon = stat_obj.shannon_entropy(msa_dataset)
    sampled_shannon = stat_obj.shannon_entropy(sampled_dataset)

    # 2nd order statistics
    mutual_msa = stat_obj.mutual_information(msa_dataset, msa_shannon)
    mutual_sampled = stat_obj.mutual_information(sampled_dataset, sampled_shannon)

    # plot 1st order data statistics
    stat_obj.plot_order_statistics(msa_shannon, sampled_shannon, 'Training Data Entropy', 'VAE Sampled Entropy',
                                   'first_order.png')
    stat_obj.plot_order_statistics(mutual_msa, mutual_sampled, 'Training Mutual Information',
                                   'Generated Mutual Information', 'second_order.png')
