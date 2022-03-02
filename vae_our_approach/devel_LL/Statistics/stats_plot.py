__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/24 14:40:00"

import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Statistics.reconstruction_ability  import Reconstructor
from Statistics.order_statistics import OrderStatistics
from parser_handler import CmdHandler
from sequence_transformer import Transformer
from VAE_accessor import VAEAccessor
from analyzer import AncestorsHandler
from project_enums import VaePaths

cmd_line = CmdHandler()
transformer = Transformer(cmd_line)
vae = VAEAccessor(cmd_line, cmd_line.get_model_to_load())
aligner = AncestorsHandler(cmd_line)


# class Overview:
#     """ Gathers plots of all statistical outputs """


def show_latent_space_features(ax):
    """ Plots statistics latent space -> where is query, Babkovas ancestors. """
    with open(cmd_line.pickles_fld + "/training_alignment.pkl", 'rb') as file_handle:
        train_dict = pickle.load(file_handle)
    msa_binary, weights, msa_keys = transformer.sequence_dict_to_binary(train_dict)
    msa_binary = torch.from_numpy(msa_binary)
    mu, sigma = vae.propagate_through_VAE(msa_binary, weights, msa_keys)

    query_index = msa_keys.index(cmd_line.query_id)
    query_coords = mu[query_index]

    ancestor_aligned = aligner.align_fasta_to_original_msa(cmd_line.in_file, already_msa=False, verbose=False)
    ancestor_msa, ancestor_weight, ancestor_keys = transformer.sequence_dict_to_binary(ancestor_aligned)
    ancestor_mus, ancestor_sigma = vae.propagate_through_VAE(ancestor_msa, ancestor_weight, ancestor_keys)

    # highlight query
    ax.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, )
    ax.plot(query_coords[0], query_coords[1], '.', color='red')

    # Highlight Babkovas ancestors
    for ancestor_idx, anc_mu in enumerate(ancestor_mus):
        ax.plot(anc_mu[0], anc_mu[1], '.', color='black', alpha=1, markersize=3,
                label=ancestor_keys[ancestor_idx] + '({})'.format(ancestor_idx))
        ax.annotate(str(ancestor_idx), (anc_mu[0], anc_mu[1]))
    ax.set_xlabel("$Z_1$")
    ax.set_ylabel("$Z_2$")


def create_bench_plot(ax, highlight_folder_path):
    """ Plots benchmarks probabilities distributions """
    pickle_bench_path = highlight_folder_path + "/" + VaePaths.BENCHMARK_DATA.value

    with open(pickle_bench_path, "rb") as file_handle:
        data_dict = pickle.load(file_handle)
        data_frame = pd.DataFrame.from_dict(data_dict["data_dict"])
        sns.histplot(ax=ax, data=data_frame, x="Probabilities", hue="Dataset", multiple="dodge", shrink=.8,
                     color=["green", "black", "red", "orange"])
        ax.set_xlabel("% probability of observing")
        ax.set_ylabel("Density")
        ax.set_title(r'Benchmark histogram $\mu={0:.2f},{1:.2f},{2:.2f},{3:.2f}$'.format(data_dict["mean_n"],
                                                                                         data_dict["mean_p"],
                                                                                         data_dict["mean_t"],
                                                                                         data_dict["mean_a"]))


def create_seq_identity_plot(ax, highlight_folder_path):
    """ Creates plot for sequence identities """
    identities_file, query_identity_file = Reconstructor.get_plot_data_file_names()
    dir_path = highlight_folder_path + "/" + VaePaths.FREQUENCIES_STATS.value + "/" + VaePaths.RECONSTRUCTOR.value + "data/"
    with open(dir_path + query_identity_file, "rb") as f:
        query_identity = pickle.load(f)
    with open(dir_path + identities_file, 'rb') as f:
        identities = pickle.load(f)
    Reconstructor.created_subplot(ax, identities, query_identity, "Input dataset reconstruction")


def create_depths_tree_corr_plot(ax, highlight_folder_path):
    """ Create plot for depths correlations """
    mapping_path = highlight_folder_path + "/" + VaePaths.TREE_EVALUATION_DIR.value + "/"
    with open(mapping_path + "correlations.pkl", "rb") as file_handle:
        correlations = pickle.load(file_handle)
    ax.hist(correlations, bins=25)
    ax.set_title("Correlation of latent center distance and depth in the tree")
    ax.set(xlabel=" Pearson correlation coefficient", ylabel="")


def create_orders_statistics_plots(axs, highlight_folder_path):
    """ Creates 4 plot with 1st 2nd order frequencies/entropies """
    ordes_dir = highlight_folder_path + VaePaths.FREQUENCIES_STATS.value + "/"

    msa_shannon, msa_frequencies, sampled_shannon, sampled_frequencies = [], [], [], []
    mutual_msa, mutual_msa_frequencies, mutual_sampled, mutual_sampled_frequencies = [], [], [], []
    data_to_store = [msa_shannon, msa_frequencies, sampled_shannon, sampled_frequencies,
                     mutual_msa, mutual_msa_frequencies, mutual_sampled, mutual_sampled_frequencies]

    data_files = OrderStatistics.get_plot_data_file_names()
    for f, data in zip(data_files, data_to_store):
        with open(ordes_dir + f, 'rb') as file_handle:
            pickle.dump(data, file_handle)

    OrderStatistics.created_subplot(axs[1, 0], msa_shannon, sampled_shannon, 'Training Data Entropy',
                                    'VAE Sampled Entropy',
                                    show_gap=False, frequencies=False)
    OrderStatistics.created_subplot(axs[1, 1], mutual_msa, mutual_sampled, 'Training Mutual Entropy',
                                    'Generated Mutual Entropy', show_gap=False, frequencies=False)
    OrderStatistics.created_subplot(axs[0, 0], msa_frequencies, sampled_frequencies,
                                    'Training Data Frequencies', 'VAE Sampled Frequencies',
                                    show_gap=True, frequencies=True)
    OrderStatistics.created_subplot(axs[0, 1], mutual_msa_frequencies, mutual_sampled_frequencies,
                                    'Training 2nd Order Frequencies', 'Generated 2nd Order Frequencies',
                                    show_gap=False, frequencies=True)


def make_overview():
    """ Creates 8 subplot plot for generative evaluation """
    cmd_line = CmdHandler()
    highlight_dir = cmd_line.high_fld

    fig, axs = plt.subplots(4, 2)
    show_latent_space_features(axs[0, 0])
    create_bench_plot(axs[0, 1], highlight_dir)
    create_seq_identity_plot(axs[1, 0], highlight_dir)
    create_depths_tree_corr_plot(axs[1, 1], highlight_dir)
    create_orders_statistics_plots(axs[2:, :], highlight_dir)

    plot_file = highlight_dir + "generative_evaluation.png"
    print("    Generative evaluation stored in : {}", format(plot_file))
    plt.savefig(plot_file, dpi=600)
