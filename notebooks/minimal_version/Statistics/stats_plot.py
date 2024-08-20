__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/24 14:40:00"

import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from Statistics.order_statistics import OrderStatistics
from Statistics.reconstruction_ability import Reconstructor
from VAE_accessor import VAEAccessor
from analyzer import AncestorsHandler
from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler
from project_enums import VaePaths
from sequence_transformer import Transformer

cmd_line = CmdHandler()
transformer = Transformer(cmd_line)
vae = VAEAccessor(cmd_line, cmd_line.get_model_to_load())
aligner = AncestorsHandler(cmd_line)


# class Overview:
#     """ Gathers plots of all statistical outputs """


def show_latent_space_features(ax, anc_msa_file, setuper):
    """ Plots statistics latent space -> where is query, Babkovas ancestors. """
    print("   Creating and plotting latent space...")
    with open(cmd_line.pickles_fld + "/training_alignment.pkl", 'rb') as file_handle:
        train_dict = pickle.load(file_handle)
    msa_binary, weights, msa_keys = transformer.sequence_dict_to_binary(train_dict)
    msa_binary = torch.from_numpy(msa_binary)

    # In the case of CVAE
    solubility = setuper.get_solubility_data()

    mu, sigma = vae.propagate_through_VAE(msa_binary, weights, msa_keys, solubility)
    with open(setuper.pickles_fld + "/embeddings.pkl", "wb") as embeddings_pkl:
        store_values = {"keys": msa_keys, "mu": mu}
        pickle.dump(store_values, embeddings_pkl)

    query_index = msa_keys.index(cmd_line.query_id)
    query_coords = mu[query_index]

    anc_msa = MSA.load_msa(anc_msa_file)
    ancestor_aligned = aligner.align_fasta_to_original_msa(anc_msa_file, False)  # Ancestors aligned to entire MSA
    ancestor_msa, ancestor_weight, ancestor_keys = transformer.sequence_dict_to_binary(ancestor_aligned)
    ancestor_mus, ancestor_sigma = vae.propagate_through_VAE(ancestor_msa, ancestor_weight, ancestor_keys)

    # Store coordinates of Babkovas sequences in the file
    anc_embeddings = os.path.join(setuper.pickles_fld, "anc_embeddings.pkl")
    with open(anc_embeddings, "wb") as anc_file:
        pickle.dump({"mus": ancestor_mus, "keys": ancestor_keys}, anc_file)

    fig_lat, ax_lat = plt.subplots(1, 1)

    # highlight query
    for ax in [ax, ax_lat]:
        ax.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, )
        ax.plot(query_coords[0], query_coords[1], '.', color='red')

        # Highlight Babkovas ancestors
        for ancestor_idx, anc_mu in enumerate(ancestor_mus):
            ax.plot(anc_mu[0], anc_mu[1], '.', color='black', alpha=1, markersize=3,
                    label=ancestor_keys[ancestor_idx] + '({})'.format(ancestor_idx))
            ax.annotate(str(ancestor_idx), (anc_mu[0], anc_mu[1]))
        ax.set_xlabel("$Z_1$")
        ax.set_ylabel("$Z_2$")
    fig_lat.savefig(setuper.high_fld + "/latent_space.png")


def create_bench_plot(ax, highlight_folder_path):
    """ Plots benchmarks probabilities distributions """
    print("   Loading and plotting benchmark stats...")
    pickle_bench_path = highlight_folder_path + "/" + VaePaths.BENCHMARK_STORE_FILE.value

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
    print("   Loading and plotting seq...")
    identities_file, query_identity_file = Reconstructor.get_plot_data_file_names()
    dir_path = highlight_folder_path + "/" + VaePaths.FREQUENCIES_STATS.value + "/" + VaePaths.RECONSTRUCTOR.value + "data/"
    with open(dir_path + query_identity_file, "rb") as f:
        query_identity = pickle.load(f)
    with open(dir_path + identities_file, 'rb') as f:
        identities = pickle.load(f)
    Reconstructor.created_subplot(ax, identities, query_identity, "Input dataset reconstruction")


def create_depths_tree_corr_plot(ax, highlight_folder_path, pickle_name, title):
    """ Create plot for depths correlations """
    print("   Loading and plotting tree correlation...")
    mapping_path = highlight_folder_path + "/" + VaePaths.TREE_EVALUATION_DIR.value + "/"
    with open(mapping_path + pickle_name, "rb") as file_handle:
        correlations = pickle.load(file_handle)
    ax.hist(correlations, bins=25)
    ax.set_title(title)
    ax.set(xlabel=" Pearson correlation coefficient", ylabel="")


def create_orders_statistics_plots(axs, highlight_folder_path):
    """ Creates 4 plot with 1st 2nd order frequencies/entropies """
    print("   Loading and plotting order statistics...")
    order_dir = highlight_folder_path + "/" + VaePaths.FREQUENCIES_STATS.value + "/data/"

    data = []
    data_files = OrderStatistics.get_plot_data_file_names()
    for i, f in enumerate(data_files):
        with open(order_dir + f, 'rb') as file_handle:
            data.append(pickle.load(file_handle))

    msa_shannon, msa_frequencies, sampled_shannon, sampled_frequencies = data[:4]
    mutual_msa, mutual_msa_frequencies, mutual_sampled, mutual_sampled_frequencies, cov_msa, cov_gen = data[4:]
    # OrderStatistics.created_subplot(axs[1, 0], msa_shannon, sampled_shannon, 'Training Data Entropy',
    #                                 'VAE Sampled Entropy',
    #                                 show_gap=False, frequencies=False)
    # OrderStatistics.created_subplot(axs[1, 1], mutual_msa, mutual_sampled, 'Training Mutual Entropy',
    #                                 'Generated Mutual Entropy', show_gap=False, frequencies=False)
    OrderStatistics.created_subplot(axs[0, 0], msa_frequencies, sampled_frequencies,
                                    'Training Data Frequencies', 'VAE Sampled Frequencies',
                                    show_gap=True, frequencies=True)
    OrderStatistics.created_subplot(axs[0, 1], cov_msa, cov_gen,
                                    'Target MSA covariances', 'Generated MSA covariances',
                                    show_gap=False, frequencies=False)


def make_overview():
    """ Creates 8 subplot plot for generative evaluation """
    cmd_line = CmdHandler()
    highlight_dir = cmd_line.high_fld

    fig, axs = plt.subplots(4, 2, figsize=(12, 16), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    show_latent_space_features(axs[0, 0], cmd_line.highlight_files, cmd_line)
    create_bench_plot(axs[0, 1], highlight_dir)
    create_seq_identity_plot(axs[1, 0], highlight_dir)
    create_depths_tree_corr_plot(axs[1, 1], highlight_dir, "correlations.pkl",
                                 "Correlation of latent center distance and depth in the tree")
    create_depths_tree_corr_plot(axs[2, 0], highlight_dir, "r2_correlations.pkl",
                                 "R2 correlation of latent space distance and depth in the tree")
    create_depths_tree_corr_plot(axs[2, 1], highlight_dir, "pcc_correlations.pkl",
                                 "PCC correlation with 1st PCA component")
    create_orders_statistics_plots(axs[3:, :], highlight_dir)

    plt.tight_layout()
    plot_file = highlight_dir + "/generative_evaluation.png"
    print("    Generative evaluation stored in : {}".format(plot_file))
    plt.savefig(plot_file, dpi=600)


def plot_latent_space():
    """ Plot latent space  """
    cmd_line = CmdHandler()

    fig, axs = plt.subplots(1, 1)
    show_latent_space_features(axs[0], cmd_line.highlight_files, cmd_line)
    print("Latent space features plotted in : {}".format(cmd_line.high_fld + "/latent_space.png"))
