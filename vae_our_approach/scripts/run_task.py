__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/07 11:30:00"
__description__ = " This file enable to run packages from the root directory "

import sys
from argparse import ArgumentParser


if sys.argv[1] not in ["--help", "-h"]:
    from Statistics.ancestral_tree import run_sampler as model_sampler_run
    from Statistics.ancestral_tree import run_tree_highlighter as model_tree_run
    from Statistics.ancestral_tree import plot_tree_to_latent_space
    from Statistics.label_mapper import run_latent_mapper as model_mapper_run
    from Statistics.label_mapper import run_latent_dhaa115_mapper as model_dhaa115_run
    from Statistics.order_statistics import run_setup as model_statistics_run
    from Statistics.reconstruction_ability import run_input_dataset_reconstruction as model_input_reconstruct_run
    from Statistics.stats_plot import make_overview as plot_overview
    from Statistics.entropy_latent import run_entropy as model_entropy

    from reconstruction.mutagenesis import run_random_mutagenesis, run_straight_evolution
    from reconstruction.evo_search import run_cma_es_evolution
    from reconstruction.riemannian import run_riemannian_ancestors
    from reconstruction.ensembleEvolution import run_ensemble_evolution

    from VAE_logger import Capturing



def run_benchmark():
    """ Bench our model """
    pass


def get_package_parser() -> ArgumentParser:
    """ Creates Argument parser """
    parser = ArgumentParser()
    parser.add_argument("--run_package_stats_order", action='store_true', default=False,
                        help="Runs 1st and 2nd order statistics over model.")
    parser.add_argument("--run_package_stats_fireprot", action='store_true', default=False,
                        help="Creates MSAs for fireprotASR")
    parser.add_argument("--run_package_stats_tree", action='store_true', default=False,
                        help="Highlights phylo tree levels in the latent space")
    parser.add_argument("--run_package_stats_tree_plot", action='store_true', default=False,
                        help="Map tree to the latent space")
    parser.add_argument("--run_package_stats_mapper", action='store_true', default=False,
                        help="Highlights sequence with label in the latent space")
    parser.add_argument("--run_package_stats_map_dhaa115", action='store_true', default=False,
                        help="Highlights sequence with label in the latent space of DhaA115 mutants")
    parser.add_argument("--run_package_stats_reconstruction", action='store_true', default=False,
                        help="Determines how well can model reconstruct sequences")
    parser.add_argument("--run_package_stats_entropy", action='store_true', default=False,
                        help="Highlight entropy in the latent space")
    parser.add_argument("--run_generative_evaluation_plot", action='store_true', default=False,
                        help="Plot all statistics in one plot")
    parser.add_argument("--run_generative_evaluation", action='store_true', default=False,
                        help="Run all statistics and then plot")
    parser.add_argument("--run_random_mutagenesis", action='store_true', default=False,
                        help="Run random mutagenesis for query sequence")
    parser.add_argument("--run_multicriterial_random_mutagenesis", action='store_true', default=False,
                        help="Run random multicriterial mutagenesis for query sequence with model sequence probability")
    parser.add_argument("--run_evolution", action='store_true', default=False,
                        help="Run evolution in the latent space using Covariance matrix adaptation evolution strategy")
    parser.add_argument("--run_straight_evolution", action='store_true', default=False,
                        help="Run direct evolution strategy")
    parser.add_argument("--run_riemannian_anc_evolution", action='store_true', default=False,
                        help="Run riemannian manifold strategy for origin ancestral strategy")
    parser.add_argument("--run_ensemble_evolution", action='store_true', default=False,
                        help="Run ensemble straight ancestor evolution for more models")
    return parser


def run_package(parser: ArgumentParser):
    """ Run package according to the choice """
    if sys.argv[1] in ["--help", "-h"]:
        parser.print_help()
        exit(0)
    args, unknown = parser.parse_known_args()
    if args.run_package_stats_order:
        model_statistics_run()
    if args.run_package_stats_fireprot:
        model_sampler_run()
    if args.run_package_stats_tree:
        model_tree_run()
    if args.run_package_stats_tree_plot:
        plot_tree_to_latent_space()
    if args.run_package_stats_mapper:
        model_mapper_run()
    if args.run_package_stats_map_dhaa115:
        model_dhaa115_run()
    if args.run_package_stats_entropy:
        model_entropy(optimized_entropy=True)
    if args.run_package_stats_reconstruction:
        model_input_reconstruct_run()
    if args.run_generative_evaluation_plot:
        plot_overview()
    if args.run_generative_evaluation:
        print("  Running order statistics.. ")
        model_statistics_run()
        print("  Running order tree reconstruction.. ")
        with Capturing() as output:
            model_tree_run()
        print("  Running order mapper.. ")
        with Capturing() as output:
            model_mapper_run()
        print("  Running order reconstruction.. ")
        with Capturing() as output:
            model_input_reconstruct_run()
    if args.run_random_mutagenesis:
        run_random_mutagenesis()
    if args.run_multicriterial_random_mutagenesis:
        run_random_mutagenesis(multicriterial=True)
    if args.run_evolution:
        run_cma_es_evolution()
    if args.run_straight_evolution:
        run_straight_evolution()
    if args.run_riemannian_anc_evolution:
        run_riemannian_ancestors()
    if args.run_ensemble_evolution:
        run_ensemble_evolution()


if __name__ == '__main__':
    package_parser = get_package_parser()
    run_package(package_parser)
