__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/07 11:30:00"
__description__ = " This file enable to run packages from the root directory "

from argparse import ArgumentParser

from Statistics.order_statistics import run_setup as model_statistics_run
from Statistics.ancestral_tree import run_sampler as model_sampler_run
from Statistics.ancestral_tree import run_tree_highlighter as model_tree_run
from Statistics.label_mapper import run_latent_mapper as model_mapper_run
from Statistics.reconstruction_ability import run_input_dataset_reconstruction as model_input_reconstruct_run


def get_package_parser() -> ArgumentParser:
    """ Creates Argument parser """
    parser = ArgumentParser()
    parser.add_argument("--run_package_stats_order", action='store_true', default=False,
                        help="Runs 1st and 2nd order statistics over model.")
    parser.add_argument("--run_package_stats_fireprot", action='store_true', default=False,
                        help="Creates MSAs for fireprotASR")
    parser.add_argument("--run_package_stats_tree", action='store_true', default=False,
                        help="Highlights phylo tree levels in the latent space")
    parser.add_argument("--run_package_stats_mapper", action='store_true', default=False,
                        help="Highlights sequence with label in the latent space")
    parser.add_argument("--run_package_stats_reconstruction", action='store_true', default=False,
                        help="Determines how well can model reconstruct sequences")
    return parser


def run_package(parser: ArgumentParser):
    """ Run package according to the choice """
    args, unknown = parser.parse_known_args()
    if args.run_package_stats_order:
        model_statistics_run()
    if args.run_package_stats_fireprot:
        model_sampler_run()
    if args.run_package_stats_tree:
        model_tree_run()
    if args.run_package_stats_mapper:
        model_mapper_run()
    if args.run_package_stats_reconstruction:
        model_input_reconstruct_run()


if __name__ == '__main__':
    package_parser = get_package_parser()
    run_package(package_parser)