__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/07 11:30:00"
__description__ = " This file enable to run packages from the root directory "

from argparse import ArgumentParser

from Statistics.order_statistics import run_setup as model_statistics_run


def get_package_parser() -> ArgumentParser:
    """ Creates Argument parser """
    parser = ArgumentParser()
    parser.add_argument("--run_package_stats", action='store_true', default=False,
                        description="Runs 1st and 2nd order statistics over model.")
    return parser


def run_package(parser: ArgumentParser):
    """ Run package according to the choice """
    args = parser.parse_args()
    if args.run_package_stats:
        model_statistics_run()


if __name__ == '__main__':
    package_parser = get_package_parser()
    run_package(package_parser)
