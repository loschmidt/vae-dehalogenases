__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/04/13 13:09:00"

import argparse
import pickle

"""
    This script is created for for preparation of sequence library with 
    known thermostability, soluble properties by applying mutations provided by 
    publications.
"""


class CommandHandler:
    def __init__(self):
        args = self._get_parser()
        self.txt_path = args.source_txt
        self.csv_path = args.source_csv

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(description='Parameters for script creating mutation library')
        parser.add_argument("--source_csv", type=str, default=None, help="The path to the mutation file in format csv. "
                                                                         "See csv parser method description")
        parser.add_argument("--source_txt", type=str, default=None, help="The path to the mutation file in format"
                                                                         "simple txt. See txt parsermethod description")
        parser.add_argument("--pickle_fld", type=str, default=None, help="the path to the pickle format with prepare"
                                                                         "training dataset to model")
        args = parser.parse_args()
        return args


class Curator:
    """
        Prepare library of mutatants with known properties
    """

    def __init__(self, cmd_hand):
        self.cmd_hand = cmd_hand
        self._load_pickles()
        if cmd_hand.txt_path:
            self.parse_txt()
        else:
            self.parse_csv()

    def parse_txt(self):
        """
            Supporting multiple point mutants
            File expected format :
            [fasta core sequence
                [X245Y delta_temp]|[#sequence in fasta temp]
                S12Y;S44U delta_temp
                ....
                ....]+ <- repeated more times
        """
        pass

    def parse_csv(self):
        """
            CSV file expected format:
            mutation column        temp      models.. .. ..
            C24S;C55S ...         0.5777     -0.454    5.555
            ..
            ..
        """
        pass

    def _load_pickles(self):
        with open(self.setuper.pickles_fld + "/keys_list.pkl", 'rb') as file_handle:
            self.seq_keys = pickle.load(file_handle)