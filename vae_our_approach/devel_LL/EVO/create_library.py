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
        self.pickle_fld = args.pickle_fld

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(description='Parameters for script creating mutation library')
        parser.add_argument("--source_csv", type=str, default=None, help="The path to the mutation file in format csv. "
                                                                         "See csv parser method description")
        parser.add_argument("--source_txt", type=str, default=None, help="The path to the mutation file in format"
                                                                         "simple txt. See txt parsermethod description")
        parser.add_argument("--pickle_fld", type=str, default=None, help="the path to the pickle format with prepare"
                                                                         "training dataset to model")
        args, unknown = parser.parse_known_args()
        return args


class Curator:
    """
        Prepare library of mutatants with known properties
    """

    def __init__(self, cmd_hand):
        self.cmd_hand = cmd_hand
        #self.seq_keys = self._load_pickles()
        if cmd_hand.txt_path:
            self.mutants, self.temps = self.parse_txt()
        else:
            self.mutants, self.temps = self.parse_csv()

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
        START = 0
        LOAD_SRC = 1
        MUT = 2
        MUT_FASTA = 3

        source_seq = []
        mutated_seqs = {}
        mutated_temp = {}
        state = START
        with open(self.cmd_hand.txt_path, "r") as file:
            for i, line in enumerate(file):
                if state == START:
                    state = LOAD_SRC
                    continue
                if state == LOAD_SRC:
                    if line[0] == '>':
                        state = MUT
                    else:
                        source_seq.extend(list(line[:-1]))
                    continue
                if state == MUT:
                    if line[0] == '>':
                        state = MUT_FASTA
                        continue
                    muts, temp = line.split()[0], line.split()[-1]
                    seq = mutate_seq(source_seq, muts)
                    mutated_seqs[muts] = seq
                    mutated_temp[muts] = float(temp)
                if state == MUT_FASTA:
                    pass
        return mutated_seqs, mutated_temp

    def parse_csv(self):
        """
            CSV file expected format:
            mutation column        temp      models.. .. ..
            C24S;C55S ...         0.5777     -0.454    5.555
            ..
            ..
        """
        pass

    def get_data(self):
        return self.mutants, self.temps

    def _load_pickles(self):
        with open(self.cmd_hand.pickles_fld + "/keys_list.pkl", 'rb') as file_handle:
            seq_keys = pickle.load(file_handle)
            return seq_keys


def mutate_seq(seq, muts):
    """ mutation format S144F[;S144K]* """
    muts = muts.split(';')
    mutated = seq.copy()
    for mut in muts:
        pos = int(mut[1:-1])-1
        if seq[pos] != mut[0]:
            print(" Curator warning : {}[{}] vs expected residue by mutation {}".format(seq[pos], pos, mut[0]))
        mutated[pos] = mut[-1]
    return mutated

if __name__ == '__main__':
    cmd_line = CommandHandler()
    Curator(cmd_line)