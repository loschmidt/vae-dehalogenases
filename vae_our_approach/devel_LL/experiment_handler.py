__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/06/10 11:05:00"

import csv
import numpy as np

from vae_our_approach.devel_LL.VAE_accessor import VAEAccessor
from benchmark import Benchmarker as ProbabilityMaker
from sequence_transformer import Transformer


class ExperimentStatistics:
    """
    The purpose of this class is to propose methods for saving the results of experiment
    in united form of csv file and the others format as they are needed.

    Support the creation of fasta format files and so on.
    """

    def __init__(self, setuper, experiment_name="BasicExperiment"):
        self.pickles = setuper.pickles_fld
        self.high_fld = setuper.high_fld
        self.exp_name = experiment_name + "_"

        self.transformer = Transformer(setuper)
        self.vae_handler = VAEAccessor(setuper, model_name=setuper.model_name)
        self.probability_maker = ProbabilityMaker(None, None, setuper, generate_negative=False)

        self.logged_items = 0
        self.log_msg = ""
        self.query_name = setuper.query_id

    def set_experiment_name(self, experiment_name):
        self.exp_name = experiment_name

    def store_ancestor_dict_in_fasta(self, seq_dict: dict, file_name, msg="storing ancestors into "):
        """ Method store sequence dictionary in fasta file named expName_file_name.fasta """
        file_name = self.exp_name + file_name
        with open(self.high_fld + "/" + file_name, 'w') as file_handle:
            for key, sequence in seq_dict.items():
                file_handle.write(">" + key + "\n" + "".join(sequence) + "\n")
        self._log_msg(msg + self.high_fld + "/" + file_name)

    def residues_likelihood_above_threshold(self, seqs_dict, threshold=0.9):
        """
        Compute likelihood of each residue in sequence
        and count those having probability above threshold
        """
        binaries = self.transformer.sequences_dict_to_binary(seqs_dict)
        p_seqs_residues = self.vae_handler.residues_probability(binaries)
        sequences_res_above = np.zeros(p_seqs_residues.shape[0], dtype=int)
        for i, seq_res_p in enumerate(p_seqs_residues):
            sequences_res_above[i] += (seq_res_p > threshold).sum()
        return sequences_res_above

    def create_and_store_ancestor_statistics(self, seq_dict_to_store, file_name, coords):
        """
        The method creates statistics for csv file which is main output for all protocols.
        It requires out of generated sequence dictionary also coordinates in the order of
        sequences in the dictionary.

        Returns observing probabilities
        """
        observing_probs = self.probability_maker.measure_seq_probability(seq_dict_to_store)
        names = list(seq_dict_to_store.keys())
        sequences = list(seq_dict_to_store.values())
        self.store_ancestor_dict_in_fasta(seq_dict=seq_dict_to_store, file_name=file_name)

        # Measuring the identity with Query decoded from binary, extend query by excluded residues
        query_dict = self.transformer.get_seq_dict_by_key(self.query_name)
        query_seq = self.transformer.add_excluded_query_residues(query_dict[self.query_name])
        self._log_msg(msg="The identity statistics are done against sequence with ID " + self.query_name)

        # Residues probabilities above 0.9 counting
        threshold = 0.9
        residues_prob_above = self.residues_likelihood_above_threshold(seq_dict_to_store, threshold=threshold)
        col_res_prob_name = "Residues count of probabilities above {}".format(threshold)

        # Find closest sequence in term of edit distance in the input dataset
        closest_sequences = self.vae_handler.get_identity_closest_dataset_sequence(sequences)

        # Store in csv file
        file_name = self.high_fld + '/{0}_probabilities_ancs.csv'.format(file_name.split('.')[0])
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Number", "Ancestor", "Sequences", "Probability of observation", "Coordinate x",
                             "Coordinate y", "Query identity [%]", col_res_prob_name, "Closest ID",
                             "Closest identity [%]", "Count of indels",
                             "Indels", "Count of substitutions", "Substitutions in WT {}".format(self.query_name)])
            for i, (name, seq, prob, c, res_p, close) in enumerate(zip(names, sequences, observing_probs, coords,
                                                                       residues_prob_above, closest_sequences)):
                seq_str = ''
                extended_seq = self.transformer.add_excluded_query_residues(seq_str.join(seq))
                query_iden = ExperimentStatistics.sequence_identity(extended_seq, query_seq)
                subs_list, indels = ExperimentStatistics.query_indels_and_substitution(extended_seq, query_seq)
                writer.writerow([i, name, extended_seq, prob, c[0], c[1], query_iden, res_p,
                                 close[0], close[1], len(indels), ", ".join(indels),
                                 len(subs_list), ", ".join(subs_list)])
        self._log_msg(msg="The CSV stats with profile were created into " + file_name)
        return observing_probs

    def _log_msg(self, msg):
        """ Logger collects all file accesses and prints it out at the end of program run """

        def add_record(record: str):
            self.log_msg += "#" + record + "\n"

        if self.logged_items == 0:
            add_record("#" * 99)
            add_record("   ExperimentStatistics log")
        add_record(" " + msg)
        self.logged_items += 1

    def __del__(self):
        """ On destruction print the log"""
        print(self.log_msg)

    @staticmethod
    def query_indels_and_substitution(seq, query):
        """ Determines how many indels were added to WT and these positions keep in list """
        indels_list = []
        substitution_list = []
        gaps_vec = [p != '-' for p in query]
        wt_positions = [sum(gaps_vec[:prefix + 1]) for prefix in range(len(query))]
        for pos, (seq_char, query_char) in enumerate(zip(seq, query)):
            if seq_char != query_char:
                if query_char == '-':
                    indels_list.append("ins{}{}".format(wt_positions[pos], seq_char))
                elif seq_char == '-':
                    indels_list.append("del{}{}".format(wt_positions[pos], query_char))
                else:  # Substitution occurs
                    substitution_list.append("{}{}{}".format(query_char, wt_positions[pos], seq_char))
        return substitution_list, indels_list

    @staticmethod
    def sequence_identity(source, target):
        """
        Returns the sequence percentage identity. Input sequences are lists of amino acids
        Example:
             source = [A G G E S - D T W A]
             target = [A G G E - D T W A A]
             returns: 50.00
        """
        identical_residues = sum([1 if i == j else 0 for i, j in zip(source, target)])
        identical_fraction = identical_residues / len(target)
        return  identical_fraction * 100