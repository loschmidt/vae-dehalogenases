"""
Preprocessing MSA for VAE
Remove unnecessary columns and encode it to the one-hot encoding
"""

__author__ = "Pavel Kohout <pavel.kohout@recetox.muni.cz>"
__date__ = "2024/08/12 11:30:00"

import itertools
import os.path
import pickle
from copy import deepcopy
from random import randint
from typing import Dict, List, Tuple

import numpy as np

# my libraries
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.utils import store_to_pkl


def weight_sequences(msa: np.array) -> np.array:
    """
    Get sequences weights by the formula:
        w{n}_j = 1/C_j x 1/C{n}_j

        where C_j is the number of unique amino acid types at the jth position of the MSA
        and C{n}_j is the number of sequences in the MSA that has the same amino acid
        type at the jth position as the nth sequence. Then the weight of the nth sequence
        is the sum of its position weights, i.e., w{n} = SUM(j=1 -> L) w{n}_j. Finally,
        the weights are renormalized as w{~n} = SUM(i=1 -> N) w{i} such that the sum of
        the normalized weights w{~n} is one.
    """
    seq_weight = np.zeros(msa.shape)
    for j in range(msa.shape[1]):
        aa_type, aa_counts = np.unique(msa[:, j], return_counts=True)
        num_type = len(aa_type)
        aa_dict = {}
        for a in aa_type:
            aa_dict[a] = aa_counts[list(aa_type).index(a)]
        for i in range(msa.shape[0]):
            seq_weight[i, j] = (1.0 / num_type) * (1.0 / aa_dict[msa[i, j]])
    tot_weight = np.sum(seq_weight)
    # Normalize weights of sequences
    seq_weight = seq_weight.sum(1) / tot_weight
    return seq_weight


class Preprocessor:
    def __init__(self, run: RunSetup):
        self.sequences_before = 0
        self.no_essentials = 0
        self.low_aa_count = 0
        self.msa_columns_removed = 0

        self.run = run
        self.pickles = run.pickles
        self.keep_keys = []
        self.aa_index = pickle.load(open(os.path.join(self.pickles, "aa_index.pkl"), "rb"))

    def trim_msa(self, msa_dict: Dict[str, str]):
        """
        Trim columns from MSA where no of the query/fixed sequences occurred in MSA
        :param msa_dict:
        :return: Trimmed msa dictionary numeric representation, msa_keys
        """
        if not isinstance(msa_dict, dict):
            print(" Trimming MSA error the parameter does not pass Dict!")
            exit(3)
        correct_names, msa_keys = True, list(msa_dict.keys())  # indices of found queries, and all keys
        keep_keys = list(set(self.run.fixed_sequences + [self.run.query]))
        self.keep_keys = keep_keys
        for fixed_seq in keep_keys:
            if fixed_seq not in msa_keys:
                print(f" Trimming Error: Fixed sequence {fixed_seq} not found in MSA!!!")
                correct_names = False
        if not correct_names:
            exit(1)

        # Get indices of occupied with any amino acids in some queries
        aa_pos_sets = [[i for i, s in enumerate(msa_dict[k]) if (s != "-" and s != ".")] for k in keep_keys]
        queries_aa_pos, keep_msa_pos = list(set(itertools.chain(*aa_pos_sets))), []
        queries_aa_pos.sort()
        queries_sequences = [[s for s in msa_dict[k] if (s != "-" and s != ".")] for k in keep_keys]
        gaped_queries_sequences = [msa_dict[k] for k in keep_keys]

        # Transfer sequences to numbers
        np_msa, msa_keys = self.filter_sequences_by_aa_and_transform_to_number(msa_dict)

        treemmer_excluded_pos_aa = {seq: [] for seq in keep_keys}

        enrichment = 0
        # Filter gap positions and calculate which position to keep
        for i in range(np_msa.shape[1]):
            if i in queries_aa_pos:  # Hold all positions as in queries
                if np.sum(np_msa[:, i] == 0) > (0.99 * np_msa.shape[0]):  # if in column is 80% of gaps
                    Preprocessor.exclude_pos_in_queries(treemmer_excluded_pos_aa,
                                                        i,
                                                        aa_pos_sets,
                                                        queries_sequences,
                                                        keep_keys)
                else:
                    keep_msa_pos.append(i)
            elif np.sum(np_msa[:, i] == 0) <= np_msa.shape[0] * 0.2:  # keep column with high frequently occurring AA
                keep_msa_pos.append(i)
                enrichment += 1
        store_to_pkl(treemmer_excluded_pos_aa, os.path.join(self.pickles, "queries_excluded_pos_and_aa.pkl"))

        print(" MSA trimming: \n"
              f"        Raw MSA width                                     : {np_msa.shape[1]}\n"
              f"        Number of positions occupied in treemmers         : {len(queries_aa_pos)}\n"
              f"        Sequences excluded due to non essential AA        : {self.no_essentials}\n"
              f"        Rich Amino Acids columns added into the train MSA : {enrichment}\n"
              f"        Query columns excluded having gaps:")
        for seq_name in treemmer_excluded_pos_aa.keys():
            for exclusion in treemmer_excluded_pos_aa[seq_name]:
                print(f"          {seq_name} pos: {exclusion[0]}, AA : {exclusion[1]} ")

        np_msa = np_msa[:, np.array(keep_msa_pos)]  # keep columns with aa in queries

        store_to_pkl(keep_msa_pos, os.path.join(self.pickles, "msa_columns.pkl"))

        # Filter sequences with >40% of gaps on query positions
        longer_sequence_indices = []

        for seq_i in range(np_msa.shape[0]):  # filter sequence having >40% gaps in positions
            if np.sum(np_msa[seq_i, :] == 0) < (0.4 * np_msa.shape[1]):
                longer_sequence_indices.append(seq_i)
        print("        MSA sequences excluded because of the gaps: {}")
        np_msa = np_msa[np.array(longer_sequence_indices), :]
        msa_keys = [k for i, k in enumerate(msa_keys) if i in longer_sequence_indices]
        return np_msa, msa_keys

    def filter_sequences_by_aa_and_transform_to_number(self, msa: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Remove everything with unexplored residues from dictionary
        Converts amino acid letter representation to numbers store in self.seq_msa
        Return ndarray of number converted sequences, key list
        """
        seq_msa, keys_list = [], []
        for k in msa.keys():
            if msa[k].count('X') > 0 or msa[k].count('Z') > 0:
                continue
            try:
                seq_msa.append([self.aa_index[s] for s in msa[k]])
                keys_list.append(k)
            except KeyError:
                self.no_essentials += 1
        return np.array(seq_msa), keys_list

    def get_keys_file_and_np_sequences(self, msa: Dict[str, List[np.array]], threshold: float = 0.5) -> np.array:
        """
        Prepare essential files for training
        Return np ndarray of a sequence encoded in numbers
        """
        seqs_to_np = []
        final_keys = []

        for k, seq in msa.items():
            seqs_to_np.append(seq)
            final_keys.append(k)

        store_to_pkl(self.keep_keys, os.path.join(self.pickles, "keep_keys.pkl"))
        store_to_pkl(final_keys, os.path.join(self.pickles, "keys_list.pkl"))
        print(" MSA preprocessing : Sequence keys and training dictionary stored")

        return np.array(seqs_to_np), msa

    def identity_filtering(self, msa):
        """ Cluster sequences by the 90 % identity """
        thresholds = np.linspace(0.2, self.run.identity, 4)
        clusters = []
        inputs = [msa]
        # Secure that query will stay in clustered MSA, and evo sequences too
        sampled_dict = {key: msa[key] for key in self.keep_keys}

        # Run through a threshold to decrease computational cost
        for th in thresholds:
            clusters = []
            for i, input in enumerate(inputs):
                ret_clusters = self.phylo_clustering(input, threshold=th)
                clusters.extend(ret_clusters)
            inputs = clusters
            print(' Identity filtering : Clustering with {} identity done'.format(th))

        for cluster in clusters:
            key, k_len = list(cluster.keys()), len(list(cluster.keys())) - 1
            random_key = key[randint(0, k_len)]
            sampled_dict[random_key] = cluster[random_key]
        return sampled_dict

    def phylo_clustering(self, msa, threshold=0.9):
        """ If a query sequence is given hold it in dataset and filter it by threshold identity """
        identity = lambda seq1, seq2: sum([1 if g == o else 0 for g, o in zip(seq1, seq2)]) / len(seq2)

        try:
            seed = msa[self.run.query]
        except KeyError:
            seed = (list(msa.values())[0])

        clusters = []

        while len(seed) != 0:
            cluster = {}
            prev_seed = []
            del_keys = []
            for k in msa.keys():
                if identity(msa[k], seed) > threshold:
                    del_keys.append(k)
                    cluster[k] = msa[k]
                else:
                    if len(prev_seed) == 0:  # First one without identity left for another iteration
                        prev_seed = msa[k]
            seed = prev_seed
            clusters.append(deepcopy(cluster))
            for k in del_keys:  # remove keys with
                del msa[k]
        return clusters

    @staticmethod
    def exclude_pos_in_queries(excl_dict: Dict[str, List[Tuple[int, str]]],
                               position: int,
                               queries_positions: List[List[int]],
                               queries_sequences: List[List[str]],
                               queries_names: List[str]):
        """
        Exclude position from all queries whose position is in queries positions
        :param excl_dict: dictionary to be updated
        :param position: position in the all queries positions!
        :param queries_positions: Positions of individual queries with some AA in original MSA
        :param queries_sequences: gap free queries sequences
        :param queries_names: names of the queries in right order
        :return: updated dictionary
        """
        for i, q in enumerate(queries_positions):
            if position in q:  # this ensures that we will insert only AA not a gaps
                no_gap_ind = q.index(position)
                excl_dict[queries_names[i]].append((no_gap_ind, queries_sequences[i][no_gap_ind]))
