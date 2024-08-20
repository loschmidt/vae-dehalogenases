"""
Profiler class for Vaencestors
This class creates final profile for given ancestral sequences
"""

__author__ = "Pavel Kohout <pavel.kohout@recetox.muni.cz>"
__date__ = "2024/08/14 11:05:00"

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch

from notebooks.minimal_version.latent_space import LatentSpace
from notebooks.minimal_version.msa import MSA
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.utils import sequence_identity, load_from_pkl, reshape_binary, store_to_fasta, \
    get_binaries_by_key


# My libraries



def get_identity_closest_dataset_sequence(dataset: Dict[str, str], sequences: List[str]):
    """
    Method searches for the most identical sequence in the input dataset.
    :param dataset: Dataset in which we search for the closest sequence
    :param sequences: list of chars representing amino acid sequence level
    Returns:
        List of tuples [(id, identity),...]
        Id - key of nearest sequence
        identity - percentage sequence identity with the corresponding sequence in sequences
    Note that the order in list corresponds to the order in sequences
    """

    def closest_seq_identity(seq, dataset):
        max_identity, id_i = 0.0, 0
        for i, d in enumerate(dataset):
            seq_identity = sequence_identity(d, seq)
            if seq_identity > max_identity:
                max_identity = seq_identity
                id_i = i
        return id_i, max_identity

    dataset_sequences = list(dataset.values())
    dataset_keys = list(dataset.keys())

    closest_sequences = []
    for iteration, sequence in enumerate(sequences):
        i_key, closest_identity = closest_seq_identity(sequence, dataset_sequences)
        closest_sequences.append((dataset_keys[i_key], closest_identity))
    return closest_sequences


def query_indels_and_substitution(seq: str, query: str) -> Tuple[List[str], List[str]]:
    """
    Find the position in sequence where insertion, deletion or substitution happened against query in MSA
    :param seq: sequence to be examined
    :param query: sequence against which the metrics are calculated (MSA query typically)
    :return: tuple of lists containing substitutions and indels in formatted string
    """
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


class Profiler:
    """
    The purpose of this class is to propose methods for saving the results of experiment
    in united form of csv file and the others format as they are needed.

    Support the creation of fasta format files and so on.
    """

    def __init__(self, run: RunSetup):
        self.pickles = run.pickles
        self.results = run.results
        self.latent = LatentSpace(run)
        self.run = run
        try:
            self.queries_excluded = load_from_pkl(os.path.join(run.pickles, "queries_excluded_pos_and_aa.pkl"))
        except FileNotFoundError:  # back compatibility with a simple query
            print("Could not find queries_excluded_pos_and_aa using one query file")
            self.queries_excluded = {run.evo_query: load_from_pkl(os.path.join(run.pickles,
                                                                               "query_excluded_pos_and_aa.pkl"))}

        self.evo_query = None
        self.excluded_query_pos_and_aa = None
        self.set_evo_query(run.evo_query)

    def set_evo_query(self, evo_query: str):
        """
        Set the evo_query to the sequence specified via parameter, also set excluded_query_pos_and_aa to those values
        :param evo_query: name of the evo_query, must be in the treemmers set in conf_path file
        :return:
        """
        try:
            self.excluded_query_pos_and_aa = self.queries_excluded[evo_query]
            self.evo_query = evo_query
        except KeyError:
            print(f" Profiler error: {evo_query} not in the treemmers configuration file")
            exit(1)

    def residues_likelihood_above_threshold(self, seq_dict: Dict[str, str], threshold: float = 0.9):
        """
        Count residues above a given threshold in sequences
        :param seq_dict: dictionary with sequences
        :param threshold: probability threshold value
        :return:
        """
        # Get binary representation for our proteins
        number_sequences = MSA.aa_to_number(seq_dict)
        binary = MSA.number_to_binary(number_sequences)
        binary = reshape_binary(binary)
        sequences_residue_probs = self.latent.residues_probability(binary)
        sequences_res_above = np.zeros(sequences_residue_probs.shape[0], dtype=int)
        for i, sequence_residue_probs in enumerate(sequences_residue_probs):
            sequences_res_above[i] += (sequence_residue_probs > threshold).sum()
        return sequences_res_above

    def log_ratio(self, seq_dict: Dict[str, str], msa_name: str) -> np.ndarray:
        """
        Get loglikelihood ratio with the query sequence for sequences within the dictionary.
        :param seq_dict: dictionary with sequences to be evaluated
        :param msa_name: sequence against which we get the ratio, should be in the input dataset
        :return: np array with log_ans / log_evo_query probabilities
        """
        # Get binary representation for our proteins
        number_sequences = MSA.aa_to_number(seq_dict)
        binary = MSA.number_to_binary(number_sequences)
        binary = reshape_binary(binary)
        ancestor_logs = self.latent.sequence_probabilities(binary)

        # Get original query for evolution log likelihood
        msa_keys = load_from_pkl(os.path.join(self.pickles, "keys_list.pkl"))
        msa_binary = load_from_pkl(os.path.join(self.pickles, "seq_msa_binary.pkl"))
        evo_query_binary = get_binaries_by_key([msa_name], msa_binary, msa_keys)
        evo_query_logs = self.latent.sequence_probabilities(torch.Tensor(evo_query_binary))
        # Return the ratio between the sequence
        return ancestor_logs / evo_query_logs

    def profile_sequences(self, ancestor_dict, file_name: str, coords: np.ndarray):
        """
        The method creates statistics for csv file which is the main output for all protocols.
        It requires out of generated sequence dictionary also coordinates in the order of
        sequences in the dictionary.
        :param ancestor_dict: generated ancestral sequences to be evaluated
        :param file_name: name the file/experiment where to store
        :param coords:  latent space coordinates of the generated ancestors
        :return:
        """
        logs = self.log_ratio(ancestor_dict, self.evo_query)
        names = list(ancestor_dict.keys())
        sequences = list(ancestor_dict.values())
        raw_vae_msa_path = os.path.join(self.run.results, f"raw_msa_{file_name}")
        store_to_fasta(ancestor_dict, raw_vae_msa_path)
        print(f" Profiler storing raw VAE ancestor results into {raw_vae_msa_path}")

        # Measuring the identity with Query
        processed_msa = MSA.load_msa(os.path.join(self.run.msa, 'training_msa.fasta'))
        extended_query = self.add_excluded_query_residues(processed_msa[self.run.query], processed_msa[self.run.query])
        identities_to_query = [sequence_identity(s, processed_msa[self.run.query]) for s in sequences]

        # Measuring the identity with Evo
        identities_to_query = [sequence_identity(s, processed_msa[self.evo_query]) for s in sequences]

        # Residue probabilities above 0.9 counting
        threshold = 0.9
        residues_prob_above = self.residues_likelihood_above_threshold(ancestor_dict, threshold=threshold)

        # Find the latent space spatially the closest sequences
        spatial_closest = self.latent.get_closest_dataset_sequence(coords)
        # Find sequentially the closest sequences
        closest_sequences = get_identity_closest_dataset_sequence(processed_msa, sequences)
        # Clear sequences from gaps and add residues
        extended_sequences = [self.add_excluded_query_residues(s, processed_msa[self.evo_query]) for s in sequences]
        no_gap_sequences = [s.replace("-", "") for s in extended_sequences]
        # Get substitutions and Indels to the MSA query
        subs_list, indels_list = [], []
        for s in extended_sequences:
            subs, indels = query_indels_and_substitution(s, extended_query)
            subs_list.append(subs)
            indels_list.append(indels)
        df = pd.DataFrame({
            'Ancestor': names,
            'Sequences': no_gap_sequences,
            'log(x_mut)/log(evo_query)': np.round(logs, decimals=3),
            'Latent Coordinates': [f"{coord[0]}, {coord[1]}" for coord in np.round(coords, decimals=3)],
            'Spatial latent closest ID': spatial_closest,
            'Sequence closest ID': closest_sequences,
            f'Identity to {self.run.query}': identities_to_query,
            f'Identity to {self.evo_query}': identities_to_query,
            'Count of substitutions': [len(subs) for subs in subs_list],
            'Count of indels': [len(indels) for indels in indels_list],
            f'Substitutions in {self.run.query}': [", ".join(subs) for subs in subs_list],
            f'Insertions and deletions in {self.run.query}': [", ".join(indels) for indels in indels_list]
        })

        # Store gap free ancestors in fasta file
        gap_free_ancestors = {k: s for k, s in zip(names, no_gap_sequences)}
        store_to_fasta(gap_free_ancestors, os.path.join(self.run.results, file_name))
        print(f" The ancestor sequences are stored in {os.path.join(self.run.results, file_name)}")

        # Create the new file path with the custom extension
        file_without_extension = os.path.splitext(file_name)[0]
        new_filename = 'profile_' + file_without_extension + '.csv'
        csv_file = os.path.join(self.run.results, new_filename)
        df.to_csv(csv_file, index=False)
        print(f" The ancestor profile csv stored in : {csv_file}")

    def add_excluded_query_residues(self, seq_to_add_in, query_sequence):
        """
        MSA preprocessing protocol excludes some MSA columns as they are not rich in information, now we have to add them
        :param seq_to_add_in: sequence to add exclude amino acids in
        :param query_sequence: sequence of the query as it is in preprocessed MSA
        :return:
        """
        aa_sequence = "".join(seq_to_add_in)
        query_sequence = "".join(query_sequence)

        inserted = 0
        for position, aa in self.excluded_query_pos_and_aa:
            seq_aa_position = 0
            for aa_ind in range(len(query_sequence)):
                if seq_aa_position == position:
                    offset = aa_ind
                    aa_sequence = aa_sequence[:offset] + aa + aa_sequence[offset:]
                    query_sequence = query_sequence[:offset] + aa + query_sequence[offset:]
                    break
                if query_sequence[aa_ind] != "-":
                    seq_aa_position += 1
            inserted += 1
        return aa_sequence
