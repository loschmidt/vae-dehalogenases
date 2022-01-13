__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/24 11:30:00"

from download_MSA import Downloader
from copy import deepcopy
from msa_preparation import MSA
from random import randint
from parser_handler import CmdHandler

import numpy as np
import pickle
import pandas as pd


class MSAPreprocessor:
    def __init__(self, setuper: CmdHandler):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.aa_index = MSA.amino_acid_dict(self.pickle)

    def proc_msa(self):
        msa = MSA.load_msa(self.setuper.msa_file)
        msa_col_num = self._remove_cols_with_gaps(msa)  # converted to numbers
        msa_no_gaps = self._remove_seqs_with_gaps(msa_col_num, threshold=0.4)
        self._save_reference_sequence(msa_no_gaps)
        msa_filtered = self.identity_filtering(msa_no_gaps)
        msa_overlap, self.keys_list = self._get_seqs_overlap_ref(msa_filtered)
        self.seq_weight = self._weighting_sequences(msa_overlap)
        self.seq_msa_binary = MSA.number_to_binary(msa_overlap, self.pickle)
        self._stats(msa_overlap)

    def prepare_aligned_msa_for_Vae(self, msa):
        """ Prepares MSA for VAE, msa aligned """
        seqs, keys = self._remove_unexplored_and_covert_aa(msa)
        dataframe = pd.DataFrame(seqs.tolist(), dtype=int)
        seqs = np.array(dataframe.fillna(0).values, dtype=int)
        weights = self._weighting_sequences(msa=seqs, gen_pickle=False)
        binary = MSA.number_to_binary(seqs, self.pickle, gen_pickle=False)
        return binary, weights, keys

    def _save_reference_sequence(self, msa):
        if not self.setuper.ref_seq:
            return None
        ref_seq = {}
        ref_seq[self.setuper.ref_n] = msa[self.setuper.ref_n]
        ref_seq = MSA.number_to_amino(ref_seq)
        with open(self.pickle + "/reference_seq.pkl", 'wb') as file_handle:
            pickle.dump(ref_seq, file_handle)

    def _remove_cols_with_gaps(self, msa, threshold=0.2):
        """
        Remove positions with too many gaps. All columns with query no gap position are held.
        The columns with proportion of gaps lower than threshold * num_msa_seqs will be held
        even if there is gap in the query.
        Returns modified dictionary with translate sequences AA to numbers
        """
        pos_idx = []
        ref_pos = self._get_ref_pos(msa)
        np_msa, key_list = self._remove_unexplored_and_covert_aa(msa)
        for i in range(np_msa.shape[1]):
            # Hold all position as in reference sequence
            if i in ref_pos:
                pos_idx.append(i)
            elif np.sum(np_msa[:, i] == 0) <= np_msa.shape[0] * threshold:
                pos_idx.append(i)
        print('MSA_filter message : The MSA is cleared by gaps columns. Width: {}, \n'
              '                     added gaps columns by threshold: {}'.format(len(pos_idx),
                                                                                len(pos_idx) - len(ref_pos)))
        with open(self.pickle + "/seq_pos_idx.pkl", 'wb') as file_handle:
            pickle.dump(pos_idx, file_handle)
        np_msa = np_msa[:, np.array(pos_idx)]
        msa_dict = {}
        # Set to dict modified sequences
        for i in range(np_msa.shape[0]):
            msa_dict[key_list[i]] = np_msa[i]  # without columns with many gaps
        return msa_dict

    def _get_ref_pos(self, msa):
        """ Returns  no gap position indexes of reference sequence """
        if not self.setuper.ref_seq:
            return []

        query_seq = msa[self.setuper.ref_n]  # with gaps
        gaps = [s == "-" or s == "." for s in query_seq]
        print('MSA_filter message : Do not search for regions in the core')
        idx = []
        for i, gap in enumerate(gaps):
            if not gap:
                idx.append(i)
        return idx

    def _remove_seqs_with_gaps(self, msa, threshold=0.2):
        """ Remove sequences with too many gaps """
        seq_id = list(msa.keys())
        seq_len = len(list(msa.values())[0])  # All seqs same len, select first
        for k in seq_id:
            if self.setuper.ref_seq and self.setuper.ref_n == k:
                continue  # keep query sequence in msa
            unique, counts = np.unique(msa[k], return_counts=True)
            cnt_gaps = dict(zip(unique, counts))
            try:
                if cnt_gaps[0] > threshold * seq_len:
                    msa.pop(k)
            except KeyError:
                # No gaps in that sequence, keep it in
                pass
        return msa

    def _get_seqs_overlap_ref(self, msa, threshold=0.5):
        """
        Left only sequence having threshold overlap with reference seq.
        Default 50 % overlap with query sequence.
        """
        trans_arr = []
        key_list = []
        ii = 0
        for k in msa.keys():
            trans_arr.append(list(msa[k]))
            key_list.append(k)
        if not self.setuper.ref_seq:
            with open(self.pickle + "/keys_list.pkl", 'wb') as file_handle:
                pickle.dump(key_list, file_handle)
            return np.array(trans_arr), key_list
        ref_seq = msa[self.setuper.ref_n]
        len_seq = len(ref_seq)
        overlap_seqs = []
        final_keys = []
        training_alignment = {}
        for key_idx, seq in enumerate(trans_arr):
            overlap = 0
            for i in range(len_seq):
                # Simply gaps or anything else
                if (ref_seq[i] == 0 and seq[i] == 0) or (ref_seq[i] != 0 and seq[i] != 0):
                    overlap += 1
            if overlap >= threshold * len_seq:
                overlap_seqs.append(seq)
                final_keys.append(key_list[key_idx])
                training_alignment[key_list[key_idx]] = seq
        with open(self.pickle + "/keys_list.pkl", 'wb') as file_handle:
            pickle.dump(final_keys, file_handle)
        training_converted = MSA.number_to_amino(training_alignment)
        with open(self.pickle + "/training_alignment.pkl", 'wb') as file_handle:
            pickle.dump(training_converted, file_handle)
        return np.array(overlap_seqs), final_keys

    def _remove_unexplored_and_covert_aa(self, msa):
        """
            Remove everything with unexplored residues from dictionary
            Converts amino acid letter representation to numbers store in self.seq_msa
        """
        seq_msa = []
        keys_list = []
        no_essential = 0
        for k in msa.keys():
            if msa[k].count('X') > 0 or msa[k].count('Z') > 0:
                continue
            try:
                seq_msa.append([self.aa_index[s] for s in msa[k]])
                keys_list.append(k)
            except KeyError:
                no_essential += 1
        self.no_essentials = no_essential
        return np.array(seq_msa), keys_list

    def _weighting_sequences(self, msa, gen_pickle=True):
        # reweighting sequences
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
        ## Normalize weights of sequences
        seq_weight = seq_weight.sum(1) / tot_weight
        if gen_pickle:
            with open(self.pickle + "/seq_weight.pkl", 'wb') as file_handle:
                pickle.dump(seq_weight, file_handle)
        return seq_weight

    def identity_filtering(self, msa):
        ''' Cluster sequences by the 90 % identity'''
        thresholds = [0.2, 0.4, 0.6, 0.8, 0.9]
        clusters = []
        inputs = [msa]
        # Secure that query will stay in clustered MSA
        sampled_dict = {} if not self.setuper.ref_seq \
            else {self.setuper.ref_n: msa[self.setuper.ref_n]}

        # Run through threshold to decrease computational cost
        for th in thresholds:
            clusters = []
            for i, input in enumerate(inputs):
                ret_clusters = self.phylo_clustering(input, threshold=th)
                clusters.extend(ret_clusters)
                if (i + 1) % max(50, len(inputs) // 10) == 0:
                    print("\tClusters proceeded {}/{}".format(i + 1, len(inputs)), flush=True)
            inputs = clusters
            print('MSA_filter message : Clustering with identity {} done, number of clusters {}'.format(th,
                                                                                                        len(clusters)))

        for cluster in clusters:
            key, k_len = list(cluster.keys()), len(list(cluster.keys())) - 1
            random_key = key[randint(0, k_len)]
            sampled_dict[random_key] = cluster[random_key]
        if self.setuper.stats:
            print("=" * 60)
            print("Statistics for clustering")
            print("\t Cluster: {}".format(len(clusters)))

        return sampled_dict

    def phylo_clustering(self, msa, threshold=0.9):
        """ If query sequence is given hold it in dataset and filter it by threshold identity """
        identity = lambda seq1, seq2: sum([1 if g == o else 0 for g, o in zip(seq1, seq2)]) / len(seq2)

        if self.setuper.ref_seq:
            try:
                seed = msa[self.setuper.ref_n]
            except KeyError:
                seed = (list(msa.values())[0])
        else:
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
                    if len(prev_seed) == 0:  ## First one without identity left for another iteration
                        prev_seed = msa[k]
            seed = prev_seed
            clusters.append(deepcopy(cluster))
            for k in del_keys:
                del msa[k]

        return clusters

    def _stats(self, msa):
        if self.setuper.stats:
            name = self.setuper.ref_n if self.setuper.ref_seq else 'No reference sequence'
            print("=" * 60)
            print("Pfam ID : {0} - {2}, Reference sequence {1}".format(self.setuper.exp_dir, name, self.setuper.rp))
            print("Sequences used: {0}".format(msa.shape[0]))
            print("Max lenght of sequence: {0}".format(max([len(i) for i in msa])))
            print('Sequences removed due to nnoessential AA: ', self.no_essentials)
            print("=" * 60)
            print()


if __name__ == '__main__':
    tar_dir = CmdHandler()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = MSAPreprocessor(tar_dir)
    msa.proc_msa()
