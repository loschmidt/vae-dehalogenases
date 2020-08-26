__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/10 11:30:00"

from download_MSA import Downloader
from msa_prepar import MSA
from pipeline import StructChecker

import numpy as np
import pickle

class MSAFilterCutOff :
    def __init__(self, setuper):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld

    def proc_msa(self):
        msa_obj = MSA(setuper=self.setuper, processMSA=False)
        msa = msa_obj.load_msa()
        self.aa, self.aa_index = msa_obj.amino_acid_dict(export=True)
        msa_col_num = self._remove_cols_with_gaps(msa) ## converted to numbers
        msa_no_gaps = self._remove_seqs_with_gaps(msa_col_num)
        msa_overlap, self.key_list = self._get_seqs_overlap_ref(msa_no_gaps)
        self.seq_weight = self._weighting_sequences(msa_overlap)
        self.seq_msa_binary = self._to_binary(msa_overlap)
        self._stats(msa_overlap)

    def _remove_cols_with_gaps(self, msa, threshold=0.2, keep_ref=False):
        '''Remove positions with too many gaps. Set threshold and
           keeping ref seq position is up to you
           Returns modified dictionary with translate sequences AA to numbers'''
        pos_idx = []
        ref_pos = []
        if keep_ref:
            ref_pos = self._get_ref_pos(msa)
        np_msa, key_list = self._remove_unexplored_and_covert_aa(msa)
        for i in range(np_msa.shape[1]):
            if i not in ref_pos:
                if np.sum(np_msa[:, i] == 0) <= np_msa.shape[0] * threshold:
                    pos_idx.append(i)
        with open(self.pickle + "/seq_pos_idx.pkl", 'wb') as file_handle:
            pickle.dump(pos_idx, file_handle)
        np_msa = np_msa[:, np.array(pos_idx)]
        msa_dict = {}
        ## Set to dict modified sequences
        for i in range(np_msa.shape[0]):
            msa_dict[key_list[i]] = np_msa[i] ## without columns with many gaps
        return msa_dict

    def _get_ref_pos(self, msa):
        """Returns position of reference sequence with no gaps"""
        if not self.setuper.ref_seq:
            return []
        query_seq = msa[self.setuper.ref_n]  ## with gaps
        idx = []
        gaps = [s == "-" or s == "." for s in query_seq]
        for i, gap in enumerate(gaps):
            if not gap:
                idx.append(i)
        return idx

    def _remove_seqs_with_gaps(self, msa, threshold=0.2):
        '''remove sequences with too many gaps'''
        seq_id = list(msa.keys())
        seq_len = len(list(msa.values())[0]) ## All seqs same len, select first
        for k in seq_id:
            unique, counts = np.unique(msa[k], return_counts=True)
            cnt_gaps = dict(zip(unique, counts))
            try:
                if cnt_gaps[0] > threshold * seq_len:
                    msa.pop(k)
            except KeyError:
                ## No gaps in that sequence, keep it in
                pass
        return msa

    def _get_seqs_overlap_ref(self, msa, threshold=0.5):
        '''Left only sequence having threshold overlap with reference seq
            Default 50 % overlap with query sequence'''
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
        for key_idx, seq in enumerate(trans_arr):
            overlap = 0
            for i in range(len_seq):
                ## Simply gaps or anything else
                if (ref_seq[i] == 0 and seq[i] == 0) or (ref_seq[i] != 0 and seq[i] != 0):
                    overlap += 1
            if overlap >= threshold * len_seq:
                overlap_seqs.append(seq)
                final_keys.append(key_list[key_idx])
        with open(self.pickle + "/keys_list.pkl", 'wb') as file_handle:
            pickle.dump(final_keys, file_handle)
        return np.array(overlap_seqs), final_keys

    def _remove_unexplored_and_covert_aa(self, msa):
        """
            Remove everything with unexplored residues from dictionary
            Converts amino acid letter representation to numbers store in self.seq_msa
        """
        seq_msa = []
        keys_list = []
        for k in msa.keys():
            if msa[k].count('X') > 0 or msa[k].count('Z') > 0:
                continue
            seq_msa.append([self.aa_index[s] for s in msa[k]])
            keys_list.append(k)
        return np.array(seq_msa), keys_list

    def _weighting_sequences(self, msa):
        ## reweighting sequences
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
        with open(self.pickle + "/seq_weight.pkl", 'wb') as file_handle:
            pickle.dump(seq_weight, file_handle)
        return seq_weight

    def _to_binary(self, msa):
        K = len(self.aa)+1  ## num of classes of aa
        D = np.identity(K)
        num_seq = msa.shape[0]
        len_seq_msa = msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[msa[i]]
        with open(self.pickle + "/seq_msa_binary.pkl", 'wb') as file_handle:
            pickle.dump(seq_msa_binary, file_handle)
        return seq_msa_binary

    def _stats(self, msa):
        if self.setuper.stats:
            name = self.setuper.ref_n if self.setuper.ref_seq else 'No reference sequence'
            print("=" * 60)
            print("Pfam ID : {0} - {2}, Reference sequence {1}".format(self.setuper.pfam_id, name, self.setuper.rp))
            print("Sequences used: {0}".format(msa.shape[0]))
            print("Max lenght of sequence: {0}".format(max([len(i) for i in msa])))
            print("=" * 60)
            print()

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = MSAFilterCutOff(tar_dir)
    msa.proc_msa()