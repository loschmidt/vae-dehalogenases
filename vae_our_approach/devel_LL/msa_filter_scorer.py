__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/08/24 11:30:00"

from download_MSA import Downloader
from msa_prepar import MSA
from pipeline import StructChecker

import numpy as np
import pickle
import pandas as pd

class MSAFilterCutOff :
    def __init__(self, setuper):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.msa_obj = MSA(setuper=self.setuper, processMSA=False)
        self.aa, self.aa_index = self.msa_obj.amino_acid_dict(export=True)

    def proc_msa(self):
        msa = self.msa_obj.load_msa()
        msa_col_num = self._remove_cols_with_gaps(msa, keep_ref=True) ## converted to numbers
        msa_no_gaps = self._remove_seqs_with_gaps(msa_col_num, threshold=0.4)
        msa_overlap, self.keys_list = self._get_seqs_overlap_ref(msa_no_gaps)
        self._save_reference_sequence(msa_no_gaps)
        self.seq_weight = self._weighting_sequences(msa_overlap)
        self.seq_msa_binary = self._to_binary(msa_overlap)
        self._stats(msa_overlap)

    def prepare_aligned_msa_for_Vae(self, msa):
        '''Prepares MSA for VAE, msa aligned'''
        seqs, keys = self._remove_unexplored_and_covert_aa(msa)
        dataframe = pd.DataFrame(seqs.tolist(), dtype=int)
        seqs = np.array(dataframe.fillna(0).values, dtype=int)
        weights = self._weighting_sequences(msa=seqs, gen_pickle=False)
        binary = self._to_binary(seqs, gen_pickle=False)
        return binary, weights, keys

    def _save_reference_sequence(self, msa):
        if not self.setuper.ref_seq:
            return None
        ref_seq = {}
        ref_seq[self.setuper.ref_n] = msa[self.setuper.ref_n]
        ref_seq = self.back_to_amino(ref_seq)
        with open(self.pickle + "/reference_seq.pkl", 'wb') as file_handle:
            pickle.dump(ref_seq, file_handle)

    def _remove_cols_with_gaps(self, msa, threshold=0.2, keep_ref=False):
        '''Remove positions with too many gaps. Set threshold and
           keeping ref seq position is up to you
           Returns modified dictionary with translate sequences AA to numbers'''
        pos_idx = []
        ref_pos = []
        if keep_ref:
            ref_pos = self._get_ref_pos(msa, search_for_regions=False)
        np_msa, key_list = self._remove_unexplored_and_covert_aa(msa)
        for i in range(np_msa.shape[1]):
            # Hold all position as in reference sequence
            if i in ref_pos:
                pos_idx.append(i)
            elif np.sum(np_msa[:, i] == 0) <= np_msa.shape[0] * threshold:
                pos_idx.append(i)
        with open(self.pickle + "/seq_pos_idx.pkl", 'wb') as file_handle:
            pickle.dump(pos_idx, file_handle)
        np_msa = np_msa[:, np.array(pos_idx)]
        msa_dict = {}
        ## Set to dict modified sequences
        for i in range(np_msa.shape[0]):
            msa_dict[key_list[i]] = np_msa[i] ## without columns with many gaps
        return msa_dict

    def _get_ref_pos(self, msa, search_for_regions=True):
        """Returns position of reference sequence with no gaps
           Cut gaps at the beginning and at the end. Gaps inside keep
           -----------[--AAAAA-AA-A-A-A---A--]------------
                    begin    keep this      end
                     of core of sequence
            Search for regions makes the reduction of width of the core
            by iterative run through regions in the core
        """
        if not self.setuper.ref_seq:
            return []

        def get_gaps_regions(gaps_arr, offset):
            prev_pos = False
            regions = []
            start_pos = 0
            for i, gap in enumerate(gaps_arr, offset):
                if not prev_pos and gap:
                    # Start of gap region
                    start_pos = i
                elif prev_pos and not gap:
                    # End of gap region, modify end index
                    regions.append((start_pos, i-1))
                prev_pos = gap
            # Correct last one region
            if start_pos != 0:
                regions.append((start_pos, offset+len(gaps_arr)-1))
            return regions

        query_seq = msa[self.setuper.ref_n]  ## with gaps
        gaps = [s == "-" or s == "." for s in query_seq]
        # first and last position of AA. Keep 2 gaps on each side.
        begin_end = [[i for i, gap in enumerate(gaps) if not gap][ii]-2-(4*ii) for ii in (0, -1)]
        # Just find the core sequence and make column reduction by threshold process
        if not search_for_regions:
            print('MSA_filter message : Do not search for regions in the core')
            return list(range(begin_end[0], begin_end[1]))

        # Now search for regions in the core and reduce them
        print('MSA_filter message : Do search for regions in the core')
        core_protein = gaps[begin_end[0]: begin_end[1]+1]
        # Search for big gap regions in the core
        max_region_size = len(core_protein)*0.4
        # Gaps region and its positions
        offset = begin_end[0]
        regions = get_gaps_regions(core_protein, offset)
        # Get maximum region and select which part of alignment will be used
        region_sizes = list(map(lambda e: e[1]-e[0], regions))
        idx_max, max_val = max(enumerate(region_sizes), key=lambda e: e[1])
        # Split up to max region only if gaps region is really big
        # To avoid sequences like AAAAA---------------A
        if max_val > max_region_size:
            fst_reg = core_protein[0:regions[idx_max][0]]
            snd_reg = core_protein[regions[idx_max][1]:]
            # Use that region with more information (amino acids)
            begin_end = [offset, regions[idx_max][0]+2] if fst_reg.count(False) > snd_reg.count(False) else [regions[idx_max][1]-2, begin_end[1]]
            # Core of sequence is modified, modify regions too
            regions = get_gaps_regions(gaps[begin_end[0]: begin_end[1] + 1], offset=begin_end[0])
            region_sizes = list(map(lambda e: e[1] - e[0], regions))
        # Amino acid positions in selected region will be hold
        idx = []
        # Get aa and their position get to idx
        for i, gap in enumerate(gaps[begin_end[0]: begin_end[1]+1], begin_end[0]):
            if not gap:
                idx.append(i)

        # Region is selected modify it to shorter version cca. 450 aa
        # Remove regions form the biggest until the length is sufficient
        actual_size = begin_end[1] - begin_end[0]
        while actual_size > 450:
            # Request for core sequence length is too small, all gap regions were remove break the loop
            if len(region_sizes) == 0:
                break
            max_region_i, max_region_value = max(enumerate(region_sizes), key=lambda e: e[1])
            del region_sizes[max_region_i]
            del regions[max_region_i]
            actual_size -= max_region_value+1 # to get correct length of region
        # Include positions of kept gaps into idx
        for r in regions:
            idx.extend(list(range(r[0], r[1]+1)))
        idx.sort()
        return idx

    def _remove_seqs_with_gaps(self, msa, threshold=0.2):
        '''remove sequences with too many gaps'''
        seq_id = list(msa.keys())
        seq_len = len(list(msa.values())[0]) ## All seqs same len, select first
        for k in seq_id:
            if self.setuper.ref_seq and self.setuper.ref_n == k:
                continue # keep query sequence in msa
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
        training_alignment = {}
        for key_idx, seq in enumerate(trans_arr):
            overlap = 0
            for i in range(len_seq):
                ## Simply gaps or anything else
                if (ref_seq[i] == 0 and seq[i] == 0) or (ref_seq[i] != 0 and seq[i] != 0):
                    overlap += 1
            if overlap >= threshold * len_seq:
                overlap_seqs.append(seq)
                final_keys.append(key_list[key_idx])
                training_alignment[key_list[key_idx]] = seq
        with open(self.pickle + "/keys_list.pkl", 'wb') as file_handle:
            pickle.dump(final_keys, file_handle)
        training_converted = self.back_to_amino(training_alignment)
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
        if gen_pickle:
            with open(self.pickle + "/seq_weight.pkl", 'wb') as file_handle:
                pickle.dump(seq_weight, file_handle)
        return seq_weight

    def _to_binary(self, msa, gen_pickle=True):
        K = len(self.aa)+1  ## num of classes of aa
        D = np.identity(K)
        num_seq = msa.shape[0]
        len_seq_msa = msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[msa[i]]
        if gen_pickle:
            with open(self.pickle + "/seq_msa_binary.pkl", 'wb') as file_handle:
                pickle.dump(seq_msa_binary, file_handle)
        return seq_msa_binary

    def back_to_amino(self, msa):
        ## Reverse transformation
        ## exports function out due to analyzer decode to sequences
        reverse_index = {}
        reverse_index[0] = '-'
        i = 1
        for a in self.aa:
            reverse_index[i] = a
            i += 1
        transformed = {}
        ## Sequencies back to aminoacid representation
        for k in msa.keys():
            to_amino = msa[k]
            amino_seq = [reverse_index[s] for s in to_amino]
            transformed[k] = amino_seq
        return transformed

    def _stats(self, msa):
        if self.setuper.stats:
            name = self.setuper.ref_n if self.setuper.ref_seq else 'No reference sequence'
            print("=" * 60)
            print("Pfam ID : {0} - {2}, Reference sequence {1}".format(self.setuper.pfam_id, name, self.setuper.rp))
            print("Sequences used: {0}".format(msa.shape[0]))
            print("Max lenght of sequence: {0}".format(max([len(i) for i in msa])))
            print('Sequences removed due to nnoessential AA: ', self.no_essentials)
            print("=" * 60)
            print()

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = MSAFilterCutOff(tar_dir)
    msa.proc_msa()