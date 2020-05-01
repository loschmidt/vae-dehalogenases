__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/05/01 12:45:00"

import pickle
import numpy as np

class MSA_VAE_loader():
    '''
    Class for transforming Stockholm file to binary seq code
    '''
    def __init__(self, file_name, src_dir):
        '''
        file_name - name of file in stockholm format to be prepared
        src_dir - name of directory in output directory with source latent space
        '''
        self.file_name = file_name
        self.pickle_aa_index = "./output/{0}/aa_index.pkl".format(src_dir)
        self.pickle_inds = "./output/{0}/seq_pos_idx.pkl".format(src_dir)
        self.seq_dict = self._init_dict()
        self.pos_ind, self.aa_index = self._read_pickles()

    def binary_seq(self, length):
        '''
        Creates binary sequence of given length
        '''
        seq_msa, key_list = self._rem_seq_with_unknown_res(self.seq_dict)
        seq_msa = self._aling_to_len(seq_msa, length)
        seq_msa_binary = self._to_binary(seq_msa)
        return seq_msa_binary, key_list

    def _init_dict(self):
        '''
        Trasforming file to dictionary without sequences with lots of gaps
        '''
        seq_dict = {}
        with open(self.file_name, 'r') as file_handle:
            for line in file_handle:
                if line[0] == "#" or line[0] == "/" or line[0] == "":
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                seq_id, seq = line.split()
                seq_dict[seq_id] = seq.upper()

        ## remove sequences with too many gaps
        seq_id = list(seq_dict.keys())
        for k in seq_id:
            if seq_dict[k].count("-") + seq_dict[k].count(".") > 10:
                seq_dict.pop(k)

        return seq_dict

    def _read_pickles(self):
        ''' Just loads pickle files '''
        aa_index = {}
        pos_idx = []
        with open(self.pickle_inds, 'rb') as file_handle:
            pos_idx = pickle.load(file_handle)
        with open(self.pickle_aa_index, 'rb') as file_handle:
            aa_index = pickle.load(file_handle)
        return pos_idx, aa_index

    def _rem_seq_with_unknown_res(self, seq_dict):
        seq_msa = []
        keys_list = []
        for k in seq_dict.keys():
            if seq_dict[k].count('X') > 0 or seq_dict[k].count('Z') > 0:
                continue
            seq_msa.append([self.aa_index[s] for s in seq_dict[k]])
            keys_list.append(k)
        seq_msa = np.array(seq_msa)
        return seq_msa, keys_list

    def _aling_to_len(self, seq_msa, length):
        '''
        Removes positions in MSA on given position according to reference processing
        This method is optimized for highlighting RPXX subfamilies in a plot
        '''
        pos_idx = []
        for i in self.pos_ind:
            if i < seq_msa.shape[1]:
                pos_idx.append(i)

        seq_msa = seq_msa[:, np.array(pos_idx)]
        if seq_msa.shape[1] == length:
            return seq_msa
        if length < seq_msa.shape[1]:
            ## Somehow reduce MSA
            gaps_idx = {}  ## count of gaps, index
            for i in range(seq_msa.shape[1]):
                gaps_idx[i] = np.sum(seq_msa[:, i] == 0)
            ## Sort by count of gapsa
            gaps_idx = {k: v for k, v in sorted(gaps_idx.items(), key=lambda item: item[1])}
            pos_idx_to_remove = []
            while (seq_msa.shape[1] - len(pos_idx_to_remove)) > length:
                ## Remove with most gaps
                k, v = gaps_idx.popitem()
                pos_idx_to_remove.append(k)
            ## Keep just position without less gaps
            pos_idx = []
            for i in range(seq_msa.shape[1]):
                if i not in pos_idx_to_remove:
                    pos_idx.append(i)
            seq_msa = seq_msa[:, np.array(pos_idx)]
            return seq_msa
        else:
            print("Preparation MSA ERROR -------------- ")
            print("MSA length is {0} but target length {1} is bigger".format(length, seq_msa.shape[1]))
            exit(1)

    def _to_binary(self, seq_msa):
        '''
        Convert to binary representation of amino acids
        '''
        K = 21  ## num of classes of aa
        D = np.identity(K)
        num_seq = seq_msa.shape[0]
        len_seq_msa = seq_msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[seq_msa[i]]