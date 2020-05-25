__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/05/01 12:45:00"

import pickle
import numpy as np

from collections import Counter

class MSA_VAE_loader():
    '''
    Class for transforming Stockholm file to binary seq code
    '''
    def __init__(self, file_name, src_dir, src_MSA_file):
        '''
        file_name - name of file in stockholm format to be prepared
        src_dir - name of directory in output directory with source latent space
        src_MSA_file - name of file with main MSAs (for RPs rp75, for seeds highlight full set ...)
        '''
        self.file_name = file_name
        self.pickle_aa_index = "./output/{0}/aa_index.pkl".format(src_dir)
        self.pickle_inds = "./output/{0}/seq_pos_idx.pkl".format(src_dir)
        self.pickle_key_list = "./output/{0}/keys_list.pkl".format(src_dir)
        self.pickle_latent = "./output/{0}/latent_space.pkl".format(src_dir)
        self.high_seq = self._init_dict(self.file_name)
        self.seq_dict = self._init_dict(src_MSA_file)
        self.pos_ind, self.aa_index, self.latent_keys, self.latent_mu, self.key_list= self._read_pickles()
        self.missing_seq = {} # Will be initialized in _name_match_with_origin function

    def get_Mus_and_unknown_binary_seq(self):
        '''
        Prepare MUs of Gaussian matched through name match and
        prepare seqs for future VAE processing with original weighting
        '''
        mus = self._name_match_with_origin()
        seq_msa = self._rem_seq_with_unknown_res(self.missing_seq)
        if len(seq_msa) == 0:
            ## Everything is removed due to unknown residues
            return mus, None, self.key_list
        seq_msa = self._aling_to_len(seq_msa)
        seq_msa_binary = self._to_binary(seq_msa)
        return mus, seq_msa_binary, self.key_list

    def _init_dict(self, file_name):
        '''
        Trasforming file to dictionary without sequences with lots of gaps
        '''
        seq_dict = {}
        with open(file_name, 'r') as file_handle:
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
            if seq_dict[k].count("-") + seq_dict[k].count(".") > 0.99*len(seq_dict[k]):
                seq_dict.pop(k)

        return seq_dict

    def _read_pickles(self):
        '''
        Just loads pickle files
        '''
        aa_index = {}
        pos_idx = []
        key_list = []
        with open(self.pickle_inds, 'rb') as file_handle:
            pos_idx = pickle.load(file_handle)
        with open(self.pickle_aa_index, 'rb') as file_handle:
            aa_index = pickle.load(file_handle)
        with open(self.pickle_key_list, 'rb') as file_handle:
            key_list = pickle.load(file_handle)
        with open(self.pickle_latent, 'rb') as file_handle:
            latent_space = pickle.load(file_handle)
        mu = latent_space['mu']
        key = latent_space['key']
        return pos_idx, aa_index, key, mu, key_list

    def _rem_seq_with_unknown_res(self, seq_dict):
        '''
        Replace unknown positions in MSA with the most frequent one in column
        '''
        seq_msa = []
        for k in seq_dict.keys():
            if seq_dict[k].count('X') > 0 or seq_dict[k].count('Z') > 0:
                unk_pos = [i for i, letter in enumerate(seq_dict[k]) if letter == 'X' or letter == 'Z']
                for pos in unk_pos:
                    ## Get the most common amino in that position and replace it with it
                    amino_acid = self._most_frequent(seq_dict[:,pos])
                    seq_dict[k][pos] = amino_acid
            seq_msa.append([self.aa_index[s] for s in seq_dict[k]])
        seq_msa = np.array(seq_msa)
        return seq_msa

    def _most_frequent(self, msa_column):
        occurence_count = Counter(msa_column)
        return occurence_count.most_common(1)[0][0]

    def _aling_to_len(self, seq_msa):
        '''
        Removes positions in MSA on given position according to reference processing
        This method is optimized for highlighting RPXX subfamilies in a plot
        '''
        seq_msa = seq_msa[:, np.array(self.pos_ind)]
        return seq_msa

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
        return seq_msa_binary

    def _name_match_with_origin(self):
        '''
        Returns mus of Gaussians with name matched sequencies and
        dictionary with not name matched seqs for future processing through VAE
            WARNING side effect sets self.missing_seq class attribute
        '''
        ## Key to representation of index
        key2idx = {}
        for i in range(len(self.latent_keys)):
            key2idx[self.latent_keys[i]] = i

        names = self.high_seq.keys()
        idx = []
        succ = 0
        fail = 0
        for n in names:
            try:
                idx.append(int(key2idx[n]))
                succ += 1
            except KeyError as e:
                try:
                    ## Get original alignment of searching sequence
                    self.missing_seq[n] = self.seq_dict[n]
                    succ += 1
                except KeyError as e:
                    fail += 1 # That seq is missing even in original seq set. Something terrifying is happening here.
        print("="*60)
        print("Printing match stats")
        print(self.file_name, " Success: ", succ, " Fails: ", fail)

        return self.latent_mu[idx,:]
