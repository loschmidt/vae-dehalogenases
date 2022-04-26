__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/06/09 00:30:00"

import numpy as np
import pickle
import torch

from project_enums import SolubilitySetting
from msa_handlers.msa_preprocessor import MSAPreprocessor as Convertor
from msa_handlers.msa_preparation import MSA
from metaclasses import Singleton


class Transformer(metaclass=Singleton):
    """
    Transformer class unites all method for transformation of sequence to/from binary one-hot encoding.
    The class is implemented as Singleton design pattern initialized by the CmdHandler class
    """

    def __init__(self, setuper):
        self.convector = Convertor(setuper)
        self.pickles = setuper.pickles_fld
        with open(self.pickles + "/keys_list.pkl", 'rb') as file_handle:
            self.keys_list = pickle.load(file_handle)
        with open(self.pickles + "/seq_msa_binary.pkl", 'rb') as file_handle:
            self.msa_binary = pickle.load(file_handle)
        with open(self.pickles + "/query_excluded_pos_and_aa.pkl", 'rb') as file_handle:
            self.excluded_query_pos_and_aa = pickle.load(file_handle)

    def get_binary_by_key(self, seq_key):
        """ Method returns the binary of given sequence by """
        try:
            key_index = self.keys_list.index(seq_key)
            return self.msa_binary[key_index]
        except ValueError:
            print(" Sequence transformer : ERROR in get_binary_by_key method the key {} not found!".format(seq_key))
            return None

    def get_seq_dict_by_key(self, seq_key):
        """ Method returns the sequence dictionary with sequence in the form of amino acid """
        binary = self.get_binary_by_key(seq_key)
        return self.binary_to_seq_dict(binary, seq_key)

    def binary_to_seq_dict(self, binary, seq_key=None):
        """
        Convert binary encoding of sequence to sequence dictionary. The desired sequence key might be given.
        [[0,1,0],[0,0,1]] -> {seq_key : [R,H]}
        """
        number_coding = self.binary_to_numbers_coding(binary)
        return self.number_coding_to_amino(number_coding, seq_key)

    def number_coding_to_amino(self, number_coding, seq_key=None):
        """
        Convert number coding into amino acid representation. The name of sequence can be given.
        [1,2,1,2,5] -> {seq_key : [R,H,R,H,E]}
        """
        if seq_key is None:
            seq_key = "key_placeholder"
        return MSA.number_to_amino({seq_key: number_coding})

    def back_to_amino(self, seq_dict):
        """
        Just call back_to_amino method of MSA. This method is uniting import in other files.
        Input : seq_dict = dictionary of sequences in number coding
        """
        return MSA.number_to_amino(seq_dict)

    def add_excluded_query_residues(self, aa_sequence):
        """ Adds excluded Amino acids from query in MSA preprocessing """
        aa_sequence = "".join(aa_sequence)  # Secure string format
        for query_pos, query_aa in self.excluded_query_pos_and_aa:
            aa_sequence = aa_sequence[:query_pos] + query_aa + aa_sequence[query_pos:]
        return aa_sequence

    def sequence_dict_to_binary(self, seq_dict):
        """
        Applies same steps for preprocessing as for the whole MSA
        Return (binary, weight, keys)
        """
        binaries, weights, keys = self.convector.prepare_aligned_msa_for_Vae(seq_dict)
        binaries = binaries.astype(np.float32)
        binaries = binaries.reshape((binaries.shape[0], -1))
        weights = weights.astype(np.float32)
        return binaries, weights, keys

    def slice_msa_by_pos_indexes(self, msa_dict):
        """
        In the msa keeps only columns kept during original preprocessing
        Useful for sequence remove during clustering in preprocessing from training msa
        ,and now you would like to highlight them
        """
        with open(self.pickles + "/seq_pos_idx.pkl", 'rb') as file_handle:
            pos_idx = pickle.load(file_handle)
        sliced_msa = {}
        for name, sequence in msa_dict.items():
            sliced_msa[name] = [item for i, item in enumerate(sequence) if i in pos_idx]
        return sliced_msa

    @staticmethod
    def binary_to_numbers_coding(binary):
        """
        Convert binary form into number coding
        [[0,1,0],[0,0,1]] -> [1,2]
        """
        return [np.where(one_hot_AA == 1)[0][0] for one_hot_AA in binary]

    @staticmethod
    def binaries_to_numbers_coding(binaries):
        """
        Convert binaries form into number coding
        [[0,1,0],[0,0,1],
         [0,1,0],[0,0,1]] -> [[1,2],[1,2]]
        """
        return np.array([Transformer.binary_to_numbers_coding(binary) for binary in binaries])

    @staticmethod
    def shape_binary_for_vae(binary):
        """
        Use in the case having binary not shaped for VAE, also prepares artificial weights
        Return (binary, weight)
        """
        weights = np.ones(binary.shape[0]) / binary.shape[0]
        weights = weights.astype(np.float32)
        binary = binary.reshape((binary.shape[0], -1))
        binary = binary.astype(np.float32)
        return binary, weights

    @staticmethod
    def seq_list_to_dict(seq_list):
        """ Convert only sequences to dict form """
        ret_dict = {}
        for i, seq in enumerate(seq_list):
            seq_n = "tmp_seq_{}".format(i)
            ret_dict[seq_n] = seq
        return ret_dict

    @staticmethod
    def idx2onehot(idx, n):
        assert torch.max(idx).item() < n

        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        onehot = torch.zeros(idx.size(0), n).to(idx.device)
        onehot.scatter_(1, idx, 1)
        return onehot

    @staticmethod
    def add_condition(x, c):
        """ Adding condition to input for the case of conditional vae"""
        x = x.to(torch.float32)
        if c is not None:
            c = Transformer.idx2onehot(c, n=SolubilitySetting.SOLUBILITY_BINS.value)
            x = torch.cat((x, c), dim=-1)
        return x