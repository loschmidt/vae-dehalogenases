__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/06/09 00:30:00"

import numpy as np
import pickle
import torch

from msa_preprocessor import MSAPreprocessor as Convertor
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

    def sequences_dict_to_binary(self, seqs_dict):
        """ Prepare sequence dictionary to be one hot encoded and retyped for VAE """
        binaries, _, _ = self.convector.prepare_aligned_msa_for_Vae(seqs_dict)
        binaries = binaries.astype(np.float32)
        binaries = binaries.reshape((binaries.shape[0], -1))
        binaries = torch.from_numpy(binaries)
        return binaries

    def get_binary_by_key(self, seq_key):
        """ Method returns the binary of given sequence by """
        try:
            key_index = self.keys_list.index(seq_key)
            return self.msa_binary[key_index]
        except ValueError:
            print(" Sequence transformer : ERROR in get_binary_by_key method the key {} not found!".format(seq_key))
            exit(0)

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
        return self.convector.back_to_amino({seq_key: number_coding})

    def back_to_amino(self, seq_dict):
        """
        Just call back_to_amino method of Convector. This method is uniting import in other files.
        Input : seq_dict = dictionary of sequences in number coding
        """
        return self.convector.back_to_amino(seq_dict)

    def prepare_aligned_msa_for_vae(self, seq_dict):
        """
        Applies same steps for preprocessing as for the whole MSA
        Return (binary, weight, keys)
        """
        binaries, weights, keys = self.convector.prepare_aligned_msa_for_Vae(seq_dict)
        binaries = binaries.astype(np.float32)
        binaries = binaries.reshape((binaries.shape[0], -1))
        weights = weights.astype(np.float32)
        return binaries, weights, keys

    @staticmethod
    def binary_to_numbers_coding(binary):
        """
        Convert binary form into number coding
        [[0,1,0],[0,0,1]] -> [1,2]
        """
        return [np.where(one_hot_AA == 1)[0][0] for one_hot_AA in binary]
