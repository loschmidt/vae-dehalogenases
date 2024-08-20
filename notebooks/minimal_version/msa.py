__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2023/06/ 13:12:00"

import os.path
import pickle
from typing import Dict, List

import numpy as np
import torch


class MSA:
    """
    Class gathers functions and mapping values, loading files
    It is static class.
    """
    # convert aa type into num 0-20
    aa = ['R', 'H', 'K',
          'D', 'E',
          'S', 'T', 'N', 'Q',
          'C', 'G', 'P',
          'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

    @staticmethod
    def load_msa(file_path: str = None) -> Dict[str, str]:
        """ Static method for loading fasta MSA from file."""
        sequences = {}
        with open(file_path, 'r') as file:
            current_seq = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences[header] = current_seq.upper()
                        current_seq = ''
                    header = line[1:]
                else:
                    current_seq += line.upper()
            if current_seq:
                sequences[header] = current_seq.upper()
        return sequences

    @staticmethod
    def load_pfam(file_path: str = None) -> Dict[str, str]:
        """ Static method for loading fasta MSA from pfam file. """
        sequences = {}
        with open(file_path, 'r') as lines:
            for line in lines:
                if line[0] == "#" or line[0] == "/" or line[0] == "":
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                seq_id, seq = line.split()
                sequences[seq_id] = seq.upper()
        return sequences

    @staticmethod
    def amino_acid_dict(pickle_dir: str=None) -> Dict[str, int]:
        """ Return number encoding dictionary of amino acid alphabet """
        aa_index = {'-': 0, '.': 0}
        i = 1
        for a in MSA.aa:
            aa_index[a] = i
            i += 1
        if pickle_dir:
            with open(os.path.join(pickle_dir, "aa_index.pkl"), 'wb') as file_handle:
                pickle.dump(aa_index, file_handle)
        return aa_index

    @staticmethod
    def aa_to_number(msa: Dict[str, str]) -> np.ndarray:
        """
        Transfer fasta dictionary (MSA ID: sequence in AA) to number representation
        :param msa: dictionary with loaded MSA
        :return: just transformed sequences to the number representation
        """
        seq_msa, keys_list = [], []
        aa_index = MSA.amino_acid_dict()
        for k in msa.keys():
            try:
                seq_msa.append([aa_index[s] for s in msa[k]])
                keys_list.append(k)
            except KeyError:
                print(" Translating to the number MSA representation: ERROR {k} has unsupported amino acid type!")
                print(f" Supported are: {MSA.aa}")
        return np.array(seq_msa, dtype=object)

    @staticmethod
    def number_to_binary(msa: np.ndarray) -> np.ndarray:
        """ Transform number array to one-hot encoding """
        K = len(MSA.aa) + 1
        D = np.identity(K)
        num_seq = msa.shape[0]
        len_seq_msa = msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[msa[i]]
        return seq_msa_binary

    @staticmethod
    def number_to_amino(msa: Dict[str, List[int]]) -> Dict[str, str]:
        """ Transform from number encoding into amino acid alphabet """
        reverse_index = {0: '-'}
        i = 1
        for a in MSA.aa:
            reverse_index[i] = a
            i += 1
        transformed = {}
        # Sequences back to amino acid representation
        for k in msa.keys():
            to_amino = msa[k]
            amino_seq = [reverse_index[s] for s in to_amino]
            transformed[k] = ''.join(amino_seq)
        return transformed

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
        return np.array([MSA.binary_to_numbers_coding(binary) for binary in binaries])

    @staticmethod
    def binaries_to_tensor(binaries) -> torch.Tensor:
        """
        Get binary reshaped for vae
        """
        # assert len(binaries.shape) >= 3  # only for batches
        binaries = binaries.astype(np.float32)
        binaries = binaries.reshape((binaries.shape[0], -1)) if binaries.ndim == 3 else binaries.reshape((-1))
        return torch.from_numpy(binaries)

    @staticmethod
    def binary_to_tensor(binary) -> torch.Tensor:
        """
        Get binary reshaped for vae
        """
        assert len(binary.shape) == 2  # only for batches
        binaries = binary.astype(np.float32)
        binaries = binary.reshape((-1))
        return torch.from_numpy(binaries)
