"""
Collection of globally useful functions
"""
import pickle
import sys
import re
from io import StringIO
from typing import List

import numpy as np
import torch


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def store_to_fasta(msa: dict, file_path):
    """
    Store msa in the dictionary of sequence IDs and AAs to given file
    :param msa: MSA dictionary of str to str
    :param file_path: path to msa
    :return:
    """
    with open(file_path, 'w') as file:
        for header, sequence in msa.items():
            file.write(f'>{header}\n')
            formatted_sequence = '\n'.join(["".join(sequence)[i:i + 80] for i in range(0, len(sequence), 80)])
            file.write(formatted_sequence + '\n')


def store_to_pkl(data, path_to_pkl):
    """
    Store data into the pickle
    :param data: data to be stored
    :param path_to_pkl: path to the file
    :return:
    """
    with open(path_to_pkl, "wb") as f:
        pickle.dump(data, f)


def load_from_pkl(path_to_pkl):
    """
    Load data from pickle format to the program
    :param path_to_pkl: path to file
    :return: loaded data from pickle file
    """
    with open(path_to_pkl, 'rb') as f:
        return pickle.load(f)


def key_to_embedding(embeddings: np.ndarray, key: str) -> np.ndarray:
    """
    Embedding from LatentSpace class, map sequence ID to latent space coordinates
    :param embeddings: dictionary with keys and mus
    :param key: which key to find
    :return: mu coordinates
    """
    try:
        key_ind = embeddings['keys'].index(key)
    except:
        print(f" Key to embedding mapping: {key} not found in dictionary")
        exit(2)
    return embeddings['mus'][key_ind]


def get_binaries_by_key(keys_to_get: List[str], msa_binary: np.ndarray, msa_keys: List[str]) -> np.ndarray:
    """
    Get binaries corresponding to the given MSA sequences from training dataset
    :param keys_to_get: which MSA ID sequence binary we want to get
    :param msa_binary: the whole dataset preprocessed binary
    :param msa_keys: all the sequence keys from preprocessed MSA
    :return: binaries of that key
    """
    np_seq_array = np.array([])
    for k in keys_to_get:
        try:
            key_index = msa_keys.index(k)
            np_seq_array = np.append(np_seq_array, msa_binary[key_index])
        except ValueError:
            print(" Sequence transformer : ERROR in get_binary_by_key method the key {} not found!".format(seq_key))
            exit(1)
    return np_seq_array


# MSA transformation methods
def reshape_binary(binary: np.ndarray) -> torch.Tensor:
    """
    Reshapes the input binary with 21 positions per AA to the latent vector with 21 X L
    :param binary:
    :return: reshaped values
    """
    binary = binary.reshape((binary.shape[0], -1))
    tensor_binary = torch.tensor(binary)
    return tensor_binary.to(torch.float32)


# Identities
def sequence_identity(source: str, target: str) -> float:
    """
    Returns the sequence percentage identity. Input sequences are lists of amino acids
    Example:
         source = [A G G E S - D T W A]
         target = [A G G E - D T W A A]
         returns: 50.00
    """
    identical_residues = sum([1 if i == j else 0 for i, j in zip(source, target)])
    identical_fraction = identical_residues / len(target)
    return identical_fraction * 100
