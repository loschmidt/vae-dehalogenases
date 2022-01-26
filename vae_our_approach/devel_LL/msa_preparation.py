__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/13 11:30:00"

import pickle
from typing import Dict, List

import numpy as np
from Bio import SeqIO

from VAE_logger import Logger


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
    def load_msa(file: str = None) -> Dict[str, str]:
        """ Static method for loading fasta MSA from file. Search for fasta/fa file extension! """
        seq_dict = {}
        Logger.print_for_update(' MSA preparation message: Loading fasta format file {}', value="in process...")
        # Check format of file and proceed fasta format here
        if file is not None:
            if file.endswith('.fasta') or file.endswith('.fa'):
                fasta_sequences = SeqIO.parse(open(file), 'fasta')
                for fasta in fasta_sequences:
                    name, seq = fasta.id, str(fasta.seq)
                    seq_dict[name] = seq.upper()
                Logger.update_msg(value="done!", new_line=True)
        else:
            Logger.update_msg(value="cannot be done! No file parameter passed to the function", new_line=True)
        return seq_dict

    @staticmethod
    def amino_acid_dict(pickle_dir: str) -> Dict[str, int]:
        """ Return number encoding dictionary of amino acid alphabet """
        aa_index = {'-': 0, '.': 0}
        i = 1
        for a in MSA.aa:
            aa_index[a] = i
            i += 1
        with open(pickle_dir + "/aa_index.pkl", 'wb') as file_handle:
            pickle.dump(aa_index, file_handle)
        return aa_index

    @staticmethod
    def number_to_binary(msa: np.ndarray, pickle_dir: str, gen_pickle: bool = True) -> np.ndarray:
        """ Transform number array to one-hot encoding """
        K = len(MSA.aa) + 1
        D = np.identity(K)
        num_seq = msa.shape[0]
        len_seq_msa = msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[msa[i]]
        if gen_pickle:
            Logger.print_for_update(' MSA preparation message: Storing binary MSA into file {}', value="in process...")
            with open(pickle_dir + "/seq_msa_binary.pkl", 'wb') as file_handle:
                pickle.dump(seq_msa_binary, file_handle)
            Logger.update_msg(value="{} is done!".format(pickle_dir+"/seq_msa_binary.pkl"), new_line=True)
        return seq_msa_binary

    @staticmethod
    def number_to_amino(msa: Dict[str, List[int]]) -> Dict[str, List[str]]:
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
            transformed[k] = amino_seq
        return transformed