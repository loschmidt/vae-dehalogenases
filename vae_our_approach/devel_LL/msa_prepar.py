__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 11:30:00"

import pickle
import numpy as np
from pipeline import StructChecker
from download_MSA import Downloader
from Bio import SeqIO

class MSA:
    def __init__(self, setuper: StructChecker, processMSA=True):
        self.setup = setuper
        self.msa_file = setuper.msa_file
        self.pickle = setuper.pickles_fld
        self.values = {"ref": setuper.ref_seq, "ref_n": setuper.ref_n, "keep_gaps": setuper.keep_gaps, "stats": setuper.stats}
        if processMSA:
            self.seq_dict = self.load_msa()
            self.amino_acid_dict()

    def proc_msa(self):
        if self.values["ref"] or self.values["keep_gaps"]:
            self._ref_filtering()
        else:
            ## get length of first element in dictionary
            self._rem_seqs_on_gaps(len(self.seq_dict[next(iter(self.seq_dict))]), threshold=0.85)
        with open(self.pickle + "/seq_dict.pkl", 'wb') as file_handle:
            pickle.dump(self.seq_dict, file_handle)

        self._remove_unexplored_and_covert_aa()
        if not self.values["keep_gaps"]:
            self._rem_gaps_msa_positions()
        else:
            ## Keep all sequence positions
            pos_idx = [i for i in range(self.seq_msa.shape[1])]
            with open(self.pickle + "/seq_pos_idx.pkl", 'wb') as file_handle:
                pickle.dump(pos_idx, file_handle)
        with open(self.pickle + "/seq_msa.pkl", 'wb') as file_handle:
            pickle.dump(self.seq_msa, file_handle)
        self._weighting_sequences()
        self._to_binary()
        self._stats()

    def load_msa(self, file=None):
        seq_dict = {}
        # Check format of file
        if file is not None:
            if file.endswith('.fasta'):
                print('Loading fasta format file')
                fasta_sequences = SeqIO.parse(open(file), 'fasta')
                for fasta in fasta_sequences:
                    name, seq = fasta.id, str(fasta.seq)
                    seq_dict[name] = seq.upper()
                return seq_dict
        with open(self.msa_file if file is None else file, 'r') as file_handle:
            for line in file_handle:
                if line[0] == "#" or line[0] == "/" or line[0] == "":
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                seq_id, seq = line.split()
                seq_dict[seq_id] = seq.upper()
        return seq_dict

    def _ref_filtering(self):
        """Filter sequences against query sequences"""
        query_seq = self.seq_dict[self.values["ref_n"]]  ## with gaps
        idx = [s == "-" or s == "." for s in query_seq]
        for k in self.seq_dict.keys():
            self.seq_dict[k] = [self.seq_dict[k][i] for i in range(len(self.seq_dict[k])) if idx[i] == False]
        query_seq = self.seq_dict[self.values["ref_n"]]  ## without gaps
        self._rem_seqs_on_gaps(len(query_seq))

    def _rem_seqs_on_gaps(self, seq_len, threshold=0.2):
            ## remove sequences with too many gaps
            seq_id = list(self.seq_dict.keys())
            num_gaps = []
            for k in seq_id:
                num_gaps.append(self.seq_dict[k].count("-") + self.seq_dict[k].count("."))
                if self.seq_dict[k].count("-") + self.seq_dict[k].count(".") > threshold * seq_len:
                    self.seq_dict.pop(k)

    def amino_acid_dict(self, export=False):
        ## convert aa type into num 0-20
        self.aa = ['R', 'H', 'K',
              'D', 'E',
              'S', 'T', 'N', 'Q',
              'C', 'G', 'P',
              'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
        self.aa_index = {}
        self.aa_index['-'] = 0
        self.aa_index['.'] = 0
        i = 1
        for a in self.aa:
            self.aa_index[a] = i
            i += 1
        with open(self.pickle + "/aa_index.pkl", 'wb') as file_handle:
            pickle.dump(self.aa_index, file_handle)
        if export:
            return self.aa, self.aa_index

    def _remove_unexplored_and_covert_aa(self):
        """
            Remove everything with unexplored residues from dictionary
            Converts amino acid letter representation to numbers store in self.seq_msa
        """
        seq_msa = []
        keys_list = []
        nonessential = 0
        for k in self.seq_dict.keys():
            if self.seq_dict[k].count('X') > 0 or self.seq_dict[k].count('Z') > 0:
                continue
            try:
                seq_msa.append([self.aa_index[s] for s in self.seq_dict[k]])
                keys_list.append(k)
            except KeyError:
                nonessential += 1
        self.seq_msa = np.array(seq_msa)
        self.keys_list = keys_list
        with open(self.pickle + "/keys_list.pkl", 'wb') as file_handle:
            pickle.dump(keys_list, file_handle)
        self.nonessential = nonessential

    def _rem_gaps_msa_positions(self):
        pos_idx = []
        for i in range(self.seq_msa.shape[1]):
            if np.sum(self.seq_msa[:, i] == 0) <= self.seq_msa.shape[0] * 0.2:
                pos_idx.append(i)
        with open(self.pickle + "/seq_pos_idx.pkl", 'wb') as file_handle:
            pickle.dump(pos_idx, file_handle)
        self.seq_msa = self.seq_msa[:, np.array(pos_idx)]

    def _weighting_sequences(self):
        ## reweighting sequences
        seq_weight = np.zeros(self.seq_msa.shape)
        for j in range(self.seq_msa.shape[1]):
            aa_type, aa_counts = np.unique(self.seq_msa[:, j], return_counts=True)
            num_type = len(aa_type)
            aa_dict = {}
            for a in aa_type:
                aa_dict[a] = aa_counts[list(aa_type).index(a)]
            for i in range(self.seq_msa.shape[0]):
                seq_weight[i, j] = (1.0 / num_type) * (1.0 / aa_dict[self.seq_msa[i, j]])
        tot_weight = np.sum(seq_weight)
        ## Normalize weights of sequences
        self.seq_weight = seq_weight.sum(1) / tot_weight
        with open(self.pickle + "/seq_weight.pkl", 'wb') as file_handle:
            pickle.dump(self.seq_weight, file_handle)

    def _to_binary(self):
        K = len(self.aa)+1  ## num of classes of aa
        D = np.identity(K)
        num_seq = self.seq_msa.shape[0]
        len_seq_msa = self.seq_msa.shape[1]
        seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
        for i in range(num_seq):
            seq_msa_binary[i, :, :] = D[self.seq_msa[i]]
        self.seq_msa_binary = seq_msa_binary
        with open(self.pickle + "/seq_msa_binary.pkl", 'wb') as file_handle:
            pickle.dump(seq_msa_binary, file_handle)

    def _stats(self):
        if self.values["stats"]:
            print("=" * 60)
            print("Pfam ID : {0} - {2}, Reference sequence {1}".format(self.setup.pfam_id, self.values["ref"], self.setup.rp))
            print("Sequences used: {0}".format(self.seq_msa.shape[0]))
            print("Max lenght of sequence: {0}".format(max([len(i) for i in self.seq_msa])))
            print('Sequences removed due to nnoessential AA: ', self.no_essentials)
            print("=" * 60)
            print()

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    msa = MSA(tar_dir)
    msa.proc_msa()
