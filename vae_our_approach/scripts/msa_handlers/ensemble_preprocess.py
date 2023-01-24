__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2023/01/22 11:30:00"

import os, sys, inspect
currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

import pickle
import random
from copy import deepcopy
from random import randint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler
from project_enums import Helper
from VAE_logger import Logger


class EnsembleMSAPreprocess:
    """
    The class process gap reduced MSA from MSAPreprocessor script and prepares input files for training
    ensemble of models on individual data splits.
    """

    def __init__(self, cmd_line: CmdHandler):
        self.cmd = cmd_line
        self.pickle = cmd_line.pickles_fld
        self.query_name = cmd_line.query_id
        self.ensemble_dir = self.init_dst_folder()

        # Init MSA meta data
        self.init_msa_meta_data()

    def init_dst_folder(self):
        path = self.pickle + "/ensemble/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def store_pickle(self, path, data):
        with open(self.pickle + "/" + path, 'wb') as file_handle:
            pickle.dump(data, file_handle)

    def store_ensemble(self, path, data):
        self.store_pickle("ensemble/" + path, data)

    def process_data(self, models_cnt: int, positive_percent: int = 5):
        """
        Split data into @models_cnt sets for training and validation.
        Keep query sequence within each of them.
        Generate just one positive control set for all of them with @positive_percent of sequences, default 5%.
        """
        cmd = self.cmd
        percentage = max(100 // positive_percent, 3)  # at least 3 percent
        # init benchmark set
        benchmark_set = np.zeros((self.num_seq // percentage, self.len_protein, self.num_res_type))

        # prepare positive control set
        random_idx = np.random.permutation(range(1, self.num_seq))
        benchmark_indices = np.array([])
        for i in range(self.num_seq // percentage):
            benchmark_set[i] = self.seq_msa_binary[random_idx[i]]
            benchmark_indices = np.append(benchmark_indices, random_idx[i])
        # store positive control
        self.store_pickle('positive_control.pkl', benchmark_set)
        if cmd.solubility_file:
            with open(cmd.pickles_fld + '/solubilities.pkl', 'rb') as file_handle:
                solubility = pickle.load(file_handle)
            self.store_pickle('solubility_positive.pkl', solubility[benchmark_indices.astype(int)])

        # remove positive control indexes
        random_idx = random_idx[(self.num_seq // percentage):]  # truncate positive control indexes out

        # split rest of sequences in ratio 4:1 -> 4 training vs 1 validation and add query
        # validation data can be repeated in several models
        index_cnt = random_idx.shape[0]
        training_size = (index_cnt * 4) // 5 + 1
        validation_size = index_cnt // 5

        # stack 2 identical arrays in the row to make available cyclic slicing
        random_idx = np.append(random_idx, random_idx)

        # for every model make a training subset with query
        for i in range(models_cnt):
            offset = random.randint(0, training_size)
            train_idx = random_idx[offset: offset + training_size]
            valid_idx = random_idx[offset + training_size: offset + training_size + validation_size]
            train_idx.sort()
            valid_idx.sort()
            train_subset = self.seq_msa_binary[train_idx,]
            validation_subset = self.seq_msa_binary[valid_idx,]

            self.store_ensemble(f'training_set{i}.pkl', train_subset)
            self.store_ensemble(f'training_keys{i}.pkl', self.seq_keys[train_idx.reshape(1, -1)])
            self.store_ensemble(f'training_weights{i}.pkl', self.seq_weight[train_idx.astype(int)])
            self.store_ensemble(f'validation_set{i}.pkl', validation_subset)
            self.store_ensemble(f'solubility_set{i}.pkl', solubility[train_idx])

            print(f"\tData generated for model {i+1} out of {models_cnt}")

    def init_msa_meta_data(self):
        try:
            with open(self.pickle + "/seq_msa_binary.pkl", 'rb') as file_handle:
                self.seq_msa_binary = pickle.load(file_handle)
            with open(self.pickle + "/seq_weight.pkl", 'rb') as file_handle:
                self.seq_weight = pickle.load(file_handle).astype(np.float32)
            with open(self.pickle + "/keys_list.pkl", 'rb') as file_handle:
                self.seq_keys = np.array(pickle.load(file_handle))
        except FileNotFoundError:
            print("\n\tERROR please run MSA preprocessing "
                  "\n\t\tpython3 runner.py msa_handlers/msa_preprocessor.py --json [your_config_path] "
                  "\n\tbefore running this command")
            exit(1)

        # Set all meta information
        self.num_seq = self.seq_msa_binary.shape[0]
        self.len_protein = self.seq_msa_binary.shape[1]
        self.num_res_type = self.seq_msa_binary.shape[2]


if __name__ == '__main__':
    tar_dir = CmdHandler()
    msa_proc = EnsembleMSAPreprocess(cmd_line=tar_dir)
    msa_proc.process_data(tar_dir.ens_cnt)
