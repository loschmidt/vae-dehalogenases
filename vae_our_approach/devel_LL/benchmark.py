__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/18 13:49:00"

import numpy as np
import pickle
import random
import torch

from pipeline import StructChecker
from VAE_model import VAE
from .analyzer import VAEHandler

class Benchmarker:
    def __init__(self, positive_control, train_data, setuper):
        self.setuper = setuper
        self.positive = positive_control
        self.train_data = train_data

        self.negative = self._generate_negative(count=positive_control.shape[0], s_len=positive_control.shape[1])
        self.vae_handler = VAEHandler(setuper)

    def make_bench(self):
        self._bench_origin(self.train_data)
        ## TODO add for others data and draw plot (kind histogram)

    def _bench(self, data):
        msa_weight = np.ones(data.shape[0]) / data.shape[0]
        msa_weight = msa_weight.astype(np.float32)
        rand_keys = ['k_'.format(i) for i in range(data.shape[0])]
        mus, sigmas = self.vae_handler.propagate_through_VAE(data, weights=msa_weight, keys=rand_keys)


    def _sample(self, mus, sigmas, data):
        '''
        Sample for each q(Z|X) for 10 000 times and make average
            1/N * SUM(p(X,Zi)/q(Zi|X))
        '''
        N = 10000
        probs = [] # propabilities
        for mu, sigma, d in zip(mus, sigmas, data):
            s = [torch.randn_like(sigma) for _ in range(N)]
            zs = [z + mu for z in s]
            seqs = self.vae_handler.decode_sequences_VAE(zs, ref_name='', )

        def seq_diff(gen, orig):
        ## TODO comparison of sequences in binary form

    def _generate_negative(self, count, s_len):
        K = self.positive.shape[2]
        D = np.identity(K)
        rand_seq_binary = np.zeros((count, s_len, K))
        for i in range(count):
            rand_seq_binary[i, :, :] = D[random.sample(range(0, K), s_len)]
        print('Negative control {} samples generated...'.format(count))
        return rand_seq_binary

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    with open(tar_dir.pickles_fld + '/positive_control.pkl', 'rb') as file_handle:
        positive = pickle.load(file_handle)

    with open(tar_dir.pickles_fld + '/training_set.pkl', 'rb') as file_handle:
        train_set = pickle.load(file_handle)
    b = Benchmarker(positive, train_set, tar_dir)