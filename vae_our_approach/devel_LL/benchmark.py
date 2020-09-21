__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/18 13:49:00"

import numpy as np
import pickle
import random
import torch
import warnings

from pipeline import StructChecker
from analyzer import VAEHandler
from matplotlib import pyplot as plt

class Benchmarker:
    def __init__(self, positive_control, train_data, setuper):
        self.setuper = setuper
        self.positive = positive_control
        self.train_data = train_data

        self.negative = self._generate_negative(count=positive_control.shape[0], s_len=positive_control.shape[1])
        self.vae_handler = VAEHandler(setuper)
        # Ignore deprecated errors
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def make_bench(self):
        marginals_train = self._bench(self.train_data)
        marginals_positive = self._bench(self.positive)
        marginals_negative = self._bench(self.negative)

        # Normalization process
        tr_w = np.empty(len(marginals_train))
        tr_w.fill(1 / len(marginals_train))
        pos_w = np.empty(len(marginals_positive))
        pos_w.fill(1 / len(marginals_positive))
        neg_w = np.empty(len(marginals_negative))
        neg_w.fill(1 / len(marginals_negative))

        # Plot it
        plt.style.use('seaborn-deep')
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 50)
        ax.hist([marginals_negative, marginals_positive, marginals_train], bins, alpha=1, label=['neg', 'pos'], color=['r', 'g', 'b'], weights=[neg_w, pos_w, marginals_train])
        ax.legend(loc='upper right')
        fig.show()
        save_path = self.setuper.high_fld + '/benchmark.png'
        print("Class highlighter saving graph to", save_path)
        fig.savefig(save_path)
        if self.setuper.stats:
            print('Benchmark results:')
            print('\tpositive mean: \t', sum(marginals_positive)/len(marginals_positive))
            print('\tnegative mean: \t', sum(marginals_negative)/len(marginals_negative))
            print('\ttrain data mean: \t', sum(marginals_train)/len(marginals_train))

    def _bench(self, data):
        msa_weight = np.ones(data.shape[0]) / data.shape[0]
        msa_weight = msa_weight.astype(np.float32)
        data = data.astype(np.float32)
        rand_keys = ['k_'.format(i) for i in range(data.shape[0])]
        mus, sigmas = self.vae_handler.propagate_through_VAE(data, weights=msa_weight, keys=rand_keys)
        return self._sample(mus, sigmas, data)


    def _sample(self, mus, sigmas, data):
        '''
        Sample for each q(Z|X) for 10 000 times and make average
            1/N * SUM(p(X,Zi)/q(Zi|X))
        '''
        N = 10000
        probs = [] # marginal propabilities

        # Lambda for the calculation of the amino sequence decoded from binary
        get_aas = lambda xs: [np.where(aa_bin == 1)[0] for aa_bin in xs]
        # Lambda for p(X,Zi)/q(Zi|X) of generated and original, given as sum of equal positions to length of original sequence
        marginal = lambda gen, orig: sum([1 if g == o else 0 for g,o in zip(gen, orig)]) / len(orig)
        print('='*60)
        print('Sampling process is begun...')
        iteration = 0

        for mu, sigma, d in zip(mus, sigmas, data):
            iteration += 1
            if iteration % 100 == 0:
                print(iteration, 'inputs were sampled for', N, 'times')
            # For each mu sample N times and compare with original data
            s = [torch.randn_like(torch.tensor(sigma)) for _ in range(N)]
            zs = [z + mu for z in s]
            seqs = self.vae_handler.decode_marginal_prob(zs)
            original, sum_p_X_Zi = (get_aas(d), 0)
            for vae_res in seqs:
                # SUM(p(X,Zi)/q(Zi|X))
                sum_p_X_Zi += marginal(get_aas(vae_res), original)
            # 1/N * SUM(p(X,Zi)/q(Zi|X))
            marginal_prob = sum_p_X_Zi / N
            probs.append(marginal_prob)
        return probs

    def _generate_negative(self, count, s_len):
        '''Generate random sequences'''
        K = self.positive.shape[2]
        D = np.identity(K)
        rand_seq_binary = np.zeros((count, s_len, K))
        for i in range(count):
            rand_seq_binary[i, :, :] = D[[i % K for i in random.sample(range(0, s_len), s_len)]]
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
    b.make_bench()
    # with open(tar_dir.pickles_fld + '/seq_msa_binary.pkl', 'rb') as file_handle:
    #     train_set = pickle.load(file_handle)
    # iden = np.identity(train_set.shape[2])
    # pos = np.zeros((2, train_set.shape[1], train_set.shape[2]))
    # for p in pos:
    #     for i in range(len(p)):
    #         p[i] = iden[i % train_set.shape[2]]
    # b = Benchmarker(pos, train_set, tar_dir)
    # b.make_bench()