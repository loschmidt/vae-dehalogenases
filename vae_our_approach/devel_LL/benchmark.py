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
import seaborn as sns

class Benchmarker:
    def __init__(self, positive_control, train_data, setuper):
        self.setuper = setuper
        self.positive = positive_control
        self.train_data = train_data[random.sample(range(0, train_data.shape[0]), positive_control.shape[0])] # Same amount of samples

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

        mean_p = sum(marginals_positive) / len(marginals_positive)
        mean_n = sum(marginals_negative) / len(marginals_negative)
        mean_t = sum(marginals_train) / len(marginals_train)
        # Plot it
        plt.style.use('seaborn-deep')
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 50)
        ax.hist([marginals_negative, marginals_positive, marginals_train], bins,
                label=['neg', 'pos', 'train_data'], color=['red', 'b', 'k'], weights=[neg_w, pos_w, tr_w])#, histtype='step', stacked=True, fill=False, density=True)
        fig2, aa = plt.subplots()
        aa = sns.distplot(marginals_negative, hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3})
        sns.kdeplot(marginals_positive, ax=aa)
        sns.kdeplot(marginals_train, ax=aa)
        ax.legend(loc='upper right')
        plt.xlabel('% of similarity [%/100]')
        plt.ylabel('Quantity')
        plt.title('Benchmark histogram')
        ax.text(6, 2, r'$\mu={0},{1},{2}$'.format(mean_n, mean_p, mean_t))
        save_path = self.setuper.high_fld + '/benchmark.png'
        print("Class highlighter saving graph to", save_path)
        fig.savefig(save_path)
        save_path = self.setuper.high_fld + '/benchmark_density.png'
        print("Class highlighter saving graph to", save_path)
        fig2.savefig(save_path)
        if self.setuper.stats:
            print('Benchmark results:')
            print('\tpositive mean: \t', mean_p)
            print('\tnegative mean: \t', mean_n)
            print('\ttrain data mean: \t', mean_t)

        file = open(self.setuper.high_fld + '/benchmark_stats.txt', 'w')
        print('Benchmark stats saved in', self.setuper.high_fld + '/benchmark_stats.txt')
        s = 'train: '+ str(mean_t) + ' positive: ' + str(mean_p) + ' negative: ' + str(mean_n)
        file.write(s)
        file.close()

    def _bench(self, data):
        msa_weight = np.ones(data.shape[0]) / data.shape[0]
        msa_weight = msa_weight.astype(np.float32)
        data_t = data.astype(np.float32)
        rand_keys = ['k_'.format(i) for i in range(data.shape[0])]
        mus, sigmas = self.vae_handler.propagate_through_VAE(data_t, weights=msa_weight, keys=rand_keys)
        return self._sample(mus, sigmas, data)

    def _sample(self, mus, sigmas, data):
        '''
        Sample for each q(Z|X) for 10 000 times and make average
            1/N * SUM(p(X,Zi)/q(Zi|X))
        '''
        N = 10
        probs = []  # marginal propabilities

        # Lambda for the calculation of the amino sequence decoded from binary
        get_aas = lambda xs: [np.where(aa_bin == 1)[0] for aa_bin in xs]
        # Lambda for p(X,Zi)/q(Zi|X) of generated and original, given as sum of equal positions to length of original sequence
        marginal = lambda gen, orig: sum([1 if g == o else 0 for g, o in zip(gen, orig)]) / len(orig)
        print('=' * 60)
        print('Sampling process has begun...')

        # Sample for mus and sigmas for N times
        batch_size = 32
        num_batches = mus.shape[0] // batch_size + 1
        for idx_batch in range(num_batches):
            if (idx_batch + 1) % 50 == 0:
                print("idx_batch: {} out of {}".format(idx_batch, num_batches))
            mus_b = mus[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            sigmas_b = sigmas[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            mus_b = torch.from_numpy(mus_b)
            sigmas_b = torch.from_numpy(sigmas_b)
            with torch.no_grad():
                decoded_seq = self.vae_handler.decode_for_marginal_prob(mus_b, sigmas_b, N)
                # Calculate marginal probability for each generated sequence
                chunks = [decoded_seq[x:x + N] for x in range(0, len(decoded_seq), N)]
                for i, ch in enumerate(chunks):
                    #ch = list(map(get_aas, ch))
                    orig = get_aas(data[(idx_batch * batch_size) + i]) # Remember chunks are from more than one mu so compare with correct orignal sequence
                    sum_p_X_Zi = 0
                    for c in ch:
                    # SUM(p(X,Zi)/q(Zi|X))
                        sum_p_X_Zi += marginal(c, orig)
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