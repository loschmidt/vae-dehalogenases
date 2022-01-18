__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/18 13:49:00"

import csv
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from analyzer import VAEHandler, Highlighter
from download_MSA import Downloader
from msa_preprocessor import MSAPreprocessor as BinaryCovertor
from msa_preparation import MSA
from parser_handler import CmdHandler
from sequence_transformer import Transformer

## LAMBDAS FUNCTIONS FOR CONVERSION AND PAIRWWISE COMPARISON OF SEQUENCES
# Lambda for the calculation of the amino sequence decoded from binary
get_aas = lambda xs: [np.where(aa_bin == 1)[0] for aa_bin in xs]
get_aasR = lambda xs: [np.where(aa_bin == 1)[0][0] for aa_bin in xs]
# Lambda for p(X,Zi)/q(Zi|X) of generated and original, given as sum of equal positions to length of original sequence
marginal = lambda gen, orig: sum([1 if g == o else 0 for g, o in zip(gen, orig)]) / len(orig)

class Benchmarker:
    def __init__(self, positive_control, train_data, setuper, generate_negative=True, ancestor_probs=[]):
        self.setuper = setuper
        self.positive = positive_control
        if train_data is not None:
            self.train_data = train_data[random.sample(range(0, train_data.shape[0]), positive_control.shape[0])] # Same amount of samples

        if generate_negative:
            self.negative = self._generate_negative(count=positive_control.shape[0], s_len=positive_control.shape[1], profile_data=train_data)
        self.vae_handler = VAEHandler(setuper, model_name=setuper.model_name)
        self.binaryConv = BinaryCovertor(self.setuper)
        self.transformer = Transformer(setuper)

        self.ancestors_probs = ancestor_probs
        # Ignore deprecated errors
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def make_bench(self):
        marginals_train = self._bench(self.train_data)
        marginals_positive = self._bench(self.positive)
        marginals_negative = self._bench(self.negative)
        self._store_marginals(marginals_train, marginals_positive, marginals_negative)

        mean_p = sum(marginals_positive) / len(marginals_positive)
        mean_n = sum(marginals_negative) / len(marginals_negative)
        mean_t = sum(marginals_train) / len(marginals_train)
        mean_a = sum(self.ancestors_probs) / len(self.ancestors_probs)
        # Plot it
        plt.style.use('seaborn-deep')
        # Prepare dataframe
        datasets = []
        probabilities = []
        for dataset, probs in [("Positive", marginals_positive),("Negative", marginals_negative),
                                ("Training", marginals_train), ("Ancestors", self.ancestors_probs)]:
            for p in probs:
                datasets.append(dataset)
                probabilities.append(p*100)
        data_dict = {"Dataset": datasets, "Probabilities": probabilities}
        dataFrame = pd.DataFrame.from_dict(data_dict)
        # sns_plot = sns.displot(dataFrame, x="Probabilities", hue="Dataset", kind="kde", fill=True, common_norm=False,
        #                        color=["green", "black", "red", "orange"])
        sns_plot = sns.histplot(data=dataFrame, x="Probabilities", hue="Dataset", multiple="dodge", shrink=.8,
                                color=["green", "black", "red", "orange"])

        plt.xlabel('% probability of observing')
        plt.ylabel('Density')
        plt.title(r'Benchmark histogram $\mu={0:.2f},{1:.2f},{2:.2f},{3:.2f}$'.format(mean_n, mean_p, mean_t, mean_a))

        save_path = self.setuper.high_fld + '/{}benchmark_density.png'.format(self.setuper.model_name)
        print(" Benchmark message : Class highlighter saving graph to", save_path)
        sns_plot.figure.savefig(save_path)
        if self.setuper.stats:
            print(' Benchmark message : Benchmark results:')
            print('\tpositive mean: \t', mean_p)
            print('\tnegative mean: \t', mean_n)
            print('\ttrain data mean: \t', mean_t)
            print('\tStraight ancestors mean: \t', mean_a)

        file = open(self.setuper.high_fld + '/{}benchmark_stats.txt'.format(self.setuper.model_name), 'w')
        print(' Benchmark stats saved in', self.setuper.high_fld + '/{}benchmark_stats.txt'.format(self.setuper.model_name))
        s = 'train: '+ str(mean_t) + ' positive: ' + str(mean_p) + ' negative: ' + str(mean_n) +' ancestors: '+ str(mean_a)
        file.write(s)
        file.close()

    def measure_seq_probability(self, seqs_dict):
        """
        Method measures the marginal observation probability of
        the sequence on the output of network
        """
        # Encode sequence to binary
        binary, _, _ = self.transformer.prepare_aligned_msa_for_vae(seqs_dict)
        observing_probs = self._bench(binary)
        return observing_probs

    def prepareMusSigmas(self, data):
        msa_weight = np.ones(data.shape[0]) / data.shape[0]
        msa_weight = msa_weight.astype(np.float32)
        data_t = data.astype(np.float32)
        rand_keys = ['k_'.format(i) for i in range(data.shape[0])]
        mus, sigmas = self.vae_handler.propagate_through_VAE(data_t, weights=msa_weight, keys=rand_keys)
        return mus, sigmas

    def _bench(self, data):
        mus, sigmas = self.prepareMusSigmas(data)
        return self._sample(mus, sigmas, data)

    def _sample(self, mus, sigmas, data):
        """
        Sample for each q(Z|X) for 10 000 times and make average
            1/N * SUM(p(X,Zi)/q(Zi|X))
        """
        N = 5
        probs = []  # marginal propabilities

        print('=' * 80)
        print(' Benchmark message : Sampling process has begun...')

        # Sample for mus and sigmas for N times
        batch_size = 32
        num_batches = mus.shape[0] // batch_size + 1
        for idx_batch in range(num_batches):
            if (idx_batch + 1) % 10 == 0:
                print("\t idx_batch: {} out of {}".format(idx_batch, num_batches))
            mus_b = mus[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            sigmas_b = sigmas[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            mus_b = torch.from_numpy(mus_b)
            sigmas_b = torch.from_numpy(sigmas_b)
            with torch.no_grad():
                decoded_seq = self.vae_handler.decode_z_marginal_probability(mus_b, sigmas_b, N)
                # Calculate marginal probability for each generated sequence
                chunks = [decoded_seq[x:x + N] for x in range(0, len(decoded_seq), N)]
                for i, ch in enumerate(chunks):
                    # Remember chunks are from more than one mu so compare with correct orignal sequence
                    orig = get_aas(data[(idx_batch * batch_size) + i])
                    sum_p_X_Zi = 0
                    for c in ch:
                    # SUM(p(X,Zi)/q(Zi|X))
                        sum_p_X_Zi += marginal(c, orig)
                    # 1/N * SUM(p(X,Zi)/q(Zi|X))
                    marginal_prob = sum_p_X_Zi / N
                    probs.append(marginal_prob)
        return probs

    def _generate_negative(self, count, s_len, profile_data):
        """Generate random sequences by the profile of family"""
        profile = self._get_profile(profile_data)
        K = self.positive.shape[2]
        D = np.identity(K)
        rand_seq_binary = np.zeros((count, s_len, K))
        for i in range(count):
            # Get profile sequence
            prof_seq = []
            for j in range(s_len):
                r = random.random()
                ssum = 0
                for aa, prob in enumerate(profile[:, j]):
                    ssum += prob
                    if r < ssum:
                        prof_seq.append(aa)
                        break
            rand_seq_binary[i, :, :] = D[prof_seq]
        print(' Benchmark message : Negative control {} samples generated...'.format(count))
        return rand_seq_binary

    def _get_profile(self, msa_binary):
        """
            Generate probabilistic profile of the MSA for further generation
            Profile format:
                    |-------len(msa)----|
                    0.2 0.3 0.15 ... 0.4
            cnt AA  0.4 0.3 0.15 ... 0.4
                    0.4 0.4 0.7  ... 0.2

                columns sum to one
        """
        msa_prof_file = self.setuper.pickles_fld + "/msa_profile.pkl"
        # check if msa_profile exists
        if os.path.exists(msa_prof_file) and os.path.getsize(msa_prof_file) > 0:
            print(' Benchmark message : Profile file exists in {}. Loading that file.'.format(msa_prof_file))
            with open(msa_prof_file, 'rb') as file_handle:
                profile = pickle.load(file_handle)
            return profile
        print(' Benchmark message : Creating the profile of MSA')
        # Convert MSA from binary to number coding
        get_aas = lambda xs: [np.where(aa_bin == 1)[0] for aa_bin in xs]
        msa = []
        for s in msa_binary:
            msa.append(get_aas(s))
        msa = np.array(msa).reshape((msa_binary.shape[0], msa_binary.shape[1]))
        profile = np.zeros((msa_binary.shape[2], msa.shape[1]))
        for j in range(msa.shape[1]):
            aa_type, aa_counts = np.unique(msa[:, j], return_counts=True)
            aa_sum = sum(aa_counts)
            for i, aa in enumerate(aa_type):
                profile[aa, j] = aa_counts[i] / aa_sum
        with open(msa_prof_file, 'wb') as file_handle:
            pickle.dump(profile, file_handle)
        return profile

    def _store_marginals(self, marginal_t, marginal_p, marginal_n):
        filename = self.setuper.high_fld + '/{}_marginals_benchmark.csv'.format(self.setuper.model_name)
        print(' Benchmark message: Storing marginals probabilities in {}'.format(filename))
        with open(filename, 'w', newline='') as file:
            # Store in csv file
            writer = csv.writer(file)
            writer.writerow(["Number", "Train", "Positive", "Negative", "Straight ancestors"])
            for i, (name, seq, prob) in enumerate(zip(marginal_t, marginal_p, marginal_n)):
                if i < len(self.ancestors_probs):
                    writer.writerow([i, name, seq, prob, self.ancestors_probs[i]])
                else:
                    writer.writerow([i, name, seq, prob])

    def model_generative_ability(self, data):
        """
        For whole training data set check how data
        points are biased by neighbouring data points
        """
        print(' Benchmark message : Measuring of model generative ability has started')
        data = data.astype(np.float32)
        batch_size = 64
        num_batches = data.shape[0] // batch_size + 1
        probs = []
        pairwise_score = []
        for idx_batch in range(num_batches):
            if (idx_batch + 1) % 10 == 0:
                print(" idx_batch: {} out of {}".format(idx_batch, num_batches))
            d = data[idx_batch * batch_size:(idx_batch + 1) * batch_size]
            # Convert original binary to AA for pairwise analyse
            originals = [get_aasR(it) for it in d]
            # Get marginal probability to see sequence
            binary = d.reshape((d.shape[0], -1))
            binary = torch.from_numpy(binary)
            probs.append(self.vae_handler.get_marginal_probability(binary))
            # Count pairwise generative ability of sequences
            mus_b, sigmas_b = self.prepareMusSigmas(d)
            mus_b, sigmas_b = torch.from_numpy(mus_b), torch.from_numpy(sigmas_b)
            generated = self.vae_handler.decode_z_marginal_probability(mus_b, sigmas_b, -1)
            for orig, gen in zip(originals, generated):
                pairwise_score.append(marginal(gen, orig))
        # Get pairwise score for query
        ref_binary = self.transformer.get_binary_by_key(self.setuper.query_id)
        orig_query = get_aasR(list(ref_binary))
        mu, sigma = self.prepareMusSigmas(ref_binary[np.newaxis, :, :])
        mu, sigma = torch.from_numpy(mu), torch.from_numpy(sigma)
        gen_query = self.vae_handler.decode_z_marginal_probability(mu, sigma, -1)
        query_score = marginal(gen_query[0], orig_query)

        print("\tModel generative ability value:", sum(probs)/num_batches)

        filename = self.setuper.high_fld + '/generative.txt'
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        hs = open(filename, append_write)
        hs.write(" Model with C = {0}, D = {1}, generative value = {2} "
                 " pairwise = {3}, DHAA pairwise score = {4}, layers: {5}, Decay = {6}\n".format(self.setuper.C,
                                                                self.setuper.dimensionality,
                                                                (sum(probs)/num_batches),
                                                                (sum(pairwise_score)/data.shape[0]),
                                                                 query_score, self.setuper.layersString,
                                                                 self.setuper.decay))
        hs.close()

    def test_loglikelihood(self):
        """Just pick point from the latent space and then generate sequence and compare loglikelihood"""
        seq = self.vae_handler.decode_z_to_aa_dict([(float(-66), float(-66))], "testing")
        binary, _, _ = self.binaryConv.prepare_aligned_msa_for_Vae(seq)
        binary = binary.astype(np.float32)
        binary = binary.reshape((binary.shape[0], -1))
        binary = torch.from_numpy(binary)
        print(" Marginal probs for test is: ", self.vae_handler.get_marginal_probability(binary))
        marginal = lambda gen, orig: sum([1 if g == o else 0 for g, o in zip(gen, orig)]) / len(orig)
        s = seq['testing'].copy()
        s[2] = 'X'
        print("Normal comparison", marginal(s, seq['testing']))


if __name__ == '__main__':
    tar_dir = CmdHandler()
    down_MSA = Downloader(tar_dir)
    with open(tar_dir.pickles_fld + '/positive_control.pkl', 'rb') as file_handle:
        positive = pickle.load(file_handle)

    with open(tar_dir.pickles_fld + '/training_set.pkl', 'rb') as file_handle:
        train_set = pickle.load(file_handle)

    # Prepare straight ancestors for making its distribution
    from mutagenesis import MutagenesisGenerator as Mutagenesis
    mut = Mutagenesis(setuper=tar_dir, num_point_mut=tar_dir.mut_points)
    _, ancs, probs = mut.get_straight_ancestors()

    b = Benchmarker(positive, train_set, tar_dir, generate_negative=True, ancestor_probs=probs)
    #b.test_loglikelihood()
    # b.model_generative_ability(train_set)
    b.make_bench()
    # Highlight training , positive, straight ancestors and negative control
    neg_mus, _ = b.prepareMusSigmas(b.negative)
    tr_subset_mus, _ = b.prepareMusSigmas(b.train_data)
    pos_mus, _ = b.prepareMusSigmas(b.positive)
    anc_npa = np.asarray(ancs, dtype=np.float32)[0::10, :] # Subsampling
    mut_names = ['Positive', 'Negative', 'Training', 'Ancestors']

    h = Highlighter(tar_dir)
    h.highlight_mutants([], [], mutants=[pos_mus, neg_mus, tr_subset_mus, anc_npa], mut_names=mut_names,
                        file_name='BenchmarkSets')
    # with open(tar_dir.pickles_fld + '/seq_msa_binary.pkl', 'rb') as file_handle:
    #     train_set = pickle.load(file_handle)
    # iden = np.identity(train_set.shape[2])
    # pos = np.zeros((2, train_set.shape[1], train_set.shape[2]))
    # for p in pos:
    #     for i in range(len(p)):
    #         p[i] = iden[i % train_set.shape[2]]
    # b = Benchmarker(pos, train_set, tar_dir)
    # b.make_bench()
