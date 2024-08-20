__author__ = "Pavel Kohout <pavel.kohout@recetox.muni.cz>"
__date__ = "2024/08/14 13:49:00"

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

from notebooks.minimal_version.latent_space import LatentSpace
# my libraries
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.utils import load_from_pkl, store_to_pkl
from notebooks.minimal_version.msa import MSA
from notebooks.minimal_version.models.ai_models import AIModel

# Lambda for p(X,Zi)/q(Zi|X) of generated and original, given as sum of equal positions to length of original sequence
marginal = lambda gen, orig: sum([1 if g == o else 0 for g, o in zip(gen, orig)]) / len(orig)


class Benchmarker:
    def __init__(self, run: RunSetup, samples=500):
        """
        Conditional param will provide N_CLASSES for one how encoding else None
        Without ancestors
        """

        self.run = run
        self.samples = samples
        positive_control_ids = load_from_pkl(os.path.join(run.pickles, "positive_control_ids.pkl"))
        self.positive_control = load_from_pkl(os.path.join(run.pickles, "positive_control.pkl"))

        test_size = self.positive_control.shape[0]
        self.binary_shape = self.positive_control[0].shape

        self.model_param = load_from_pkl(os.path.join(run.model, "model_params.pkl"))
        self.latent_space = LatentSpace(run)
        self.vae, conditional = self.latent_space.load_model()

        # Get training set
        binary_msa = load_from_pkl(os.path.join(run.pickles, "seq_msa_binary.pkl"))
        training_ids = np.concatenate(np.array(load_from_pkl(os.path.join(run.pickles, "idx_subsets.pkl")), dtype=object).flatten())
        # print("Overlap", [a for a in positive_control_ids if a in training_ids])
        training_ids = training_ids[:test_size]  # same amount of samples
        training_ids.sort()
        self.train_data = binary_msa[training_ids]
        # print("Training first ", training_ids[:5])


        # Get negative set
        self.negative = self._generate_negative(count=test_size, s_len=self.positive_control.shape[1],
                                                profile_data=binary_msa)

        # Labels
        self.train_labels = None
        self.positive_labels = None
        self.negative_labels = None
        if conditional:
            all_labels = load_from_pkl(os.path.join(self.run.conditional_data, "conditional_labels_to_categories.pkl"))
            self.train_labels = all_labels[training_ids]
            self.positive_labels = all_labels[positive_control_ids]
            # Assign labels from whole dataset randomly to the generated sequences
            random_ids = np.random.choice(np.arange(0, binary_msa.shape[0]), size=test_size, replace=False)
            self.negative_labels = all_labels[random_ids]

        # Store stats
        os.makedirs(self.run.benchmark, exist_ok=True)
        self.bench_dir = run.benchmark
        self.bench_file = os.path.join(run.benchmark, "benchmark_data.pkl")
        # Ignore deprecated errors
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def _sample(self, data, c_labels):
        """
        Sample for each q(Z|X) for 10 000 times and make average
            1/N * SUM(p(X,Zi)/q(Zi|X))
        """
        # print("sample", data.shape)
        probabilities = []
        with torch.no_grad():
            for i, d in enumerate(data):
                c = c_labels[i].unsqueeze(0) if c_labels is not None else None
                mus, sigma = self.vae.encoder(d.unsqueeze(0), c)
                numerical_sequences = self.vae.decode_samples(mus, sigma, self.samples, c)
                or_reshaped = d.reshape(-1, self.binary_shape[-1])
                original_sequence = MSA.binary_to_numbers_coding(or_reshaped)
                sum_p_X_Zi = 0
                for decoded in numerical_sequences:
                    sum_p_X_Zi += marginal(decoded, original_sequence)
                probabilities.append(sum_p_X_Zi / self.samples)
        return probabilities

    def bench_dataset(self):
        # print("STATS")
        # print(self.positive_control.shape, MSA.binaries_to_tensor(self.positive_control).shape)
        #
        # input_Q = MSA.binaries_to_tensor(self.train_data[0])
        # print(input_Q[:42])
        # z_query, _ = self.vae.encoder(input_Q, c=None)
        # z_query = torch.tensor([-4.00096035, 1.92028594])
        # rec = self.vae.z_to_number_sequences(z_query)
        # print("QUERY")
        # print(MSA.number_to_amino({"query": MSA.binary_to_numbers_coding(self.train_data[0])}))
        # print("RECON")
        # print(z_query, MSA.number_to_amino({"rec": rec.numpy()}))
        # qury = MSA.binary_to_numbers_coding(self.train_data[0])
        #
        #
        # print(marginal(rec, qury))
        # exit(0)

        marginals_positive = self._sample(MSA.binaries_to_tensor(self.positive_control), self.positive_labels)
        marginals_train = self._sample(MSA.binaries_to_tensor(self.train_data), self.train_labels)
        marginals_negative = self._sample(MSA.binaries_to_tensor(self.negative), self.negative_labels)

        self._store_marginals(marginals_train, marginals_positive, marginals_negative)

        mean_p = sum(marginals_positive) / len(marginals_positive)
        mean_n = sum(marginals_negative) / len(marginals_negative)
        mean_t = sum(marginals_train) / len(marginals_train)

        # Plot it
        plt.style.use('seaborn-deep')
        # Prepare dataframe
        datasets = []
        probabilities = []
        for dataset, probs in [("Positive", marginals_positive), ("Negative", marginals_negative),
                               ("Training", marginals_train)]:
            for p in probs:
                datasets.append(dataset)
                probabilities.append(p * 100)
        data_dict = {"Dataset": datasets, "Probabilities": probabilities}
        dataFrame = pd.DataFrame.from_dict(data_dict)
        sns_plot = sns.histplot(data=dataFrame, x="Probabilities", hue="Dataset", multiple="dodge", shrink=.8,
                                color=["green", "black", "red"])

        plt.xlabel('% reconstruction from samples')
        plt.ylabel('Density')
        plt.title(r'Benchmark histogram $\mu={0:.2f},{1:.2f},{2:.2f}$'.format(mean_n, mean_p, mean_t))

        save_path = os.path.join(self.bench_dir, 'benchmark_density.png')
        print("  Benchmark message : Benchmark histogram saving to", save_path)
        sns_plot.figure.savefig(save_path)

        print(' Benchmark message : Benchmark results:')
        print('\tpositive mean: \t', mean_p)
        print('\tnegative mean: \t', mean_n)
        print('\ttrain data mean: \t', mean_t)

        bench_data_dict = {
            "mean_p": mean_p,
            "mean_n": mean_n,
            "mean_t": mean_t,
            "data_dict": data_dict
        }
        with open(self.bench_file, "wb") as file_handle:
            pickle.dump(bench_data_dict, file_handle)

    def _generate_negative(self, count, s_len, profile_data):
        """ Generate random sequences by the profile of family """
        profile = self._get_profile(profile_data)
        K = self.positive_control.shape[2]
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
        msa_prof_file = os.path.join(self.run.pickles, "msa_profile.pkl")
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
        filename = os.path.join(self.bench_dir, 'marginals_benchmark.csv')
        print(' Benchmark message: Storing marginals probabilities in {}'.format(filename))
        with open(filename, 'w', newline='') as file:
            # Store in csv file
            writer = csv.writer(file)
            writer.writerow(["Number", "Train", "Positive", "Negative"])
            for i, (name, seq, prob) in enumerate(zip(marginal_t, marginal_p, marginal_n)):
                writer.writerow([i, name, seq, prob])
