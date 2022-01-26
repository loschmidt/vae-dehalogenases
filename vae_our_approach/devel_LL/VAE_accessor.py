__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/06 14:33:00"

import os.path as path
import pickle
from os import path as path

import numpy as np
import torch
from scipy.spatial import distance
from torch import tensor
from torch.utils.data import DataLoader

from VAE_model import MSA_Dataset, VAE
from project_enums import Helper
from sequence_transformer import Transformer


class VAEAccessor:
    def __init__(self, setuper, model_name=None):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.model_name = model_name
        self.transformer = Transformer(setuper)
        self.vae = None
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

    def _prepare_model(self):
        # prepare model to mapping from highlighting files
        with open(self.pickle + "/seq_msa_binary.pkl", 'rb') as file_handle:
            msa_original_binary = pickle.load(file_handle)
        num_seq = msa_original_binary.shape[0]
        len_protein = msa_original_binary.shape[1]
        num_res_type = msa_original_binary.shape[2]

        msa_binary = msa_original_binary.reshape((num_seq, -1))
        msa_binary = msa_binary.astype(np.float32)

        # build a VAE model
        vae = VAE(num_res_type, self.setuper.dimensionality, len_protein * num_res_type, self.setuper.layers)
        print(self.model_name, "\n\n\n\n")
        if self.model_name:
            vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/" + self.model_name))

        # move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()

        self.vae = vae
        return vae, msa_binary, num_seq

    def _load_pickles(self):
        """ """
        with open(self.pickle + "/keys_list.pkl", 'rb') as file_handle:
            msa_keys = pickle.load(file_handle)
        with open(self.pickle + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        with open(self.pickle + "/training_alignment.pkl", 'rb') as file_handle:
            train_dict = pickle.load(file_handle)
        return seq_weight.astype(np.float32), msa_keys, train_dict

    def latent_space(self, check_exists=False):
        pickle_name = self.pickle + "/latent_space.pkl"
        if check_exists and path.isfile(pickle_name):
            print(" Latent space file already exists in {0}. \n"
                  " Loading and returning that file...".format(pickle_name))
            with open(pickle_name, 'rb') as file_handle:
                latent_space = pickle.load(file_handle)
            return latent_space['mu'], latent_space['sigma'], latent_space['key']

        msa_weight, msa_keys, _ = self._load_pickles()
        vae, msa_binary, num_seq = self._prepare_model()

        batch_size = num_seq
        train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
        train_data_loader = DataLoader(train_data, batch_size=batch_size)

        mu_list = []
        sigma_list = []
        keys_list = []
        for idx, data in enumerate(train_data_loader):
            msa, weight, key = data
            with torch.no_grad():
                if self.use_cuda:
                    msa = msa.cuda()
                mu, sigma = vae.encoder(msa)
                if self.use_cuda:
                    mu = mu.cpu().data.numpy()
                    sigma = sigma.cpu().data.numpy()
                mu_list.append(mu)
                sigma_list.append(sigma)
                keys_list.append(key)
        mu = np.vstack(mu_list)
        sigma = np.vstack(sigma_list)
        keys = np.vstack(keys_list)

        pickle_name = "/{}latent_space.pkl".format(self.model_name + "_" if self.model_name else "")
        with open(self.pickle + pickle_name, 'wb') as file_handle:
            pickle.dump({'key': keys, 'mu': mu, 'sigma': sigma}, file_handle)
        print(' The latent space was created....')
        return mu, sigma, keys

    def propagate_through_VAE(self, binaries, weights, keys):
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()

        train_data = MSA_Dataset(binaries, weights, keys)
        train_data_loader = DataLoader(train_data, batch_size=binaries.shape[0])
        mu_list = []
        sigma_list = []
        for idx, data in enumerate(train_data_loader):
            msa, weight, key = data
            with torch.no_grad():
                if self.use_cuda:
                    msa = msa.cuda()
                mu, sigma = vae.encoder(msa)
                if self.use_cuda:
                    mu = mu.cpu().data.numpy()
                    sigma = sigma.cpu().data.numpy()
                mu_list.append(mu)
                sigma_list.append(sigma)
        mu = np.vstack(mu_list)
        sigma = np.vstack(sigma_list)
        return mu, sigma

    def decode_z_to_aa_dict(self, lat_sp_pos, ref_name):
        """
        Methods decodes latent space coordinates to sequences in amino acid form.

        Returns dictionary of decoded sequences where first sequence name is ref_name
        """
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, msa_binary, num_seq = self._prepare_model()
        num_seqs = {}
        for i, z in enumerate(lat_sp_pos, 1):
            anc_name = 'ancestor_{}'.format(i) if i > 1 else ref_name
            z = tensor(z)
            if self.use_cuda:
                z = z.cuda()
            num_seqs[anc_name] = vae.z_to_number_sequences(z)
        # Convert from numbers to amino acid sequence
        anc_dict = self.transformer.back_to_amino(num_seqs)
        return anc_dict

    def decode_z_marginal_probability(self, z, sigma, samples):
        """
        Decode binary representation of z. Method optimized for
        marginal probability computation
        """
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                z = z.cuda()
                sigma = sigma.cuda()
            # indices already on cpu(not tensor)
            if samples == -1:
                # Decode exact value of z
                ret = vae.z_to_number_sequences(z)
            else:
                ret = vae.decode_samples(z, sigma, samples)
        return ret

    def get_marginal_probability(self, x, multiple_likelihoods=False):
        """
        This method returns the log softmax probability as it is obtained by VAE

        To return log likelihoods far each sequence position set multiple_likelihoods to True
        Otherwise it returns the average likelihood in the given batch of sequences
        """
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                x = x.cuda()
            # indices already on cpu(not tensor)
            ret = vae.get_sequence_log_likelihood(x, multiple_likelihoods)
        return ret

    def residues_probability(self, x):
        """ Return classic probability of each position to be seen on output. Input binary"""
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()
        return vae.residues_probabilities(x)

    def get_closest_dataset_sequence(self, x_coords):
        """
        Method searches for the closest sequence in the input dataset and returns its ID and sequence
        Parameters:
            x_coords : list of tuples with coordinates into the latent space.
                       The nearest points to these are found
        Returns:
            List of tuples [(id, sequence),...]
                id - key of nearest sequence
                sequence - sequence of the closest point from input dataset
            Note that the order in list corresponds to the order in x_coords
        """

        def closest_node(node, nodes):
            closest_index = distance.cdist(np.array([node]), nodes).argmin()
            return closest_index

        _, keys, train_dict = self._load_pickles()
        mu, _, _ = self.latent_space(check_exists=True)

        closest_sequences = []
        for coords in x_coords:
            key_ind = closest_node(coords, mu)
            seq_id = keys[key_ind]
            closest_sequences.append((seq_id, train_dict[seq_id]))
        return closest_sequences

    def get_identity_closest_dataset_sequence(self, sequences):
        """
        Method searches for the most identical sequence in the input dataset.
        Parameters:
            sequences : list of chars representing amino acid sequence level
        Returns:
            List of tuples [(id, identity),...]
            id - key of nearest sequence
            identity - percentage sequence identity with the corresponding sequence in sequences
        Note that the order in list corresponds to the order in sequences
        """
        from experiment_handler import ExperimentStatistics

        def closest_seq_identity(seq, dataset):
            max_identity, id_i = 0.0, 0
            for i, d in enumerate(dataset):
                seq_identity = ExperimentStatistics.sequence_identity(d, seq)
                if seq_identity > max_identity:
                    max_identity = seq_identity
                    id_i = i
            return id_i, max_identity

        _, keys, train_dict = self._load_pickles()
        train_seqs = train_dict.values()

        print(Helper.LOG_DELIMETER.value)
        closest_sequences = []
        for iteration, sequence in enumerate(sequences):
            i_key, closest_identity = closest_seq_identity(sequence, train_seqs)
            closest_sequences.append((keys[i_key], closest_identity))
            if iteration % (len(sequences) // 5) == 0:
                print(" Closest sequence search : iteration {} out of {}".format((iteration+1), len(sequences)))
        return closest_sequences
