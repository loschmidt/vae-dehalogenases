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

from VAE_model import MSA_Dataset
from vae_models.cnn_vae import VaeCnn
from vae_models.conditional_vae import CVAE
from vae_models.vae_interface import VAEInterface as VAE
from VAE_logger import Logger
from project_enums import Helper, SolubilitySetting
from sequence_transformer import Transformer


class VAEAccessor:
    def __init__(self, setuper, model_name=None):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.model_name = model_name
        self.transformer = Transformer(setuper)

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.vae, self.seq_cnt = self._prepare_model()

    def _prepare_model(self):
        # prepare model to mapping from highlighting files
        with open(self.pickle + "/seq_msa_binary.pkl", 'rb') as file_handle:
            msa_original_binary = pickle.load(file_handle)
        num_seq = msa_original_binary.shape[0]
        len_protein = msa_original_binary.shape[1]
        num_res_type = msa_original_binary.shape[2]

        # build a VAE model
        if self.setuper.convolution:
            vae = VaeCnn(self.setuper.dimensionality, len_protein)
        elif self.setuper.conditional:
            vae = CVAE(num_res_type, self.setuper.dimensionality, len_protein * num_res_type, self.setuper.layers)
        else:
            vae = VAE(num_res_type, self.setuper.dimensionality, len_protein * num_res_type, self.setuper.layers)
        if self.model_name:
            vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/" + self.model_name))

        # move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()

        return vae, num_seq

    def _load_pickles(self):
        """ """
        with open(self.pickle + "/keys_list.pkl", 'rb') as file_handle:
            msa_keys = pickle.load(file_handle)
        with open(self.pickle + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        with open(self.pickle + "/training_alignment.pkl", 'rb') as file_handle:
            train_dict = pickle.load(file_handle)
        if self.setuper.solubility_file and self.setuper.conditional:
            with open(self.setuper.pickles_fld + '/solubilities.pkl', 'rb') as file_handle:
                solubility = torch.from_numpy(pickle.load(file_handle))
        else:
            solubility = None
        return seq_weight.astype(np.float32), msa_keys, train_dict, solubility

    def latent_space(self, check_exists=False):
        pickle_name = self.pickle + "/latent_space.pkl"
        if check_exists and path.isfile(pickle_name):
            print(" Latent space file already exists in {0}. \n"
                  " Loading and returning that file...".format(pickle_name))
            with open(pickle_name, 'rb') as file_handle:
                latent_space = pickle.load(file_handle)
            return latent_space['mu'], latent_space['sigma'], latent_space['key']

        msa_weight, msa_keys, training_dict, solubility = self._load_pickles()
        msa_binary, weights, _ = self.transformer.sequence_dict_to_binary(training_dict)
        msa_binary = torch.from_numpy(msa_binary)
        mu, sigma = self.propagate_through_VAE(msa_binary, weights, msa_keys, solubility)

        pickle_name = "/{}latent_space.pkl".format(self.model_name + "_" if self.model_name else "")
        with open(self.pickle + pickle_name, 'wb') as file_handle:
            pickle.dump({'key': msa_keys, 'mu': mu, 'sigma': sigma}, file_handle)
        print(' The latent space was created....')
        return mu, sigma, msa_keys

    def propagate_through_VAE(self, binaries, weights, keys, c=None):
        """ Send binaries through encoder to latent space """
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, _ = self._prepare_model()

        solubility = self.get_conditional_label(binaries.shape[0], c)
        train_data = MSA_Dataset(binaries, weights, keys, solubility)
        train_data_loader = DataLoader(train_data, batch_size=binaries.shape[0])
        mu_list = []
        sigma_list = []
        for idx, data in enumerate(train_data_loader):
            msa, weight, key, cond = data
            cond = cond if self.setuper.conditional else None
            with torch.no_grad():
                if self.use_cuda:
                    msa = msa.cuda()
                    cond = cond.cuda() if cond is not None else cond
                mu, sigma = vae.encoder(msa, cond)
                if self.use_cuda:
                    mu = mu.cpu().data.numpy()
                    sigma = sigma.cpu().data.numpy()
                mu_list.append(mu)
                sigma_list.append(sigma)
        mu = np.vstack(mu_list)
        sigma = np.vstack(sigma_list)
        return mu, sigma

    def decode_z_to_aa_dict(self, lat_sp_pos, ref_name, query_pos=0):
        """
        Methods decodes latent space coordinates to sequences in amino acid form.
        @param lat_sp_pos  - latent space coordinates to be decoded to sequence (array)
        @param ref_name    - name for query sequence posed on query_pos index in 'lat_sp_pos'
        @param query_pos   - position of query sequence in 'lat_sp_pos' list

        Returns dictionary of decoded sequences where first sequence name is ref_name
        """
        # if CVAE sample with selected target
        solubility = self.get_conditional_label(len(lat_sp_pos))
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, num_seq = self._prepare_model()
        num_seqs = {}
        for i, z in enumerate(lat_sp_pos, 0):
            seq_name = 'ancestor_{}'.format(i - query_pos) if i > query_pos else "successor_{}".format(query_pos - i)
            anc_name = ref_name if i == query_pos else seq_name
            z = tensor(z)
            if self.use_cuda:
                z = z.cuda()
            num_seqs[anc_name] = vae.z_to_number_sequences(z, solubility[i] if solubility is not None else None)
        # Convert from numbers to amino acid sequence
        anc_dict = self.transformer.back_to_amino(num_seqs)
        return anc_dict

    def decode_z_to_number(self, z: np.ndarray) -> np.ndarray:
        """ Decode z from latent space and return amino acid in numbers """
        z = tensor(z)
        solubility = self.get_conditional_label(z.shape[0])
        if self.use_cuda:
            z = z.cuda()
            # solubility = solubility.cuda() if solubility else None
        sequences_in_numbers = self.vae.z_to_number_sequences(z, solubility)
        return np.array(sequences_in_numbers)

    def decode_z_marginal_probability(self, z, sigma, samples):
        """
        Decode binary representation of z. Method optimized for
        marginal probability computation
        """
        solubility = self.get_conditional_label(z.shape[0])
        vae = self.vae
        if vae is None:
            vae, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                z = z.cuda()
                sigma = sigma.cuda()
                # solubility = solubility.cuda() if solubility else None
            # indices already on cpu(not tensor)
            if samples == -1:
                # Decode exact value of z
                ret = vae.z_to_number_sequences(z, solubility)
            else:
                ret = vae.decode_samples(z, sigma, samples, solubility)
        return ret

    def get_marginal_probability(self, x, multiple_likelihoods=False):
        """
        This method returns the log softmax probability as it is obtained by VAE

        To return log likelihoods far each sequence position set multiple_likelihoods to True
        Otherwise it returns the average likelihood in the given batch of sequences
        """
        solubility = self.get_conditional_label(x.shape[0])
        vae = self.vae
        if vae is None:
            vae, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                x = x.cuda()
                # solubility = solubility.cuda() if solubility else None
            # indices already on cpu(not tensor)
            ret = vae.get_sequence_log_likelihood(x, multiple_likelihoods, solubility)
            if self.use_cuda:
                ret = ret.cpu()
        return ret

    def residues_probability(self, x):
        """ Return classic probability of each position to be seen on output. Input binary"""
        solubility = self.get_conditional_label(x.shape[0])
        vae = self.vae
        if vae is None:
            vae, _ = self._prepare_model()
        if self.use_cuda:
            x = x.cuda()
            # solubility = solubility.cuda() if solubility else None
        return vae.residues_probabilities(x, solubility)

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

        _, keys, train_dict, _ = self._load_pickles()
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

        _, keys, train_dict, _ = self._load_pickles()
        train_seqs = train_dict.values()

        print(Helper.LOG_DELIMETER.value)
        print("  Searching for the closest sequences from the input dataset")
        Logger.print_for_update(" Closest sequence search : iteration {} out of " + str(len(sequences)), value='0')
        closest_sequences = []
        for iteration, sequence in enumerate(sequences):
            i_key, closest_identity = closest_seq_identity(sequence, train_seqs)
            closest_sequences.append((keys[i_key], closest_identity))
            if iteration % (len(sequences) // 5) == 0:
                Logger.update_msg(value=str((iteration + 1)), new_line=False)
        Logger.update_msg(value=str(len(sequences)), new_line=True)
        return closest_sequences

    def get_conditional_label(self, cnt, numpy_flags=None):
        """ Generate conditional flags, by the template if numpy_flag provided otherwise the highest possible"""
        if self.setuper.conditional:
            if numpy_flags is not None:
                flags = numpy_flags
            else:
                flags = torch.tensor([SolubilitySetting.SOL_BIN_HIGH.value for _ in range(cnt)]).unsqueeze(1)
            if torch.is_tensor(flags):
                return flags.cuda() if self.use_cuda else flags
            return torch.from_numpy(flags).cuda() if self.use_cuda else flags
        return None
