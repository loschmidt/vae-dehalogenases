"""
Latent space accessor for VAEncestors
Encapsulates all methods for latent space operations: sampling, embedding and so on
"""
__author__ = "Pavel Kohout <pavel.kohout@recetox.muni.cz>"
__date__ = "2024/08/14 14:33:00"

import os
from typing import Tuple, List, Union, Dict

import numpy as np
from scipy.spatial import distance
import torch
from torch import tensor

from notebooks.minimal_version.models.ai_models import AIModel
from notebooks.minimal_version.msa import MSA
# My libraries
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.utils import load_from_pkl, store_to_pkl


class LatentSpace:
    def __init__(self, run: RunSetup):
        self.run = run
        self.pickle = run.pickles
        self.model_name = run.weights
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.vae, self.conditional = self.load_model()
        self.msa_embeddings = self.prepare_latent_embeddings()

    def load_model(self):
        """
        Load model and prepare it for latent space embedding
        :return: initialized VAE model
        """
        model_param = load_from_pkl(os.path.join(self.run.model, "model_params.pkl"))
        vae, conditional = AIModel(model_param).get_model()
        vae.load_state_dict(torch.load(self.model_name))
        # move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()
        return vae, conditional

    def prepare_latent_embeddings(self):
        """
        Embed all sequences from preprocessed MSA into the latent space
        :return:
        """
        embedding_file_path = os.path.join(self.run.mapping, "msa_embedding.pkl")
        print("Preparing new latent space")
        binary = load_from_pkl(os.path.join(self.run.pickles, "seq_msa_binary.pkl"))
        msa_keys = load_from_pkl(os.path.join(self.run.pickles, "keys_list.pkl"))
        conditional_labels = None
        if self.conditional:
            conditional_labels = load_from_pkl(
                os.path.join(self.run.conditional_data, "conditional_labels_to_categories.pkl"))
        if self.use_cuda:
            binary = binary.cuda()

        binaries_tensor = MSA.binaries_to_tensor(binary)
        with torch.no_grad():
            mu, sigma = self.vae.encoder(binaries_tensor, conditional_labels)
        embedding = {"keys": msa_keys, "mus": mu, "sigma": sigma}
        store_to_pkl(embedding, embedding_file_path)
        return embedding

    def key_to_embedding(self, key: Union[str, List[str]]) -> np.ndarray:
        """
        Map sequence ID to latent space coordinates
        :param key: which key/keys to find
        :return: mu coordinates
        """
        if isinstance(key, str):
            key = [key]
        np_array_mus = np.array([])
        for k in key:
            try:
                key_ind = self.msa_embeddings['keys'].index(k)
            except:
                print(f" Key to embedding mapping: {k} not found in dictionary")
                exit(2)
            np_array_mus = np.append(np_array_mus, np.array(self.msa_embeddings["mus"][key_ind]))
        return np_array_mus

    def encode(self, sequences: Union[torch.Tensor, List[str], np.ndarray, str, Dict[str, str]],
               c: torch.Tensor = None, is_binary=False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode sequences in sequence/number/binary representation into the latent space
        :param sequences: sequences to be encoded
        :param c: conditional labels
        :param is_binary: whether it is already binary representation
        :return: mu, sigma by the encoder of VAE
        """
        if sequences is None or (isinstance(sequences, List) and len(sequences) == 0):
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if not is_binary:
            if isinstance(sequences, str):
                sequences = [sequences]
            if isinstance(sequences, List) and isinstance(sequences[0], str):
                tmp_dict = {f'seq_{i}': seq for i, seq in enumerate(sequences)}
                sequences = MSA.aa_to_number(tmp_dict)
            if isinstance(sequences, Dict):
                sequences = MSA.aa_to_number(sequences)
            if isinstance(sequences, np.ndarray):
                sequences = MSA.number_to_binary(sequences.astype(np.int))
        if not isinstance(sequences, torch.Tensor):
            sequences = MSA.binaries_to_tensor(sequences)
        else:
            sequences = MSA.binaries_to_tensor(sequences)

        if c:
            if c.ndim == 1:
                c = c.unsqueeze(0)
            if c.shape[0] != sequences.shape[0]:
                c = c.repeat(sequences.shape[0], 1)

        with torch.no_grad():
            mus, sigmas = self.vae.encoder(sequences, c)
        return mus, sigmas

    def decode_z_to_aa_dict(self, latent_points: np.ndarray, prefix: str = "ancestor"):
        """
        Methods decodes latent space coordinates to sequences in amino acid form.
        @param latent_points - latent space coordinates to be decoded to sequence (array)
        :param prefix: prefix for naming purposes

        Returns dictionary of decoded sequences where a first sequence is decoding of the query
        """
        num_seqs = {}
        for i, z in enumerate(latent_points, 0):
            seq_name = f'{prefix}_{i}'
            z = tensor(z)
            if self.use_cuda:
                z = z.cuda()
            num_seqs[seq_name] = self.vae.z_to_number_sequences(z)
        # Convert from numbers to amino acid sequence
        anc_dict = MSA.number_to_amino(num_seqs)
        return anc_dict

    def decode_z_to_number(self, z: np.ndarray, c) -> np.ndarray:
        """ Decode z from latent space and return amino acid in numbers """
        z = tensor(z)
        if self.use_cuda:
            z = z.cuda()
        sequences_in_numbers = self.vae.z_to_number_sequences(z)
        return np.array(sequences_in_numbers)

    def decode_z_marginal_probability(self, z, sigma, samples):
        """
        Decode binary representation of z. Method optimized for
        marginal probability computation
        """
        solubility = self.get_conditional_label(z.shape[0])
        vae = self.vae
        if vae is None:
            vae, _ = self.load_model()
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

    def sequence_probabilities(self, binary) -> np.ndarray:
        """
        This method returns the log softmax probability as it is obtained by VAE
        :param binary: binary of sequences we want to evaluate
        :return: np.ndarray of shape (binary.shape[0])
        """
        with torch.no_grad():
            if self.use_cuda:
                binary = binary.cuda()
            # indices already on cpu(not tensor)
            log_likelihoods = self.vae.get_sequence_log_likelihood(binary, True)
            return log_likelihoods.cpu().detach().numpy()

    def residues_probability(self, x):
        """
        Get individual amino acid probabilities as seen be autoencoder
        :param x: binary representation
        :return:
        """
        if self.use_cuda:
            x = x.cuda()
        return self.vae.residues_probabilities(x)

    def get_closest_dataset_sequence(self, x_coords):
        """
        Method searches for the closest sequence in the input dataset and returns its ID and sequence
        Parameters:
            x_coords : list of tuples with coordinates into the latent space.
                       The nearest points to these are found
        Returns:
            List of sequence IDs closest to the corresponding points
            Note that the order in list corresponds to the order in x_coords
        """

        def closest_node(node, nodes):
            closest_index = distance.cdist(np.array([node]), nodes).argmin()
            return closest_index

        closest_sequences = []
        for coords in x_coords:
            key_ind = closest_node(coords, self.msa_embeddings['mus'])
            seq_id = self.msa_embeddings['keys'][key_ind]
            closest_sequences.append(seq_id)
        return closest_sequences
