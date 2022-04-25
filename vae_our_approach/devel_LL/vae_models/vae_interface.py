__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/03/28 02:50:08"
__description__ = " This interface serves as a template for new VAE models variants as Wasserstein, " \
                  " providing all needed methods for further analysis"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEInterface(nn.Module):
    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        super(VAEInterface, self).__init__()

        ## num of amino acid types
        self.num_aa_type = num_aa_type

        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        ## dimension of binary representation of sequences
        self.dim_msa_vars = dim_msa_vars

        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        ## encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias=True)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """

        h = x.to(torch.float32)
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        """

        h = z.to(torch.float32)
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))

        log_p = F.log_softmax(h, dim=-1)
        log_p = log_p.view(fixed_shape + (-1,))

        return log_p

    def z_to_number_sequences(self, z):
        """
        Decode VAE result to protein sequence again. Get max value indices for each position.

        Returns protein sequence in number representation.
        Capable of decoding more sequences at once.
        """
        h = self.decoder(z)
        # Reshape vae output
        final_shape = () if z.dim() == 1 else tuple(z.size()[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(final_shape + (-1, self.num_aa_type))
        idxs = h.max(dim=-1).indices.tolist()
        return idxs

    def decode_samples(self, mu, sigma, num_samples):
        with torch.no_grad():
            store_shape = mu.shape[0]
            mu = mu.expand(num_samples, mu.shape[0], mu.shape[1])
            # Expand in specific way, stack same tensor next to each other
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            # Decode it
            expanded_number_sequences = self.z_to_number_sequences(z)
            # Flat list and group same sequences in a row
            flatten_idxs = []
            for i in range(store_shape):
                for batch_of_sequences in expanded_number_sequences:
                    flatten_idxs.append(batch_of_sequences[i])
            return flatten_idxs

    def get_sequence_log_likelihood(self, x, likelihoods=False):
        """ Return average likelihood for batch or multiple likelihoods one by one """
        with torch.no_grad():
            # Decode exact point in the latent space
            mu, sigma = self.encoder(x)
            z = mu

            # log p(x|z)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            # Return simple sum saying it is log probability of seeing our sequences
            if likelihoods:
                return log_PxGz
            return torch.sum(log_PxGz).double() / log_PxGz.shape[0]

    def residues_probabilities(self, x):
        """ Get true probability of each position to be seen on the output. Return as numpy. """
        with torch.no_grad():
            z, _ = self.encoder(x)
            log_p = self.decoder(z)

        # Reshape x
        final_shape = tuple(x.shape[0:-1])
        x = torch.unsqueeze(x, -1)
        x = x.view(final_shape + (-1, self.num_aa_type))
        # Reshape vae output
        log_p = torch.unsqueeze(log_p, -1)
        log_p = log_p.view(final_shape + (-1, self.num_aa_type))
        p = torch.exp(log_p)

        PxGz = torch.sum(p * x, dim=-1)

        if torch.cuda.is_available():
            PxGz = PxGz.cpu()

        return PxGz.detach().numpy()

    def compute_weighted_elbo(self, x, weight, c_fx_x=2):
        """
        TO BE OVERWRITTEN
        c_fx_x - the trade off between latent space regularization parameter and reconstruction accuracy in flipped
                 order -> 1/c_fx_x
        """

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        """
        TO BE OVERWRITTEN
        c_fx_x - the trade off between latent space regularization parameter and reconstruction accuracy in flipped
                 order -> 1/c_fx_x
        """
