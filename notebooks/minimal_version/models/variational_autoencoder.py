__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/03/28 02:50:08"
__description__ = " This interface serves as a template for new VAE models variants as Wasserstein, " \
                  " providing all needed methods for further analysis"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Conditional variational autoencoder class, in model parameters you have to specify number of classes for label
    """

    def __init__(self, model_params):
        super(VariationalAutoencoder, self).__init__()

        # num of amino acid types
        self.num_aa_type = model_params["aa_count"]

        # dimension of latent space
        self.dim_latent_vars = model_params["lat_dim"]

        # dimension of binary representation of sequences
        self.dim_msa_vars = model_params["in_out_size"]

        # num of hidden neurons in encoder and decoder networks
        self.hidden_units = model_params["hidden_units"]

        # conditional label dim
        self.label_dim = model_params["label_dim"]

        # encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_msa_vars + self.label_dim, self.hidden_units[0]))
        for i in range(1, len(self.hidden_units)):
            self.encoder_linears.append(nn.Linear(self.hidden_units[i - 1], self.hidden_units[i]))
        self.encoder_mu = nn.Linear(self.hidden_units[-1], self.dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(self.hidden_units[-1], self.dim_latent_vars, bias=True)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(self.dim_latent_vars + self.label_dim, self.hidden_units[0]))
        for i in range(1, len(self.hidden_units)):
            self.decoder_linears.append(nn.Linear(self.hidden_units[i - 1], self.hidden_units[i]))
        self.decoder_linears.append(nn.Linear(self.hidden_units[-1], self.dim_msa_vars))

    def encoder(self, x, c=None):
        """
        encoder transforms x into latent space z, c has to be encoded in to one hot before
        """
        h = x
        if c is not None:
            h = torch.cat([x, c], dim=1)
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z, c=None):
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        """
        h = z
        if c is not None:
            h = torch.cat([z, c], dim=1)
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

    def z_to_number_sequences(self, z, c=None):
        """
        Decode VAE result to protein sequence again. Get max value indices for each position.

        Returns protein sequence in number representation.
        Capable of decoding more sequences at once.
        """
        h = self.decoder(z, c)
        # Reshape vae output
        final_shape = () if z.dim() == 1 else tuple(z.size()[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(final_shape + (-1, self.num_aa_type))
        idxs = h.max(dim=-1).indices
        return idxs

    def decode_samples(self, mu, sigma, num_samples, c=None):
        """ Decode samples from latent space with variance given by VAE, if c is proposed behaves as CVAE """
        with torch.no_grad():
            store_shape = mu.shape[0]
            mu = mu.expand(num_samples, mu.shape[0], mu.shape[1])
            # Expand in specific way, stack same tensor next to each other
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            if c is not None:
                c = c.expand(num_samples, c.shape[0], c.shape[1])
                z = torch.cat([z, c], dim=2)
            # Decode it
            expanded_number_sequences = self.z_to_number_sequences(z)
            # Flat list and group same sequences in a row
            flatten_idxs = []
            for i in range(store_shape):
                for batch_of_sequences in expanded_number_sequences:
                    flatten_idxs.append(batch_of_sequences[i])
            return flatten_idxs

    def get_sequence_log_likelihood(self, x, likelihoods=False, c=None):
        """
        Compute the log likelihood of the input sequences.

        Args:
            x (Tensor): Input sequences.
            likelihoods (bool): If True, return the log likelihood for each sequence individually.
                                If False, return the average log likelihood across the batch.
            c (Tensor, optional): Conditioning variable. If provided, the model behaves as a CVAE.

        Returns:
            Tensor: Log likelihood for each sequence if `likelihoods=True`.
                    Average log likelihood across the batch if `likelihoods=False`.
        """
        with torch.no_grad():
            # Decode exact point in the latent space
            mu, sigma = self.encoder(x, c)
            z = mu

            # log p(x|z)
            log_p = self.decoder(z, c)
            log_PxGz = torch.sum(x * log_p, -1)
            # Return simple sum saying it is log probability of seeing our sequences
            if likelihoods:
                return log_PxGz
            return torch.sum(log_PxGz).double() / log_PxGz.shape[0]

    def residues_probabilities(self, x, c=None):
        """
        Get true probability of each position to be seen on the output. Return as numpy.
        if c value is set then behaves as CVAE
        """
        with torch.no_grad():
            z, _ = self.encoder(x, c)
            log_p = self.decoder(z, c)

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

    def compute_weighted_elbo(self, x, weight, c_fx_x=2, c=None):
        """
        TO BE OVERWRITTEN, for different learning models as Wasserstein
        c_fx_x - the trade off between latent space regularization parameter and reconstruction accuracy in flipped
                 order -> 1/c_fx_x
        """
        # sample z from q(z|x)
        mu, sigma = self.encoder(x, c)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z, c)
        log_PxGz = torch.sum(x * log_p, -1)

        # Set parameter for training
        # loss = (x - f(x)) - (1/C * KL(qZ, pZ)
        #      reconstruction    normalization
        #         parameter        parameter
        # The bigger C is more accurate the reconstruction will be
        # default value is 2.0
        c = 1 / c_fx_x

        # compute elbo
        elbo = log_PxGz - torch.sum(c * (sigma ** 2 + mu ** 2 - 2 * torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo * weight)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples, c=None):
        """
        c_fx_x - the trade off between latent space regularization parameter and reconstruction accuracy in flipped
                 order -> 1/c_fx_x
        """
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1])
            if c is not None:
                c = c.expand(num_samples, c.shape[0], c.shape[1])
                h = torch.cat([x, c], dim=2)
            else:
                h = x
            mu, sigma = self.encoder(h)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)

            if c is not None:
                z = torch.cat([z, c], dim=2)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5 * (eps) ** 2 -
                                 0.5 * torch.log(2 * z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo
