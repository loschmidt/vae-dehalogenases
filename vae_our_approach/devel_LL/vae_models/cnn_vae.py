__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/03/17 10:30:00"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from msa_handlers.msa_preparation import MSA


class Flatten(nn.Module):
    def forward(self, input):
        # print("Flatten before ", input.shape)
        # print("Flatten after ", input.view(input.size(0), -1).shape)
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=64):
        # print("UnFlatten before ", input.shape)
        # print("UnFlatten after ", input.view(-1, size, 33, 6).shape)
        return input.view(-1, size, 10, 8)


class VaeCnn(nn.Module):
    """
    VAE with convolutional layers...
    Against classical VAE the input is 2D matrix (num_msa_position, aa_cnt) -> has to be reshaped
    """

    def __init__(self, dim_latent_vars, dim_msa_positions):
        super(VaeCnn, self).__init__()

        # num of amino acid types, plus gap symbol
        self.num_aa_type = len(MSA.aa) + 1

        # dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        # dimension of binary representation of sequences
        self.dim_msa_positions = dim_msa_positions

        # encoder
        self.encoder_conv = nn.Sequential(
            # transform via 128 filters
            nn.Conv2d(1, 128, 5, stride=(3, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(1, inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 1)),
            # transform to 96 filters
            nn.Conv2d(128, 96, 4, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(0.95, inplace=True),
            nn.MaxPool2d((2, 1), stride=1),
            # # last trasform to 64 channels
            nn.Conv2d(96, 64, 2, stride=(2, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(0.8, inplace=True),
            nn.MaxPool2d((2, 1), stride=1),

            Flatten(),
        )
        self.encoder_fc1 = nn.Linear(5120, 1024)
        self.encoder_mu = nn.Linear(1024, dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(1024, dim_latent_vars, bias=True)

        # decoder
        self.decoder_conv = nn.Sequential(
            UnFlatten(),
            nn.Conv2d(64, 96, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(0.8, inplace=True),
            nn.Upsample(scale_factor=(2, 1)),
            #
            nn.Conv2d(96, 128, kernel_size=(3, 2), stride=(1, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(0.95, inplace=True),
            nn.Upsample(scale_factor=(2, 2)),

            nn.Conv2d(128, 1, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ELU(1, inplace=True),
            nn.Upsample(size=(291, 21)),
        )
        self.decoder_fc1 = nn.Linear(dim_latent_vars, 1024)
        self.decoder_linear = nn.Linear(1024, 5120)

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """
        init_shape = x.shape[0:-1]
        x = x.view(init_shape + (-1, self.num_aa_type))
        x = x.unsqueeze(1)
        h = x.to(torch.float32)
        conv = self.encoder_conv(h)
        conv = self.encoder_fc1(conv)
        mu = self.encoder_mu(conv)
        sigma = torch.exp(self.encoder_logsigma(conv))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        """
        h = z.to(torch.float32)
        h = self.decoder_fc1(h)
        h = self.decoder_linear(h)
        h = self.decoder_conv(h)

        h = h.squeeze(1)
        h = h.view((h.shape[0], -1))
        fixed_shape = tuple(h.shape[0:-1])
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
        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z)
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

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        with torch.no_grad():
            x_expanded = torch.zeros((x.shape[0]*num_samples, x.shape[1]))
            print(x.shape)
            for i, x_sample in enumerate(x):
                x_expanded[i*num_samples: (i+1)*num_samples] = x_sample
            x = x_expanded.cuda()
            print(x.shape)
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5 * z ** 2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
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
