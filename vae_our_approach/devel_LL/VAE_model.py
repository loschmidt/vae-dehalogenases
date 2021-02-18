__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/16 02:50:08"

"""Modified by Pavel Kohout"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MSA_Dataset(Dataset):
    '''
    Dataset class for multiple sequence alignment.
    '''
    
    def __init__(self, seq_msa_binary, seq_weight, seq_keys):
        '''
        seq_msa_binary: a two dimensional np.array. 
                        size: [num_of_sequences, length_of_msa*num_amino_acid_types]
        seq_weight: one dimensional array. 
                    size: [num_sequences]. 
                    Weights for sequences in a MSA. 
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seq_keys: name of sequences in MSA
        '''
        super(MSA_Dataset).__init__()
        self.seq_msa_binary = seq_msa_binary
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys
        
    def __len__(self):
        assert(self.seq_msa_binary.shape[0] == len(self.seq_weight))
        assert(self.seq_msa_binary.shape[0] == len(self.seq_keys))        
        return self.seq_msa_binary.shape[0]
    
    def __getitem__(self, idx):
        return self.seq_msa_binary[idx, :], self.seq_weight[idx], self.seq_keys[idx]

class VAE(nn.Module):
    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        super(VAE, self).__init__()

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
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))

    def encoder(self, x):
        '''
        encoder transforms x into latent space z
        '''
        
        h = x
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        '''
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        '''
        
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))
        
        #h = torch.reshape(h, fixed_shape + (-1, self.num_aa_type))        
        log_p = F.log_softmax(h, dim = -1)
        log_p = log_p.view(fixed_shape + (-1,))
        #log_p = torch.reshape(log_p, fixed_shape + (-1,))
        
        # h = h.view(h.size(0), -1, self.num_aa_type)
        # log_p = F.log_softmax(h, dim = 2)
        # log_p = log_p.view(log_p.size(0), -1)
        
        return log_p

    def decoder_seq(self, z):
        '''Decode VAE result to protein sequence again. Get max value indice for each position'''
        h = z
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)
        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))
        h = F.softmax(h, dim=-1)
        idxs = (h.max(dim=-1).indices).tolist()
        return idxs

    def decode_samples(self, mu, sigma, num_samples):
        with torch.no_grad():
            store_shape = mu.shape[0]
            mu = mu.expand(num_samples, mu.shape[0], mu.shape[1])
            # Expand in specific way, stack same tensor next to each other
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            # Decode it
            h = z
            for i in range(len(self.decoder_linears) - 1):
                h = self.decoder_linears[i](h)
                h = torch.tanh(h)
            h = self.decoder_linears[-1](h)
            # And now convert to numbers from one hot encoding
            idxs = []
            for i in range(store_shape):
                for h_s in h:
                    fixed_shape = tuple(h_s.shape[0:-1])
                    h_s = torch.unsqueeze(h_s, -1)
                    h_s = h_s.view(fixed_shape + (-1, self.num_aa_type))
                    idxs.append((h_s.max(dim=-1).indices).tolist()[i])
            #print(len(idxs), idxs[0], 'to je velikost po samplovani')
            return idxs

    def marginal_sequence(self, x, num_samples=1000):
        with torch.no_grad():
            ## Decode exact point in the latent space
            mu, sigma = self.encoder(x)
            z = mu

            ## log p(x|z)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            ## Return simple sum saying it is log probability of seeing our sequences
            return torch.sum(log_PxGz).double() / log_PxGz.shape[0]

    def compute_weighted_elbo(self, x, weight, c_fx_x=2):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, -1)

        # Set parameter for training
        # loss = (x - f(x)) - (1/C * KL(qZ, pZ)
        #      reconstruction    normalization
        #         parameter        parameter
        # The bigger C is more accurate the reconstruction will be
        # default value is 2.0
        c = 1/c_fx_x

        ## compute elbo
        elbo = log_PxGz - torch.sum(c*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)
        
        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5*z**2 - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x*log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo        

    #def Wasserstein_distance_objective(self, x, weight):
        
    # def sample_latent_var(self, mu, sigma):
    #     eps = torch.ones_like(sigma).normal_()
    #     z = mu + sigma * eps
    #     return z
    
    # def forward(self, x):
    #     mu, sigma = self.encoder(x)
    #     z = self.sample_latent_var(mu, sigma)
    #     p = self.decoder(z)        
    #     return mu, sigma, p

    
# def loss_function(msa, weight, mu, sigma, p):    
#     cross_entropy = -torch.sum(torch.sum(msa*p, dim = 1) * weight)    
#     KLD = - 0.5 * torch.sum(torch.sum((1.0 + torch.log(sigma**2) - mu**2 - sigma**2), dim = 1) * weight)
#     return cross_entropy + KLD
