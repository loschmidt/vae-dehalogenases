from abc import ABC

from notebooks.minimal_version.models.architectures.base import BaseVAEArchitecture
from torch import nn


class DenseEncDec(BaseVAEArchitecture):

    def __init__(self, model_params):
        super(DenseEncDec, self).__init__()

        # num of amino acid types
        self.num_aa_type = model_params["aa_count"]

        # dimension of latent space
        self.dim_latent_vars = model_params["lat_dim"]

        # dimension of binary representation of sequences
        self.dim_msa_vars = model_params["in_out_size"]

        # num of hidden neurons in encoder and decoder networks
        self.hidden_units = model_params["hidden_units"]

        # encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(self.dim_msa_vars, self.hidden_units[0]))
        for i in range(1, len(self.hidden_units)):
            self.encoder_linears.append(nn.Linear(self.hidden_units[i - 1], self.hidden_units[i]))
        self.encoder_mu = nn.Linear(self.hidden_units[-1], self.dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(self.hidden_units[-1], self.dim_latent_vars, bias=True)

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(self.dim_latent_vars, self.hidden_units[0]))
        for i in range(1, len(self.hidden_units)):
            self.decoder_linears.append(nn.Linear(self.hidden_units[i - 1], self.hidden_units[i]))
        self.decoder_linears.append(nn.Linear(self.hidden_units[-1], self.dim_msa_vars))

    def decoder(self) -> nn.ModuleList:
        return self.decoder_linears

    def encoder(self) -> nn.ModuleList:
        return self.encoder_linears
