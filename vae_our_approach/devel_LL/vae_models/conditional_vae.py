__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/03/28 02:50:08"
__description__ = " Implementation of conditonal variational autoencoder " \
                  "held out of original pipeline to keep it clear and save."

import torch.nn as nn

from vae_models.vae_interface import VAEInterface
from project_enums import SolubilitySetting


class CVAE(VAEInterface):
    """
    CVAE stands for conditional variational autoencoder.
    In our case we are working with variational autoencoder conditioned on soluble prediction by SoluProt
        the original input sequences are divided into 3 bins as following:
            1 - predicted low (<40%) solubility
            2 - predicted medium (40 - 72.5 %) solubility
            3 - predicted high (>72.5 %) solubility
    """
    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        input_size_conditional = dim_msa_vars + SolubilitySetting.SOLUBILITY_BINS.value
        super(CVAE, self).__init__(num_aa_type, dim_latent_vars, input_size_conditional, num_hidden_units)

        # latent space dimensionality
        latent_input_size = dim_latent_vars + SolubilitySetting.SOLUBILITY_BINS.value

        # Change decoder architecture
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(latent_input_size, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i - 1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))
        # super(CVAE, self).decoder_linears = self.decoder_linears
