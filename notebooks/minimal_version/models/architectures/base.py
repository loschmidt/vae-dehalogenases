import abc
import torch.nn as nn


class BaseVAEArchitecture(abc.ABC, nn.Module):
    """
    Abstract base class for VAE architecture.
    Requires implementation of encoder and decoder methods.
    """

    def __init__(self):
        super(BaseVAEArchitecture, self).__init__()

    @abc.abstractmethod
    def encoder(self) -> nn.ModuleList:
        """
        Should return the encoder as an nn.ModuleList.
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def decoder(self) -> nn.ModuleList:
        """
        Should return the decoder as an nn.ModuleList.
        Must be implemented by subclasses.
        """
        pass

    def encoder_forward(self, x):
        """
        Forward pass through the encoder.
        """
        for layer in self.encoder():
            x = layer(x)
        return x

    def decoder_forward(self, x):
        """
        Forward pass through the decoder.
        """
        for layer in self.decoder():
            x = layer(x)
        return x
