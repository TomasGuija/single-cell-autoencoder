"""Base class for autoencoder models."""

import abc
from typing import Dict, Tuple

import torch.nn as nn


class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
    """Clase base para los autoencoders."""

    # pylint: disable=W0221
    @abc.abstractmethod
    def forward(self, x):
        """Produce la predicción del autoencoder.

        Args:
            x: Input

        Returns:
            Tensor: Reconstrucción
        """
        pass
    
    def compute_loss(self, x):
        """Produce la pérdida del autoencoder.

        Args:
            x: Input 

        Returns:
            Tuple: Pérdida

        """
        pass

    @abc.abstractmethod
    def encode(self, x):
        """Produce la representación latente."""

    @abc.abstractmethod
    def decode(self, z):
        """Produce la reconstrucción a partir de la representación latente."""

