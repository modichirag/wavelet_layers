import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

import numpy as np
from typing import Sequence

class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """ This module applies a two-layer convolutional ResNet block with input gate """
    c_in : int  # Number of input channels
    c_hidden : int  # Number of hidden dimensions

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            ConcatELU(),
            nn.Conv(self.c_hidden, kernel_size=(3, 3)),
            ConcatELU(),
            nn.Conv(2*self.c_in, kernel_size=(1, 1))
        ])(x)
        val, gate = out.split(2, axis=-1)
        return x + val * nn.sigmoid(gate)


class GatedConvNet(nn.Module):
    c_hidden : int  # Number of hidden dimensions to use within the network
    c_out : int  # Number of output channels
    num_layers : int = 3 # Number of gated ResNet blocks to apply

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3, 3))]
        for layer_index in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden),
                       nn.LayerNorm()]
        layers += [ConcatELU(),
                   nn.Conv(self.c_out, kernel_size=(3, 3),
                   )]
                          #  kernel_init=nn.initializers.zeros)]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)
