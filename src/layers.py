import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax

import numpy as np
from typing import Sequence

from utils import haar_decompose_2D, haar_recompose_2D, softplus
from cnn import GatedConvNet




class CouplingLayer(nn.Module):
    network : nn.Module  # NN to use in the flow for predicting mu and sigma
    mask : np.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in : int  # Number of input channels

    def setup(self):
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (self.c_in,))

    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(jnp.concatenate([z_in, orig_img], axis=-1))
        s, t = nn_out.split(2, axis=-1)

        # Stabilize scaling output
        s_fac = jnp.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1,2,3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1,2,3])

        return z, ldj, rng




class WaveletLayer(nn.Module):
    
    L : int         # number of levels 
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    renorm : callable = softplus
    
    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [GatedConvNet(c_hidden=self.c_hidden, c_out=6, num_layers=2)  for _ in range(self.L)]
        assert self.L == len(self.networks)
        pass
    
    def _forward(self, x):
        details = []
        logdetjac = 0
        s = x*1.
        for nl in range(self.L):
            s, d = haar_decompose_2D(s)
            details.append(d)
        details = details[::-1]

        for i, detail in enumerate(details):
            wb = self.networks[i](s)
            wh, wv, wd = jnp.split(self.renorm(wb[..., :3]), 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3:], 3, axis=-1)
            logdetjac += jnp.sum(jnp.log(jnp.abs(self.renorm(wb[..., :3]))), axis=[1, 2, 3])
            h, v, d = detail
            h = h*wh + bh
            v = v*wv + bv
            d = d*wd + bd
            detail = [h, v, d]
            s = haar_recompose_2D(s, detail)

        return s, logdetjac

    
    def _reverse(self, x):            
        details = []
        logdetjac = 0.
        s = x*1.
    
        for i in range(self.L):
            s, detail = haar_decompose_2D(s)
            wb = self.networks[::-1][i](s)
            wh, wv, wd = jnp.split(self.renorm(wb[..., :3]), 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3:], 3, axis=-1)
            logdetjac += jnp.sum(jnp.log(jnp.abs(self.renorm(wb[..., :3]))), axis=[1, 2, 3])
            h, v, d = detail
            h = (h-bh)/wh
            v = (v-bv)/wv
            d = (d-bd)/wd
            detail = [h, v, d]
            details.append(detail)
        details = details[::-1]

        for i, d in enumerate(details):
            s = haar_recompose_2D(s, d)
        return s, logdetjac


    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            rng - PRNG state
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Affine transformation
        if  not reverse:
            z, ldj_layer = self._forward(z)
            ldj += ldj_layer
        else:
            z, ldj_layer = self._reverse(z)
            ldj -= ldj_layer

        return z, ldj, rng
