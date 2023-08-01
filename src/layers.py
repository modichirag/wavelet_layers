import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from jax import scipy as jscipy
from jax.nn.initializers import orthogonal

import numpy as np
from typing import Sequence, Any

from utils import haar_decompose_2D, haar_recompose_2D
from cnn import GatedConvNet


class ActNorm(nn.Module):
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    @nn.compact
    def __call__(self, x, ldj, rng, reverse=False):
        # Affine transformation
        
        log_weight = self.param('logweight', lambda rng, x: jnp.log(1.0 / (x.std(0) + 1e-6)), x)
        bias = self.param('bias', lambda rng, x: x.mean(0), x)

        if (not reverse):
            outputs = (x - bias) * jnp.exp(log_weight)
            log_det_jacobian = jnp.full(x.shape[:1], log_weight.sum())
            ldj += log_det_jacobian
            return outputs, ldj, rng
            
        else:
            outputs = x * jnp.exp(-log_weight) + bias
            log_det_jacobian = jnp.full(x.shape[:1], -log_weight.sum())
            ldj += log_det_jacobian
            return outputs, ldj, rng
            



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





class InvertibleLinear(nn.Module):
    """An implementation of an invertible linear layer from `Glow: Generative Flow with Invertible 1x1 Convolutions`
    (https://arxiv.org/abs/1605.08803).
    
    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    input_dim : int
    key : Any
    
    def setup(self):

        key = self.key
        rng = jax.random.split(key, 4)
        rngl, rngs, rngu = rng[1:]

        W = orthogonal()(rng[0], (self.input_dim, self.input_dim))
        self.P, L, U = jscipy.linalg.lu(W)
        #self.P = self.param('P', lambda rngp, x: x, P)
        self.L = self.param('L', lambda rngl, x: x, L)
        self.S = self.param('S', lambda rngs, x: x, jnp.diag(U))        
        self.U = self.param('U', lambda rngu, x: x, jnp.triu(U, 1))
        self.identity = jnp.eye(self.input_dim)
        
    def _forward(self, x):

        L = jnp.tril(self.L, -1) + self.identity
        U = jnp.triu(self.U, 1)
        W = self.P @ L @ (U + jnp.diag(self.S))
        outputs = x @ W
        log_det_jacobian = jnp.full(x.shape[:1], jnp.log(jnp.abs(self.S)).sum())
        return outputs, log_det_jacobian

    def _reverse(self, x):

        L = jnp.tril(self.L, -1) + self.identity
        U = jnp.triu(self.U, 1)
        W = self.P @ L @ (U + jnp.diag(self.S))
        outputs = x @ jscipy.linalg.inv(W)
        log_det_jacobian = jnp.full(x.shape[:1], - jnp.log(jnp.abs(self.S)).sum())
        return outputs, log_det_jacobian

    
    def __call__(self, x, ldj=None, rng=None, reverse=False):
        #if ldj is given, then this is treated as layer
        #if ldj is not given, then this is a submodule of another NF layer.
        #Then depending on the context, one might need to negate returned ldj in reverse mode. 
        if  not reverse:
            x, ldj_layer = self._forward(x)
            if ldj is not None: ldj += ldj_layer            
        else:
            x, ldj_layer = self._reverse(x)
            if ldj is not None: ldj += ldj_layer            

        if ldj is not None:
            return x, ldj, rng
        else:
            return x, ldj_layer
    



        
class WaveletLayer(nn.Module):
    
    L : int         # number of levels 
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    renorm : callable = jnp.exp
    nchannels : int = 1
    
    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [GatedConvNet(c_hidden=self.c_hidden, c_out=self.nchannels*6, num_layers=2)  for _ in range(self.L)]
        assert self.L == len(self.networks)
    
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
            #wh, wv, wd = jnp.split(self.renorm(wb[..., :3*self.nchannels]), 3, axis=-1)
            #bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            #logdetjac += jnp.sum(jnp.log(self.renorm(wb[..., :3*self.nchannels])), axis=[1, 2, 3])
            wh, wv, wd = jnp.split(jnp.exp(wb[..., :3*self.nchannels]), 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            logdetjac += jnp.sum(jnp.log(jnp.exp(wb[..., :3*self.nchannels])), axis=[1, 2, 3])
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
            #wh, wv, wd = jnp.split(self.renorm(wb[..., :3*self.nchannels]), 3, axis=-1)
            #bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            #logdetjac += jnp.sum(jnp.log(self.renorm(wb[..., :3*self.nchannels])), axis=[1, 2, 3])
            wh, wv, wd = jnp.split(jnp.exp(wb[..., :3*self.nchannels]), 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            logdetjac += jnp.sum(jnp.log(jnp.exp(wb[..., :3*self.nchannels])), axis=[1, 2, 3])
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





class WaveletLayer_sadapt(nn.Module):
    
    D : int
    L : int         # number of levels 
    key : Any       # rng key
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    nchannels : int = 1

    
    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [GatedConvNet(c_hidden=self.c_hidden, c_out=6*self.nchannels, num_layers=2)  for _ in range(self.L)]
        self.s_flow = InvertibleLinear(int((self.D/2**self.L)**2*self.nchannels), key=self.key)
        assert self.L == len(self.networks)
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (self.L, 3, self.nchannels))
    
    def _forward(self, x):
        details = []
        logdetjac = 0
        s = x*1.
        for nl in range(self.L):
            s, d = haar_decompose_2D(s)
            details.append(d)
        details = details[::-1]

        #adapt s
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac += lj_layer

        for i, detail in enumerate(details):
            wb = self.networks[i](s)
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            # Stabilize scaling output
            sfac = jnp.exp(self.scaling_factor[i]).reshape(1, 1, 3, self.nchannels)
            sh, sv, sd = jnp.split(sfac, 3, axis=-2)
            wh = nn.tanh(wh / sh) * sh
            wv = nn.tanh(wv / sv) * sv
            wd = nn.tanh(wd / sd) * sd
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            #
            h, v, d = detail
            h = h*jnp.exp(wh) + bh
            v = v*jnp.exp(wv) + bv
            d = d*jnp.exp(wd) + bd
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
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            # Stabilize scaling output
            sfac = jnp.exp(self.scaling_factor[i]).reshape(1, 1, 3, self.nchannels)
            sh, sv, sd = jnp.split(sfac, 3, axis=-2)
            wh = nn.tanh(wh / sh) * sh
            wv = nn.tanh(wv / sv) * sv
            wd = nn.tanh(wd / sd) * sd
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            h, v, d = detail
            h = (h-bh) * jnp.exp(-wh)
            v = (v-bv) * jnp.exp(-wv)
            d = (d-bd) * jnp.exp(-wd)
            detail = [h, v, d]
            details.append(detail)
        details = details[::-1]

        #adapt s reversed
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat, reverse=True)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac -= lj_layer  #because it is already negated in s_flow
        
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


class WaveletLayer_sadapt2(nn.Module):
    
    D : int
    L : int         # number of levels 
    key : Any       # rng key
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    s_flow : list
    nchannels : int 
    
    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [GatedConvNet(c_hidden=self.c_hidden, c_out=6*self.nchannels, num_layers=2)  for _ in range(self.L)]
        assert self.L == len(self.networks)
    
    def _forward(self, x):
        details = []
        logdetjac = 0
        s = x*1.
        for nl in range(self.L):
            s, d = haar_decompose_2D(s)
            details.append(d)
        details = details[::-1]

        #adapt s
        for ll in self.s_flow:
            s, lj_layer, _ = ll(s, 0., 0.)
            logdetjac += lj_layer

        for i, detail in enumerate(details):
            wb = self.networks[i](s)
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            # Stabilize scaling output
            # sfac = jnp.exp(self.scaling_factor[i]).reshape(1, 1, 3, self.nchannels)
            # sh, sv, sd = jnp.split(sfac, 3, axis=-2)
            # wh = nn.tanh(wh / sh) * sh
            # wv = nn.tanh(wv / sv) * sv
            # wd = nn.tanh(wd / sd) * sd
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            #
            h, v, d = detail
            h = h*jnp.exp(wh) + bh
            v = v*jnp.exp(wv) + bv
            d = d*jnp.exp(wd) + bd
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
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            # Stabilize scaling output
            # sfac = jnp.exp(self.scaling_factor[i]).reshape(1, 1, 3, self.nchannels)
            # sh, sv, sd = jnp.split(sfac, 3, axis=-2)
            # wh = nn.tanh(wh / sh) * sh
            # wv = nn.tanh(wv / sv) * sv
            # wd = nn.tanh(wd / sd) * sd
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            h, v, d = detail
            h = (h-bh) * jnp.exp(-wh)
            v = (v-bv) * jnp.exp(-wv)
            d = (d-bd) * jnp.exp(-wd)
            detail = [h, v, d]
            details.append(detail)
        details = details[::-1]

        #adapt s reversed
        for ll in self.s_flow[::-1]:
            s, lj_layer, _ = ll(s, 0., 0., reverse=True)
            logdetjac -= lj_layer  #because it is already negated in s_flow layers
        
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
    



class WaveletLayer_sadapt_noscaling(nn.Module):
    
    D : int
    L : int         # number of levels 
    key : Any       # rng key
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    nchannels : int = 1

    
    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [GatedConvNet(c_hidden=self.c_hidden, c_out=6*self.nchannels, num_layers=2)  for _ in range(self.L)]
        self.s_flow = InvertibleLinear(int((self.D/2**self.L)**2*self.nchannels), key=self.key)
        assert self.L == len(self.networks)
    
    def _forward(self, x):
        details = []
        logdetjac = 0
        s = x*1.
        for nl in range(self.L):
            s, d = haar_decompose_2D(s)
            details.append(d)
        details = details[::-1]

        #adapt s
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac += lj_layer

        for i, detail in enumerate(details):
            wb = self.networks[i](s)
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            #
            h, v, d = detail
            h = h*jnp.exp(wh) + bh
            v = v*jnp.exp(wv) + bv
            d = d*jnp.exp(wd) + bd
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
            wh, wv, wd = jnp.split(wb[..., :3*self.nchannels], 3, axis=-1)
            bh, bv, bd = jnp.split(wb[..., 3*self.nchannels:], 3, axis=-1)
            #
            logdetjac += jnp.sum(wh, axis=[1, 2, 3]) + \
                jnp.sum(wv, axis=[1, 2, 3]) + \
                jnp.sum(wd, axis=[1, 2, 3])
            h, v, d = detail
            h = (h-bh) * jnp.exp(-wh)
            v = (v-bv) * jnp.exp(-wv)
            d = (d-bd) * jnp.exp(-wd)
            detail = [h, v, d]
            details.append(detail)
        details = details[::-1]

        #adapt s reversed
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat, reverse=True)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac -= lj_layer  #because it is already negated in s_flow
        
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
    
    
class WaveletLayer_sadapt_permute(nn.Module):
    
    D : int
    L : int         # number of levels 
    permutations : list 
    key : Any       # rng key
    c_hidden : int  # NN to use in the flow for predicting mu and sigma
    nchannels : int = 1


    def setup(self):
        '''Setup conv-net for every level
        '''
        self.networks = [[GatedConvNet(c_hidden=self.c_hidden, c_out=2*self.nchannels, num_layers=1) for _ in range(3)] for _ in range(self.L)]
        self.s_flow = InvertibleLinear(int((self.D/2**self.L)**2*self.nchannels), key=self.key)
        assert self.L == len(self.networks)
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (self.L, 3, self.nchannels))
        
        
    def _forward(self, x):
        details = []
        logdetjac = 0
        s = x*1.
        for nl in range(self.L):
            s, d = haar_decompose_2D(s)
            details.append(d)
        details = details[::-1]
                
        networks = self.networks
        permutations = self.permutations

        #adapt s
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac += lj_layer

        for i, detail in enumerate(details):
            #h, v, d = [detail[p] for p in permutations[i][0]]
            tmp = jnp.stack(detail, axis=-1)
            h = jnp.take(tmp, permutations[i][0][0], -1)
            v = jnp.take(tmp, permutations[i][0][1], -1)
            d = jnp.take(tmp, permutations[i][0][2], -1)

            xinp = s*1.
            sfac = jnp.exp(self.scaling_factor[i, 0]).reshape(1, 1, 1, -1)
            wb = networks[i][0](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            h = h*jnp.exp(w) + b
            
            #
            xinp = jnp.concatenate([s, h], axis=-1)
            sfac = jnp.exp(self.scaling_factor[i, 1]).reshape(1, 1, 1, -1)
            wb = networks[i][1](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            v = v*jnp.exp(w) + b
            
            #
            xinp = jnp.concatenate([s, h, v], axis=-1)
            sfac = jnp.exp(self.scaling_factor[i, 2]).reshape(1, 1, 1, -1)
            wb = networks[i][2](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            d = d*jnp.exp(w) + b
            
            detail = [h, v, d]
            detail = [detail[p] for p in permutations[i][1]]
            s = haar_recompose_2D(s, detail)
            
        return s, logdetjac

    
    def _reverse(self, x):            
        details = []
        permutations = self.permutations[::-1]
        networks = self.networks[::-1]        
        logdetjac = 0.
        s = x*1.
        
        for i in range(self.L):
            s, detail = haar_decompose_2D(s)
            # h, v, d = detail
            #h, v, d = [detail[p] for p in permutations[i][0]]
            tmp = jnp.stack(detail, axis=-1)
            h = jnp.take(tmp, permutations[i][0][0], -1)
            v = jnp.take(tmp, permutations[i][0][1], -1)
            d = jnp.take(tmp, permutations[i][0][2], -1)

            xinp = jnp.concatenate([s, h, v], axis=-1)
            sfac = jnp.exp(self.scaling_factor[i, 2]).reshape(1, 1, 1, -1)
            wb = networks[i][2](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            d = (d - b)*jnp.exp(-w)

            xinp = jnp.concatenate([s, h], axis=-1)
            sfac = jnp.exp(self.scaling_factor[i, 1]).reshape(1, 1, 1, -1)
            wb = networks[i][1](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            v = (v - b)*jnp.exp(-w)

            xinp = s*1.
            sfac = jnp.exp(self.scaling_factor[i, 0]).reshape(1, 1, 1, -1)
            wb = networks[i][0](xinp)           
            w, b =  jnp.split(wb, 2, axis=-1)
            w = nn.tanh(w / sfac) * sfac
            logdetjac += jnp.sum(w, axis=[1, 2, 3])
            h = (h - b)*jnp.exp(-w)
            
            detail = [h, v, d]
            detail = [detail[p] for p in permutations[i][1]]
            details.append(detail)
        details = details[::-1]

        #adapt s reversed
        s_flat = jnp.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        s_flat, lj_layer = self.s_flow(s_flat, reverse=True)
        s = jnp.reshape(s_flat, s.shape)
        logdetjac -= lj_layer  #because it is already negated in s_flow
        
        for i, d in enumerate(details):
            s = haar_recompose_2D(s, d)
        return s, logdetjac


    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        """
        """
        # Affine transformation
        if  not reverse:
            z, ldj_layer = self._forward(z)
            ldj += ldj_layer
        else:
            z, ldj_layer = self._reverse(z)
            ldj -= ldj_layer

        return z, ldj, rng
    
    
