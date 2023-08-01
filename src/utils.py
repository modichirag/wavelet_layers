import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

def get_permutations(n, L, rng):
    perms = []
    key = rng
    for i in range(L):
        key, key_use = jax.random.split(key, 2)
        p = jax.random.permutation(key_use, n)
        s = np.empty_like(p)
        s[p] = jnp.arange(p.size)
        perms.append([p, s])
    return perms
    
def create_checkerboard_mask(h, w, invert=False):
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(c_in, invert=False):
    mask = jnp.concatenate([
                jnp.ones((c_in//2,), dtype=jnp.float32),
                jnp.zeros((c_in-c_in//2,), dtype=jnp.float32)
            ])
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask


        
def haar_decompose_2D(x):
    '''One step application of haar wavelets (x_l -> s_(l+1), d_(l+1))
    https://www-users.cse.umn.edu/~jwcalder/5467/lec_dwt.pdf
    '''

    batch = x.shape[0]
    dims = x.shape[1]
    c_in  = x.shape[-1]
    A = lax.slice(x, (0, 0, 0, 0), (batch, dims, dims, c_in), strides=(1, 2, 2, 1)) #x[::2,::2]; 
    B = lax.slice(x, (0, 0, 1, 0), (batch, dims, dims, c_in), strides=(1, 2, 2, 1)) #x[::2,1::2]; 
    C = lax.slice(x, (0, 1, 0, 0), (batch, dims, dims, c_in), strides=(1, 2, 2, 1)) #x[1::2,::2]; 
    D = lax.slice(x, (0, 1, 1, 0), (batch, dims, dims, c_in), strides=(1, 2, 2, 1)) #x[1::2,1::2]
    s = (+ A + B + C + D) #approximation coeff
    h = (- A - B + C + D) #Horizontal detail
    v = (- A + B - C + D) #Vertical detail
    d = (+ A - B - C + D) #Diagonal detail

    return s, [h, v, d]

def haar_recompose_2D(s, details):
    '''One step application of inverse haar wavelets (s_l, d_l -> x_(l-1))
    '''

    h, v, d = details        
    batch = s.shape[0]
    dims = s.shape[1]
    c_in = s.shape[-1]

    x = jnp.zeros((batch, 2*dims, 2*dims, c_in), dtype=float)
    x = x.at[:, 0::2,0::2, :].set((s - h - v + d)/4)
    x = x.at[:, 0::2,1::2, :].set((s - h + v - d)/4)
    x = x.at[:, 1::2,0::2, :].set((s + h - v - d)/4)
    x = x.at[:, 1::2,1::2, :].set((s + h + v + d)/4)
    return x




def softplus(x, eps=1e-3):
    r"""Softplus activation function.

      Computes the element-wise function

      .. math::
        \mathrm{softplus}(x) = \log(1 + e^x)
    """
    return jnp.logaddexp(x, 0) + eps

