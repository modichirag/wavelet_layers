## Standard libraries
import numpy as np
from functools import partial
import sys, os, json, time
from typing import Sequence

## Imports for plotting
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import jax
from jax import grad, jit
import jax.numpy as jnp
from jax import random
import flax
from flax import linen as nn
from flax.training import train_state
import optax
#import torch, torchvision

# Local imports
sys.path.append('../../src/')
from dequant import Dequantization, VariationalDequantization
from layers import CouplingLayer, WaveletLayer
from cnn import GatedConvNet
from flow import ImageFlow
from utils import create_channel_mask, create_checkerboard_mask, softplus

sys.path.append('../')
from trainer import TrainerModule

from tools import power_spectrum

print("\nUsing which device : ", jax.lib.xla_bridge.get_backend().platform)
print()
############################


sys.path.append('../../dataloaders/')
from quijote2d import Quijote2d_dataloader

d, L = 256, 8
#d, L = 128, 7
#d, L = 32, 5
bs = 1000
dataloader = Quijote2d_dataloader(d=d, train_batch=64, train_workers=1, test_workers=1)
train_loader = dataloader.train_loader
test_loader = dataloader.test_loader
val_loader = dataloader.val_loader
debug = False

#####
n_vardeq_layers = 4
n_flow_layers = 8
c_hidden = 24
renorm = jnp.exp
num_epochs = 500

if debug:
    n_flow_layers = 2
    c_hidden = 6
    

model_name = 'waveletflow_b64'
ckpt_path = f'/mnt/ceph/users/cmodi/wavelet_layers/quijote2d_fid_{d}/{model_name}/ckpt/'
os.makedirs(ckpt_path, exist_ok=True)

def create_wavelet_flow(use_vardeq=True):
    flow_layers = []
    for i in range(n_flow_layers):
        flow_layers += [WaveletLayer(L=L, c_hidden=c_hidden, renorm=renorm)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=2),
                                      mask=create_checkerboard_mask(h=d, w=d, invert=(i%2==1)),
                                      c_in=1)]

    flow_model = ImageFlow(flow_layers)
    return flow_model



def init_model(model, x, seed=42):
    # Initialize model
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, flow_rng = jax.random.split(rng, 3)
    params = model.init(init_rng, x, flow_rng)['params']
    return params, rng



def callback(step, model, method, params, rng, ckpt_path):
    # rng = trainer.rng;
    sample, rng = model.apply({"params":params}, img_shape=[16, d, d, 1], rng=rng, method=method)

    fig_path = f'{ckpt_path}/figs/'
    os.makedirs(fig_path, exist_ok=True)
    samples_path = f'{ckpt_path}/samples/'
    os.makedirs(samples_path, exist_ok=True)

    fig, ax = plt.subplots(4, 4, figsize=(4, 4), sharex=True, sharey=True)
    for i in range(16):
        ax.flatten()[i].imshow(sample[i, ..., 0])
        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])
    plt.savefig(fig_path + f"/{step}")   
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
    for i in range(16):
        k, p = power_spectrum(sample[i, ..., 0], kmin=np.pi/bs/2, dk=np.pi/bs, boxsize=np.array([bs]*2))
        ax.plot(k, p)
    ax.loglog()
    ax.grid(which='both', lw=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')
    plt.savefig(fig_path + f"/pk_{step}")   
    plt.close()

    np.save(f"{samples_path}/{step}.npy", sample)
    return rng


#####
model = create_wavelet_flow(use_vardeq=True)
x_test = next(iter(val_loader))[0]
#x_test = jnp.array(np.random.normal(size=(2*32**2)).reshape(2, 32, 32, 1))
params, model_rng  = init_model(model, x_test)
param_count = sum(x.size for x in jax.tree_leaves(params))
print("Number of params : ", param_count)

# Initialize learning rate schedule and optimizer
lr = 1e-3
lr_schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=len(train_loader),
        decay_rate=0.99,
        end_value=0.01*lr
    )
optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at 1
        optax.adam(lr_schedule)
    )

# Initialize training state
fig_path = f'{ckpt_path}/figs/'
os.makedirs(fig_path, exist_ok=True)
samples_path = f'{ckpt_path}/samples/'
os.makedirs(samples_path, exist_ok=True)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
trainer = TrainerModule(f'{model_name}', state, model, nckpt=1, ckpt_path=ckpt_path)

if debug:
    trainer.train_model(val_loader, val_loader, num_epochs=num_epochs,
                    callback=partial(callback, model=model, method=model.sample, ckpt_path=ckpt_path))
else:
    trainer.train_model(train_loader, val_loader, num_epochs=num_epochs,
                    callback=partial(callback, model=model, method=model.sample, ckpt_path=ckpt_path))


