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
sys.path.append('../src/')
from dequant import Dequantization, VariationalDequantization
from layers import CouplingLayer, WaveletLayer
from cnn import GatedConvNet
from flow import ImageFlow
from utils import create_channel_mask, create_checkerboard_mask, softplus

from trainer import TrainerModule

print("Using which device : ", jax.lib.xla_bridge.get_backend().platform)
############################


sys.path.append('../dataloaders/')
from mnist import MNIST_dataloader

mnist_dataloader = MNIST_dataloader(train_workers=1, test_workers=1)
train_loader = mnist_dataloader.train_loader
test_loader = mnist_dataloader.test_loader
val_loader = mnist_dataloader.val_loader
train_exmp_loader = mnist_dataloader.train_exmp_loader

#####
n_vardeq_layers = 4
n_flow_layers = 8
c_hidden = 24
renorm = jnp.exp
L = 5
num_epochs = 500
model_name = 'waveletflow2'
ckpt_path = f'/mnt/ceph/users/cmodi/wavelet_layers/mnist/{model_name}/ckpt/'

def create_wavelet_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=16),
                                       mask=create_checkerboard_mask(h=32, w=32, invert=(i%2==1)),
                                       # mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
                                       c_in=1) for i in range(n_vardeq_layers)]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]
    for i in range(n_flow_layers):
        flow_layers += [WaveletLayer(L=L, c_hidden=c_hidden, renorm=renorm)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=2),
                                      mask=create_checkerboard_mask(h=32, w=32, invert=(i%2==1)),
                                      c_in=1)]

    flow_model = ImageFlow(flow_layers)
    return flow_model



def init_model(model, x, seed=42):
    # Initialize model
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, flow_rng = jax.random.split(rng, 3)
    params = model.init(init_rng, x, flow_rng)['params']
    return params, rng



def callback(step, model, method, params, rng, figpath):
    # rng = trainer.rng;
    sample, rng = model.apply({"params":params}, img_shape=[16, 32, 32, 1], rng=rng, method=method)

    fig, ax = plt.subplots(4, 4, figsize=(4, 4), sharex=True, sharey=True)
    for i in range(16):
        ax.flatten()[i].imshow(sample[i, ..., 0])
        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])
    plt.savefig(figpath + f"/{step}")
    plt.close()
    return rng


#####
model = create_wavelet_flow(use_vardeq=True)
x_test = next(iter(test_loader))[0]
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

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
trainer = TrainerModule(f'{model_name}', state, model, nckpt=1, ckpt_path=ckpt_path)

trainer.train_model(train_loader, val_loader, num_epochs=num_epochs,
                    callback=partial(callback, model=model, method=model.sample, figpath=fig_path))


