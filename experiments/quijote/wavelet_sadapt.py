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
from layers import CouplingLayer, WaveletLayer_sadapt, InvertibleLinear, ActNorm
from cnn import GatedConvNet
from flow import ImageFlow
from utils import create_channel_mask, create_checkerboard_mask, softplus, get_permutations

sys.path.append('../')
from trainer import TrainerModule
from tools import power_spectrum

print("\nUsing which device : ", jax.lib.xla_bridge.get_backend().platform)
print()

############################


sys.path.append('../../dataloaders/')
from quijote2d import Quijote2d_dataloader

#D, L = 256, 6
#D, L = 128, 7
D, L = 64, 6
#D, L = 32, 5
bs = 1000
nchannels = 1
model_name = 'wavelet_sadapt_L6_new'

batch_size = 32
dataloader = Quijote2d_dataloader(d=D, train_batch=batch_size, train_workers=1, test_workers=1)
train_loader = dataloader.train_loader
test_loader = dataloader.test_loader
val_loader = dataloader.val_loader
debug = False

#####
n_flow_layers = 5
c_hidden = int(2*6*nchannels)

debug = False
if debug:
    n_flow_layers = 1
    c_hidden = 2
    

num_epochs = 500
ckpt_path = f'/mnt/ceph/users/cmodi/wavelet_layers/quijote2d_fid_{D}/{model_name}/ckpt/'
fig_path = f'{ckpt_path}/figs/'
os.makedirs(fig_path, exist_ok=True)
samples_path = f'{ckpt_path}/samples/'
os.makedirs(samples_path, exist_ok=True)

def create_wavelet_flow(seed_init=99):
    flow_layers = []
    key_init = random.PRNGKey(seed_init)
    for i in range(n_flow_layers):
        key_init, key_perm = random.split(key_init)
        flow_layers += [ActNorm()]
        permutations = get_permutations(n=3, L=L, rng=key_perm)
        flow_layers += [WaveletLayer_sadapt(D=D, L=L, permutations=permutations,
                                            nchannels=nchannels, c_hidden=c_hidden, key=key_init)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2*nchannels, c_hidden=2*nchannels),
                                      mask=create_checkerboard_mask(h=D, w=D, invert=(i%2==0)),
                                      c_in=nchannels)]
        # flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2*nchannels, c_hidden=2*nchannels),
        #                               mask=create_checkerboard_mask(h=D, w=D, invert=(i%2==1)),
        #                               c_in=nchannels)]
        
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
    sample, rng = model.apply({"params":params}, img_shape=[batch_size, D, D, 1], rng=rng, method=method)

    fig_path = f'{ckpt_path}/figs/'
    os.makedirs(fig_path, exist_ok=True)
    samples_path = f'{ckpt_path}/samples/'
    os.makedirs(samples_path, exist_ok=True)

    fig, ax = plt.subplots(4, 4, figsize=(4, 4), sharex=True, sharey=True)
    for i in range(min(16, batch_size)):
        ax.flatten()[i].imshow(sample[i, ..., 0])
        ax.flatten()[i].set_xticks([])
        ax.flatten()[i].set_yticks([])
    plt.savefig(fig_path + f"/{step}")   
    plt.close()

    try:
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
    except Exception as e:
        print("Exception in estimating power spectrum : ", e)

    np.save(f"{samples_path}/{step}.npy", sample)
    return rng




#####
model = create_wavelet_flow()
x_init = next(iter(train_loader))[0]
print("input data shape : ", x_init.shape)
params, model_rng  = init_model(model, x_init)
param_count = sum(x.size for x in jax.tree_leaves(params))
print("Number of params : ", param_count)
#print(jax.tree_map(lambda p: p.shape, params))

#sanity check of data by saving sample
fig, ax = plt.subplots(4, 4, figsize=(4, 4), sharex=True, sharey=True)
for i in range(min(16, x_init.shape[0])):
    ax.flatten()[i].imshow(x_init[i, ...])
    ax.flatten()[i].set_xticks([])
    ax.flatten()[i].set_yticks([])
plt.savefig(fig_path + f"/data")
plt.close()


####
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
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
trainer = TrainerModule(f'{model_name}', state, model, nckpt=1, ckpt_path=ckpt_path)

if debug:
    trainer.train_model(val_loader, val_loader, num_epochs=num_epochs,
                    callback=partial(callback, model=model, method=model.sample, ckpt_path=ckpt_path))
else:
    trainer.train_model(train_loader, val_loader, num_epochs=num_epochs,
                    callback=partial(callback, model=model, method=model.sample, ckpt_path=ckpt_path))


