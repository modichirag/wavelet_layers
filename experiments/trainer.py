import numpy as np
import sys, os, json

import jax
from torch.utils.tensorboard import SummaryWriter

from flax.training import checkpoints
import matplotlib.pyplot as plt

class TrainerModule:

    def __init__(self, model_name, state, model, ckpt_path, nckpt=5, seed=99, ckpt_to_keep=2, ):
        super().__init__()
        self.model_name = model_name
        self.rng = jax.random.PRNGKey(seed)
        self.nckpt = nckpt
        self.state = state
        # Create empty model. Note: no parameters yet
        self.model = model
        # Prepare logging
        self.log_dir = os.path.join(ckpt_path, 'logs')
        print(self.log_dir)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        self.train_loss = []
        self.epoch_loss = []
        #checkpoints
        self.ckpt_path = ckpt_path

        
    def create_functions(self):
        # Training function
        def train_step(state, rng, batch):
            imgs, _ = batch
            loss_fn = lambda params: self.model.apply({'params': params}, imgs, rng, testing=False)
            (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  # Get loss and gradients for loss
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, rng, loss
        self.train_step = jax.jit(train_step)

        # Eval function, which is separately jitted for validation and testing        
        def eval_step(state, rng, batch, testing):
            return self.model.apply({'params': state.params}, batch[0], rng, testing=testing)
        self.eval_step = jax.jit(eval_step, static_argnums=(3,))

        
    def train_model(self, train_loader, val_loader, num_epochs=500, callback=None):
        # Train model for defined number of epochs
        best_eval = 1e6

        for epoch_idx in range(1, num_epochs+1):
            print(f"Epoch : {epoch_idx}")
            self.train_epoch(train_loader, epoch=epoch_idx)
            if callback is not None:
                print("executing callback")
                self.rng = callback(params=self.state.params, rng=self.rng, step=epoch_idx)
            if epoch_idx % self.nckpt == 0:
                eval_bpd = self.eval_model(val_loader, testing=False)
                self.logger.add_scalar('val/bpd', eval_bpd, global_step=epoch_idx)
                print("evaluation bpd : ", eval_bpd)
                if eval_bpd < best_eval:
                    best_eval = eval_bpd
                    self.save_model(step=epoch_idx)
                self.logger.flush()

                
    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_loss)
        plt.loglog()
        plt.grid()
        plt.savefig(self.ckpt_path + '/figs/loss')
        plt.close()
    
    def train_epoch(self, data_loader, epoch):
        # Train model for one epoch, and log avg loss
        avg_loss = 0.
        ib = 0
        for batch in data_loader:
            ib +=1 
            self.state, self.rng, loss = self.train_step(self.state, self.rng, batch)
            self.train_loss.append(loss)
            avg_loss += loss
        avg_loss /= len(data_loader)
        self.epoch_loss.append(avg_loss)
        print(f"Loss at epoch : {avg_loss.item()}")
        #make loss figure here
        self.plot_loss()
        self.logger.add_scalar('train/bpd', avg_loss.item(), global_step=epoch)

        
    def eval_model(self, data_loader, testing=False):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss, self.rng = self.eval_step(self.state, self.rng, batch, testing=testing)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss
    
    
    def save_model(self, step):
        print(f"Saving model for step : {step}")
        ckpt = {}
        ckpt['params'] = self.state.params
        ckpt['tx'] = self.state.tx
        ckpt['name'] = self.model_name
        checkpoints.save_checkpoint(ckpt_dir=self.ckpt_path,
                            target=ckpt,
                            step=step,
                            overwrite=True,
                            keep=2)

        
    def load_model(self, step):
        ckpt = {}
        ckpt['params'] = self.state.params
        ckpt['params'] = self.state.tx
        ckpt['name'] = self.model_name
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_path, target=ckpt)
        self.state = train_state.TrainState.create(apply_fn=self.state.apply_fn, 
                                                   params=ckpt['params'], 
                                                   tx=ckpt['tx'])
        return ckpt



