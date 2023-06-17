import numpy as np
from dataclasses import dataclass

# ## PyTorch Data Loading
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets import MNIST


# Transformations applied on each image => bring them into a numpy array
# Note that we keep them in the range 0-255 (integers)
def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[...,None]  # Make image [28, 28, 1]
    img32 = np.zeros((32, 32, 1))
    img32[2:-2, 2:-2] = img
    return img32

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


@dataclass
class MNIST_dataloader():

    DATASET_PATH : str = '/mnt/ceph/users/cmodi/ML_data/'
    train_frac : float = 0.8
    train_batch : float = 32
    val_batch : float = 32
    test_batch : float = 32
    train_workers : float = 8
    test_workers : float = 4
    
    def __post_init__(self ):
        
        # Loading the training dataset. We need to split it into a training and validation part
        train_dataset = MNIST(root=self.DATASET_PATH, train=True, transform=image_to_numpy, download=True)
        self.train_size = train_dataset.data.size()[0]
        self.train_set, self.val_set = torch.utils.data.random_split(train_dataset,
                                                           [int(self.train_frac * self.train_size),  self.train_size-int(self.train_frac * self.train_size)],
                                                           generator=torch.Generator().manual_seed(42))

        # Loading the test set
        self.test_set = MNIST(root=self.DATASET_PATH, train=False, transform=image_to_numpy, download=True)

        # We define a set of data loaders that we can use for various purposes
        # Data loader for loading examples throughout the notebook
        self.train_exmp_loader = data.DataLoader(self.train_set, batch_size=self.test_batch, shuffle=False, drop_last=False, collate_fn=numpy_collate)
        # Actual data loaders for training, validation, and testing
        self.train_data_loader = data.DataLoader(self.train_set,
                                            batch_size=self.train_batch,
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=numpy_collate,
                                            num_workers=self.train_workers,
                                            persistent_workers=True)
        self.val_loader = data.DataLoader(self.val_set, batch_size=self.val_batch, shuffle=False, drop_last=False, num_workers=self.test_workers, collate_fn=numpy_collate)
        self.test_loader = data.DataLoader(self.test_set, batch_size=self.test_batch, shuffle=False, drop_last=False, num_workers=self.test_workers, collate_fn=numpy_collate)

