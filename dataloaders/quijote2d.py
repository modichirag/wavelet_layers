import numpy as np
from dataclasses import dataclass

# ## PyTorch Data Loading
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

    
@dataclass
class Quijote2d_dataloader():

    d : int = 32
    train_frac : float = 0.8
    train_batch : float = 32
    val_batch : float = 32
    test_batch : float = 32
    train_workers : float = 8
    test_workers : float = 4
    
    def __post_init__(self ):
        
        self.DATASET_PATH : str = f'/mnt/ceph/users/cmodi/ML_data/quijote_fid_{self.d}/'
        
        # Loading the training dataset. 
        data = np.load(f'{self.DATASET_PATH}/data.npy') 
        targets = np.load(f'{self.DATASET_PATH}/targets.npy')
        transform = None 
        train_dataset = CustomDataset(data, targets, transform=transform)

        # We need to split it into a training and validation part
        self.train_size = train_dataset.data.shape[0]
        self.train_set, self.val_set = torch.utils.data.random_split(train_dataset,
                                                           [int(self.train_frac * self.train_size),  self.train_size-int(self.train_frac * self.train_size)],
                                                           generator=torch.Generator().manual_seed(42))

        # Loading the test set
        self.test_set = None

        # Actual data loaders for training, validation, and testing
        self.train_data_loader = DataLoader(self.train_set,
                                            batch_size=self.train_batch,
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=numpy_collate,
                                            num_workers=self.train_workers,
                                            persistent_workers=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch, shuffle=False, drop_last=False, num_workers=self.test_workers, collate_fn=numpy_collate)
        self.test_loader = None

