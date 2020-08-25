import torch as ch
import numpy as np
import h5py as h5

import torchvision
from torch.utils.data import Dataset, DataLoader

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        z = self.tensors[2][index]

        return x, y, z

    def __len__(self):
        return self.tensors[0].size(0)
    
class NeuralDataset(Dataset):
    def __init__(self, datapath, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.datapath = datapath
	self.normalize = torchvision.transforms.Normalize(
            mean=mean,
            std=std
        )

        partitions = ['Train', 'Test']
        
        name = 'Stimuli'
        Stimuli_Partitioned = [
            np.load(os.path.join(datapath, f'{name}_{P}.npy'))
            for P in partitions
        ]
        
        name = 'Reps'
        Reps_Partitioned = [
            np.load(os.path.join(datapath, f'{name}_{P}.npy'))
            for P in partitions
        ]
        
        name = 'Labels'
        Labels_Partitioned = [
            np.load(os.path.join(datapath, f'{name}_{P}.npy'))
            for P in partitions
        ]
        
	self.Train, self.Test = [
	    CustomTensorDataset([
	        ch.Tensor(Stimuli_Partitioned[i]), 
	        ch.Tensor(Reps_Partitioned[i]), 
	        ch.Tensor(Y_Partitioned[i])
	    ], transform = torchvision.transforms.Compose([
	        torchvision.transforms.ToPILImage(),
	        torchvision.transforms.Resize(224),
	        torchvision.transforms.ToTensor(),
	        self.normalize
	    ]))
	    for i in range(len(partitions))
	]


# and concat the two:
class ConcatDataset(ch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
