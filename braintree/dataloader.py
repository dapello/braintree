import torch as ch
import numpy as np
import h5py as h5

import torchvision
from torch.utils.data import TensorDataset, DataLoader


# need to add the transformation to normalize images etc. maybe do it in run.py?
class NeuralDataset(ch.utils.data.Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
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
            TensorDataset(
                ch.Tensor(Stimuli_Partitioned[i]), 
                ch.Tensor(Reps_Partitioned[i]), 
                ch.Tensor(Y_Partitioned[i])
            )
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


