import os

import numpy as np
import torch as ch
import h5py as h5

import torchvision
from torchvision import transforms as transform_lib
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .imagenet_datamodule import ImagenetDataModule
from .normalization import imagenet_normalization
from .wrapper import Wrapper

class ImageNetAndNeuralDataModule(LightningDataModule):
    """
    merges both ImageNet loader and the NeuralData loader below through
    Wrapper. Wrapper returns batches inside a dictionary, with dataset name 
    as the key.
    """
    def __init__(
        self, 
        hparams,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.name = hparams.datamodule
        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size
        self.ImageNet = ImagenetDataModule(hparams)
        self.NeuralData = NeuralDataModule(hparams)

    def train_dataloader(self):
        dataset = Wrapper(
            self.ImageNet._get_dataset('train', self.ImageNet.train_transform()),
            self.NeuralData._get_dataset('train', self.NeuralData.train_transform())
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader

    def val_dataloader(self):
        dataset = Wrapper(
            self.ImageNet._get_dataset('val', self.ImageNet.val_transform()),
            self.NeuralData._get_dataset('val', self.NeuralData.val_transform())
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader
        
class NeuralDataModule(LightningDataModule):
    """
    A DataLoader for neural data. uses a dataconstructer class (KKTemporal) to
    format neural data, formats it and then returns it wrapped in a dictionary 
    through Wrapper
    """
    def __init__(
        self, 
        hparams,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.name = hparams.datamodule
        self.image_size = hparams.image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size
        self.constructor = SOURCES[hparams.neuraldataset](hparams)

    def _get_dataset(self, type_, transforms):
        # construct data here
        X = self.constructor.get_stimuli()[type_]
        Y = self.constructor.get_neural_responses()[type_]
        dataset = CustomTensorDataset(X, Y, transforms)
        dataset.name = self.name
        import pdb; pdb.set_trace()
        return Wrapper(dataset)

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)
    
    def train_dataloader(self):
        """
        Uses the train split from provided neural data path 
        """
        transforms = self.train_transform() 
        dataset = self._get_dataset('train', transforms)

        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """
        Uses the validation split of imagenet2012 for testing
        """
        transforms = self.val_transform()
        dataset = self._get_dataset('test', transforms)
        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_transform(self):
        preprocessing = transform_lib.Compose([
            transform_lib.ToPILImage(),
            transform_lib.Resize(self.image_size),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

    def val_transform(self):
        preprocessing = transform_lib.Compose([
            transform_lib.ToPILImage(),
            transform_lib.Resize(self.image_size),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    takes    
        Stimuli : nd.array,
        Target : nd.array
    and returns [transform(stimuli), target]

    __getitem__ operates with any index through modulo
    """
    def __init__(self, X, Y, transform=None):
        assert X.shape[0] == Y.shape[0] 
        self.X = ch.Tensor(X)
        self.Y = ch.Tensor(Y)
        self.transform = transform

    def __getitem__(self, index):
        # modulo index by length of data, so that we can any index
        N = self.__len__()
        index = index%N
        
        X = self.transform(self.X[index])
        Y = self.Y
        return tuple(X,Y)

    def __len__(self):
        return self.X.size(0)


############ Neural Data construction tools ############

class KKTemporalDataConstructer(object):
    def __init__(
        self, hparams, partition_scheme=(1100, 900, 100, 100), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.data = h5.File('/om2/user/dapello/neural_data/kk_temporal_data.h5', 'r')
        self.partition = Partition(*partition_scheme)
        self.regions = hparams.regions
        self.animals = self.expand(hparams.animals)
        self.n_fit_images = int(1e10) if hparams.stimuli=='All' else hparams.stimuli
        self.n_fit_neurons = int(1e10) if hparams.neurons=='All' else hparams.neurons
        self.n_trials = int(1e10) if hparams.trials=='All' else hparams.trials
        self.n_heldout_neurons=50 
        self.window = int(1e10) if hparams.window=='All' else hparams.window
        self.return_heldout=0
        self.verbose = hparams.verbose

    def get_stimuli(self):
        # correct flipped axes
        X = self.data['images']['raw'][:].transpose(0,1,3,2)
        # partition the stimuli
        X_Partitioned = self.partition(X)
        X_Partitioned['train'] = X_Partitioned['train'][:self.n_fit_images]
        return X_Partitioned

    def get_neural_responses(self):
        X = np.concatenate([
            self._get_neural_responses(
                animal
            )
            for animal in self.animals
        ], axis=1)

        if self.verbose:
            print(
                f'{self.animals} shape:\n\
                (stimuli, sites) : {X.shape}'
            )
        
        X_Partitioned = self.partition(X)
        X_Partitioned['train'] = X_Partitioned['train'][:self.n_fit_images]

        return X_Partitioned

    def _get_neural_responses(self, animal):
        animal, region = animal.split('.')
        X = self.data['neural'][animal][region]

        if self.verbose:
            print(
                f'{animal} {region} shape:\n\
                (timestep, stimuli, sites, trials) : {X.shape}'
            )

        # get mean over time window
        start, stop = [int(s) for s in self.window.split('t')]
        X = np.nanmean(X[start:stop], axis=0)

        if self.verbose:
            print(
                f'{animal} {region} shape:\n\
                (stimuli, sites, trials) : {X.shape}'
            )

        """
        get subset of neurons to fit/test on. 
        return_heldout==0 => fitting set,
        return_heldout==1 => heldout set
        """
        X = self.partition_neurons(X, X.shape[1]-self.n_heldout_neurons)[self.return_heldout]
        X = X[:, :self.n_fit_neurons, :]

        if self.verbose:
            print(
                f'{animal} {region} shape:\n\
                (stimuli, sites, trials) : {X.shape}'
            )

        # take mean over trials
        if self.n_trials!='All':
            X = X[:,:,:self.n_trials]
        X = np.nanmean(X, axis=2)

        if self.verbose:
            print(
                f'{animal} {region} shape:\n\
                (stimuli, sites) : {X.shape}'
            )

        return X
    
    @staticmethod
    def expand(animals):
        if animals[0] == 'All':
            animals = ['nano.right', 'nano.left', 'magneto.right']
        return animals

    @staticmethod
    def partition_neurons(X, ntrain):
        np.random.seed(0)
        idx = np.random.choice(X.shape[1], X.shape[1], replace=False)
        return X[:,idx[:ntrain]], X[:,idx[ntrain:]]

class Partition(object):
    """ 
    generate random indices dividing data into train, test, and val sets.
    saves the indices, so you can easily use the same partition scheme on 
    multiple datasets with the same original order. ie:
        
        partion(images)
        partion(response)
    
        is equivalent to
    
        idx = random_index
        images[idx]
        responses[idx]
    
    """
    def __init__(self, ntotal, ntrain, ntest, nval, idx=None):
        super(Partition, self).__init__()
        # always generate the same random partition, for now
        np.random.seed(0)
        self.ntotal = ntotal
        self.ntrain = ntrain
        self.ntest = ntest
        self.nval = nval
        
        # so we can supply the idx if we want to use the same partition scheme
        if idx:
            self.idx = idx
        else:
            self.idx = np.random.choice(ntotal, ntotal, replace=False)
            
        self.train_idx = self.idx[:ntrain]
        
        test_and_val_idx = self.idx[ntrain:]
        self.test_idx = test_and_val_idx[:ntest]
        self.val_idx = test_and_val_idx[ntest:]
    
        # make sure none of the indices are overlapping
        assert 0 == len(
            set(self.train_idx)
            .intersection(set(self.test_idx))
            .intersection(set(self.val_idx))
        )

    def __call__(self, X):
        return {
            'train' : X[self.train_idx], 
            'test' : X[self.test_idx], 
            'val' : X[self.val_idx]
        }


SOURCES = {
    'kktemporal' : KKTemporalDataConstructer
}

"""
# datasets to scan
animals = ['magneto.right,nano.right,nano.left','magneto_right', 'nano_right', 'nano_left']
trials = [2,4,8,16,32,'All']
neurons = [42, 84, 168, 336]
stimuli = [225, 450, 900]

datasets = [
    f'ds_kktemporal-animal_{animal}-trials_{trials}-neurons_{neurons}-stimuli_{stimuli}' 
    for animal in animals
    for trial in trials
    for neuron in neurons
    for stimulus in stimuli
]

len(datasets) => 288
"""
