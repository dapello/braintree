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

class NeuralDataModule(LightningDataModule):
    name = 'NeuralData'
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
        self.image_size = hparams.image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size
        self.constructor = SOURCES[hparams.neuraldataset](hparams)

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)
    
    def train_dataloader(self):
        """
        Uses the train split from provided neural data path 
        """
        hparams = self.hparams

        X = self.constructor.get_stimuli(stimuli_partition='train').astype('float32')
        Y = self.constructor.get_neural_responses(
            animals=hparams.fit_animals, n_neurons=hparams.neurons,
            n_trials=hparams.trials, neuron_partition=0, stimuli_partition='train', 
            hparams=hparams
        ).astype('float32')

        transforms = self.train_transform() 

        # number of stimuli to fit to
        n_stimuli = int(1e10) if hparams.stimuli=='All' else int(hparams.stimuli)
        dataset = CustomTensorDataset(X[:n_stimuli], Y[:n_stimuli], transforms)

        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        if self.hparams.verbose:
            print(f'neural train set shape: {X.shape}, {Y.shape}')
        return loader

    def val_dataloader(self, stimuli_partition='test', neuron_partition=0):
        """
        Uses the validation split of imagenet2012 for testing
        """
        hparams = self.hparams

        X = self.constructor.get_stimuli(stimuli_partition=stimuli_partition).astype('float32')
        Y = self.constructor.get_neural_responses(
            animals=hparams.test_animals, n_neurons=hparams.neurons,
            n_trials='All', neuron_partition=neuron_partition, stimuli_partition=stimuli_partition,
            hparams=hparams
        ).astype('float32')
        
        transforms = self.val_transform()

        dataset = CustomTensorDataset(X, Y, transforms)
        
        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

        if self.hparams.verbose:
            print(f'neural validation set shape: {X.shape}, {Y.shape}')
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
        Y = self.Y[index]
        return (X,Y)

    def __len__(self):
        return self.X.size(0)


############ Neural Data construction tools ############

## should abstract base class and then build on it for individual constructors
class KKTemporalDataConstructer(object):
    def __init__(
        self, hparams, partition_scheme=(1100, 900, 100, 100), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.data = h5.File('/om2/user/dapello/neural_data/kk_temporal_data.h5', 'r')
        self.partition = Partition(*partition_scheme, seed=hparams.seed)
        self.n_heldout_neurons=50 
        self.verbose = hparams.verbose

    def get_stimuli(self, stimuli_partition):
        # correct flipped axes
        X = self.data['images']['raw'][:].transpose(0,1,3,2)
        # partition the stimuli
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def get_neural_responses(self, animals, n_neurons, n_trials, neuron_partition, stimuli_partition, hparams):
        if self.verbose:
            print(
                f'constructing {stimuli_partition} data with\n' +
                f'animals:{animals}\n' +
                f'neurons:{n_neurons}\n' +
                f'trials:{n_trials}\n'
            )
        # transform "All" to all dataset's animals
        animals = self.expand(animals)
        n_neurons = int(1e10) if n_neurons=='All' else int(n_neurons)
        n_trials = int(1e10) if n_trials=='All' else int(n_trials)
        X = np.concatenate([
            self._get_neural_responses(animal, n_trials, neuron_partition, hparams)
            for animal in animals
        ], axis=1)

        # only return [:n_neurons] if it's not the heldout set of neurons
        if neuron_partition == 0:
            # should be taking a random sample not just first n. can we reuse partition neurons?
            X = X[:, :n_neurons]

        if self.verbose: print(f'Neural data shape:\n(stimuli, sites) : {X.shape}')
        
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def _get_neural_responses(self, animal, n_trials, neuron_partition, hparams):
        animal, region = animal.split('.')
        X = self.data['neural'][animal][region]

        if self.verbose:
            print(
                f'{animal} {region} shape:\n(timestep, stimuli, sites, trials) : {X.shape}'
            )

        # get mean over time window
        start, stop = [int(s) for s in hparams.window.split('t')]
        X = np.nanmean(X[start:stop], axis=0)

        if self.verbose:
            print(f'(stimuli, sites, trials) : {X.shape}')

        """
        get subset of neurons to fit/test on. 
        return_heldout==0 => fitting set,
        return_heldout==1 => heldout set
        """
        X = self.partition_neurons(
            X, X.shape[1]-self.n_heldout_neurons, seed=hparams.seed
        )[neuron_partition]

        if self.verbose:
            print(f'(stimuli, sites, trials) : {X.shape}')

        # take mean over trials
        X = X[:,:,:n_trials]
        X = np.nanmean(X, axis=2)

        if self.verbose:
            print(f'(stimuli, sites) : {X.shape}')

        assert ~np.isnan(np.sum(X))
        return X
    
    @staticmethod
    def expand(animals):
        if animals[0] == 'All':
            animals = ['nano.right', 'nano.left', 'magneto.right']
        return animals

    @staticmethod
    def partition_neurons(X, ntrain, seed=0):
        np.random.seed(seed)
        idx = np.random.choice(X.shape[1], X.shape[1], replace=False)
        return X[:,idx[:ntrain]], X[:,idx[ntrain:]]

class ManyMonkeysDataConstructer(object):
    def __init__(
        self, hparams, partition_scheme=(640, 540, 100, 0), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.data = h5.File('/om2/user/dapello/neural_data/many_monkeys.h5', 'r')
        self.partition = Partition(*partition_scheme, seed=hparams.seed)
        self.n_heldout_neurons=0
        self.verbose = hparams.verbose

    def get_stimuli(self, stimuli_partition):
        X = self.data['stimuli'][:].transpose(0,3,1,2)*255
        # partition the stimuli
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def get_neural_responses(self, animals, n_neurons, n_trials, neuron_partition, stimuli_partition, hparams):
        if self.verbose:
            print(
                f'constructing {stimuli_partition} data with\n' +
                f'animals:{animals}\n' +
                f'neurons:{n_neurons}\n' +
                f'trials:{n_trials}\n'
            )
        # transform "All" to all dataset's animals
        animals = self.expand(animals)
        n_neurons = int(1e10) if n_neurons=='All' else int(n_neurons)
        n_trials = int(1e10) if n_trials=='All' else int(n_trials)
        X = np.concatenate([
            self._get_neural_responses(animal, n_trials, neuron_partition, hparams)
            for animal in animals
        ], axis=1)

        # only return [:n_neurons] if it's not the heldout set of neurons
        if neuron_partition == 0:
            # should be taking a random sample not just first n. can we reuse partition neurons?
            X = X[:, :n_neurons]

        if self.verbose: print(f'Neural data shape:\n(stimuli, sites) : {X.shape}')
        
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def _get_neural_responses(self, animal, n_trials, neuron_partition, hparams):
        animal, region = animal.split('.')
        X = self.data[animal][region]['rates']

        if self.verbose:
            print(
                f'{animal} {region} shape:\n(stimuli, sites, trials) : {X.shape}'
            )

        """
        get subset of neurons to fit/test on. 
        return_heldout==0 => fitting set,
        return_heldout==1 => heldout set
        """
        if self.n_heldout_neurons != 0:
            X = self.partition_neurons(
                X, X.shape[1]-self.n_heldout_neurons, seed=hparams.seed
            )[neuron_partition]

        if self.verbose:
            print(f'(stimuli, sites, trials) : {X.shape}')

        # take mean over trials
        X = X[:,:,:n_trials]
        X = np.nanmean(X, axis=2)

        if self.verbose:
            print(f'(stimuli, sites) : {X.shape}')

        assert ~np.isnan(np.sum(X))
        return X
    
    @staticmethod
    def expand(animals):
        if animals[0] == 'All':
            animals = ['nano.right', 'nano.left', 'magneto.right']
        return animals

    @staticmethod
    def partition_neurons(X, ntrain, seed=0):
        np.random.seed(seed)
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
    def __init__(self, ntotal, ntrain, ntest, nval, seed=0, idx=None):
        super(Partition, self).__init__()
        # always generate the same random partition, for now
        np.random.seed(seed)
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
    'kktemporal' : KKTemporalDataConstructer,
    'manymonkeys' : ManyMonkeysDataConstructer
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
