# https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py

import os
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms as transform_lib
from torch.utils.data.distributed import DistributedSampler

from wrapper import Wrapper

#default_Imagenet_dir = '/data/ImageNet/ILSVRC2012'
default_Imagenet_dir = '/om/data/public/imagenet/images_complete/ilsvrc/'

class ImagenetDataModule(LightningDataModule):

    name='ImageNet'

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
        self.data_dir = default_Imagenet_dir
        self.meta_dir = None
        self.num_workers = hparams.num_workers
        self.batch_size = hparams.batch_size

    def _get_dataset(self, type_, transforms):
        dir = os.path.join(self.data_dir, type_)
        dataset = datasets.ImageFolder(dir, transforms)
        dataset.name = self.name
        return dataset

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def train_dataloader(self):
        transforms = self.train_transform() 
        dataset = self._get_dataset('train', transforms)

        loader = self._get_DataLoader(
            #Wrapper(dataset),
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
        dataset = self._get_dataset('val', transforms)
        loader = self._get_DataLoader(
            #Wrapper(dataset),
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_transform(self):
        """
        The standard imagenet transforms     
        """
        preprocessing = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.image_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            #imagenet_normalization(), # model does it's own normalization!
        ])

        return preprocessing

    def val_transform(self):
        """
        The standard imagenet transforms for validation
        """
        preprocessing = transform_lib.Compose([
            transform_lib.Resize(self.image_size + 32),
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            #imagenet_normalization(), # model does it's own normalization!
        ])
        return preprocessing
