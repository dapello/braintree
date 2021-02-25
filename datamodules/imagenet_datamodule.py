# https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py

import os
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from torchvision import transforms as transform_lib
from .normalization import imagenet_normalization
from torch.utils.data.distributed import DistributedSampler
from .wrapper import Wrapper

default_Imagenet_dir = '/data/ImageNet/ILSVRC2012'

class ImagenetDataModule(LightningDataModule):
    def __init__(
        self,
        name: str = 'ImageNet',
        data_dir: str = None,
        meta_dir: str = None,
        image_size: int = 224,
        num_workers: int = 16,
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir or default_Imagenet_dir
        self.meta_dir = meta_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def _get_dataset(self, type_, transforms):
        dir = os.path.join(self.data_dir, type_)
        dataset = datasets.ImageFolder(dir, transforms)
        dataset.name = self.name
        return Wrapper(dataset)

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def train_dataloader(self):
        transforms = self.train_transforms or self.train_transform() 
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
        """
        The standard imagenet transforms     
        """
        preprocessing = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.image_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

    def val_transform(self):
        """       The standard imagenet transforms for validation      """
        preprocessing = transform_lib.Compose([
            transform_lib.Resize(self.image_size + 32),
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])
        return preprocessing
