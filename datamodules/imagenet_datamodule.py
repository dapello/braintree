# https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/datamodules/imagenet_datamodule.py

import os
from warnings import warn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from torchvision import transforms as transform_lib
from .normalization_images import imagenet_normalization
from torch.utils.data.distributed import DistributedSampler

default_Imagenet_dir = '/data/ImageNet/ILSVRC2012'

class ImagenetDataModule(LightningDataModule):

    name = 'imagenet'

    def __init__(
            self,
            data_dir: str = None,
            meta_dir: str = None,
            num_imgs_per_val_class: int = 50,
            image_size: int = 224,
            num_workers: int = 16,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir or default_Imagenet_dir
        self.meta_dir = meta_dir
        self.num_workers = num_workers
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.batch_size = batch_size

    @property
    def num_classes(self):
        return 1000

    def _get_dataset(self, type_, transforms):
        dir = os.path.join(self.data_dir, type_)
        dataset = datasets.ImageFolder(dir, transforms)
        return dataset

    def _get_DataLoader(self, *args, **kwargs):
        return  DataLoader(*args, **kwargs)

    def train_dataloader(self):
        """
        Uses the train split of imagenet2012 and puts away a portion of it for the validation split
        """
        transforms = self.train_transforms or self.train_transform() 
        dataset = self._get_dataset('train', transforms)
        # dataset = UnlabeledImagenet(self.data_dir, num_imgs_per_class=-1, meta_dir=self.meta_dir, split='train', transform=transforms)

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
        Uses the part of the train split of imagenet2012  that was not used for training via `num_imgs_per_val_class`
        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        # transforms = self.val_transforms or self.train_transform()  # huh?: train_transform() instead of val_transform() ??
        transforms = self.val_transforms or self.val_transform()  # huh?: train_transform() instead of val_transform() ??

        dataset = self._get_dataset('val', transforms)
        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        """
        Uses the validation split of imagenet2012 for testing
        """
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms
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
        """      The standard imagenet transforms      """
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
    
    
