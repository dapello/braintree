import os, glob
from collections import OrderedDict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from braintree.losses import CenteredKernelAlignment

########### Network Models ##############

import torchvision.models as torchvision_models
import models as custom_models

models_dict = {**torchvision_models.__dict__, **custom_models.__dict__}  # Merge two dictionaries

MODEL_NAMES = sorted(
    name for name in models_dict
    if name.islower() and not name.startswith("__") and callable(models_dict[name])
)

###########

def get_model(arch, pretrained, *args, **kwargs): 
    model_arch = models_dict[arch]
    # remove kwargs for torchvision_models
    kwargs = dict_remove_none(kwargs) if arch in custom_models.__dict__ else {} 
    model = model_arch(pretrained=pretrained, *args, **kwargs)      
    return model

def dict_remove_none(kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

###########

class Model_Lightning(LightningModule):
    
    def __init__(self, hparams, *args, **kwargs): 
        super().__init__()
        
        if isinstance(hparams, dict):  
            # for load_from_checkpoint bug: which uses dict instead of namespace
            hparams = argparse.Namespace(**hparams)
            
        self.hparams = hparams
        self.record_time = hparams.record_time
        
        self.model = get_model(hparams.arch, pretrained=hparams.pretrained, *args, **kwargs)
        self.regions = self.hook_layers()
        self.dissimilarity_loss = CenteredKernelAlignment()

        print('record_time = ', self.record_time)
        
    def forward(self, x):
        return self.model(x)

    def hook_layers(self):
        """
        need to make a more generic layer committer here; this assumes layers in the net are
        named after brain regions like CORnets
        """
        layer_hooks = {
            f'{region}' : Hook(self.model._modules['module']._modules[region])
            for region in self.hparams.regions
        }

        return layer_hooks
    
    def training_step(self, batch, batch_idx):
        losses = []
        if 'ImageNet' in batch.keys():
            losses.append(
                self.classification(batch['ImageNet'], 'train')
            )

        for region in self.hparams.regions:
            if region in batch.keys():
                losses.append(
                    self.dissimilarity(batch[region], region, 'train')
                )
        # is this really working?
        return sum(losses)

    def dissimilarity(self, batch, region, mode):
        X, Y = batch
        _ = self.model(X)
        Y_hat = self.regions[region].output
        loss = self.dissimilarity_loss(Y, Y_hat)
        log = {
            f'{mode}_{self.dissimilarity_loss.name}' : loss
        }
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        losses = []
        if 'ImageNet' in batch.keys():
            losses.append(
                self.classification(batch['ImageNet'], 'val')
            )

        for region in self.hparams.regions:
            if region in batch.keys():
                losses.append(
                    self.dissimilarity(batch[region], region, 'val')
                )

        return sum(losses)

    def classification(self, batch, mode):
        X, Y = batch
        Y_hat = self.model(X)
        loss = F.cross_entropy(Y_hat, Y)
        acc1, acc5 = self.__accuracy(Y_hat, Y, topk=(1,5))
        log = {
            f'{mode}_loss' : loss,
            f'{mode}_acc1' : acc1,
            f'{mode}_acc5' : acc5
        }
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        param_list, lr =  self.parameters(), self.hparams.lr
        lr_list = lr
        
        optimizer = optim.SGD(
            param_list, 
            lr = lr, 
            weight_decay=self.hparams.weight_decay, 
            momentum=0.9, 
            nesterov=True
        )
        scheduler = {
            'scheduler' : lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size
            ),
            'interval' : 'epoch'
        }

        return [optimizer], [scheduler]

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = [correct[:k].sum().item() for k in topk]
            return res

    @staticmethod
    def add_model_specific_args(parent_parser):  
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--v_num', type=int)
        parser.add_argument('-a', '--arch', metavar='ARCH', choices=MODEL_NAMES, default = 'CORnet-S', 
                            help='model architecture: ' + ' | '.join(MODEL_NAMES))
        parser.add_argument('--regions', choices=['V1', 'V2', 'V4', 'IT'], action='append', 
                            help='which CORnet layer to match')
        parser.add_argument('--image_size', default=224, type=int)
        parser.add_argument('--epochs', default=100, type=int, metavar='N')
        parser.add_argument('-b', '--batch-size', type=int, metavar='N', default = 256, 
                            help='this is the total batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--scheduler', type=str, default='ExponentialLR')
        parser.add_argument('--lr', '--learning-rate', metavar='LR', dest='lr', type=float, default = 0.1)
        parser.add_argument('--step_size', default=10, type=int,
                            help='after how many epochs learning rate should be decreased 10x')
        parser.add_argument('--momentum', metavar='M', type=float, default=0.9)
        parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
        # change to step LR
        parser.add_argument('--wd', '--weight-decay', metavar='W', dest='weight_decay', type=float, default = 1e-4)  # set to 1e-2 for cifar10
        parser.add_argument('--optim', dest='optim', default='sgd') # := {'sgd'}
        parser.add_argument('--pretrained', dest='pretrained', action='store_true', default = False)
        parser.add_argument('--record-time', dest='record_time', action='store_true')
        
        return parser

# extract intermediate representations
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output#.clone()

    def close(self):
        self.hook.remove()
