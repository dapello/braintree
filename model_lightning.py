import os, glob
from collections import OrderedDict
import argparse

import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from braintree.losses import CenteredKernelAlignment, LogCenteredKernelAlignment

########### Network Models ##############

import torchvision.models as torchvision_models
import models as custom_models

models_dict = {**torchvision_models.__dict__, **custom_models.__dict__}  # Merge two dictionaries

MODEL_NAMES = sorted(
    name for name in models_dict
    if name.islower() and not name.startswith("__") and callable(models_dict[name])
)

###########


class Model_Lightning(LightningModule):
    
    neural_losses = {
        'CKA' : CenteredKernelAlignment,
        'logCKA' : LogCenteredKernelAlignment
    }

    BENCHMARKS=['fneurons.fstimuli', 'fneurons.ustimuli', 'uneurons.fstimuli', 'uneurons.ustimuli']

    def __init__(self, hparams, dm, *args, **kwargs): 
        super().__init__()
        
        if isinstance(hparams, dict):  
            # for load_from_checkpoint bug: which uses dict instead of namespace
            hparams = argparse.Namespace(**hparams)
            
        
        self.dm = dm
        self.hparams = hparams
        self.record_time = hparams.record_time
        self.loss_weights = hparams.loss_weights
        
        self.model = self.get_model(hparams.arch, pretrained=hparams.pretrained, *args, **kwargs)
        self.regions = self.hook_layers()
        self.neural_loss = self.neural_losses[hparams.neural_loss]()
        self.benchmarks = self.load_benchmarks()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

        print('record_time = ', self.record_time)
        
    def forward(self, x):
        return self.model(x)

    def hook_layers(self):
        """
        need to make a more generic layer committer here; this assumes layers in the net are
        named after brain regions like CORnets
        """
        if self.hparams.verbose: print(f'Hooking regions {self.hparams.regions}')
        layer_hooks = {
            f'{region}' : Hook(self.model._modules['module']._modules[region])
            for region in self.hparams.regions
        }

        return layer_hooks

    def train_dataloader(self):
        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        # loaders = {key : self.dm[key].train_dataloader() for key in self.dm}
        loaders = [self.dm[key].train_dataloader() for key in self.dm]

        return loaders

    def val_dataloader(self):
        # loaders = {key : self.dm[key].val_dataloader() for key in self.dm}
        # inspect hparams for validation settings;
        # validation options: 
        # [fitted neurons, heldout stimuli] # dm.val_loader(stimuli_partion='test', neuron_partition=0) 
        # [heldout neurons, fitted stimuli] # dm.val_loader(stimuli_partion='train', neuron_partition=1) 
        # [heldout neurons, heldout stimuli] # dm.val_loader(stimuli_partion='test', neuron_partition=1) 

        loaders = [self.dm[key].val_dataloader() for key in self.dm if "ImageNet" in key]

        return loaders

    def training_step(self, batch, batch_idx):
        losses = []
        for dataloader_idx, batch_ in enumerate(batch):
            if dataloader_idx == 0:
                losses.append(
                    self.loss_weights[dataloader_idx]*self.classification(
                        batch_, 'train'
                    )
                )

            elif dataloader_idx == 1:
                losses.append(
                    self.loss_weights[dataloader_idx]*self.similarity(
                        batch_, 'IT', 'train'
                    )
                )

        return sum(losses)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None, mode='val'):
        ## need a proper map here for the dataloader_idx
        losses = []
        if dataloader_idx is None:
            dataloader_idx = 0

        if dataloader_idx == 0:
            losses.append(
                self.classification(batch, mode)
            )

        if dataloader_idx == 1:
            losses.append(
                self.similarity(batch, 'IT', mode)
            )

        return sum(losses)

    def validation_epoch_end(self, outputs):
        # what if we do the real neural validation work here?
        # validation options: 
        # [fitted neurons, heldout stimuli] # dm.val_loader(stimuli_partion='test', neuron_partition=0) 
        # [heldout neurons, fitted stimuli] # dm.val_loader(stimuli_partion='train', neuron_partition=1) 
        # [heldout neurons, heldout stimuli] # dm.val_loader(stimuli_partion='test', neuron_partition=1) 
        if 'NeuralData' in self.dm.keys():
            with ch.no_grad():
                self.model.eval()
                for key in self.benchmarks:
                    X, Y = [], []
                    for X_, Y_ in self.benchmarks[key]:
                        X.append(X_)
                        Y.append(Y_)
                    X = ch.cat(X).cuda()
                    Y = ch.cat(Y).cuda()
                    similarity_loss = self.similarity((X,Y), 'IT', key)

                    del X, Y, similarity_loss
                    ch.cuda.empty_cache()

    def load_benchmarks(self):
        benchmarks = {}
        if 'NeuralData' in self.dm.keys():
            if self.hparams.benchmarks[0] == 'All':
                self.hparams.benchmarks = self.BENCHMARKS

            if 'fneurons.fstimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on fitted neurons and fitted stimuli')

                benchmarks['fneurons.fstimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='train', neuron_partition=0
                )
                
            if 'fneurons.ustimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on fitted neurons and unfitted stimuli')

                benchmarks['fneurons.ustimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='test', neuron_partition=0
                )

            if 'uneurons.fstimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on unfitted neurons and fitted stimuli')
                
                benchmarks['uneurons.fstimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='train', neuron_partition=1
                )

            if 'uneurons.ustimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on unfitted neurons and unfitted stimuli')
                
                benchmarks['uneurons.ustimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='test', neuron_partition=1
                )

        return benchmarks


    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, mode='val')

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
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        #if mode == 'train':
        #    self.train_acc(Y_hat, Y)
        #    self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        #elif mode == 'val':
        #    self.valid_acc(Y_hat, Y)
        #    self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

        return loss

    def similarity(self, batch, region, mode):
        X, Y = batch
        _ = self.model(X)
        Y_hat = self.regions[region].output
        loss = self.neural_loss(Y, Y_hat)
        log = {
            f'{self.neural_loss.name}_{mode}' : loss
        }
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        t = ch.cuda.get_device_properties(0).total_memory
        r = ch.cuda.memory_reserved(0) 
        a = ch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f'>>>sim: total {t}, reserved {r}, allocated {a}, free {f}')

        return loss

    def configure_optimizers(self):
        param_list, lr = self.parameters(), self.hparams.lr
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
    def __accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with ch.no_grad():
            _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            total = output.shape[0]
            res = [correct[:k].sum().item()/total for k in topk]
            return res

    @staticmethod
    def get_model(arch, pretrained, *args, **kwargs): 
        
        def dict_remove_none(kwargs):
            return {k: v for k, v in kwargs.items() if v is not None}

        model_arch = models_dict[arch]
        # remove kwargs for torchvision_models
        kwargs = dict_remove_none(kwargs) if arch in custom_models.__dict__ else {} 
        print(f'Using pretrained model: {pretrained}')
        model = model_arch(pretrained=pretrained, *args, **kwargs)      
        return model

    @classmethod
    def add_model_specific_args(cls, parent_parser):  
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--v_num', type=int)
        parser.add_argument('-a', '--arch', metavar='ARCH', choices=MODEL_NAMES, default = 'cornet_s', 
                            help='model architecture: ' + ' | '.join(MODEL_NAMES))
        parser.add_argument('--regions', choices=['V1', 'V2', 'V4', 'IT'], nargs="*", default=['IT'], 
                            help='which CORnet layer to match')
        parser.add_argument('--neural_loss', default='CKA', choices=cls.neural_losses.keys(), type=str)
        parser.add_argument('--loss_weights', nargs="*", default=[1,1], type=float,
                            help="how to weight losses; [1,1] => equal weighting of imagenet and neural loss")
        parser.add_argument('--image_size', default=224, type=int)
        parser.add_argument('--epochs', default=25, type=int, metavar='N')
        parser.add_argument('-b', '--batch-size', type=int, metavar='N', default = 128, 
                            help='this is the total batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--scheduler', type=str, default='StepLR')
        parser.add_argument('--lr', '--learning-rate', metavar='LR', dest='lr', type=float, default = 0.001)
        parser.add_argument('--step_size', default=10000, type=int,
                            help='after how many epochs learning rate should be decreased 10x')
        parser.add_argument('--momentum', metavar='M', type=float, default=0.9)
        parser.add_argument('--wd', '--weight-decay', metavar='W', dest='weight_decay', type=float, 
                            default = 1e-4)  # set to 1e-2 for cifar10
        parser.add_argument('--optim', dest='optim', default='sgd') # := {'sgd'}
        parser.add_argument('--pretrained', dest='pretrained', action='store_true', default = True)
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
