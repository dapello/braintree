import os, glob, time
from collections import OrderedDict
import argparse

import psutil
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

from braintree.losses import CenteredKernelAlignment, LogCenteredKernelAlignment
from braintree.benchmarks import score_model
from braintree.adversary import Adversary
from models.helpers import layer_maps, add_normalization, add_outputs, Hook

##### models
import torchvision.models as torchvision_models
import models as custom_models 

process = psutil.Process()
models_dict = {**torchvision_models.__dict__, **custom_models.__dict__}  # Merge two dictionaries

MODEL_NAMES = sorted(
    name for name in models_dict
    if name.islower() and not name.startswith("__") and callable(models_dict[name])
)

#####

class Model_Lightning(LightningModule):
    
    NEURAL_LOSSES = {
        'CKA' : CenteredKernelAlignment,
        'logCKA' : LogCenteredKernelAlignment
    }

    # f = fitted, u = unfitted. ie fnuerons.ustimuli => run benchmark on fitted neurons and unfitted stimuli
    # this should not be called BENCHMARKS, to be consistent with brainscore terminology. PARTITION maybe?
    BENCHMARKS=['fneurons.fstimuli', 'fneurons.ustimuli', 'uneurons.fstimuli', 'uneurons.ustimuli']
    LAYER_MAPS=layer_maps

    def __init__(self, hparams, dm, *args, **kwargs): 
        super().__init__()
        
        self.dm = dm
        #self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.record_time = hparams.record_time
        self.loss_weights = hparams.loss_weights
        
        assert self.hparams.arch in self.LAYER_MAPS
        self.layer_map = self.LAYER_MAPS[hparams.arch]
        self.model = self.get_model(hparams.arch, pretrained=hparams.pretrained, *args, **kwargs)
        self.regions = self.hook_layers()
        self.neural_loss = self.NEURAL_LOSSES[hparams.neural_loss]()
        self.neural_val_loss = self.NEURAL_LOSSES[hparams.neural_val_loss]()
        self.benchmarks = self.load_benchmarks()
        self.adversaries = self.generate_adversaries()

        print('record_time = ', self.record_time)
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)

    def hook_layers(self):
        if self.hparams.verbose: print(f'Hooking regions {self.hparams.regions}')

        layer_hooks = {}

        for region in self.hparams.regions:
            # this allows us to specify layer4.downsample0.maxpool for instance to get the maxpool in layer4.downsample0
            # [1] gets model instead of normalization layer [0]
            model = self.model[1]
            layer = model.module if hasattr(model, 'module') else model
            # iteratively find layer to hook
            for id_ in self.layer_map[region].split('.'):
                layer = getattr(layer, id_)

            if f'{region}_temp' in self.layer_map.keys():
                layer_hooks[region] = Hook(layer, **self.layer_map[f'{region}_temp'])
            else:
                layer_hooks[region] = Hook(layer)

        return layer_hooks

    def generate_adversaries(self):
        adversaries = {}
        if self.hparams.adv_eval_images:
            ## make class adversary
            adversaries['class_adversary'] = Adversary(
                model=self.model,
                eps=self.hparams.eps
            )

        if self.hparams.adv_eval_neural:
            ## make region adversaries
            print('Neural Adversaries not implemented')
            raise
        return adversaries

    def train_dataloader(self):
        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        # loaders = {key : self.dm[key].train_dataloader() for key in self.dm}
        loaders = [self.dm[key].train_dataloader() for key in self.dm]

        return loaders

    def val_dataloader(self):
        # loaders = {key : self.dm[key].val_dataloader() for key in self.dm}

        loaders = [self.dm[key].val_dataloader() for key in self.dm if "ImageNet" in key]

        return loaders

   # def train_dataloader(self):
   #     # pass loaders as a dict. This will create batches like this:
   #     # {'a': batch from loader_a, 'b': batch from loader_b}
   #     loaders = {key : self.dm[key].train_dataloader() for key in self.dm}
   #     # loaders = [self.dm[key].train_dataloader() for key in self.dm]

   #     return loaders

   # def val_dataloader(self):
   #     loaders = {key : self.dm[key].val_dataloader() for key in self.dm}

   #     # loaders = [self.dm[key].val_dataloader() for key in self.dm if "ImageNet" in key]

   #     return loaders

    def training_step(self, batch, batch_idx):
        losses = []
        #import pdb; pdb.set_trace()

        for dataloader_idx, batch_ in enumerate(batch):
            if dataloader_idx == 0:
                losses.append(
                    self.loss_weights[dataloader_idx]*self.classification(
                        batch_, 'train'
                    )
                )

            # this assumes dataloader_idx is the dataloader for IT. 
            # fine for now, but need to generalize if we wanted to fit multiple layers.
            elif (dataloader_idx == 1) & (self.loss_weights[dataloader_idx] != 0):
                if not self.hparams.adapt_bn_to_stim: self.model.eval()
                losses.append(
                    self.loss_weights[dataloader_idx]*self.similarity(
                        batch_, 'IT', 'train'
                    )
                )
                if not self.hparams.adapt_bn_to_stim: self.model.train()

            elif (dataloader_idx == 2) & (self.loss_weights[dataloader_idx] != 0):
                if not self.hparams.adapt_bn_to_stim: self.model.eval()
                losses.append(
                    self.loss_weights[dataloader_idx]*self.classification(
                        batch_, 'train', output_inds=[1000, 1008], dataset='Stimuli'
                    )
                )
                if not self.hparams.adapt_bn_to_stim: self.model.train()

        #loss = self.loss_weights[0]*self.classification(batch[0], 'train')
        #loss += self.loss_weights[1]*self.similarity(batch[1], 'IT', 'train')
        #return loss

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
            if self.hparams.adv_eval_images:
                losses.append(
                    self.classification(batch, f'adv_{mode}', adversarial=True)
                )


        # this assumes dataloader_idx is the dataloader for IT. 
        # fine for now, but need to generalize if we wantedto fit multiple layers.
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
                    X = ch.cat(X)#.cuda()
                    Y = ch.cat(Y)#.cuda()
                    similarity_loss = self.similarity((X,Y), 'IT', key)

                    # we were having mem issues for a while, maybe they've been resolved?
                    #del X, Y, similarity_loss
                    #ch.cuda.empty_cache()

        if self.hparams.BS_benchmarks[0] != 'None':
            self.model.eval()
            benchmark_log = {}
            for benchmark_identifier in self.hparams.BS_benchmarks:
                model_id = f'{self.hparams.file_name}-v_{self.hparams.v_num}-{int(time.time())}'
                print('>>>', model_id)
                layer = '1.module.' if hasattr(self.model[1], 'module') else '1.'
                if 'V1' in benchmark_identifier:
                    layers = [layer + self.layer_map['V1']]
                elif 'V2' in benchmark_identifier:
                    layers = [layer + self.layer_map['V2']]
                elif 'V4' in benchmark_identifier:
                    layers = [layer + self.layer_map['V4']]
                elif 'IT' in benchmark_identifier:
                    layers = [layer + self.layer_map['IT']]
                else:
                    layers = [layer + self.layer_map['decoder']]
                score = score_model(
                    model_identifier=model_id,
                    model=self.model,
                    layers=layers,
                    benchmark_identifier=benchmark_identifier,
                )

                benchmark_log[benchmark_identifier] = score.values[0]
                if self.hparams.verbose: print(f'layers: {layers}, {benchmark_log}')

            self.log_dict(benchmark_log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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

    def classification(self, batch, mode, output_inds=[0,1000], dataset='ImageNet', adversarial=False):
        X, Y = batch
        Y = Y.long()
        if adversarial:
            X = self.adversaries['class_adversary'].generate(X, Y, F.cross_entropy, output_inds=output_inds)

        Y_hat = self.model(X)[:, output_inds[0]:output_inds[1]]

        loss = F.cross_entropy(Y_hat, Y)
        acc1, acc5 = self.__accuracy(Y_hat, Y, topk=(1,5))
        if mode == 'train':
            pass

        log = {
            f'{dataset}_{mode}_loss' : loss,
            f'{dataset}_{mode}_acc1' : acc1,
            f'{dataset}_{mode}_acc5' : acc5
        }
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def similarity(self, batch, region, mode):
        X, Y = batch
        if 'adv_' in mode:
            # adversarially attack on labels. requires HVM readouts to be trained.
            X = self.adversaries['class_adversary'].generate(X, Y, F.cross_entropy, output_inds=[1000,1008])

        _ = self.model(X)
        Y_hat = self.regions[region].output

        # this allows to test with a different loss than the train loss.
        neural_loss_fnc = self.neural_loss if mode == 'train' else self.neural_val_loss
        loss = neural_loss_fnc(Y, Y_hat)
        log = {f'{neural_loss_fnc.name}_{mode}' : loss}

        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # for debugging memory issues
        #t = ch.cuda.get_device_properties(0).total_memory
        #r = ch.cuda.memory_reserved(0) 
        #a = ch.cuda.memory_allocated(0)
        #f = r-a  # free inside reserved
        #print(f'>>>sim: total {t}, reserved {r}, allocated {a}, free {f}')

        return loss

    def configure_optimizers(self):
        param_list, lr = self.parameters(), self.hparams.lr
        
        ## ???
        #lr_list = lr
        
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

    def get_model(self, arch, pretrained, *args, **kwargs): 
        """gets a model and prepends a normalization layer"""
        def dict_remove_none(kwargs):
            return {k: v for k, v in kwargs.items() if v is not None}

        model_arch = models_dict[arch]
        # remove kwargs for torchvision_models
        kwargs = dict_remove_none(kwargs) if arch in custom_models.__dict__ else {} 
        print(f'Using pretrained model: {pretrained}')
        model = model_arch(pretrained=pretrained, *args, **kwargs)
        model = add_normalization(model)
        model = add_outputs(model, out_name=self.layer_map['output'], n_outputs=8)
        return model

    @classmethod
    def add_model_specific_args(cls, parent_parser):  
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--v_num', type=int)
        parser.add_argument('-a', '--arch', metavar='ARCH', choices=MODEL_NAMES, default = 'cornet_s', 
                            help='model architecture: ' + ' | '.join(MODEL_NAMES))
        parser.add_argument('--regions', choices=['V1', 'V2', 'V4', 'IT'], nargs="*", default=['IT'], 
                            help='which CORnet layer to match')
        parser.add_argument('--neural_loss', default='logCKA', choices=cls.NEURAL_LOSSES.keys(), type=str)
        parser.add_argument('--neural_val_loss', default='CKA', choices=cls.NEURAL_LOSSES.keys(), type=str)
        parser.add_argument('--loss_weights', nargs="*", default=[1,1,0], type=float,
                            help="how to weight losses; [1,1,1] => equal weighting of imagenet, neural loss, and stimuli classification")
        parser.add_argument('--image_size', default=224, type=int)
        parser.add_argument('--epochs', default=150, type=int, metavar='N')
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
        parser.add_argument('--pretrained', dest='pretrained', type=int, default=1)
        parser.add_argument('-adapt', '--adapt_bn_to_stim', dest='adapt_bn_to_stim', type=int, default=0)
        parser.add_argument('--record-time', dest='record_time', action='store_true')
        
        return parser
