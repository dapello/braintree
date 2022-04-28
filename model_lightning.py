import os, glob, time, gc
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

#from braintree.losses import CenteredKernelAlignment, LogCenteredKernelAlignment, LogCenteredKernelAlignment0
from braintree.losses import NEURAL_LOSSES
from braintree.benchmarks import score_model, score_model_behavior
from braintree.adversary import Adversary
from datamodules.neural_datamodule import NeuralDataModule
from models.helpers import layer_maps, add_normalization, add_outputs, Hook, copy_bns, paste_bns

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
    
    NEURAL_LOSSES = NEURAL_LOSSES

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

        # initialize bn modes
        if hparams.multi_bn:
            self.bn_imnet = copy_bns(self.model)
            self.bn_hvm = copy_bns(self.model)
            self.set_bn('ImageNet')

        print('record_time = ', self.record_time)
        self.save_hyperparameters()
    
    def set_bn(self, mode):
        if self.hparams.multi_bn:
            if 'ImageNet' in mode:
                bns = self.bn_imnet
                print('swapped to ImageNet bn')

            if 'Stimuli' in mode:
                bns = self.bn_hvm
                print('swapped to Stimuli bn')

            if self.model.training:
                for name, module in bns.items(): module.train()
            else:
                for name, module in bns.items(): module.eval()
            
            self.model = paste_bns(self.model, bns)

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
        if self.hparams.adv_train_images:
            ## make class adversary
            adversaries['train_class_adversary'] = Adversary(
                model=self.model,
                eps=self.hparams.train_eps
            )

        if self.hparams.adv_eval_images:
            ## make class adversary
            adversaries['adv_val_class_adversary'] = Adversary(
                model=self.model,
                eps=self.hparams.eps
            )

        if self.hparams.adv_eval_neural:
            ## make region adversaries
            adversaries['adv_val_neural_adversary'] = Adversary(
                model=self.model,
                eps=self.hparams.eps
            )
        return adversaries

    def train_dataloader(self):
        # pass loaders as a dict. This will create batches like this:
        # {'a': batch from loader_a, 'b': batch from loader_b}
        loaders = {key : self.dm[key].train_dataloader() for key in self.dm}
        #loaders = [self.dm[key].train_dataloader() for key in self.dm]

        return loaders

    def val_dataloader(self):
        # we just run ImageNet val through the normal val_dataloader -- the neural validation is handled in validation_epoch_end

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

        return sum(losses)

    def validation_epoch_end(self, outputs):
        # we do the real neural validation work here
        if 'NeuralData' in self.dm.keys():
            ch.cuda.empty_cache()
            with ch.no_grad():
                self.set_bn(mode='Stimuli')
                self.model.eval()
                # loop over benchmarks (here, dataloaders)
                for key in self.benchmarks:
                    # draw the data from the data loader (large batch_size => 1 batch for validation)
                    if 'coco' in key:
                        self.set_bn(mode='ImageNet')
                    else:
                        self.set_bn(mode='Stimuli')

                    for batch in self.benchmarks[key]:
                        pass

                    # score the model on the data. similarity function will take care of logging here.
                    self.similarity(batch, 'IT', key)
 
                    # and similarity on adversarially attacked stimuli
                    if self.hparams.adv_eval_neural:
                        self.similarity(batch, 'IT', f'adv_{key}', adversarial=True)

                    # and classification of HVM stimuli, if fitting HVM labels
                    if self.hparams.loss_weights[2] > 0:
                        self.classification(
                            batch, 'val', output_inds=[1000,1008], 
                            dataset=f'Stimuli_{key}', adversarial=False
                        )

                        # and also adversarial classification of HVM stimuli
                        if self.hparams.adv_eval_images:
                            self.classification(
                                batch, 'adv_val', output_inds=[1000,1008], 
                                dataset=f'Stimuli_{key}', adversarial=True
                            )

                    # we were having mem issues for a while, maybe they've been resolved?
                    del batch
                    ch.cuda.empty_cache()
                    gc.collect()

        if self.hparams.BS_benchmarks[0] != 'None':
            ## set bn for each?
            self.set_bn(mode='ImageNet')
            self.model.eval()
            benchmark_log = {}
            for benchmark_identifier in self.hparams.BS_benchmarks:
                model_id = f'{self.hparams.file_name}-v_{self.hparams.v_num}-{int(time.time())}'
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


                if 'Rajalingham' in benchmark_identifier:
                    self.set_bn(mode='Stimuli')
                else:
                    self.set_bn(mode='ImageNet')

                score = score_model(
                    model_identifier=model_id,
                    model=self.model,
                    layers=layers,
                    benchmark_identifier=benchmark_identifier,
                )

                benchmark_log[benchmark_identifier] = score.values[0]
                if self.hparams.verbose: print(f'layers: {layers}, {benchmark_log}')

            self.log_dict(benchmark_log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if (self.hparams.behavior_benchmarks[0] != 'None') & ~(self.hparams.test):
            self.set_bn(mode='Stimuli')
            self.model.eval()
            benchmark_log = {}
            for benchmark_identifier in self.hparams.behavior_benchmarks:
                print(benchmark_identifier)
                model_id = f'{self.hparams.file_name}-v_{self.hparams.v_num}-{int(time.time())}'
                layer = '1.module.' if hasattr(self.model[1], 'module') else '1.'
                
                ## score behavior from decoder
                layer_ = layer + self.layer_map['decoder']
                
                score = score_model_behavior(
                    model_id=model_id,
                    model=self.model,
                    layer=layer_,
                    benchmark=benchmark_identifier,
                )

                benchmark_log[benchmark_identifier+'_decoder'] = score
                if self.hparams.verbose: print(f'layer: {layer}, {benchmark_log}')
                
            self.log_dict(benchmark_log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        gc.collect()

    def load_benchmarks(self):
        # benchmark loaders use very large batch_size
        # add custom val benchmark here.
        benchmarks = {}
        batch_size = 10000
        if 'NeuralData' in self.dm.keys():
            if self.hparams.benchmarks[0] == 'All':
                self.hparams.benchmarks = self.BENCHMARKS

            # these are not really necessary anymore..
            #if 'fneurons.fstimuli' in self.hparams.benchmarks:
            #    if self.hparams.verbose:
            #        print('\nvalidating on fitted neurons and fitted stimuli')

            #    benchmarks['fneurons.fstimuli'] = self.dm['NeuralData'].val_dataloader(
            #        stimuli_partition='train', neuron_partition=0, batch_size=batch_size
            #    )
            #    
            if 'fneurons.ustimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on fitted neurons and unfitted stimuli')

                benchmarks['fneurons.ustimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='test', neuron_partition=0, batch_size=batch_size
                )

            #if 'uneurons.fstimuli' in self.hparams.benchmarks:
            #    if self.hparams.verbose:
            #        print('\nvalidating on unfitted neurons and fitted stimuli')
            #    
            #    benchmarks['uneurons.fstimuli'] = self.dm['NeuralData'].val_dataloader(
            #        stimuli_partition='train', neuron_partition=1, batch_size=batch_size
            #    )

            if 'uneurons.ustimuli' in self.hparams.benchmarks:
                if self.hparams.verbose:
                    print('\nvalidating on unfitted neurons and unfitted stimuli')
                
                benchmarks['uneurons.ustimuli'] = self.dm['NeuralData'].val_dataloader(
                    stimuli_partition='test', neuron_partition=1, batch_size=batch_size
                )
        
        # if test flag, don't load the rest of the benchmarks.
        if self.hparams.test:
            return benchmarks

        if 'nano.coco' in self.hparams.benchmarks:
            # load manymonkeys test set, animal nano, with COCO stimuli
            benchmarks['nano.coco'] = NeuralDataModule(
                self.hparams, neuraldataset='COCO', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['nano.left'],
                neurons='All', batch_size=batch_size,
            )

        if 'bento.coco' in self.hparams.benchmarks:
            # load manymonkeys test set, animal nano, with COCO stimuli
            benchmarks['bento.coco'] = NeuralDataModule(
                self.hparams, neuraldataset='COCO', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['bento.left'],
                neurons='All', batch_size=batch_size,
            )

        if 'magneto.var6' in self.hparams.benchmarks:
            # load manymonkeys test set, animal magneto, var 6
            benchmarks['magneto.var6'] = NeuralDataModule(
                self.hparams, neuraldataset='manymonkeysval', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['magneto.left', 'magneto.right'],
                neurons='All', batch_size=batch_size, 
            )

        if 'magneto.var6' in self.hparams.benchmarks:
            # load manymonkeys test set, animal magneto, var 6
            benchmarks['magneto.left.var6'] = NeuralDataModule(
                self.hparams, neuraldataset='manymonkeysval', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['magneto.left'],
                neurons='All', batch_size=batch_size, 
            )

        if 'nano.var6' in self.hparams.benchmarks:
            # load manymonkeys test set, animal nano, var 6
            benchmarks['nano.var6'] = NeuralDataModule(
                self.hparams, neuraldataset='manymonkeysval', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['nano.left', 'nano.right'],
                neurons='All', batch_size=batch_size,
            )

        if 'nano.left.var6' in self.hparams.benchmarks:
            # load manymonkeys test set, animal nano, var 6
            benchmarks['nano.left.var6'] = NeuralDataModule(
                self.hparams, neuraldataset='manymonkeysval', num_workers=1
            ).val_dataloader(
                stimuli_partition='test', neuron_partition=0, 
                animals=['nano.left'],
                neurons='All', batch_size=batch_size,
            )

        return benchmarks

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx, mode='val')

    # old train step -- uses two dataloaders for neural fit and stimuli classification
    #def training_step(self, batch, batch_idx):
    #    #### refactor -- take a batch, run stimuli, and apply whatever losses to all outcomes at once.
    #    losses = []
    #    import pdb; pdb.set_trace()

    #    for dataloader_idx, batch_ in enumerate(batch):
    #        if dataloader_idx == 0:
    #            losses.append(
    #                self.loss_weights[dataloader_idx]*self.classification(
    #                    batch_, 'train'
    #                )
    #            )
    #        
    #            # for less than even mix of neural data
    #            if ch.rand(1) > self.hparams.mix_rate:
    #                return sum(losses)

    #        # this assumes dataloader_idx is the dataloader for IT. 
    #        # fine for now, but need to generalize if we wanted to fit multiple layers.
    #        elif (dataloader_idx == 1) & (self.loss_weights[dataloader_idx] != 0):
    #            if not self.hparams.adapt_bn_to_stim: self.model.eval()
    #            losses.append(
    #                self.loss_weights[dataloader_idx]*self.similarity(
    #                    batch_, 'IT', 'train'
    #                )
    #            )
    #            if not self.hparams.adapt_bn_to_stim: self.model.train()

    #        elif (dataloader_idx == 2) & (self.loss_weights[dataloader_idx] != 0):
    #            if not self.hparams.adapt_bn_to_stim: self.model.eval()
    #            losses.append(
    #                self.loss_weights[dataloader_idx]*self.classification(
    #                    batch_, 
    #                    'train', 
    #                    output_inds=[1000, 1008], 
    #                    dataset='Stimuli',
    #                    adversarial=self.hparams.adv_train_images
    #                )
    #            )
    #            if not self.hparams.adapt_bn_to_stim: self.model.train()

    #    return sum(losses)

    def loss_weights_map(self, loss_name):
        mapping = {
            'ImageNet' : self.loss_weights[0],
            'Neural' : self.loss_weights[1],
            'StimClass' : self.loss_weights[2]
        }

        return mapping[loss_name]

    def training_step(self, batch, batch_idx):
        if self.hparams.causal:
            return self._training_step_causal(batch, batch_idx)
        else:
            return self._training_step(batch, batch_idx)

    def _training_step(self, batch, batch_idx):
        losses = []

        losses.append(
            self.loss_weights_map('ImageNet')*self.classification(
                batch['ImageNet'], 'train'
            )
        )
                
        # for less than even mix of neural data
        if ch.rand(1) > self.hparams.mix_rate:
            return sum(losses)

        neural_loss, stim_class_loss = self.similarity_and_classification(
            batch['NeuralData'], 'IT', 'train', adversarial=self.hparams.adv_train_images
        )

        losses.append(self.loss_weights_map('Neural')*neural_loss)
        losses.append(self.loss_weights_map('StimClass')*stim_class_loss)

        return sum(losses)

    def _training_step_causal(self, batch, batch_idx):
        # stochastically zero grads for neural similarity. always zero before step 2500, so HVM accuracy is equilabrated
        if (ch.rand(1) > self.hparams.mix_rate) or (self.global_step < 2500):
            neural_loss_weight = 0
        else:
            neural_loss_weight = self.loss_weights_map('Neural')


        losses = []

        losses.append(
            self.loss_weights_map('ImageNet')*self.classification(
                batch['ImageNet'], 'train'
            )
        )

        neural_loss, stim_class_loss = self.similarity_and_classification(
            batch['NeuralData'], 'IT', 'train', adversarial=self.hparams.adv_train_images
        )

        losses.append(neural_loss_weight*neural_loss)
        losses.append(self.loss_weights_map('StimClass')*stim_class_loss)

        return sum(losses)

    def unpack_batch(self, batch, flag):
        X, H, Y = None, None, None
        if flag == 'classification':
            if len(batch) == 3:
                # batches from neural dataloader
                X, H, Y = batch
            elif len(batch) == 2:
                # batches from imagenet dataloader
                X, Y = batch
            else:
                raise NameError(f'Unexpected batch length {len(batch)}!')

        elif flag == 'similarity':
            if len(batch) == 3:
                X, H, Y = batch
            elif len(batch) == 2:
                X, H = batch
            else:
                raise NameError(f'Unexpected batch length {len(batch)}!')
        
        if Y is not None:
            Y = Y.long().cuda()

        return X, H, Y

    def classification(self, batch, mode, output_inds=[0,1000], dataset='ImageNet', adversarial=False):
        self.set_bn(mode=dataset)
        X, H, Y = self.unpack_batch(batch, flag='classification')
        
        if adversarial:
            X = self.adversaries[f'{mode}_class_adversary'].generate(
                X, Y, F.cross_entropy, output_inds=output_inds
            )

        Y_hat = self.model(X)[:, output_inds[0]:output_inds[1]]

        loss = F.cross_entropy(Y_hat, Y)
        acc1, acc5 = self.__accuracy(Y_hat, Y, topk=(1,5))

        # ??
        if mode == 'train':
            pass

        log = {
            f'{dataset}_{mode}_loss' : loss,
            f'{dataset}_{mode}_acc1' : acc1,
            f'{dataset}_{mode}_acc5' : acc5
        }
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def similarity(self, batch, region, mode, adversarial=False):
        self.set_bn(mode='Stimuli')
        X, H, Y = self.unpack_batch(batch, flag='similarity')

        if adversarial:
            # adversarially attack on labels. requires HVM readouts to be trained.
            X = self.adversaries[f'{mode}_neural_adversary'].generate(
                X, Y, F.cross_entropy, output_inds=[1000,1008]
            )

        _ = self.model(X)
        H_hat = self.regions[region].output

        # this allows to test with a different loss than the train loss.
        neural_loss_fnc = self.neural_loss if mode == 'train' else self.neural_val_loss
        loss = neural_loss_fnc(H, H_hat)
        log = {f'{neural_loss_fnc.name}_{mode}' : loss}

        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def similarity_and_classification(self, batch, region, mode, dataset='Stimuli', adversarial=False):
        self.set_bn(mode=dataset)
        X, H, Y = self.unpack_batch(batch, flag='similarity')

        if adversarial:
            # adversarial attack on labels. requires HVM readouts to be trained.
            X = self.adversaries[f'{mode}_class_adversary'].generate(
                X, Y, F.cross_entropy, output_inds=[1000,1008]
            )

        Y_hat = self.model(X)[:, 1000:1008]
        H_hat = self.regions[region].output

        # this allows to test with a different loss than the train loss.
        neural_loss_fnc = self.neural_loss if mode == 'train' else self.neural_val_loss
        neural_loss = neural_loss_fnc(H, H_hat)

        # and compute classification loss  accuracy
        class_loss = F.cross_entropy(Y_hat, Y)
        acc1, acc5 = self.__accuracy(Y_hat, Y, topk=(1,5))

        log = {
            f'{neural_loss_fnc.name}_{mode}' : neural_loss,
            f'{dataset}_{mode}_loss' : class_loss,
            f'{dataset}_{mode}_acc1' : acc1,
            f'{dataset}_{mode}_acc5' : acc5
        }

        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return (neural_loss, class_loss)

    def configure_optimizers(self):
        param_list, lr = self.parameters(), self.hparams.lr
        
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

    def on_train_epoch_end(self):
        # better memory management
        gc.collect()

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
        model = add_normalization(model, normalization=self.layer_map['normalization'])
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
        parser.add_argument('-adapt', '--adapt_bn_to_stim', dest='adapt_bn_to_stim', type=int, default=1)
        parser.add_argument('-multi_bn', '--multi_bn', dest='multi_bn', type=int, default=0)
        parser.add_argument('-mix_rate', '--mix_rate', dest='mix_rate', type=float, default=1)
        parser.add_argument('-causal', '--causal', dest='causal', type=int, default=0)
        parser.add_argument('--record-time', dest='record_time', action='store_true')
        
        return parser
