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

from functools import partial


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


####################

class Model_Lightning(LightningModule):
    
    def __init__(self, hparams, train_size = None, *args, **kwargs): 
        super().__init__()
        
        if isinstance(hparams, dict):  
            # for load_from_checkpoint bug: which uses dict instead of namespace
            hparams = argparse.Namespace(**hparams)
            
        self.hparams = hparams
        self.train_size = train_size
        self.record_time = hparams.record_time
        
        self.model = get_model(hparams.arch, pretrained=hparams.pretrained, *args, **kwargs)

        print('record_time = ', self.record_time)
        
    def forward(self, x):
        return self.model(x)
    
    def loss_fn(self, output, target):
        l = F.cross_entropy(output, target)
        return l

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1,5))            
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc1', acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1,5))            
        self.log('val_loss', loss)
        self.log('val_acc1', acc1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    @classmethod
    def __accuracy(cls, outputs, targets, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        # outputs.shape = [batch, num_class, time_steps]
        # pred.shape = [batch, maxk, time_steps]
        # targets.shape = [batch]

        with torch.no_grad():
            _, pred = outputs.topk(max(topk), 1, True, True)
            view_idx = [-1]+[1]*(pred.dim()-1)
            permute_idx = [i for i in range(1,pred.dim())] + [0]
            correct = pred.eq(targets.view(view_idx).expand_as(pred)).permute(permute_idx)

            res = []
            for k in topk:
                correct_k = correct[:k].float().sum(0, keepdim=False).mean(-1, keepdim=False)   # shape: [time_steps]
                res.append(correct_k)     
            return res
             
    def configure_optimizers(self):
        param_list, lr =  self.parameters(), self.hparams.lr
        lr_list = lr
        
        if self.hparams.optim == 'sgd':
            optimizer = optim.SGD(param_list, lr = lr, weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        elif self.hparams.optim == 'adam':
            optimizer = optim.Adam(param_list, lr = lr, weight_decay=self.hparams.weight_decay)
        else:
            NotImplemented
            
        if self.hparams.scheduler == 'OneCycleLR':
            scheduler = {
                'scheduler' : torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr= lr_list, 
                    steps_per_epoch=self.train_size//self.hparams.batch_size, 
                    epochs=self.hparams.epochs
                ),
                'interval' : 'step'
            }
        elif self.hparams.scheduler == 'ExponentialLR':
            scheduler = {
                'scheduler' : lr_scheduler.ExponentialLR(
                    optimizer, gamma = 0.926
                ),
                'interval' : 'epoch'
            }
        else:
            return [optimizer]

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', choices=MODEL_NAMES, default = 'CORnet-S', 
                            help='model architecture: ' + ' | '.join(MODEL_NAMES))
        parser.add_argument('--epochs', default=100, type=int, metavar='N')
        parser.add_argument('-b', '--batch-size', type=int, metavar='N', default = 256, 
                            help='this is the total batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--scheduler', type=str, default='ExponentialLR')
        parser.add_argument('--lr', '--learning-rate', metavar='LR', dest='lr', type=float, default = 0.01)
        # use lr = 0.1 for OneCycleLR,  0.01 for ExponentialLR
        parser.add_argument('--momentum', metavar='M', type=float, default=0.9)
        parser.add_argument('--lr_exp', help='learning rate modifier', type=float,   default=1.5 )
        parser.add_argument('--wd', '--weight-decay', metavar='W', dest='weight_decay', type=float, default = 1e-4)  # set to 1e-2 for cifar10
        parser.add_argument('--optim', dest='optim', default='sgd') # := {'sgd', 'adam'}1
        parser.add_argument('--dataset', type=str, default='ImageNet')
        parser.add_argument('--v_num', type=int)
        parser.add_argument('--pretrained', dest='pretrained', action='store_true', default = False)
        parser.add_argument('--no-bn', dest='bn', action='store_false', default = True)
        parser.add_argument('--record-time', dest='record_time', action='store_true')
        
        return parser

##############

def get_filename(hparams):
    bn = '_noBN' if not hparams.bn else ''
    optim = '_' + hparams.optim if hparams.optim is not None else ''
    file_name = hparams.arch + bn + optim
    return os.path.join(hparams.dataset, file_name) #, save_path


def get_path_names(hparams):
    file_name = get_filename(hparams)    
    save_path = hparams.save_path or os.getcwd()
    log_save_path = os.path.join(save_path, 'logs')
    statedict_path = os.path.join(save_path, 'trained_models', file_name + '.pt')  
#     print('save_path:', save_path, 'file_name:', file_name)
    return file_name, save_path, log_save_path, statedict_path

################
