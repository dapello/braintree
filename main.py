import os, glob, argparse, random

import torch as ch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 
from model_lightning import Model_Lightning as MODEL
from datamodules import DATAMODULES

default_save_path = "./save"

def main(hparams):
    if hparams.seed is not None:
        seed_everything(hparams.seed)
        deterministic = True     
    else: 
        deterministic = False
        
    hparams = add_path_names(hparams)
    dm = DATAMODULES[hparams['datamodule']](hparams)
    
    #file_name, save_path, log_save_path, statedict_path = get_path_names(hparams)
    
    # Load checkpoint from previous version 
    ckpt_file = get_ckpt(hparams)
    if ckpt_file is not None:   
        model = MODEL.load_from_checkpoint(ckpt_file)
    else:
        model = MODEL(hparams)

    print(f'save_path = {hparams.save_path}')
    print(f'ckpt_file = {ckpt_file}')
    
    logger = TensorBoardLogger(
        hparams.log_save_path, name=hparams.file_name, 
        version=hparams.v_num
    ) 
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        default_root_dir=hparams.log_save_path,
        gpus=hparams.gpus,   
        max_epochs=hparams.epochs,   
        resume_from_checkpoint = ckpt_file,
        distributed_backend=hparams.distributed_backend, 
        num_nodes=hparams.num_nodes,
        logger=logger, callbacks=[lr_monitor],  #   PrintingCallback()],
        deterministic=deterministic,
    ) 
    
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model, dm)
    
        print('post-training test')
        
        # Load best checkpoint (I don't think that's what's happening -- we're just loading last ckpt?)
        ckpt_file = get_ckpt(hparams)
        model = MODEL.load_from_checkpoint(ckpt_file)
        
        # Save weights from checkpoint
        ch.save(model.model.state_dict(), hparams.statedict_path)
        
        # Test model
        trainer.test(model)


def get_args(*args):
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--seed', type=int, default=42,
                                help='seed for initializing training. ')
    parent_parser.add_argument('--save-path', metavar='DIR', type=str, default=default_save_path,  help='path to save output')
    parent_parser.add_argument('--private', dest='save_path', action='store_false')
    
    parent_parser.add_argument('--num_workers', type=int, default=16,
                               help='how many workers')
    parent_parser.add_argument('--num_nodes', type=int, default=1,
                               help='how many nodes')
    parent_parser.add_argument('--gpus', default=4,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    # data specific arguments. maybe move to DATAMODULES like MODELS?
    parser.add_argument('--datamodule', type=str, default='ImageNet', 
        choices=DATAMODULES.keys(), help='which datamodule to use.')
    parent_parser.add_argument('--neuraldataset', default='None',
        help='which source neural dataset to construct from')
    parser.add_argument('--animals', action='append', default=[],
        help='which animals to load from the dataset. should be of form "nano.right"')
    parent_parser.add_argument('--neurons', default='All',
        help='how many neurons to fit to')
    parent_parser.add_argument('--stimuli', default='All',
        help='how many stimuli to fit to')
    parent_parser.add_argument('--trials', default='All',
        help='how many trials of stimuli presentation to average over')
    parent_parser.add_argument('--window', default='70t170',
        help='time window to average neural data over')

    parser = MODEL.add_model_specific_args(parent_parser)
    args, unknown = parser.parse_known_args(*args) 
    return args 

################

def add_path_names(hparams):
    hparams.file_name = get_filename(hparams)
    hparams.log_save_path = os.path.join(hparams.save_path, 'logs')
    hparams.statedict_path = os.path.join(
        hparams.save_path, 'trained_models', hparams.file_name + '.pt'
    )  
    return hparams

def get_filename(hparams):
    return f'model_{hparams.arch}\
        -loss_{hparams.loss}\
        -ds_kktemporal\
        -animal_{hparams.animals}\
        -regions_{hparams.regions}\
        -trials_{hparams.trials}\
        -neurons_{hparams.neurons}\
        -stimuli_{hparams.stimuli}'

def get_ckpt(hparams):
    if hparams.v_num is None:
        return None 
    else:
        ckpt_path = os.path.join(
            hparams.log_save_path,
            hparams.file_name, 
            'version_' + str(hparams.v_num),
            'checkpoints'
        ) 
        return get_latest_file(ckpt_path)


def get_latest_file(ckpt_path):
    files_path = os.path.join(ckpt_path, '*')
    list_of_ckpt = glob.glob(files_path)
    ckpt_sorted = sorted(
        list_of_ckpt, 
        key = lambda f: (
            int(f.split('-')[0].split('=')[1]), 
            int(f.split('-')[1].split('=')[1].split('.')[0])
        ),
        reverse = True
    )
    print(ckpt_sorted)
    return ckpt_sorted[0]

#################
    
class PrintingCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        print('Scheduler epoch %d' % trainer.lr_schedulers[0]['scheduler'].last_epoch)
        print('Trainer epoch %d' % trainer.current_epoch)
        print('-'*80)    

#################


if __name__ == '__main__':
    main(get_args())
