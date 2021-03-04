import os, glob, argparse, random
from pprint import pprint

import torch as ch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 
from model_lightning import Model_Lightning as MODEL, CheckpointEveryNSteps
from datamodules import DATAMODULES
from datamodules.neural_datamodule import SOURCES

default_save_path = "./"

def main(hparams):
    if hparams.seed is not None:
        seed_everything(hparams.seed)
        deterministic = True     
    else: 
        deterministic = False
        
    logger = TensorBoardLogger(
        hparams.log_save_path, name=hparams.file_name, 
        version=hparams.v_num
    ) 
    hparams.v_num = logger.version

    dm = DATAMODULES[hparams.datamodule](hparams)
    model = MODEL(hparams)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_callback = ModelCheckpoint(
        verbose=hparams.verbose,
        save_top_k=-1
    )

    trainer = Trainer(
        default_root_dir=hparams.log_save_path,
        gpus=hparams.gpus,   
        max_epochs=hparams.epochs,   
        checkpoint_callback=ckpt_callback,
        distributed_backend=hparams.distributed_backend, 
        num_nodes=hparams.num_nodes,
        logger=logger, callbacks=[lr_monitor],  #   PrintingCallback()],
        deterministic=deterministic,
    ) 
    
    if hparams.evaluate:
        trainer.test(model, test_dataloaders=dm.val_dataloader())
    else:
        trainer.fit(model, dm)

def get_args(*args):
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--seed', type=int, default=42,
                                help='seed for initializing training. ')
    parent_parser.add_argument('--save-path', metavar='DIR', type=str, default=default_save_path,  help='path to save output')
    parent_parser.add_argument('--num_workers', type=int, default=16,
                               help='how many workers')
    parent_parser.add_argument('--num_nodes', type=int, default=1,
                               help='how many nodes')
    parent_parser.add_argument('--gpus', default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    # data specific arguments. maybe move to DATAMODULES like MODELS?
    parent_parser.add_argument('-d', '--datamodule', dest='datamodule', default='ImageNetAndNeuralData', 
                               choices=DATAMODULES.keys(), help='which datamodule to use.')
    parent_parser.add_argument('-nd', '--neuraldataset', dest='neuraldataset', default='kktemporal',
                               choices=SOURCES.keys(), help='which source neural dataset to construct from')
    parent_parser.add_argument('--animals', dest='animals',  nargs='*', default=['All'],
                               help='which animals to load from the dataset. should be of form "nano.right"')
    parent_parser.add_argument('-n', '--neurons', dest='neurons', default='All',
                               help='how many of the train neurons to fit to')
    parent_parser.add_argument('-s', '--stimuli', dest='stimuli', default='All',
                               help='how many of the train stimuli to fit to')
    parent_parser.add_argument('-t', '--trials', dest='trials', default='All',
                               help='how many trials of stimuli presentation to average over')
    parent_parser.add_argument('--window', default='7t17',
                               help='time window to average neural data over. 7t17 => 70ms through 170ms')

    parser = MODEL.add_model_specific_args(parent_parser)
    args, unknown = parser.parse_known_args(*args) 
    args = add_path_names(args)
    if args.verbose: pprint(args)
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
    filename = f'model_{hparams.arch}'\
        + f'-loss_{hparams.neural_loss}'\
        + f'-ds_kktemporal'\
        + f'-animals_{"+".join(hparams.animals)}'\
        + f'-regions_{"+".join(hparams.regions)}'\
        + f'-trials_{hparams.trials}'\
        + f'-neurons_{hparams.neurons}'\
        + f'-stimuli_{hparams.stimuli}'

    return filename

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
