import os, glob, argparse, random
from pprint import pprint

import torch as ch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 
from model_lightning import Model_Lightning as MODEL

from datamodules import DATAMODULES
from datamodules.neural_datamodule import SOURCES
from braintree.benchmarks import list_brainscore_benchmarks

default_save_path = "dev" 

def main(hparams):
    deterministic = seed(hparams)    
    logger = set_logger(hparams)

    dm = { 
        module_name : DATAMODULES[module_name](hparams)
        for module_name in hparams.datamodule
    }

    model = MODEL(hparams, dm)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    ckpt_callback = ModelCheckpoint(
        verbose = hparams.verbose,
        #monitor='val_loss',
        save_last=True,
        #save_top_k = hparams.save_top_k
    )

    trainer = Trainer(
        default_root_dir=hparams.log_save_path,
        devices=hparams.gpus,   
        accelerator='gpu',
        max_epochs=hparams.epochs,   
        #num_sanity_val_steps=-1,
        check_val_every_n_epoch=hparams.val_every,
        limit_val_batches=hparams.val_batches,
        checkpoint_callback=ckpt_callback,
        #distributed_backend=hparams.distributed_backend, 
        num_nodes=hparams.num_nodes,
        logger=logger, callbacks=[lr_monitor],  #   PrintingCallback()],
        deterministic=deterministic,
        multiple_trainloader_mode='min_size',
        profiler="simple",
        log_gpu_memory=True,
        precision=16
    ) 
    
    if hparams.evaluate:
        trainer.test(model, test_dataloaders=[dm[key].val_dataloader() for key in dm])
    else:
        trainer.validate(model)
        trainer.fit(model)

def seed(hparams):
    deterministic = False
    if hparams.seed is not None:
        seed_everything(hparams.seed)
        deterministic = True     

    return deterministic

def set_logger(hparams):
    logger = TensorBoardLogger(
        hparams.log_save_path, name=hparams.file_name, 
        version=hparams.v_num
    ) 
    hparams.v_num = logger.version

    return logger

class PrintingCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        print('Scheduler epoch %d' % trainer.lr_schedulers[0]['scheduler'].last_epoch)
        print('Trainer epoch %d' % trainer.current_epoch)
        print('-'*80)    

#################

def get_args(*args):
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--seed', type=int, default=42,
                                help='seed for initializing training. ')
    parent_parser.add_argument('--save_path', metavar='DIR', type=str, default=default_save_path, 
                               help='path to save output')
    parent_parser.add_argument('--num_workers', type=int, default=4,
                               help='how many workers')
    parent_parser.add_argument('--num_nodes', type=int, default=1,
                               help='how many nodes')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--save_top_k', dest='save_top_k', type=int, default=1,
                               help='how many model checkpoints to save. -1 for all')
    parent_parser.add_argument('--val_batches', dest='val_batches', type=int, default=0.1,
                               help='how many batches (10) / what percent (0.25) of the validation set to run.')
    parent_parser.add_argument('--val_every', dest='val_every', type=int, default=20,
                               help='how frequently to run the validation set.')
    parent_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                               help='prints more details of dataloading / etc')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-aei', '--adv_eval_images', dest='adv_eval_images', action='store_true',
                               help='adversarially evaluate model on validation set')
    parent_parser.add_argument('-aen', '--adv_eval_neural', dest='adv_eval_neural', action='store_true',
                               help='adversarially evaluate model on CenteredKernelAlignment')
    parent_parser.add_argument('-eps', '--epsilon', dest='eps', type=float, default=1/1020,
                               help='maximum L_inf perturbation strength')

    # data specific arguments. maybe move to DATAMODULES like MODELS?
    parent_parser.add_argument('-d', '--datamodule', dest='datamodule', nargs='+', 
                               default=['ImageNet', 'NeuralData', 'StimuliClassification'], choices=DATAMODULES.keys(), 
                               help='which datamodule to use.')
    parent_parser.add_argument('-nd', '--neuraldataset', dest='neuraldataset', default='manymonkeys',
                               choices=SOURCES.keys(), help='which source neural dataset to construct from')
    parent_parser.add_argument('--benchmarks', dest='benchmarks',  nargs='*', default=['fneurons.ustimuli'],
                               choices=['None', 'All'] + MODEL.BENCHMARKS,
                               help='which metrics to collect at the end of the epoch')
    parent_parser.add_argument('-BS', '--BS_benchmarks', dest='BS_benchmarks',  nargs='*', default=['None'],
                               choices=['None'] + list_brainscore_benchmarks(),
                               help='which metrics to collect at the end of the epoch')
    parent_parser.add_argument('--fit_animals', dest='fit_animals',  nargs='*', default=['All'],
                               help='which animals to fit from the dataset, should be of form "nano.right"')
    parent_parser.add_argument('--test_animals', dest='test_animals',  nargs='*', default=['All'],
                               help='which animals to test on from the dataset, of form "nano.right"')
    parent_parser.add_argument('-n', '--neurons', dest='neurons', default='All',
                               help='how many of the train neurons to fit to')
    parent_parser.add_argument('-s', '--stimuli', dest='stimuli', default='All',
                               help='how many of the train stimuli to fit to')
    parent_parser.add_argument('-t', '--trials', dest='trials', default='All',
                               help='how many trials of stimuli presentation to average over')
    parent_parser.add_argument('-ntt', '--neural-train-transform', dest='neural_train_transform', type=int, default=1,
                               help='if 1, train with input aug on neural data; if 0, no input aug')
    parent_parser.add_argument('-gn', '--gaussian-noise', dest='gaussian_noise', type=float, default=0.01,
                               help='data augmentation with Gaussian noise')
    parent_parser.add_argument('-gb', '--gaussian-blur', dest='gaussian_blur', type=str, default='3,(0.1,3.0)',
                               help='data augmentation with Gaussian blur')
    parent_parser.add_argument('--translate', dest='translate', type=str, default='(0.0625, 0.0625)',
                               help='data augmentation vertical or horizontal translation by up to .5 degrees')
    parent_parser.add_argument('--rotate', dest='rotate', type=str, default='(-0.5, 0.5)',
                               help='data augmentation rotation by up to .5 degrees')
    parent_parser.add_argument('--scale', dest='scale', type=str, default='(0.9, 1.1)',
                               help='data augmentation size jitter by up to a little more than .5 degrees')
    parent_parser.add_argument('--shear', dest='shear', type=str, default='(0.9375, 1.0625, 0.9375, 1.0625)',
                               help='data augmentation shear jitter by up to .5 degrees')
    parent_parser.add_argument('--brightness', dest='brightness', type=str, default='0.2',
                               help='data augmentation brightness jitter')
    parent_parser.add_argument('--contrast', dest='contrast', type=str, default='(0.5,1.5)',
                               help='data augmentation contrast jitter')
    parent_parser.add_argument('--saturation', dest='saturation', type=str, default='0.',
                               help='data augmentation saturation jitter')
    parent_parser.add_argument('--hue', dest='hue', type=str, default='0.',
                               help='data augmentation hue jitter')
    parent_parser.add_argument('--window', default='7t17',
                               help='time window to average neural data over. 7t17 => 70ms through 170ms')

    parser = MODEL.add_model_specific_args(parent_parser)
    args, unknown = parser.parse_known_args(*args) 
    args = add_path_names(args)
    if args.verbose: pprint(args)
    return args 

def add_path_names(hparams):
    hparams.file_name = get_filename(hparams)
    hparams.log_save_path = os.path.join('./logs', hparams.save_path)
    hparams.statedict_path = os.path.join(
        hparams.save_path, 'trained_models', hparams.file_name + '.pt'
    )  
    return hparams

def get_filename(hparams):
    filename = f'model_{hparams.arch}'\
        + f'-loss_{hparams.neural_loss}'\
        + f'-ds_{hparams.neuraldataset}'\
        + f'-fanimals_{"+".join(hparams.fit_animals)}'\
        + f'-tanimals_{"+".join(hparams.test_animals)}'\
        + f'-regions_{"+".join(hparams.regions)}'\
        + f'-trials_{hparams.trials}'\
        + f'-neurons_{hparams.neurons}'\
        + f'-stimuli_{hparams.stimuli}'

    return filename

################

if __name__ == '__main__':
    main(get_args())
