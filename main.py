import argparse
import random

import torch
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger #, WandbLogger
from model_lightning import Model_Lightning as MODEL, get_path_names
from datamodules import DATAMODULES

default_save_path = "./save"

def main(hparams):
    
    if hparams.seed is not None:
        seed_everything(hparams.seed)
        deterministic = True     
        # import torch.backends.cudnn as cudnn
        # cudnn.deterministic = True
    else: 
        deterministic = False
        

    dm = DATAMODULES[hparams.dataset](  # data_dir = hparams.data_path, 
        batch_size = hparams.batch_size, 
        num_workers = hparams.num_workers
    ) 
    
    model = MODEL(hparams, num_classes = dm.num_classes, train_size = len(dm.train_dataloader().dataset))
    
    file_name, save_path, log_save_path, statedict_path = get_path_names(hparams)
    # maybe save them into hparams?
    
    # Load checkpoint from previous version 
    ckpt_file =  get_ckpt(hparams.v_num, log_save_path, file_name)
    if ckpt_file is not None:   
        model = MODEL.load_from_checkpoint(ckpt_file, num_classes=dm.num_classes, train_size=len(dm.train_dataloader().dataset))
#     https://github.com/PyTorchLightning/pytorch-lightning/issues/1772    # lr_scheduler's epoch off by one   

    print(f'save_path = {save_path}')
    print(f'ckpt_file = {ckpt_file}')
    
    logger = TensorBoardLogger(log_save_path, name=file_name, version=hparams.v_num) 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    

    trainer = Trainer(
        default_root_dir=log_save_path,
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
        
        # Load best checkpoint
        ckpt_file =  get_ckpt(model.logger.version, log_save_path,  file_name)
        model = MODEL.load_from_checkpoint(ckpt_file,  num_classes = dm.num_classes)
        
        # Save weights from checkpoint
        torch.save(model.model.state_dict(), statedict_path)
        
        # Test model
        trainer.test(model)
