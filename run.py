import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

import cornet
from braintree.losses import CenteredKernelAlignment
from braintree.dataloader import ConcatDataset, get_neural_dataset

from PIL import Image
Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='Z',
                    help='which model to train')
parser.add_argument('--pretrained', default=False,
                    help='start from pretrained CORnet')
parser.add_argument('--regions', choices=['V1', 'V2', 'V4', 'IT'], action='append', 
                    help='which CORnet layer to match')
parser.add_argument('--neuraldata', required=True,
                    help='which neural dataset to load (not implemented)')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R model)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')


FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


if FLAGS.ngpus > 0:
    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=FLAGS.times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)

    if FLAGS.ngpus == 0:
        model = model.module  # remove DataParallel
    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.01,  # how often save output during training
          save_val_epochs=.01,  # how often save output during validation
          save_model_epochs=.01,  # how often save model weigths
          save_model_secs=60 * 10  # how often save model (in sec)
          ):

    model = get_model(pretrained=FLAGS.pretrained)
    
    # add hooks to extract intermediate layers, for matching to neural data
    model.intermediate = {
        f'region-{region}' : Hook(model._modules['module']._modules[region])
        for region in FLAGS.regions
    }
    
    print('>>> model loaded and hooked')

    # ImageNetAndSimilarity trains the model on imagenet and a neural similarity loss
    trainer = ImageNetAndSimilarityTrain(model, FLAGS.neuraldata)
    validator = ImageNetAndSimilarityVal(model, FLAGS.neuraldata)

    print('>>> trainer and validator loaded')

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
        print(f'save train progress at steps: {save_train_steps}')

    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
        print(f'save val steps: {save_val_steps}')

    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)
        print(f'save model steps: {save_model_steps}')

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan

        print('>>> start epoch,', epoch)

        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    trainer.model.train()

            if FLAGS.output_path is not None:
                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{global_step}.pth.tar'))
                                                           #f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record
                        print(f'results:\n{results}')

            data_load_start = time.time()


class ImageNetAndSimilarityTrain(object):
    def __init__(self, model, neural_datapath, Similarity_Loss=CenteredKernelAlignment):
        self.name = 'train'
        self.model = model 
        self.neural_datapath = neural_datapath 
        self.data_loader = self.get_dataloader() 
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         FLAGS.lr,
                                         momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.classification_loss = nn.CrossEntropyLoss()
        self.similarity_loss = Similarity_Loss()
        if FLAGS.ngpus > 0:
            self.classification_loss = self.classification_loss.cuda()
            self.similarity_loss = self.similarity_loss.cuda()
            
    def get_dataloader(self):
        print('>>> getting dataloader')
        print('>>> loading ImageNet')
        ImageNet_Train = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        print('>>> ImageNet loader retrieved')

        # neural data is already formatted for the network.
        NeuralData_Train, NeuralData_Test = get_neural_dataset(self.neural_datapath)
        print('>>> NeuralData loader retrieved')

        data_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 ImageNet_Train,
                 NeuralData_Train
             ),
             batch_size=FLAGS.batch_size, 
             shuffle=True,
             num_workers=FLAGS.workers, 
             pin_memory=True
        )
        print('>>> full data loader retrieved')

        return data_loader

    # need to adapt this to accomodate new dataloader (do we still wanna use a dict as input?)
    def __call__(self, frac_epoch, data):
        #print('>>> called trainer, ',frac_epoch)
        ((imnet_inp, imnet_label), neural_data) = data
        #print(f'>>> imnet_inp shape: {imnet_inp.shape}, imnet_label shape: {imnet_label.shape}')
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if FLAGS.ngpus > 0:
            imnet_label = imnet_label.cuda(non_blocking=True)
            if 'Labels' in neural_data.keys():
                neural_data['Labels'] = neural_data['Labels'].cuda(non_blocking=True)

        imnet_output = self.model(imnet_inp)
        #print(f'>>> output shape: {output.shape}')

        # quantify imnet classification loss
        classification_loss = self.classification_loss(imnet_output, imnet_label)
        
        stimuli_output = self.model(neural_data['Stimuli'])
        #classification_loss += self.classification_loss(output, neural_data['Labels'])

        
        # record training data
        record = {}
        record['classification_loss'] = classification_loss.item()
        record['top1'], record['top5'] = accuracy(imnet_output, imnet_label, topk=(1, 5))
        record['top1'] /= len(imnet_output)
        record['top5'] /= len(imnet_output)
        record['learning_rate'] = self.lr.get_lr()[0]

        similarity_losses = {
            key : self.similarity_loss(
                self.model.intermediate[key].output,
                neural_data[key].cuda()
            ) 
            for key in neural_data if 'region-' in key
        }

        for key in similarity_losses:
            record[f'{key}_loss'] = similarity_losses[key].item()

        self.optimizer.zero_grad()
        classification_loss.backward()
        for key in similarity_losses:
            similarity_losses[key].backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record

class ImageNetAndSimilarityVal(object):
    def __init__(self, model, neural_datapath, Similarity_Loss=CenteredKernelAlignment):
        self.name = 'val'
        self.model = model
        self.neural_datapath = neural_datapath 
        self.data_loader = self.get_dataloader() 
        self.classification_loss = nn.CrossEntropyLoss()
        self.similarity_loss = Similarity_Loss()
        #self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.classification_loss = self.classification_loss.cuda()
            self.similarity_loss = self.similarity_loss.cuda()

    def get_dataloader(self):
        print('>>> getting dataloader')
        print('>>> loading ImageNet')
        ImageNet_Val = torchvision.datasets.ImageFolder(
                os.path.join(FLAGS.data_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        print('>>> ImageNet loader retrieved')

        # neural data is already formatted for the network.
        NeuralData_Train, NeuralData_Test = get_neural_dataset(self.neural_datapath)
        print('>>> NeuralData loader retrieved')

        data_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 ImageNet_Val,
                 NeuralData_Test
             ),
             batch_size=FLAGS.batch_size, 
             shuffle=False,
             num_workers=FLAGS.workers, 
             pin_memory=True
        )
        print('>>> full data loader retrieved')

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for data in tqdm.tqdm(self.data_loader, desc=self.name):
                ((imnet_inp, imnet_label), neural_data) = data
                if FLAGS.ngpus > 0:
                    imnet_label = imnet_label.cuda(non_blocking=True)
                    if 'Labels' in neural_data.keys():
                        neural_data['Labels'] = neural_data['Labels'].cuda(non_blocking=True)

                imnet_output = self.model(imnet_inp)

                # quantify imnet classification loss
                classification_loss = self.classification_loss(imnet_output, imnet_label)
                record['loss'] += classification_loss.item()
                p1, p5 = accuracy(imnet_output, imnet_label, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5
                
                stimuli_output = self.model(neural_data['Stimuli'])

                similarity_losses = {
                    key : self.similarity_loss(
                        self.model.intermediate[key].output,
                        neural_data[key].cuda()
                    ) 
                    for key in neural_data if 'region-' in key
                }
                
                for key in similarity_losses:
                    if f'{key}_loss' not in record:
                        record[f'{key}_loss'] = similarity_losses[key].item()
                    else:
                        record[f'{key}_loss'] += similarity_losses[key].item()

        for key in record:
            if 'loss' in key:
                record[key] /= len(self.data_loader)
            else:
                record[key] /= self.data_loader.dataset.__len__()

        record['dur'] = (time.time() - start) / len(self.data_loader)
        print(record)

        return record

class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
                os.path.join(FLAGS.data_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)
        print(record)

        return record

# extract intermediate representations
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output#.clone()
    def close(self):
        self.hook.remove()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
