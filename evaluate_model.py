import os, glob, argparse, random

import pandas as pd
import numpy as np
import torch as ch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier

from cornet import cornet_s

from datamodules.neural_datamodule import NeuralDataModule
from models.helpers import layer_maps, add_normalization, add_outputs, Mask

def main(args):
    weight_paths = [y for x in os.walk(f'logs/{args.logdir}') for y in glob.glob(os.path.join(x[0], 'epoch*.ckpt'))]
    for weight_path in weight_paths:
        do_the_thing(weight_path, args)

def do_the_thing(weight_path, args):
    # load model
    model = load_model(weight_path, args)
    
    # load dataset
    X, Y = load_dataset(args)

    # load attack
    adversary = load_adversary(args, model)
    X_adv = adversary.generate(x=X, y=Y)

    clean_accuracy = evaluate(model, X, Y)
    print(f"Accuracy on benign test examples: {clean_accuracy * 100}")

    adv_accuracy = evaluate(model, X_adv, Y)
    print(f"Accuracy on eps={args.eps} adversarial test examples: {adv_accuracy * 100}")
    results = {k:[v] for (k,v) in vars(args).items()}
    results['weight_path'] = [weight_path]
    results['clean_acc'] = [clean_accuracy]
    results['adv_acc'] = [adv_accuracy]

    save_path = weight_path.split('checkpoint')[0]
    save_path += f'ds_{args.dataset}-attack_{args.attack}-norm_{args.norm}-eps_{args.eps}-iter_{args.max_iter}.csv'

    pd.DataFrame(results).to_csv(save_path)

def evaluate(model, X, Y):
    predictions = model.predict(X)
    accuracy = np.sum(np.argmax(predictions, axis=1) == Y) / len(Y)
    return accuracy

def load_dataset(args):
    args.image_size = 224
    args.dims = (3, 224, 224)
    args.num_workers = 1
    args.batch_size = 320
    args.stimuli = 'All' 

    # data augmentation parameters -- off
    args.neural_train_transform = 0
    args.translate = "None"
    args.rotate = "None"
    args.scale = "None"
    args.shear = "None"

    if args.dataset == 'HVM_var3':
        loader = NeuralDataModule(
            args, neuraldataset='SachiMajajHongPublic', num_workers=1,
        ).val_dataloader(
            stimuli_partition='test', neuron_partition=1, batch_size=320
        )
    elif args.dataset == 'HVM_var6':
        loader = NeuralDataModule(
            args, neuraldataset='manymonkeysval', num_workers=1
        ).val_dataloader(
            stimuli_partition='test', neuron_partition=0, 
            animals=['magneto.left', 'magneto.right'],
            neurons='All', batch_size=320, 
        )

    for batch in loader:
        pass

    X, H, Y = batch
    return X.numpy(), Y.numpy()

def load_adversary(args, model):
    if args.attack == 'PGD':
        from art.attacks.evasion import ProjectedGradientDescent

        if args.norm == 'inf':
            norm = args.norm
        else:
            norm = eval(args.norm)

        attack = ProjectedGradientDescent(
            estimator=model, 
            norm=norm,
            max_iter=args.max_iter, 
            eps=args.eps, 
            eps_step=(args.eps / args.stepf),
            targeted=False,
            verbose=bool(args.verbose)
        )
    return attack
    
def load_model(path_to_weights, args):
    # load the model architecture with the normalization preprocessor
    model = add_normalization(cornet_s(pretrained=False))
    model = add_outputs(model, out_name='decoder.linear', n_outputs=8)

    # load weights and strip pesky 'model.' prefix
    state_dict = ch.load(path_to_weights)
    weights = {k.replace('model.', '') :v for k,v in state_dict['state_dict'].items()}

    # load the architecture with the trained model weights
    model.load_state_dict(weights)

    # mask to just the HVM outputs
    model = Mask(model, start=1000, stop=1008)

    # model in eval mode............
    model.eval()

    # wrap with ART for adv attacks
    model = ART_wrap_model(model)
    return model

def ART_wrap_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    mean = np.array([0, 0, 0]).reshape((3, 1, 1))
    std = np.array([1, 1, 1]).reshape((3, 1, 1))
    
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        preprocessing=(mean, std),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1008,
    )

    return classifier


def get_args(*args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training. ')
    parser.add_argument('--logdir', type=str, help='Path to the network weights')
    parser.add_argument('--dataset', type=str, default='HVM_var6', help='Path to the network weights')
    parser.add_argument('--attack', dest='attack', default='PGD', help='tag for finding and naming features')
    parser.add_argument('--norm', dest='norm', default='inf', help='tag for finding and naming features')
    parser.add_argument('--eps', dest='eps', type=float, help='tag for finding and naming features')
    parser.add_argument('--max_iter', dest='max_iter', type=int, default=64, help='tag for finding and naming features')
    parser.add_argument('--ensemble', dest='ensemble', type=int, default=1, help='tag for finding and naming features')
    parser.add_argument('--stepf', dest='stepf', type=int, default=4, help='tag for finding and naming features')
    parser.add_argument('--restarts', dest='restarts', type=int, default=0, help='tag for finding and naming features')
    parser.add_argument('--fix', dest='fix', type=int, default=0, help='Fix stochasticity in stochastic models')
    parser.add_argument('--overwrite', dest='overwrite', type=int, default=0, help='tag for finding and naming features')
    parser.add_argument('--meancase', dest='meancase', type=int, default=0, help='tag for finding and naming features')
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='tag for finding and naming features')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_args())
