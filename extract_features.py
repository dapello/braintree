import os
import functools

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import torch as ch
import torch.nn as nn

from cornet import cornet_s
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images


def main():
    # an example path to model weights
    identifier = 'manymonkeys_btCORnet_S'
    path_to_weights = 'trained_models/model_cornet_s-loss_logCKA-ds_manymonkeys-fanimals_All-tanimals_All-regions_IT-trials_All-neurons_All-stimuli_640.ckpt'
    #identifier = 'btCORnet_S'
    #path_to_weights = 'trained_models/model_cornet_s-loss_logCKA-ds_sachimajajhong-fanimals_All-tanimals_All-regions_IT-trials_All-neurons_188-stimuli_5760.ckpt'
    #identifier = 'CORnet_S_control'
    #path_to_weights = 'trained_models/model_cornet_s-control.ckpt'
    stimuli_path = 'stimuli/shined'
    print(f'Extracting {stimuli_path} features from {identifier} at {path_to_weights}')
    
    model = load_model(path_to_weights, loc='cpu')
    wrapped_model = wrap_model(model, identifier)
    layers = ['1.module.'+layer for layer in ['V1', 'V2', 'V4', 'IT', 'decoder.avgpool']]
    #layers = ['1.module.decoder.avgpool']
    features = extract_features(wrapped_model, layers, path=stimuli_path)
    save_features(features, identifier, stimuli_path)
    
# this normalization preprocessor makes the model take in images with pixel ranges between [0-1]
class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = ch.tensor(mean).reshape(3,1,1)#.cuda()
        self.std = ch.tensor(std).reshape(3,1,1)#.cuda()

    def forward(self, x): 
        x = x - self.mean
        x = x / self.std
        return x

# wrap a model, returning a new model with the Normalize then the model
def add_normalization(model, **kwargs):
    return nn.Sequential(Normalize(**kwargs), model)


def load_model(path_to_weights, loc='cpu'):

    # load the model architecture with the normalization preprocessor
    model = add_normalization(cornet_s(pretrained=False))

    # load weights and strip pesky 'model.' prefix
    state_dict = ch.load(path_to_weights, map_location=ch.device(loc))
    weights = {k.replace('model.', '') :v for k,v in state_dict['state_dict'].items()}
    #weights['1.module.decoder.linear.weight'] = model[1].module.decoder.linear.weight
    #weights['1.module.decoder.linear.bias'] = model[1].module.decoder.linear.bias


    import pdb; pdb.set_trace()

    # load the architecture with the trained model weights
    model.load_state_dict(weights)
    return model

def wrap_model(model, identifier, image_size=224):
    image_size = 224
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size, normalize_mean=(0,0,0), normalize_std=(1,1,1))
    wrapper = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def extract_features(wrapped_model, layers, path):
    files = np.sort([os.path.join(path, f) for f in os.listdir(path)])
    X = load_preprocess_images(files, image_size=224, normalize_mean=(0,0,0), normalize_std=(1,1,1))
    print(f'feature details: min:{X.min()}, max:{X.max()}, shape:{X.shape}')
    
    activations = wrapped_model.get_activations(X, layer_names=layers)
    activations['files'] = np.array([os.path.basename(file) for file in files]).astype('S10')
    
    return activations

def save_features(features, identifier, stimuli_path):
    save_path = f'features/model_{identifier}-stimuli_{os.path.basename(stimuli_path)}.h5'
    f = h5.File(save_path, 'w')
    
    for key in features.keys():
        f.create_dataset(key.split('.')[-1], data=features[key])
    
    f.close()


if __name__ == '__main__':
    main()
