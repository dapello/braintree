import torch as ch
import torch.nn as nn

layer_maps = {
    'efficientnet_b0' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : '_blocks.15',
        'decoder' : '_blocks.15',
    },
    'mobilenet_v2' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'features.14',
        'decoder' : 'classifier'
    },
    'mobilenet_v3_large' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'features.13',
        'decoder' : 'avgpool'
    },
    'resnet18' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'layer4.1',
        'decoder' : 'avgpool'
    },
    'resnet50' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'layer4.0',
        'decoder' : 'avgpool'
    },
    'cornet_s' : {
        'V1' : 'V1',
        'V2' : 'V2',
        'V4' : 'V4',
        'IT' : 'IT',
        'decoder' : 'decoder.avgpool',
        'output' : 'decoder.linear'
    },
    'vonecornet_s' : {
        'V1' : 'V1',
        'V2' : 'V2',
        'V4' : 'V4',
        'IT' : 'IT',
        'decoder' : 'decoder.avgpool',
        'output' : 'decoder.linear'
    },
}

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = ch.tensor(mean).reshape(3,1,1)#.cuda()
        self.std = ch.tensor(std).reshape(3,1,1)#.cuda()

    def forward(self, x):
        x = x - self.mean.type_as(x)
        x = x / self.std.type_as(x)
        return x

def add_normalization(model, **kwargs):
    return nn.Sequential(Normalize(**kwargs), model)

def add_outputs(model, out_name, n_outputs=8, verbose=False):
    # walk model to get to final output module (layer4.3.linear => linear)
    model_ = model[1]
    model_ = model_.module if hasattr(model_, 'module') else model_

    layer_tree = [model_]
    for l in out_name.split('.'):
        layer_tree.append(getattr(layer_tree[-1], l))

    old_out = layer_tree[-1]
        
    if verbose: print(old_out)

    # assuming output is always linear w a bias..
    new_out = ch.nn.Linear(
        in_features=old_out.in_features, 
        out_features=old_out.out_features + n_outputs
    ).cuda()

    # write params of old output to new output
    for old_p, new_p in zip(old_out.parameters(), new_out.parameters()):
        if verbose: print(f'old_p.shape: {old_p.shape}')
        if verbose: print(f'new_p.shape: {new_p.shape}')
        new_p.requires_grad = False
        new_p[:old_p.shape[0]] = old_p.data 
        new_p.requires_grad = True
        
    # set the new output to the output layer
    setattr(layer_tree[-2], l, new_out)
    return model

# extract intermediate representations
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        
        self.output = None
        #self.output = []

    def hook_fn(self, module, input, output):
        self.output = output#.clone()
        #self.output.append(output)

    def close(self):
        self.hook.remove()
