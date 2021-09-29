import torch as ch
import torch.nn as nn

layer_maps = {
    'mobilenet_v2' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'features.14',
        'decoder' : 'classifier'
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
        'decoder' : 'decoder.avgpool'
    }
}


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = ch.tensor(mean).reshape(3,1,1).cuda()
        self.std = ch.tensor(std).reshape(3,1,1).cuda()

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

def add_normalization(model, **kwargs):
    return nn.Sequential(Normalize(**kwargs), model)

# extract intermediate representations
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output#.clone()

    def close(self):
        self.hook.remove()
