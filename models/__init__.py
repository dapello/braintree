from cornet import *

from efficientnet_pytorch import EfficientNet

def efficientnet_b0(pretrained=True):
    assert pretrained == True
    return EfficientNet.from_pretrained('efficientnet-b0')

layer_maps = {
    'efficientnet_b0' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : '_blocks.11',
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
        'decoder' : 'decoder.avgpool'
    }
}
