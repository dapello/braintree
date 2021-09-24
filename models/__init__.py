from cornet import *

<<<<<<< HEAD
def efficientnet_b0(pretrained=True):
    from efficientnet_pytorch import EfficientNet
    assert pretrained == True
    return EfficientNet.from_pretrained('efficientnet-b0')
=======
from efficientnet_pytorch import EfficientNet

def efficientnet_b0(pretrained=True):
    assert pretrained == True
    return EfficientNet.from_pretrained('efficientnet-b0')

layer_maps = {
    'efficientnet_b0' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : '_blocks.15',
        #'decoder' : '_blocks.14',
        'decoder' : '_avg_pooling',
    },
    'mobilenet_v2' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'features.14',
        'decoder' : 'features.14'
    },
    'mobilenet_v3_large' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'features.13',
        'decoder' : 'features.13'
    },
    'resnet18' : {
        'V1' : None,
        'V2' : None,
        'V4' : None,
        'IT' : 'layer4.1',
        'decoder' : 'layer4.1'
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
        'decoder' : 'IT'
    }
}
>>>>>>> 8467cdc282eb1293f14be2d19e1ded1dcacf2601
