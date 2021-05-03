from cornet import *

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
        'IT' : 'V4',
        'decoder' : 'decoder.avgpool'
    }
}
