from cornet import *
from vonenet import *

def efficientnet_b0(pretrained=True):
    from efficientnet_pytorch import EfficientNet
    assert pretrained == True
    return EfficientNet.from_pretrained('efficientnet-b0')
