from cornet import *

def vonecornet_s(pretrained=True):
    from vonenet import get_model
    return get_model(model_arch='cornets', pretrained=pretrained, map_location='cpu')

def efficientnet_b0(pretrained=True):
    from efficientnet_pytorch import EfficientNet
    assert pretrained == True
    return EfficientNet.from_pretrained('efficientnet-b0')
