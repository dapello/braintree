from .imagenet_datamodule import ImagenetDataModule
from .neural_datamodule import NeuralDataModule

DATAMODULES = {
    'ImageNet' : ImagenetDataModule,
    'NeuralData' : NeuralDataModule
}
