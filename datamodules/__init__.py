from .imagenet_datamodule import ImagenetDataModule
from .neural_datamodule import NeuralDataModule, ImageNetAndNeuralDataModule

DATAMODULES = {
    'ImageNet' : ImagenetDataModule,
    'NeuralData' : NeuralDataModule,
    'ImageNetAndNeuralData' : ImageNetAndNeuralDataModule
}
