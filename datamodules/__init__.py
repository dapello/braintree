from .imagenet_datamodule import ImagenetDataModule
from .neural_datamodule import NeuralDataModule, NeuralDataModule2
from .neural_datamodule import StimuliDataModule

DATAMODULES = {
    'ImageNet' : ImagenetDataModule,
    'NeuralData' : NeuralDataModule2,
    'StimuliClassification' : StimuliDataModule
}
