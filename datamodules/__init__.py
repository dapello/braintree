from .imagenet_datamodule import ImagenetDataModule
from .neural_datamodule import NeuralDataModule
from .neural_datamodule import StimuliDataModule

DATAMODULES = {
    'ImageNet' : ImagenetDataModule,
    'NeuralData' : NeuralDataModule,
    'StimuliClassification' : StimuliDataModule
}
