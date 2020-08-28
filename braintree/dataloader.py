import os

import numpy as np
import torch as ch
import h5py as h5

import torchvision
from torch.utils.data import Dataset, DataLoader

def data_as_dict(datapath, partition):
    files = np.sort([f for f in os.listdir(datapath) if partition+'.npy' in f])
    keys = [f.split('_')[0] for f in files]
    return {
        key:ch.Tensor(np.load(os.path.join(datapath,file)))
        for (key,file) in zip(keys, files) 
    }

def get_neural_dataset(datapath, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = torchvision.transforms.Normalize(
        mean=mean,
        std=std
    )   

    train = data_as_dict(datapath, 'Train')
    test = data_as_dict(datapath, 'Test')
    
    return [
        CustomTensorDataset(dataset, transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            normalize
        ]))
        for dataset in [train, test]
    ]


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    takes and returns dictionaries of tensors like:
    {
        stimuli -> tensor,
        layerkey1 -> tensor,
        layerkey2 -> tensor,
        ...,
        labels -> tensor
    }

    __getitem__ operates with any index through modulo
    """
    def __init__(self, tensor_dict, transform=None, mean_fill=True):
        self.key0 = list(tensor_dict.keys())[0]
        assert all(
            tensor_dict[self.key0].size(0) == tensor_dict[key].size(0) 
            for key in tensor_dict
        )
        self.tensor_dict = tensor_dict
        self.transform = transform
        if mean_fill:
            print('filling means!')
            mask = ch.isnan(tensor_dict['region-IT'])
            self.tensor_dict['region-IT'][mask] = tensor_dict['region-IT'][mask==0].mean()


    def __getitem__(self, index):
        # modulo index by length of data, so that we can any index
        N = self.__len__()
        index = index%N
        return {
            key:(
                self.transform(self.tensor_dict[key][index]) 
                if key=='Stimuli' 
                else self.tensor_dict[key][index]
            )
            for key in self.tensor_dict
        }

    def __len__(self):
        return self.tensor_dict[self.key0].size(0)
    
# and concat the two:
class ConcatDataset(ch.utils.data.Dataset):
    """
    takes the length as the greater of the two datasets. assumes the lesser length
    dataset can operate with any index, ie CustomTensorDataset class above.
    """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
