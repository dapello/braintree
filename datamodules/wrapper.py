from torch.utils.data import Dataset

class Wrapper(Dataset):
    """
    takes the length as the greater of the two datasets. assumes the lesser length
    dataset can operate with any index, ie CustomTensorDataset class above.
    """
    def __init__(self, *datasets):
        assert all([hasattr(d, 'name') for d in datasets])
        self.datasets = datasets
        self.names = [d.name for d in datasets]

    def __getitem__(self, i):
        #return tuple(d[i] for d in self.datasets)
        return {d.name : d[i] for d in self.datasets}

    def __len__(self):
        # return max length. requires input datasets to use modulo __getitem__
        return max(len(d) for d in self.datasets)
