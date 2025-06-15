import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader


# # pandas dataset iterator
# class PandasDatasetIterator(IterableDataset):
#     def __init__(self, file_path):
#         self.dataset = pd.read_csv(file_path, sep='\t', iterator=True, chunksize=1)

#     def __iter__(self):
#         for data in self.dataset:
#             yield torch.from_numpy(data.values)

class PandasDatasetIterator(IterableDataset):
    def __init__(self, file_path, process=None):
        self.dataset = pd.read_csv(file_path, sep='\t', iterator=True, chunksize=1)  # , chunksize=1
        self.process = process

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for data in self.dataset:
                yield torch.from_numpy(data.values)
        else:
            # worker_info.id is the index of this worker
            # worker_info.num_workers is the total number of workers
            for i, data in enumerate(self.dataset):
                if i % worker_info.num_workers == worker_info.id:
                    # yield (data.index[0], worker_info.id)
                    if self.process is not None:
                        yield self.process(data)
                    else: 
                        yield data.values
                    

# torch dataLoader
class DatasetLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
class DatasetLoader_X(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]
    
    
class DatasetLoader_w(Dataset):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]
    

class DatasetLoader_www(Dataset):
    def __init__(self, x, y, w1, w2, w3):
        self.x = x
        self.y = y
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w1[idx], self.w2[idx], self.w3[idx]
    
    
if __name__ == '__main__':
    FILEPATH = '../../data/test_dataset_0124_sample.txt'
    dataset = PandasDatasetIterator(FILEPATH)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=12)
    for i, f in enumerate(dataloader):
        print(f)
        if i == 2:
            break
