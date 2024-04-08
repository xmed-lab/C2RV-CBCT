import numpy as np
from torch.utils.data import Dataset



class Mixed_CBCT_dataset(Dataset):
    def __init__(self, dst_list, DST_CLASS, **kwargs) -> None:
        super().__init__()
        
        print('mixed_dataset:', dst_list)
        self.name_list = dst_list
        self.datasets = []
        for dst_name in self.name_list:
            self.datasets.append(DST_CLASS(dst_name=dst_name, **kwargs))
        
        self.is_train = self.datasets[0].is_train
    
    def __len__(self):
        dst_len = [len(d) for d in self.datasets]
        return np.sum(dst_len)
    
    @property
    def num_dst(self):
        return len(self.datasets)
    
    def find_dst(self, index):
        for i, dst in enumerate(self.datasets):
            if index >= len(dst):
                index -= len(dst)
            else:
                return i, index

    def __getitem__(self, index):
        dst_idx, index = self.find_dst(index)
        return self.datasets[dst_idx][index]
