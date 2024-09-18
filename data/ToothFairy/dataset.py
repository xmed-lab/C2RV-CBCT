import os
import numpy as np

from base.dataset import Dataset



class Dataset_Tooth(Dataset):
    def __init__(self, root_dir, config):
        super().__init__(config)

        names = os.listdir(os.path.join(root_dir, 'raw'))

        self._data_list = []
        for name in names:
            self._data_list.append({
                'name': name,
                'path': os.path.join(root_dir, f'raw/{name}/data.npy')
            })
    
    def _load_raw(self, data):
        image = np.load(data['path'])
        image = image.transpose(1, 2, 0)
        spacing = np.array([0.3, 0.3, 0.3])
        return {
            'name': data['name'],
            'image': image,
            'spacing': spacing
        }
