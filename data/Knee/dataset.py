import os

from base.dataset import Dataset
from base.utils import sitk_load



class Dataset_Knee(Dataset):
    def __init__(self, root_dir, config):
        super().__init__(config)

        # root_dir: ./datasets/Knee_zhao/
        with open(os.path.join(root_dir, 'raw/names.txt'), 'r') as f:
            names = f.read().splitlines()

        self._data_list = []
        for name in names:
            tag, obj_id = name.split('-')
            self._data_list.append({
                'name': name,
                'path': os.path.join(root_dir, 'raw', tag, f'{obj_id}.mhd')
            })
    
    def _load_raw(self, data):
        image, spacing = sitk_load(data['path'], spacing_unit='m')
        return {
            'name': data['name'],
            'image': image,
            'spacing': spacing
        }
