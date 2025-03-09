import os
import pickle
import numpy as np
from copy import deepcopy

from base.utils import sitk_save
from base.projector import visualize_projections



PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy'
}


class Saver:
    def __init__(self, root_dir, path_dict):
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(root_dir, self._path_dict[key])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._path_dict[key] = path

        self._projs_list = []
        self._projs_max = 0.

        self._is_blocks_coords_saved = False

    @property
    def path_dict(self):
        return self._path_dict

    def _save_CT(self, data):
        name = data['name']
        sitk_save(
            self._path_dict['image'].format(name), 
            image=data['image'],
            spacing=data['spacing'],
            uint8=True
        ) # dtype: uint8

    def _save_blocks(self, data):
        if not self._is_blocks_coords_saved:
            np.save(self._path_dict['blocks_coords'], data['blocks_coords']) # dtype: float32
            self._is_block_coords_saved = True

        for i, block in enumerate(data['blocks_vals']):
            save_path = self._path_dict['blocks_vals'].format(data['name'], i)
            block = (block * 255).astype(np.uint8)
            np.save(save_path, block) # dtype: uint8

    def _save_projs(self, data, projs_vis=True):
        projs = data['projs']
        projs_max = projs.max()
        # projs_max = np.ceil(projs_max * 100) / 100

        projs = (projs / projs_max * 255).astype(np.uint8)
        with open(self._path_dict['projs'].format(data['name']), 'wb') as f:
            pickle.dump({
                'projs': projs,          # uint8
                'projs_max': projs_max,  # float
                'angles': data['angles'] # float
            }, f, pickle.HIGHEST_PROTOCOL)

        if projs_vis:
            visualize_projections(
                self._path_dict['projs_vis'].format(data['name']), 
                data['projs'], data['angles']
            )

    def save(self, data):
        self._save_CT(data)
        self._save_blocks(data)
        self._save_projs(data)
