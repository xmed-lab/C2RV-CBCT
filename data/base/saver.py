import os
import pickle
import numpy as np
from copy import deepcopy
from base.utils import sitk_save
from base.projector import visualize_projections


# Define default paths for saving data
PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy'
}



class Saver:
    """
    Class for saving processed data, including images, projections, and blocks.
    
    Attributes:
        _path_dict (dict): Dictionary mapping data types to file paths.
        _projs_list (list): List of projection data.
        _projs_max (float): Maximum projection value for normalization.
        _is_blocks_coords_saved (bool): Flag indicating if block coordinates have been saved.
    """
    
    def __init__(self, root_dir, path_dict):
        """
        Initialize the Saver with a root directory and path dictionary.
        
        Args:
            root_dir (str): Root directory where data will be saved.
            path_dict (dict): Dictionary mapping data types to relative file paths.
        """
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(root_dir, self._path_dict[key])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._path_dict[key] = path
        
        self._projs_list = []
        self._projs_max = 0.0
        self._is_blocks_coords_saved = False
    
    
    @property
    def path_dict(self):
        """
        Get the dictionary of data paths.
        
        Returns:
            dict: Dictionary of data paths.
        """
        
        return self._path_dict
    
    
    def _save_CT(self, data):
        """
        Save the processed CT image.
        
        Args:
            data (dict): Data dictionary containing 'name', 'image', and 'spacing'.
        """
        name = data['name']
        sitk_save(
            self._path_dict['image'].format(name),
            image=data['image'],
            spacing=data['spacing'],
            uint8=True  # Save as uint8 image
        )
    
    
    def _save_blocks(self, data):
        """
        Save the blocks of image data.
        
        Args:
            data (dict): Data dictionary containing 'name', 'blocks_vals', and 'blocks_coords'.
        """
        if not self._is_blocks_coords_saved:
            # Save block coordinates once
            np.save(self._path_dict['blocks_coords'], data['blocks_coords'])  # dtype: float32
            self._is_blocks_coords_saved = True
        
        for i, block in enumerate(data['blocks_vals']):
            save_path = self._path_dict['blocks_vals'].format(data['name'], i)
            block = (block * 255).astype(np.uint8)  # Scale block values to [0, 255]
            np.save(save_path, block)
    
    
    def _save_projs(self, data, projs_vis=True):
        """
        Save the projections and optionally generate a visualization.
        
        Args:
            data (dict): Data dictionary containing 'name', 'projs', and 'angles'.
            projs_vis (bool): If True, generate a visualization of the projections.
        """
        projs = data['projs']
        # Normalize projections to [0, 255]
        projs_max = projs.max()
        projs_max = np.ceil(projs_max * 100) / 100  # Round up to nearest 0.01
        
        projs_scaled = (projs / projs_max * 255).astype(np.uint8)
        with open(self._path_dict['projs'].format(data['name']), 'wb') as f:
            pickle.dump({
                'projs': projs_scaled,      # uint8 projections
                'projs_max': projs_max,     # Maximum value before scaling
                'angles': data['angles']    # Projection angles
            }, f, pickle.HIGHEST_PROTOCOL)
        
        if projs_vis:
            # Generate and save a visualization of the projections
            visualize_projections(
                self._path_dict['projs_vis'].format(data['name']),
                projs_scaled,
                data['angles']
            )
    
    
    def save(self, data):
        """
        Save all data components for a single data item.
        
        Args:
            data (dict): Data dictionary containing all necessary data.
        """
        self._save_CT(data)
        self._save_blocks(data)
        self._save_projs(data)
