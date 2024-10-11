import os
import sys
import yaml
import json
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import Dataset_LUNA16
from base.utils import sitk_save
from base.projector import Projector
from base.saver import Saver, PATH_DICT


class Saver_LUNA16(Saver):
    """
    Extended Saver class for LUNA16 dataset to include nodule masks.
    """
    
    def __init__(self, root_dir):
        """
        Initialize the Saver with paths specific to LUNA16 dataset.
        
        Args:
            root_dir (str): Root directory where data will be saved.
        """
        path_dict = PATH_DICT.copy()
        path_dict['nodule_mask'] = 'nodule_masks/{}.nii.gz'
        
        super().__init__(root_dir, path_dict)
    
    def _save_nodule_mask(self, data):
        """
        Save the nodule mask as a medical image file.
        
        Args:
            data (dict): Data dictionary containing 'name', 'nodule_mask', and 'spacing'.
        """
        sitk_save(
            self._path_dict['nodule_mask'].format(data['name']),
            data['nodule_mask'] / 255.0,  # Scale mask to [0, 1]
            data['spacing'],
            uint8=True  # Save as uint8 image
        )
    
    def save(self, data):
        """
        Save all data components, including the nodule mask.
        
        Args:
            data (dict): Data dictionary containing all necessary data.
        """
        super().save(data)
        
        self._save_nodule_mask(data)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process LUNA16 dataset')
    
    parser.add_argument('-n', '--name', type=str, default=None, help='Name of a specific patient to process')
    parser.add_argument('-r', '--root_dir', type=str, default='./', help='Root Directory')
    parser.add_argument('-d', '--data_dir', type=str, default='raw', help='Name of directory with subsets of Luna')
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    processed_dir = os.path.join(root_dir, 'processed/')
    config_path = os.path.join(root_dir, 'config.yaml')
    
    saver = Saver_LUNA16(root_dir=processed_dir)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure that projector and dataset configurations are consistent
    # 1. config.projector.nVoxel == config.dataset.resolution
    # 2. config.projector.dVoxel == config.dataset.spacing
    
    dataset = Dataset_LUNA16(
        root_dir=root_dir,
        config=config['dataset'],
        data_dir= args.data_dir
    ).return_nodule_mask(True).init_projector(
        Projector(config=config['projector'])
    )
    
    if args.name is not None:
        # Process a specific patient (useful for debugging)
        dataset.filter_names([args.name])
        saver.save(dataset[0])
    else:
        # Process all patients in the dataset
        for data in tqdm(dataset, ncols=50):
            saver.save(data)
    
    # Save meta information and data splits
    info = {}
    info['dataset_config'] = config_path
    info.update(saver.path_dict)
    
    with open(os.path.join(root_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
        info.update(splits)
    
    with open(os.path.join(root_dir, 'meta_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
#cloner174