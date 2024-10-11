import os
import sys
import csv
import numpy as np
from glob import glob
import SimpleITK as sitk
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base.dataset import Dataset



def sitk_load_luna16(path):
    # load as float32
    itk_img = sitk.ReadImage(path)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32) # to locate tumors in the CT; [0, 0, 0] if not provided
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    
    return image, spacing, origin



class Dataset_LUNA16(Dataset):
    """
    Dataset class for handling the LUNA16 dataset.

    This class extends the base Dataset class and adds functionality specific to LUNA16, such as loading nodule annotations.
    """
    
    def __init__(self, root_dir, config, data_dir = 'raw'):
        super().__init__(config)
        """
        Initialize the LUNA16 dataset.
        
        Args:
            root_dir (str): Root directory of the LUNA16 dataset.
            config (dict): Configuration dictionary containing dataset parameters.
        """
        # root_dir: ./datasets/LUNA16/
        self._data_list = []
        for sub_id in range(10):
            tag = f'subset{sub_id}'
            for path in glob(os.path.join(root_dir, data_dir, tag, '*.mhd')):
                name = path.split('/')[-1].replace('.mhd', '')
                name = f'{tag}_{name}'
                self._data_list.append({
                    'name': name,
                    'path': path,
                    'nodules': []
                })
        
        with open(os.path.join(root_dir, f'{data_dir}/annotations.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                for k in range(len(self._data_list)):
                    if row[0] == self._data_list[k]['name'].split('_')[-1]:
                        self._data_list[k]['nodules'].append([eval(x) for x in row[1:]])
        
        self._return_nodule_mask = False
    
    def return_nodule_mask(self, flag):
        self._return_nodule_mask = flag
        return self
    
    def _make_mask(self, data):
        # to extract [instance] segmentation mask from the annotation (for LUNA16 only)
        # data can be raw or processed
        spacing = data['spacing']
        mask = np.zeros_like(data['image'], dtype=int)
        for i, nodule in enumerate(data['nodules']):
            center = np.array(mask.shape) / 2 + nodule[:3] / spacing
            x, y, z = np.round(center).astype(int)
            
            r = nodule[-1] / 2
            r += 1.5 # additional 1.5 mm to show the tumor more obviously
            r = np.ceil(r / spacing).astype(int)
            
            # TODO: should further check if the nodule falls outside the (processed) image
            mask[
                x - r[0]: x + r[0],
                y - r[1]: y + r[1],
                z - r[2]: z + r[2]
            ] = i + 1 # or (= 1) if semantic segmentation
        
        # NOTE: the key 'mask' is defined in _crop_pad
        # data['nodule_mask'] = mask 
        
        return mask
    
    def _load_raw(self, data):
        image, spacing, origin = sitk_load_luna16(data['path']) # additionally load origin
        nodules = deepcopy(data['nodules'])
        for i in range(len(nodules)):
            nodule = np.array(nodules[i])
            nodule[:3] = nodule[:3] - origin - np.array(image.shape) / 2 * spacing
            nodules[i] = nodule
        
        return {
            'name': data['name'],
            'image': image,
            'spacing': spacing,
            'nodules': nodules
        }
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self._return_nodule_mask:
            mask = self._make_mask(data)
            data['nodule_mask'] = mask
        
        return data
    
#cloner174