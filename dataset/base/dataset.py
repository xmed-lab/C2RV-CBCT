import scipy
import numpy as np
from copy import deepcopy
from base.utils import sitk_load


class Dataset:
    
    """
    Base Dataset class for handling volumetric medical images (e.g., CT scans).
    
    This class provides methods for:
    - Data loading
    - Resampling to a consistent spacing
    - Cropping or padding to a consistent resolution
    - Normalizing voxel intensities
    - Splitting the volume into blocks
    - Applying a projector to simulate 2D projections
    
    Attributes:
        _data_list (list): List of data items, each containing 'name' and 'path' keys.
        _spacing (np.ndarray): Desired voxel spacing for resampling.
        _resolution (np.ndarray): Desired image resolution after cropping/padding.
        _value_range (np.ndarray): Value range for normalization (e.g., [-1000, 400] for CT HU values).
        _block_size (np.ndarray): Size of blocks for splitting the volume.
        _block_info (dict or None): Cached block information.
        _return_raw (bool): If True, returns raw data without processing.
        _projector (callable or None): Projector function to simulate 2D projections.
    """
    def __init__(self, config):
        """
        Initialize the Dataset with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing dataset parameters:
                - 'spacing' (list or array): Desired voxel spacing for resampling.
                - 'resolution' (list or array): Desired image resolution after cropping/padding.
                - 'value_range' (list or array): Value range for normalization (e.g., [-1000, 400] for CT HU values).
                - 'block_size' (list or array): Size of blocks for splitting the volume (optional).
        """
        self._data_list = []
        self._spacing = np.array(config['spacing'])
        self._resolution = np.array(config['resolution'])
        self._value_range = np.array(config['value_range'])
        self._block_size = np.array(config['block_size'])
        self._block_info = None
        self._return_raw = False
        self._projector = None
    
    
    def filter_names(self, names):
        """
        Filter the dataset to include only data items with names in the provided list.
        
        Args:
            names (list): List of names to include.
        
        Returns:
            Dataset: Returns self to allow method chaining.
        """
        new_list = []
        for item in self._data_list:
            if item['name'] in names:
                new_list.append(item)
        self._data_list = new_list
        
        return self
    
    
    def return_raw(self, flag):
        """
        Set whether to return raw data without processing.
        
        Args:
            flag (bool): If True, the dataset will return raw data.
        
        Returns:
            Dataset: Returns self to allow method chaining.
        """
        self._return_raw = flag
        
        return self
    
    
    def init_projector(self, projector):
        """
        Initialize a projector to simulate 2D projections for all data items.
        
        Args:
            projector (callable): A projector function that takes an image and returns projections.
        
        Returns:
            Dataset: Returns self to allow method chaining.
        """
        self._projector = projector
        
        return self
    
    
    def _generate_blocks(self):
        """
        Generate block coordinates and indices for splitting the volume into smaller blocks.
        
        Returns:
            dict: A dictionary containing block coordinates and lists.
        """
        if self._block_info is not None:
            return self._block_info
        
        nx, ny, nz = self._block_size
        # Ensure that the resolution is divisible by the block size
        assert (self._resolution % self._block_size).sum() == 0, \
            f'Resolution {self._resolution} is not divisible by block_size {self._block_size}'
        
        offsets = (self._resolution / self._block_size).astype(int)
        base = np.mgrid[:nx, :ny, :nz]  # Create a grid of indices [3, nx, ny, nz]
        base = base.reshape(3, -1).transpose(1, 0)  # Reshape to [*, 3]
        base = base * offsets  # Scale base indices by offsets
        
        block_list = []
        for x in range(offsets[0]):
            for y in range(offsets[1]):
                for z in range(offsets[2]):
                    # Shift blocks by the offset
                    block = base + np.array([x, y, z])
                    block_list.append(block)
        
        blocks_coords = np.stack(block_list, axis=0)  # [N, *, 3]
        # Normalize coordinates to [0, 1]
        blocks_coords = blocks_coords / (self._resolution - 1)
        blocks_coords = blocks_coords.astype(np.float32)
        
        self._block_info = {
            'coords': blocks_coords,
            'list': block_list
        }
        
        return self._block_info
    
    
    def _convert_blocks(self, data):
        """
        Split the image data into blocks and add block information to the data dictionary.
        
        Args:
            data (dict): Data dictionary containing 'image'.
        
        Returns:
            dict: Updated data dictionary with 'blocks_vals' and 'blocks_coords'.
        """
        block_info = self._generate_blocks()
        # Extract block values from the image
        blocks_vals = [
            data['image'][b[:, 0], b[:, 1], b[:, 2]]
            for b in block_info['list']
        ]
        data['blocks_vals'] = blocks_vals
        data['blocks_coords'] = block_info['coords']
        
        return data
    
    
    def _process(self, data):
        """
        Process the raw data through resampling, cropping/padding, and normalization.
        
        Args:
            data (dict): Raw data dictionary.
        
        Returns:
            dict: Processed data dictionary.
        """
        # data -> resample -> crop/pad -> normalize
        return self._normalize(
            self._crop_pad(
                self._resample(data)
            )
        )
    
    
    def _resample(self, data):
        """
        Resample the image data to the desired spacing.
        Args:
            data (dict): Data dictionary containing 'image' and 'spacing'.
        Returns:
            dict: Updated data dictionary with resampled 'image' and updated 'spacing'.
        """
        # Calculate zoom factors for each dimension
        zoom_factors = data['spacing'] / self._spacing
        # Resample image using cubic interpolation (order=3)
        data['image'] = scipy.ndimage.zoom(
            data['image'],
            zoom=zoom_factors,
            order=3,
            prefilter=False
        )
        # Update spacing to the desired spacing
        data['spacing'] = deepcopy(self._spacing)
        return data
    
    
    def _crop_pad(self, data):
        """
        Crop or pad the image data to the desired resolution.
        
        Args:
            data (dict): Data dictionary containing 'image'.
        
        Returns:
            dict: Updated data dictionary with cropped/padded 'image'.
        """
        processed = []  # Indices for the processed (output) array
        original = []   # Indices for the original (input) array
        shape = data['image'].shape
        for i in range(3):
            if shape[i] >= self._resolution[i]:
                # Center crop
                processed.append({
                    'left': 0,
                    'right': self._resolution[i]
                })
                offset = (shape[i] - self._resolution[i]) // 2
                original.append({
                    'left': offset,
                    'right': offset + self._resolution[i]
                })
            else:
                # Padding
                offset = (self._resolution[i] - shape[i]) // 2
                processed.append({
                    'left': offset,
                    'right': offset + shape[i]
                })
                original.append({
                    'left': 0,
                    'right': shape[i]
                })
        
        def slice_array(a, index_a, b, index_b):
            """
            Helper function to copy data from array b to array a using provided indices.
            
            Args:
                a (np.ndarray): Destination array.
                index_a (list): Indices for the destination array.
                b (np.ndarray): Source array.
                index_b (list): Indices for the source array.
            
            Returns:
                np.ndarray: Updated destination array.
            """
            a[
                index_a[0]['left']:index_a[0]['right'],
                index_a[1]['left']:index_a[1]['right'],
                index_a[2]['left']:index_a[2]['right']
            ] = b[
                index_b[0]['left']:index_b[0]['right'],
                index_b[1]['left']:index_b[1]['right'],
                index_b[2]['left']:index_b[2]['right']
            ]
            
            return a
        
        # Create a mask (not currently used) indicating the region of valid data
        data['mask'] = slice_array(
            np.zeros(self._resolution),
            processed,
            np.ones_like(data['image']),
            original
        )
        # Initialize an array filled with the minimum value (e.g., -1000 HU for air)
        data['image'] = slice_array(
            np.full(self._resolution, fill_value=self._value_range[0], dtype=np.float32),
            processed,
            data['image'],
            original
        )
        
        return data
    
    
    def _normalize(self, data):
        """
        Normalize the image data to the [0, 1] range based on the specified value range.
        
        Args:
            data (dict): Data dictionary containing 'image'.
        
        Returns:
            dict: Updated data dictionary with normalized 'image'.
        """
        min_value, max_value = self._value_range
        image = data['image']
        # Clip values to the specified range
        image = np.clip(image, a_min=min_value, a_max=max_value)
        # Normalize to [0, 1]
        image = (image - min_value) / (max_value - min_value)
        data['image'] = image
        
        return data
    
    
    def _load_raw(self, data):
        """
        Load raw image data using SimpleITK.
        
        Args:
            data (dict): Data item containing 'name' and 'path'.
        
        Returns:
            dict: Data dictionary with loaded 'image' and 'spacing'.
        """
        image, spacing = sitk_load(data['path'])
        
        return {
            'name': data['name'],
            'image': image,
            'spacing': spacing
        }
    
    
    def __len__(self):
        """
        Get the number of data items in the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self._data_list)
    
    
    def __getitem__(self, index):
        """
        Get a data item by index.
        
        Args:
            index (int): Index of the data item.
        
        Returns:
            dict: Data dictionary containing processed image and additional information.
        """
        item = self._data_list[index]
        data = self._load_raw(item)
        if not self._return_raw:
            data = self._process(data)
            data = self._convert_blocks(data)
            if self._projector is not None:
                # Apply the projector to simulate 2D projections
                projs = self._projector(data['image'])
                data.update(projs)  # Add 'projs' and 'angles' to data
        
        return data
    
#cloner174