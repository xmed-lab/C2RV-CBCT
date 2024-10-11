import numpy as np
import SimpleITK as sitk


def sitk_load(path, uint8=False, spacing_unit='mm'):
    """
    Load a medical image using SimpleITK.
    
    Args:
        path (str): Path to the image file.
        uint8 (bool): If True, scales image to [0, 1] assuming it's uint8.
        spacing_unit (str): Unit of spacing, 'mm' or 'm'.
    
    Returns:
        tuple: (image array [x, y, z], spacing array [x, y, z])
    """
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    
    # Convert spacing to the desired unit
    if spacing_unit == 'm':
        spacing *= 1000.0
    elif spacing_unit != 'mm':
        raise ValueError("Invalid spacing unit. Must be 'mm' or 'm'.")
    
    # Get the image array and transpose to [x, y, z]
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0)
    image = image.astype(np.float32)
    
    if uint8:
        # If the image is uint8, normalize to [0, 1]
        image /= 255.0
    
    return image, spacing


def sitk_save(path, image, spacing=None, uint8=False):
    """
    Save an image array as a medical image file using SimpleITK.
    
    Args:
        path (str): Path to save the image file.
        image (np.ndarray): Image array [x, y, z].
        spacing (np.ndarray or None): Spacing array [x, y, z] in mm.
        uint8 (bool): If True, scales image to [0, 255] and saves as uint8.
    """
    image = image.astype(np.float32)
    # Transpose image back to [z, y, x] for SimpleITK
    image = image.transpose(2, 1, 0)
    
    if uint8:
        # Scale image to [0, 255] and convert to uint8
        image = (image * 255).astype(np.uint8)
    
    out = sitk.GetImageFromArray(image)
    
    if spacing is not None:
        # Set the image spacing (in mm)
        out.SetSpacing(spacing.astype(np.float64))
    
    sitk.WriteImage(out, path)


def check_range(dataset):
    from tqdm import tqdm
    
    # NOTE: set dataset.return_raw(True) to load raw data
    # just to check the range of the image size
    size_list = []
    for item in tqdm(dataset, ncols=50):
        image = item['image']
        spacing = item['spacing']
        shape = np.array(image.shape)
        image_size = shape * spacing
        size_list.append(image_size)
    
    size_list = np.stack(size_list, axis=0) # [N, 3]
    size_min = np.min(size_list, axis=0)
    size_max = np.max(size_list, axis=0)
    # TODO: histgram
    return size_min, size_max
#cloner174