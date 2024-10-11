# The LUNA16

## 1. Installation

```shell
#torch 1.13, cuda 11.6
pip install -r requirements.txt

# TIGRE: Needs More try to be installed!

git clone https://github.com/CERN/TIGRE.git

cd TIGRE/Python

python setup.py install
```

---

## **Overview**

To train a model on the LUNA16 dataset, here are main steps:

1. **Prepare the Dataset**:
   - Process the LUNA16 dataset using existing code.
   - Organize the processed data for use with PyTorch.

2. **Set Up the Environment**:
   - Install necessary libraries.
   - Clone the Denoising-Diffusion-GAN repository.
    ```
    git clone https://github.com/cloner174/Denoising-Diffusion-GAN.git
    ```

3. **Train the Model**:
   - Run the training script and monitor the training process.

---

## **Step 1: Prepare the Dataset**

### **a. Process the LUNA16 Dataset**

After processing the LUNA16 dataset using this repo, ensure the following:

- **Processed Images**: The CT images are saved in `.nii.gz` format in a directory, e.g., `data/LUNA16/processed/images/`.
- **Nodule Masks**: The corresponding nodule masks are saved in `.nii.gz` format, e.g., `data/LUNA16/processed/nodule_masks/`.

### **b. Organize the Data**

Ensure your directory structure looks like this:

```
C2RV-CBCT
├─ data
│  ├─ LUNA16
│     ├─ processed
│        ├─ images
│        │  ├─ scan1.nii.gz
│        │  ├─ scan2.nii.gz
│        │  └─ ...
│        ├─ nodule_masks
│        │  ├─ scan1.nii.gz
│        │  ├─ scan2.nii.gz
│        │  └─ ...
│        └─ meta_info.json
```

---

## **Step 2: Set Up the Environment**

### **a. Clone the Denoising-Diffusion-GAN Repository**

```bash
git clone https://github.com/cloner174/Denoising-Diffusion-GAN.git
```

Navigate to the cloned repository:

```bash
cd Denoising-Diffusion-GAN
```

### **b. Install Required Libraries**

Install necessary Python packages:

```bash
pip install -r requirements.txt
```

---

## **Step 3: Create a PyTorch Dataset Class**

Create a new Python script named `luna16_dataset.py` in the project directory:

```python
import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class LUNA16Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.mask_names = sorted(os.listdir(masks_dir))
        self.transform = transform

        # Ensure images and masks are aligned
        assert len(self.image_names) == len(self.mask_names), "Number of images and masks do not match."

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)  # Shape: [z, y, x]

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)

        # Normalize image to [0, 1]
        min_value = -1000  # Hounsfield Unit for air
        max_value = 400    # Typical upper bound for lung tissue
        image_array = np.clip(image_array, min_value, max_value)
        image_array = (image_array - min_value) / (max_value - min_value)

        # Convert to torch tensors and add channel dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Shape: [1, z, y, x]
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)    # Shape: [1, z, y, x]

        # Apply transformations if any
        if self.transform:
            image_tensor = self.transform(image_tensor)
            mask_tensor = self.transform(mask_tensor)

        return image_tensor, mask_tensor
```

**Notes**:

- **Data Alignment**: Ensure that the images and masks are aligned correctly by matching their filenames.
- **Normalization**: Adjust the `min_value` and `max_value` based on dataset's intensity range.
- **Data Types**: Convert arrays to `float32` before converting to tensors.

---

## **Step 4: Modify the Model for 3D Data**

The Denoising-Diffusion-GAN model supports uses 3D convolutional layers. But, you'll need to adjust the model setting by command line arg or editing the config file.

---

## **Step 5: Training**

### **a. Define Training Parameters**

```python
num_epochs = 100
batch_size = 1  # Adjust based on your GPU!
learning_rate = 1e-4
# Paths to processed data
images_dir = 'data/LUNA16/processed/images/'
masks_dir = 'data/LUNA16/processed/nodule_masks/'

#dataset = LUNA16Dataset(images_dir=images_dir, masks_dir=masks_dir)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
```

- **Adjust Hyperparameters**: The weight factor for `loss_L1` and other hyperparameters may need tuning.

### **b. Save Model Checkpoints**

Add code to save model checkpoints periodically:

Run the train script always with --save-content

---

## **Step 6: Train the Model**

### **a. Run the Training Script**

Execute the training script!


### **b. Monitor Training**

- **Loss Values**: Keep an eye on the discriminator and generator losses.
- **GPU Usage**: Monitor GPU memory usage to prevent out-of-memory errors.
- **Adjust Parameters**: If the model isn't converging, consider adjusting learning rates, loss weights, or other hyperparameters.

---

## **Troubleshooting**

- **Memory Errors**: If you encounter `CUDA out of memory` errors, reduce the batch size or use patch-based training.

- **Model Not Converging**: Try different learning rates, optimizers, or loss function weights.

- **Data Loading Issues**: Double-check file paths and ensure data is correctly loaded.

- **Model Bugs**: Verify that all layers and operations in your model are compatible with 3D data.

---

## **Additional Resources**

- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

- **Medical Imaging Libraries**:
  - `torchio`: [https://torchio.readthedocs.io/](https://torchio.readthedocs.io/)
  - `MONAI`: [https://monai.io/](https://monai.io/)

- **Denoising Diffusion Models**:
  - Research papers and tutorials on diffusion models can provide insights into the training process.

```
Yiqun Lin, Jiewen Yang, Hualiang Wang, Xinpeng Ding, Wei Zhao, and Xiaomeng Li. "C^2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction." CVPR 2024. [arXiv](https://arxiv.org/abs/2406.03902)

@InProceedings{lin2024c2rv,
    author    = {Lin, Yiqun and Yang, Jiewen and Wang, Hualiang and Ding, Xinpeng and Zhao, Wei and Li, Xiaomeng},
    title     = {C{\textasciicircum}2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11205-11214}
}
```

---

## **Final Remarks**

Training complex models like Denoising-Diffusion-GANs on 3D medical imaging data is a challenging task. It requires careful consideration of data handling, model architecture, and training procedures.

Don't hesitate to seek help from the community if you encounter issues. Forums like Stack Overflow, PyTorch Forums, and specialized medical imaging communities can be valuable resources.

Good luck!