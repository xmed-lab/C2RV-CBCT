# C^2RV-CBCT
Yiqun Lin, Jiewen Yang, Hualiang Wang, Xinpeng Ding, Wei Zhao, and Xiaomeng Li. "C^2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction." CVPR 2024. [arXiv](https://arxiv.org/abs/2406.03902)

```
@InProceedings{lin2024c2rv,
    author    = {Lin, Yiqun and Yang, Jiewen and Wang, Hualiang and Ding, Xinpeng and Zhao, Wei and Li, Xiaomeng},
    title     = {C{\textasciicircum}2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11205-11214}
}
```

## 1. Installation

```shell
torch 1.13, cuda 11.6
fvcore, SimpleITK, easydict, scikit-image, cython, matplotlib, scipy

# TIGRE: https://github.com/CERN/TIGRE/blob/master/Frontispiece/python_installation.md
git clone https://github.com/CERN/TIGRE
cd Python
python setup.py install
```
