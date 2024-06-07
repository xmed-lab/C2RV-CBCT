# C^2RV-CBCT
Yiqun Lin, Jiewen Yang, Hualiang Wang, Xinpeng Ding, Wei Zhao, and Xiaomeng Li. "C^2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction." CVPR 2024. [arxiv](https://arxiv.org/abs/2406.03902)

```
@misc{lin2024c2rv,
      title={C^2RV: Cross-Regional and Cross-View Learning for Sparse-View CBCT Reconstruction}, 
      author={Yiqun Lin and Jiewen Yang and Hualiang Wang and Xinpeng Ding and Wei Zhao and Xiaomeng Li},
      year={2024},
      eprint={2406.03902},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
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
