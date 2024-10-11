## LUNA16 (Chest CT)

### 1. Download

Download data (subset0-9, annotations.csv) from https://luna16.grand-challenge.org/Download/ to `./data/LUNA16/raw/`. Files required for data preprocessing are organized as follows:

```
├── ./data/LUNA16/
│   ├── raw/
│   │   ├── subset0/
│   │   ├── ...
│   │   ├── subset9/
│   │   ├── annotations.csv [optional]
│   │   └── names.txt
│   ├── splits.json
```

Note that it is not necessary to download the annotations (`annotations.csv`), which are not used in model training and evaluation. Annotations are used to help locate nodules in the CT for radiologists to evaluate the reconstruction quality in our future study. Totally, there are 888 chest CT scans with the size (unit: mm) ranging from [236, 236, 165] to [500, 500, 416].

### 2. Data Preprocessing

Files (scripts and config) for data preprocessing:

```
├── ./data/LUNA16/
│   ├── dataset.py
│   ├── main.py
│   ├── run.sh
│   ├── config.yaml
```

Run the following commands:

```
export PYTHONPATH=$(pwd)/data/:$PYTHONPATH
cd ./data/LUNA16/
bash run.sh
```

The processed data are organized as:

```
├── ./data/LUNA16/
│   ├── processed/
│   │   ├── blocks/
│   │   ├── images/
│   │   ├── nodule_masks/ [optional]
│   │   ├── projections/
│   │   └── projections_vis/ [optional]
│   ├── meta_info.json
```
