import os
import yaml
import json
import argparse
from tqdm import tqdm

from dataset import Dataset_Tooth
from base.projector import Projector
from base.saver import Saver, PATH_DICT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='project')
    parser.add_argument('-n', '--name', type=str, default=None)
    args = parser.parse_args()

    root_dir = './'
    processed_dir = os.path.join(root_dir, 'processed/')
    config_path = os.path.join(root_dir, 'config.yaml')

    saver = Saver(
        root_dir=processed_dir, 
        path_dict=PATH_DICT
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # TODO: check
    # 1. config.projector.nVoxel == config.dataset.resolution
    # 2. config.projector.dVoxel == config.dataset.spacing

    dataset = Dataset_Tooth(
        root_dir=root_dir, 
        config=config['dataset']
    ).init_projector(Projector(config=config['projector']))

    if args.name is not None:
        # NOTE:
        # bugs (?) of tigre 2.2.0 (and older):
        # the processing will become slower and slower when calling tigre.Ax multiple times
        dataset.filter_names([args.name])
        saver.save(dataset[0])
    else:
        for data in tqdm(dataset, ncols=50):
            saver.save(data)

    # split data and save meta info (in json)
    info = {}
    info['dataset_config'] = config_path
    info.update(saver.path_dict)

    with open(os.path.join(root_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
        info.update(splits)

    with open(os.path.join(root_dir, 'meta_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    