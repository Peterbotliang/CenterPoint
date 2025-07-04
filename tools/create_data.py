import copy
from pathlib import Path
import pickle

import fire, os
# import os
import sys
sys.path.append(os.path.dirname(__file__))
path_to_c = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this path to the beginning of Python's search paths
if path_to_c not in sys.path:
    sys.path.insert(0, path_to_c)

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    if version == 'v1.0-trainval':
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    fire.Fire()
