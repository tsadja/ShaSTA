import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, save_path, version, nsweeps=10, filter_zero=True, virtual=False):
    nu_ds.create_nuscenes_infos(root_path, save_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    

if __name__ == "__main__":
    fire.Fire()
