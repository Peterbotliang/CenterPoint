import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import json
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.splits import create_splits_scenes

def plot_measurements(ax, measurements, xlim = None, ylim = None, color=None):
    num_meas, dim_meas = measurements.shape
    for ind in range(num_meas):
        t = mpl.transforms.Affine2D().rotate_around(measurements[ind, 0], measurements[ind, 1], measurements[ind, 6]) + ax.transData

        rect = mpl.patches.Rectangle((measurements[ind, 0] - measurements[ind, 3] / 2,
                                        measurements[ind, 1] - measurements[ind, 4] / 2),
                                        measurements[ind, 3],
                                        measurements[ind, 4],
                                        edgecolor = color,
                                        facecolor = 'none',
                                        transform = t)

        ax.add_patch(rect)

    if xlim is None:
        ax.set(xlim=(np.min(measurements[:, 0]) - 10, np.max(measurements[:, 0]) + 10))
    else:
        ax.set(xlim=xlim)

    if ylim is None:
        ax.set(ylim=(np.min(measurements[:, 1]) - 10, np.max(measurements[:, 1]) + 10))
    else:
        ax.set(ylim=ylim)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert nuscenes tracking submission to  array formats')
    parser.add_argument('--result_path',
                        type = str,
                        help = 'The .pt file that stores detections and features')
    parser.add_argument('--dataroot', type=str, default='./data/nuScenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--output_dir', type=str, default='./centerPoint_feat',
                        help='Folder to store converted arrays.')

    args = parser.parse_args()
    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version

    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)

    detections = torch.load(result_path_)

    nusc = NuScenes(version=version_, dataroot=dataroot_, verbose=False)

    detections_by_scene = dict()

    tokens_traversed = set()
    for sample_token in tqdm(detections.keys()):
        if sample_token in tokens_traversed:
            continue

        # tokens_traversed.add(sample_token)
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        scene_record = nusc.get('scene', scene_token)
        scene = scene_record['name']
        detections_by_scene[scene] = []

        scene_first_token = scene_record['first_sample_token']
        sample_token = scene_first_token

        while sample_token != '':
            tokens_traversed.add(sample_token)
            sample_record = nusc.get('sample', sample_token)
            detection = detections[sample_token]
            detection['token'] = sample_token
            detections_by_scene[scene].append(detection)
            sample_token = sample_record['next']

    print(detections_by_scene.keys())

    for scene, detection in detections_by_scene.items():
        detection_feat = [x['backbone_feat'] for x in detection]
        torch.save(detection_feat, os.path.join(output_dir_, scene + '.pt'))
        torch.save(detection, os.path.join(output_dir_, scene + '_all.pt'))
