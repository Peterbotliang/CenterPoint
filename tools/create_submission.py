import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--prediction_dir", required=True, 
                        default = '/content/drive/MyDrive/nuScenes/', 
                        help="the dir that store prediction of 10 blobs")
    parser.add_argument("--split", type=str, default='val')

    args = parser.parse_args()

    return args

def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    print(cfg.class_names)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.split == 'test':
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    elif args.split == 'val':
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)
    elif args.split == 'train':
        print("Use Train Set")
        dataset = build_dataset(cfg.data.train)
    else:
        raise NotImplementedError

    predictions_all = {}
    for i in range(1, 11):
        intstr = str(i).zfill(2)
        prediction_path = os.path.join(args.prediction_dir, 'prediction_blob' + intstr + '_' + args.split + '.pkl')
        if os.path.exists(prediction_path):
            print(i)
            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)
            predictions_all.update(predictions)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions_all), 
                                        output_dir=args.work_dir, 
                                        testset=False,
                                        save_only = True)

    # if result_dict is not None:
    #     for k, v in result_dict["results"].items():
    #         print(f"Evaluation {k}: {v}")

    # if args.txt_result:
    #     assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
