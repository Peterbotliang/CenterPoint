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



def create_model(config_path, checkpoint_path=None):

    cfg = Config.fromfile(config_path)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if checkpoint_path is not None:
        _ = load_checkpoint(model, checkpoint_path, map_location="cpu")

    return model

if __name__ == "__main__":
    main()
