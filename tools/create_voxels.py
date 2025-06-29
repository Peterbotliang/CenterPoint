import argparse
import copy
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
path_to_c = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this path to the beginning of Python's search paths
if path_to_c not in sys.path:
    sys.path.insert(0, path_to_c)

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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--output_path", type=str, default='output.pt')
    parser.add_argument('--voxel_dir', type=str, default='./voxels')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    print(cfg.class_names)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

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

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    print('len(dataset)', len(dataset))
    print('batch_size', data_loader.batch_size)

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if not os.path.exists(args.voxel_dir):
        os.mkdir(args.voxel_dir)

    for i, data_batch in enumerate(data_loader):
        # if i > 0:
        #     break

        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        data = dict(
            features=torch.sum(data_batch['voxels'], dim = 1, keepdim = True) / data_batch['num_points'][:, None, None],
            num_voxels=torch.ones_like(data_batch['num_points']),
            coors=data_batch['coordinates'],
            batch_size=len(data_batch['num_voxels']),
            input_shape=data_batch["shape"][0],
        )

        torch.save(data, os.path.join(args.voxel_dir, data_batch['metadata'][0]['token'] + '.pt'))



if __name__ == "__main__":
    main()
