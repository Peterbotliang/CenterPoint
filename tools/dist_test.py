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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
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
    # parser.add_argument("--testset", action="store_true")
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

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

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
    # print(type(dataset._nusc_infos))
    # print(dataset._nusc_infos[0])

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

    if not os.path.exists(args.voxel_dir):
        os.mkdir(args.voxel_dir)

    for i, data_batch in enumerate(data_loader):
        # if i > 0:
        #     break
        # print(data_batch.keys())
        # print(type(data_batch['points']))
        # print('metadata:', data_batch['metadata'])
        # print('shape', data_batch['shape'])
        # print('points shape:', data_batch['points'].shape)
        # print('voxels shape:', data_batch['voxels'].shape)
        # print('num_points shape:', data_batch['num_points'].shape)
        # print(data_batch['num_points'])
        # print(data_batch['coordinates'])
        # print(data_batch['coordinates'].shape)
        # print(data_batch['points'][0 , :])
        # print(torch.min(data_batch['points'], dim = 0)[0], torch.max(data_batch['points'], dim = 0)[0])
        # print(torch.min(torch.sum(data_batch['voxels'], dim = 1) / data_batch['num_points'][:, None], dim = 0)[0], 
        #       torch.max(torch.sum(data_batch['voxels'], dim = 1) / data_batch['num_points'][:, None], dim = 0)[0])
        # print(data_batch['voxels'][0, :, :])
        # print(data_batch['voxels'][-1, :, :])
        # print('0', torch.sum(data_batch['coordinates'][:, 0] == 0))
        # print('1', torch.sum(data_batch['coordinates'][:, 0] == 1))
        # print('2', torch.sum(data_batch['coordinates'][:, 0] == 2))
        # print('3', torch.sum(data_batch['coordinates'][:, 0] == 3))
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        # mask = data_batch['coordinates'][:, 0] == 0
        # data_batch['voxels'] = data_batch['voxels'][mask, :, :]
        # data_batch['num_points'] = data_batch['num_points'][mask]
        # data_batch['coordinates'] = data_batch['coordinates'][mask, :]
        # data_batch['num_voxels'] = data_batch['num_voxels'][[0], ...]
        # print('len num_voxels', len(data_batch['num_voxels']))
        # print('len num_voxels', len(data_batch['num_voxels'].to('cuda')))
        

        # with torch.no_grad():
        #     outputs = batch_processor(
        #         model, data_batch, train_mode=False, local_rank=args.local_rank,
        #     )

            # data = dict(
            #     features=data_batch['voxels'].to('cuda'),
            #     num_voxels=data_batch['num_points'].to('cuda'),
            #     coors=data_batch['coordinates'].to('cuda'),
            #     batch_size=len(data_batch['num_voxels']),
            #     input_shape=data_batch["shape"][0],
            # )
            # x, multi_feat = model.extract_feat(data)

        # data = dict(
        #     features=torch.sum(data_batch['voxels'], dim = 1, keepdim = True) / data_batch['num_points'][:, None, None],
        #     num_voxels=torch.ones_like(data_batch['num_points']),
        #     coors=data_batch['coordinates'],
        #     batch_size=len(data_batch['num_voxels']),
        #     input_shape=data_batch["shape"][0],
        # )
        # print(data['features'].shape)
        # print(data['num_voxels'].shape)
        # print(data['coors'].shape)
        # print(data['coors'])
        # print(data['batch_size'])
        # print(data['input_shape'])

        mask = data_batch['coordinates'][:, 0] == 0
        data = dict(
            features=torch.sum(data_batch['voxels'][mask, :, :], dim = 1, keepdim = True) / data_batch['num_points'][mask, None, None],
            num_voxels=torch.ones_like(data_batch['num_points'][mask]),
            coors=data_batch['coordinates'][mask, :],
            batch_size=len(data_batch['num_voxels']),
            input_shape=data_batch["shape"][0],
        )
        # print(data['features'].shape)
        # print(data['num_voxels'].shape)
        # print(data['coors'].shape)
        # print(data['coors'])
        # print(data['batch_size'])
        # print(data['input_shape'])


        model = model.cuda()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )

        torch.save(data, os.path.join(args.voxel_dir, data_batch['metadata'][0]['token'] + '.pt'))

        # print(model.with_neck)
        # print(outputs[0].keys())
        # print(outputs[0]['metadata'])
        # print(outputs[0]['box3d_lidar'].shape)
        # print(outputs[0]['box3d_lidar'][: 10, :])
        # print(torch.min(outputs[0]['box3d_lidar'][:, :], dim = 0)[0], torch.max(outputs[0]['box3d_lidar'][:, :], dim = 0)[0])
        

        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in ["metadata", "shape"]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )

        if args.local_rank == 0:
            prog_bar.update()

    
    for k in detections.keys():
        detections[k].pop('backbone_feat', None)

    torch.save(detections, os.path.join(args.work_dir, args.output_path))

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)

    # result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    # if result_dict is not None:
    #     for k, v in result_dict["results"].items():
    #         print(f"Evaluation {k}: {v}")

    # if args.txt_result:
    #     assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
