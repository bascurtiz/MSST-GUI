# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'


import torch

# Compiled layout: the bundled runtime may not put this script's folder on
# sys.path (inference.py carries the same preamble). Make utils/ importable
# regardless of how the process is launched.
import os as _entry_os
import sys as _entry_sys
_entry_dir = _entry_os.path.dirname(_entry_os.path.abspath(__file__))
if _entry_dir not in _entry_sys.path:
    _entry_sys.path.insert(0, _entry_dir)

from utils.model_utils import load_start_checkpoint, ensure_readable_checkpoint
from utils.settings import get_model_from_config, parse_args_valid, initialize_environment_ddp
import warnings
import torch.multiprocessing as mp
from valid import valid_multi_gpu
import torch.distributed as dist

warnings.filterwarnings("ignore")


def check_validation_single(rank: int, world_size: int, args=None):
    args = parse_args_valid(args)

    initialize_environment_ddp(rank, world_size)
    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        checkpoint = torch.load(
            ensure_readable_checkpoint(args.start_check_point),
            weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='valid')

    if dist.get_rank() == 0:
        print(f"Instruments: {config.training.instruments}")

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    valid_multi_gpu(model, args, config, args.device_ids, verbose=False)


def check_validation(args=None):
    world_size = torch.cuda.device_count()
    mp.spawn(check_validation_single, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    check_validation()
