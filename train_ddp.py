# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'

import torch
import torch.multiprocessing as mp

# Compiled layout: the bundled runtime may not put this script's folder on
# sys.path (inference.py carries the same preamble). Make utils/ importable
# regardless of how the process is launched.
import os as _entry_os
import sys as _entry_sys
_entry_dir = _entry_os.path.dirname(_entry_os.path.abspath(__file__))
if _entry_dir not in _entry_sys.path:
    _entry_sys.path.insert(0, _entry_dir)

from train import train_model
from utils.settings import  cleanup_ddp
import warnings

warnings.filterwarnings("ignore")


def train_model_single(rank: int, world_size: int, args=None):
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        world_size:
        rank:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """
    train_model(args, rank, world_size)  # Close DDP


def train_model_ddp(args=None):
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(train_model_single, args=(world_size, args), nprocs=world_size, join=True)
    except Exception as e:
        cleanup_ddp()
        raise e


if __name__ == "__main__":
    train_model_ddp()
