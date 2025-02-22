# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later

from train import goTraining
from test import goTesting
from utils.config import json2args
import os
import torch


def print_mode(mode):
    print(f"Running in {mode} mode")


def validate_path(path):
    if not path or not os.path.exists(path):
        raise ValueError(f"Invalid path: {path}")


def main(config_path="./params.json", test_path=None):
    try:
        args = json2args(config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        assert torch.cuda.is_available(), "Currently, we only support CUDA version"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if args.mode == "train":
            print_mode("Training")
            goTraining(args)
        elif args.mode == "test":
            args.test_path = test_path
            validate_path(args.test_path)
            print_mode("Testing")
            goTesting(args)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except AssertionError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main(config_path="/data/root/wyq/FAST/checkpoint/spe202502171643/config.json",
         test_path="/data/root/wyq/FAST/data/train/spe")
