# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later


from train import goTraining
from test import goTesting
from utils.config import json2args
import os
import torch

FASTdir = '/home/schollab-gaga/Documents/FAST'
dataDir = '/mnt/bigdata/BRUKER/TSeries-07132025-1042-002'

# ===== CONFIGURATION =====
CONFIG_PATH = FASTdir + '/inference_params.json'
TRAIN_DATA_PATH = dataDir
TEST_DATA_PATH = dataDir
# =========================


def print_mode(mode):
    print(f"Running in {mode} mode")

def validate_path(path):
    if not path or not os.path.exists(path):
        raise ValueError(f"Invalid path: {path}")

def main():
    try:
        print(f"Load configuration file path: {CONFIG_PATH}")
        args = json2args(CONFIG_PATH)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        assert torch.cuda.is_available(), "Currently, we only support CUDA version"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if args.mode == "train":
            args.train_folder = TRAIN_DATA_PATH
            validate_path(args.train_folder)
            print_mode("Training")
            print(f"Training data path: {args.train_folder}")
            goTraining(args)
        elif args.mode == "test":
            args.test_path = TEST_DATA_PATH
            validate_path(args.test_path)
            print_mode("Testing")
            print(f"Test data path: {args.test_path}")
            goTesting(args)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()