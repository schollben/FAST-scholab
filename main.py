# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later

from train import goTraining
from test import goTesting
from utils.config import json2args
import os
import torch
import json

# ===== CONFIGURATION =====
#update plan:  hazy-brewing-squirrel.md

train = False
test = True

dataFolder = '/mnt/bigdata/BRUKER/TSeries-07132025-1042-002/'
FASTdir = '/home/schollab-gaga/Documents/FAST'
TRAIN_DATA_PATH = dataFolder + 'training/'
TEST_DATA_PATH = dataFolder + 'registered/'

if train:
    CONFIG_PATH = FASTdir + '/userparams.json'
elif test:
    # Auto-find the most recent config.json from dataFolder/checkpoint/
    checkpoint_root = os.path.join(dataFolder, 'checkpoint')
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_root}")
    # Pick the latest timestamped subfolder
    subdirs = sorted([d for d in os.listdir(checkpoint_root)
                      if os.path.isdir(os.path.join(checkpoint_root, d))])
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories in {checkpoint_root}")
    CONFIG_PATH = os.path.join(checkpoint_root, subdirs[-1], 'config.json')


#update json
with open(CONFIG_PATH, 'r') as f:
    params = json.load(f) 
if train:
    params['train_frames'] = 1000
    params['miniBatch_size'] = 8
    params['batch_size'] = 1
    params['num_workers'] = 16
    params['save_freq'] = 10
    params['epochs'] = 10 #set to 100 for actual running
    params['results_dir'] = dataFolder
elif test:
    params['results_dir'] = dataFolder

with open(CONFIG_PATH, 'w') as f:
    json.dump(params, f, indent=4)
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