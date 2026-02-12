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
    #SPECIFY TEST CONFIG PATH EACH TIME RIGHT NOW AND FIX LATER
    CONFIG_PATH = '/home/schollab-gaga/Documents/FAST/checkpoint/202602120922/config.json' #where saved config file from training


#update json
with open(CONFIG_PATH, 'r') as f:
    params = json.load(f) 
if train:   
    params['train_frames'] = 2000
    params['miniBatch_size'] = 8
    params['batch_size'] = 1
    params['num_workers'] = 16
    params['save_freq'] = 10
    params['epochs'] = 10
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