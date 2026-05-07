# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later
# # updated by Scholl Lab, 2025-07-13
# FAST Pipeline Main Module

# This module orchestrates the complete FAST (Fluorescence Analysis and Source Tracking) 
# pipeline for processing two-photon microscopy data. It handles the conversion of HDF5 
# registered data to training formats, model training, inference on test data, and 
# conversion of results back to HDF5 format.

# PREREQUISITES:
#     - Motion correction must be completed FIRST
#     - Input data must contain a 'registered.h5' file (output from motion correction)
#     - CUDA-compatible GPU is required
#     - All dependencies from requirements.txt installed
#     - userparams.json configuration file present in FAST_DIR
#     - This script should be run with the FAST environment activated
#     - to WATCH GPU use this command in terminal: 'watch -n 2 nvidia-smi'

# WORKFLOW:
#     1. Convert registered.h5 to TIFF stacks for processing
#     2. Train deep learning model on selected frames
#     3. Run inference (testing) on all registered data
#     4. Convert denoised results back to inference.h5
#     5. Generate example TIFF output and cleanup intermediate files

# INPUT:
#     - DATA_FOLDERS: List of paths containing registered.h5 files
#     - Each folder should contain output from motion correction pipeline

# OUTPUT (per folder):
#     - checkpoint/: Trained model weights and configuration
#     - inference.h5: Denoised output in HDF5 format
#     - *.tif: Example result TIFF stack

# NOTES:
#     - CUDA is mandatory; CPU-only execution is not supported
#     - WATCH GPU MEMORY WITH THIS COMMAND IN BASH: watch -n 2 nvidia-smi 
#     - Training time depends on dataset size and GPU memory
#     - Intermediate TIFF directories are automatically deleted post-processing

import os
import json
import shutil
import glob
import torch
from train import goTraining
from test import goTesting
from utils.config import json2args
from utils.h5_utils import h5_to_tiff, tif_stacks_to_h5

# ===== DATA FOLDERS TO PROCESS =====
# Add paths to folders containing registered.h5 (one per line)
DATA_FOLDERS = [
    '/mnt/bigdata/SCANIMAGE/TSeries-04022026-1315-003/2026-04-23/',
    # '/mnt/bigdata/BRUKER/TSeries-07132025-1042-005/',
    # '/mnt/bigdata/BRUKER/TSeries-07132025-1042-003/',
]
# ====================================

# ===== CONFIGURATION =====
FAST_DIR = '/home/schollab-dion/Documents/FAST-scholab/'
BASE_CONFIG_PATH = os.path.join(FAST_DIR, 'userparams.json')
# Training hyperparameters
TRAIN_FRAMES = 1000
MINIBATCH_SIZE = 16 
BATCH_SIZE = 1
NUM_WORKERS = 16
SAVE_FREQ = 25
EPOCHS = 100

# Set to True to skip Steps 1 & 2 (h5→TIFF conversion + training).
# Use this when training already completed and you want to resume from inference.
# registered/ and training/ dirs must already exist in each data folder,
# and a checkpoint/ dir with a valid config.json must be present.
SKIP_TRAINING = False
# =========================


def setup_cuda():
    """Configure CUDA environment."""
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def process_folder(dataFolder):
    """
    Run the full FAST pipeline on a single data folder.

    Expected input: dataFolder containing registered.h5
    Output: dataFolder/inference.h5, dataFolder/checkpoint/, one example TIFF
    """
    h5_path = os.path.join(dataFolder, 'registered.h5')
    registered_dir = os.path.join(dataFolder, 'registered')
    training_dir = os.path.join(dataFolder, 'training')
    result_dir = os.path.join(dataFolder, 'result')

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"registered.h5 not found in {dataFolder}")

    print(f"\n{'='*60}")
    print(f"Processing: {dataFolder}")
    print(f"{'='*60}")

    if SKIP_TRAINING:
        print("\n[Step 1/5] SKIPPED (SKIP_TRAINING=True)")
        print("\n[Step 2/5] SKIPPED (SKIP_TRAINING=True)")
        if not os.path.isdir(registered_dir):
            raise FileNotFoundError(f"registered/ not found in {dataFolder} — cannot skip Step 1")
    else:
        # --- Step 1: Convert registered.h5 to TIFF stacks ---
        print("\n[Step 1/5] Converting registered.h5 to TIFF stacks...")
        os.makedirs(registered_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
        h5_to_tiff(h5_path, output_dir=registered_dir)

        # Copy the first TIFF stack to training/
        tif_files = sorted(glob.glob(os.path.join(registered_dir, '*.tif')))
        if not tif_files:
            raise FileNotFoundError(f"No TIFF files created in {registered_dir}")
        first_tif = tif_files[0]
        shutil.copy2(first_tif, os.path.join(training_dir, os.path.basename(first_tif)))
        print(f"  Copied {os.path.basename(first_tif)} to training/")

        # --- Step 2: Train ---
        print("\n[Step 2/5] Training...")
        with open(BASE_CONFIG_PATH, 'r') as f:
            params = json.load(f)

        params['train_frames'] = TRAIN_FRAMES
        params['miniBatch_size'] = MINIBATCH_SIZE
        params['batch_size'] = BATCH_SIZE
        params['num_workers'] = NUM_WORKERS
        params['save_freq'] = SAVE_FREQ
        params['epochs'] = EPOCHS
        params['results_dir'] = dataFolder
        params['mode'] = 'train'

        # Write a working copy of the config for this run
        run_config_path = os.path.join(dataFolder, '_run_config.json')
        with open(run_config_path, 'w') as f:
            json.dump(params, f, indent=4)

        args = json2args(run_config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        args.train_folder = training_dir
        print(f"  Training data: {args.train_folder}")
        goTraining(args)

    # --- Step 3: Test (inference) ---
    print("\n[Step 3/5] Running inference...")
    # Find the latest checkpoint config saved by goTraining
    checkpoint_root = os.path.join(dataFolder, 'checkpoint')
    subdirs = sorted([d for d in os.listdir(checkpoint_root)
                      if os.path.isdir(os.path.join(checkpoint_root, d))])
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories in {checkpoint_root}")
    test_config_path = os.path.join(checkpoint_root, subdirs[-1], 'config.json')
    print(f"  Using checkpoint config: {test_config_path}")

    # Ensure results_dir is set correctly in the checkpoint config
    with open(test_config_path, 'r') as f:
        test_params = json.load(f)
    test_params['results_dir'] = dataFolder
    with open(test_config_path, 'w') as f:
        json.dump(test_params, f, indent=4)

    args = json2args(test_config_path)
    args.test_path = registered_dir
    print(f"  Test data: {args.test_path}")
    goTesting(args)

    # --- Step 4: Convert result TIFFs to inference.h5 ---
    print("\n[Step 4/5] Converting results to inference.h5...")
    inference_h5_path = os.path.join(dataFolder, 'inference.h5')
    tif_stacks_to_h5(result_dir, inference_h5_path, h5_key='mov',
                     delete_tiffs=False, frame_offset=False)
    print(f"  Saved: {inference_h5_path}")

    # --- Step 5: Copy example TIFF and cleanup ---
    print("\n[Step 5/5] Cleanup...")
    # Copy first result TIFF to main data folder as a sample
    result_tifs = sorted(glob.glob(os.path.join(result_dir, '*.tif')))
    if result_tifs:
        example_tif = result_tifs[0]
        dest = os.path.join(dataFolder, os.path.basename(example_tif))
        shutil.copy2(example_tif, dest)
        print(f"  Copied example: {os.path.basename(example_tif)}")

    # Delete all created subfolders except checkpoint/
    for subdir in [registered_dir, training_dir, result_dir]:
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
            print(f"  Deleted: {subdir}")

    # Clean up temp config (only exists when SKIP_TRAINING=False)
    run_config_path = os.path.join(dataFolder, '_run_config.json')
    if os.path.exists(run_config_path):
        os.remove(run_config_path)

    print(f"\nDone: {dataFolder}")
    print(f"  checkpoint/  - model weights + config")
    print(f"  inference.h5 - denoised output")
    print(f"  *.tif        - example result stack")


def main():
    setup_cuda()
    total = len(DATA_FOLDERS)
    print(f"FAST Pipeline: {total} folder(s) to process")
    for i, folder in enumerate(DATA_FOLDERS, 1):
        print(f"\n{'#'*60}")
        print(f"  Folder {i}/{total}")
        print(f"{'#'*60}")
        process_folder(folder)
    print(f"\n{'='*60}")
    print(f"All {total} folder(s) complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
