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
import signal
import threading
import datetime
import psutil
import torch
from train import goTraining
from test import goTesting
from utils.config import json2args
from utils.h5_utils import h5_to_tiff, tif_stacks_to_h5

# ===== DATA FOLDERS TO PROCESS =====
# Add paths to folders containing registered.h5 (one per line)
DATA_FOLDERS = [
    '/mnt/bigdata/BRUKER/TSeries-04032026-1406-001/',
    '/mnt/bigdata/BRUKER/TSeries-04032026-1406-003/',
    '/mnt/bigdata/BRUKER/TSeries-04032026-1406-004/',
    '/mnt/bigdata/BRUKER/TSeries-04032026-1406-005/',
    '/mnt/bigdata/BRUKER/TSeries-04032026-1406-006/',
    '/mnt/bigdata/BRUKER/TSeries-04152026-1333-001/',
    '/mnt/bigdata/BRUKER/TSeries-04152026-1636-002/',
    '/mnt/bigdata/BRUKER/TSeries-04152026-1636-003/',
    '/mnt/bigdata/BRUKER/TSeries-04152026-1636-004/',
    '/mnt/bigdata/BRUKER/TSeries-04302026-1323-001/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-001/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-002/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-004/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-005/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-006/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-007/',
    '/mnt/bigdata/BRUKER/TSeries-05012026-1510-008/',
    '/mnt/bigdata/BRUKER/TSeries-05062026-1510-002/',
    '/mnt/bigdata/BRUKER/TSeries-05062026-1510-003/',
    '/mnt/bigdata/BRUKER/TSeries-05062026-1510-004/',
]
# ====================================

# ===== CONFIGURATION =====
FAST_DIR = '/home/schollab-gaga/Documents/FAST/'
BASE_CONFIG_PATH = os.path.join(FAST_DIR, 'userparams.json')
# Training hyperparameters
TRAIN_FRAMES = 2000
MINIBATCH_SIZE = 16 
BATCH_SIZE = 1
NUM_WORKERS = 16
EPOCHS = 5
SAVE_FREQ = EPOCHS

# Set to True to skip Steps 1 & 2 (h5→TIFF conversion + training).
# Use this when training already completed and you want to resume from inference.
# registered/ and training/ dirs must already exist in each data folder,
# and a checkpoint/ dir with a valid config.json must be present.
SKIP_TRAINING = False
# =========================


def _log(msg, log_path=None):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(line + '\n')

class MemoryMonitor:
    """Logs RAM and GPU memory to terminal + file every `interval` seconds."""

    def __init__(self, log_path, interval=30):
        self.log_path = log_path
        self.interval = interval
        self._step = 'init'
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_step(self, step):
        self._step = step

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        vm = psutil.virtual_memory()
        while not self._stop.wait(self.interval):
            ram_used = psutil.virtual_memory().used / 1e9
            ram_total = vm.total / 1e9
            ram_pct = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                gpu_used = torch.cuda.memory_allocated() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_str = f"  GPU {gpu_used:.1f}/{gpu_total:.1f} GB"
            else:
                gpu_str = ""
            line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [MEM] step={self._step}  RAM {ram_used:.1f}/{ram_total:.1f} GB ({ram_pct:.0f}%){gpu_str}\n"
            with open(self.log_path, 'a') as f:
                f.write(line)

def setup_cuda():
    """Configure CUDA environment."""
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def process_folder(dataFolder, monitor):
    """
    Run the full FAST pipeline on a single data folder.

    Expected input: dataFolder containing registered.h5
    Output: dataFolder/inference.h5, dataFolder/checkpoint/, one example TIFF
    """
    h5_path = os.path.join(dataFolder, 'registered.h5')
    registered_dir = os.path.join(dataFolder, 'registered')
    training_dir = os.path.join(dataFolder, 'training')
    result_dir = os.path.join(dataFolder, 'result')
    log_path = monitor.log_path

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"registered.h5 not found in {dataFolder}")

    _log(f"\n{'='*60}", log_path)
    _log(f"Processing: {dataFolder}", log_path)
    _log(f"{'='*60}", log_path)

    if SKIP_TRAINING:
        _log("[Step 1/5] SKIPPED (SKIP_TRAINING=True)", log_path)
        _log("[Step 2/5] SKIPPED (SKIP_TRAINING=True)", log_path)
        if not os.path.isdir(registered_dir):
            raise FileNotFoundError(f"registered/ not found in {dataFolder} — cannot skip Step 1")
    else:
        # --- Step 1: Convert registered.h5 to TIFF stacks ---
        monitor.set_step('step1_tiff_export')
        _log("[Step 1/5] Converting registered.h5 to TIFF stacks...", log_path)
        os.makedirs(registered_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)
        h5_to_tiff(h5_path, output_dir=registered_dir)

        # Copy the first TIFF stack to training/
        tif_files = sorted(glob.glob(os.path.join(registered_dir, '*.tif')))
        if not tif_files:
            raise FileNotFoundError(f"No TIFF files created in {registered_dir}")
        first_tif = tif_files[0]
        shutil.copy2(first_tif, os.path.join(training_dir, os.path.basename(first_tif)))
        _log(f"  Copied {os.path.basename(first_tif)} to training/", log_path)

        # --- Step 2: Train ---
        monitor.set_step('step2_training')
        _log("[Step 2/5] Training...", log_path)
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
        _log(f"  Training data: {args.train_folder}", log_path)
        goTraining(args)

    # --- Step 3: Test (inference) ---
    monitor.set_step('step3_inference')
    _log("[Step 3/5] Running inference...", log_path)
    # Find the latest checkpoint config saved by goTraining
    checkpoint_root = os.path.join(dataFolder, 'checkpoint')
    subdirs = sorted([d for d in os.listdir(checkpoint_root)
                      if os.path.isdir(os.path.join(checkpoint_root, d))])
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories in {checkpoint_root}")
    test_config_path = os.path.join(checkpoint_root, subdirs[-1], 'config.json')
    _log(f"  Using checkpoint config: {test_config_path}", log_path)

    # Ensure results_dir is set correctly in the checkpoint config
    with open(test_config_path, 'r') as f:
        test_params = json.load(f)
    test_params['results_dir'] = dataFolder
    with open(test_config_path, 'w') as f:
        json.dump(test_params, f, indent=4)

    args = json2args(test_config_path)
    args.test_path = registered_dir
    _log(f"  Test data: {args.test_path}", log_path)
    goTesting(args)

    # --- Step 4: Convert result TIFFs to inference.h5 ---
    monitor.set_step('step4_h5_export')
    _log("[Step 4/5] Converting results to inference.h5...", log_path)
    inference_h5_path = os.path.join(dataFolder, 'inference.h5')
    tif_stacks_to_h5(result_dir, inference_h5_path, h5_key='mov',
                     delete_tiffs=False, frame_offset=False)
    _log(f"  Saved: {inference_h5_path}", log_path)

    # --- Step 5: Copy example TIFF and cleanup ---
    monitor.set_step('step5_cleanup')
    _log("[Step 5/5] Cleanup...", log_path)
    # Copy first result TIFF to main data folder as a sample
    result_tifs = sorted(glob.glob(os.path.join(result_dir, '*.tif')))
    if result_tifs:
        example_tif = result_tifs[0]
        dest = os.path.join(dataFolder, os.path.basename(example_tif))
        shutil.copy2(example_tif, dest)
        _log(f"  Copied example: {os.path.basename(example_tif)}", log_path)

    # Delete all created subfolders except checkpoint/
    for subdir in [registered_dir, training_dir, result_dir]:
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
            _log(f"  Deleted: {subdir}", log_path)

    # Clean up temp config (only exists when SKIP_TRAINING=False)
    run_config_path = os.path.join(dataFolder, '_run_config.json')
    if os.path.exists(run_config_path):
        os.remove(run_config_path)

    _log(f"Done: {dataFolder}", log_path)
    _log(f"  checkpoint/  - model weights + config", log_path)
    _log(f"  inference.h5 - denoised output", log_path)
    _log(f"  *.tif        - example result stack", log_path)


def main():
    setup_cuda()

    # Single log file for the whole run, named by start time
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = os.path.join(FAST_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'_pipeline_log_{run_ts}.txt')

    monitor = MemoryMonitor(log_path, interval=30)
    monitor.start()

    # Write a clean-exit marker on normal exit; its absence means we were killed
    clean_exit = {'status': 'running', 'started': run_ts}
    marker_path = os.path.join(logs_dir, '_pipeline_status.json')

    def _write_marker(status, extra=None):
        clean_exit['status'] = status
        clean_exit['updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if extra:
            clean_exit.update(extra)
        with open(marker_path, 'w') as f:
            json.dump(clean_exit, f, indent=2)

    _write_marker('running')

    def _on_signal(signum, _):
        _log(f"Caught signal {signum} — pipeline interrupted", log_path)
        _write_marker('interrupted', {'signal': signum})
        monitor.stop()
        raise SystemExit(1)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)

    total = len(DATA_FOLDERS)
    _log(f"FAST Pipeline: {total} folder(s) to process  |  log: {log_path}", log_path)

    try:
        for i, folder in enumerate(DATA_FOLDERS, 1):
            _log(f"\n{'#'*60}", log_path)
            _log(f"  Folder {i}/{total}: {folder}", log_path)
            _log(f"{'#'*60}", log_path)
            clean_exit['current_folder'] = folder
            _write_marker('running')
            process_folder(folder, monitor)
            _write_marker('running', {'last_completed_folder': folder})
        _log(f"\n{'='*60}", log_path)
        _log(f"All {total} folder(s) complete!", log_path)
        _log(f"{'='*60}", log_path)
        _write_marker('complete')
    except Exception as e:
        _log(f"ERROR: {e}", log_path)
        _write_marker('error', {'error': str(e)})
        raise
    finally:
        monitor.stop()


if __name__ == '__main__':
    main()
