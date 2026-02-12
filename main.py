# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import json
import shutil
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from train import goTraining
from test import goTesting
from utils.config import json2args
from utils.h5_utils import h5_to_tiff, tif_stacks_to_h5

# ===== CONFIGURATION =====
FAST_DIR = '/home/schollab-gaga/Documents/FAST'
BASE_CONFIG_PATH = os.path.join(FAST_DIR, 'userparams.json')

# Training hyperparameters
TRAIN_FRAMES = 1000
MINIBATCH_SIZE = 8
BATCH_SIZE = 1
NUM_WORKERS = 16
SAVE_FREQ = 10
EPOCHS = 10  # set to 100 for actual running
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

    print(f"\n{'='*60}")
    print(f"Processing: {dataFolder}")
    print(f"{'='*60}")

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
    result_tifs = sorted(glob.glob(os.path.join(result_dir, '*.tif')))
    if result_tifs:
        example_tif = result_tifs[0]
        dest = os.path.join(dataFolder, os.path.basename(example_tif))
        shutil.copy2(example_tif, dest)
        print(f"  Copied example: {os.path.basename(example_tif)}")

    shutil.rmtree(registered_dir)
    print(f"  Deleted: {registered_dir}")
    shutil.rmtree(result_dir)
    print(f"  Deleted: {result_dir}")

    # Clean up temp config
    if os.path.exists(run_config_path):
        os.remove(run_config_path)

    print(f"\nDone: {dataFolder}")
    print(f"  checkpoint/  - model weights + config")
    print(f"  training/    - training TIFF stack")
    print(f"  inference.h5 - denoised output")


def run_pipeline(folders):
    """Run the full pipeline on all selected folders."""
    setup_cuda()
    total = len(folders)
    for i, folder in enumerate(folders, 1):
        print(f"\n{'#'*60}")
        print(f"  Folder {i}/{total}")
        print(f"{'#'*60}")
        process_folder(folder)
    print(f"\n{'='*60}")
    print(f"All {total} folder(s) complete!")
    print(f"{'='*60}")


# ===== GUI =====
class FolderSelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FAST Pipeline")
        self.root.geometry("700x400")
        self.folders = []

        # Title
        tk.Label(root, text="FAST Denoising Pipeline", font=("Arial", 16, "bold")).pack(pady=(10, 5))
        tk.Label(root, text="Select folders containing registered.h5").pack()

        # Button frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Add Folder", command=self.add_folder, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Remove Selected", command=self.remove_folder, width=15).pack(side=tk.LEFT, padx=5)

        # Listbox with scrollbar
        list_frame = tk.Frame(root)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Run button
        self.run_btn = tk.Button(root, text="Run Pipeline", command=self.run,
                                 width=20, height=2, bg="#4CAF50", fg="white",
                                 font=("Arial", 12, "bold"))
        self.run_btn.pack(pady=15)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select data folder containing registered.h5")
        if not folder:
            return
        h5_path = os.path.join(folder, 'registered.h5')
        if not os.path.exists(h5_path):
            messagebox.showerror("Error", f"No registered.h5 found in:\n{folder}")
            return
        if folder in self.folders:
            messagebox.showinfo("Info", "Folder already in list.")
            return
        self.folders.append(folder)
        self.listbox.insert(tk.END, folder)

    def remove_folder(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.listbox.delete(idx)
        self.folders.pop(idx)

    def run(self):
        if not self.folders:
            messagebox.showwarning("Warning", "No folders selected.")
            return
        msg = f"Run pipeline on {len(self.folders)} folder(s)?\n\n"
        msg += "\n".join(f"  {f}" for f in self.folders)
        if not messagebox.askyesno("Confirm", msg):
            return

        # Disable UI during processing
        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.root.update()

        try:
            run_pipeline(list(self.folders))
            messagebox.showinfo("Complete", f"Pipeline finished for {len(self.folders)} folder(s)!")
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline failed:\n{e}")
            raise
        finally:
            self.run_btn.config(state=tk.NORMAL, text="Run Pipeline")


def main():
    root = tk.Tk()
    FolderSelectionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
