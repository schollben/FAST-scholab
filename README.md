# FAST: FrAme-multiplexed SpatioTemporal Learning Strategy

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

FAST is a real-time self-supervised denoising framework for fluorescence neural imaging, achieving enhanced image quality in low-SNR scenarios through spatiotemporal joint optimization.

## ✨ Key Features

- 🚀 **Real-Time Processing**: >1000 fps denoising (hardware-dependent)
- 🤖 **Self-Supervised Learning**: Eliminates need for clean ground truth
- 🔄 **Spatiotemporal Optimization**: Frame-multiplexing enhances SNR
- 📊 **High Adaptability**: Suitable for various fluorescence imaging data

## 🛠 Installation

### Requirements
- Python 3.9+
- PyTorch 2.x (CUDA version aligned with your hardware)
- CUDA-capable GPU (recommended)

### Quick Setup
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate FAST
```

## 🚀 Quick Start

### Command Line Mode
```bash
# Training mode
python main.py --config_path "./params.json"

# Testing mode (with pretrained model)
python main.py --config_path "./checkpoint/model_name/config.json"
```

### GUI Mode
```bash
# Launch training GUI
python Train_GUI.py

# Launch testing GUI
python Test_GUI.py
```

## 📁 Directory Structure
```
FAST/
├── data/
│   └── train/          # Training data
├── checkpoint/         # Model checkpoints
├── result/            # Output results
├── src/               # Source code
└── configs/           # Configuration files
```

## ⚙️ Configuration

Customize model parameters by modifying `params.json`:

```json
{
    "data_extension": "tif",
    "epochs": 100,
    "miniBatch_size": 4,
    "lr": 0.0001,
    "weight_decay": 0.9,
    "gpu_ids": "0",
    "train_frames": 2000,
    "data_type": "3D",
    "denoising_strategy": "FAST",
    "seed": 123,
    "save_freq": 25,
    "clip_gradients": 20.0,
    "num_workers": 0,
    "batch_size": 1
}
```

## 🤝 Contributing

We welcome contributions, particularly:

1. 🐛 Bug reports and fixes
2. ✨ New feature proposals and implementations
3. 📚 Documentation improvements
4. 🎨 Code optimizations

### Coding Standards
- Use `UpperCamelCase` for class names
- Use `lowercase_with_underscores` for functions and variables
- Include docstrings for core functions
- Follow PEP8 standards (validate using `flake8`)

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. This means you are free to:

- ✅ Use
- ✅ Modify
- ✅ Distribute

But you must:
- ⚠️ Disclose source
- ⚠️ Include original copyright
- ⚠️ Use the same license

See [LICENSE](LICENSE) file for full text.

## ❓ FAQ

<details>
<summary>To do</summary>
--
</details>



## 📮 Contact

- 📧 Email: yiqunwang22@fudan.edu.cn
- 🌐 Project Page: [GitHub Repository](https://github.com/yourusername/FAST)

---

### Citation

If you use FAST in your research, please cite our paper:

```bibtex
@article{wang2024real,
    title={Real-time self-supervised denoising for high-speed fluorescence neural imaging},
    author={Wang, Yiqun and Others},
    journal={Coming Soon},
    year={2025}
}
```