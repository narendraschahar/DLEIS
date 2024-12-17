# Deep Learning-Empowered Image Steganography: Architectural Innovations and Performance Benchmarking

## Overview
This framework implements and evaluates multiple state-of-the-art deep learning architectures for image steganography. It includes eight models:
- RDN (Residual Dense Network)
- ViT-AA (Vision Transformer with Adaptive Attention)
- PGN (Progressive Generation Network)
- DSA (Dual-Stream Architecture)
- WHN (Wavelet-based Hybrid Network)
- MAT (Mutual Attention Transformer)
- EAPT (Efficient Attention Pyramid Transformer)


## System Requirements
- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 32GB+ RAM
- 100GB+ disk space for datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/narendraschahar/DLEIS
cd DLEIS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download and prepare the DIV2K dataset:
```bash
python scripts/prepare_datasets.py --dataset div2k
```

2. Download additional test datasets:
```bash
python scripts/prepare_datasets.py --dataset bossbase
python scripts/prepare_datasets.py --dataset sipi
```

## Training

1. Train a specific model:
```bash
python scripts/train.py --config configs/model_configs/whn_config.yaml --model WHN
```

2. Monitor training:
```bash
tensorboard --logdir runs
```

## Evaluation

1. Evaluate all models:
```bash
python scripts/evaluate.py --config configs/default_config.yaml
```

2. Generate comparison plots:
```bash
python scripts/generate_plots.py --results_dir results
```

## Directory Structure
```
DLEIS/
├── configs/               # Configuration files
├── src/                  # Source code
│   ├── models/          # Model implementations
│   ├── data/            # Dataset handling
│   ├── utils/           # Utilities
│   └── losses/          # Loss functions
├── scripts/              # Training/evaluation scripts
├── tests/                # Unit tests
└── results/              # Results and plots
```

## Model Configuration

Each model can be configured through its respective YAML file in `configs/model_configs/`. Key parameters include:

- Network architecture
- Training parameters
- Loss weights
- Data augmentation settings

Example configuration:
```yaml
model_name: "WHN"
num_blocks: 6
base_channels: 64
wavelet_type: "haar"
training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 200
```

## Results

The framework generates comprehensive results including:
- PSNR and SSIM metrics
- Robustness against attacks
- Visual comparisons
- Performance benchmarks

Results are saved in the `results/` directory and can be visualized using the provided plotting utilities.


