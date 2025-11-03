# Loss Landscape Analysis of CNN Normalization Methods

A PyTorch-based experimental framework for comparing normalization techniques in convolutional neural networks through loss landscape visualization and Hessian eigenvalue analysis on MNIST.

## Overview

This project investigates how different normalization methods affect neural network optimization by analyzing:

- **Loss landscape geometry** using filter-normalized random direction visualization
- **Hessian eigenvalue spectrum** to measure local curvature at converged solutions
- **Training dynamics** including convergence speed and generalization performance

## Supported Normalization Methods

- **Batch Normalization** - Standard batch statistics normalization
- **Layer Normalization** - Per-instance normalization (GroupNorm with 1 group)
- **Group Normalization** - Normalization over channel groups (8 groups)
- **RMS Normalization** - Root mean square normalization without mean centering
- **Dynamic Tanh** - Custom learnable tanh-based normalization
- **Piecewise Linear** - Custom piecewise linear activation normalization
- **None** - Baseline without normalization

## Architecture

**MNIST CNN** (3-layer convolutional + 2 fully connected):

- Conv layers: 32 → 64 → 64 channels (3×3 kernels)
- FC layers: 3136 → 128 → 10
- Normalization applied after conv1, conv2, conv3, and fc1
- Dropout (p=0.25) before final classification layer
- ~459k parameters

## Installation

```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install torch torchvision numpy pyhessian
```

**Requirements:**

- Python 3.8+
- PyTorch 1.12+
- numpy

## Usage

### Basic Training

Edit `main.py` to select normalization method:

```python
# Train single normalization method
train_phase('group')
analyze_loss_landscape('group')
analyze_hessian('group')
```

### Batch Experiments

Train and analyze all normalization methods:

```python
norm_types = ['batch', 'layer', 'group', 'rms', 'dyt', 'piecewise', 'none']

for norm_type in norm_types:
    train_phase(norm_type)
    analyze_loss_landscape(norm_type)
    analyze_hessian(norm_type)
```

### Output Structure

Results are saved to `output/` directory:

```py
output/
├── json/
│   ├── training_<norm>.json      # Training history and metrics
│   ├── landscape_<norm>.json     # Loss landscape data
│   └── hessian_<norm>.json       # Hessian eigenvalue analysis
└── weights/
    └── model_<norm>.pth          # Trained model weights
```

## Key Modules

### `neuralnetwork/`

- `architecture.py` - MNISTNet CNN with configurable normalization
- `trainer.py` - Training loop with validation tracking

### `methods/customnorm/`

- `dyt.py` - Dynamic Tanh normalization
- `linear.py` - Piecewise Linear normalization
- `rms_norm.py` - RMS Normalization

### `methods/trainingutils/`

- `dataloader.py` - MNIST dataset loading with train/val split
- `tensorprep.py` - Data preprocessing and tensor preparation
- `hessian.py` - Hessian eigenvalue computation using PyHessian
- `landscaper.py` - 2D loss landscape visualization

## Citation

If you use this code for research, please cite the relevant normalization papers:

```bibtex
@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:1607.06450},
  year={2016}
}

@inproceedings{wu2018group,
  title={Group normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={ECCV},
  year={2018}
}
```

## License

This project is licensed under the MIT License.
