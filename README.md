# Enhancing Visual Paralinguistics: Motion-Guided Spatial Denoising for Non-Verbal Interaction Analysis

**Interspeech 2026** | [Paper](#) | [arXiv](#)

*Junjie Wan*  
Harbin Institute of Technology, Shenzhen

## Overview

We propose a lightweight dual-stream framework for robust micro-gesture recognition in unconstrained HCI environments, featuring:

- **MG-SRM**: Motion-Guided Spatial Refinement Module — suppresses static background noise in pose heatmaps via multi-order temporal differences
- **CLF**: Class-Learnable Fusion — adaptively balances RGB and pose modalities based on class-specific preferences

On the MA-52 dataset, our method achieves **67.02% Top-1 accuracy**, outperforming prior state-of-the-art with only **0.072M extra parameters**.

## Installation

```bash
# 1. Create environment
conda create -n mgsrm python=3.8 -y
conda activate mgsrm

# 2. Install PyTorch (tested with torch 1.13, CUDA 11.6)
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116

# 3. Install MMAction2 dependencies
pip install -U openmim
mim install mmengine mmcv mmdet mmpose

# 4. Install this repo
pip install -v -e .
```

## Dataset

Download the [MA-52 dataset](https://github.com/VUT-HFUT/MicroAction) and organize as:

```
data/
└── ma52/
    ├── raw_videos/
    └── MA-52_openpose_28kp/
        ├── MA52_train.pkl
        ├── MA52_val.pkl
        └── MA52_test.pkl
```

## Key Files

Our contributions are implemented in the following files:

| File | Description |
|------|-------------|
| `mmaction/models/ops/v3_branch/heatmap_v10.py` | MG-SRM module |
| `mmaction/models/heads/rgbpose_head.py` | CLF head (+ ablation variants) |
| `mmaction/models/backbones/rgbposeconv3d.py` | Dual-stream backbone |
| `configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py` | Training config |

## Training

```bash
python tools/train.py configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py
```

## Testing

```bash
python tools/test.py configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py <checkpoint>
```

## Results

| Method | Top-1 (%) | F1-Mean |
|--------|-----------|---------|
| PoseConv3D | 63.52 | 0.6666 |
| PCAN* (reproduced) | 66.40 | 0.6950 |
| **Ours (MG-SRM + CLF)** | **67.02** | **0.6993** |

## Citation

```bibtex
@inproceedings{wan2026mgsrm,
  title={Enhancing Visual Paralinguistics: Motion-Guided Spatial Denoising for Non-Verbal Interaction Analysis},
  author={Wan, Junjie},
  booktitle={Interspeech},
  year={2026}
}
```

## Acknowledgement

This codebase is built on [MMAction2](https://github.com/open-mmlab/mmaction2).
