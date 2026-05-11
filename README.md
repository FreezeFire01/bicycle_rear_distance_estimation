# Bicycle Rear Distance Estimation

> **Bachelor thesis project** — Vehicle distance estimation from a bicycle's rear-facing camera using YOLO + Random Forest and Dist-YOLO end-to-end approach.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

## About

This is the official code release for the bachelor thesis *"Increasing the safety of non-motorized vehicles on roads"* (FIIT STU Bratislava, 2026).

The system estimates distances to approaching vehicles from a Garmin Varia RCT715 rear bicycle camera using two complementary approaches:

1. **Modular pipeline** — YOLO detector + Random Forest distance regressor
2. **End-to-end Dist-YOLO** — YOLO architecture extended with distance prediction head

## Key Results

| Method | MAE ≤50m | MAE full range | mAP@0.5 |
|---|---|---|---|
| YOLO + Random Forest | 4.69m | 6.34m (3-143m) | 0.736 |
| Dist-YOLO end-to-end | 3.50m | 5.60m (3-90m) | 0.271 |

Cross-dataset generalization (nuScenes → Garmin):
- Pixel features: MAE 18.35m
- FOV-invariant features: **MAE 12.46m (32% improvement)**

All results use **block-split methodology** preventing clip-boundary data leakage.

## Modified Ultralytics Framework

This project includes a **modified version of Ultralytics 8.4.41** with custom Dist-YOLO architecture support. The modifications add an integrated distance regression head to the standard YOLO detector.

**Modified files in `ultralytics/`:**

| File | Modification |
|---|---|
| `ultralytics/nn/modules/head.py` | Added `DetectDist` class with regression branch for distance estimation |
| `ultralytics/utils/loss.py` | Extended `v8DistDetectionLoss` with $L_1$ loss for distance |
| `ultralytics/data/dataset.py` | New `YOLODistDataset` class for 6-column labels (cls, cx, cy, w, h, distance) |
| `ultralytics/data/utils.py` | Disabled column count validation for Dist-YOLO mode |
| `ultralytics/nn/tasks.py` | Registered `DetectDist` in supported heads |

**License compliance:** Because Ultralytics is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), this entire repository (including all modifications and original code) is also released under **AGPL-3.0** to maintain license compatibility. Any derivative work that uses or modifies this code must therefore also be released under AGPL-3.0 if distributed.

## Installation

### Requirements
- Linux (Ubuntu 22.04+) or Windows 11 with WSL2
- Python 3.12
- NVIDIA GPU with CUDA 12.8+ (tested on RTX 5070 Ti)
- 16+ GB RAM (32 GB recommended for training)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/<user>/bicycle-rear-distance-estimation.git
cd bicycle-rear-distance-estimation

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install modified Ultralytics (REQUIRED for Dist-YOLO)
cd ultralytics && pip install -e . && cd ..

# 5. Few models on Github/all of them + dataset can be asked from supervisor
```

### Running inference on a video

The scripts use **argument-based paths** so you can run them from any directory.

```bash
# Modular pipeline (YOLO + Random Forest)
python scripts/evaluation/analyse_video.py \
    --video your_video.mp4 \
    --yolo final_models/yolo_garmin_v3.pt \
    --regressor final_models/rf_garmin_full.pkl

# End-to-end Dist-YOLO
python scripts/evaluation/dist_yolo_test_video.py \
    --video your_video.mp4 \
    --model final_models/dist_yolo_garmin_v3.pt

# Use webcam instead of video file
python scripts/evaluation/analyse_video.py --video 0
```

## Hardware Requirements

### For training
- NVIDIA GPU with 8+ GB VRAM (tested on RTX 5070 Ti)
- 32 GB system RAM recommended
- CUDA 12.8+ for Blackwell architecture

### For inference
- **Desktop:** Any NVIDIA GPU with 4+ GB VRAM
- **Edge:** Raspberry Pi 5 (8 GB RAM)
- **CPU-only:** Possible but slow (~2 FPS)

## Datasets

### Garmin Dataset (custom)

Recorded with **Garmin Varia RCT715** rear bicycle radar camera. Synchronized radar distance measurements with video clips during real cycling rides in Slovakia.

| Split | Frames | Bounding boxes |
|---|---|---|
| Train | 1,574 | ~1,930 |
| Val | 150 | ~120 |
| Test | 69 | ~62 |
| **Total** | **1,793** | **2,112** |

**Distance range:** 3-143m
**Methodology:** Block-split with 1-clip buffer between splits

**Download:** You have to ask the supervisor for the DATASET and bigger models

### nuScenes Subset

Custom subset prepared from [nuScenes v1.0-trainval](https://www.nuscenes.org/) with 31,054 vehicle annotations filtered for ≤110m range.

## Repository Structure

```
.
├── ultralytics/                     # Modified Ultralytics framework (AGPL-3.0)
├── scripts/
│   ├── annotation/                  # Labeling pipeline
│   ├── dataset_prep/                # Dataset preparation
│   ├── evaluation/                  # Video preview & evaluation
│   ├── synchronization/             # Garmin radar-video sync
│   ├── compare_regressors.py        # 7 regressors benchmark
│   ├── cross_dataset_regressor.py   # Cross-dataset with FOV features
│   ├── dist_yolo.py                 # DistYOLO API class
│   └── train_dist_yolo_garmin.py    # Dist-YOLO training
├── YOLO_training/
│   ├── train_yolo.py
│   └── eval_yolo_crossval.py
├── final_models/                    # Pretrained models (download from Zenodo)
├── figures_for_bp/                  # Generated figures
├── requirements.txt
├── LICENSE                          # AGPL-3.0(ultralytics required)
└── README.md
```

## Reproducing Training

```bash
# 1. Prepare datasets
python scripts/dataset_prep/make_garmin_dataset.py
python scripts/dataset_prep/block_split_dataset.py

# 2. Train YOLO baseline
python YOLO_training/train_yolo.py --strategy garmin_only --model yolo11n

# 3. Train Random Forest regressor
python scripts/train_random_forest.py

# 4. Train Dist-YOLO
python scripts/train_dist_yolo_garmin.py --epochs 100 --batch 16

# 5. Evaluate
python scripts/eval_dist_yolo.py
python scripts/compare_regressors.py
```

Total training time: ~16 hours on RTX 5070 Ti.

## Methodology Highlights

### Block-Split for Temporal Data

Garmin Varia automatically splits continuous rides into 30-second clips, which can cause **clip-boundary leakage** with traditional group-by-clip splits. We did block-split with buffer to overcome leakage.

At 50-90 km/h, vehicles travel 420-750m during the 30s buffer, exceeding the 140m radar range.

### FOV-Invariant Features

For cross-dataset generalization, we convert pixel features to angular space using camera FOV:

```python
w_angle = (bbox_width / image_width) * fov_horizontal
h_angle = (bbox_height / image_height) * fov_vertical
```

Result: 32% MAE reduction when transferring nuScenes-trained models to Garmin.


## Acknowledgments

- **Supervisor:** doc. Ing. Rastislav Bencel, PhD.
- Modified Ultralytics framework based on [official Ultralytics 8.4.41](https://github.com/ultralytics/ultralytics)
- Dist-YOLO architecture inspired by [Vajgl et al. 2022](https://doi.org/10.3390/s22134801)
- Garmin FIT SDK integration via [fitparse](https://github.com/dtcooper/python-fitparse)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** in compliance with the upstream Ultralytics framework license.

The full license text is available in [LICENSE](LICENSE) or at [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html).

---

**Contact:** Jozef Magula | jojo.magula@gmail.com
