# STA-TFM: Spatio-Temporal Aggregation Across Views TransForMer for Pose Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2609.24482-b31b1b.svg)](https://arxiv.org/abs/2609.24482)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22832620-blue)](https://doi.org/10.5281/zenodo.22832620)
[![License: Non-Commercial](https://img.shields.io/badge/license-non--commercial-lightgrey)](LICENSE.txt)

Official code for **STA-TFM** ([arXiv:2609.24482](https://arxiv.org/abs/2609.24482)).

Monocular 3D human pose estimation (HPE) remains challenging due to depth ambiguity, occlusions, and the need for temporal consistency. While multi-view methods provide superior accuracy over monocular approaches, they often require complex setups. We introduce STA-TFM, a transformer-based architecture that combines spatial and temporal information for multi-view pose estimation. The approach leverages DSTformer, a monocular feature extractor, to capture long-range pose dependencies within each view. A fusion transformer then aggregates information across views to produce coherent 3D estimates. To address training data scarcity, we use a data generation pipeline that transforms any existing 3D pose dataset into multi-view setups with controllable parameters. Experiments on various datasets demonstrate that STA-TFM outperforms existing camera-parameter-free multi-view methods. STA-TFM achieves 50.9% and 49.5% reductions in mean per joint position error (MPJPE) and mean per joint velocity error (MPJVE) on the DHP19 dataset. Furthermore, it achieves 6.7% and 7.7% respective reductions on HAA4D, and a 15.2% MPJPE reduction on TotalCapture. STA-TFM handles noisy and missing 2D inputs, supporting potential deployment in healthcare monitoring, athletic assessment, and immersive technologies.

![Graphical Abstract](figures/Graphical%20Abstract.png)

---

## Contents

- [Quick start](#quick-start)
- [1. Installation](#1-installation)
- [2. Download pre-trained weights](#2-download-pre-trained-weights)
- [3. Reproducing the paper (DHP19 + HAA4D)](#3-reproducing-the-paper-dhp19--haa4d)
- [4. Robustness experiments](#4-robustness-experiments)
- [5. Visualization](#5-visualization)
- [6. End-to-end demo on raw videos](#6-end-to-end-demo-on-raw-videos)
- [7. Using your own data (NPZ format)](#7-using-your-own-data-npz-format)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Quick start

Assuming conda is installed and you have downloaded DHP19 and HAA4D (see [§3.1](#31-download-the-raw-datasets)), the full pipeline is:

```bash
git clone https://github.com/Sanofi-Public/STA-TFM.git && cd STA-TFM
bash setup_py_env.sh ./env && conda activate ./env

# Synthetic multi-view data
python -m src.data.data_processing --config conf/data_processing/dhp19.yaml
python -m src.data.data_processing --config conf/data_processing/haa4d.yaml

# Train/test splits
python -m src.data.split_dataset --dataset union --output_dir datasets/haa4d_dhp19_union/split
python -m src.data.split_dataset --dataset dhp19 --output_dir datasets/dhp19/split
python -m src.data.split_dataset --dataset haa4d --output_dir datasets/haa4d/split

# Train (or skip and use the released checkpoint from Zenodo)
python -m src.train

# Inference + evaluation on DHP19
python -m src.infer --fusion_head_ckpt <checkpoint.pth> --input datasets/dhp19/split/test --output results/dhp19
python -m src.eval.evaluate --pred_path results/dhp19/<checkpoint_stem> --gt_path datasets/dhp19/split/test \
    --output_path results/dhp19/eval --dataset dhp19
```

All commands are run from the repository root. Every script accepts `--help`.

---

## 1. Installation

**Requirements**
- Linux or macOS, Python **3.10**, conda (Miniconda/Miniforge)
- An NVIDIA GPU is recommended for training. The paper used a single V100 16 GB (~2.5 h of training on 73,590 four-view windows). Inference and evaluation also run on CPU.
- The raw-video demo ([§6](#6-end-to-end-demo-on-raw-videos)) additionally needs MMPose, which is built from source and requires **Linux + a CUDA build of PyTorch**.

### Option A — setup script (recommended)

```bash
bash setup_py_env.sh ./env               # core: data generation, training, inference, evaluation
bash setup_py_env.sh ./env --with-demo   # also installs MMPose/MMDet/MMCV for the raw-video demo
conda activate ./env
```

The script creates the environment if needed, installs `requirements.txt`, and creates the `datasets/`, `weights/` and `results/` folders.

With `--with-demo`, the script builds the demo stack from source at fixed commits, so every install gets identical code:

| Component | Version | Commit |
|---|---|---|
| MMCV | 2.2.0 | `d2111bd` |
| MMDetection | main, with [patches/mmdet-mmcv-max-version.patch](patches/mmdet-mmcv-max-version.patch) applied | `cfd5d3a` |
| MMPose | 1.3.2 | `759b39c` |

The other demo dependencies are pinned in [requirements-demo.txt](requirements-demo.txt). The commits are set at the top of the demo section in [setup_py_env.sh](setup_py_env.sh).

### Option B — manual

```bash
conda create --prefix ./env python=3.10 -y
conda activate ./env
pip install -r requirements.txt
mkdir -p datasets/dhp19 datasets/haa4d weights results
```

> `requirements.txt` pins `torch==2.13.0`, which pip installs with its default CUDA build on Linux. For a different CUDA version, install PyTorch first following https://pytorch.org/get-started/locally/, then run `pip install -r requirements.txt`.

### Verify the install

```bash
pip install -r requirements-dev.txt
pytest
```

The tests are CPU-only and need no datasets or weights; they check synthetic data generation, seeding, and the train/test split.

---

## 2. Download pre-trained weights

Download `STA-TFM Supplementary Material.zip` (2.7 GB) from Zenodo: https://doi.org/10.5281/zenodo.22832620. It contains:
- `dstformer_weights.pth`: pre-trained DSTformer (MotionBERT) backbone, needed for both training and inference
- STA-TFM checkpoints trained on synthetic HAA4D + DHP19
- Sample output videos and baseline comparison results

Copy the weights into `weights/`:

```
weights/
├── dstformer_weights.pth            # default --dstformer_path for train/infer
└── <sta-tfm-checkpoint>.pth         # pass as --fusion_head_ckpt
```

> Checkpoints trained on CMU Panoptic and TotalCapture cannot be shared because of dataset licensing.

---

## 3. Reproducing the paper (DHP19 + HAA4D)

The released model is trained on the **union** of synthetic DHP19 and HAA4D and evaluated **separately** on each dataset's test split. Each dataset uses its own joint mask and units.

### 3.1 Download the raw datasets

| Dataset | Source | What to keep | Place under |
|---|---|---|---|
| DHP19 | https://sites.google.com/view/dhp19/download | `Vicon_data/` (`S<subject>_<session>_<task>.mat`) | `datasets/dhp19/raw/Vicon_data/` |
| HAA4D | https://cse.hkust.edu.hk/haa4d/dataset/skeletons_3d/primary_classes.zip | `primary_classes.zip` | `datasets/haa4d/raw/primary_classes/` |

```bash
unzip <path-to>/primary_classes.zip -d datasets/haa4d/raw
```

### 3.2 Generate synthetic multi-view data

Each 3D sequence is projected into 4 virtual cameras placed radially around the subject (0°, 90°, 180°, 270°), with random noise on the extrinsics, intrinsics and 2D joints. The output is one `.npz` file per sequence (format: [§7](#7-using-your-own-data-npz-format)).

All settings live in a config file: dataset, paths, scaling, seed and the camera rig. Ready-to-use configs that reproduce the paper are in [conf/data_processing/](conf/data_processing/):

```bash
python -m src.data.data_processing --config conf/data_processing/dhp19.yaml
python -m src.data.data_processing --config conf/data_processing/haa4d.yaml
```

Example ([conf/data_processing/dhp19.yaml](conf/data_processing/dhp19.yaml)):

```yaml
dataset_type: dhp19                       # dhp19 | haa4d | blendmimic | totalcapture
data_dir: datasets/dhp19/raw/Vicon_data   # raw dataset, searched recursively
output_dir: datasets/dhp19/synthetic      # one .npz per sequence is written here
scale: local                              # keep "local" to match STA-TFM
seed: 42                                  # seeds all camera / intrinsic / joint noise

views:                                    # one entry per camera
  - {axis: y, angle: 0,   dist: 5.0, fx: 2751, fy: 2751, cx: 1512, cy: 2016, angle_noise: 15, dist_noise: 4, intrinsic_noise: [50, 10], joint_noise: 0.01}
  - {axis: y, angle: 90,  dist: 5.0, ...}   # second view
  - {axis: y, angle: 180, dist: 5.0, ...}   # third view
  - {axis: y, angle: 270, dist: 5.0, ...}   # fourth view
```

| View field | Meaning |
|---|---|
| `axis`, `angle` | Camera position: rotate `angle` degrees around `axis` (`x`/`y`/`z`). With `y`, the cameras circle the subject. |
| `dist` | Distance from the subject's root. Skeletons are normalised to roughly unit size. |
| `fx`, `fy`, `cx`, `cy` | Intrinsics in pixels. The image size is `(2·cx, 2·cy)`, here 3024 × 4032 ([reference](https://arxiv.org/pdf/2201.10865)). |
| `angle_noise` | ± degrees (uniform), added to `angle` and applied as a random rotation around the other two axes |
| `dist_noise` | ± uniform offset added to `dist` |
| `intrinsic_noise` | `[± focal length, ± principal point]` in pixels (uniform) |
| `joint_noise` | Gaussian 2D joint noise with std = U(0, `joint_noise`) × the 2D body height in that view |

Noise terms are re-sampled for every sequence and view, and are reproducible through `seed`. We recommend keeping the camera intrinsics and extrinsics at their defaults. To try another setup, for example more or fewer cameras, other distances, or no noise, copy a config and edit it.

Command-line flags override the config file, so you don't have to edit it just to change paths:

```bash
python -m src.data.data_processing --config conf/data_processing/dhp19.yaml \
    --data_dir /data/DHP19/Vicon_data --output_dir /data/dhp19_synthetic
```

| Flag | Description |
|---|---|
| `--config` | YAML config file |
| `--dataset`, `--data_dir`, `--output_dir`, `--scale`, `--seed` | Override the matching config entry. Without `--config`, these flags alone are enough, and the paper's camera rig is used. |

The effective config, including the seed, is saved as `config.yaml` in `--output_dir`.

### 3.3 Split into train/test

The splits are defined in [conf/dataset_splits.json](conf/dataset_splits.json): a subject split for DHP19 and an action split for HAA4D. A file goes to the test set if its name starts with one of the `test` entries; every other file goes to the train set.

```bash
# Training set: union of both datasets
python -m src.data.split_dataset --dataset union --output_dir datasets/haa4d_dhp19_union/split
# Per-dataset test sets used for evaluation
python -m src.data.split_dataset --dataset dhp19 --output_dir datasets/dhp19/split
python -m src.data.split_dataset --dataset haa4d --output_dir datasets/haa4d/split
```

`source_dir` in the JSON defaults to `datasets/<name>/synthetic`. Override it with `--dhp19_dir` / `--haa4d_dir`.

### 3.4 Train

```bash
python -m src.train                       # uses the default paths below
python -m src.train --dataset_path <train-dir> --dstformer_path <weights.pth> --epochs 50
```

| Argument | Default |
|---|---|
| `--dataset_path` | `datasets/haa4d_dhp19_union/split/train` |
| `--dstformer_path` | `weights/dstformer_weights.pth` |
| `--model_save_dir` | `weights/training/haa4d_dhp19_union/checkpoints` |
| `--metrics_dir` | `weights/training/haa4d_dhp19_union/metrics` |
| `--epochs`, `--batch_size`, `--lr`, `--num_workers`, `--seed`, `--resume_from`, `--finetune_backbone` | see `configs` in [train.py](src/train.py) |

The remaining architecture hyperparameters (window length 36, stride 12, fusion depth/heads, bottleneck, and others) are set in the `configs` dict in [train.py](src/train.py). The DSTformer backbone is frozen by default. Each run writes:
- `<model_save_dir>/<timestamp>_mv_fusion_head_best.pth`: best checkpoint by validation MPJPE
- `<metrics_dir>/<timestamp>_config.json`: the exact configuration used
- `<metrics_dir>/<timestamp>_metrics.json`: per-epoch metrics

Sample output:
```
Epoch 001 | train_loss 0.094 | val_mpjpe 0.075 | val_p_mpjpe 0.096
```
`val_mpjpe` and `val_p_mpjpe` are computed on normalized skeletons (`keypoints_3d_processed`), so they are **unitless**. Millimetre metrics come from [§3.6](#36-evaluate).

To train in the background:
```bash
TQDM_DISABLE=1 nohup python -m src.train > train.log 2>&1 &
```

### 3.5 Inference

```bash
python -m src.infer --fusion_head_ckpt <checkpoint.pth> \
    --input datasets/dhp19/split/test --output results/dhp19
```

Predictions are written to `<output>/<checkpoint_stem>/<sample>.pred3d.npz`.

| Argument | Description | Default |
|---|---|---|
| `--fusion_head_ckpt` | STA-TFM checkpoint | required |
| `--dstformer_path` | DSTformer weights | `weights/dstformer_weights.pth` |
| `--input` | Folder of `.npz` files or a single `.npz` | `datasets/haa4d_dhp19_union/split/test` |
| `--output` | Output root | `results/haa4d_dhp19_union` |
| `--num_views_test` | Active views (4/3/2/1) for missing-view experiments | `4` |
| `--view_drop_test` | `rand` or `order`: which views to drop | `rand` |
| `--use_gt_2d` | Use oracle GT 2D (`keypoints_2d_gt_processed`) | off |
| `--seed` | Seed for random view dropping | `42` |

### 3.6 Evaluate

```bash
python -m src.eval.evaluate \
    --pred_path   results/dhp19/<checkpoint_stem> \
    --gt_path     datasets/dhp19/split/test \
    --output_path results/dhp19/eval \
    --dataset     dhp19
```

| Argument | Description |
|---|---|
| `--pred_path` | Folder of prediction `.npz` files |
| `--gt_path` | Folder of ground-truth `.npz` files |
| `--output_path` | Output prefix for the CSV files |
| `--dataset` | `dhp19`, `haa4d`, `cmu`, `totalcapture`, `blendmimic3d`. Selects the joint mask ([conf/joints.yaml](conf/joints.yaml)) and the units. |
| `--use_gt_2d` | Use ground-truth 2D keypoints instead of predicted ones |

Outputs:
- `<output_path>_averages.csv`: mean ± std over samples
- `<output_path>_per_sample.csv`: per-sample metrics
- `<output_path>_ap_metrics.csv`: AP and recall at distance thresholds

Metrics: `p_mpjpe` (mm), `p_pck` (%), `p_mpjve` (mm/frame), `AP25/50/100/150`, `Recall@500`. All are Procrustes-aligned.

**Expected results (paper, P-MPJPE in mm):** DHP19 **17.56**, HAA4D **34.77**. Because the synthetic camera noise is random, regenerated data will vary slightly unless you keep the default `--seed`.

---

## 4. Robustness experiments

Noisy 2D inputs and missing keypoints:

```bash
python -m src.eval.assess_noise_robustness --input datasets/dhp19/split/test --output datasets/dhp19/noise_0.02 --mode noise --noise_std 0.02
python -m src.eval.assess_noise_robustness --input datasets/dhp19/split/test --output datasets/dhp19/missing --mode missing --max_remove 4
```

Then run [§3.5](#35-inference) and [§3.6](#36-evaluate) on the generated folders.

Missing views: pass `--num_views_test {3,2,1}` to `src.infer`.

---

## 5. Visualization

Interactive 3D skeleton plus 2D camera views (HTML):

```bash
python -m src.eval.plot --npz datasets/haa4d/split/test/air_guitar_001.npz --output_dir results/visuals
python -m src.eval.plot --npz results/haa4d/<checkpoint_stem>/air_guitar_001.pred3d.npz --output_dir results/visuals
```

- For dataset files, this writes `<sample>.html`.
- For prediction files (which contain `pred_3d`), it writes `<sample>_raw.html` and `<sample>_aligned.html` (Procrustes-aligned).

| Argument | Description | Default |
|---|---|---|
| `--npz` | Path to `.npz` file | required |
| `--output_dir` | Directory for HTML output | required |
| `--yaml` | Joint connectivity YAML | `conf/connectivities/default.yaml` |
| `--world_coord` | `True`/`False`: use world-frame GT (`gt_world`) | `True` |
| `--show_raw` | Use raw (pixel/unnormalized) keypoints | off |
| `--start`, `--end`, `--max_frames` | Frame range | full sequence |

Side-by-side MP4 of the videos with 2D overlays next to the 3D prediction:

```bash
python -m src.eval.video_overlay \
  --pred_npz <inference-result.pred3d.npz> --data_npz <input.npz> \
  --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 \
  --out overlay.mp4
```

---

## 6. End-to-end demo on raw videos

Requires the demo install (`bash setup_py_env.sh ./env --with-demo`). The input is 4 synchronized videos filmed roughly 90° apart around the subject.

```bash
python -m src.demo.demo \
    --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 \
    --dstformer_path weights/dstformer_weights.pth \
    --fusion_head_ckpt weights/<sta-tfm-checkpoint>.pth \
    --output_dir results/demo
```

The demo:
1. Runs [MMPose RTMPose](https://github.com/open-mmlab/mmpose) on every frame
2. Converts COCO-17 keypoints to the canonical 17-joint layout and saves an `.npz`
3. Runs STA-TFM inference
4. Writes `<output_dir>/<checkpoint_stem>/<name>.pred3d.npz` and `<name>_viz.html`

| Argument | Description | Default |
|---|---|---|
| `--videos` | 4 synchronized video files (one per view) | required |
| `--dstformer_path` | DSTformer weights | required |
| `--fusion_head_ckpt` | STA-TFM checkpoint | required |
| `--output_dir` | Output folder | required |
| `--name` | Output file stem | `demo` |
| `--mmpose_model` | MMPose model id | `rtmpose-m_8xb256-420e_coco-256x192` |
| `--frame_skip` | Process every N-th frame | `1` |
| `--device` | Compute device | `cuda:0` if available |

Overlay video of the demo output. `pred3d.npz` already contains the 2D keypoints, so pass it twice. `--frame_skip` must match the value used in the demo.

```bash
python -m src.eval.video_overlay \
    --pred_npz results/demo/<checkpoint_stem>/demo.pred3d.npz \
    --data_npz results/demo/<checkpoint_stem>/demo.pred3d.npz \
    --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 \
    --out results/demo/overlay.mp4 --frame_skip 1
```

---

## 7. Using your own data (NPZ format)

STA-TFM inference needs one `.npz` per sequence with at least `keypoints_2d_processed` (or `keypoints_2d`) and `view_mask`. Training also needs `keypoints_3d_processed`.

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `keypoints_2d` | `[T, V, J, 3]` | `float32` | Raw 2D keypoints in pixels, `[x, y, confidence]` |
| `keypoints_2d_processed` | `[T, V, J, 3]` | `float32` | MotionBERT-style normalized 2D (aspect-ratio aware): X ∈ [-1, 1], Y ∈ [-(H/W), (H/W)], root-centred |
| `keypoints_3d` | `[T, J, 3]` | `float32` | 3D keypoints after `transform_skeleton()` |
| `keypoints_3d_processed` | `[T, J, 3]` | `float32` | Root-centred 3D used as the training target |
| `view_mask` | `[T, V]` | `bool` | Which views are valid per frame |
| `img_wh` | `[V, 2]` | `int64` | Image `[width, height]` per view |
| `cam_params` | `[V]` | `object` | Dicts with `K`, `extrinsic` (3×4 [R\|t]), `axis`, `angle` |
| `gt_world` | `[T, J, 3]` | `float32` | Original 3D skeleton in dataset units (mm for DHP19), used by evaluation |

- **T**: frames, **V**: views (4), **J**: 17 joints, in the order `root, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, torso, neck, nose, head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist`.
- Use `normalize_2d_motionbert()` from [data_processing.py](src/data/data_processing.py) to build `keypoints_2d_processed` from pixel keypoints. `coco_to_canonical()` in [demo.py](src/demo/demo.py) converts COCO-17 detections.

```python
import numpy as np
d = np.load("datasets/haa4d/split/test/air_guitar_001.npz", allow_pickle=True)
d["keypoints_2d_processed"].shape   # (T, 4, 17, 3)
d["view_mask"].shape                # (T, 4)
```

---

## Repository layout

```
conf/                     dataset splits, joint masks, skeleton connectivity
src/data/                 synthetic multi-view data generation and splitting
src/models/               DSTformer backbone and STA-TFM multi-view fusion model
src/train.py, infer.py    training and inference entry points
src/eval/                 metrics, robustness test generation, visualization
src/demo/                 raw-video demo (MMPose -> STA-TFM)
tests/                    CPU-only sanity tests
datasets/, weights/, results/   created locally, git-ignored
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run commands from the repository root with `python -m src.<module>`, not `python src/<file>.py`.
- **`No trainable sequences ... under <path>`**: the training folder is empty or the path is wrong. Run §3.2–3.3 first or pass `--dataset_path`.
- **DataLoader pickling errors on macOS/Windows**: try `--num_workers 0`.
- **CUDA out of memory**: lower `--batch_size`. The paper used 16 on a 16 GB GPU.
- **MMCV build fails**: the demo stack builds MMCV from source and needs a CUDA toolkit that matches `torch.version.cuda`. The setup script installs one into the env. The core pipeline (§3–5) does not need MMCV.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{kamel2026statfm,
  title={STA-TFM: Spatio-Temporal Aggregation Across Views TransForMer for Pose Estimation},
  author={Kamel, Mena and Won, Natalie and Sarangi, Amrut and Jager, Sven and Pla Planas, Albert},
  journal={arXiv preprint arXiv:2609.24482},
  year={2026}
}
```

## License

Released under a non-commercial, academic-research license. See [LICENSE.txt](LICENSE.txt). For commercial use, please contact Sanofi.
