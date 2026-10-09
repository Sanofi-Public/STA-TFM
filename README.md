# STA-TFM: Spatio-Temporal Aggregation Across Views TransForMer for Pose Estimation

## Overview
Official code implementation of **STA-TFM** ([arXiv:2609.24482](https://arxiv.org/abs/2609.24482)).
Monocular 3D human pose estimation (HPE) remains challenging due to depth ambiguity, occlusions, and the need for temporal consistency. While multi-view methods provide superior accuracy over monocular approaches, they often require complex setups. We introduce STA-TFM, a transformer-based architecture that combines spatial and temporal information for multi-view pose estimation. The approach leverages DSTformer, a monocular feature extractor, to capture long-range pose dependencies within each view. A fusion transformer then aggregates information across views to produce coherent 3D estimates. To address training data scarcity, we use a data generation pipeline that transforms any existing 3D pose dataset into multi-view setups with controllable parameters. Experiments on various datasets demonstrate that STA-TFM outperforms existing camera-parameter-free multi-view methods. STA-TFM achieves 50.9% and 49.5% reductions in mean per joint position error (MPJPE) and mean per joint velocity error (MPJVE) on the DHP19 dataset. Furthermore, it achieves 6.7% and 7.7% respective reductions on HAA4D, and a 15.2% MPJPE reduction on TotalCapture. STA-TFM handles noisy and missing 2D inputs, supporting potential deployment in healthcare monitoring, athletic assessment, and immersive technologies.

![Graphical Abstract](figures/Graphical%20Abstract.png)

---

## Installation

### 0. Download Supplementary Files

Optional step to download the supporting files provided with STA-TFM. Please download `STA-TFM Supplementary Material.zip` from Zenodo (https://zenodo.org/records/22832620), which contains pre-trained model weights, example datasets, and helper scripts.


### 1. Environment Setup

Create a conda environment with Python 3.9:

```bash
# Set environment path
export PY_ENV_PATH="/path/to/envs/sta-tfm" # path to store Conda environment

# Create and activate environment
bash setup_py_env.sh $PY_ENV_PATH
```

Or manually:

```bash
conda create --prefix "$PY_ENV_PATH" python=3.10 -y
conda activate "$PY_ENV_PATH"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
mkdir -p datasets/{dhp19,haa4d}
```

---


### 3. Synthetic Dataset Generation from DHP19 and HAA4D datasets

#### A. Downloading DHP19 dataset
Download the DHP19 dataset from:
https://sites.google.com/view/dhp19/download.


For STA-TFM, only the 'VICON_DATA' folder is relevant from the DHP19 dataset.

#### B. Downloading HAA4D dataset
Download the HAA4D dataset from:
https://cse.hkust.edu.hk/haa4d/dataset/skeletons_3d/primary_classes.zip

For STA-TFM, only the 'primary_classes.zip' folder is relevant from the HAA4D dataset.
Run the following command to unzip it
```bash 
unzip <path-to-haa4d-dataset>/primary_classes.zip -d datasets/haa4d/raw
# unzip <path-to-haa4d-dataset>/haa4d.zip -d datasets/haa4d/raw_new
```
This will extract the HAA4D dataset to <STA-TFM-repo-path>/datasets/haa4d/raw/primary_classes


#### C. Generating Synthetic Dataset 
The following steps will load each sample from the DHP19 or HAA4D dataset and generate 4 synthetic
Convert to NPZ format for training STA-TFM. 

Update the configurations in the following file:
`src/data/data_processing.py`

You can change the camera intrinsics and extrinsics accordingly. 
Intrinsics Reference: https://arxiv.org/pdf/2201.10865
It is recommended to leave the camera instrinsics/ extrinsics in their default values.

```python
config = {
    "dataset_type": "dhp19", # set for DHP19
    "data_dir": "/home/raw_data/DHP19/Vicon_data", # Update to the location where DHP19 dataset is stored
    "output_dir": "/home/processed_data/DHP19_npz", # Desired output location
    "scale": "local",
    "views": [
        {
            "axis": "y", "angle": 0, "dist": 5.0,
            "fx": 2751, "fy": 2751, "cx": 1512, "cy": 2016,
            "angle_noise": 15, "dist_noise": 4,
            "intrinsic_noise": [50, 10],
            "joint_noise": 0.01
        }, # first view
        {"axis": "y", "angle": 90, "dist": 5.0, ...}, # second view
        {"axis": "y", "angle": 180, "dist": 5.0, ...}, # third view
        {"axis": "y", "angle": 270, "dist": 5.0, ...} # fourth view
    ]
}
```

Run preprocessing:
```bash
python -m src.data.data_processing
```

Visualize the 3D skeleton and 2D camera projections using `plot.py`:
```bash
python -m src.eval.plot --npz <path-to-.npz> --output_dir <output-directory>
```
```bash
# Show raw (un-normalized) keypoints:
python -m src.eval.plot --npz <path-to-.npz> --output_dir <output-directory> --show_raw
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--npz` | Path to `.npz` file | — |
| `--output_dir` | Directory to save output HTML files | — |
| `--yaml` | Path to joint connectivity YAML | `conf/connectivities/default.yaml` |
| `--world_coord` | Use world-frame GT (`gt_world`) instead of normalized (`keypoints_3d_processed`) | `False` |
| `--show_raw` | Use raw (pixel/unnormalized) keypoints instead of processed versions | `False` |
| `--start` | Start frame index | `0` |
| `--end` | End frame index (exclusive) | end of sequence |
| `--max_frames` | Maximum number of frames to visualize | all |

**Output** (saved to `--output_dir`):
- Dataset inspection (no `pred_3d` in file): `<sample_name>.html` — GT skeleton + 2D camera views
- Inference results (file contains `pred_3d`): `<sample_name>_raw.html` and `<sample_name>_aligned.html` — 3D pred vs GT + 2D views, raw and Procrustes-aligned


**Example:**
```bash
python -m src.eval.plot \
    --npz /home/sta-tfm-helpers/datasets/haa4d/split/test/air_guitar_001.npz \
    --output_dir /home/sta-tfm-helpers/datasets/haa4d/synthetic_visuals \
    --world_coord True \
    --show_raw
```

### Generated NPZ File Format

The synthetic data generation pipeline produces `.npz` files with the following structure:

#### Data Arrays

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `keypoints_2d` | `[T, V, J, 3]` | `float32` | **Raw 2D keypoints** in pixel coordinates. Last dimension: `[x, y, confidence]` |
| `keypoints_2d_processed` | `[T, V, J, 3]` | `float32` | **Preprocessed 2D keypoints** normalized using MotionBERT-style (aspect-ratio aware). X ∈ [-1, 1], Y ∈ [-(H/W), (H/W)]. Root-centered at (0, 0). |
| `keypoints_3d` | `[T, J, 3]` | `float32` | **Transformed 3D keypoints** after applying `transform_skeleton()` with the configured scale (e.g., "local" or "global") |
| `keypoints_3d_processed` | `[T, J, 3]` | `float32` | **Preprocessed 3D keypoints** root-centered and scaled by `scale_factor` (default: 1.0 for HAA4D, 500.0 for CMU) |
| `view_mask` | `[T, V]` | `bool` | Boolean mask indicating which views have valid data for each frame |
| `img_wh` | `[V, 2]` | `int64` | Image dimensions `[width, height]` for each view (e.g., `[[3024, 4032], ...]`) |
| `cam_params` | `[V]` | `object` | List of camera parameter dicts with keys: `K` (intrinsics), `extrinsic` (4x3 [R|t]), `axis`, `angle` |
| `gt_world` | `[T, J, 3]` | `float32` | **Original 3D skeleton** in world coordinates (before `transform_skeleton()` is applied) |

**Dimensions:**
- **T**: Number of frames (temporal dimension)
- **V**: Number of views/cameras (typically 4)
- **J**: Number of joints (17 in canonical COCO format)
  - Joint order: `root, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, torso, neck, nose, head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist`

**Coordinate Systems:**
- **2D Raw**: Pixel coordinates with origin at top-left `(0, 0)`, Y-axis points down
- **2D Processed**: Normalized coordinates with root at `(0, 0)`, aspect-ratio preserved
- **3D `gt_world`**: Original skeleton data (untransformed, dataset-specific units)
- **3D `keypoints_3d`**: After `transform_skeleton()` (scaled by config: "local" or "global")
- **3D `keypoints_3d_processed`**: Root-centered, scaled to approximate [-1, 1] range for model training

**Example Usage:**
```python
import numpy as np

# Load data
data = np.load('abseiling_003.npz', allow_pickle=True)

# Access raw 2D keypoints (pixels)
kp2d_raw = data['keypoints_2d']  # [102, 4, 17, 3]

# Access preprocessed 2D keypoints (for training)
kp2d_proc = data['keypoints_2d_processed']  # [102, 4, 17, 3]

# Access 3D keypoints
kp3d_raw = data['keypoints_3d']  # [102, 17, 3]

# Check which views have valid data
view_mask = data['view_mask']  # [102, 4]
```
---

### 4. Splitting Datasets into Train/Test

After generating synthetic data, split it into train and test sets:

```bash
python -m src.data.split_dataset
```

Configure the split in [split_dataset.py](src/data/split_dataset.py):
```python
output_dir = "/path/to/output"  # Where to create train/ and test/ folders
dataset_type = "haa4d"  # Options: "haa4d", "dhp19", or "union"
```

- **`haa4d`** or **`dhp19`**: Splits a single dataset using predefined splits from [dataset_splits.json](conf/dataset_splits.json)
- **`union`**: Combines both HAA4D and DHP19 datasets into shared train/test folders

The script copies files from source directories (specified in `dataset_splits.json`) to `output_dir/train/` and `output_dir/test/`.

---

### 5. Train on Synthetic Dataset (DHP19 + HAA4D)

#### A. Download DSTFormer Weights

Download the pre-trained DSTFormer weights (feature extraction network) from Zenodo (https://zenodo.org/records/22832620):
- **File:** `dstformer_weights.pth`
- **Location:** Store in `weights/` directory

> **Note:** Ensure all model weights are stored under the `datasets/` directory.



#### B. Configure Training Paths

Update the following paths in `src/train.py`:

```python
"dataset_path": str(REPO_ROOT / "datasets/haa4d_dhp19_union/split/train"), # Path to training set with .npz files
"dstformer_path": str(REPO_ROOT / "weights/sta-tfm-paper/dstformer_weights.pth"), # Path to DSTFormer weights
"model_save_dir": str(REPO_ROOT / "weights/training/haa4d_dhp19_union/checkpoints"), # Output directory for model weights
"metrics_dir": str(REPO_ROOT / "weights/training/haa4d_dhp19_union/metrics"), # Output directory for model training metrics
```

**Optional - Weights & Biases Logging:**
- Set `"use_wandb": True` in the config
- Export your API key: `export WANDB_API_KEY=your_api_key`


#### C. Run Training

**Standard training:**
```bash
python -m src.train
```

**Headless training (background process with logging):**
```bash
TQDM_DISABLE=1 python -m src.train > output.log 2>&1 & echo $! > script.pid
```

**Monitor/control background process:**
```bash
# Check if running
ps -p $(cat script.pid)

# Stop training
kill $(cat script.pid)
```

Training logs will be saved to `output.log`, and the process ID stored in `script.pid`.


### Training Metrics

**Sample output:**
```
Epoch 001 | train_loss 0.094 | val_mpjpe 0.075 | val_p_mpjpe 0.096
```

**Metrics explained:**
- **`val_mpjpe`**: Mean Per Joint Position Error on validation set
- **`val_p_mpjpe`**: Procrustes-aligned MPJPE on validation set

Both metrics compare predicted vs. ground truth 3D keypoints (`keypoints_3d_processed`). Since skeletons are normalized to the range [-1, 1] for all axes (x, y, z), these are **unitless measurements**.

---

### 6. Run Inference on Synthetic Dataset

STA-TFM inference requires a folder with .npz files containing 2D poses from each view. For other datasets/videos, ensure your folder contains .npz files with 2D pose keypoints following the format in [Generated NPZ File Format](#generated-npz-file-format).

#### A. Configure Inference Paths
Update paths in `src/infer.py`:

```python
"dstformer_path": str(REPO_ROOT / "weights/sta-tfm-paper/dstformer_weights.pth"), # Path to DSTFormer weights
"fusion_head_ckpt": "<path-to-best-model>", # Path to STA-TFM training weights
"input": str(REPO_ROOT / "datasets/haa4d_dhp19_union/split/test"), # Folder with .npz files for inference
"output":  str(REPO_ROOT / "datasets/haa4d_dhp19_union/results/"), # Output .npz location
```

#### B. Run Inference

```bash
python -m src.infer
```

---

### 7. Visualize Inference Results

Use `evaluate.py` (see [§3C](#c-generating-synthetic-dataset) for the full argument reference). Since inference `.npz` files contain `pred_3d`, the script produces two HTML files per sample: a raw comparison and a Procrustes-aligned comparison.

```bash
cd src/eval
python evaluate.py \
    --npz <path-to-inference-results.npz> \
    --output_dir <output-directory> \
    --world_coord True
```

Use `video_overlay.py` to generate a synchronized side-by-side comparisons of 2D joint overlays on videos and 3D skeleton predictions.

```bash
python -m src.eval.video_overlay \
  --pred_npz "<path-to-inference-results.npz>" \
  --data_npz "<path-to-ground-truth.npz>" \
  --videos \
    "<path-to-video-1.mp4>" \
    "<path-to-video-2.mp4>" \
    "<path-to-video-3.mp4>" \
    "<path-to-video-4.mp4>" \
  --out "<path-to-output.mp4>" \
  --frame_skip 1
```

---

### 8. Evaluate STA-TFM Results

Run `evaluate.py` to compute quantitative metrics against ground truth across a folder of prediction `.npz` files.

```bash
python -m src.eval.evaluate \
    --pred_path   <predictions-directory> \
    --gt_path     <ground-truth-directory> \
    --output_path <output-prefix> \
    --dataset     <dhp19|haa4d|cmu|totalcapture|blendmimic3d>
```

**Example:**
```bash
python -m src.eval.evaluate \
    --pred_path   /home/sta-tfm-helpers/results/haa4d/test \
    --gt_path     /home/sta-tfm-helpers/datasets/haa4d/split/test \
    --output_path /home/sta-tfm-helpers/results/haa4d/eval \
    --dataset     haa4d
```

**Arguments:**

| Argument | Description |
|---|---|
| `--pred_path` | Folder containing prediction `.npz` files |
| `--gt_path` | Folder containing ground-truth `.npz` files |
| `--output_path` | Output file prefix for CSV results |
| `--dataset` | Dataset name — controls joint mask and unit scaling |
| `--use_gt_2d` | Use ground-truth 2D keypoints instead of predicted ones |

Joint evaluation masks per dataset are defined in [conf/joints.yaml](conf/joints.yaml).

**Output files:**
- `<output_path>_averages.csv` — mean ± std across all samples
- `<output_path>_per_sample.csv` — per-sample metrics
- `<output_path>_ap_metrics.csv` — AP and Recall at distance thresholds

**Metrics:**
- `p_mpjpe` — Procrustes-aligned MPJPE (mm)
- `p_pck` — Procrustes-aligned Percentage of Correct Keypoints
- `p_mpjve` — Procrustes-aligned Mean Per Joint Velocity Error
- `AP25/50/100/150` — Average Precision at 25/50/100/150 mm thresholds
- `Recall@500` — Recall at 500 mm threshold

---

### 9. End-to-End Demo on Raw Videos

This section walks through running STA-TFM on 4 synchronized videos from scratch and visualizing the results as an overlay video.

> **Prerequisites:** Videos should be synchronized and filmed approximately 90° apart around the subject. All commands are run from the repo root.

#### Step 1 — Run the demo

`src/demo/demo.py` handles the full pipeline automatically:
1. Runs [MMPose RTMPose](https://github.com/open-mmlab/mmpose) on each video to extract 2D keypoints frame-by-frame
2. Converts COCO-17 keypoints to the canonical 17-joint order and saves them to an intermediate `.npz`
3. Runs STA-TFM inference on the `.npz` to produce 3D poses
4. Saves an interactive HTML visualization alongside the results

```bash
python -m src.demo.demo \
    --videos \
        /home/BlendMimic3D/S1/Videos/WalkCircle_0.mp4 \
        /home/BlendMimic3D/S1/Videos/WalkCircle_1.mp4 \
        /home/BlendMimic3D/S1/Videos/WalkCircle_3.mp4 \
        /home/BlendMimic3D/S1/Videos/WalkCircle_2.mp4 \
    --dstformer_path  checkpoints/DSTFormer_weights/dstformer_weights.pth \
    --fusion_head_ckpt /home/sta-tfm-helpers/20260316_2052_mv_fusion_head_best.pth \
    --output_dir  /home/sta-tfm-helpers/demo
```

When it completes, `--output_dir` will contain:
- `<checkpoint_stem>/<name>.pred3d.npz` — 3D pose predictions with 2D keypoints and metadata embedded
- `<checkpoint_stem>/<name>_viz.html` — interactive 3D+2D Plotly visualization

| Argument | Description | Default |
|---|---|---|
| `--videos` | 4 synchronized video files (one per camera view) | required |
| `--dstformer_path` | Path to pre-trained DSTFormer backbone weights | required |
| `--fusion_head_ckpt` | Path to trained STA-TFM fusion head checkpoint | required |
| `--output_dir` | Directory to save outputs | required |
| `--name` | Stem for output file names | `demo` |
| `--mmpose_model` | MMPose model identifier | `rtmpose-m_8xb256-420e_coco-256x192` |
| `--frame_skip` | Process every N-th frame (1 = all frames) | `1` |
| `--device` | Compute device | `cuda:0` if available, else `cpu` |

#### Step 2 — Generate a video overlay

Once the demo has run, use `src.eval.video_overlay` to produce a side-by-side MP4 showing the original camera feeds with 2D skeleton overlays alongside a 3D pose panel. Since `pred3d.npz` already contains the 2D keypoints, pass it as both `--pred_npz` and `--data_npz`:

```bash
python -m src.eval.video_overlay \
    --pred_npz  /path/to/output/demo/<checkpoint_stem>/demo.pred3d.npz \
    --data_npz  /path/to/output/demo/<checkpoint_stem>/demo.pred3d.npz \
    --videos \
        /path/to/videos/cam0.mp4 \
        /path/to/videos/cam1.mp4 \
        /path/to/videos/cam2.mp4 \
        /path/to/videos/cam3.mp4 \
    --out  /path/to/output/demo/overlay.mp4
```

| Argument | Description | Default |
|---|---|---|
| `--pred_npz` | Path to `pred3d.npz` (contains `pred_3d` and `keypoints_3d`) | required |
| `--data_npz` | Path to `.npz` with `keypoints_2d` — same as `--pred_npz` for demo output | required |
| `--videos` | Camera video files in the same order used in Step 1 | required |
| `--out` | Output MP4 path | `overlay_result.mp4` |
| `--frame_skip` | Must match the `--frame_skip` value used in Step 1 so video frames stay in sync with skeleton frames | `1` |
| `--limit` | Max frames to process (`-1` = all) | `-1` |
| `--no_procrustes` | Disable Procrustes alignment of predicted poses | off |

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
