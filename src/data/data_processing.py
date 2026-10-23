""" Processing pipeline """
import os
import glob
import yaml
import numpy as np
from tqdm import tqdm

from src.data.utils.dataset.totalcapture import TotalCaptureLoader
from src.data.utils.dataset.DHP19 import DHP19Loader, dhp19_joint_map
from src.data.utils.dataset.blendmimic import BlendMimicLoader
from src.data.utils.dataset.haa4d import HAA4DLoader
from src.data.utils.transform import transform_skeleton
from src.data.utils.projection import get_rotation_matrix, project_2d


def normalize_2d_motionbert(keypoints_2d, img_wh):
    """Normalize 2D keypoints using MotionBERT preprocessing (aspect-ratio aware).
    
    Matches the preprocessing from MotionBERT/DSTFormer pretraining on H36M:
    keypoint_labels[..., :2] = keypoint_labels[..., :2] / w * 2 - [1, h / w]
    
    This normalizes both X and Y by width (preserving aspect ratio):
    - X: (x / W) * 2 - 1  → ranges from -1 to 1
    - Y: (y / W) * 2 - (H/W) → ranges from -(H/W) to (H/W), centered at 0
    
    For 1920x1080: Y ranges from -0.5625 to 0.5625
    """
    out = keypoints_2d.copy()
    T, V, J, _ = out.shape

    for v in range(V):
        W, H = img_wh[v]
        aspect_ratio = H / W  # h/w in MotionBERT formula
        
        for t in range(T):
            # Normalize both axes by width, then apply offsets
            out[t, v, :, 0] = (out[t, v, :, 0] / W) * 2 - 1
            out[t, v, :, 1] = (out[t, v, :, 1] / W) * 2 - aspect_ratio

            root_xy = out[t, v, 0, :2]  # Root joint (index 0)
            shift = root_xy
            
            # Shift all keypoints so root is at image center
            out[t, v, :, :2] -= shift

    return out



def normalize_3d_root_centered(keypoints_3d, scale_factor=1.0):
    """Normalize 3D keypoints: root-center and scale to ~[-1, 1] range.
    
    Args:
        keypoints_3d: (T, J, 3) array in original units
        scale_factor: typical range of data (default 500 for CMU centimeters)
    
    Returns:
        (T, J, 3) array with root at origin and scaled by scale_factor
    """
    out = keypoints_3d.copy()
    T, J, _ = out.shape
    
    for t in range(T):
        root_pos = out[t, 0:1, :]  # (1, 3) - root joint
        out[t] = out[t] - root_pos  # Center at root
    
    # Scale to ~[-1, 1]
    out = out / scale_factor
    
    return out


class DataProcessor:

    def __init__(self, config: dict):
        self.cfg = config
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        # Select dataset loader
        if self.cfg["dataset_type"].lower() == "dhp19":
            self.loader_cls = DHP19Loader
            self.file_ext = "*.mat"
            self.joint_map = dhp19_joint_map
        elif self.cfg["dataset_type"].lower() == "blendmimic":
            self.loader_cls = BlendMimicLoader
            self.file_ext = "*.npz"
        elif self.cfg["dataset_type"].lower() == "haa4d":
            self.loader_cls = HAA4DLoader
            self.file_ext = "*.npy"
        elif self.cfg["dataset_type"].lower() == "totalcapture":
            self.loader_cls = TotalCaptureLoader
            self.file_ext = "*.npz"
        else:
            raise ValueError(f"Unknown dataset_type: {self.cfg['dataset_type']}")

    def process_single(self, filepath: str):
        base = os.path.splitext(os.path.basename(filepath))[0]
        if self.cfg["dataset_type"].lower() == "dhp19": #DHP19
            # Parse subject/session/task
            parts = base.split("_")
            if len(parts) != 3 or not parts[0].startswith("S"):
                return
            subject = int(parts[0][1:])
            session = int(parts[1])
            task = int(parts[2])
            loader = self.loader_cls(os.path.dirname(filepath))
            skeleton = loader.load_skeleton_data(subject, session, task)
        else: # DHP19 or Blendmimic
            loader = self.loader_cls(self.cfg["data_dir"])
            skeleton = loader.load_skeleton_data(filepath)

        # Normalize
        skeleton_norm = transform_skeleton(skeleton, scale=self.cfg["scale"])
        
        # Build array [T,17,J]
        joint_order = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'] # Second check for order
        keypoints_3d = np.stack([skeleton_norm[j].T for j in joint_order], axis=1)  # [T,17,3]

        T = keypoints_3d.shape[0]
        J = keypoints_3d.shape[1]

        # Project views
        keypoints_2d, view_mask, cam_params = [], [], []
        for view in self.cfg["views"]:
            R_cam = get_rotation_matrix(view["axis"], view["angle"], noise=view["angle_noise"])
            if view["dist_noise"]:
                dist = np.random.uniform(-view["dist_noise"],view["dist_noise"])
                cam_position = -R_cam.T @ np.array([0, 0, view["dist"]+dist])
            else:
                cam_position = -R_cam.T @ np.array([0, 0, view["dist"]])
            t_cam = -R_cam @ cam_position.reshape(3, 1)
            extrinsic = np.hstack((R_cam, t_cam))

            if view["intrinsic_noise"]:
                f_noise = np.random.uniform(-view["intrinsic_noise"][0],view["intrinsic_noise"][0])
                c_noise = np.random.uniform(-view["intrinsic_noise"][1],view["intrinsic_noise"][1])
                K = np.array([
                [view["fx"]+f_noise, 0, view["cx"]+c_noise],
                [0, view["fy"]+f_noise, view["cy"]+c_noise],
                [0, 0, 1]
            ])
            else:
                K = np.array([
                    [view["fx"], 0, view["cx"]],
                    [0, view["fy"], view["cy"]],
                    [0, 0, 1]
                ])

            if view.get("joint_noise", None) is not None: 
                joint_noise = np.random.uniform(0,view["joint_noise"])
            else:
                joint_noise = None

            # Use correct image height for Y-flip in projection
            img_h = view["cy"] * 2
            proj = project_2d(keypoints_3d, K, extrinsic, joint_noise, img_height=img_h)   # [T,J,2]
            conf = np.ones((T,J,1), dtype=np.float32)
            proj_conf = np.concatenate([proj, conf], axis=-1)   # [T,J,3]
            keypoints_2d.append(proj_conf)
            view_mask.append(np.ones((T,), dtype=bool))
            cam_params.append({"K": K, "extrinsic": extrinsic, "axis": view["axis"], "angle": view["angle"]})

        keypoints_2d = np.stack(keypoints_2d, axis=1)   # [T,V,J,3]
        # keypoints_3d = skeleton_norm.transpose(2,0,1)   # [T,J,3]
        view_mask = np.stack(view_mask, axis=1)         # [T,V]

        gt_joint_names = list(skeleton.keys())
        T = next(iter(skeleton.values())).shape[1]
        gt_world = np.full((T, len(joint_order), 3), np.nan, dtype=np.float32)

        for j_idx, j_name in enumerate(joint_order):
            if j_name in gt_joint_names:
                gt_world[:, j_idx, :] = skeleton[j_name].T

        # Apply preprocessing to match convert_pkl_to_npz_simple.py format
        img_wh_array = np.array([[v["cx"]*2, v["cy"]*2] for v in self.cfg["views"]])
        
        # Preprocess 2D predictions (MotionBERT-style: aspect-ratio aware)
        keypoints_2d_processed = normalize_2d_motionbert(keypoints_2d, img_wh_array)
        
        # Preprocess 3D GT (root-centered, scaled to [-1, 1])
        keypoints_3d_processed = normalize_3d_root_centered(keypoints_3d, scale_factor=1.0)

        # Save npz
        out_file = os.path.join(self.cfg["output_dir"], base + ".npz")
        np.savez_compressed(out_file,
            keypoints_2d=keypoints_2d.astype(np.float32),
            keypoints_2d_processed=keypoints_2d_processed.astype(np.float32),
            keypoints_3d=keypoints_3d.astype(np.float32),
            keypoints_3d_processed=keypoints_3d_processed.astype(np.float32),
            view_mask=view_mask,
            img_wh=img_wh_array,
            cam_params=cam_params,
            gt_world = gt_world
        )

    def run(self):
        cfg_path = os.path.join(self.cfg["output_dir"], "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)

        files = glob.glob(os.path.join(self.cfg["data_dir"], "**", self.file_ext), recursive=True)
        if not files:
            print(f"[Error] No {self.file_ext} files found in {self.cfg['data_dir']}")
            return

        if self.cfg.get("seed") is not None:
            np.random.seed(self.cfg["seed"])

        for file_path in tqdm(sorted(files), desc="Sequences"):
            self.process_single(file_path)

# Default synthetic camera rig used in the paper: 4 views radially distributed around the subject
DEFAULT_VIEWS = [
    {"axis":"y","angle":0,   "dist":5.0,"fx":2751,"fy":2751,"cx":1512,"cy":2016, "angle_noise":15,"dist_noise":4,"intrinsic_noise":[50,10], "joint_noise":0.01},
    {"axis":"y","angle":90,  "dist":5.0,"fx":2751,"fy":2751,"cx":1512,"cy":2016, "angle_noise":15,"dist_noise":4,"intrinsic_noise":[50,10], "joint_noise":0.01},
    {"axis":"y","angle":180, "dist":5.0,"fx":2751,"fy":2751,"cx":1512,"cy":2016, "angle_noise":15,"dist_noise":4,"intrinsic_noise":[50,10], "joint_noise":0.01},
    {"axis":"y","angle":270, "dist":5.0,"fx":2751,"fy":2751,"cx":1512,"cy":2016, "angle_noise":15,"dist_noise":4,"intrinsic_noise":[50,10], "joint_noise":0.01},
]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate synthetic multi-view .npz files from a 3D pose dataset")
    ap.add_argument("--dataset", required=True, choices=["dhp19", "haa4d", "blendmimic", "totalcapture"])
    ap.add_argument("--data_dir", required=True, help="Raw dataset dir, e.g. datasets/dhp19/raw/Vicon_data")
    ap.add_argument("--output_dir", required=True, help="Output dir for .npz files, e.g. datasets/dhp19/synthetic")
    ap.add_argument("--scale", default="local", choices=["local", "global"],
                    help="Skeleton scaling. Keep 'local' to match STA-TFM")
    ap.add_argument("--views_yaml", default=None,
                    help="Optional YAML file with a list of view dicts overriding the default 4-camera rig")
    ap.add_argument("--seed", type=int, default=42, help="Seed for camera/intrinsic/joint noise sampling")
    args = ap.parse_args()

    views = DEFAULT_VIEWS
    if args.views_yaml:
        with open(args.views_yaml) as f:
            views = yaml.safe_load(f)

    config = {
        "dataset_type": args.dataset,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "scale": args.scale,
        "seed": args.seed,
        "views": views,
    }
    DataProcessor(config).run()


if __name__ == "__main__":
    main()
