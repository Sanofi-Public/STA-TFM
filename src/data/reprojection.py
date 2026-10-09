import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.dataset.dannceAI import dannceAILoader
from src.utils.projection import project_2d
from src.utils.conversions import mediapipe_to_canonical
from plot import plot_multiview

def load_calibration_h5(calib_path):
    """Load intrinsics (K) and extrinsics ([R|t]) for each camera."""
    cams = {}
    with h5py.File(calib_path, "r") as f:
        for cam_name in f["cameras"].keys():
            base = f[f"cameras/{cam_name}"]
            K = np.array(base["intrinsic/camera_mtx"], dtype=np.float64)
            R = np.array(base["extrinsic/rotation_mtx"], dtype=np.float64)
            t = np.array(base["extrinsic/translation_vec"], dtype=np.float64).reshape(3, 1)
            cams[cam_name] = {"K": K, "R": R, "t": t, "extrinsics": np.hstack([R, t])}
            print(K),
            print(R),
            print(t)
    return cams


class ReprojectionChecker:
    def __init__(self, cfg):
        self.data_root = cfg["data_root"]
        self.calib_path = cfg["calibration_h5"]
        self.output_dir = cfg.get("output_dir", "./reproj_check")
        os.makedirs(self.output_dir, exist_ok=True)

        self.loader = dannceAILoader()
        self.calib = load_calibration_h5(self.calib_path)

    def run(self):
        seq_dirs = glob.glob(os.path.join(self.data_root, "**/body_2DLandmarks"), recursive=True)
        for seq_dir in tqdm(seq_dirs, desc="Sequences"):
            self.check_sequence(seq_dir)

    def check_sequence(self, seq_dir):
        seq_root = os.path.dirname(seq_dir)
        seq_name = os.path.basename(seq_root)
        body_pose_path = os.path.join(seq_root, "body_pose.hdf5")

        if not os.path.exists(body_pose_path):
            print(f"[WARN] Missing 3D skeletons for {seq_name}")
            return

        skel_dict = self.loader.load_3D_keypoints(body_pose_path)
        if skel_dict is None:
            print(f"[WARN] Empty 3D file: {body_pose_path}")
            return
        joints = list(skel_dict.keys())
        x3d = np.stack([skel_dict[j].T for j in joints], axis=1)  # [T,J,3]

        cam_files = sorted(glob.glob(os.path.join(seq_dir, "Camera*.hdf5")))
        if not cam_files:
            print(f"[WARN] No Camera*.hdf5 in {seq_dir}")
            return

        print(f"Checking {seq_name} ({len(cam_files)} cams)")

        n_views = len(cam_files)
        n_frames = len(x3d)
        raw2d_all  = np.ones((n_frames, n_views, 17, 3), dtype=np.float32)
        proj2d_all = np.ones((n_frames, n_views, 17, 3), dtype=np.float32)

        for v, cam_path in enumerate(cam_files):
            cam_name = os.path.splitext(os.path.basename(cam_path))[0]
            if cam_name not in self.calib:
                print(f"[WARN] {cam_name} missing in calibration file")
                continue

            K = self.calib[cam_name]["K"]
            extrinsics = self.calib[cam_name]["extrinsics"]
            raw2d_seq = self.loader.load_2D_keypoints(cam_path)
            if raw2d_seq is None:
                continue

            n_frames = min(len(x3d), len(raw2d_seq))
            errs = []

            for i in range(n_frames):                  # (33,3)
                gt_2d_17  = mediapipe_to_canonical(raw2d_seq[i])  # (17,3)
                raw2d_all[i, v] = gt_2d_17

                # Canonical
                proj_2d_17 = project_2d(x3d[i][None, ...], K, extrinsics)[0]  # (17,2)
                proj2d_all[i, v, :, :2] = proj_2d_17

                errs.append(np.linalg.norm(gt_2d_17[:, :2] - proj_2d_17, axis=1).mean())

            mean_err = np.nanmean(errs)
            print(f"  {cam_name}: mean reprojection error = {mean_err:.2f}px")
        

        # Plot 2D keypoints and Projected Keypoints
        # Flip Y to match standard CV format
        fig = plot_multiview(k2d=[raw2d_all,proj2d_all])
        fig.show()


if __name__ == "__main__":
    config = {
        "data_root": "<path-to-dataset>",
        "calibration_h5": "<path-to-calibration-h5>",
        "output_dir": "<path-to-output-dir>",
    }
    checker = ReprojectionChecker(config)
    checker.run()
