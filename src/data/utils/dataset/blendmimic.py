""" Read BlendMimic .npz skeletons into numpy arrays """
import os
import numpy as np

JOINT_ORDER = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

class BlendMimicLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_skeleton_data(self, npz_file: str) -> dict:
        """Load skeleton data from BlendMimic .npz file as dict of arrays"""
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"File not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        if "positions_3d" in data:
            arr = data["positions_3d"]
        else:
            arr = data[list(data.keys())[0]]

        if arr.ndim != 3 or arr.shape[1] != len(JOINT_ORDER):
            raise ValueError(f"Expected shape (T,17,3), got {arr.shape}")

        # Build dict {joint_name: [T,3]}
        skeleton_data = {
            joint: arr[:, j, :].T for j, joint in enumerate(JOINT_ORDER)
        }
        return skeleton_data

    def load_2D_keypoints(self, npz_file: str) -> np.ndarray:
        """Load 2D keypoints from BlendMimic .npz file"""
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"File not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        cam_keys = sorted([k for k in data.keys() if k.startswith("Cam_")])
        if not cam_keys:
            raise ValueError(f"No 'Cam_*' arrays found in {npz_file}")

        cam_arrays = [data[k].astype(np.float32) for k in cam_keys]
        for c in cam_arrays:
            if c.ndim != 3 or c.shape[1:] != (len(JOINT_ORDER), 2):
                raise ValueError(f"Expected (T,17,2) for all cameras, got {c.shape}")
        keypoints_2d = np.stack(cam_arrays, axis=1)

        # Add dummy confidence = 1.0
        conf = np.ones((*keypoints_2d.shape[:3], 1), dtype=np.float32)
        keypoints_2d = np.concatenate([keypoints_2d, conf], axis=-1)  # [T, V, 17, 3]

        #Swap views to make them ordered
        order =  [3, 2, 0, 1]
        if keypoints_2d.shape[1] == 4:
            keypoints_2d = keypoints_2d[:, order]

        return keypoints_2d

    def load_cameras(self, subject="S1"):
        """
        Loads camera intrinsics/extrinsics in the same structure
        as synthetic cam_params.
        """
        cam_file = os.path.join(
            self.data_dir, "..", "Cameras", f"matrices_{subject}.npz"
        )
        cam_file = os.path.abspath(cam_file)

        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Camera file not found: {cam_file}")

        cam = np.load(cam_file, allow_pickle=True)
        K_all = cam["K"]
        RT_all = cam["RT"]
        
        # Reorder to match 2d keypoints
        order = [3, 2, 0, 1]
        K_all = K_all[order]
        RT_all = RT_all[order]

        cameras = []
        for v in range(K_all.shape[0]):
            cameras.append({
                "K": K_all[v],
                "extrinsic": RT_all[v],
                "axis": None,
                "angle": None
            })
        return cameras
