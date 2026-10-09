""" Read HAA4D .npy skeletons into numpy arrays """
import os
import numpy as np

JOINT_ORDER = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

class HAA4DLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_skeleton_data(self, npy_file: str) -> dict:
        """Load skeleton data from HAA4D .npy file as dict of arrays"""
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"File not found: {npy_file}")

        data = np.load(npy_file, allow_pickle=True)

        if data.ndim != 3 or data.shape[1] != len(JOINT_ORDER):
            raise ValueError(f"Expected shape (T,17,3), got {data.shape}")
        data = data[:, :, [0, 2, 1]] # Flip y and z
        data[:, :, 2] = -data[:, :, 2] # Flip in y

        # Build dict {joint_name: [T,3]}
        skeleton_data = {joint: data[:, j, :].T for j, joint in enumerate(JOINT_ORDER)}
        return skeleton_data

