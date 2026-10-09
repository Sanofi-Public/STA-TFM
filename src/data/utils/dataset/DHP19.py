""" Read DHP19 .mat skeletons into numpy arrays """
import os
import numpy as np
import scipy.io

dhp19_joint_map = {
    "hipR": "RHip",
    "kneeR": "RKnee",
    "footR": "RAnkle",
    "hipL": "LHip",
    "kneeL": "LKnee",
    "footL": "LAnkle",
    "shoulderR": "RShoulder",
    "elbowR": "RElbow",
    "handR": "RWrist",
    "shoulderL": "LShoulder",
    "elbowL": "LElbow",
    "handL": "LWrist",
    "head": "head",
}

class DHP19Loader:
    def __init__(self, data_dir: str, joint_map: dict=dhp19_joint_map):
        self.data_dir = data_dir
        self.joint_map = joint_map

    def load_skeleton_data(self, subject: int, session: int, task: int) -> dict:
        """Load skeleton data for given subject/session/task as dict of arrays"""
        filename = os.path.join(self.data_dir, f"S{subject}_{session}_{task}.mat")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        mat = scipy.io.loadmat(filename, squeeze_me=True)
        xyzpos = mat.get("XYZPOS", None)
        if xyzpos is None:
            raise ValueError(f"No 'XYZPOS' in {filename}")

        skeleton_data = {}
        for mat_name, canon_name in dhp19_joint_map.items():
            if mat_name in xyzpos.dtype.names:
                skeleton_data[canon_name] = xyzpos[mat_name].item().T 
        return skeleton_data
