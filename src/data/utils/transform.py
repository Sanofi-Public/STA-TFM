""" Skeleton alignment and normalization"""
import numpy as np

JOINT_ORDER = [
    "root","RHip","RKnee","RAnkle",
    "LHip","LKnee","LAnkle",
    "torso","neck","nose","head",
    "LShoulder","LElbow","LWrist",
    "RShoulder","RElbow","RWrist"
]

def fill_missing_joints_dict(frame_dict):
    """Fill virtual joints for one frame (dict[name] = [3,])"""
    filled = dict(frame_dict)

    if "LHip" in filled and "RHip" in filled:
        filled["root"] = 0.5*(filled["LHip"] + filled["RHip"])
    if "LShoulder" in filled and "RShoulder" in filled:
        filled["neck"] = 0.5*(filled["LShoulder"] + filled["RShoulder"])
    if "root" in filled and "neck" in filled:
        filled["torso"] = 0.5*(filled["root"] + filled["neck"])
    if "head" in filled and "neck" in filled:
        filled["nose"] = 0.3*filled["neck"] + 0.7*filled["head"]

    return filled

def transform_skeleton(skel_dict, scale="local"):
    """
    Input: dict{name -> [3,T]} from loader
    Output: dict{name -> [3,T]} with all 17 canonical joints
    """
    T = next(iter(skel_dict.values())).shape[1]

    # center on hip midpoint 
    if "LHip" in skel_dict and "RHip" in skel_dict:
        hip_mid = 0.5*(skel_dict["LHip"] + skel_dict["RHip"])
    else:
        hip_mid = np.nanmean(np.stack(list(skel_dict.values())), axis=0)

    aligned = {}
    for name, arr in skel_dict.items():
        centered = arr - hip_mid
        centered = centered[[0,2,1], :]  # swap y/z
        aligned[name] = centered

    # fill missing joints per frame 
    out = {name: np.full((3,T), np.nan, dtype=np.float32) for name in JOINT_ORDER}

    for t in range(T):
        frame_dict = {name: aligned[name][:,t] for name in aligned}
        frame_filled = fill_missing_joints_dict(frame_dict)

        for name in JOINT_ORDER:
            if name in frame_filled:
                out[name][:,t] = frame_filled[name]

    # Normalize scale 
    arr = np.stack([out[name] for name in JOINT_ORDER], axis=0)  # [17,3,T]
    lengths = np.array([np.max(np.ptp(arr[:,:,t], axis=0)) for t in range(T)])
    max_len = np.max(lengths)
    safe_lengths = np.where(lengths==0, max_len, lengths)

    if scale == "local":
        scale_factors = 1.0 / safe_lengths
        for name in JOINT_ORDER:
            out[name] = out[name] * scale_factors[np.newaxis,:]
    else:
        factor = 1.0/max_len if max_len > 0 else 1.0
        for name in JOINT_ORDER:
            out[name] = out[name] * factor


    return out
