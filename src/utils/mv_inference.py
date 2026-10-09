'''Helpers for Inference'''
import os
import numpy as np
import torch

def sliding_windows(T, L, stride):
    if T <= L:
        return [(0, L)]
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] + L < T:
        starts.append(T - L)
    return [(s, s + L) for s in starts]


def pad_time(x, target_len):
    T = x.shape[0]
    if T == target_len:
        return x, 0
    if T > target_len:
        return x[:target_len], 0
    pad_front = (target_len - T) // 2
    pad_back  = target_len - T - pad_front
    pad_cfg = [(0,0)] * x.ndim
    pad_cfg[0] = (pad_front, pad_back)
    x_pad = np.pad(x, pad_cfg, mode="edge")
    return x_pad, pad_front


def ensure_views(x2d, view_mask, max_views):
    T, V, J, D = x2d.shape
    if V == max_views:
        return x2d, view_mask
    if V > max_views:
        scores = view_mask.sum(axis=0)
        keep = np.argsort(-scores)[:max_views]
        keep = np.sort(keep)
        return x2d[:, keep], view_mask[:, keep]
    pad_v = max_views - V
    pad_cfg_x = [(0,0),(0,pad_v),(0,0),(0,0)]
    pad_cfg_m = [(0,0),(0,pad_v)]
    x2d_pad = np.pad(x2d, pad_cfg_x, mode="edge")
    m_pad   = np.pad(view_mask, pad_cfg_m, mode="constant", constant_values=0)
    return x2d_pad, m_pad


def norm_2d(x2d, img_wh, mode='01'):
    if mode == 'none' or img_wh is None:
        return x2d
    if np.all(img_wh == 0):
        return x2d
    W = img_wh[:,0].reshape(1,-1,1)
    H = img_wh[:,1].reshape(1,-1,1)
    x = x2d[...,0] / W
    y = x2d[...,1] / H
    c = x2d[...,2]
    if mode == 'minus1_1':
        x = x*2 - 1
        y = y*2 - 1
    return np.stack([x,y,c], axis=-1)


def load_npz(path, norm_mode='01', use_gt_2d=False):
    """
    Load NPZ file for inference.
    
    Args:
        path: Path to NPZ file
        norm_mode: Normalization mode ('01', 'minus1_1', or 'none')
        use_gt_2d: If True, load keypoints_2d_gt_processed (preprocessed GT) instead of keypoints_2d_processed (preprocessed predictions)
                   Used for oracle/upper-bound analysis
    """
    data = np.load(path, allow_pickle=True)
    
    # Select which 2D keypoints to use (both preprocessed versions)
    kp_key = "keypoints_2d_gt_processed" if use_gt_2d else "keypoints_2d_processed"
    
    if kp_key not in data or "view_mask" not in data:
        if use_gt_2d and "keypoints_2d_gt_processed" not in data:
            print(f"Warning: keypoints_2d_gt_processed not found in {path}, falling back to keypoints_2d_processed")
            kp_key = "keypoints_2d_processed"
        if kp_key not in data or "view_mask" not in data:
            # Final fallback to raw keypoints_2d
            if "keypoints_2d" in data and "view_mask" in data:
                kp_key = "keypoints_2d"
            else:
                return None, None, None, None
    
    x2d = data[kp_key].astype(np.float32)
    vm  = data["view_mask"].astype(bool)
    img_wh = data["img_wh"].astype(np.int32) if "img_wh" in data else None
    x2d = norm_2d(x2d, img_wh, mode=norm_mode)

    if "keypoints_3d" in data:
        x3d = data["keypoints_3d"].astype(np.float32)
        return x2d, vm, data, x3d
    else:
        return x2d, vm, data, None


def save_npz(out_path, pred3d, vm, keypoints_2d=None, keypoints_3d=None, masked_mpjpe=None, p_mpjpe=None, pck=None, mpjve=None, meta=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path,
        pred_3d=pred3d.astype(np.float32),
        view_mask=vm.astype(bool),
        **({"keypoints_2d": keypoints_2d.astype(np.float32)} if keypoints_2d is not None else {}),
        **({"keypoints_3d": keypoints_3d.astype(np.float32)} if keypoints_3d is not None else {}),
        **({"masked_mpjpe": masked_mpjpe.astype(np.float32)} if masked_mpjpe is not None else {}),
        **({"p_mpjpe": p_mpjpe.astype(np.float32)} if p_mpjpe is not None else {}),
        **({"pck": pck.astype(np.float32)} if pck is not None else {}),
        **({"mpjve":mpjve.astype(np.float32)}if mpjve is not None else {}),
        **(meta or {})
    )
