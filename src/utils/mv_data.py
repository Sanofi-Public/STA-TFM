import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class MVSequenceNPZ(Dataset):
    """
    Expects each .npz to contain:
      view_mask:    [T, V] (bool) 1=has 2D for that frame/view
      img_wh:       [V, 2] optional (ints)
      cam_params:   dict optional
    """
    def __init__(self, root_npz):
        self.files = sorted(glob.glob(os.path.join(root_npz, "*.npz")))
        
        # Quick filter: keep only samples that have preprocessed 2D data
        kept = []
        for f in self.files:
            with np.load(f, allow_pickle=True) as z:
                has_2d = 'keypoints_2d_processed' in z
                if has_2d and z.get('keypoints_2d_processed', np.array([])).size > 0:
                    kept.append(f)
        self.files = kept
        if not self.files:
            raise RuntimeError(f"No trainable sequences (with keypoints_2d_processed) under {root_npz}")
            
    def __len__(self): return len(self.files)

    def _norm_2d(self, x2d, img_wh, mode='01'):
        """Normalize 2D keypoints by image dimensions.
        
        Args:
            x2d: [T,V,J,3] array with (x,y,conf)
            img_wh: [V,2] array with (W,H) per view
            mode: '01' for [0,1] range, 'minus1_1' for [-1,1] range, 'none' for no normalization
        
        Returns:
            Normalized [T,V,J,3] array
        """
        if mode == 'none' or img_wh is None:
            return x2d
        
        x = x2d[..., 0]
        y = x2d[..., 1]
        c = x2d[..., 2]
        W = img_wh[:, 0]  # [V]
        H = img_wh[:, 1]
        
        # Avoid div0
        W = np.where(W <= 0, 1, W)
        H = np.where(H <= 0, 1, H)
        
        # Broadcast to [T,V,J]
        Wb = W.reshape(1, -1, 1)
        Hb = H.reshape(1, -1, 1)
        
        x = x / Wb
        y = y / Hb
        
        if mode == 'minus1_1':
            x = x * 2.0 - 1.0
            y = y * 2.0 - 1.0
        
        out = np.stack([x, y, c], axis=-1)
        return out

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        
        # Load preprocessed data (already normalized to [-1, 1])
        x2d = data['keypoints_2d_processed'].astype(np.float32)  # [T,V,J,3], already [-1,1]
        x3d = data['keypoints_3d_processed'].astype(np.float32)  # [T,J,3], already root-centered/500
        vmask = data['view_mask'].astype(bool)                   # [T,V]
        img_wh = data['img_wh'].astype(np.int32) if 'img_wh' in data else None

        T, V, J, D = x2d.shape
        assert D == 3, "2D must be (x,y,conf)"
        # time mask for valid 3D joints (ignore NaN)
        valid3d = np.isfinite(x3d).all(axis=-1).astype(np.bool_)   # [T,J]

        sample = {
            "x2d": x2d,               # [T,V,J,3]
            "x3d": x3d,               # [T,J,3]
            "view_mask": vmask,       # [T,V]
            "valid3d": valid3d,       # [T,J]
            "img_wh": img_wh,         # [V,2] or None
            "path": path
        }
        return sample

def mv_collate_pad(batch, max_views=None):
    B = len(batch)
    T_max = max(b["x2d"].shape[0] for b in batch)
    V_max = max(b["x2d"].shape[1] for b in batch)

    if max_views is not None:
        V_max = min(V_max, max_views)

    J = batch[0]["x2d"].shape[2]

    x2d = torch.zeros(B, T_max, V_max, J, 3, dtype=torch.float32)
    x3d = torch.zeros(B, T_max, J, 3, dtype=torch.float32)
    view_mask = torch.zeros(B, T_max, V_max, dtype=torch.bool)
    valid3d_mask = torch.zeros(B, T_max, J, dtype=torch.bool)
    time_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        T, V = b["x2d"].shape[:2]
        V_eff = min(V, V_max)

        x2d[i, :T, :V_eff] = torch.as_tensor(b["x2d"][:, :V_eff])
        x3d[i, :T] = torch.as_tensor(b["x3d"])
        view_mask[i, :T, :V_eff] = torch.as_tensor(b["view_mask"][:, :V_eff])

        #handle either "valid3d" or "valid3d_mask"
        if "valid3d" in b:
            valid3d_mask[i, :T] = torch.as_tensor(b["valid3d"])
        elif "valid3d_mask" in b:
            valid3d_mask[i, :T] = torch.as_tensor(b["valid3d_mask"])

        time_mask[i, :T] = True

    return {
        "x2d": x2d,
        "x3d": x3d,
        "view_mask": view_mask,
        "valid3d_mask": valid3d_mask,
        "time_mask": time_mask,
        "meta": [b["path"] for b in batch],
    }
