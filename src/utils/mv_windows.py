''' Slice the videos to be compatible with MotionBERT'''
import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from typing import List, Optional, Union

class MVNPZWindows(Dataset):
    """
    Map-style windowed dataset compatible with random_split.
    Each __getitem__ returns a single window:
      x2d: [L, V, J, 3]
      x3d: [L, J, 3]
      view_mask: [L, V]
      valid3d: [L, J]
      path: str
    """
    def __init__(self,
                 L: int = 243,
                 stride: int = 81,
                 from_subset: Optional[Dataset] = None,
                 from_base: Optional[Dataset] = None,
                 files: Optional[List[str]] = None,
                 root: Optional[str] = None):
        super().__init__()
        self.L = L
        self.S = stride

        # Files
        if from_subset is not None:
            # Subset of MVSequenceNPZ
            base = from_subset.dataset
            assert hasattr(base, "files"), "Base dataset must expose .files"
            self.files = [base.files[i] for i in from_subset.indices]
        elif from_base is not None:
            assert hasattr(from_base, "files"), "Base dataset must expose .files"
            self.files = list(from_base.files)
        elif files is not None:
            self.files = list(files)
        elif root is not None:
            self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        else:
            raise ValueError("Provide one of: from_subset, from_base, files, or root")

        if not self.files:
            raise RuntimeError("No .npz files found for MVNPZWindows")

        self.index = []
        self._lens_cache = []  # store (T,V,J) for quick checks
        for fi, p in enumerate(self.files):
            with np.load(p, allow_pickle=True) as z:
                kp_key = "keypoints_2d_processed" if "keypoints_2d_processed" in z else "keypoints_2d"
                if kp_key not in z:
                    continue
                T, V, J, _ = z[kp_key].shape
            self._lens_cache.append((T, V, J))
            # compute tail pad so last window fits
            pad = (self.L - (T - self.L) % self.S - self.L) % self.S
            T_eff = T + pad
            for s in range(0, T_eff - self.L + 1, self.S):
                e = s + self.L
                self.index.append((fi, s, e, T, pad))

        if not self.index:
            raise RuntimeError("No valid windows built (check your files and L/stride)")

    def __len__(self):
        return len(self.index)

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
        
        W = np.where(img_wh[:, 0] <= 0, 1, img_wh[:, 0]).reshape(1, -1, 1)
        H = np.where(img_wh[:, 1] <= 0, 1, img_wh[:, 1]).reshape(1, -1, 1)
        
        x = x2d[..., 0] / W
        y = x2d[..., 1] / H
        c = x2d[..., 2]
        
        if mode == 'minus1_1':
            x = x * 2 - 1
            y = y * 2 - 1
        
        return np.stack([x, y, c], axis=-1)

    def __getitem__(self, idx):
        fi, s, e, T_orig, pad = self.index[idx]
        path = self.files[fi]
        z = np.load(path, allow_pickle=True)
        
        # Load preprocessed data (already normalized to [-1, 1])
        x2d = z["keypoints_2d_processed"].astype(np.float32)  # Already [-1,1]
        x3d = z["keypoints_3d_processed"].astype(np.float32)  # Already root-centered/500
        vmsk = z["view_mask"].astype(bool)                    # [T,V]
        img_wh = z["img_wh"].astype(np.int32) if "img_wh" in z else None
            
        V, J = x2d.shape[1], x2d.shape[2]
        valid3d = np.isfinite(x3d).all(-1)                # [T,J]

        if e > T_orig:
            # resample tail region
            needed = e - s
            if T_orig - s < self.L:
                idx = np.linspace(s, T_orig - 1, needed, dtype=int)
            else:
                idx = np.arange(s, e)
        else:
            idx = np.arange(s, e)
        x2d = x2d[idx]
        x3d = x3d[idx]
        vmsk = vmsk[idx]
        valid3d = valid3d[idx]

        clip2d = torch.from_numpy(x2d)       # [L,V,J,3]
        clip3d = torch.from_numpy(x3d)       # [L,J,3]
        clipvm = torch.from_numpy(vmsk)      # [L,V]
        clipv3 = torch.from_numpy(valid3d)   # [L,J]

        # Cap to 4 views
        K = 4  # max views
        Vcur = clip2d.shape[1]
        if Vcur > K:
            keep = torch.arange(K)
            keep, _ = torch.sort(keep)                              # keep original order

            clip2d = clip2d[:, keep]                                # [L,K,J,3]
            clipvm = clipvm[:, keep]                                # [L,K]

        return {
            "x2d": clip2d,
            "x3d": clip3d,
            "view_mask": clipvm,
            "valid3d_mask": clipv3,
            "path": path,
            "T_orig": T_orig,
        }


def collate_pad_V(batch, max_views=None):
    B = len(batch)
    L = batch[0]["x3d"].shape[0]
    J = batch[0]["x3d"].shape[1]
    V_max = max(b["x2d"].shape[1] for b in batch)

    if max_views is not None:
        V_max = min(V_max, max_views)

    x2d = torch.zeros(B, L, V_max, J, 3, dtype=torch.float32)
    x3d = torch.zeros(B, L, J, 3, dtype=torch.float32)
    view_mask = torch.zeros(B, L, V_max, dtype=torch.bool)
    valid3d = torch.zeros(B, L, J, dtype=torch.bool)
    paths = []

    for i, b in enumerate(batch):
        V = b["x2d"].shape[1]
        V_eff = min(V, V_max)

        x2d[i, :, :V_eff] = torch.as_tensor(b["x2d"][:, :V_eff])
        x3d[i] = torch.as_tensor(b["x3d"])
        view_mask[i, :, :V_eff] = torch.as_tensor(b["view_mask"][:, :V_eff])
        valid3d[i] = torch.as_tensor(b["valid3d_mask"])
        paths.append(b["path"])

    return {
        "x2d": x2d,
        "x3d": x3d,
        "view_mask": view_mask,
        "valid3d_mask": valid3d,
        "meta": paths,
    }
