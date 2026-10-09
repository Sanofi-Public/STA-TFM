# Losses
import torch


def masked_mpjpe(pred, gt, valid_mask):
    """
    pred, gt: [B,T,J,3]
    valid_mask: [B,T,J] (bool)
    returns mean L2 over valid joints
    """
    m = valid_mask.unsqueeze(-1)  # [B,T,J,1]
    diff = torch.where(m, pred - gt, torch.zeros_like(pred))  # zeros where invalid
    sq = (diff**2).sum(-1).sqrt()  # [B,T,J]
    m = valid_mask
    denom = m.sum().clamp_min(1)
    return (sq * m).sum() / denom


def masked_mpjve(pred, gt, valid_mask):
    """
    pred, gt: [B, T, J, 3]
    valid_mask: [B, T, J] (bool)
    Returns mean L2 velocity difference over valid joints
    """
    # Compute velocities along the temporal axis
    v_pred = pred[:, 1:] - pred[:, :-1]  # [B, T-1, J, 3]
    v_gt = gt[:, 1:] - gt[:, :-1]  # [B, T-1, J, 3]

    # Update mask for valid frames (both current & next frame valid)
    v_mask = valid_mask[:, 1:] & valid_mask[:, :-1]  # [B, T-1, J]
    v_mask = v_mask.unsqueeze(-1)

    # Compute L2 velocity difference
    diff = torch.where(v_mask, v_pred - v_gt, torch.zeros_like(v_pred))
    sq = diff.pow(2).sum(-1).sqrt()  # [B, T-1, J]

    denom = v_mask.sum().clamp_min(1)
    return (sq * v_mask.squeeze(-1)).sum() / denom


def procrustes_align(
    gt_pose: torch.Tensor, pred_pose: torch.Tensor, mask: torch.Tensor = None
):
    """
    Apply Procrustes alignment (rotation + translation + scale) to align predicted pose to ground truth.

    Args:
        gt_pose:   [N,3] ground truth joints (torch.Tensor)
        pred_pose: [N,3] predicted joints (torch.Tensor)
        mask:      [N] boolean mask (optional) → valid joints only

    Returns:
        pred_aligned: [N,3] aligned prediction
        R: [3,3] rotation matrix
    """
    if mask is not None:
        gt_pose = gt_pose[mask]
        pred_pose = pred_pose[mask]

    # Center
    mu_gt = gt_pose.mean(dim=0, keepdim=True)
    mu_pred = pred_pose.mean(dim=0, keepdim=True)
    gt_centered = gt_pose - mu_gt
    pred_centered = pred_pose - mu_pred

    # Cross-covariance
    H = pred_centered.T @ gt_centered
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Isotropic scale
    s = (S.sum() / (pred_centered.pow(2).sum())).clamp(min=1e-8)

    # Apply transform
    pred_aligned = (pred_centered @ R) * s + mu_gt

    return pred_aligned, R


@torch.no_grad()
def p_mpjpe(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
    """
    Procrustes-aligned MPJPE.
    Args:
        pred: [B,T,J,3] predicted poses
        gt:   [B,T,J,3] ground truth poses
        mask: [B,T,J] boolean mask (optional)
    Returns:
        mpjpe_total: scalar tensor
        mpjpe_joint: [J] tensor
    """
    B, T, J, _ = pred.shape
    aligned = torch.zeros_like(pred)

    for b in range(B):
        for t in range(T):
            joint_mask = mask[b, t] if mask is not None else None
            P_aligned, _ = procrustes_align(gt[b, t], pred[b, t], joint_mask)
            aligned[b, t] = P_aligned

    # Compute MPJPE on aligned poses
    errors = torch.norm(gt - aligned, dim=-1)  # [B,T,J]
    if mask is not None:
        errors = errors * mask  # zero-out invalid joints
        denom = mask.sum().clamp(min=1)
    else:
        denom = errors.numel()

    mpjpe_total = errors.sum() / denom
    mpjpe_joint = errors.sum(dim=(0, 1)) / denom  # mean per joint

    return mpjpe_total


import numpy as np
from scipy.linalg import orthogonal_procrustes


class Inference_Loss:
    """
    Compute evaluation metrics for 3D human pose estimation after inference.
    No masking (assumes all joints are valid).
    """

    def __init__(self):
        pass

    def _ensure_batch(self, pred, gt):
        """Ensure tensors are [B,T,J,3]."""
        if pred.ndim == 3:  # [T,J,3]
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        return pred, gt

    def align_poses(self, pred, gt, joints_to_eval=None, confidence=None, confidence_threshold=0.1):
        if joints_to_eval is None:
            return (pred, gt)
        ix = 0
        joints_to_keep = []
        for _, included in joints_to_eval.items():
            if included:
                joints_to_keep.append(ix)
            ix += 1
        if pred.shape[1] == gt.shape[1] == len(joints_to_keep):
            aligned_pred = pred
            aligned_gt = gt
        else:
            aligned_pred = pred
            aligned_gt = gt
            if pred.shape[1] == len(joints_to_eval):
                aligned_pred = pred[:, joints_to_keep]
            if gt.shape[1] == len(joints_to_eval):
                aligned_gt = gt[:, joints_to_keep]
        
        # Apply confidence masking if provided
        if confidence is not None:
            # Filter confidence to match aligned joints
            if confidence.shape[1] == len(joints_to_eval):
                aligned_confidence = confidence[:, joints_to_keep]
            else:
                aligned_confidence = confidence
            
            # Mask out low-confidence joints
            low_conf_mask = aligned_confidence < confidence_threshold
            aligned_gt = aligned_gt.copy()
            aligned_gt[low_conf_mask] = np.nan
        
        return (aligned_pred, aligned_gt)

    def mpjpe(self, pred, gt):
        """Calculate Mean Per Joint Position Error"""

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")

        gt = np.array(gt, dtype=float)
        pred = np.array(pred, dtype=float)
        # Calculate L2 distance for each joint
        joint_errors = np.linalg.norm(gt - pred, axis=2)  # (num_frames, num_joints)

        # Mean across all joints and frames
        mpjpe = np.nanmean(joint_errors)
        
        # Per-joint MPJPE
        per_joint_mpjpe = np.nanmean(joint_errors, axis=0)
        
        return mpjpe, per_joint_mpjpe
    
    def compute_frame_errors(self, pred, gt, correct_scale=False):
        """
        Compute per-frame MPJPE for already Procrustes-aligned poses.
        
        Args:
            pred: [T, J, 3] already aligned predictions
            gt: [T, J, 3] ground truth
            
        Returns:
            frame_errors: [T] array of MPJPE per frame
        """
        num_frames = gt.shape[0]
        aligned_poses = np.zeros_like(pred)

        # Align each frame individually
        for frame_idx in range(num_frames):
            aligned_poses[frame_idx], _ = self.procrustes_align(
                gt[frame_idx], pred[frame_idx], correct_scale
            )
        # Compute per-frame MPJPE: [T, J, 3] -> [T, J] -> [T]
        frame_errors = np.sqrt(np.sum((aligned_poses - gt) ** 2, axis=-1)).mean(axis=-1)
        return frame_errors

    def p_mpjpe(self, pred, gt, correct_scale=False):
        num_frames = gt.shape[0]
        aligned_poses = np.zeros_like(pred)

        # Align each frame individually
        for frame_idx in range(num_frames):
            aligned_poses[frame_idx], _ = self.procrustes_align(
                gt[frame_idx], pred[frame_idx], correct_scale
            )

        # Calculate MPJPE on aligned poses
        pmpjpe, per_joint_pmpjpe = self.mpjpe(pred=aligned_poses, gt=gt)

        return pmpjpe, per_joint_pmpjpe

    def procrustes_align(self, gt_pose, pred_pose, correct_scale=False):
        """Apply Procrustes alignment to align predicted pose to ground truth"""
        # Mask out invalid joints
        gt_pose = np.array(gt_pose, dtype=float)
        pred_pose = np.array(pred_pose, dtype=float)
        valid_mask = np.isfinite(gt_pose).all(axis=1) & np.isfinite(pred_pose).all(
            axis=1
        )

        if valid_mask.sum() < 3:
            return pred_pose.copy(), np.eye(3)

        gt_valid = gt_pose[valid_mask]
        pred_valid = pred_pose[valid_mask]

        # Center the valid points
        mu_gt = gt_valid.mean(axis=0, keepdims=True)
        mu_pred = pred_valid.mean(axis=0, keepdims=True)
        gt_centered = gt_valid - mu_gt
        pred_centered = pred_valid - mu_pred

        # Rotation via Orthogonal Procrustes
        R, _ = orthogonal_procrustes(pred_centered, gt_centered)

        # Isotropic scale
        s = np.linalg.norm(gt_centered) / (np.linalg.norm(pred_centered) + 1e-8)

        # Apply transform to the entire skeleton
        pred_aligned = (pred_pose - mu_pred) @ R * s + mu_gt

        if correct_scale:
            pred_aligned = self.correct_anisotropic_scale(pred_aligned, gt_pose)
        return pred_aligned, R

    def p_PCK(self, pred, gt, alpha=0.2, correct_scale=False):
        """Calculate Procrustes aligned PCK"""
        gt = np.array(gt, dtype=float)
        pred = np.array(pred, dtype=float)
        num_frames = gt.shape[0]
        aligned_poses = np.zeros_like(pred)

        # Align each frame individually
        for frame_idx in range(num_frames):
            aligned_poses[frame_idx], _ = self.procrustes_align(
                gt[frame_idx], pred[frame_idx], correct_scale
            )

        pred = aligned_poses

        # Torso Diameter
        torso_diameter = np.linalg.norm(gt[:, 1] - gt[:, 11], axis=1)
        threshold = alpha * torso_diameter[:, None]

        # Calculate MPJPE
        joint_errors = np.linalg.norm(gt - pred, axis=2)
        correct = joint_errors < threshold
        pck = np.nanmean(correct)
        per_joint_pck = np.nanmean(correct, axis=0)

        return pck, per_joint_pck

    def PCK(self, pred, gt, alpha=0.2):
        """Calculate PCK"""
        # Torso Diameter
        gt = np.array(gt, dtype=float)
        pred = np.array(pred, dtype=float)
        torso_diameter = np.linalg.norm(gt[:, 1] - gt[:, 11], axis=1)
        threshold = alpha * torso_diameter[:, None]

        # Calculate MPJPE
        joint_errors = np.linalg.norm(gt - pred, axis=2)
        correct = joint_errors < threshold
        pck = np.nanmean(correct)
        per_joint_pck = np.nanmean(correct, axis=0)
        return pck, per_joint_pck

    def mpjve(self, pred, gt):
        """Calculate Mean Per Joint Velocity Error"""
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")

        gt = np.array(gt, dtype=float)
        pred = np.array(pred, dtype=float)

        # Calculate Velocities
        v_pred = np.diff(pred, axis=0)
        v_gt = np.diff(gt, axis=0)

        # Calculate L2 distance for each joint
        joint_errors = np.linalg.norm(v_gt - v_pred, axis=2)  # (num_frames, num_joints)

        # Mean across all joints and frames
        mpjve = np.nanmean(joint_errors)

        # Per-joint MPJVE
        per_joint_mpjve = np.nanmean(joint_errors, axis=0)

        return mpjve, per_joint_mpjve

    def p_mpjve(self, pred, gt, correct_scale=False):
        """Calculate Mean Per Joint Velocity Error"""
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")

        gt = np.array(gt, dtype=float)
        pred = np.array(pred, dtype=float)

        num_frames = gt.shape[0]
        aligned_poses = np.zeros_like(pred)

        # Align each frame individually
        for frame_idx in range(num_frames):
            aligned_poses[frame_idx], _ = self.procrustes_align(
                gt[frame_idx], pred[frame_idx], correct_scale
            )

        pred = aligned_poses

        # Calculate Velocities
        v_pred = np.diff(pred, axis=0)
        v_gt = np.diff(gt, axis=0)

        # Calculate L2 distance for each joint
        joint_errors = np.linalg.norm(v_gt - v_pred, axis=2)  # (num_frames, num_joints)

        # Mean across all joints and frames
        mpjve = np.nanmean(joint_errors)

        # Per-joint MPJVE
        per_joint_mpjve = np.nanmean(joint_errors, axis=0)

        return mpjve, per_joint_mpjve

    def correct_anisotropic_scale(self, pred, gt):
        """
        Calculates and applies anisotropic (per-axis) scaling to bring
        the already-aligned 'pred' data closer to 'gt'.

        Args:
            pred (np.ndarray): The prediction data, already aligned
                            via Procrustes, shape (N, K, 3).
            gt (np.ndarray): The ground truth data, shape (N, K, 3).

        Returns:
            np.ndarray: The prediction data with anisotropic scale correction applied.
        """
        # 1. Center the Data
        # Calculate the mean across all points and samples (axes 0 and 1)
        C_pred = np.nanmean(pred, axis=(0, 1))
        C_gt = np.nanmean(gt, axis=(0, 1))

        # Center both datasets
        P_centered = pred - C_pred
        Q_centered = gt - C_gt

        # Flatten the data to (N*K, 3) for vectorized calculation
        P_flat = P_centered.reshape(-1, 3)
        Q_flat = Q_centered.reshape(-1, 3)

        # 2. Calculate Scale Factors (sx, sy, sz)
        # The optimal scale factor s_i is: sum(Q'_i * P'_i) / sum(P'_i^2)

        # Numerator: Sum of products for each axis (columns of the flattened arrays)
        numerator = np.nansum(Q_flat * P_flat, axis=0)

        # Denominator: Sum of squares for each axis (variance proxy of P')
        denominator = np.nansum(P_flat**2, axis=0)

        # Prevent division by zero if an axis is degenerate (rare, but safe)
        denominator[denominator == 0] = 1e-12

        # Calculate the scale factors [sx, sy, sz]
        scale_factors = numerator / denominator

        # 3. Apply Scale and Re-center

        # Apply the scale factors element-wise to the centered prediction data.
        # NumPy broadcasting applies the (3,) array to the last dimension of P_centered.
        P_scaled = P_centered * scale_factors

        # Apply the final translation (add back the ground truth centroid C_gt)
        pred_final_anisotropic = P_scaled + C_gt

        return pred_final_anisotropic