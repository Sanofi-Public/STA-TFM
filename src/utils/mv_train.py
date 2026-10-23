# Train multiview model
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.utils.mv_data import MVSequenceNPZ, mv_collate_pad
from src.utils.mv_losses import masked_mpjpe, p_mpjpe, masked_mpjve
from src.models.model_multiview import MotionBERT_MV
from contextlib import nullcontext
from src.utils.train_monitor import step_bar, format_mem, grad_norm
import math

try:
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as _GradScaler_new
    _HAS_NEW_AMP = True
except Exception:
    _HAS_NEW_AMP = False
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler as _GradScaler_old

def make_autocast(device, enabled: bool):
    """Return a context manager for autocast, or a no-op on CPU / when disabled."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    if _HAS_NEW_AMP:
        return _autocast_new(device_type="cuda")
    else:
        return _autocast_old()

def make_scaler(device, enabled: bool):
    """Return a GradScaler compatible with your torch version, or a dummy on CPU/disabled."""
    if not enabled or device.type != "cuda":
        class _Dummy:
            def scale(self, x): return x
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return _Dummy()
    return _GradScaler_new("cuda") if _HAS_NEW_AMP else _GradScaler_old()

def build_model(backbone, num_joints=17, feat_dim=None, bottleneck=4,
                fusion_heads=4, fusion_depth=2, fusion_mlp=4.0, max_views=4):
    """
    Ensure feat_dim matches the backbone's output channels C (J-preserving).
    We'll infer it with a tiny forward if not given.
    
    NOTE: This function no longer forces the backbone to eval/frozen.
    The caller should set backbone.requires_grad before calling this.
    """
    # Store original training state
    was_training = backbone.training
    
    # Temporarily set to eval for shape inference if needed
    backbone.eval()

    # Infer feat_dim if needed
    if feat_dim is None:
        with torch.no_grad():
            dummy = torch.zeros(1, 4, num_joints, 3)  # [B,T,J,3]
            out = backbone(dummy)                     # [B,T,J,C]
            feat_dim = out.shape[-1]
    # Restore original state
    if was_training:
        backbone.train()

    # Divisibility for multi-head attention
    assert feat_dim % fusion_heads == 0, f"feat_dim({feat_dim}) % fusion_heads({fusion_heads}) != 0"
    model = MotionBERT_MV(
        backbone=backbone,
        num_joints=num_joints,
        in_dim_2d=3,
        feat_dim=feat_dim,
        bottleneck=bottleneck,
        fusion_depth=fusion_depth,
        fusion_heads=fusion_heads,
        fusion_mlp=fusion_mlp,
        max_views=max_views,
        use_time_in_fusion=False,
    )

    # Do NOT force backbone state here - let caller manage it
    return model, feat_dim

def select_params_for_training(model):
    # Only fusion + 3D head
    params = list(model.fusion.parameters()) + list(model.head_3d.parameters())
    return params

def train_one_epoch(model, loader, optimizer, device, amp=True, max_grad_norm=1.0, loss_lambda=0.0):
    model.train()
    
    # Check if backbone should be fine-tuned or frozen
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    
    if backbone_trainable:
        print("[INFO] Training with backbone fine-tuning enabled")
        model.backbone.train()
        params_to_clip = list(model.parameters())  # Include all model parameters
    else:
        print("[INFO] Training with frozen backbone")
        model.backbone.eval()
        params_to_clip = list(model.proj.parameters()) + list(model.fusion.parameters()) + list(model.head_3d.parameters())
    
    scaler = make_scaler(device, amp)

    finite_losses = []
    n_skipped_no_gt = 0
    n_skipped_nonfinite_pred = 0
    n_skipped_nonfinite_loss = 0
    n_skipped_no_views = 0 

    loop = step_bar(loader, desc="train")
    for batch in loop:
        x2d = batch["x2d"].to(device)               # [B,L,V,J,3]
        x3d = batch["x3d"].to(device)               # [B,L,J,3]
        view_mask = batch["view_mask"].to(device)   # [B,L,V]
        valid3d = batch["valid3d_mask"].to(device)  # [B,L,J]

        drop_prob = getattr(model, "view_dropout", 0.0)
        B, L, V, J, C = x2d.shape

        if drop_prob > 0:
            # Per-sequence camera dropout
            seq_mask = (torch.rand(B, V, device=device) > drop_prob).unsqueeze(1).expand(-1, L, -1)
            # Mild per-frame flicker
            flicker_mask = (torch.rand(B, L, V, device=device) > 0.01)

            # Combine with dataset-provided availability
            view_mask = (view_mask & seq_mask & flicker_mask)

            # Ensure at least ONE view per (b,l)
            dead = ~view_mask.any(dim=2)  # [B,L]
            if dead.any():
                b_idx, l_idx = dead.nonzero(as_tuple=True)
                v_idx = torch.randint(0, V, (b_idx.numel(),), device=view_mask.device)
                view_mask[b_idx, l_idx, v_idx] = True

            # Apply to inputs
            x2d = x2d * view_mask.unsqueeze(-1).unsqueeze(-1)

        # Require every frame to have >=1 view 
        has_view = view_mask.any(dim=2)   # [B,L]
        ok_view  = has_view.all(dim=1)    # [B]
        if not ok_view.all():
            keep = ok_view.nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                n_skipped_no_views += 1
                loop.set_postfix_str(
                    f"skip(noViews) x{n_skipped_no_views} | skip(noGT) x{n_skipped_no_gt} "
                    f"| skip(pred NaN) x{n_skipped_nonfinite_pred} | skip(loss NaN) x{n_skipped_nonfinite_loss}"
                )
                continue
            x2d = x2d.index_select(0, keep)
            x3d = x3d.index_select(0, keep)
            view_mask = view_mask.index_select(0, keep)
            valid3d = valid3d.index_select(0, keep)

        # keep only samples with at least one valid GT joint in the window
        has_valid = (valid3d.view(valid3d.size(0), -1).sum(dim=1) > 0)
        if not has_valid.all():
            keep = has_valid.nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                n_skipped_no_gt += 1
                loop.set_postfix_str(
                    f"skip(noViews) x{n_skipped_no_views} | skip(noGT) x{n_skipped_no_gt} "
                    f"| skip(pred NaN) x{n_skipped_nonfinite_pred} | skip(loss NaN) x{n_skipped_nonfinite_loss}"
                )
                continue
            x2d = x2d.index_select(0, keep)
            x3d = x3d.index_select(0, keep)
            view_mask = view_mask.index_select(0, keep)
            valid3d = valid3d.index_select(0, keep)

        with make_autocast(device, amp):
            pred3d = model(x2d, view_mask=view_mask)
            if not torch.isfinite(pred3d).all():
                n_skipped_nonfinite_pred += 1
                loop.set_postfix_str(
                    f"skip(noViews) x{n_skipped_no_views} | skip(noGT) x{n_skipped_no_gt} "
                    f"| skip(pred NaN) x{n_skipped_nonfinite_pred} | skip(loss NaN) x{n_skipped_nonfinite_loss}"
                )
                continue
            if loss_lambda != 0:
                mpjpe_loss = masked_mpjpe(pred3d, x3d, valid3d)

                if torch.isfinite(pred3d).all() and pred3d.size(1) > 1:
                    mpjve_loss = masked_mpjve(pred3d, x3d, valid3d)
                    if not torch.isfinite(mpjve_loss):
                        mpjve_loss = torch.tensor(0.0, device=device)
                else:
                    mpjve_loss = torch.tensor(0.0, device=device)

                loss = mpjpe_loss + loss_lambda *mpjve_loss
            else:
                loss = masked_mpjpe(pred3d, x3d, valid3d)

        if not torch.isfinite(loss):
            n_skipped_nonfinite_loss += 1
            loop.set_postfix_str(
                f"skip(noViews) x{n_skipped_no_views} | skip(noGT) x{n_skipped_no_gt} "
                f"| skip(pred NaN) x{n_skipped_nonfinite_pred} | skip(loss NaN) x{n_skipped_nonfinite_loss}"
            )
            continue

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            g_clipped = grad_norm(params_to_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            g_clipped = grad_norm(params_to_clip)
            # Mirror GradScaler (AMP path above), which skips steps with non-finite gradients
            if math.isfinite(g_clipped):
                optimizer.step()

        finite_losses.append(float(loss.detach()))
        avg = sum(finite_losses) / max(1, len(finite_losses))
        lr = optimizer.param_groups[0]["lr"]
        if loss_lambda != 0:
            loop.set_postfix_str(
                f"loss {loss.item():.3f} | avg {avg:.3f} | mpjpe {mpjpe_loss:.3f} | mpjve {mpjve_loss:.3f} | grad {g_clipped:.2f} | lr {lr:.2e}"
                f"{format_mem()} | skipGT {n_skipped_no_gt} | skipPred {n_skipped_nonfinite_pred} "
                f"| skipLoss {n_skipped_nonfinite_loss} | skipViews {n_skipped_no_views}"
            )
        else:
            loop.set_postfix_str(
                f"loss {loss.item():.3f} | avg {avg:.3f} | grad {g_clipped:.2f} | lr {lr:.2e}"
                f"{format_mem()} | skipGT {n_skipped_no_gt} | skipPred {n_skipped_nonfinite_pred} "
                f"| skipLoss {n_skipped_nonfinite_loss} | skipViews {n_skipped_no_views}"
            )

    return sum(finite_losses) / max(1, len(finite_losses))



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    model.backbone.eval()

    losses, p_losses = [], []
    n_skipped_no_views = 0
    n_skipped_no_gt = 0
    n_skipped_pred_nan = 0
    n_skipped_loss_nan = 0

    for batch in loader:
        x2d = batch["x2d"].to(device)               # [B,L,V,J,3]
        x3d = batch["x3d"].to(device)               # [B,L,J,3]
        view_mask = batch["view_mask"].to(device)   # [B,L,V]
        valid3d = batch["valid3d_mask"].to(device)  # [B,L,J]

        has_view = view_mask.any(dim=2)             # [B,L]
        ok_view  = has_view.all(dim=1)              # [B]
        if not ok_view.all():
            keep = ok_view.nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                n_skipped_no_views += 1
                continue
            x2d = x2d.index_select(0, keep)
            x3d = x3d.index_select(0, keep)
            view_mask = view_mask.index_select(0, keep)
            valid3d = valid3d.index_select(0, keep)

        # Drop samples with no GT joints in the window
        has_gt = (valid3d.view(valid3d.size(0), -1).sum(dim=1) > 0)
        if not has_gt.all():
            keep = has_gt.nonzero(as_tuple=False).squeeze(1)
            if keep.numel() == 0:
                n_skipped_no_gt += 1
                continue
            x2d = x2d.index_select(0, keep)
            x3d = x3d.index_select(0, keep)
            view_mask = view_mask.index_select(0, keep)
            valid3d = valid3d.index_select(0, keep)

        pred3d = model(x2d, view_mask=view_mask)

        if not torch.isfinite(pred3d).all():
            n_skipped_pred_nan += 1
            continue

        loss = masked_mpjpe(pred3d, x3d, valid3d)
        if not torch.isfinite(loss):
            n_skipped_loss_nan += 1
            continue
        losses.append(loss.item())

        ploss = p_mpjpe(pred3d, x3d, valid3d)
        if torch.isfinite(ploss):
            p_losses.append(ploss.item())

    mpjpe  = float(np.mean(losses))  if losses  else float("inf")
    pmpjpe = float(np.mean(p_losses)) if p_losses else float("inf")

    if not hasattr(evaluate, "_dbg"):
        print(f"[Eval] skips -> no_views:{n_skipped_no_views} no_gt:{n_skipped_no_gt} "
              f"pred_nan:{n_skipped_pred_nan} loss_nan:{n_skipped_loss_nan}")
        evaluate._dbg = True

    return {"mpjpe": mpjpe, "p_mpjpe": pmpjpe}
