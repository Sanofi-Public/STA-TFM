""" Run this script to perform inference with 2D skeletons.
Match settings to the MV model."""
import argparse
import os
import glob
import json
import numpy as np
import torch
import time
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime

from src.models.DSTformer import DSTformer
from src.utils.mv_train import build_model
from src.utils.mv_losses import Inference_Loss
from src.utils.mv_inference import sliding_windows, pad_time, ensure_views, norm_2d, load_npz, save_npz

REPO_ROOT = Path(__file__).resolve().parent.parent

class MultiviewInference:
    def __init__(self,
        configs = {
            "dstformer_path": str(REPO_ROOT / "weights/dstformer_weights.pth"), # Path to DSTFormer weights
            "fusion_head_ckpt": "<path-to-STA-TFM-fusion-head-weights>", # Path to STA-TFM training weights
            "input": str(REPO_ROOT / "datasets/haa4d_dhp19_union/split/test"), # Folder with .npz files for inference
            "output": str(REPO_ROOT / "results/haa4d_dhp19_union"), # Predictions are written to <output>/<checkpoint_stem>/
            "seed": 42,  # Seed for random view dropping (view_drop_test="rand")
            "norm_2d": "none",  # Match to Model: '01' | 'none' | 'minus1_1'
            "max_length": 36,
            "stride": 12,
            "max_views": 4,
            "batch_size": 16,
            "num_joints": 17,
            "keep_meta": True,
            "num_views_test":4,  # set to 3 or 2 to test fewer views
            "view_drop_test": "rand", # Set to rand or order
            "use_gt_2d": False,  # If True, use keypoints_2d_gt_processed (preprocessed GT) instead of keypoints_2d (MMPose predictions)
        },
    ):
        self.cfg = {**configs}
        np.random.seed(self.cfg.get("seed", 42))
        torch.manual_seed(self.cfg.get("seed", 42))
        self.date_str = datetime.now().strftime("%Y%m%d_%H%M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AMP = (self.device.type == "cuda")
        print("device:", self.device)

        # Load backbone + full model
        self.backbone = self.load_backbone()
        self.model = self.build_model()

        self.mpjpe_all = []
        self.p_mpjpe_all = []
        self.pck_all = []
        self.mpjve_all = []
        self.model_name = Path(self.cfg["fusion_head_ckpt"]).stem


    def load_backbone(self):
        import torch.nn.functional as F
        cfg = self.cfg

        # Build the DSTformer backbone for the current inference window size
        backbone = DSTformer(
            dim_in=3,
            dim_feat=256,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            num_joints=cfg["num_joints"],
            maxlen=cfg["max_length"],   
            att_fuse=True
        )

        # Load pretrained checkpoint (may be longer window)
        ckpt = torch.load(cfg["dstformer_path"], map_location="cpu")
        state_dict = ckpt.get("model", ckpt)

        # Handle temporal embedding mismatch 
        if "temp_embed" in state_dict:
            old_embed = state_dict["temp_embed"]
            new_embed = backbone.temp_embed
            if old_embed.shape[1] != new_embed.shape[1]:
                print(f"[Info] Interpolating temp_embed from {old_embed.shape[1]} → {new_embed.shape[1]}")
                temp = old_embed.squeeze(2).permute(0, 2, 1)  # [1,256,L_old]
                temp = F.interpolate(temp, size=new_embed.shape[1], mode="linear", align_corners=False)
                state_dict["temp_embed"] = temp.permute(0, 2, 1).unsqueeze(2).contiguous()

        # Load weights (ignore small mismatches)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print("[Backbone] Loaded with missing:", missing, "unexpected:", unexpected)

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        return backbone


    def build_model(self):
        import torch.nn.functional as F
        cfg = self.cfg

        model, feat_dim = build_model(
            backbone=self.backbone,
            num_joints=cfg["num_joints"],
            feat_dim=256,
            bottleneck=8,
            fusion_heads=16,
            fusion_depth=12,
            fusion_mlp=3.0,
            max_views=cfg["max_views"]
        )

        ckpt = torch.load(cfg["fusion_head_ckpt"], map_location="cpu")
        if "backbone" in ckpt:
            print("Loading fine-tuned DSTFormer backbone from fusion_head_ckpt")
            state_dict = ckpt["backbone"]

            # Handle temporal embedding mismatch
            if "temp_embed" in state_dict:
                old_embed = state_dict["temp_embed"]
                new_embed = model.backbone.temp_embed
                if old_embed.shape[1] != new_embed.shape[1]:
                    print(f"[Info] Interpolating temp_embed from {old_embed.shape[1]} → {new_embed.shape[1]}")
                    temp = old_embed.squeeze(2).permute(0, 2, 1)  # [1,256,L_old]
                    temp = F.interpolate(temp, size=new_embed.shape[1],
                                        mode="linear", align_corners=False)
                    state_dict["temp_embed"] = temp.permute(0, 2, 1).unsqueeze(2).contiguous()

            missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
            print("[Fusion ckpt backbone] missing:", missing, "unexpected:", unexpected)
        else:
            print("Using pretrained DSTFormer backbone from dstformer_path")

        # Load fusion & head
        if "proj" in ckpt:
            model.proj.load_state_dict(ckpt["proj"], strict=True)
        if "fusion" in ckpt:
            model.fusion.load_state_dict(ckpt["fusion"], strict=True)
        if "head_3d" in ckpt:
            model.head_3d.load_state_dict(ckpt["head_3d"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(self.device)
        return model

    def infer_sequence(self, x2d_np, view_mask_np):
        cfg = self.cfg
        x2d_np, view_mask_np = ensure_views(x2d_np, view_mask_np, max_views=cfg["max_views"])
        # x2d_np[:,:,:,-1] = 1 # REMOVE THIS
        T, V = view_mask_np.shape
        J = x2d_np.shape[2]

        if T <= cfg["max_length"]:
            x2d_pad, offset = pad_time(x2d_np, cfg["max_length"])
            vm_pad, _ = pad_time(view_mask_np, cfg["max_length"])
            x2d_t = torch.from_numpy(x2d_pad).float().unsqueeze(0).to(self.device)
            vm_t  = torch.from_numpy(vm_pad).bool().unsqueeze(0).to(self.device)
            ctx = torch.amp.autocast(device_type="cuda") if self.AMP else nullcontext()
            with torch.no_grad(), ctx:
                pred = self.model(x2d_t, view_mask=vm_t)
            return pred.squeeze(0).cpu().numpy()[offset:offset+T]

        # Sliding windows
        windows = sliding_windows(T, cfg["max_length"], cfg["stride"])
        out_accum = np.zeros((T, J, 3), dtype=np.float32)
        out_count = np.zeros((T, 1, 1), dtype=np.float32)

        def batches(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        ctx = torch.amp.autocast(device_type="cuda") if self.AMP else nullcontext()
        for batch_wins in batches(windows, cfg["batch_size"]):
            x_batch = np.stack([x2d_np[s:e] for (s,e) in batch_wins], axis=0)
            m_batch = np.stack([view_mask_np[s:e] for (s,e) in batch_wins], axis=0)
            x_t = torch.from_numpy(x_batch).float().to(self.device)
            m_t = torch.from_numpy(m_batch).bool().to(self.device)
            with torch.no_grad(), ctx:
                pred = self.model(x_t, view_mask=m_t)
            pred = pred.cpu().numpy()
            for b, (s, e) in enumerate(batch_wins):
                out_accum[s:e] += pred[b]
                out_count[s:e] += 1.0

        out_count[out_count == 0] = 1.0
        return out_accum / out_count

    def run(self):
        cfg = self.cfg
        input_path = cfg["input"]
        if os.path.isdir(input_path):
            all_npz = sorted(glob.glob(os.path.join(input_path, "*.npz")))
            in_paths = [p for p in all_npz if not p.endswith(".pred3d.npz")]
        else:
            in_paths = [input_path]
        
        total_frames = 0
        start_time = time.time()

        for i, in_path in enumerate(in_paths):
            print(f"[{i+1}/{len(in_paths)}] {in_path}")
            x2d, vm, meta_all, x3d = load_npz(in_path, norm_mode=cfg["norm_2d"], use_gt_2d=cfg.get("use_gt_2d", False))
            if cfg.get("use_gt_2d", False):
                print(" -> Using ground truth 2D keypoints (Oracle mode)")
            # Mask out extra views
            if cfg.get("num_views_test", cfg["max_views"]) < cfg["max_views"]:
                n_keep = cfg["num_views_test"]
                V = x2d.shape[1]
                if n_keep < V:
                    if cfg.get("view_drop_test") == "rand":         
                        # Option 2: randomly choose which views to keep
                        keep = np.sort(np.random.choice(V, n_keep, replace=False))
                        drop = [v for v in range(V) if v not in keep]
                        x2d[:, drop, :, :] = 0.0
                        vm[:, drop] = False
                    else:
                        # Option 1: keep the first n_keep views
                        x2d[:, n_keep:, :, :] = 0.0
                        vm[:, n_keep:] = False

                    print(f"Masked out {V - n_keep} views -> using {n_keep} active views")
            if x2d is None:
                print(" -> skip (no keypoints)")
                continue
            T = x2d.shape[0]           
            total_frames += T
            pred3d = self.infer_sequence(x2d, vm)

            # Save
            base = os.path.splitext(os.path.basename(in_path))[0]
            fname = Path(cfg["fusion_head_ckpt"]).stem
            output_location = os.path.join(cfg["output"], fname)
            os.makedirs(output_location, exist_ok=True)
            out_path = os.path.join(output_location, f"{base}.pred3d.npz")

            meta = {}
            if cfg["keep_meta"] and meta_all is not None:
                for k in meta_all.files:
                    if k not in ("keypoints_3d", "view_mask"):
                        meta[k] = meta_all[k]

            save_npz(out_path, pred3d, vm, keypoints_2d=None, keypoints_3d=x3d,
                     meta=meta)
            print(f" -> saved {out_path}")
        
        elapsed = time.time() - start_time   
        if total_frames > 0:                 
            fps = total_frames / elapsed     
            spf = elapsed / total_frames    
            print("\n=== Inference Speed ===")      
            print(f"Total frames: {total_frames}")   
            print(f"Elapsed time: {elapsed:.3f} s")  
            print(f"Throughput: {fps:.2f} frames/s") 
            print(f"Per-frame:  {spf*1000:.3f} ms/frame ({spf:.6f} s/frame)")  

def parse_args():
    defaults = MultiviewInference.__init__.__defaults__[0]
    ap = argparse.ArgumentParser(description="Run STA-TFM inference on a folder (or single file) of multi-view 2D .npz files.")
    ap.add_argument("--fusion_head_ckpt", required=True, help="STA-TFM checkpoint (.pth)")
    ap.add_argument("--dstformer_path", help="Pre-trained DSTformer weights (.pth)")
    ap.add_argument("--input", help="Folder with .npz files, or a single .npz")
    ap.add_argument("--output", help="Output root; predictions go to <output>/<checkpoint_stem>/")
    ap.add_argument("--num_views_test", type=int, help="Number of active views (4, 3, 2, 1) for missing-view experiments")
    ap.add_argument("--view_drop_test", choices=["rand", "order"], help="Which views to drop when num_views_test < 4")
    ap.add_argument("--use_gt_2d", action="store_true", default=None, help="Use keypoints_2d_gt_processed (oracle 2D)")
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--seed", type=int)
    args = vars(ap.parse_args())
    return {**defaults, **{k: v for k, v in args.items() if v is not None}}


if __name__ == "__main__":
    runner = MultiviewInference(configs=parse_args())
    runner.run()
