''' Run this script to train FPT (DSTformer is frozen)'''

import argparse
import random
import torch
import json
import os
import numpy as np
from functools import partial
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from datetime import datetime

from src.models.DSTformer import DSTformer
from src.utils.mv_windows import MVNPZWindows, collate_pad_V
from src.utils.mv_data import MVSequenceNPZ, mv_collate_pad
from src.utils.mv_train import build_model, select_params_for_training, train_one_epoch, evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

class MultiviewTrainer:

    def __init__(
        self,
        configs = {
            "dataset_path": str(REPO_ROOT / "datasets/haa4d_dhp19_union/split/train"), # Path to training set with .npz files
            "dstformer_path": str(REPO_ROOT / "weights/dstformer_weights.pth"), # Path to DSTFormer weights
            "model_save_dir": str(REPO_ROOT / "weights/training/haa4d_dhp19_union/checkpoints"), # Output directory for model weights
            "metrics_dir": str(REPO_ROOT / "weights/training/haa4d_dhp19_union/metrics"), # Output directory for model training metrics
            "seed": 42,
            "resume_from": None,  # Path to your best model checkpoint to resume from
            "batch_size": 16,
            "epochs": 100,
            "val_ratio": 0.1,
            "max_length": 36,
            "stride": 12,
            "num_joints": 17,
            "bottleneck": 8,
            "fusion_heads": 16,
            "fusion_depth": 12,
            "fusion_mlp": 3.0,
            "max_views": 4,
            "lr": 1e-3,
            "eta_min": 1e-5,
            "scheduler_Tmax": 10,
            "weight_decay": 1e-4,
            "loss_lambda": 1.0,
            "num_workers": 4,
            "finetune_backbone": False,
            "view_dropout": 0.1,
        },
    ):
    
        self.configs = {**configs}
        seed_everything(self.configs["seed"])
        self.date_str = datetime.now().strftime("%Y%m%d_%H%M")
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AMP = (self.device.type == "cuda") 
        self.pin = (self.device.type == "cuda")
        print("device count:", torch.cuda.device_count())
        print("name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
            

    def load_backbone(self):
        import torch.nn.functional as F
        cfg = self.configs

        # Build the DSTformer backbone exactly as during pretraining
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

        # Load weights from resume checkpoint if provided, else from pretrained DSTformer
        if cfg.get("resume_from") is not None:
            print(f"[INFO] Resuming from checkpoint: {cfg['resume_from']}")
            resume_ckpt = torch.load(cfg["resume_from"], map_location="cpu")
            
            # Load backbone weights from checkpoint
            if "backbone" in resume_ckpt:
                state_dict = resume_ckpt["backbone"]
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print("[Backbone] Loaded from resume checkpoint with missing:", missing, "unexpected:", unexpected)
            else:
                print("[WARNING] No 'backbone' key in resume checkpoint, loading from dstformer_path instead")
                ckpt = torch.load(cfg["dstformer_path"], map_location="cpu")
                state_dict = ckpt.get("model", ckpt)
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print("[Backbone] Loaded from dstformer_path with missing:", missing, "unexpected:", unexpected)
        else:
            # Load pretrained checkpoint
            ckpt = torch.load(cfg["dstformer_path"], map_location="cpu")
            state_dict = ckpt.get("model", ckpt)

            # Handle temporal embedding mismatch 
            if "temp_embed" in state_dict:
                old_embed = state_dict["temp_embed"]
                new_embed = backbone.temp_embed
                if old_embed.shape[1] != new_embed.shape[1]:
                    print(f"[Info] Interpolating temp_embed from {old_embed.shape[1]} → {new_embed.shape[1]}")
                    import torch.nn.functional as F
                    # old_embed: [1, L_old, 1, 256]
                    # Convert to [1, 256, L_old]
                    temp = old_embed.squeeze(2).permute(0, 2, 1)  # [1,256,L_old]
                    temp = F.interpolate(temp, size=new_embed.shape[1], mode="linear", align_corners=False)
                    # Back to [1, L_new, 1, 256]
                    state_dict["temp_embed"] = temp.permute(0, 2, 1).unsqueeze(2).contiguous()

            # Load the state dict safely (ignore size mismatches that are fixed above)
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            print("[Backbone] Loaded with missing:", missing, "unexpected:", unexpected)

        # Freeze or finetune as configured
        if cfg["finetune_backbone"]:
            print("[INFO] Fine-tuning DSTformer backbone (gradients enabled)")
            backbone.train()
            for p in backbone.parameters():
                p.requires_grad = True
        else:
            print("[INFO] Freezing DSTformer backbone (no gradients)")
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False

        return backbone


    def initialize_datasets(self):
        cfg = self.configs
        base = MVSequenceNPZ(cfg["dataset_path"])
        
        print(f"\n{'='*80}")
        print(f"DATASET STATISTICS")
        print(f"{'='*80}")
        print(f"Total sequences found: {len(base)}")
        print(f"Dataset path: {cfg['dataset_path']}")
        
        # Split into training and validation
        n_val = max(1, int(len(base) * cfg["val_ratio"]))
        n_train = len(base) - n_val
        train_subset, val_subset = random_split(base, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        print(f"Train sequences: {n_train} ({(1-cfg['val_ratio'])*100:.1f}%)")
        print(f"Val sequences:   {n_val} ({cfg['val_ratio']*100:.1f}%)")
        
        # Split into windows with overlap
        train_ds = MVNPZWindows(
            L=cfg["max_length"],
            stride=cfg["stride"],
            from_subset=train_subset
        )
        val_ds = MVNPZWindows(
            L=cfg["max_length"],
            stride=cfg["stride"],
            from_subset=val_subset
        )
        
        print(f"Train windows: {len(train_ds)} (window_size={cfg['max_length']}, stride={cfg['stride']})")
        print(f"Val windows:   {len(val_ds)} (window_size={cfg['max_length']}, stride={cfg['stride']})")
        print(f"{'='*80}\n")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=partial(mv_collate_pad, max_views=cfg["max_views"]),
            num_workers=cfg["num_workers"],
            pin_memory=self.pin
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=partial(mv_collate_pad, max_views=cfg["max_views"]),
            num_workers=max(1, cfg["num_workers"] // 2),
            pin_memory=self.pin
        )

        return train_loader, val_loader

    def build_model(self,backbone):
        cfg = self.configs

        model, feat_dim = build_model(
            backbone=backbone, 
            num_joints=cfg["num_joints"], 
            feat_dim=None,
            bottleneck=cfg["bottleneck"], 
            fusion_heads=cfg["fusion_heads"], 
            fusion_depth=cfg["fusion_depth"], 
            fusion_mlp=cfg["fusion_mlp"], 
            max_views=cfg["max_views"],
        )
        
        # Load fusion and head weights if resuming from checkpoint
        if cfg.get("resume_from") is not None:
            print(f"[INFO] Loading fusion and head weights from: {cfg['resume_from']}")
            resume_ckpt = torch.load(cfg["resume_from"], map_location="cpu")
            
            if "proj" in resume_ckpt:
                model.proj.load_state_dict(resume_ckpt["proj"])
                print("[INFO] Loaded proj weights from checkpoint")
            
            if "fusion" in resume_ckpt:
                model.fusion.load_state_dict(resume_ckpt["fusion"])
                print("[INFO] Loaded fusion weights from checkpoint")
            
            if "head_3d" in resume_ckpt:
                model.head_3d.load_state_dict(resume_ckpt["head_3d"])
                print("[INFO] Loaded head_3d weights from checkpoint")
        
        model.to(self.device)

        return model, feat_dim

    def train(self):
        cfg = self.configs
        backbone = self.load_backbone()
        train_loader, val_loader = self.initialize_datasets()
        model, feat_dim = self.build_model(backbone)
        # Count params
        total_params, trainable_params = count_parameters(model)
        print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
        
        # Optimizer 
        if cfg["finetune_backbone"]:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            # the modules you expect to train
            modules = [model.proj, model.fusion, model.head_3d]
            trainable_params = []
            for m in modules:
                trainable_params += [p for p in m.parameters() if p.requires_grad]

        # HARD ASSERTS (catch silent empties)
        n_all = sum(p.numel() for p in model.parameters())
        n_req = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_opt = sum(p.numel() for p in trainable_params)
        print(f"[Param counts] all={n_all:,} requires_grad={n_req:,} optimizer={n_opt:,}")
        assert n_opt > 0, "Optimizer param list is empty!"
        assert n_opt <= n_req, "Optimizer param list exceeds requires_grad params?"

        optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_Tmax"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg["scheduler_Tmax"],
            eta_min=cfg["eta_min"]
        )
        
        os.makedirs(cfg["metrics_dir"], exist_ok=True)
        with open(os.path.join(cfg["metrics_dir"], f"{self.date_str}_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        history = {"train": [], "val": []}
        best_val = 1e9
        best_epoch = 0
        
        try:
            for ep in range(1, cfg["epochs"] + 1):
                tr = train_one_epoch(
                    model, train_loader, optimizer, self.device,
                    amp=self.AMP, max_grad_norm=1.0,
                    loss_lambda=cfg["loss_lambda"]
                )
                val = evaluate(model, val_loader, self.device)
                scheduler.step()
                
                print(f"Epoch {ep:03d} | "
                      f"train_loss {tr:.3f} | "
                      f"val_mpjpe {val['mpjpe']:.3f} | "
                      f"val_p_mpjpe {val['p_mpjpe']:.3f}")
                
                if val["mpjpe"] < best_val:
                    best_epoch = ep
                    best_val = val["mpjpe"]
                    os.makedirs(cfg["model_save_dir"], exist_ok=True)
                    save_path = os.path.join(cfg["model_save_dir"],
                                             f"{self.date_str}_mv_fusion_head_best.pth")
                    torch.save({
                        "backbone": model.backbone.state_dict(),
                        "proj": model.proj.state_dict(),
                        "fusion": model.fusion.state_dict(),
                        "head_3d": model.head_3d.state_dict(),
                        "feat_dim": feat_dim,
                        "num_joints": cfg["num_joints"],
                    }, save_path)
                    
                history["train"].append(tr)
                history["val"].append(val)
                os.makedirs(cfg["metrics_dir"], exist_ok=True)
                with open(os.path.join(cfg["metrics_dir"], f"{self.date_str}_metrics.json"), "w") as f:
                    json.dump(history, f, indent=2)
            
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted; best checkpoint so far is kept.")

        print(f"BEST MODEL EPOCH: {best_epoch} (val_mpjpe {best_val:.4f})")

def parse_args():
    defaults = MultiviewTrainer.__init__.__defaults__[0]
    ap = argparse.ArgumentParser(description="Train the STA-TFM fusion head (DSTformer frozen by default).")
    ap.add_argument("--dataset_path", help="Folder with training .npz files")
    ap.add_argument("--dstformer_path", help="Pre-trained DSTformer weights (.pth)")
    ap.add_argument("--model_save_dir", help="Output directory for checkpoints")
    ap.add_argument("--metrics_dir", help="Output directory for metrics/config JSON")
    ap.add_argument("--resume_from", help="Checkpoint to resume from")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--finetune_backbone", action="store_true", default=None, help="Also fine-tune DSTformer")
    args = vars(ap.parse_args())
    return {**defaults, **{k: v for k, v in args.items() if v is not None}}


if __name__ == "__main__":
    trainer = MultiviewTrainer(configs=parse_args())
    print("Running with config:")
    for k, v in trainer.configs.items():
        print(f"  {k}: {v}")

    trainer.train()
