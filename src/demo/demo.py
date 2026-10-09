"""
End-to-end demo: 4 synchronized videos → MMPose 2D detections → NPZ → STA-TFM inference → visualize.

Usage:
    python -m src.demo.demo \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 \
        --dstformer_path  checkpoints/DSTFormer_weights/dstformer_weights.pth \
        --fusion_head_ckpt checkpoints/FPT_weights/best.pth \
        --output_dir      /tmp/demo_out \
        [--mmpose_model rtmpose-m_8xb256-420e_coco-256x192] \
        [--device cuda:0]

"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]

from src.infer import MultiviewInference
from src.data.data_processing import normalize_2d_motionbert

# COCO-17 → canonical-17 mapping (same as convert_totalcapture_to_npz.py)
COCO_TO_CANONICAL = {
    0: 9,  1: 10, 2: 10, 3: 10, 4: 10,
    5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16,
    11: 4, 12: 1, 13: 5, 14: 2, 15: 6, 16: 3,
}


# ── 2D detection ──────────────────────────────────────────────────────────────

def extract_keypoints(video_path: str, inferencer, frame_skip: int = 1, label: str = ""):
    """Run MMPose on a video; return list of per-frame COCO-17 keypoints+scores."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = (total + frame_skip - 1) // frame_skip
    frames_kps = []
    frame_idx = 0
    desc = label or Path(video_path).name
    with tqdm(total=expected, desc=desc, unit="frame", dynamic_ncols=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                with torch.no_grad():
                    results = list(inferencer(frame, show=False, draw_bbox=False, draw_heatmap=False))
                kps, scores = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
                if results and results[0].get("predictions"):
                    preds = results[0]["predictions"][0]
                    if preds:
                        best = max(preds, key=lambda p: p.get("bbox_score", 0))
                        kps    = np.array(best["keypoints"],       dtype=np.float32)[:, :2]
                        scores = np.array(best["keypoint_scores"], dtype=np.float32)
                frames_kps.append((kps, scores))
                pbar.update(1)
            frame_idx += 1
    cap.release()
    return frames_kps


def coco_to_canonical(kps_coco, scores_coco):
    """(17,2) COCO + (17,) scores → (17,3) canonical [x,y,conf]."""
    out = np.zeros((17, 3), np.float32)
    for ci, cj in COCO_TO_CANONICAL.items():
        if ci < len(kps_coco):
            out[cj, :2] = kps_coco[ci]
            out[cj, 2]  = float(scores_coco[ci]) if ci < len(scores_coco) else 1.0

    # root = midpoint of hips
    if scores_coco[11] > 0 and scores_coco[12] > 0:
        out[0, :2] = (kps_coco[11] + kps_coco[12]) / 2
        out[0, 2]  = min(scores_coco[11], scores_coco[12])
    # torso = midpoint of shoulder-centre and hip-centre
    if scores_coco[5] > 0 and scores_coco[6] > 0:
        sh_c = (kps_coco[5] + kps_coco[6]) / 2
        out[7, :2] = (sh_c + out[0, :2]) / 2
        out[7, 2]  = min(scores_coco[5], scores_coco[6], out[0, 2])
    # neck = midpoint of shoulders
    if scores_coco[5] > 0 and scores_coco[6] > 0:
        out[8, :2] = (kps_coco[5] + kps_coco[6]) / 2
        out[8, 2]  = min(scores_coco[5], scores_coco[6])
    # head = mean of eyes/ears
    head_pts = [kps_coco[i] for i in [1,2,3,4] if scores_coco[i] > 0]
    if head_pts:
        out[10, :2] = np.mean(head_pts, axis=0)
        out[10, 2]  = 1.0
    return out


def get_img_wh(video_paths):
    """Read resolution from each video file."""
    wh = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        wh.append([w, h])
    return wh


def build_npz(video_paths, inferencer, out_npz, frame_skip=1):
    """Run MMPose on all views, assemble and save a single NPZ."""
    V = len(video_paths)
    per_view = [
        extract_keypoints(p, inferencer, frame_skip, label=f"cam {i} ({Path(p).name})")
        for i, p in enumerate(video_paths)
    ]

    T = min(len(v) for v in per_view)
    img_wh = get_img_wh(video_paths)
    img_wh_arr = np.array(img_wh, dtype=np.int64)  # (V, 2)

    # Raw 2D: [T, V, J, 3]
    kps_raw = np.zeros((T, V, 17, 3), np.float32)
    for v, frames in enumerate(per_view):
        for t in range(T):
            kps_raw[t, v] = coco_to_canonical(*frames[t])

    view_mask = (kps_raw[..., 2].mean(axis=2) > 0.3)  # [T, V]

    kps_processed = normalize_2d_motionbert(kps_raw, img_wh_arr)

    np.savez_compressed(
        out_npz,
        keypoints_2d=kps_raw,
        keypoints_2d_processed=kps_processed,
        view_mask=view_mask,
        img_wh=img_wh_arr,
    )
    print(f"Saved NPZ ({T} frames, {V} views): {out_npz}")
    return out_npz


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(npz_path, dstformer_path, fusion_head_ckpt, output_dir):
    cfg = {
        "dstformer_path": dstformer_path,
        "fusion_head_ckpt": fusion_head_ckpt,
        "input": str(Path(npz_path).parent),
        "output": output_dir,
        "norm_2d": "none",
        "max_length": 36,
        "stride": 12,
        "max_views": 4,
        "batch_size": 16,
        "num_joints": 17,
        "keep_meta": True,
        "num_views_test": 4,
        "view_drop_test": "rand",
        "use_gt_2d": False,
    }
    infer = MultiviewInference(configs=cfg)
    infer.run()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Videos → MMPose → NPZ → STA-TFM inference")
    ap.add_argument("--videos", nargs=4, required=True, metavar="VIDEO",
                    help="4 synchronized video files (one per camera view)")
    ap.add_argument("--dstformer_path",  required=True)
    ap.add_argument("--fusion_head_ckpt", required=True)
    ap.add_argument("--output_dir",      required=True)
    ap.add_argument("--mmpose_model", default="rtmpose-m_8xb256-420e_coco-256x192")
    ap.add_argument("--frame_skip", type=int, default=1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", default="demo", help="Stem for output files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: MMPose
    print(f"\n[1/3] Running MMPose ({args.mmpose_model}) on {len(args.videos)} videos...")
    if "cuda" in args.device:
        torch.backends.cudnn.benchmark = True
    inferencer = MMPoseInferencer(args.mmpose_model, device=args.device)
    npz_path = os.path.join(args.output_dir, f"{args.name}.npz")
    build_npz(args.videos, inferencer, npz_path, frame_skip=args.frame_skip)

    # Step 2: STA-TFM inference
    print("\n[2/3] Running STA-TFM inference...")
    run_inference(npz_path, args.dstformer_path, args.fusion_head_ckpt, args.output_dir)

    # Step 3: Visualize
    print("\n[3/3] Generating visualization...")
    pred_npz = next(Path(args.output_dir).rglob(f"{args.name}*.pred3d.npz"), None)
    if pred_npz:
        from src.eval.plot import load_npz, load_connectivities, build_frames
        yaml_path = str(_REPO_ROOT / "conf" / "connectivities" / "default.yaml")
        pred, gt, joint_names, k2d, view_mask, img_wh_loaded = load_npz(str(pred_npz))
        connectivities = load_connectivities(yaml_path)
        fig = build_frames(pred, gt, joint_names, connectivities,
                           k2d=k2d, view_mask=view_mask, img_wh=img_wh_loaded)
        out_html = str(pred_npz).replace(".pred3d.npz", "_viz.html")
        fig.write_html(out_html)
        print(f"Visualization saved to: {out_html}")
    else:
        print("No pred3d.npz found — skipping visualization.")


if __name__ == "__main__":
    main()
