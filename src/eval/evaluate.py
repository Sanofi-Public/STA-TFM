import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.utils.mv_losses import Inference_Loss

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JOINTS_YAML = _REPO_ROOT / "conf" / "joints.yaml"


def _load_joints_yaml(path=_JOINTS_YAML):
    with open(path, "r") as f:
        return yaml.safe_load(f)


_JOINTS = _load_joints_yaml()

# Backward-compatible module-level constants
DHP19_JOINTS = _JOINTS["dhp19"]
HAA4D_JOINTS = _JOINTS["haa4d"]
BLENDMIMIC = _JOINTS["blendmimic3d"]
CMU_JOINTS = _JOINTS["cmu"]
TOTALCAPTURE_JOINTS = _JOINTS["totalcapture"]


def get_joints_for_dataset(dataset, joints_yaml=_JOINTS_YAML):
    """Return the joint evaluation mask for the given dataset name."""
    joints = _load_joints_yaml(joints_yaml)
    if dataset not in joints:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(joints.keys())}")
    return joints[dataset]


def compute_ap_at_threshold(eval_list, total_gt, threshold):
    """
    Compute Average Precision at a given error threshold.
    
    Args:
        eval_list: List of dicts with 'mpjpe' (frame errors from sequence-aligned poses), 'score', 'gt_id'
        total_gt: Total number of ground truth frames
        threshold: Error threshold in mm
        
    Returns:
        ap: Average Precision
        recall: Final recall value
    """
    # eval_list = sorted(eval_list, key=lambda k: k["score"], reverse=True)
    total_num = len(eval_list)
    
    tp = np.zeros(total_num)
    fp = np.zeros(total_num)
    gt_det = []
    
    for i, item in enumerate(eval_list):
        if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
            tp[i] = 1
            gt_det.append(item["gt_id"])
        else:
            fp[i] = 1
            
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (total_gt + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    
    # Smooth precision curve (make it monotonically decreasing)
    for n in range(total_num - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])
    
    # Compute AP using precision-recall curve
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    index = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])
    
    return ap, recall[-2]

def load_npz(fpath):
    return np.load(fpath, allow_pickle=True)

def evaluate_sample(pred_dict, gt_dict, joints_to_eval=None, confidence_threshold=0.1, use_gt_2d=False):
    loss_eval = Inference_Loss()

    pred = pred_dict["pred_3d"].squeeze()
    try: 
        gt_world = gt_dict["gt_world"].squeeze()  # TO UNCOMMENT
    except:
        gt_world = gt_dict["keypoints_3d"].squeeze()  # TO COMMENT OUT
    norm_results = False
    if "keypoints_3d" in pred_dict:
        gt_normalized = pred_dict["keypoints_3d"].squeeze()
        norm_results = True

    # Extract confidence if available
    confidence = None
    if use_gt_2d and "keypoints_2d_gt_processed" in pred_dict:
        # Use preprocessed GT 2D keypoints instead of predictions (oracle/upper-bound)
        kps_2d = pred_dict["keypoints_2d_gt_processed"]
        confidence = kps_2d[..., 2].mean(axis=1)  # [T, J] - average across views
    elif "keypoints_2d_processed" in pred_dict:
        # Use preprocessed predicted 2D keypoints (default)
        kps_2d = pred_dict["keypoints_2d_processed"]
        confidence = kps_2d[..., 2].mean(axis=1)  # [T, J] - average across views
    elif "keypoints_2d" in pred_dict:
        # Fallback to raw keypoints_2d for backwards compatibility
        kps_2d = pred_dict["keypoints_2d"]
        confidence = kps_2d[..., 2].mean(axis=1)  # [T, J] - average across views

    pred_raw = pred.copy()
    pred, gt_world = loss_eval.align_poses(pred_raw, gt_world, joints_to_eval, 
                                           confidence=confidence, 
                                           confidence_threshold=confidence_threshold)
    
    p_mpjpe, per_joint_p_mpjpe = loss_eval.p_mpjpe(pred, gt_world, correct_scale=False)
    p_pck, per_joint_p_pck = loss_eval.p_PCK(pred, gt_world, correct_scale=False)
    p_mpjve, per_joint_p_mpjve = loss_eval.p_mpjve(pred, gt_world, correct_scale=False)
    metrics = {
        "p_mpjpe": p_mpjpe,
        "per_joint_p_mpjpe": per_joint_p_mpjpe,
        "p_pck": p_pck,
        "per_joint_p_pck": per_joint_p_pck,
        "p_mpjve": p_mpjve,
        "per_joint_p_mpjve": per_joint_p_mpjve,
    }
    if norm_results:
        _, gt_normalized = loss_eval.align_poses(
            pred_raw, gt_normalized, joints_to_eval
        )

        mpjpe, per_joint_mpjpe = loss_eval.mpjpe(pred, gt_normalized)
        pck, per_joint_pck = loss_eval.PCK(pred, gt_normalized)
        mpjve, per_joint_mpjve = loss_eval.mpjve(pred, gt_normalized)
        metrics.update(
            {
                "mpjpe": mpjpe,
                "per_joint_mpjpe": per_joint_mpjpe,
                "pck": pck,
                "per_joint_pck": per_joint_pck,
                "mpjve": mpjve,
                "per_joint_mpjve": per_joint_mpjve,
            }
        )
    
    # Add frame-level data for AP computation
    # Compute per-frame errors on the already Procrustes-aligned sequence
    frame_errors = loss_eval.compute_frame_errors(pred, gt_world, correct_scale=False)
    
    metrics["frame_errors"] = frame_errors  # [T] array
    return metrics


def consolidate_results(results_df, dataset):
    if dataset in ["blendmimic3d"]:
        multiplier = 1000  # Since blendmimic GT coords are in meters. multiply by 1000 to convert to mm.
    elif dataset in ["cmu"]:
        multiplier = 10  # CMU Panoptic coords are in centimeters. multiply by 10 to convert to mm.
    else:
        multiplier = 1  # Rest of the datasets are in mm
    consolidated_results = {
        "p_mpjpe": f"{multiplier * results_df.p_mpjpe.mean():.2f} ± {multiplier * results_df.p_mpjpe.std():.2f}",
        "p_pck": f"{results_df.p_pck.mean()*100:.2f} ± {results_df.p_pck.std()*100:.2f}",
        "p_mpjve": f"{multiplier*results_df.p_mpjve.mean():.2f} ± {multiplier*results_df.p_mpjve.std():.2f}",
    }
    consolidated_df = pd.DataFrame(
        list(consolidated_results.items()), columns=["metric", "mean ± std"]
    )
    return consolidated_df, consolidated_results


def evaluate_folder(pred_path, gt_path, outpath, joints_to_eval=None, dataset="dhp19", use_gt_2d=False):
    f_list = sorted(os.listdir(pred_path))
    results_list = []
    
    # For AP metrics: collect frame-level data
    eval_list = []
    gt_frame_id = 0
    
    for fname in f_list:
        sample_name = fname.split(".")[0]
        pred_fpath = os.path.join(pred_path, fname)
        if not os.path.exists(pred_fpath):
            for pattern in [f"{sample_name}.aligned.npz", f"{sample_name}.npz"]:
                pred_fpath = os.path.join(pred_path, pattern)
                if os.path.exists(pred_fpath):
                    break

        gt_fpath = os.path.join(gt_path, fname)
        if not os.path.exists(gt_fpath):
            for pattern in [f"{sample_name}.aligned.npz", f"{sample_name}.npz"]:
                gt_fpath = os.path.join(gt_path, pattern)
                if os.path.exists(gt_fpath):
                    break        
        if not ".npz" in fname:
            continue
        if not os.path.exists(gt_fpath):
            print(f"Missing GT value for sample {pred_fpath}")
            continue

        pred = load_npz(pred_fpath)
        gt = load_npz(gt_fpath)
        
        # Replace predicted 2D with preprocessed GT 2D if flag is set
        if use_gt_2d and "keypoints_2d_gt_processed" in pred:
            pred = dict(pred)  # Convert to mutable dict
            # Replace both processed and raw versions for consistency
            pred["keypoints_2d_processed"] = pred["keypoints_2d_gt_processed"]
            if "keypoints_2d" in pred:
                pred["keypoints_2d"] = pred["keypoints_2d_gt"]
        
        results = evaluate_sample(pred, gt, joints_to_eval, use_gt_2d=use_gt_2d)
        
        # Extract frame-level data for AP computation
        if 'frame_errors' in results:
            frame_errors = results.pop('frame_errors')
            
            # Apply multiplier for correct units (mm)
            if dataset in ["blendmimic3d"]:
                multiplier = 1000  # meters to mm
            elif dataset in ["cmu"]:
                multiplier = 10  # cm to mm
            else:
                multiplier = 1  # already in mm
            
            # Add each frame to eval_list
            for t in range(len(frame_errors)):
                eval_list.append({
                    'mpjpe': float(frame_errors[t] * multiplier),
                    'gt_id': gt_frame_id
                })
                gt_frame_id += 1
        
        results_list.append(results)
    
    results_df = pd.DataFrame(results_list)

    # Compute AP metrics at different thresholds
    ap_metrics = {}
    if len(eval_list) > 0:
        total_gt_frames = gt_frame_id
        thresholds = [25, 50, 100, 150]  # mm
        
        for thresh in thresholds:
            ap, recall = compute_ap_at_threshold(eval_list, total_gt_frames, thresh)
            ap_metrics[f'AP{thresh}'] = ap
            ap_metrics[f'Recall@{thresh}'] = recall
        
        # Compute Recall@500
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < 500]
        recall_500 = len(np.unique(gt_ids)) / total_gt_frames
        ap_metrics['Recall@500'] = recall_500
        
        print(f"\nAP Metrics (frame-level errors from sequence-aligned P-MPJPE):")
        for metric, value in ap_metrics.items():
            if 'AP' in metric:
                print(f"  {metric}: {value*100:.2f}%")
        print(f"  Recall@500: {ap_metrics['Recall@500']*100:.2f}%")
    consolidated_df, consolidated_results = consolidate_results(results_df, dataset)
    
    # Add AP metrics to consolidated results
    for metric, value in ap_metrics.items():
        if 'AP' in metric:
            consolidated_results[metric] = f"{value*100:.2f}%"
        elif metric == 'Recall@500':
            consolidated_results[metric] = f"{value*100:.2f}%"
    
    consolidated_df.to_csv(f"{outpath}_averages.csv", index=False)
    print(f"Folder: {pred_path}\nNum Videos: {len(results_df)}\n{consolidated_results}")
    results_df.to_csv(f"{outpath}_per_sample.csv", index=False)
    
    # Save AP metrics separately
    if ap_metrics:
        ap_df = pd.DataFrame(list(ap_metrics.items()), columns=["metric", "value"])
        ap_df.to_csv(f"{outpath}_ap_metrics.csv", index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate 3D HPE predictions against ground truth."
    )
    ap.add_argument("--pred_path", required=True, help="Folder containing prediction .npz files")
    ap.add_argument("--gt_path", required=True, help="Folder containing ground-truth .npz files")
    ap.add_argument("--output_path", required=True, help="Output path prefix for CSV results")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=list(_load_joints_yaml().keys()),
        help="Dataset name — controls joint evaluation mask and unit scaling",
    )
    ap.add_argument(
        "--use_gt_2d",
        action="store_true",
        default=False,
        help="Use ground-truth 2D keypoints (keypoints_2d_gt_processed) instead of predicted ones",
    )
    ap.add_argument(
        "--joints_yaml",
        default=str(_JOINTS_YAML),
        help=f"Path to joints YAML (default: {_JOINTS_YAML})",
    )
    args = ap.parse_args()

    if not os.path.exists(args.pred_path):
        ap.error(f"Prediction path not found: {args.pred_path}")
    if not os.path.exists(args.gt_path):
        ap.error(f"Ground truth path not found: {args.gt_path}")

    joints_to_eval = get_joints_for_dataset(args.dataset, args.joints_yaml)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print("=" * 70)
    print(f"Predictions : {args.pred_path}")
    print(f"Ground Truth: {args.gt_path}")
    print(f"Dataset     : {args.dataset}")
    print(f"Use GT 2D   : {args.use_gt_2d}")
    print("=" * 70)

    evaluate_folder(
        pred_path=args.pred_path,
        gt_path=args.gt_path,
        outpath=args.output_path,
        joints_to_eval=joints_to_eval,
        dataset=args.dataset,
        use_gt_2d=args.use_gt_2d,
    )

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_path}_averages.csv")
    print(f"Per-sample results: {args.output_path}_per_sample.csv")


if __name__ == "__main__":
    main()
    
