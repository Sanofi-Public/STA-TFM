# Plot Pred vs GT Skeletons
import os
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from src.utils.mv_losses import Inference_Loss

def load_npz(path: str, world=False, show_raw=False):
    data = np.load(path, allow_pickle=True)
    # Joint names from H3.6
    joint_names = [
            "root", "RHip", "RKnee", "RAnkle",
            "LHip", "LKnee", "LAnkle",
            "torso", "neck", "nose", "head",
            "LShoulder", "LElbow", "LWrist",
            "RShoulder", "RElbow", "RWrist",
        ]

    pred = data["pred_3d"] if "pred_3d" in data else None
    if pred is not None and pred.ndim == 4:  # [T,V,J,3] → fuse views
        pred = pred.mean(axis=1)

    if world:
        gt = data["gt_world"] if "gt_world" in data else None
        if gt is not None and gt.shape[1] == 13:
            joint_names = [
                "RHip", "RKnee", "RAnkle",
                "LHip", "LKnee", "LAnkle", "head",
                "LShoulder", "LElbow", "LWrist",
                "RShoulder", "RElbow", "RWrist",
            ]  # coco dataset
    elif show_raw:
        gt = data["keypoints_3d"] if "keypoints_3d" in data else None
    else:
        # Use normalized/processed 3D for data-inspection mode
        gt = data["keypoints_3d_processed"] if "keypoints_3d_processed" in data else (
             data["keypoints_3d"] if "keypoints_3d" in data else None)

    if show_raw:
        k2d = data["keypoints_2d"] if "keypoints_2d" in data else None
    else:
        k2d = data["keypoints_2d_processed"] if "keypoints_2d_processed" in data else (
              data["keypoints_2d"] if "keypoints_2d" in data else None)
    view_mask = data["view_mask"] if "view_mask" in data else None
    img_wh = data["img_wh"] if "img_wh" in data else None
    return pred, gt, joint_names, k2d, view_mask, img_wh

def load_connectivities(yaml_path: str):
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f)
    return yml["connectivities.default"]

def build_frames(preds, gt, joint_names, connectivities, color_pred="blue", color_gt="red",
                 k2d=None, view_mask=None, img_wh=None, show_raw=False):
    # Pick min length if both exist
    T = min(len(preds) if preds is not None else np.inf,
            len(gt) if gt is not None else np.inf)
    if not np.isfinite(T):
        raise ValueError("No preds or GT to plot!")
    T = int(T)
    if preds is not None: preds = preds[:T]
    if gt is not None:    gt    = gt[:T]
    if k2d is not None:   k2d   = k2d[:T]
    if view_mask is not None: view_mask = view_mask[:T]

    J = preds.shape[1] if preds is not None else gt.shape[1]
    labels = joint_names if (joint_names and len(joint_names) == J) else [f"J{j}" for j in range(J)]
    joint_to_idx = {name: i for i, name in enumerate(labels)}

    V = k2d.shape[1] if k2d is not None else 0

    # Count edges once for consistent per-frame trace counts
    n_edges = sum(
        1 for j, neighbors in connectivities.items() if j in joint_to_idx
        for n in neighbors if n in joint_to_idx
    )

    # --- Trace helpers ---
    def joint_trace_3d(X, name, color, showlegend=True):
        return go.Scatter3d(
            x=X[:,0], y=X[:,1], z=X[:,2],
            mode="markers",
            marker=dict(size=4, color=color),
            name=name, text=labels,
            legendgroup=name, showlegend=showlegend,
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
        )

    def bone_traces_3d(X, name, color):
        segs = []
        for j, neighbors in connectivities.items():
            if j not in joint_to_idx:
                continue
            j_idx = joint_to_idx[j]
            for n in neighbors:
                if n not in joint_to_idx:
                    continue
                n_idx = joint_to_idx[n]
                segs.append(go.Scatter3d(
                    x=[X[j_idx,0], X[n_idx,0]],
                    y=[X[j_idx,1], X[n_idx,1]],
                    z=[X[j_idx,2], X[n_idx,2]],
                    mode="lines",
                    line=dict(color=color, width=2),
                    legendgroup=name, showlegend=False
                ))
        return segs

    def joint_trace_2d(xs, ys, name, color, showlegend=True):
        return go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=5, color=color),
            name=name, text=labels,
            legendgroup=name, showlegend=showlegend,
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )

    def bone_traces_2d(xs, ys, name, color):
        segs = []
        for j, neighbors in connectivities.items():
            if j not in joint_to_idx:
                continue
            j_idx = joint_to_idx[j]
            for n in neighbors:
                if n not in joint_to_idx:
                    continue
                n_idx = joint_to_idx[n]
                segs.append(go.Scatter(
                    x=[xs[j_idx], xs[n_idx]],
                    y=[ys[j_idx], ys[n_idx]],
                    mode="lines",
                    line=dict(color=color, width=2),
                    legendgroup=name, showlegend=False
                ))
        return segs

    # Global 3D bounds
    all_xyz = []
    if preds is not None: all_xyz.append(preds.reshape(-1,3))
    if gt is not None:    all_xyz.append(gt.reshape(-1,3))
    all_xyz = np.concatenate(all_xyz, axis=0)
    xmin, ymin, zmin = np.min(all_xyz, axis=0)
    xmax, ymax, zmax = np.max(all_xyz, axis=0)
    pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    xmin, ymin, zmin = xmin - pad, ymin - pad, zmin - pad
    xmax, ymax, zmax = xmax + pad, ymax + pad, zmax + pad

    # Number of 3D traces per frame (pred block + GT block)
    n_3d = ((1 + n_edges) if preds is not None else 0) + \
           ((1 + n_edges) if gt is not None else 0)

    # Build flat trace list for timestep t
    # Layout: [3D pred (1+n_edges)] [3D GT (1+n_edges)] [2D view0 (1+n_edges)] ...
    def build_trace_list(t, showlegend=False):
        traces = []
        if preds is not None:
            traces.append(joint_trace_3d(preds[t], "Prediction", color_pred, showlegend=showlegend))
            traces.extend(bone_traces_3d(preds[t], "Prediction", color_pred))
        if gt is not None:
            traces.append(joint_trace_3d(gt[t], "Ground Truth", color_gt, showlegend=showlegend))
            traces.extend(bone_traces_3d(gt[t], "Ground Truth", color_gt))
        for v in range(V):
            masked = view_mask is not None and not view_mask[t, v]
            if not masked:
                xs, ys = k2d[t, v, :, 0], k2d[t, v, :, 1]
                traces.append(joint_trace_2d(xs, ys, f"Cam {v}", "green", showlegend=showlegend))
                traces.extend(bone_traces_2d(xs, ys, f"Cam {v}", "green"))
            else:
                # Empty placeholders to keep trace count consistent
                traces.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))
                traces.extend([go.Scatter(x=[], y=[], mode="lines", showlegend=False)
                                for _ in range(n_edges)])
        return traces

    # Create figure
    if V > 0:
        specs = [{"type": "scene"}] + [{"type": "xy"} for _ in range(V)]
        col_titles = [("3D Pred vs GT" if preds is not None else "3D Skeleton")] + [f"Cam {v}" for v in range(V)]
        fig = make_subplots(rows=1, cols=1 + V, specs=[specs], subplot_titles=col_titles)
        t0_traces = build_trace_list(0, showlegend=True)
        for idx, tr in enumerate(t0_traces):
            if idx < n_3d:
                fig.add_trace(tr, row=1, col=1)
            else:
                v = (idx - n_3d) // (1 + n_edges)
                fig.add_trace(tr, row=1, col=2 + v)
    else:
        fig = go.Figure()
        for tr in build_trace_list(0, showlegend=True):
            fig.add_trace(tr)

    # Animation frames
    fig.frames = [go.Frame(data=build_trace_list(t), name=str(t)) for t in range(T)]

    # Layout
    fig.update_layout(
        title="Pred vs GT Skeletons",
        scene=dict(
            xaxis=dict(range=[xmin, xmax], title="X"),
            yaxis=dict(range=[ymin, ymax], title="Y"),
            zaxis=dict(range=[zmin, zmax], title="Z"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0)),
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",  method="animate", args=[None, {"frame":{"duration":10,"redraw":True}, "fromcurrent":True}]),
                dict(label="Pause", method="animate", args=[[None], {"mode":"immediate", "frame":{"duration":0}, "transition":{"duration":0}}]),
            ]
        )],
        **(dict(width=420 * (1 + V), height=620) if V > 0 else {}),
    )

    # 2D axis ranges per camera view
    if V > 0:
        for v in range(V):
            col = 2 + v
            if show_raw and img_wh is not None:
                w, h = img_wh[v, 0], img_wh[v, 1]
                fig.update_xaxes(range=[0, w], row=1, col=col)
                fig.update_yaxes(range=[h, 0], row=1, col=col)  # flip y (image origin top-left)
            else:
                valid_xy = k2d[:, v, :, :2].reshape(-1, 2)
                x0, y0 = valid_xy.min(axis=0) - 0.05
                x1, y1 = valid_xy.max(axis=0) + 0.05
                fig.update_xaxes(range=[x0, x1], row=1, col=col)
                fig.update_yaxes(range=[y1, y0], row=1, col=col)  # flip y

    return fig

def main():
    ap = argparse.ArgumentParser(
        description="Visualize 3D skeletons and 2D camera projections from an NPZ file. "
                    "If the file contains model predictions (pred_3d), overlays pred vs GT and "
                    "produces Procrustes-aligned output. Otherwise, shows GT skeleton only."
    )
    ap.add_argument("--npz", required=True, help="Path to .npz file")
    ap.add_argument("--output_dir", required=True, help="Directory to save output HTML files")
    ap.add_argument("--yaml", default=str(Path(__file__).resolve().parents[2] / "conf" / "connectivities" / "default.yaml"))
    ap.add_argument("--world_coord", default=True, type=lambda v: str(v).lower() in ("1", "true", "yes"),
                    help="Use gt_world (world-frame GT) instead of keypoints_3d (True/False)")
    ap.add_argument("--show_raw", action="store_true", default=False,
                    help="Use raw (un-normalized) keypoints_2d / keypoints_3d instead of processed versions")
    ap.add_argument("--start", type=int, default=0, help="Start frame index")
    ap.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    ap.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to visualize")
    args = ap.parse_args()

    pred, gt, joint_names, k2d, view_mask, img_wh = load_npz(args.npz, args.world_coord, args.show_raw)
    connectivities = load_connectivities(args.yaml)

    # Apply frame range
    total_frames = min(len(pred) if pred is not None else float('inf'),
                       len(gt) if gt is not None else float('inf'))
    start = args.start
    end = args.end if args.end is not None else total_frames
    if args.max_frames is not None:
        end = min(end, start + args.max_frames)
    end = int(min(end, total_frames))

    print(f"Visualizing frames {start} to {end} (total: {end - start} frames)")

    if pred is not None:     pred      = pred[start:end]
    if gt is not None:       gt        = gt[start:end]
    if k2d is not None:      k2d       = k2d[start:end]
    if view_mask is not None: view_mask = view_mask[start:end]

    fname = Path(args.npz).stem
    os.makedirs(args.output_dir, exist_ok=True)

    if pred is None:
        # Data-inspection mode: no predictions, just show GT skeleton + 2D views
        fig = build_frames(None, gt, joint_names, connectivities, k2d=k2d, view_mask=view_mask, img_wh=img_wh, show_raw=args.show_raw)
        out_path = os.path.join(args.output_dir, f"{fname}.html")
        fig.write_html(out_path)
        print(f"Saved visualization to: {out_path}")
    else:
        # Evaluation mode: overlay pred vs GT + Procrustes-aligned variant
        loss_eval = Inference_Loss()
        pred_aligned = np.zeros_like(pred)
        for t in range(min(len(pred), len(gt))):
            pred_aligned[t], _ = loss_eval.procrustes_align(gt[t], pred[t])

        fig = build_frames(pred, gt, joint_names, connectivities, k2d=k2d, view_mask=view_mask, img_wh=img_wh, show_raw=args.show_raw)
        fig.write_html(os.path.join(args.output_dir, f"{fname}_raw.html"))
        print(f"Saved raw comparison to: {os.path.join(args.output_dir, f'{fname}_raw.html')}")

        fig_aligned = build_frames(pred_aligned, gt, joint_names, connectivities, k2d=k2d, view_mask=view_mask, img_wh=img_wh, show_raw=args.show_raw)
        fig_aligned.write_html(os.path.join(args.output_dir, f"{fname}_aligned.html"))
        print(f"Saved aligned comparison to: {os.path.join(args.output_dir, f'{fname}_aligned.html')}")

if __name__ == "__main__":
    main()