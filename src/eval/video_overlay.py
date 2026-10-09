import sys
import os
import argparse
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm 

# Project Path Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.utils.mv_losses import Inference_Loss
except ImportError:
    print("Error: Could not find 'src.utils.mv_losses'. Ensure you are running from the project root or PYTHONPATH is set.")
    sys.exit(1)

def load_connectivities(yaml_path: str):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Connectivity YAML not found at {yaml_path}")
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f)
    return yml["connectivities.default"]

def draw_2d_skeleton(frame, points, connectivities, joint_to_idx):
    """Draws 2D skeleton with optimized lookups."""
    WHITE = (255, 255, 255)
    
    # Draw Lines
    for j, neighbors in connectivities.items():
        if j in joint_to_idx:
            idx1 = joint_to_idx[j]
            p1 = (int(points[idx1, 0]), int(points[idx1, 1]))
            if p1 == (0, 0): continue 
            for n in neighbors:
                if n in joint_to_idx:
                    idx2 = joint_to_idx[n]
                    p2 = (int(points[idx2, 0]), int(points[idx2, 1]))
                    if p2 == (0, 0): continue
                    cv2.line(frame, p1, p2, WHITE, 1, cv2.LINE_AA)
    
    # Draw Joints
    for pt in points:
        if pt[0] > 2 and pt[1] > 2:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, WHITE, -1, cv2.LINE_AA)
    
    return frame

def render_frame_to_bgr(pred, gt, connectivities, joint_to_idx, size=(600, 600)):
    """Renders 3D comparison plot."""
    plt.ioff()
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasAgg(fig)

    def plot_skel(points, color, alpha=1.0, lw=2):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=15, alpha=alpha)
        for j, neighbors in connectivities.items():
            if j in joint_to_idx:
                idx1 = joint_to_idx[j]
                for n in neighbors:
                    if n in joint_to_idx:
                        idx2 = joint_to_idx[n]
                        ax.plot([points[idx1,0], points[idx2,0]], 
                                [points[idx1,1], points[idx2,1]], 
                                [points[idx1,2], points[idx2,2]], 
                                c=color, lw=lw, alpha=alpha)

    plot_skel(gt, color='red', alpha=0.4, lw=1)
    plot_skel(pred, color='blue', alpha=1.0, lw=2)

    # Fit axis limits to the actual data range so joints are visible at any scale
    all_pts = np.concatenate([pred, gt], axis=0)
    center = all_pts.mean(axis=0)
    half = max(np.abs(all_pts - center).max() * 1.2, 1e-3)
    ax.set_xlim([center[0] - half, center[0] + half])
    ax.set_ylim([center[1] - half, center[1] + half])
    ax.set_zlim([center[2] - half, center[2] + half])
    ax.view_init(elev=15, azim=-45, vertical_axis='y') 
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Multi-view 3D Pose Visualization Overlay")
    # Data arguments
    parser.add_argument("--pred_npz", required=True, help="Path to predicted 3D .npz")
    parser.add_argument("--data_npz", required=True, help="Path to ground truth/2D .npz")
    parser.add_argument("--videos", nargs='+', required=True, help="Paths to camera MP4 files (provide 4)")
    parser.add_argument("--yaml", default=os.path.join(PROJECT_ROOT, "conf/connectivities/default.yaml"))
    
    # Output arguments
    parser.add_argument("--out", default="overlay_result.mp4", help="Output video filename")
    parser.add_argument("--limit", type=int, default=-1, help="Max frames to process (-1 = process all)")
    parser.add_argument("--frame_skip", type=int, default=1, help="Frame skip used when generating the NPZ (must match demo --frame_skip)")
    parser.add_argument("--no_procrustes", action="store_true", help="Disable Procrustes alignment")
    
    args = parser.parse_args()

    # Load Data
    d_pred = np.load(args.pred_npz, allow_pickle=True)
    d_orig = np.load(args.data_npz, allow_pickle=True)
    
    pred_3d = d_pred["pred_3d"]
    gt_3d = d_pred["keypoints_3d"] if "keypoints_3d" in d_pred else d_pred["pred_3d"]
    kp2d_all = d_orig["keypoints_2d"] 
    
    # Collapse multi-hypothesis if present
    if pred_3d.ndim == 4: pred_3d = pred_3d.mean(axis=1)
    if gt_3d.ndim == 4: gt_3d = gt_3d.mean(axis=1)
    
    joint_names = ["root", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                   "torso", "neck", "nose", "head", "LShoulder", "LElbow", "LWrist",
                   "RShoulder", "RElbow", "RWrist"]
    joint_to_idx = {name: i for i, name in enumerate(joint_names)}
    connectivities = load_connectivities(args.yaml)
    loss_eval = Inference_Loss()

    # Video Setup
    caps = [cv2.VideoCapture(v) for v in args.videos]
    for i, v_path in enumerate(args.videos):
        if not caps[i].isOpened():
            print(f"Error: Could not open video {v_path}")
            sys.exit(1)

    # Layout Calculations
    plot_dim = 600
    sub_h = plot_dim // 2
    ret, first_frame = caps[0].read()
    if not ret:
        print("Error: Could not read first frame.")
        sys.exit(1)
    
    orig_h, orig_w = first_frame.shape[:2]
    sub_w = int(sub_h * (orig_w / orig_h))
    
    # Reset to start
    for c in caps:
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Use FPS from the first video unless overridden
    video_fps = caps[0].get(cv2.CAP_PROP_FPS)
    out_fps = video_fps

    # Writer Setup
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    canvas_w = (2 * sub_w) + plot_dim
    out_video = cv2.VideoWriter(args.out, fourcc, out_fps, (canvas_w, plot_dim))

    max_frames = min(len(pred_3d), args.limit if args.limit > 0 else len(pred_3d))
    
    print(f"Generating {args.out}...")
    for t in tqdm(range(max_frames)):
        sub_frames = []

        for v_idx, c in enumerate(caps):
            ret, frame = c.read()
            # Skip extra frames to stay in sync when frame_skip > 1
            for _ in range(args.frame_skip - 1):
                c.grab()

            if not ret:
                frame_res = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
            else:
                frame_res = cv2.resize(frame, (sub_w, sub_h))

            # Scale 2D Keypoints to the new sub-window size
            pts_2d = kp2d_all[t, v_idx][:, :2].copy()
            pts_2d[:, 0] *= (sub_w / orig_w)
            pts_2d[:, 1] *= (sub_h / orig_h)
            
            cv2.putText(frame_res, f"F:{t} Cam:{v_idx}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            frame_overlay = draw_2d_skeleton(frame_res, pts_2d, connectivities, joint_to_idx)
            sub_frames.append(frame_overlay)

        # 3D Alignment & Rendering
        if not args.no_procrustes:
            p_viz, _ = loss_eval.procrustes_align(gt_3d[t], pred_3d[t])
        else:
            p_viz = pred_3d[t]

        skel_3d = render_frame_to_bgr(p_viz, gt_3d[t], connectivities, joint_to_idx, size=(plot_dim, plot_dim))
        
        # Grid Layout: [Cam0 Cam1 | 3D]
        #              [Cam2 Cam3 | Plot]
        top_row = np.hstack((sub_frames[0], sub_frames[1]))
        bot_row = np.hstack((sub_frames[2], sub_frames[3]))
        grid_2d = np.vstack((top_row, bot_row))
        final_canvas = np.hstack((grid_2d, skel_3d))

        out_video.write(final_canvas)

    # Cleanup
    for c in caps: c.release()
    out_video.release()
    print(f"\nSuccess! Video saved to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()

