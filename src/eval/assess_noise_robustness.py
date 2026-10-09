"""
Generate test cases for DHP-19 dataset to evaluate model robustness.

Usage:
  # Add noise
  python assess_noise_robustness.py --input /path/to/data --output /path/out --mode noise --noise_std 0.02
  
  # Remove keypoints
  python assess_noise_robustness.py --input /path/to/data --output /path/out --mode missing --max_remove 4
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def add_noise(keypoints_2d, std=0.02):
    """
    Add Gaussian noise to x,y coordinates relative to pose height.
    std=0.02 means 2% of the pose height (max_y - min_y) per frame per view.
    """
    noisy = keypoints_2d.copy()
    T, V, J, C = noisy.shape
    
    for t in range(T):
        for v in range(V):
            # Get pose height from y-coordinate range in this view
            y_coords = noisy[t, v, :, 1]
            pose_height = np.max(y_coords) - np.min(y_coords) if len(y_coords) > 0 else 1.0
            
            # Noise scaled to pose height
            noise_std = std * pose_height
            noise = np.random.normal(0, noise_std, size=(J, 2))
            noisy[t, v, :, :2] += noise
    
    return noisy


def remove_keypoints(keypoints_2d, n_remove):
    """
    Randomly remove n_remove keypoints per view per frame.
    Strategy: Replace (x,y) with previous frame's value (temporal interpolation),
    set conf=0 to mark as invalid. Most realistic for tracking failures/occlusions.
    Falls back to mean of other joints for first frame.
    """
    if n_remove == 0:
        return keypoints_2d.copy()
    
    keypoints = keypoints_2d.copy()
    T, V, J, C = keypoints.shape
    
    for t in range(T):
        for v in range(V):
            idxs = np.random.choice(J, min(n_remove, J), replace=False)
            for j in idxs:
                if t > 0:
                    # Use previous frame's position (temporal interpolation)
                    keypoints[t, v, j, :2] = keypoints[t-1, v, j, :2]
                else:
                    # First frame: use mean of other joints in this view
                    other_joints = np.concatenate([keypoints[t, v, :j, :2], keypoints[t, v, j+1:, :2]])
                    keypoints[t, v, j, :2] = np.mean(other_joints, axis=0)
                
                keypoints[t, v, j, 2] = 0.0  # Mark as invalid with conf=0
    
    return keypoints


def process_noise(input_folder, output_folder, noise_std, seed):
    """Generate noisy keypoint files."""
    np.random.seed(seed)
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_path.glob("*.npz"))
    print(f"Adding {noise_std*100}% noise to {len(files)} files...")
    
    for fpath in tqdm(files):
        data = dict(np.load(fpath, allow_pickle=True))
        data['keypoints_2d'] = add_noise(data['keypoints_2d'], noise_std)
        np.savez(output_path / fpath.name, **data)
    
    print(f"✓ Saved to {output_path}")


def process_missing(input_folder, output_folder, max_remove, seed):
    """Generate files with missing keypoints (0 to max_remove)."""
    np.random.seed(seed)
    input_path = Path(input_folder)
    base_output = Path(output_folder)
    
    files = sorted(input_path.glob("*.npz"))
    print(f"Generating missing keypoint cases (0-{max_remove}) for {len(files)} files...")
    
    for n_remove in range(0, max_remove + 1):
        output_path = base_output / f"missing_{n_remove}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fpath in tqdm(files, desc=f"Missing {n_remove}"):
            data = dict(np.load(fpath, allow_pickle=True))
            data['keypoints_2d'] = remove_keypoints(data['keypoints_2d'], n_remove)
            np.savez(output_path / fpath.name, **data)
        
        print(f"✓ Saved missing_{n_remove} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DHP-19 robustness test cases")
    parser.add_argument('--input', required=True, help='Input folder with .npz files')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--mode', required=True, choices=['noise', 'missing'], 
                        help='noise: add Gaussian noise | missing: remove keypoints')
    parser.add_argument('--noise_std', type=float, default=0.02, 
                        help='Noise std dev (for noise mode)')
    parser.add_argument('--max_remove', type=int, default=4, 
                        help='Max keypoints to remove (for missing mode)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'noise':
        process_noise(args.input, args.output, args.noise_std, args.seed)
    elif args.mode == 'missing':
        process_missing(args.input, args.output, args.max_remove, args.seed)


if __name__ == "__main__":
    main()