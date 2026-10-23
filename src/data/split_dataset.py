"""
Split dataset into train and test sets based on predefined splits.
Supports HAA4D (action split) and DHP19 (subject split).

Usage:
    python -m src.data.split_dataset --dataset union --output_dir datasets/haa4d_dhp19_union/split
"""

import os
import shutil
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = REPO_ROOT / "conf" / "dataset_splits.json"


def load_splits(splits_file=SPLITS_FILE):
    """Load dataset splits from JSON file."""
    with open(splits_file, "r") as f:
        return json.load(f)


def split_dataset(output_dir, dataset_type, source_dirs=None, splits_file=SPLITS_FILE):
    """Split dataset into train and test sets.

    source_dirs: optional {dataset_name: dir} overriding `source_dir` in the splits file.
    Relative paths are resolved against the repository root.
    """
    dataset_splits = load_splits(splits_file)
    if dataset_type not in dataset_splits and dataset_type != "union":
        raise ValueError(f"Dataset '{dataset_type}' not defined. Available: {list(dataset_splits.keys())} or 'union'")

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_count = test_count = 0

    datasets = dataset_splits.items() if dataset_type == "union" else [(dataset_type, dataset_splits[dataset_type])]

    for ds_type, config in datasets:
        source_dir = (source_dirs or {}).get(ds_type) or config["source_dir"]
        source_dir = Path(source_dir) if os.path.isabs(source_dir) else REPO_ROOT / source_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory for '{ds_type}' not found: {source_dir}")

        npz_files = sorted(f for f in os.listdir(source_dir) if f.endswith(".npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {source_dir}. Run src.data.data_processing first.")

        test_names = config["test"]
        for filename in npz_files:
            is_test = any(filename.startswith(name) for name in test_names)
            dst_path = os.path.join(test_dir if is_test else train_dir, filename)
            shutil.copy2(source_dir / filename, dst_path)
            test_count += is_test
            train_count += not is_test

    print(f"Split complete: {train_count} train, {test_count} test")
    print(f"Train: {train_dir}")
    print(f"Test: {test_dir}")


def main():
    ap = argparse.ArgumentParser(description="Split synthetic .npz files into train/test using conf/dataset_splits.json")
    ap.add_argument("--dataset", default="union", choices=["haa4d", "dhp19", "union"],
                    help="Dataset to split; 'union' merges HAA4D and DHP19 into shared train/test folders")
    ap.add_argument("--output_dir", required=True, help="Where to create train/ and test/ folders")
    ap.add_argument("--haa4d_dir", default=None, help="Override HAA4D source dir (default: from splits file)")
    ap.add_argument("--dhp19_dir", default=None, help="Override DHP19 source dir (default: from splits file)")
    ap.add_argument("--splits_file", default=str(SPLITS_FILE))
    args = ap.parse_args()

    split_dataset(args.output_dir, args.dataset,
                  source_dirs={"haa4d": args.haa4d_dir, "dhp19": args.dhp19_dir},
                  splits_file=args.splits_file)


if __name__ == "__main__":
    main()
