"""
Split dataset into train and test sets based on predefined splits.
Supports HAA4D (action split) and DHP19 (subject split).
"""

import os
import shutil
import json

SPLITS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "conf", "dataset_splits.json")


def load_splits():
    """Load dataset splits from JSON file."""
    with open(SPLITS_FILE, "r") as f:
        return json.load(f)


def split_dataset(output_dir, dataset_type):
    """Split dataset into train and test sets."""
    dataset_splits = load_splits()
    if dataset_type not in dataset_splits and dataset_type != "union":
        print(f"Dataset '{dataset_type}' not defined. Available: {list(dataset_splits.keys())} or 'union'")
        return
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_count = test_count = 0
    
    datasets = dataset_splits.items() if dataset_type == "union" else [(dataset_type, dataset_splits[dataset_type])]
    
    for ds_type, config in datasets:
        source_dir = config["source_dir"]
        if not os.path.exists(source_dir):
            print(f"Source directory not found: {source_dir}")
            continue
        
        test_names = config["test"]
        npz_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.npz')])
        
        for filename in npz_files:
            src_path = os.path.join(source_dir, filename)
            is_test = any(filename.startswith(name) for name in test_names)
            dst_path = os.path.join(test_dir if is_test else train_dir, filename)
            shutil.copy2(src_path, dst_path)
            test_count += is_test
            train_count += not is_test
    
    print(f"Split complete: {train_count} train, {test_count} test")
    print(f"Train: {train_dir}")
    print(f"Test: {test_dir}")


if __name__ == "__main__":
    # Configuration
    output_dir = "<path-to-output-split-dataset>"  # Output directory for split dataset
    dataset_type = "union"  # Options: "haa4d", "dhp19", or "union"
    
    split_dataset(output_dir, dataset_type)