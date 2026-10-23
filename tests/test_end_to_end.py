"""CPU smoke test of the full pipeline via the CLIs: generate -> split -> train -> infer -> evaluate.

Uses random skeletons and a randomly initialised DSTformer, so metrics are meaningless; it only checks
that every stage runs and produces its outputs.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _run(*args):
    env = {**os.environ, "TQDM_DISABLE": "1"}
    subprocess.run([sys.executable, "-m", *args], check=True, env=env)


def test_pipeline(tmp_path):
    from src.models.DSTformer import DSTformer

    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for name in ["abseiling", "archery", "burpee", "air_guitar"]:
        for i in (1, 2):
            t = np.linspace(0, 6, 60)[:, None, None]
            skel = rng.normal(size=(1, 17, 3)) * 300 + 20 * np.sin(t + rng.normal(size=(1, 17, 3)))
            np.save(raw / f"{name}_{i:03d}.npy", skel.astype(np.float32))
    weights = tmp_path / "dstformer.pth"
    torch.save({"model": DSTformer(num_joints=17, maxlen=243).state_dict()}, weights)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"haa4d": {"source_dir": str(tmp_path / "synthetic"),
                                            "train": ["abseiling", "archery", "burpee"], "test": ["air_guitar"]}}))

    _run("src.data.data_processing", "--dataset", "haa4d", "--data_dir", str(raw), "--output_dir", str(tmp_path / "synthetic"))
    _run("src.data.split_dataset", "--dataset", "haa4d", "--output_dir", str(tmp_path / "split"), "--splits_file", str(splits))
    _run("src.train", "--dataset_path", str(tmp_path / "split/train"), "--dstformer_path", str(weights),
         "--model_save_dir", str(tmp_path / "ckpt"), "--metrics_dir", str(tmp_path / "metrics"),
         "--epochs", "1", "--num_workers", "0")
    ckpt = next((tmp_path / "ckpt").glob("*.pth"))
    _run("src.infer", "--fusion_head_ckpt", str(ckpt), "--dstformer_path", str(weights),
         "--input", str(tmp_path / "split/test"), "--output", str(tmp_path / "results"))
    preds = tmp_path / "results" / ckpt.stem
    assert len(list(preds.glob("*.pred3d.npz"))) == 2
    _run("src.eval.evaluate", "--pred_path", str(preds), "--gt_path", str(tmp_path / "split/test"),
         "--output_path", str(tmp_path / "eval"), "--dataset", "haa4d")
    assert (tmp_path / "eval_averages.csv").exists()
