"""CPU-only checks of synthetic data generation and train/test splitting (no datasets or weights needed)."""
import json

import numpy as np
import pytest

from src.data.data_processing import DataProcessor, DEFAULT_VIEWS

T, J, V = 20, 17, 4


def _fake_haa4d(raw_dir, names):
    rng = np.random.default_rng(0)
    raw_dir.mkdir(parents=True)
    for n in names:
        np.save(raw_dir / f"{n}.npy", rng.normal(size=(T, J, 3)).astype(np.float32))


def _generate(tmp_path, out_name, seed=42):
    out = tmp_path / out_name
    DataProcessor({"dataset_type": "haa4d", "data_dir": str(tmp_path / "raw"), "output_dir": str(out),
                   "scale": "local", "seed": seed, "views": DEFAULT_VIEWS}).run()
    return out


def test_generated_npz_format(tmp_path):
    _fake_haa4d(tmp_path / "raw", ["air_guitar_001"])
    d = np.load(_generate(tmp_path, "out") / "air_guitar_001.npz", allow_pickle=True)
    assert d["keypoints_2d"].shape == (T, V, J, 3)
    assert d["keypoints_2d_processed"].shape == (T, V, J, 3)
    assert d["keypoints_3d"].shape == (T, J, 3)
    assert d["keypoints_3d_processed"].shape == (T, J, 3)
    assert d["view_mask"].shape == (T, V) and d["view_mask"].all()
    assert d["img_wh"].shape == (V, 2)
    np.testing.assert_allclose(d["keypoints_2d_processed"][:, :, 0, :2], 0, atol=1e-5)  # root-centred


def test_generation_is_seeded(tmp_path):
    _fake_haa4d(tmp_path / "raw", ["air_guitar_001", "abseiling_001"])
    a, b, c = (_generate(tmp_path, n, seed=s) for n, s in [("a", 1), ("b", 1), ("c", 2)])
    for f in ["air_guitar_001.npz", "abseiling_001.npz"]:
        np.testing.assert_array_equal(np.load(a / f)["keypoints_2d"], np.load(b / f)["keypoints_2d"])
        assert not np.array_equal(np.load(a / f)["keypoints_2d"], np.load(c / f)["keypoints_2d"])


def test_split_dataset(tmp_path):
    from src.data.split_dataset import split_dataset
    src = tmp_path / "haa4d"
    src.mkdir()
    for n in ["air_guitar_001", "abseiling_001"]:
        np.savez(src / f"{n}.npz", x=np.zeros(1))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"haa4d": {"source_dir": str(src), "train": ["abseiling"], "test": ["air_guitar"]}}))
    split_dataset(str(tmp_path / "split"), "haa4d", splits_file=splits)
    assert [p.name for p in (tmp_path / "split/test").iterdir()] == ["air_guitar_001.npz"]
    assert [p.name for p in (tmp_path / "split/train").iterdir()] == ["abseiling_001.npz"]


def test_example_configs_match_paper_defaults():
    import yaml
    from pathlib import Path
    for name in ["dhp19", "haa4d"]:
        with open(Path(__file__).resolve().parents[1] / f"conf/data_processing/{name}.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["dataset_type"] == name
        assert cfg["scale"] == "local" and cfg["seed"] == 42
        assert cfg["views"] == DEFAULT_VIEWS


def test_dhp19_subject_split(tmp_path):
    """DHP19 subject split: test = S1, S4, S12; train = the other 14 subjects (S1 must not match S10-S17).
    Uses empty placeholder files, so no dataset is needed."""
    from src.data.split_dataset import load_splits, split_dataset
    src = tmp_path / "dhp19"
    src.mkdir()
    for s in range(1, 18):
        np.savez(src / f"S{s}_1_1.npz", x=np.zeros(1))
    cfg = {**load_splits()["dhp19"], "source_dir": str(src)}
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"dhp19": cfg}))
    split_dataset(str(tmp_path / "split"), "dhp19", splits_file=splits)
    subjects = lambda d: sorted(p.name.split("_")[0] for p in (tmp_path / "split" / d).iterdir())
    assert subjects("test") == ["S1", "S12", "S4"]
    assert subjects("train") == sorted(f"S{s}" for s in range(1, 18) if s not in (1, 4, 12))
