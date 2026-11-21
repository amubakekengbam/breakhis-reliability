import numpy as np
import pandas as pd

from breakhis.data import build_metadata, parse_filename
from breakhis.metrics import bootstrap_ci, image_metrics, patient_metrics
from breakhis.splits import image_level_split, patient_level_split, patient_overlap


def test_parse_malignant():
    m = parse_filename("SOB_M_DC-14-2523-400-005.png")
    assert m == {"label": 1, "subtype": "ductal_carcinoma", "patient_id": "14-2523",
                 "patient_base": "14-2523", "magnification": 400, "seq": 5}


def test_parse_benign_with_letters_in_patient_id():
    m = parse_filename("SOB_B_A-14-22549AB-40-001.png")
    assert m["label"] == 0 and m["patient_id"] == "14-22549AB" and m["magnification"] == 40
    assert m["patient_base"] == "14-22549"


def test_parse_rejects_other_files():
    assert parse_filename("readme.png") is None


def _fake_df(n_patients=20, imgs_per_patient=10):
    rows = []
    for p in range(n_patients):
        label = int(p % 3 != 0)  # roughly 2/3 malignant, like BreakHis
        for i in range(imgs_per_patient):
            rows.append({"patient_id": f"p{p}", "label": label, "path": f"p{p}_{i}.png"})
    return pd.DataFrame(rows)


def test_patient_split_has_no_overlap():
    df = _fake_df()
    for seed in range(3):
        tr, te = patient_level_split(df, seed=seed)
        assert patient_overlap(df, tr, te) == 0


def test_image_split_leaks_patients():
    df = _fake_df()
    tr, te = image_level_split(df, seed=0)
    assert patient_overlap(df, tr, te) > 0


def test_metrics_and_patient_bootstrap():
    df = _fake_df()
    rng = np.random.default_rng(0)
    df["prob"] = np.clip(df["label"] * 0.6 + rng.uniform(0, 0.4, len(df)), 0, 1)
    assert image_metrics(df["label"], df["prob"])["auroc"] > 0.9
    assert patient_metrics(df)["n_test_patients"] == 20
    lo, hi = bootstrap_ci(df, n_boot=200)
    assert 0 <= lo <= hi <= 1


def test_build_metadata_scans_nested_folders(tmp_path):
    d = tmp_path / "benign" / "SOB" / "adenosis" / "40X"
    d.mkdir(parents=True)
    (d / "SOB_B_A-14-22549AB-40-001.png").write_bytes(b"")
    (d / "notes.png").write_bytes(b"")
    df = build_metadata(tmp_path)
    assert len(df) == 1 and df.loc[0, "patient_id"] == "14-22549AB"


def test_cv_folds_test_every_patient_once():
    from breakhis.splits import patient_cv_folds
    df = _fake_df(n_patients=25)
    seen = []
    for tr, te in patient_cv_folds(df, "patient_id", n_folds=5, seed=0):
        assert patient_overlap(df, tr, te) == 0
        seen += list(df.iloc[te]["patient_id"].unique())
    assert sorted(seen) == sorted(df["patient_id"].unique())


def test_grouping_by_number_keeps_shared_numbers_together():
    from breakhis.splits import group_overlap, patient_cv_folds
    df = _fake_df(n_patients=20)
    # pairs of IDs share a number (p0+p1 -> n0, p2+p3 -> n1, ...)
    df["patient_base"] = "n" + (df["patient_id"].str[1:].astype(int) // 2).astype(str)
    for tr, te in patient_cv_folds(df, "patient_base", n_folds=4, seed=0):
        assert group_overlap(df, tr, te, "patient_base") == 0


def test_subtype_metrics_counts():
    from breakhis.metrics import subtype_metrics
    df = pd.DataFrame({"patient_id": ["a", "a", "b", "c"], "label": [1, 1, 1, 0],
                       "subtype": ["ductal", "ductal", "lobular", "adenosis"],
                       "prob": [0.9, 0.8, 0.2, 0.1]})
    out = subtype_metrics(df)
    assert out.loc["ductal", "patient_correct_rate"] == 1.0
    assert out.loc["lobular", "patient_correct_rate"] == 0.0
    assert out.loc["adenosis", "class"] == "benign"


def test_multiple_subtypes_found():
    from breakhis.data import patients_in_multiple_subtypes
    df = pd.DataFrame({"patient_id": ["x", "x", "y"], "subtype": ["ductal", "lobular", "ductal"]})
    assert list(patients_in_multiple_subtypes(df).index) == ["x"]


def test_cv_experiment_end_to_end(tmp_path, monkeypatch):
    """Run Experiment 2 on fake files with a fake model (no GPU, no torch)."""
    import sys
    import types

    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    root.mkdir()
    for p in range(12):
        label, sub = ("M", "DC") if p % 3 else ("B", "F")
        pid = f"14-{1000 + p}" + ("AB" if p < 2 else "")
        for mag in (40, 200):
            for i in range(4):
                (root / f"SOB_{label}_{sub}-{pid}-{mag}-{i:03d}.png").write_bytes(b"")
    fake = types.ModuleType("breakhis.train")
    fake.train_and_predict = lambda train_df, test_df, **kw: np.clip(
        test_df["label"].to_numpy() * 0.7 + rng.uniform(0, 0.3, len(test_df)), 0, 1)
    monkeypatch.setitem(sys.modules, "breakhis.train", fake)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["cv", "--data_dir", str(root), "--out_dir", str(out),
                                      "--magnifications", "40", "200", "--n_folds", "3"])
    from breakhis.experiments import cv
    cv.main()
    summary = pd.read_csv(out / "cv_summary.csv")
    assert len(summary) == 4  # 2 groupings x 2 magnifications
    assert (summary["n_test_patients"] == 12).all()
    assert (out / "patient_base" / "200x" / "per_subtype.csv").exists()
    cv.main()  # second run resumes from saved folds
    assert len(pd.read_csv(out / "cv_summary.csv")) == 4
