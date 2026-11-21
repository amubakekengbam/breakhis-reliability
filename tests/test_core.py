import numpy as np
import pandas as pd

from breakhis.data import build_metadata, parse_filename
from breakhis.metrics import bootstrap_ci, image_metrics, patient_metrics
from breakhis.splits import image_level_split, patient_level_split, patient_overlap


def test_parse_malignant():
    m = parse_filename("SOB_M_DC-14-2523-400-005.png")
    assert m == {"label": 1, "subtype": "ductal_carcinoma", "patient_id": "14-2523",
                 "magnification": 400, "seq": 5}


def test_parse_benign_with_letters_in_patient_id():
    m = parse_filename("SOB_B_A-14-22549AB-40-001.png")
    assert m["label"] == 0 and m["patient_id"] == "14-22549AB" and m["magnification"] == 40


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
