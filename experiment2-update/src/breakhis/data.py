"""Load BreakHis image metadata from file names.

BreakHis file names encode everything we need, e.g.
    SOB_M_DC-14-2523-400-005.png
    -> biopsy=SOB, label=M (malignant), subtype=DC, patient=14-2523,
       magnification=400, sequence=005
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<biopsy>[A-Z]+)_(?P<label>[BM])_(?P<subtype>[A-Z]+)-"
    r"(?P<patient>\d+-[0-9A-Z]+)-(?P<magnification>\d+)-(?P<seq>\d+)\.png$"
)

SUBTYPES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes_tumor", "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma", "LC": "lobular_carcinoma", "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}


def parse_filename(name: str) -> dict | None:
    """Return metadata for one BreakHis file name, or None if it doesn't match."""
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    d = m.groupdict()
    return {
        "label": 1 if d["label"] == "M" else 0,  # 1 = malignant
        "subtype": SUBTYPES.get(d["subtype"], d["subtype"]),
        "patient_id": d["patient"],
        # Same number with different letters (14-22549AB / 14-22549CD) may be slides from one person.
        "patient_base": re.sub(r"[A-Z]+$", "", d["patient"]),
        "magnification": int(d["magnification"]),
        "seq": int(d["seq"]),
    }


def build_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Scan a BreakHis folder (any layout) and return one row per image."""
    rows, skipped = [], 0
    for path in sorted(Path(data_dir).rglob("*.png")):
        meta = parse_filename(path.name)
        if meta is None:
            skipped += 1
            continue
        rows.append({"path": str(path), **meta})
    if not rows:
        raise FileNotFoundError(f"No BreakHis .png files found under {data_dir}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    if skipped:
        print(f"[build_metadata] skipped {skipped} files with unexpected names")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Images and patients per magnification and class — check against the paper."""
    imgs = df.pivot_table(index="magnification", columns="label", values="path",
                          aggfunc="count").rename(columns={0: "benign_imgs", 1: "malignant_imgs"})
    pats = df.groupby("label")["patient_id"].nunique().rename({0: "benign", 1: "malignant"})
    print("Patients per class:", pats.to_dict(), "| total:", df["patient_id"].nunique())
    return imgs


def patients_in_multiple_subtypes(df: pd.DataFrame) -> pd.Series:
    """Patient IDs filed under more than one tumor type (explains 81 vs 82 patients)."""
    subtypes = df.groupby("patient_id")["subtype"].unique()
    return subtypes[subtypes.apply(len) > 1]


def id_report(df: pd.DataFrame) -> None:
    """Print the two patient-ID quirks we check before trusting any split."""
    print("Unique patient IDs:", df["patient_id"].nunique(),
          "| unique patient numbers (letters stripped):", df["patient_base"].nunique())
    multi = patients_in_multiple_subtypes(df)
    print("IDs under more than one tumor type:",
          {k: list(v) for k, v in multi.items()} if len(multi) else "none")
    shared = df.groupby("patient_base")["patient_id"].unique()
    shared = shared[shared.apply(len) > 1]
    print(f"Numbers shared by several IDs ({len(shared)}):")
    for base in shared.index:
        labels = df[df["patient_base"] == base].groupby("patient_id")["subtype"].first().to_dict()
        print(f"  {base}: {labels}")
