"""Experiment 1: does an image-level split inflate accuracy vs a patient-level split?

Example (Colab):
    python -m breakhis.experiments.leakage --data_dir /content/BreaKHis_v1 \
        --out_dir /content/drive/MyDrive/breakhis-results --seeds 0 1 2 --epochs 5 --magnification 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from breakhis.data import build_metadata
from breakhis.metrics import bootstrap_ci, image_metrics, patient_metrics
from breakhis.splits import SPLITTERS, patient_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--magnification", type=int, default=None,
                    help="Use one magnification (40/100/200/400). Default: all.")
    args = ap.parse_args()

    from breakhis.train import (
        train_and_predict,  # imported here so tests don't need torch
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_metadata(args.data_dir)
    if args.magnification:
        df = df[df["magnification"] == args.magnification].reset_index(drop=True)
    print(f"{len(df)} images from {df['patient_id'].nunique()} patients")

    results_path = out / "leakage_results.csv"
    done = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")

    for split_name, splitter in SPLITTERS.items():
        for seed in args.seeds:
            if any(r["split"] == split_name and r["seed"] == seed for r in rows):
                print(f"skip {split_name} seed {seed} (already done)")
                continue
            tr, te = splitter(df, seed=seed)
            train_df, test_df = df.iloc[tr], df.iloc[te].copy()
            print(f"\n== split={split_name} seed={seed} | train {len(train_df)} test {len(test_df)}"
                  f" | patients in both: {patient_overlap(df, tr, te)}")
            test_df["prob"] = train_and_predict(train_df, test_df, seed=seed,
                                                arch=args.arch, epochs=args.epochs)
            test_df.to_csv(out / f"preds_{split_name}_seed{seed}.csv", index=False)
            lo, hi = bootstrap_ci(test_df, "balanced_accuracy", seed=seed)
            row = {"split": split_name, "seed": seed, "arch": args.arch,
                   "magnification": args.magnification or "all",
                   "patients_in_both": patient_overlap(df, tr, te),
                   **image_metrics(test_df["label"], test_df["prob"]),
                   "bal_acc_ci_low": lo, "bal_acc_ci_high": hi,
                   **patient_metrics(test_df)}
            print(json.dumps(row, indent=2))
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)  # save after every run

    res = pd.DataFrame(rows)
    summary = res.groupby("split")[["accuracy", "balanced_accuracy", "auroc",
                                    "patient_balanced_accuracy"]].agg(["mean", "std"]).round(3)
    print("\n=== Summary (mean ± std over seeds) ===\n", summary)
    summary.to_csv(out / "leakage_summary.csv")


if __name__ == "__main__":
    main()
