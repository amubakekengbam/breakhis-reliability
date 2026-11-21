"""Experiment 2: honest baseline with 5-fold patient-grouped cross-validation.

For each grouping (patient ID, and patient *number* with letters stripped) and each magnification,
every patient is tested exactly once. We pool the out-of-fold predictions and report image-level,
patient-level and per-tumor-type results with patient-bootstrap confidence intervals.

Example (Colab):
    python -m breakhis.experiments.cv --data_dir /content/BreaKHis_v1 \
        --out_dir /content/drive/MyDrive/breakhis-results/cv --epochs 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from breakhis.data import build_metadata, id_report
from breakhis.metrics import pooled_summary, subtype_metrics
from breakhis.splits import group_overlap, patient_cv_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--groups", nargs="+", default=["patient_id", "patient_base"])
    ap.add_argument("--magnifications", type=int, nargs="+", default=[40, 100, 200, 400])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--arch", default="resnet18")
    args = ap.parse_args()

    from breakhis.train import (
        train_and_predict,  # imported here so tests don't need torch
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_df = build_metadata(args.data_dir)
    id_report(all_df)

    summary_rows = []
    for group_col in args.groups:
        for mag in args.magnifications:
            df = all_df[all_df["magnification"] == mag].reset_index(drop=True)
            run_dir = out / group_col / f"{mag}x"
            run_dir.mkdir(parents=True, exist_ok=True)
            fold_preds = []
            for k, (tr, te) in enumerate(patient_cv_folds(df, group_col, args.n_folds, args.seed)):
                fold_file = run_dir / f"fold{k}.csv"
                if fold_file.exists():
                    print(f"skip {group_col} {mag}x fold {k} (already done)")
                    fold_preds.append(pd.read_csv(fold_file))
                    continue
                overlap_ids = group_overlap(df, tr, te, "patient_id")
                overlap_nums = group_overlap(df, tr, te, "patient_base")
                print(f"\n== group={group_col} mag={mag}x fold={k} | train {len(tr)} test {len(te)}"
                      f" | shared IDs: {overlap_ids} | shared numbers: {overlap_nums}")
                test_df = df.iloc[te].copy()
                test_df["prob"] = train_and_predict(df.iloc[tr], test_df, seed=args.seed + k,
                                                    arch=args.arch, epochs=args.epochs)
                test_df["fold"] = k
                test_df["shared_numbers_in_fold"] = overlap_nums
                test_df.to_csv(fold_file, index=False)  # saved after every fold, so runs resume
                fold_preds.append(test_df)

            oof = pd.concat(fold_preds, ignore_index=True)
            oof.to_csv(run_dir / "oof_predictions.csv", index=False)
            subtype_metrics(oof).round(3).to_csv(run_dir / "per_subtype.csv")
            row = {"grouping": group_col, "magnification": mag, "images": len(oof),
                   "shared_numbers_max": int(oof["shared_numbers_in_fold"].max()),
                   **pooled_summary(oof, seed=args.seed)}
            print(json.dumps(row, indent=2))
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(out / "cv_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    cols = ["grouping", "magnification", "accuracy", "balanced_accuracy",
            "balanced_accuracy_ci_low", "balanced_accuracy_ci_high", "auroc",
            "patient_balanced_accuracy", "n_test_patients", "shared_numbers_max"]
    print("\n=== Experiment 2 summary ===\n", summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
