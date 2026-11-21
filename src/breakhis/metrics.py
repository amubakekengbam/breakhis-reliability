"""Metrics at image level and patient level, with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def image_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan")
    return out


def patient_metrics(df_pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Average image probabilities per patient, then score patients.

    df_pred needs columns: patient_id, label, prob.
    """
    per_patient = df_pred.groupby("patient_id").agg(label=("label", "first"), prob=("prob", "mean"))
    m = image_metrics(per_patient["label"], per_patient["prob"], threshold)
    return {f"patient_{k}": v for k, v in m.items()} | {"n_test_patients": len(per_patient)}


def bootstrap_ci(df_pred: pd.DataFrame, metric: str = "balanced_accuracy",
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Patient-level bootstrap: resample PATIENTS (not images), since images are correlated."""
    rng = np.random.default_rng(seed)
    patients = df_pred["patient_id"].unique()
    groups = {p: g for p, g in df_pred.groupby("patient_id")}
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(patients, size=len(patients), replace=True)
        boot = pd.concat([groups[p] for p in sample])
        if boot["label"].nunique() < 2:
            continue
        stats.append(image_metrics(boot["label"], boot["prob"])[metric])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
