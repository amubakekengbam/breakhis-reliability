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


def subtype_metrics(df_pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Per tumor type: how often the model gets it right, by image and by patient.

    For malignant types this is sensitivity (cancer caught); for benign types, specificity.
    """
    d = df_pred.assign(correct=((df_pred["prob"] >= threshold).astype(int) == df_pred["label"]))
    per_patient = d.groupby(["subtype", "patient_id"]).agg(
        label=("label", "first"), prob=("prob", "mean"))
    per_patient["correct"] = (per_patient["prob"] >= threshold).astype(int) == per_patient["label"]
    out = pd.DataFrame({
        "class": d.groupby("subtype")["label"].first().map({0: "benign", 1: "malignant"}),
        "images": d.groupby("subtype").size(),
        "image_correct_rate": d.groupby("subtype")["correct"].mean(),
        "patients": per_patient.groupby("subtype").size(),
        "patients_correct": per_patient.groupby("subtype")["correct"].sum(),
    })
    out["patient_correct_rate"] = out["patients_correct"] / out["patients"]
    return out.sort_values(["class", "patients"], ascending=[True, False])


def pooled_summary(df_pred: pd.DataFrame, seed: int = 0) -> dict:
    """Metrics over out-of-fold predictions for every patient, with patient-bootstrap CIs."""
    row = image_metrics(df_pred["label"], df_pred["prob"]) | patient_metrics(df_pred)
    for metric in ["balanced_accuracy", "auroc"]:
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_ci(df_pred, metric, seed=seed)
    return row
