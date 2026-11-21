"""Train/test splits: the leaky way (by image) and the honest way (by patient)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def image_level_split(df: pd.DataFrame, seed: int, n_folds: int = 5, fold: int = 0):
    """LEAKY on purpose: images from the same patient can land in train AND test."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = list(skf.split(df, df["label"]))[fold]
    return np.asarray(train_idx), np.asarray(test_idx)


def patient_level_split(df: pd.DataFrame, seed: int, n_folds: int = 5, fold: int = 0):
    """Honest: every patient is entirely in train or entirely in test."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = list(sgkf.split(df, df["label"], groups=df["patient_id"]))[fold]
    return np.asarray(train_idx), np.asarray(test_idx)


def patient_overlap(df: pd.DataFrame, train_idx, test_idx) -> int:
    """How many patients appear on both sides of the split (should be 0 for honest splits)."""
    train_p = set(df.iloc[train_idx]["patient_id"])
    test_p = set(df.iloc[test_idx]["patient_id"])
    return len(train_p & test_p)


SPLITTERS = {"image": image_level_split, "patient": patient_level_split}


def patient_cv_folds(df: pd.DataFrame, group_col: str = "patient_id", n_folds: int = 5,
                     seed: int = 0):
    """All folds of a patient-grouped, class-stratified CV: each patient is tested exactly once.

    group_col="patient_base" groups IDs that share a number (e.g. 14-22549AB and 14-22549CD).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in sgkf.split(df, df["label"], groups=df[group_col]):
        yield np.asarray(train_idx), np.asarray(test_idx)


def group_overlap(df: pd.DataFrame, train_idx, test_idx, group_col: str) -> int:
    """How many groups (e.g. patient numbers) appear on both sides of a split."""
    return len(set(df.iloc[train_idx][group_col]) & set(df.iloc[test_idx][group_col]))
