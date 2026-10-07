"""Data loading utilities for In-hospital Death modelling experiments.

This module provides clean access to the frozen 1,474-feature matrices and the
three-way split in `preprocessed/task2_mortality/` (written by `preprocess.py`; frozen
2026-09-21, SHA-256 manifest alongside).

Usage:
    from src.task2_mortality.data import load_data, get_split, get_feature_names
    
    labels_df, X_unimputed, X_imputed_splits = load_data()
    X_train, y_train = get_split('train')
    feature_names = get_feature_names()
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "preprocessed" / "task2_mortality"

SplitName = Literal["train", "validation", "test"]

_FEATURE_NAMES = None
_LABELS_DF = None
_X_UNIMPUTED = None
_X_IMPUTED_SPLITS = None


def get_feature_names() -> list[str]:
    """Return the 1,474 feature column names (excludes RecordID).
    
    Returns:
        List of feature column names in the order they appear in the data files.
    """
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        _FEATURE_NAMES = _load_feature_names()
    return _FEATURE_NAMES


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[SplitName, pd.DataFrame]]:
    """Load all data files: labels, unimputed features, and imputed splits.
    
    Returns:
        tuple of:
            - labels_df: DataFrame with RecordID, In-hospital_death, split, and baselines
            - X_unimputed: DataFrame with RecordID + 1,474 unimputed features
            - X_imputed_splits: dict mapping split name to imputed feature DataFrame
    """
    global _LABELS_DF, _X_UNIMPUTED, _X_IMPUTED_SPLITS
    
    if _LABELS_DF is None:
        _LABELS_DF = pd.read_csv(DATA_DIR / "labels_and_baselines.csv")
    
    if _X_UNIMPUTED is None:
        _X_UNIMPUTED = pd.read_csv(DATA_DIR / "patient_features_unimputed.csv.gz")
    
    if _X_IMPUTED_SPLITS is None:
        _X_IMPUTED_SPLITS = {
            "train": pd.read_csv(DATA_DIR / "X_train_imputed.csv.gz"),
            "validation": pd.read_csv(DATA_DIR / "X_validation_imputed.csv.gz"),
            "test": pd.read_csv(DATA_DIR / "X_test_imputed.csv.gz"),
        }
    
    return _LABELS_DF, _X_UNIMPUTED, _X_IMPUTED_SPLITS


def get_split(split_name: SplitName) -> tuple[pd.DataFrame, pd.Series]:
    """Get X and y for a specific split.
    
    Args:
        split_name: One of 'train', 'validation', 'test'
    
    Returns:
        tuple of:
            - X: DataFrame with 1,474 imputed features (no RecordID)
            - y: Series with In-hospital_death labels
    """
    labels_df, _, X_imputed_splits = load_data()
    
    X = X_imputed_splits[split_name].copy()
    record_ids = X["RecordID"]
    X = X.drop(columns=["RecordID"])
    
    split_labels = labels_df[labels_df["split"] == split_name].copy()
    split_labels = split_labels.merge(
        pd.DataFrame({"RecordID": record_ids}),
        on="RecordID", how="right", validate="one_to_one"
    )
    
    if len(split_labels) != len(X):
        raise AssertionError(
            f"label/feature length mismatch for {split_name}: "
            f"{len(split_labels)} labels vs {len(X)} feature rows"
        )
    
    y = split_labels["In-hospital_death"]
    
    return X, y


def _load_feature_names() -> list[str]:
    """Load feature names from the unimputed data header (internal use)."""
    import gzip
    with gzip.open(DATA_DIR / "patient_features_unimputed.csv.gz", "rt") as f:
        header = f.readline().strip()
        cols = header.split(",")
    return cols[1:]  # Exclude RecordID
