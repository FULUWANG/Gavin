"""Common credit risk metrics for analysis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ks_score(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """Calculate the Kolmogorov-Smirnov statistic for binary credit models."""
    frame = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    if frame["y"].nunique() != 2:
        raise ValueError("ks_score requires a binary target with both classes present.")

    ordered = frame.sort_values("score", ascending=False)
    positives = ordered["y"].sum()
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ks_score requires at least one positive and one negative sample.")

    cum_pos = ordered["y"].cumsum() / positives
    cum_neg = (1 - ordered["y"]).cumsum() / negatives
    return float((cum_pos - cum_neg).abs().max())


def auc_score(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    """Calculate AUC with light input cleanup."""
    frame = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    if frame["y"].nunique() != 2:
        raise ValueError("auc_score requires a binary target with both classes present.")
    return float(roc_auc_score(frame["y"], frame["score"]))


def psi(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, bins: int = 10) -> float:
    """Population stability index using quantile bins from the expected sample."""
    expected_s = pd.Series(expected).dropna()
    actual_s = pd.Series(actual).dropna()
    if expected_s.empty or actual_s.empty:
        raise ValueError("psi requires non-empty expected and actual samples.")

    _, edges = pd.qcut(expected_s, q=bins, retbins=True, duplicates="drop")
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_dist = pd.cut(expected_s, bins=edges).value_counts(normalize=True, sort=False)
    actual_dist = pd.cut(actual_s, bins=edges).value_counts(normalize=True, sort=False)

    expected_dist = expected_dist.clip(lower=1e-6)
    actual_dist = actual_dist.reindex(expected_dist.index, fill_value=1e-6).clip(lower=1e-6)
    return float(((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum())

