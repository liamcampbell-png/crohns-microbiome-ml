"""Tests for :mod:`evaluation`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from evaluation import bootstrap_ci, cross_validate_grouped, permutation_test


def test_grouped_cv_returns_correct_number_of_folds(grouped_dataset) -> None:
    """Both grouped and naive results have one metric value per fold."""
    X, y, groups = grouped_dataset
    results = cross_validate_grouped(
        LogisticRegression(max_iter=1000), X, y, groups, n_splits=3
    )
    assert set(results) == {"grouped", "naive"}
    for scheme in results.values():
        assert set(scheme) >= {"auc", "auprc", "sensitivity", "specificity", "brier"}
        assert len(scheme["auc"]) == 3


def test_bootstrap_ci_contains_point_estimate() -> None:
    """The bootstrap CI brackets the observed metric on synthetic scores."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    # Separable-ish scores: positives shifted up.
    y_score = np.concatenate([rng.normal(0.3, 0.1, 50), rng.normal(0.7, 0.1, 50)])
    observed = roc_auc_score(y_true, y_score)
    lo, hi = bootstrap_ci(roc_auc_score, y_true, y_score, n_resamples=500, random_state=0)
    assert lo <= observed <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_perfect_separation() -> None:
    """Perfectly separable scores give an AUC CI pinned at 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    lo, hi = bootstrap_ci(roc_auc_score, y_true, y_score, n_resamples=200, random_state=0)
    assert lo == 1.0 and hi == 1.0


def test_permutation_test_nonsignificant_on_random_labels() -> None:
    """With random features and labels, the permutation p-value is not significant."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.integers(0, 2, size=60))
    result = permutation_test(
        LogisticRegression(max_iter=1000), X, y, groups=None,
        n_permutations=50, n_splits=5, random_state=3,
    )
    assert result["p_value"] > 0.05
    assert result["null_scores"].shape == (50,)
