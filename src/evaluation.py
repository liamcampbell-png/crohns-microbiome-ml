"""Rigorous model-evaluation utilities for the microbiome project.

The headline concern is *patient leakage*: HMP2 contributes several samples per
participant, so sample-level cross-validation overstates how well a model
generalises to new patients. Every routine here is built to make the honest,
patient-grouped number easy to obtain and to compare against the naive one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    permutation_test_score,
)

__all__ = [
    "SEED",
    "LeakageReport",
    "cross_validate_grouped",
    "bootstrap_ci",
    "calibration_analysis",
    "permutation_test",
]

SEED: int = 42

logger = logging.getLogger(__name__)

ScalarMetric = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------- #
# Leakage report
# --------------------------------------------------------------------------- #
@dataclass
class LeakageReport:
    """Summary of the naive-vs-grouped cross-validation gap.

    Parameters
    ----------
    naive_auc : float
        Mean ROC AUC under sample-level (StratifiedKFold) cross-validation.
    grouped_auc : float
        Mean ROC AUC under patient-grouped (GroupKFold) cross-validation.
    delta : float
        ``naive_auc - grouped_auc``; how much leakage inflated the naive score.
    p_value : float or None
        Permutation-test p-value for the grouped AUC, if computed.
    bootstrap_ci : tuple of float or None
        Confidence interval for the grouped AUC, if computed.
    """

    naive_auc: float
    grouped_auc: float
    delta: float
    p_value: float | None = None
    bootstrap_ci: tuple[float, float] | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_cv(
        cls,
        cv_results: dict[str, dict[str, np.ndarray]],
        p_value: float | None = None,
        bootstrap_ci: tuple[float, float] | None = None,
    ) -> "LeakageReport":
        """Build a report from :func:`cross_validate_grouped` output."""
        naive = float(np.mean(cv_results["naive"]["auc"]))
        grouped = float(np.mean(cv_results["grouped"]["auc"]))
        return cls(
            naive_auc=naive,
            grouped_auc=grouped,
            delta=naive - grouped,
            p_value=p_value,
            bootstrap_ci=bootstrap_ci,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of the report."""
        return {
            "naive_auc": self.naive_auc,
            "grouped_auc": self.grouped_auc,
            "delta": self.delta,
            "p_value": self.p_value,
            "bootstrap_ci": list(self.bootstrap_ci) if self.bootstrap_ci else None,
            "extra": self.extra,
        }

    def __str__(self) -> str:
        lines = [
            "Leakage report (CD vs healthy)",
            "------------------------------",
            f"  Naive   k-fold AUC : {self.naive_auc:.3f}",
            f"  Grouped k-fold AUC : {self.grouped_auc:.3f}",
            f"  Leakage delta      : {self.delta:+.3f}",
        ]
        if self.bootstrap_ci is not None:
            lo, hi = self.bootstrap_ci
            lines.append(f"  Grouped 95% CI     : [{lo:.3f}, {hi:.3f}]")
        if self.p_value is not None:
            lines.append(f"  Permutation p      : {self.p_value:.4f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def _scores_from_estimator(est, X: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Return continuous scores and whether they are calibrated probabilities."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1], True
    if hasattr(est, "decision_function"):
        return est.decision_function(X), False
    raise AttributeError("Estimator exposes neither predict_proba nor decision_function.")


def _fold_metrics(
    y_true: np.ndarray, scores: np.ndarray, is_proba: bool, threshold: float
) -> dict[str, float]:
    """Compute AUC, AUPRC, sensitivity, specificity, and Brier for one fold."""
    preds = (scores >= (threshold if is_proba else 0.0)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    brier = brier_score_loss(y_true, scores) if is_proba else np.nan
    return {
        "auc": roc_auc_score(y_true, scores),
        "auprc": average_precision_score(y_true, scores),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": brier,
    }


def _run_cv(estimator, X, y, splits) -> dict[str, np.ndarray]:
    """Run CV over pre-computed (train, test) index splits; collect metrics."""
    rows: list[dict[str, float]] = []
    y_arr = np.asarray(y)
    for train_idx, test_idx in splits:
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y_arr[train_idx])
        scores, is_proba = _scores_from_estimator(est, X.iloc[test_idx])
        rows.append(_fold_metrics(y_arr[test_idx], scores, is_proba, threshold=0.5))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def cross_validate_grouped(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = SEED,
) -> dict[str, dict[str, np.ndarray]]:
    """Cross-validate with both grouped and naive splits in one call.

    Parameters
    ----------
    estimator : sklearn estimator
        A fresh (unfitted) classifier or pipeline; cloned per fold.
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Binary labels.
    groups : pandas.Series
        Patient id per sample (used only for the grouped split).
    n_splits : int, default=5
        Number of folds for both schemes.
    random_state : int, default=:data:`SEED`
        Seed for the shuffled StratifiedKFold (GroupKFold is not randomised).

    Returns
    -------
    dict
        ``{"grouped": {metric: array}, "naive": {metric: array}}`` where each
        metric array has one entry per fold. Metrics: ``auc``, ``auprc``,
        ``sensitivity``, ``specificity``, ``brier``.
    """
    gkf = GroupKFold(n_splits=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grouped = _run_cv(estimator, X, y, list(gkf.split(X, y, groups)))
    naive = _run_cv(estimator, X, y, list(skf.split(X, y)))

    logger.info(
        "cross_validate_grouped: grouped AUC %.3f vs naive AUC %.3f (delta %+.3f)",
        grouped["auc"].mean(), naive["auc"].mean(),
        naive["auc"].mean() - grouped["auc"].mean(),
    )
    return {"grouped": grouped, "naive": naive}


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    metric_fn: ScalarMetric,
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(y_true, y_score) -> float`` (e.g. :func:`roc_auc_score`).
    y_true : array-like
        Ground-truth labels.
    y_score : array-like
        Predicted scores or probabilities.
    n_resamples : int, default=1000
        Number of bootstrap resamples.
    alpha : float, default=0.05
        Two-sided significance level; returns the central ``1 - alpha`` interval.
    random_state : int, default=:data:`SEED`
        RNG seed.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` bounds. Resamples containing a single class are
        skipped.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    estimates: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        estimates.append(metric_fn(y_true[idx], y_score[idx]))
    if not estimates:
        raise ValueError("All bootstrap resamples were single-class; cannot form CI.")
    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return lo, hi


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_analysis(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict[str, object]:
    """Reliability-diagram data with expected and maximum calibration error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of equal-width probability bins.

    Returns
    -------
    dict
        ``prob_true`` / ``prob_pred`` (from :func:`sklearn.calibration.calibration_curve`),
        ``bin_counts``, ``ece`` (expected calibration error) and ``mce`` (maximum
        calibration error).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    counts: list[int] = []
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        counts.append(c)
        if c == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap = abs(acc - conf)
        ece += (c / n) * gap
        mce = max(mce, gap)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": np.array(counts),
        "ece": float(ece),
        "mce": float(mce),
    }


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #
def permutation_test(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    n_permutations: int = 1000,
    scoring: str = "roc_auc",
    n_splits: int = 5,
    random_state: int = SEED,
) -> dict[str, object]:
    """Label-permutation test for the (grouped) cross-validated score.

    Parameters
    ----------
    estimator : sklearn estimator
        Classifier or pipeline to evaluate.
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Binary labels.
    groups : pandas.Series, optional
        Patient ids; if given, a :class:`~sklearn.model_selection.GroupKFold` is
        used so the null distribution respects patient structure.
    n_permutations : int, default=1000
        Number of label shuffles.
    scoring : str, default="roc_auc"
        Any sklearn scoring string.
    n_splits : int, default=5
        Folds per evaluation.
    random_state : int, default=:data:`SEED`
        RNG seed.

    Returns
    -------
    dict
        ``observed`` score, ``p_value``, and the ``null_scores`` array.
    """
    cv = GroupKFold(n_splits=n_splits) if groups is not None else n_splits
    observed, null_scores, p_value = permutation_test_score(
        estimator,
        X,
        np.asarray(y),
        groups=groups,
        cv=cv,
        n_permutations=n_permutations,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info(
        "permutation_test: observed %s=%.3f, p=%.4f (%d permutations)",
        scoring, observed, p_value, n_permutations,
    )
    return {"observed": float(observed), "p_value": float(p_value), "null_scores": null_scores}
