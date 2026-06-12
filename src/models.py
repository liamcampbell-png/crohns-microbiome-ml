"""Model registry and training utilities.

Provides a small registry of classifier families with sensible hyperparameter
grids, a composition-aware Random Forest, grouped-CV training that returns
out-of-fold predictions, isotonic calibration, and model persistence with a
metadata sidecar.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from xgboost import XGBClassifier

__all__ = [
    "SEED",
    "CompositionAwareRF",
    "get_model_grid",
    "train_with_cv",
    "calibrate_model",
    "save_model",
    "load_model",
]

SEED: int = 42

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Composition-aware Random Forest
# --------------------------------------------------------------------------- #
def _array_clr(X: np.ndarray, pseudocount: float) -> np.ndarray:
    """CLR transform on a non-negative array (rows = samples)."""
    logX = np.log(np.asarray(X, dtype=float) + pseudocount)
    return logX - logX.mean(axis=1, keepdims=True)


class CompositionAwareRF(BaseEstimator, ClassifierMixin):
    """Random Forest that CLR-transforms its inputs internally.

    Unlike a plain RF fed pre-transformed data, this estimator expects
    non-negative count or relative-abundance input and applies the centered
    log-ratio transform itself. That makes it a drop-in classifier for
    compositional data when the surrounding pipeline has *not* already
    transformed the features (use ``transform_method="none"`` or ``"relative"``
    upstream).

    Parameters
    ----------
    n_estimators : int, default=500
        Number of trees.
    max_features : {"sqrt", "log2"} or float, default="sqrt"
        Features considered per split.
    min_samples_leaf : int, default=1
        Minimum samples per leaf.
    pseudocount : float, default=0.5
        Pseudocount for the internal CLR.
    class_weight : str or dict or None, default="balanced"
        Passed to the underlying forest.
    random_state : int, default=:data:`SEED`
        RNG seed.
    n_jobs : int, default=-1
        Parallelism for the forest.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_features: str | float = "sqrt",
        min_samples_leaf: int = 1,
        pseudocount: float = 0.5,
        class_weight: str | dict | None = "balanced",
        random_state: int = SEED,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.pseudocount = pseudocount
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> "CompositionAwareRF":
        """Fit the internal forest on CLR-transformed ``X``."""
        Xc = _array_clr(np.asarray(X), self.pseudocount)
        self._forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(Xc, y)
        self.classes_ = self._forest.classes_
        self.n_features_in_ = Xc.shape[1]
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self._forest.predict(_array_clr(np.asarray(X), self.pseudocount))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self._forest.predict_proba(_array_clr(np.asarray(X), self.pseudocount))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity-based importances from the internal forest."""
        return self._forest.feature_importances_


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def get_model_grid() -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Return the model registry.

    Returns
    -------
    dict
        Maps a short name to ``(estimator, param_grid)``. Param-grid keys are
        bare estimator parameter names (no pipeline prefix); callers tuning
        inside a pipeline should prefix them (e.g. ``"clf__C"``).

    Notes
    -----
    Names: ``"logreg"`` (L1/L2 logistic regression), ``"rf"`` (Random Forest),
    ``"xgb"`` (XGBoost), ``"corf"`` (:class:`CompositionAwareRF`).
    """
    registry: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]] = {
        "logreg": (
            LogisticRegression(
                solver="liblinear", max_iter=5000,
                class_weight="balanced", random_state=SEED,
            ),
            {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 0.5, 1.0, 10.0]},
        ),
        "rf": (
            RandomForestClassifier(
                random_state=SEED, class_weight="balanced", n_jobs=-1,
            ),
            {
                "n_estimators": [300, 500],
                "max_features": ["sqrt", 0.3],
                "min_samples_leaf": [1, 3, 5],
            },
        ),
        "xgb": (
            XGBClassifier(
                eval_metric="logloss", random_state=SEED, n_jobs=-1,
                n_estimators=300, tree_method="hist",
            ),
            {
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "scale_pos_weight": [0.5, 1.0],
            },
        ),
        "corf": (
            CompositionAwareRF(random_state=SEED),
            {
                "n_estimators": [300, 500],
                "max_features": ["sqrt", 0.3],
                "min_samples_leaf": [1, 3, 5],
            },
        ),
    }
    return registry


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _extract_importances(model: BaseEstimator, feature_names: list[str]) -> pd.Series | None:
    """Pull feature importances/coefficients from a model or pipeline tail."""
    est = model
    if hasattr(est, "steps"):  # sklearn Pipeline
        est = est.steps[-1][1]
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        values = np.abs(np.asarray(est.coef_)).ravel()
    else:
        return None
    if len(values) != len(feature_names):
        feature_names = [f"f{i}" for i in range(len(values))]
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def train_with_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = SEED,
) -> dict[str, Any]:
    """Fit a model and collect grouped out-of-fold predictions.

    Parameters
    ----------
    estimator : sklearn estimator
        Classifier or pipeline (cloned for CV, then refit on all data).
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Binary labels.
    groups : pandas.Series
        Patient ids for :class:`~sklearn.model_selection.GroupKFold`.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=:data:`SEED`
        Seed (forwarded to the final clone via its own params).

    Returns
    -------
    dict
        ``model`` (fitted on full data), ``oof_proba`` (pandas.Series aligned to
        ``y``), ``feature_importances`` (pandas.Series or ``None``), and
        ``cv_auc`` (mean grouped out-of-fold ROC AUC).
    """
    from sklearn.metrics import roc_auc_score

    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    oof = cross_val_predict(clone(estimator), X, y, cv=splits, method="predict_proba")
    oof_proba = pd.Series(oof[:, 1], index=y.index, name="oof_proba")
    cv_auc = float(roc_auc_score(y, oof_proba))

    model = clone(estimator).fit(X, y)
    if hasattr(model, "steps"):  # pipeline: names come from the transformed space
        try:
            feature_names = list(model[:-1].get_feature_names_out())
        except (AttributeError, ValueError):
            feature_names = list(X.columns)
    else:
        feature_names = list(X.columns)
    importances = _extract_importances(model, feature_names)

    logger.info("train_with_cv: grouped OOF AUC %.3f", cv_auc)
    return {
        "model": model,
        "oof_proba": oof_proba,
        "feature_importances": importances,
        "cv_auc": cv_auc,
    }


def calibrate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wrap a classifier in isotonic (or sigmoid) probability calibration.

    Parameters
    ----------
    estimator : sklearn estimator
        Base classifier (unfitted).
    X, y : pandas.DataFrame, pandas.Series
        Training data.
    method : {"isotonic", "sigmoid"}, default="isotonic"
        Calibration method.
    cv : int, default=5
        Internal cross-validation folds for calibration. Note: this is
        stratified, not patient-grouped — calibrating with grouped folds would
        require a custom CV splitter and is out of scope here.

    Returns
    -------
    sklearn.calibration.CalibratedClassifierCV
        The fitted, calibrated classifier.
    """
    calibrated = CalibratedClassifierCV(clone(estimator), method=method, cv=cv)
    calibrated.fit(X, y)
    logger.info("calibrate_model: fitted %s calibration (cv=%d)", method, cv)
    return calibrated


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(
    model: BaseEstimator,
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Pickle a model with a JSON metadata sidecar.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model to persist.
    path : pathlib.Path or str
        Output *directory*; created if absent. Writes ``model.pkl`` and
        ``metadata.json`` inside it.
    metadata : dict, optional
        Extra metadata (e.g. ``feature_names``, ``cv_auc``, ``n_samples``);
        merged with an auto-added ``training_date``.

    Returns
    -------
    pathlib.Path
        The directory written to.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")

    meta: dict[str, Any] = {"training_date": datetime.now(timezone.utc).isoformat()}
    if metadata:
        meta.update(metadata)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("save_model: wrote %s and metadata.json", out / "model.pkl")
    return out


def load_model(path: Path | str) -> tuple[BaseEstimator, dict[str, Any]]:
    """Load a model and its metadata saved by :func:`save_model`.

    Parameters
    ----------
    path : pathlib.Path or str
        Directory containing ``model.pkl`` and ``metadata.json``.

    Returns
    -------
    tuple
        ``(model, metadata)``.
    """
    src = Path(path)
    model = joblib.load(src / "model.pkl")
    meta_path = src / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, metadata
