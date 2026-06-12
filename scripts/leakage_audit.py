"""Stand-alone patient-leakage audit across model families.

Quantifies how much sample-level cross-validation overstates performance on the
HMP2 cohort by comparing three evaluation schemes for every model family:

- naive k-fold (StratifiedKFold) — leaks patients,
- patient-grouped k-fold (GroupKFold),
- leave-one-patient-out (LeaveOneGroupOut).

Outputs a results table (CSV + LaTeX), a figure of the bootstrap distribution of
the naive-minus-grouped AUC gap, and the within-patient intraclass correlation
(ICC) of the outcome, which measures how much patient structure exists to leak.

Example
-------
::

    python scripts/leakage_audit.py --data data/ --output results/leakage/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from models import get_model_grid  # noqa: E402
from preprocessing import MicrobiomePreprocessor, load_hmp2  # noqa: E402

logger = logging.getLogger("leakage_audit")

AUDIT_ESTIMATOR_OVERRIDES = {"n_estimators": 200}


def build_pipeline(model_name: str, base_estimator) -> Pipeline:
    """Wrap a base estimator with the right preprocessing for the audit."""
    transform = "none" if model_name == "corf" else "clr"
    prep = MicrobiomePreprocessor(prevalence_threshold=0.10, transform_method=transform)
    return Pipeline([("prep", prep), ("clf", base_estimator)])


def oof_proba(estimator, X, y, cv, groups=None) -> np.ndarray:
    """Pooled out-of-fold positive-class probabilities for a CV scheme."""
    proba = cross_val_predict(estimator, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def icc1(y: pd.Series, groups: pd.Series) -> dict[str, float]:
    """One-way ICC(1) of a binary outcome within patient groups.

    Parameters
    ----------
    y : pandas.Series
        Outcome (0/1) per sample.
    groups : pandas.Series
        Patient id per sample.

    Returns
    -------
    dict
        ``icc``, ``ms_between``, ``ms_within``, and mean group size ``k``.

    Notes
    -----
    Because diagnosis is constant within a patient, the within-group mean square
    is ~0 and the ICC is ~1.0 — the quantitative statement that essentially all
    outcome variance lives *between* patients, which is exactly why patient
    leakage is so damaging here.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "g": np.asarray(groups)})
    grand = df["y"].mean()
    sizes = df.groupby("g").size()
    group_means = df.groupby("g")["y"].mean()
    a = len(group_means)
    n = len(df)
    k = sizes.mean()

    ss_between = float((sizes * (group_means - grand) ** 2).sum())
    ss_within = float(
        sum(((df.loc[df.g == g, "y"] - group_means[g]) ** 2).sum() for g in group_means.index)
    )
    ms_between = ss_between / (a - 1) if a > 1 else np.nan
    ms_within = ss_within / (n - a) if n > a else 0.0
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 1.0
    return {"icc": float(icc), "ms_between": ms_between, "ms_within": ms_within, "k": float(k)}


def bootstrap_delta(
    naive_p: np.ndarray, grouped_p: np.ndarray, y: np.ndarray,
    n_bootstrap: int, seed: int,
) -> np.ndarray:
    """Bootstrap the naive-minus-grouped AUC gap from pooled OOF predictions."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], naive_p[idx]) - roc_auc_score(y[idx], grouped_p[idx]))
    return np.array(deltas)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=None, help="Data directory.")
    p.add_argument("--output", required=True, type=Path, help="Results output directory.")
    p.add_argument("--task", default="cd_vs_nonibd")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--models", nargs="+", default=["logreg", "rf", "xgb", "corf"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_hmp2(task=args.task, data_dir=args.data)
    y_arr = np.asarray(y)
    registry = get_model_grid()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    gkf = GroupKFold(n_splits=args.n_splits)
    logo = LeaveOneGroupOut()

    rows: list[dict[str, object]] = []
    bootstrap_by_model: dict[str, np.ndarray] = {}
    for name in args.models:
        base, _ = registry[name]
        if "n_estimators" in base.get_params():
            base = base.set_params(**AUDIT_ESTIMATOR_OVERRIDES)
        pipe = build_pipeline(name, base)
        logger.info("Auditing %s ...", name)

        naive_p = oof_proba(pipe, X, y, cv=skf)
        grouped_p = oof_proba(pipe, X, y, cv=gkf, groups=groups)
        lopo_p = oof_proba(pipe, X, y, cv=logo, groups=groups)

        naive_auc = roc_auc_score(y_arr, naive_p)
        grouped_auc = roc_auc_score(y_arr, grouped_p)
        lopo_auc = roc_auc_score(y_arr, lopo_p)
        rows.append({
            "model": name,
            "naive_kfold_auc": round(naive_auc, 3),
            "grouped_kfold_auc": round(grouped_auc, 3),
            "leave_one_patient_out_auc": round(lopo_auc, 3),
            "leakage_delta": round(naive_auc - grouped_auc, 3),
        })
        bootstrap_by_model[name] = bootstrap_delta(naive_p, grouped_p, y_arr, args.n_bootstrap, args.seed)

    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(args.output / "leakage_results.csv")
    (args.output / "leakage_results.tex").write_text(
        results.to_latex(caption="Cross-validation scheme comparison (ROC AUC).", label="tab:leakage")
    )
    logger.info("\n%s", results.to_string())

    # Bootstrap leakage-gap distribution figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, deltas in bootstrap_by_model.items():
        if deltas.size:
            ax.hist(deltas, bins=40, alpha=0.5, label=f"{name} (mean {deltas.mean():+.3f})")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Naive − grouped AUC (bootstrap resamples)")
    ax.set_ylabel("Frequency")
    ax.set_title("Patient-leakage gap across bootstrap resamples", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "leakage_distribution.png", dpi=150)
    plt.close(fig)

    icc = icc1(y, groups)
    icc["interpretation"] = (
        "ICC ~ 1.0: diagnosis is constant within each patient, so essentially "
        "all outcome variance is between patients — maximal structure for "
        "sample-level CV to leak."
    )
    (args.output / "icc.json").write_text(json.dumps(icc, indent=2))
    logger.info("Within-patient outcome ICC(1) = %.3f (k=%.2f)", icc["icc"], icc["k"])
    logger.info("Done. Results in %s", args.output)


if __name__ == "__main__":
    main()
