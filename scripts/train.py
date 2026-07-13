"""Train and honestly evaluate a Crohn's-vs-healthy microbiome classifier.

Loads the HMP2 16S data, builds a preprocessing + model pipeline, evaluates it
with patient-grouped cross-validation alongside the naive sample-level baseline
(so the leakage gap is always reported), runs a label-permutation test, tunes
hyperparameters with Optuna under a time or trial budget, retrains on all data,
and persists the model plus a leakage report and ROC curve.

Examples
--------
::

    python scripts/train.py --model xgb --cv grouped --output models/xgb_v1/
    python scripts/train.py --model rf --n-trials 40 --output models/rf_v1/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never display
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from evaluation import (  # noqa: E402
    LeakageReport,
    bootstrap_ci,
    cross_validate_grouped,
    permutation_test,
)
from models import get_model_grid, save_model, train_with_cv  # noqa: E402
from preprocessing import MicrobiomePreprocessor, load_hmp2  # noqa: E402

logger = logging.getLogger("train")


def build_pipeline(model_name: str, base_estimator) -> Pipeline:
    """Wrap a base estimator in the appropriate preprocessing.

    The composition-aware forest transforms internally, so it receives raw
    (prevalence-filtered) counts; every other model receives CLR features.

    Parameters
    ----------
    model_name : str
        Registry key (``"logreg"``, ``"rf"``, ``"xgb"``, ``"corf"``).
    base_estimator : sklearn estimator
        The classifier to place at the end of the pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``prep`` + ``clf`` pipeline.
    """
    transform = "none" if model_name == "corf" else "clr"
    prep = MicrobiomePreprocessor(prevalence_threshold=0.10, transform_method=transform)
    return Pipeline([("prep", prep), ("clf", base_estimator)])


def tune(
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X,
    y,
    groups,
    cv_kind: str,
    n_splits: int,
    n_trials: int | None,
    time_budget: int,
    seed: int,
) -> dict:
    """Optuna hyperparameter search over the registry grid.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline whose ``clf`` step is tuned.
    param_grid : dict
        Bare-parameter grid from the registry; keys are prefixed with ``clf__``.
    X, y, groups : array-like
        Data and patient groups.
    cv_kind : {"grouped", "naive"}
        Cross-validation scheme used for the objective.
    n_splits : int
        Folds for the objective (capped at 3 for speed).
    n_trials : int or None
        Trial budget; if ``None``, ``time_budget`` seconds is used instead.
    time_budget : int
        Wall-clock budget in seconds when ``n_trials`` is ``None``.
    seed : int
        Sampler seed.

    Returns
    -------
    dict
        Best ``clf__*`` parameters found.
    """
    tune_splits = min(n_splits, 3)

    def objective(trial: optuna.Trial) -> float:
        params = {
            f"clf__{key}": trial.suggest_categorical(key, values)
            for key, values in param_grid.items()
        }
        candidate = pipeline.set_params(**params)
        if cv_kind == "grouped":
            cv = GroupKFold(n_splits=tune_splits).split(X, y, groups)
        else:
            cv = StratifiedKFold(n_splits=tune_splits, shuffle=True, random_state=seed).split(X, y)
        scores = cross_val_score(candidate, X, y, cv=list(cv), scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    if n_trials is not None:
        study.optimize(objective, n_trials=n_trials)
    else:
        study.optimize(objective, timeout=time_budget)
    logger.info(
        "Optuna: %d trials, best objective AUC %.3f", len(study.trials), study.best_value
    )
    return {f"clf__{k}": v for k, v in study.best_params.items()}


def save_roc(y_true, y_score, path: Path, title: str) -> None:
    """Save a single ROC curve PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["logreg", "rf", "xgb", "corf"], default="rf")
    p.add_argument("--task", default="cd_vs_nonibd")
    p.add_argument("--cv", choices=["grouped", "naive"], default="grouped",
                   help="CV scheme used for tuning and the headline score.")
    p.add_argument("--output", required=True, type=Path, help="Output model directory.")
    p.add_argument("--data", type=Path, default=None, help="Override data/ directory.")
    p.add_argument("--n-trials", type=int, default=None, help="Optuna trial budget.")
    p.add_argument("--time-budget", type=int, default=300, help="Optuna seconds (if no --n-trials).")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--no-tune", action="store_true", help="Skip Optuna; use registry defaults.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    X, y, groups = load_hmp2(task=args.task, data_dir=args.data)
    base_estimator, grid = get_model_grid()[args.model]
    pipeline = build_pipeline(args.model, base_estimator)

    if args.no_tune:
        best_params: dict = {}
    else:
        best_params = tune(
            pipeline, grid, X, y, groups, args.cv, args.n_splits,
            args.n_trials, args.time_budget, args.seed,
        )
        pipeline.set_params(**best_params)

    cv_results = cross_validate_grouped(pipeline, X, y, groups, n_splits=args.n_splits, random_state=args.seed)
    trained = train_with_cv(pipeline, X, y, groups, n_splits=args.n_splits, random_state=args.seed)

    ci = bootstrap_ci(roc_auc_score, y, trained["oof_proba"], n_resamples=1000, random_state=args.seed)
    perm = permutation_test(
        pipeline, X, y, groups, n_permutations=args.permutations,
        n_splits=args.n_splits, random_state=args.seed,
    )
    report = LeakageReport.from_cv(cv_results, p_value=perm["p_value"], bootstrap_ci=ci)
    logger.info("\n%s", report)

    out = save_model(
        trained["model"],
        args.output,
        metadata={
            "model": args.model,
            "task": args.task,
            "best_params": best_params,
            "grouped_cv_auc": report.grouped_auc,
            "naive_cv_auc": report.naive_auc,
            "n_samples": int(len(y)),
            "n_participants": int(groups.nunique()),
            "feature_names": list(X.columns),
        },
    )
    (out / "leakage_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    save_roc(
        y, trained["oof_proba"], out / "roc_curve.png",
        title=f"{args.model} — grouped OOF ROC (CD vs healthy)",
    )
    logger.info("Done. Artifacts in %s", out)


if __name__ == "__main__":
    main()
