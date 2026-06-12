"""SHAP interpretation for a saved microbiome classifier.

Loads a model saved by ``scripts/train.py``, computes TreeExplainer SHAP values
on the CLR feature space, and writes: a beeswarm summary, waterfall plots for one
Crohn's and one healthy example, per-taxon dependence plots for the top features,
and a ``shap_summary.csv`` ranking taxa with their direction of effect.

Example
-------
::

    python scripts/explain.py --model models/xgb_v1/ --output figures/shap/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from models import CompositionAwareRF, load_model  # noqa: E402
from preprocessing import load_hmp2  # noqa: E402

logger = logging.getLogger("explain")


def _to_clr(frame: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    """CLR-transform a count frame (used to feed the corf internal forest)."""
    log_vals = np.log(frame.astype(float) + pseudocount)
    return log_vals.subtract(log_vals.mean(axis=1), axis=0)


def resolve_tree_and_input(model, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Return the underlying tree model and the matrix SHAP should explain.

    Handles both ordinary pipelines (``prep`` does CLR; explain the CLR features)
    and the composition-aware forest (``prep`` keeps counts; explain the CLR of
    those counts against the internal forest).

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The loaded ``prep`` + ``clf`` pipeline.
    X : pandas.DataFrame
        Raw count matrix from :func:`load_hmp2`.

    Returns
    -------
    tuple
        ``(tree_model, shap_input_df)``.
    """
    prep = model[:-1]
    clf = model[-1]
    transformed = pd.DataFrame(
        prep.transform(X), index=X.index, columns=prep.get_feature_names_out()
    )
    if isinstance(clf, CompositionAwareRF):
        return clf._forest, _to_clr(transformed)
    return clf, transformed


def normalize_shap(values, expected) -> tuple[np.ndarray, float]:
    """Reduce SHAP outputs to a 2-D positive-class array and scalar base value."""
    arr = values[1] if isinstance(values, list) else np.asarray(values)
    if arr.ndim == 3:  # (samples, features, classes)
        arr = arr[:, :, 1]
    if isinstance(expected, (list, np.ndarray)) and np.ndim(expected) > 0:
        base = float(expected[1] if len(expected) > 1 else expected[0])
    else:
        base = float(expected)
    return arr, base


def save_beeswarm(shap_arr: np.ndarray, shap_df: pd.DataFrame, path: Path) -> None:
    """Save a beeswarm summary plot coloured by CLR abundance."""
    plt.figure()
    shap.summary_plot(shap_arr, shap_df, plot_type="dot", max_display=20, show=False)
    plt.title("SHAP — top taxa (CD vs healthy)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_waterfall(
    shap_arr: np.ndarray, base: float, shap_df: pd.DataFrame, row: int, path: Path
) -> None:
    """Save a single-sample waterfall plot."""
    explanation = shap.Explanation(
        values=shap_arr[row],
        base_values=base,
        data=shap_df.iloc[row].to_numpy(),
        feature_names=list(shap_df.columns),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_summary(
    shap_arr: np.ndarray, shap_df: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Rank taxa by mean |SHAP| and annotate direction + (placeholder) taxonomy.

    ``direction`` is ``"+"`` when the taxon is enriched in CD (higher mean CLR in
    CD than healthy) and ``"-"`` when depleted. ``phylum``/``family`` are filled
    with ``"unclassified"`` because the HMP2 OTU ids carry no lineage; pass a
    taxonomy table upstream to populate them.
    """
    mean_abs = np.abs(shap_arr).mean(axis=0)
    cd_mean = shap_df[y.to_numpy() == 1].mean(axis=0)
    healthy_mean = shap_df[y.to_numpy() == 0].mean(axis=0)
    direction = np.where(cd_mean.to_numpy() > healthy_mean.to_numpy(), "+", "-")
    summary = pd.DataFrame(
        {
            "taxon_id": shap_df.columns,
            "mean_abs_shap": mean_abs,
            "direction": direction,
            "phylum": "unclassified",
            "family": "unclassified",
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path, help="Saved model directory.")
    p.add_argument("--output", required=True, type=Path, help="Figure/CSV output directory.")
    p.add_argument("--task", default="cd_vs_nonibd")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--n-top", type=int, default=5, help="Number of dependence plots.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model(args.model)
    logger.info("Loaded %s model (grouped AUC %.3f)", metadata.get("model"), metadata.get("grouped_cv_auc", float("nan")))
    X, y, _ = load_hmp2(task=args.task, data_dir=args.data)

    tree_model, shap_df = resolve_tree_and_input(model, X)
    explainer = shap.TreeExplainer(tree_model)
    shap_arr, base = normalize_shap(explainer.shap_values(shap_df), explainer.expected_value)

    save_beeswarm(shap_arr, shap_df, args.output / "beeswarm.png")

    cd_row = int(np.where(y.to_numpy() == 1)[0][0])
    healthy_row = int(np.where(y.to_numpy() == 0)[0][0])
    save_waterfall(shap_arr, base, shap_df, cd_row, args.output / "waterfall_cd.png")
    save_waterfall(shap_arr, base, shap_df, healthy_row, args.output / "waterfall_healthy.png")

    summary = build_summary(shap_arr, shap_df, y)
    summary.to_csv(args.output / "shap_summary.csv", index=False)

    col_index = {c: i for i, c in enumerate(shap_df.columns)}
    for taxon in summary["taxon_id"].head(args.n_top):
        plt.figure()
        shap.dependence_plot(
            col_index[taxon], shap_arr, shap_df, interaction_index=None, show=False
        )
        plt.tight_layout()
        plt.savefig(args.output / f"dependence_{taxon}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("Wrote beeswarm, 2 waterfalls, %d dependence plots, shap_summary.csv to %s",
                args.n_top, args.output)


if __name__ == "__main__":
    main()
