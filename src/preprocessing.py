"""Production preprocessing pipeline for the Crohn's microbiome project.

Expands the original :mod:`microbiome` helper into a scikit-learn compatible
transformer plus a set of compositional-data preprocessing primitives.

The central object is :class:`MicrobiomePreprocessor`, which learns a taxon
filter at ``fit`` time and applies a compositional transform at ``transform``
time without ever changing the number of samples (so it composes cleanly inside
an sklearn :class:`~sklearn.pipeline.Pipeline`).

Sample-dropping operations (read-depth filtering, rarefaction) change the number
of rows and therefore live as *standalone* functions that operate on
``(X, y, groups)`` jointly, to be called before fitting an estimator.

Notes
-----
Microbiome counts are compositional: the per-sample total is an arbitrary
sequencing depth, so only relative information is meaningful. The default
transform is the centered log-ratio (CLR), which removes the sum-to-constant
constraint.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "SEED",
    "MicrobiomePreprocessor",
    "load_hmp2",
    "clr_transform",
    "relative_abundance",
    "prevalence_filter",
    "low_read_filter",
    "rarefy",
]

SEED: int = 42

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_CSV = DATA_DIR / "hmp2_metadata_2018-08-20.csv"
OTU_TSV = DATA_DIR / "taxonomic_profiles.tsv.gz"

_TASK_LABELS: dict[str, tuple[str, str]] = {
    # task name -> (positive_class, negative_class); positive is encoded as 1
    "cd_vs_nonibd": ("CD", "nonIBD"),
    "cd_vs_uc": ("CD", "UC"),
    "uc_vs_nonibd": ("UC", "nonIBD"),
}

TransformMethod = Literal["clr", "relative", "none"]


# --------------------------------------------------------------------------- #
# Standalone compositional primitives
# --------------------------------------------------------------------------- #
def relative_abundance(counts: pd.DataFrame) -> pd.DataFrame:
    """Total-sum-scaling (TSS) normalization to relative abundances.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples (rows) by taxa (columns) count matrix.

    Returns
    -------
    pandas.DataFrame
        Each row divided by its sum. Rows that sum to zero are returned as
        zeros and trigger a warning.
    """
    totals = counts.sum(axis=1)
    zero_rows = int((totals == 0).sum())
    if zero_rows:
        warnings.warn(
            f"{zero_rows} sample(s) have zero total reads; "
            "their relative abundances are undefined and set to 0.",
            stacklevel=2,
        )
        logger.warning("relative_abundance: %d zero-total samples set to 0", zero_rows)
    safe = totals.replace(0, np.nan)
    return counts.div(safe, axis=0).fillna(0.0)


def clr_transform(counts: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    """Centered log-ratio (CLR) transform.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples by taxa count (or abundance) matrix.
    pseudocount : float, default=0.5
        Added before the log to keep zeros finite.

    Returns
    -------
    pandas.DataFrame
        CLR values; every row sums to ~0 by construction.
    """
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive to keep the log finite.")
    log_vals = np.log(counts.astype(float) + pseudocount)
    return log_vals.subtract(log_vals.mean(axis=1), axis=0)


def prevalence_filter(
    counts: pd.DataFrame, threshold: float = 0.10
) -> pd.DataFrame:
    """Drop taxa present in fewer than ``threshold`` of samples.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples by taxa count matrix.
    threshold : float, default=0.10
        Minimum fraction of samples in which a taxon must have a non-zero count.

    Returns
    -------
    pandas.DataFrame
        Column-subset of ``counts`` keeping only prevalent taxa.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    prevalence = (counts > 0).mean(axis=0)
    keep = prevalence[prevalence >= threshold].index
    dropped = counts.shape[1] - len(keep)
    logger.info(
        "prevalence_filter(%.2f): kept %d / %d taxa (dropped %d)",
        threshold, len(keep), counts.shape[1], dropped,
    )
    return counts[keep]


def low_read_filter(
    counts: pd.DataFrame, min_reads: int
) -> pd.DataFrame:
    """Drop samples whose total read count is below ``min_reads``.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples by taxa count matrix.
    min_reads : int
        Minimum total reads required to retain a sample.

    Returns
    -------
    pandas.DataFrame
        Row-subset of ``counts``. Use :func:`align_labels` (or index the labels
        with the returned index) to keep ``y`` / ``groups`` aligned.
    """
    totals = counts.sum(axis=1)
    keep = totals[totals >= min_reads].index
    dropped = counts.shape[0] - len(keep)
    if dropped:
        warnings.warn(
            f"low_read_filter: dropping {dropped} sample(s) below "
            f"{min_reads} reads.",
            stacklevel=2,
        )
        logger.warning(
            "low_read_filter(%d): dropped %d / %d samples",
            min_reads, dropped, counts.shape[0],
        )
    return counts.loc[keep]


def rarefy(
    counts: pd.DataFrame,
    depth: int,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Rarefy every sample to a uniform sequencing ``depth``.

    Each sample is subsampled without replacement to exactly ``depth`` reads.
    Samples with fewer than ``depth`` total reads cannot be rarefied and are
    dropped (with a warning).

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples by taxa integer count matrix.
    depth : int
        Target read depth.
    random_state : int, default=:data:`SEED`
        Seed for the subsampling RNG.

    Returns
    -------
    pandas.DataFrame
        Rarefied integer counts, one row per retained sample.
    """
    if depth <= 0:
        raise ValueError("depth must be a positive integer.")
    rng = np.random.default_rng(random_state)
    totals = counts.sum(axis=1)
    keep = totals[totals >= depth].index
    dropped = counts.shape[0] - len(keep)
    if dropped:
        warnings.warn(
            f"rarefy: dropping {dropped} sample(s) with fewer than {depth} reads.",
            stacklevel=2,
        )
        logger.warning("rarefy(%d): dropped %d samples below depth", depth, dropped)

    out = np.zeros((len(keep), counts.shape[1]), dtype=int)
    kept_counts = counts.loc[keep].to_numpy(dtype=int)
    for i, row in enumerate(kept_counts):
        # multivariate hypergeometric == sampling `depth` reads without replacement
        out[i] = rng.multivariate_hypergeometric(row, depth)
    return pd.DataFrame(out, index=keep, columns=counts.columns)


# --------------------------------------------------------------------------- #
# sklearn-compatible transformer
# --------------------------------------------------------------------------- #
class MicrobiomePreprocessor(BaseEstimator, TransformerMixin):
    """Learn a taxon filter, then apply a compositional transform.

    The transformer is deliberately *sample-preserving*: ``transform`` never
    changes the number of rows, so it is safe inside a
    :class:`~sklearn.pipeline.Pipeline`. Read-depth filtering and rarefaction,
    which drop samples, are exposed as the module-level functions
    :func:`low_read_filter` and :func:`rarefy`.

    Parameters
    ----------
    prevalence_threshold : float, default=0.10
        Taxon kept if present in at least this fraction of *training* samples.
    transform_method : {"clr", "relative", "none"}, default="clr"
        Compositional transform applied at ``transform`` time.
    pseudocount : float, default=0.5
        Pseudocount for the CLR transform.
    variance_filter : bool, default=False
        If True, also drop taxa whose CLR value exceeds the per-sample geometric
        mean (CLR > 0) in fewer than 5% of training samples.
    correlation_threshold : float or None, default=None
        If set, drop one taxon from each pair with absolute correlation above
        this value (computed on transformed training data).
    random_state : int, default=:data:`SEED`
        Unused by the deterministic transforms; kept for API consistency.

    Attributes
    ----------
    kept_taxa_ : list of str
        Taxa retained after fitting.
    feature_names_in_ : numpy.ndarray
        Column names seen during ``fit``.
    n_features_in_ : int
        Number of columns seen during ``fit``.
    """

    def __init__(
        self,
        prevalence_threshold: float = 0.10,
        transform_method: TransformMethod = "clr",
        pseudocount: float = 0.5,
        variance_filter: bool = False,
        correlation_threshold: float | None = None,
        random_state: int = SEED,
    ) -> None:
        self.prevalence_threshold = prevalence_threshold
        self.transform_method = transform_method
        self.pseudocount = pseudocount
        self.variance_filter = variance_filter
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

    # -- internal helpers --------------------------------------------------- #
    def _apply_transform(self, counts: pd.DataFrame) -> pd.DataFrame:
        if self.transform_method == "clr":
            return clr_transform(counts, self.pseudocount)
        if self.transform_method == "relative":
            return relative_abundance(counts)
        if self.transform_method == "none":
            return counts.astype(float)
        raise ValueError(
            f"Unknown transform_method={self.transform_method!r}; "
            "expected 'clr', 'relative', or 'none'."
        )

    # -- sklearn API -------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MicrobiomePreprocessor":
        """Learn the taxon filter from training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training samples by taxa count matrix.
        y : pandas.Series, optional
            Ignored; present for API compatibility.

        Returns
        -------
        MicrobiomePreprocessor
            The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MicrobiomePreprocessor expects a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        kept = prevalence_filter(X, self.prevalence_threshold)
        transformed = self._apply_transform(kept)

        if self.variance_filter:
            presence = (transformed > 0).sum(axis=0) / len(transformed)
            kept_cols = presence[presence >= 0.05].index
            logger.info(
                "variance_filter: kept %d / %d taxa", len(kept_cols), transformed.shape[1]
            )
            transformed = transformed[kept_cols]
            kept = kept[kept_cols]

        if self.correlation_threshold is not None:
            corr = transformed.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if (upper[c] > self.correlation_threshold).any()]
            logger.info("correlation_filter: dropping %d redundant taxa", len(to_drop))
            kept = kept.drop(columns=to_drop)

        self.kept_taxa_ = list(kept.columns)
        logger.info("Preprocessor fitted: %d features retained", len(self.kept_taxa_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned filter and compositional transform.

        Parameters
        ----------
        X : pandas.DataFrame
            Samples by taxa count matrix with the fitted columns available.

        Returns
        -------
        pandas.DataFrame
            Transformed feature matrix with the same row index as ``X`` and the
            retained taxa as columns.
        """
        if not hasattr(self, "kept_taxa_"):
            raise RuntimeError("Call fit before transform.")
        missing = [t for t in self.kept_taxa_ if t not in X.columns]
        if missing:
            raise ValueError(f"{len(missing)} fitted taxa are missing from X.")
        subset = X[self.kept_taxa_]
        return self._apply_transform(subset)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the retained taxon names (sklearn metadata API)."""
        if not hasattr(self, "kept_taxa_"):
            raise RuntimeError("Call fit before get_feature_names_out.")
        return np.asarray(self.kept_taxa_)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_hmp2(
    task: str = "cd_vs_nonibd",
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the HMP2 16S table joined to metadata for a binary task.

    Parameters
    ----------
    task : str, default="cd_vs_nonibd"
        One of ``{"cd_vs_nonibd", "cd_vs_uc", "uc_vs_nonibd"}``. The first named
        class is encoded as the positive label (1).
    data_dir : pathlib.Path or str, optional
        Override the default ``data/`` directory.

    Returns
    -------
    X : pandas.DataFrame
        Raw count matrix, samples by taxa (no metadata columns).
    y : pandas.Series
        Binary labels (positive class = 1), indexed by sample.
    groups : pandas.Series
        Participant id per sample, for :class:`~sklearn.model_selection.GroupKFold`.

    Raises
    ------
    ValueError
        If ``task`` is not recognised.
    FileNotFoundError
        If the data files are not present.
    """
    if task not in _TASK_LABELS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(_TASK_LABELS)}.")

    base = Path(data_dir) if data_dir is not None else DATA_DIR
    meta_path = base / METADATA_CSV.name
    otu_path = base / OTU_TSV.name
    for p in (meta_path, otu_path):
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}")

    metadata = pd.read_csv(meta_path, low_memory=False)
    otu_table = pd.read_csv(otu_path, sep="\t", index_col=0)  # taxa x samples
    taxa = list(otu_table.index)

    otu_t = otu_table.T
    otu_t.index.name = "External ID"
    meta_16s = metadata[metadata["data_type"].str.contains("16S", na=False)]
    joined = otu_t.join(
        meta_16s.set_index("External ID")[["diagnosis", "Participant ID"]], how="inner"
    )

    pos, neg = _TASK_LABELS[task]
    mask = joined["diagnosis"].isin([pos, neg])
    joined = joined[mask]

    X = joined[taxa].astype(float)
    y = (joined["diagnosis"] == pos).astype(int)
    y.name = task
    groups = joined["Participant ID"]
    groups.name = "participant"

    logger.info(
        "load_hmp2(%s): %d samples x %d taxa across %d participants (%s=%d, %s=%d)",
        task, X.shape[0], X.shape[1], groups.nunique(),
        pos, int(y.sum()), neg, int((1 - y).sum()),
    )
    return X, y, groups
