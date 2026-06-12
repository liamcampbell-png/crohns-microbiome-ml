"""Microbiome-specific feature engineering.

Functions here turn a sample-by-taxa table into ecological and taxonomic
features: alpha diversity (within-sample), beta diversity (between-sample
distances), taxonomic-rank aggregation, and per-patient temporal stability.

Diversity metrics are implemented directly rather than via scikit-bio to avoid a
heavy, numpy-version-sensitive dependency; the formulas are standard.

Taxonomy note
-------------
The HMP2 16S table identifies taxa with opaque Greengenes OTU ids that carry no
embedded lineage. :func:`phylogenetic_features` therefore accepts an *optional*
taxonomy mapping and degrades gracefully to ``unclassified`` when none is
supplied (which is the situation for the shipped data).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from preprocessing import clr_transform

__all__ = [
    "alpha_diversity",
    "beta_diversity",
    "parse_greengenes_lineage",
    "phylogenetic_features",
    "stability_features",
]

logger = logging.getLogger(__name__)

#: Taxonomic ranks from broad to fine, with their Greengenes prefixes.
_RANK_PREFIX: dict[str, str] = {
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
}
DEFAULT_LEVELS: tuple[str, ...] = ("phylum", "class", "order", "family", "genus")


# --------------------------------------------------------------------------- #
# Alpha diversity
# --------------------------------------------------------------------------- #
def alpha_diversity(counts: pd.DataFrame) -> pd.DataFrame:
    """Per-sample alpha-diversity indices.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples (rows) by taxa (columns) count matrix.

    Returns
    -------
    pandas.DataFrame
        Indexed by sample, with columns ``["shannon", "simpson", "chao1",
        "observed_otus"]``.

    Notes
    -----
    - ``shannon`` is the Shannon entropy ``-sum(p log p)`` over relative
      abundances (natural log).
    - ``simpson`` is the Gini-Simpson index ``1 - sum(p^2)``.
    - ``chao1`` is ``S_obs + f1 (f1 - 1) / (2 (f2 + 1))`` where ``f1`` and ``f2``
      are the per-sample singleton and doubleton counts.
    - ``observed_otus`` is the count of taxa with non-zero abundance.
    """
    counts = counts.astype(float)
    totals = counts.sum(axis=1).replace(0, np.nan)
    rel = counts.div(totals, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rel = np.where(rel > 0, np.log(rel), 0.0)
    shannon = -(rel.to_numpy() * log_rel).sum(axis=1)
    simpson = 1.0 - (rel.to_numpy() ** 2).sum(axis=1)

    observed = (counts > 0).sum(axis=1).to_numpy()
    f1 = (counts == 1).sum(axis=1).to_numpy().astype(float)
    f2 = (counts == 2).sum(axis=1).to_numpy().astype(float)
    chao1 = observed + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))

    return pd.DataFrame(
        {
            "shannon": shannon,
            "simpson": simpson,
            "chao1": chao1,
            "observed_otus": observed,
        },
        index=counts.index,
    )


# --------------------------------------------------------------------------- #
# Beta diversity
# --------------------------------------------------------------------------- #
def beta_diversity(
    counts: pd.DataFrame,
    metric: str = "braycurtis",
    pseudocount: float = 0.5,
) -> pd.DataFrame:
    """Pairwise between-sample distance matrix.

    Parameters
    ----------
    counts : pandas.DataFrame
        Samples by taxa count matrix.
    metric : {"braycurtis", "aitchison"}, default="braycurtis"
        ``braycurtis`` is computed on raw counts; ``aitchison`` is the Euclidean
        distance on CLR-transformed data (the compositionally-correct analogue).
    pseudocount : float, default=0.5
        Pseudocount used for the CLR step when ``metric="aitchison"``.

    Returns
    -------
    pandas.DataFrame
        Square, symmetric distance matrix indexed and columned by sample id.

    Raises
    ------
    ValueError
        If ``metric`` is not recognised.
    """
    if metric == "braycurtis":
        mat = counts.astype(float).to_numpy()
        dist = pdist(mat, metric="braycurtis")
    elif metric == "aitchison":
        clr = clr_transform(counts, pseudocount).to_numpy()
        dist = pdist(clr, metric="euclidean")
    else:
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'braycurtis' or 'aitchison'."
        )
    square = squareform(dist)
    return pd.DataFrame(square, index=counts.index, columns=counts.index)


# --------------------------------------------------------------------------- #
# Taxonomic aggregation
# --------------------------------------------------------------------------- #
def parse_greengenes_lineage(lineage: str) -> dict[str, str]:
    """Parse a Greengenes lineage string into a rank -> name mapping.

    Parameters
    ----------
    lineage : str
        Semicolon-delimited string such as
        ``"k__Bacteria; p__Firmicutes; ...; g__Faecalibacterium"``.

    Returns
    -------
    dict
        Mapping from rank name (``"phylum"``, ``"class"``, ...) to taxon name.
        Ranks absent or with an empty name (e.g. ``"g__"``) are omitted.
    """
    out: dict[str, str] = {}
    if not isinstance(lineage, str):
        return out
    fields = [f.strip() for f in lineage.split(";")]
    for rank, prefix in _RANK_PREFIX.items():
        for field in fields:
            if field.startswith(prefix):
                name = field[len(prefix):].strip()
                if name:
                    out[rank] = name
                break
    return out


def _taxon_to_rank(
    taxon: str,
    rank: str,
    taxonomy: Mapping[str, str | Mapping[str, str]] | pd.DataFrame | None,
) -> str:
    """Resolve a taxon id to its label at ``rank`` or ``"unclassified"``."""
    if taxonomy is None:
        return "unclassified"
    if isinstance(taxonomy, pd.DataFrame):
        if taxon in taxonomy.index and rank in taxonomy.columns:
            val = taxonomy.at[taxon, rank]
            return str(val) if pd.notna(val) and str(val) else "unclassified"
        return "unclassified"
    entry = taxonomy.get(taxon)
    if entry is None:
        return "unclassified"
    if isinstance(entry, str):
        entry = parse_greengenes_lineage(entry)
    name = entry.get(rank) if isinstance(entry, Mapping) else None
    return name if name else "unclassified"


def phylogenetic_features(
    clr_df: pd.DataFrame,
    taxonomy: Mapping[str, str | Mapping[str, str]] | pd.DataFrame | None = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate CLR abundances at one or more taxonomic ranks.

    Parameters
    ----------
    clr_df : pandas.DataFrame
        Samples by taxa CLR-transformed matrix.
    taxonomy : mapping or pandas.DataFrame, optional
        Taxon-id to lineage mapping. May be a dict of ``{otu: lineage_string}``,
        a dict of ``{otu: {rank: name}}``, or a DataFrame indexed by taxon id
        with one column per rank. If ``None``, every taxon is treated as
        ``unclassified`` and the function still returns without error.
    levels : sequence of str, default=:data:`DEFAULT_LEVELS`
        Ranks to aggregate at.
    agg : {"mean", "sum"}, default="mean"
        How to combine taxa within a clade.

    Returns
    -------
    pandas.DataFrame
        Samples by aggregated-clade matrix; columns are ``"{level}:{clade}"``.
    """
    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be 'mean' or 'sum'.")
    if taxonomy is None:
        logger.warning(
            "phylogenetic_features: no taxonomy supplied; all %d taxa treated as "
            "unclassified (one column per level).", clr_df.shape[1]
        )

    frames: list[pd.DataFrame] = []
    for level in levels:
        labels = pd.Index(
            [_taxon_to_rank(t, level, taxonomy) for t in clr_df.columns],
            name=level,
        )
        grouped = clr_df.T.groupby(labels).agg(agg).T
        grouped.columns = [f"{level}:{c}" for c in grouped.columns]
        frames.append(grouped)
    return pd.concat(frames, axis=1)


# --------------------------------------------------------------------------- #
# Temporal stability
# --------------------------------------------------------------------------- #
def stability_features(
    longitudinal_df: pd.DataFrame,
    patient_col: str = "Participant ID",
    min_timepoints: int = 2,
) -> pd.DataFrame:
    """Per-patient temporal stability as coefficient of variation across visits.

    Parameters
    ----------
    longitudinal_df : pandas.DataFrame
        Indexed by sample, containing ``patient_col`` plus numeric feature
        columns. All non-patient numeric columns are treated as features.
    patient_col : str, default="Participant ID"
        Column holding the patient identifier.
    min_timepoints : int, default=2
        Patients with fewer than this many samples get ``NaN`` (CV undefined).

    Returns
    -------
    pandas.DataFrame
        Indexed by patient, one column per feature holding the coefficient of
        variation ``std / |mean|`` across that patient's timepoints. A feature
        that is constant for a patient yields 0; an all-zero-mean feature yields
        ``NaN``.
    """
    if patient_col not in longitudinal_df.columns:
        raise ValueError(f"patient_col {patient_col!r} not in DataFrame.")
    features = longitudinal_df.drop(columns=[patient_col]).select_dtypes("number")
    if features.empty:
        raise ValueError("No numeric feature columns found.")

    rows: dict[object, pd.Series] = {}
    for patient, idx in longitudinal_df.groupby(patient_col).groups.items():
        block = features.loc[idx]
        if len(block) < min_timepoints:
            rows[patient] = pd.Series(np.nan, index=features.columns)
            continue
        mean = block.mean(axis=0)
        std = block.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = std / mean.abs()
        cv = cv.where(mean.abs() > 0, np.nan)  # undefined when mean is 0
        rows[patient] = cv
    out = pd.DataFrame(rows).T
    out.index.name = patient_col
    return out
