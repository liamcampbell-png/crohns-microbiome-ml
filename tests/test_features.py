"""Tests for :mod:`features`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    alpha_diversity,
    beta_diversity,
    parse_greengenes_lineage,
    phylogenetic_features,
    stability_features,
)


def test_shannon_matches_manual_calculation(synthetic_counts: pd.DataFrame) -> None:
    """Shannon entropy from the function matches a direct computation."""
    div = alpha_diversity(synthetic_counts)
    row = synthetic_counts.iloc[0]
    p = (row[row > 0] / row.sum()).to_numpy()
    manual = -(p * np.log(p)).sum()
    assert div["shannon"].iloc[0] == pytest.approx(manual)


def test_observed_otus_counts_nonzero(synthetic_counts: pd.DataFrame) -> None:
    """observed_otus equals the number of non-zero taxa per sample."""
    div = alpha_diversity(synthetic_counts)
    expected = (synthetic_counts > 0).sum(axis=1)
    assert (div["observed_otus"].to_numpy() == expected.to_numpy()).all()


def test_beta_diversity_is_symmetric_zero_diagonal(synthetic_counts: pd.DataFrame) -> None:
    """Distance matrices are symmetric with a zero diagonal."""
    for metric in ("braycurtis", "aitchison"):
        d = beta_diversity(synthetic_counts, metric=metric)
        assert d.shape == (len(synthetic_counts), len(synthetic_counts))
        assert np.allclose(np.diag(d), 0.0)
        assert np.allclose(d.to_numpy(), d.to_numpy().T)


def test_phylogenetic_aggregation_handles_unclassified(synthetic_counts: pd.DataFrame) -> None:
    """With no taxonomy, aggregation returns one 'unclassified' column per level."""
    clr = np.log(synthetic_counts + 0.5)
    clr = clr.subtract(clr.mean(axis=1), axis=0)
    out = phylogenetic_features(clr, taxonomy=None, levels=("phylum", "genus"))
    assert list(out.columns) == ["phylum:unclassified", "genus:unclassified"]
    assert out.shape[0] == synthetic_counts.shape[0]
    assert not out.isna().any().any()


def test_phylogenetic_aggregation_with_partial_taxonomy(synthetic_counts: pd.DataFrame) -> None:
    """A partial taxonomy maps known taxa and leaves the rest unclassified."""
    clr = np.log(synthetic_counts + 0.5)
    clr = clr.subtract(clr.mean(axis=1), axis=0)
    taxonomy = {
        "t0": "k__Bacteria;p__Firmicutes;g__Faecalibacterium",
        "t1": {"phylum": "Bacteroidetes"},
    }
    out = phylogenetic_features(clr, taxonomy=taxonomy, levels=("phylum",))
    assert "phylum:Firmicutes" in out.columns
    assert "phylum:Bacteroidetes" in out.columns
    assert "phylum:unclassified" in out.columns


def test_parse_greengenes_lineage() -> None:
    """Lineage strings parse into rank -> name, skipping empty ranks."""
    parsed = parse_greengenes_lineage("k__Bacteria; p__Firmicutes; c__Clostridia; g__")
    assert parsed["phylum"] == "Firmicutes"
    assert parsed["class"] == "Clostridia"
    assert "genus" not in parsed  # empty g__ is skipped


def test_stability_features_shape(grouped_dataset) -> None:
    """Stability returns one row per patient and one column per feature."""
    X, _, groups = grouped_dataset
    longitudinal = X.copy()
    longitudinal["Participant ID"] = groups
    stab = stability_features(longitudinal, patient_col="Participant ID")
    assert stab.shape == (groups.nunique(), X.shape[1])
