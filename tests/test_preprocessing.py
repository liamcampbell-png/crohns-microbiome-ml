"""Tests for :mod:`preprocessing`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    MicrobiomePreprocessor,
    clr_transform,
    low_read_filter,
    prevalence_filter,
    relative_abundance,
)


def test_clr_is_zero_sum(synthetic_counts: pd.DataFrame) -> None:
    """CLR rows must sum to ~0 (the centering property)."""
    clr = clr_transform(synthetic_counts, pseudocount=0.5)
    row_sums = clr.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 0.0, atol=1e-9)


def test_prevalence_filter_removes_correct_columns(synthetic_counts: pd.DataFrame) -> None:
    """A 0.10 threshold drops both the absent (0%) and 1/20 (5%) taxa, keeps commons."""
    filtered = prevalence_filter(synthetic_counts, threshold=0.10)
    assert "t_absent" not in filtered.columns  # 0% prevalence -> dropped
    assert "t_rare" not in filtered.columns     # 5% prevalence < 10% -> dropped
    assert all(f"t{i}" in filtered.columns for i in range(6))  # common taxa kept


def test_prevalence_filter_keeps_rare_below_threshold(synthetic_counts: pd.DataFrame) -> None:
    """A 0.04 threshold retains the 1/20 taxon but still drops the absent one."""
    filtered = prevalence_filter(synthetic_counts, threshold=0.04)
    assert "t_rare" in filtered.columns
    assert "t_absent" not in filtered.columns


def test_low_read_filter_removes_correct_rows() -> None:
    """Samples below the read threshold are dropped; others retained."""
    counts = pd.DataFrame(
        {"a": [10, 1, 100], "b": [10, 0, 100]},
        index=["keep1", "drop", "keep2"],
    )
    filtered = low_read_filter(counts, min_reads=20)
    assert list(filtered.index) == ["keep1", "keep2"]


def test_relative_abundance_rows_sum_to_one(synthetic_counts: pd.DataFrame) -> None:
    """TSS normalization yields rows summing to 1 (non-empty samples)."""
    rel = relative_abundance(synthetic_counts)
    assert np.allclose(rel.sum(axis=1).to_numpy(), 1.0)


def test_fit_transform_interface(synthetic_counts: pd.DataFrame) -> None:
    """fit/transform is sample-preserving and learns a taxon subset."""
    pp = MicrobiomePreprocessor(prevalence_threshold=0.10, transform_method="clr")
    out = pp.fit_transform(synthetic_counts)
    # Same number of samples, fewer or equal taxa, real names recorded.
    assert out.shape[0] == synthetic_counts.shape[0]
    assert out.shape[1] <= synthetic_counts.shape[1]
    assert list(out.columns) == pp.kept_taxa_
    assert "t_absent" not in pp.kept_taxa_
    # CLR centering preserved through the transformer.
    assert np.allclose(out.sum(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_transform_before_fit_raises(synthetic_counts: pd.DataFrame) -> None:
    """Calling transform before fit is an error."""
    with pytest.raises(RuntimeError):
        MicrobiomePreprocessor().transform(synthetic_counts)
