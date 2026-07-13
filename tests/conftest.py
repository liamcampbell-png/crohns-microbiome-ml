"""Shared pytest fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_counts() -> pd.DataFrame:
    """A small samples x taxa count table with a known sparsity pattern.

    Taxon ``t_rare`` is present in only one sample; ``t_absent`` in none. These
    let prevalence/low-read filters be checked against hand-computed answers.
    """
    rng = np.random.default_rng(0)
    n_samples, n_common = 20, 6
    common = rng.integers(1, 50, size=(n_samples, n_common))
    rare = np.zeros((n_samples, 1), dtype=int)
    rare[0, 0] = 7  # present in exactly one sample
    absent = np.zeros((n_samples, 1), dtype=int)  # present in none
    mat = np.hstack([common, rare, absent])
    cols = [f"t{i}" for i in range(n_common)] + ["t_rare", "t_absent"]
    idx = [f"s{i}" for i in range(n_samples)]
    return pd.DataFrame(mat, index=idx, columns=cols)


@pytest.fixture
def grouped_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic (X, y, groups): 30 samples, 6 patients of 5 visits, 10 features."""
    rng = np.random.default_rng(1)
    n_groups, per_group, n_features = 6, 5, 10
    n = n_groups * per_group
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
        index=[f"s{i}" for i in range(n)],
    )
    y = pd.Series(rng.integers(0, 2, size=n), index=X.index, name="y")
    groups = pd.Series(
        np.repeat(np.arange(n_groups), per_group), index=X.index, name="patient"
    )
    return X, y, groups
