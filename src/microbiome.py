"""Shared data-loading and preprocessing for the Crohn's microbiome project.

Centralises the pipeline first worked out interactively in
`notebooks/01_exploration.ipynb` so the modeling, longitudinal, and
interpretation notebooks can reuse it without copy-pasting 60 lines each.

Pipeline (matches notebook 01, yields 179 taxa features):
    raw counts -> prevalence filter (>10% of samples)
               -> CLR transform (pseudocount 0.5)
               -> presence filter (CLR>0 in >=5% of samples)
               -> drop one of each |r|>0.90 correlated pair
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_CSV = DATA_DIR / "hmp2_metadata_2018-08-20.csv"
OTU_TSV = DATA_DIR / "taxonomic_profiles.tsv.gz"

META_COLS = ["diagnosis", "week_num", "Participant ID"]


def load_otu_meta():
    """Return (otu_meta, otu_counts, taxa).

    otu_meta : samples x (taxa + META_COLS), 16S samples joined to metadata.
    otu_counts : samples x taxa, raw counts as float (no metadata columns).
    taxa : list of all taxon IDs in the raw table.
    """
    metadata = pd.read_csv(METADATA_CSV, low_memory=False)
    otu_table = pd.read_csv(OTU_TSV, sep="\t", index_col=0)  # taxa x samples
    taxa = list(otu_table.index)

    otu_t = otu_table.T
    otu_t.index.name = "External ID"

    meta_16s = metadata[metadata["data_type"].str.contains("16S", na=False)]
    otu_meta = otu_t.join(
        meta_16s.set_index("External ID")[META_COLS], how="inner"
    )
    otu_counts = otu_meta[taxa].astype(float)
    return otu_meta, otu_counts, taxa


def clr(counts, pseudocount=0.5):
    """Centered log-ratio transform of a samples x taxa count table."""
    log_vals = np.log(counts.astype(float) + pseudocount)
    return log_vals.subtract(log_vals.mean(axis=1), axis=0)


def preprocess(otu_counts, prevalence=0.10, presence=0.05, corr_thresh=0.90):
    """Full preprocessing. Returns the 179-feature CLR table (samples x taxa)."""
    prev = (otu_counts > 0).mean(axis=0)
    prevalent = otu_counts[prev[prev > prevalence].index]

    otu_clr = clr(prevalent)

    pres = (otu_clr > 0).sum() / len(otu_clr)
    otu_clr = otu_clr[pres[pres >= presence].index]

    corr = otu_clr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    return otu_clr.drop(columns=to_drop)


def cd_vs_nonibd(otu_final, otu_meta):
    """Return (X, y) for the CD-vs-healthy task. y: CD=1, nonIBD=0."""
    mask = otu_meta["diagnosis"].isin(["CD", "nonIBD"])
    X = otu_final[mask]
    y = (otu_meta.loc[mask, "diagnosis"] == "CD").astype(int)
    return X, y
