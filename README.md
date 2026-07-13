# IBD Microbiome Leakage: Does Data Leakage Change the Biological Story?

## Overview

Can a single stool-sample microbiome profile tell Crohn's disease from a healthy gut — and *how honestly* can we measure that?

This project builds an interpretable machine-learning pipeline on the HMP2 (iHMP) IBD cohort, then stress-tests its own results against the trap that sinks most microbiome-ML projects: **patient-level data leakage**.

Most microbiome-ML work stops at "leakage inflates accuracy." This project goes further and asks a sharper question:

> **Does leakage change the biological story the model tells you — not just how well it performs, but *which taxa it says matter*?**

A naive model might hit 0.90 AUC and confidently point to Taxon X as the key driver of Crohn's disease. A properly validated ("honest") model might hit 0.75 AUC and point to Taxon Y instead. If those disagree, leakage isn't just making results look better than they are — it's actively misleading the biological conclusions drawn from the model. That's the headline finding this project is built to surface.

## Data

This project uses the **HMP2 / iHMP** (Integrative Human Microbiome Project) IBD cohort — a longitudinal dataset tracking gut microbiome composition in subjects with Crohn's disease (CD), ulcerative colitis (UC), and non-IBD controls over multiple timepoints.

Key structural fact this project is built around: **subjects contribute multiple stool samples over time.** Naively splitting at the *sample* level (rather than the *subject* level) lets samples from the same person appear in both train and test sets — the core source of leakage this project investigates.

## Repo Structure

```
ibd-microbiome-leakage/
├── src/
│   ├── data.py          # load HMP2 tables, merge taxonomy/metadata, subject ID mapping
│   ├── features.py       # CLR transform, filtering, feature matrix construction
│   ├── splits.py         # naive split, GroupKFold, StratifiedGroupKFold, nested CV
│   ├── models.py         # RF, LogReg, XGBoost wrappers, consistent interface
│   ├── importance.py      # SHAP computation, importance extraction per model/fold
│   ├── stability.py       # rank correlation, top-k overlap, bootstrap CI on importance rankings
│   └── evaluate.py        # AUC, calibration, significance testing
├── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_naive_model_the_trap.ipynb
│   ├── 04_leakage_diagnosis_and_fix.ipynb
│   ├── 05_importance_stability.ipynb      ← headline notebook
│   └── 06_paper_grounding.ipynb            ← Topçuoğlu framework + Lloyd-Price validation
├── tests/
│   └── test_splits.py     # unit tests: assert zero subject overlap in grouped CV
├── config/
│   └── experiment.yaml
└── README.md
```

## Project Narrative

### Act 1 — Data Foundation (`01`, `02`)
Load HMP2 taxonomic abundance and metadata tables, resolve the subject-sample structure, apply a CLR (centered log-ratio) transform to handle the compositional nature of microbiome data, and build the final feature matrix. Document class balance (CD vs. healthy) and how many samples exist per subject.

### Act 2 — The Trap (`03`)
Train three model families (Logistic Regression, Random Forest, XGBoost) on a **naive, random sample-level split**. Report AUC — expect it to look strong. Compute SHAP feature importances and save them as the "naive" importance rankings.

### Act 3 — The Fix (`04`)
Show the smoking gun: subject ID overlap between the train and test sets from Act 2. Re-run everything using `StratifiedGroupKFold` grouped by subject, with nested cross-validation for hyperparameter tuning so tuning itself doesn't leak. Report the corrected ("honest") AUC — expect a real drop. Recompute SHAP importances as the "honest" rankings.

### Act 4 — Stability Analysis (`05`, the headline result)
For each model family, compare naive vs. honest feature importance rankings using:
- Spearman rank correlation between naive and honest taxon rankings
- Top-k overlap (e.g., do the top 10 taxa agree between naive and honest models?)
- Bootstrapped confidence intervals across CV folds
- A summary rank-shift figure visualizing how each taxon's importance moves from naive to honest

**Target finding:** importance rankings are *less stable* than the AUC gap alone would suggest — even models with only moderate accuracy inflation can substantially reorder which taxa they implicate, because leakage lets models exploit subject-specific noise that happens to correlate with a convenient subset of taxa.

### Act 5 — Grounding (`06`)
Cross-check the honest model's top taxa against known CD-associated findings from Lloyd-Price et al. (2019) — e.g., *Faecalibacterium* depletion, increased Proteobacteria — to confirm the honest model is capturing real biology, not just noise. Cite Topçuoğlu et al.'s (2020) ML methodology framework to justify the cross-validation design choices made throughout.

## Key Finding (fill in once complete)

> *One-paragraph summary once results are in: naive cross-validation doesn't just overestimate performance — it changes which biology you'd report.*

## Methodology References

- **Lloyd-Price et al., 2019, *Nature*** — source paper for the HMP2/iHMP cohort; used here to validate that the honest model's top taxa align with established CD-associated findings.
- **Topçuoğlu et al., 2020, *mSystems*** — "A Framework for Effective Application of Machine Learning to Microbiome-Based Classification Problems"; used here to justify cross-validation and split design choices.

## Setup

```bash
git clone <repo-url>
cd ibd-microbiome-leakage
pip install -r requirements.txt
```

Run notebooks in order, `01` through `06`. Configuration (model hyperparameters, CV folds, random seeds) is centralized in `config/experiment.yaml`.

## Testing

```bash
pytest tests/
```

`test_splits.py` asserts zero subject-ID overlap between train and test folds under the grouped CV strategy — this is the correctness check for the entire leakage-fix logic.

## Status

🚧 In progress — see `notebooks/` for current state of each phase.