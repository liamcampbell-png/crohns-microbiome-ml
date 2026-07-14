# Crohn's Disease Prediction from Gut Microbiome Data

Predicting Crohn's disease status from 16S/metagenomic sequencing data, combining classical ML, microbiome foundation model embeddings, and an LLM agent for literature-grounded biomarker interpretation.

## Overview

This project predicts Crohn's disease (an IBD subtype) from gut microbiome composition and identifies which microbial taxa drive that prediction. It goes beyond a standard classification pipeline in three ways:

1. **Baseline vs. foundation model comparison** — classical ML on hand-engineered abundance features, benchmarked against a pretrained microbiome foundation model's embeddings as input features
2. **Cross-study validation** — evaluation designed to avoid batch-effect leakage (train/test splits respect study origin, not just random splits), a common failure mode in published microbiome ML work
3. **Agentic biomarker interpretation** — an LLM agent takes the model's top predictive taxa (via SHAP) and cross-references them against literature (RAG over PubMed) to flag known vs. novel candidate biomarkers, producing a cited report rather than a bare feature-importance list

## Motivation

Most microbiome disease-prediction pipelines stop at a feature-importance plot. This project treats that plot as the *start* of the interesting question — is this taxon a known IBD biomarker, or a potentially novel one worth flagging? — and automates that literature check.

## Pipeline

```
Raw sequencing data (16S/metagenomic)
        │
        ▼
Preprocessing (OTU/ASV table, relative abundance normalization)
        │
        ├──► Hand-engineered features ──► Classical ML (RF, XGBoost)
        │
        └──► Foundation model embeddings ──► Classifier
        │
        ▼
Evaluation (cross-study CV, accuracy/AUC, calibration)
        │
        ▼
SHAP feature importance
        │
        ▼
Biomarker interpretation agent (RAG over literature)
        │
        ▼
Cited report: known vs. novel candidate biomarkers
```

## Data

- Source: [fill in — e.g. iHMP/HMP2 IBD cohort, public 16S dataset]
- Samples: [n cases / n controls]
- Modality: 16S rRNA / shotgun metagenomic sequencing
- Preprocessing: [QIIME2 / DADA2 / other — fill in]

## Methods

### Baseline model
Random Forest / XGBoost on relative abundance features at [genus/species] level.

### Foundation model embeddings
Taxa/sample embeddings from a pretrained microbiome transformer, used as input features to a lightweight downstream classifier. Compared head-to-head against the baseline on identical train/test splits.

### Cross-study validation
Splits constructed so that no study appears in both train and test sets, to measure generalization rather than within-study fit.

### Biomarker interpretation agent
1. Extract top-*k* predictive taxa via SHAP
2. For each taxon, retrieve relevant literature (PubMed/RAG index)
3. LLM agent classifies each as "established IBD biomarker" (with citations) or "understudied/novel candidate" (with biological rationale)
4. Output: structured, cited markdown report

## Results

| Approach | Accuracy | AUC | Notes |
|---|---|---|---|
| Baseline (RF/XGBoost, raw abundance) | — | — | random split |
| Baseline (RF/XGBoost, raw abundance) | — | — | cross-study split |
| Foundation model embeddings + classifier | — | — | random split |
| Foundation model embeddings + classifier | — | — | cross-study split |

*[Fill in after running experiments]*

## Repo structure

```
├── data/               # raw + processed data (not committed if large/sensitive)
├── preprocessing/       # OTU/ASV table generation, normalization
├── models/
│   ├── baseline.py       # classical ML pipeline
│   └── embeddings.py     # foundation model embedding extraction + classifier
├── eval/
│   └── cross_study_cv.py
├── agent/
│   ├── biomarker_agent.py  # SHAP -> RAG -> report pipeline
│   └── literature_index/   # RAG index over relevant literature
├── reports/              # generated biomarker reports
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## Usage

```bash
# Run baseline model
python models/baseline.py --data data/processed/

# Extract foundation model embeddings and train classifier
python models/embeddings.py --data data/processed/

# Run cross-study evaluation
python eval/cross_study_cv.py

# Generate biomarker interpretation report
python agent/biomarker_agent.py --model models/best_model.pkl
```

## Tech stack

- **Data processing:** QIIME2 / DADA2, pandas
- **Classical ML:** scikit-learn, XGBoost
- **Foundation model:** [name/checkpoint used]
- **Agent/RAG:** [LLM API used], vector store for literature retrieval
- **Explainability:** SHAP

## Limitations

- Batch effects across studies remain a known confounder in microbiome ML; cross-study validation mitigates but does not eliminate this
- The biomarker agent's "novel candidate" flags are hypotheses for further investigation, not validated findings
- [Add dataset-size, demographic, or other limitations specific to your data]

## Future work

- Extend to shotgun metagenomic functional (pathway-level) features, not just taxonomic
- Persist agent-generated hypotheses across runs to build a growing candidate-biomarker knowledge base
- Multimodal integration with clinical metadata

## License

[fill in]