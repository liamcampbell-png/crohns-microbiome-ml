# crohns-microbiome-ml

Exploring Crohn's Disease through Machine Learning and the Gut Microbiome.

**Goal:** Build a Random Forest classifier to distinguish Crohn's Disease, Ulcerative Colitis, and healthy controls using gut microbiome abundance data from the HMP2 cohort.

**Stack:** Python · scikit-learn · Jupyter · pandas · UMAP · SHAP · GitHub

---

## Progress

### ✅ Day 1 — Environment Setup (May 22)
- Created GitHub repo and project skeleton
- Set up Python virtual environment
- Installed core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `umap-learn`, `shap`
- Created folder structure (`data/`, `notebooks/`, `figures/`)
- Confirmed Jupyter runs and imports work (`01_exploration.ipynb`)

### ✅ Day 2 — Biology Crash Course + Dataset Download (May 23)
- Studied gut microbiome biology: OTU/ASV tables, 16S rRNA sequencing, compositional data
- Downloaded HMP2 dataset from [ibdmdb.org](https://ibdmdb.org/):
  - `hmp2_metadata_2018-08-20.csv` — sample metadata (diagnosis, subject IDs, timepoints)
  - `taxonomic_profiles.tsv` — species-level taxonomic abundance table
- Generated initial sample count figure by diagnosis (`figures/sample_counts_by_diagnosis.png`)

---

## Dataset

**HMP2 / iHMP IBD Cohort** — [ibdmdb.org](https://ibdmdb.org/)

The Human Microbiome Project 2 (HMP2) longitudinal IBD study includes stool samples from patients with Crohn's Disease (CD), Ulcerative Colitis (UC), and healthy controls (nonIBD). Samples were collected over time with paired metagenomics, 16S rRNA, and clinical metadata.

---

## Project Structure

```
crohns-microbiome-ml/
├── data/                        ← HMP2 dataset files
│   ├── hmp2_metadata_2018-08-20.csv
│   ├── taxonomic_profiles.tsv
│   └── taxonomic_profiles.tsv.gz
├── notebooks/
│   └── 01_exploration.ipynb    ← initial data exploration
├── figures/
│   └── sample_counts_by_diagnosis.png
├── microbiome_10day_plan.md    ← full 10-day project plan
└── README.md
```

---

## Roadmap (10-Day Plan)

| Day | Goal | Status |
|-----|------|--------|
| 1 | Environment setup | ✅ Done |
| 2 | Biology crash course + download HMP2 dataset | ✅ Done |
| 3 | Data loading + EDA | ⬜ Up next |
| 4 | Preprocessing + feature engineering | ⬜ |
| 5 | Baseline model (Random Forest) | ⬜ |
| 6 | Model tuning + evaluation | ⬜ |
| 7 | UMAP visualization | ⬜ |
| 8 | SHAP feature importance | ⬜ |
| 9 | Biological interpretation | ⬜ |
| 10 | Write-up + polish | ⬜ |

---

## Getting Started

```bash
git clone https://github.com/liamc0004/crohns-microbiome-ml.git
cd crohns-microbiome-ml
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter umap-learn shap
jupyter lab
```
