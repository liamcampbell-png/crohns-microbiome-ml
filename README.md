# crohns-microbiome-ml

Exploring Crohn's Disease through Machine Learning and the Gut Microbiome.

**Goal:** Build a Random Forest classifier to distinguish Crohn's Disease, Ulcerative Colitis, and healthy controls using gut microbiome abundance data from the HMP2 cohort.

**Stack:** Python · scikit-learn · Jupyter · pandas · UMAP · SHAP · GitHub

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


## Getting Started

```bash
git clone https://github.com/liamc0004/crohns-microbiome-ml.git
cd crohns-microbiome-ml
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter umap-learn shap
jupyter lab
```
