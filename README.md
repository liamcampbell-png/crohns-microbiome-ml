# crohns-microbiome-ml

Exploring Crohn's Disease through Machine Learning and the Gut Microbiome.

**Goal:** Build a Random Forest classifier to distinguish Crohn's Disease, Ulcerative Colitis, and healthy controls using gut microbiome abundance data from the HMP2 cohort.

**Stack:** Python · scikit-learn · Jupyter · pandas · UMAP · SHAP · GitHub

---

## Progress

**Week 1 — Complete ✅** (May 23–28)
- Environment set up, HMP2 dataset downloaded
- Data explored: 178 samples, 982 taxa, 91.7% sparsity, 9 low-quality samples flagged
- CLR transform implemented for compositional data
- Alpha diversity: Shannon diversity computed, CD vs. nonIBD Mann-Whitney test
- Beta diversity: Bray-Curtis distance matrix computed and saved
- UMAP projection: 2D visualisation coloured by diagnosis, visual separation confirmed

**Week 2 — In progress** (May 29–Jun 4)
- [ ] Notebook cleaned up — clear markdown commentary, runs top-to-bottom on a fresh kernel
- [ ] Random Forest baseline trained — CLR features, 5-fold cross-validated AUC
- [ ] Feature importance: top 20 taxa identified
- [ ] SHAP values: beeswarm plot + biological sanity check

---

## Dataset

**HMP2 / iHMP IBD Cohort** — [ibdmdb.org](https://ibdmdb.org/)

The Human Microbiome Project 2 (HMP2) longitudinal IBD study includes stool samples from patients with Crohn's Disease (CD), Ulcerative Colitis (UC), and healthy controls (nonIBD). Samples were collected over time with paired metagenomics, 16S rRNA, and clinical metadata.

---

## Project Structure

```
crohns-microbiome-ml/
├── data/
│   ├── hmp2_metadata_2018-08-20.csv
│   ├── taxonomic_profiles.tsv
│   └── taxonomic_profiles.tsv.gz
├── notebooks/
│   └── 01_exploration.ipynb    ← exploration, CLR, diversity, UMAP
├── figures/
│   ├── sample_counts_by_diagnosis.png
│   ├── feature_presence_histogram.png
│   ├── shannon_diversity_boxplot.png
│   ├── pca_clr_projection.png
│   ├── pca_scatter.png
│   └── umap_scatter.png
├── 2month_roadmap.md           ← full roadmap (May 23 – Jul 16)
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
