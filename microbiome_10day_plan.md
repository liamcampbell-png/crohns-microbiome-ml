# Crohn's Disease Microbiome ML Project
## 10-Day Kickoff Plan: May 22–31, 2026

**Goal:** Get from zero to a working Random Forest model with biological interpretation.
**Stack:** Python, scikit-learn, Jupyter, GitHub
**Dataset:** HMP2 / iHMP (Human Microbiome Project 2 — IBD cohort)

---

## Day 1 — Friday May 22 · 30 min
### Environment Setup & GitHub Repo

**Goal:** Have a working project skeleton before the weekend deep dives.

- [x] Create a new GitHub repo called `crohns-microbiome-ml` (public, add a README)
- [x] Set up a Python virtual environment: `python -m venv venv`
- [x] Install core libraries:
  ```
  pip install pandas numpy scikit-learn matplotlib seaborn jupyter umap-learn shap
  ```
- [x] Create this folder structure:
  ```
  crohns-microbiome-ml/
  ├── data/          ← raw downloaded files go here
  ├── notebooks/     ← all Jupyter notebooks
  ├── figures/       ← saved plots
  └── README.md
  ```
- [x] Open a blank notebook `01_exploration.ipynb` and confirm Jupyter runs

**Done when:** You can open Jupyter and import pandas with no errors.

---

## Day 2 — Saturday May 23 · 2 hours
### Biology Crash Course + Download the Dataset

**Goal:** Understand what you're working with before touching any data.

**Hour 1 — Learn the biology (don't skip this, it'll pay off later):**
- Read: [What is the human gut microbiome?](https://www.nature.com/articles/nrmicro2695) — just the abstract + intro (~15 min)
- Watch: Search YouTube for "16S rRNA sequencing explained" — pick any 10-min explainer
- Key concepts to understand:
  - **OTU/ASV table**: rows = samples, columns = bacterial species, values = abundance counts
  - **16S rRNA**: the gene used to identify bacteria without culturing them
  - **Compositional data**: abundances are relative (sum to 1), not absolute counts — this matters for ML

**Hour 2 — Get the dataset:**
- Go to [ibdmdb.org](https://ibdmdb.org/) — this is the official HMP2 portal
- Navigate to "Data" → look for the **16S rRNA** (Amplicon) dataset from the HMP2 study
- Download:
  - The **OTU abundance table** (species-level)
  - The **metadata file** (has columns like `diagnosis`, `sample_id`, `week_num`)
- Save both to your `data/` folder
- Alternative if ibdmdb.org is slow: search "HMP2 IBD 16S dataset" on [NCBI BioProject](https://www.ncbi.nlm.nih.gov/bioproject/) PRJEB2054

**Done when:** You have two files in `data/` — an abundance table and a metadata CSV.

---

## Day 3 — Sunday May 24 · 2 hours
### Data Exploration Part 1 — Understanding the Dataset

**Goal:** Know your data inside out before touching ML.

**In `01_exploration.ipynb`:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load
meta = pd.read_csv("../data/metadata.csv")
otu  = pd.read_csv("../data/otu_table.csv", index_col=0)

# Understand structure
print(meta.shape, otu.shape)
print(meta["diagnosis"].value_counts())   # How many Crohn's, UC, healthy?
print(otu.head())
```

**Questions to answer in the notebook (add a markdown cell for each):**
- How many samples total? How many per disease group?
- How many bacterial taxa (columns) are there?
- What percentage of the OTU table is zeros? (Microbiome data is very sparse)
- What are the top 10 most abundant bacterial genera?
- Are there any samples with very few reads (possible quality issues)?

**Make at least one plot:** A bar chart of sample counts by diagnosis group.

**Done when:** You can answer all 5 questions above and have at least one saved figure.

---

## Day 4 — Monday May 25 · 30 min
### Why Microbiome Data is Weird (Compositional Data)

**Goal:** Understand the one key preprocessing concept that separates good microbiome ML from bad.

- Read this short explainer: search "compositional data microbiome CLR transformation" — aim for a blog post or tutorial, not a paper
- Key insight: because abundances sum to 1 (or a constant), standard ML assumptions break. The fix is the **CLR (Centered Log-Ratio) transformation**
- Write the CLR transform in your notebook (you'll use it later):

```python
from scipy.stats import gmean

def clr_transform(df):
    # Add small pseudocount to handle zeros
    df_pseudo = df + 0.5
    log_df = np.log(df_pseudo)
    geometric_mean = log_df.mean(axis=1)
    return log_df.subtract(geometric_mean, axis=0)
```

- Add a markdown cell explaining in your own words what CLR does and why it matters

**Done when:** You have the CLR function written and can explain it in one sentence.

---

## Day 5 — Tuesday May 26 · 30 min
### Alpha Diversity — First Biological Signal

**Goal:** Reproduce a classic Crohn's finding: patients have lower gut microbial diversity.

```python
from scipy.stats import entropy, mannwhitneyu

# Shannon diversity (higher = more diverse)
def shannon(row):
    p = row / row.sum()
    p = p[p > 0]
    return entropy(p)

otu_meta = otu.join(meta.set_index("sample_id")["diagnosis"])
otu_meta["shannon"] = otu.apply(shannon, axis=1)

# Plot
sns.boxplot(data=otu_meta, x="diagnosis", y="shannon")
plt.title("Alpha Diversity (Shannon) by Diagnosis")
plt.savefig("../figures/alpha_diversity.png")

# Statistical test
cd = otu_meta[otu_meta.diagnosis == "CD"]["shannon"]
hc = otu_meta[otu_meta.diagnosis == "nonIBD"]["shannon"]
stat, p = mannwhitneyu(cd, hc)
print(f"Crohn's vs Healthy: p = {p:.4f}")
```

**Done when:** You have a boxplot saved and can see whether Crohn's samples have lower diversity.

---

## Day 6 — Wednesday May 27 · 30 min
### Beta Diversity — Do Samples Cluster by Disease?

**Goal:** Compute Bray-Curtis dissimilarity between samples (the standard microbiome distance metric).

```python
from sklearn.metrics import pairwise_distances

# Use relative abundances (normalize rows to sum to 1)
otu_rel = otu.div(otu.sum(axis=1), axis=0)

# Bray-Curtis distance matrix
bc_dist = pairwise_distances(otu_rel, metric="braycurtis")
bc_df = pd.DataFrame(bc_dist, index=otu.index, columns=otu.index)
print("Distance matrix shape:", bc_df.shape)
```

- This produces a sample × sample distance matrix — save it, you'll use it tomorrow for visualization
- Spend remaining time reading about what Bray-Curtis measures (it's a dissimilarity based on shared species)

**Done when:** Distance matrix is computed and saved as a CSV.

---

## Day 7 — Thursday May 28 · 30 min
### UMAP Visualization — See the Data

**Goal:** Create a 2D visualization that shows how samples relate to each other by disease.

```python
import umap

reducer = umap.UMAP(metric="precomputed", random_state=42)
embedding = reducer.fit_transform(bc_dist)

plt.figure(figsize=(8, 6))
colors = {"CD": "red", "UC": "orange", "nonIBD": "steelblue"}
for diag, color in colors.items():
    mask = otu_meta["diagnosis"] == diag
    plt.scatter(embedding[mask, 0], embedding[mask, 1],
                label=diag, color=color, alpha=0.6, s=20)
plt.legend()
plt.title("UMAP of Gut Microbiome Samples by Diagnosis")
plt.savefig("../figures/umap_diagnosis.png")
```

- Do Crohn's and healthy samples separate visually? (They should, partially)
- This plot will be the hero image of your blog post

**Done when:** UMAP plot is saved. Note in a markdown cell what you observe.

---

## Day 8 — Friday May 29 · 30 min
### Notebook Cleanup + GitHub Push

**Goal:** Make your work presentable and push everything to GitHub.

- Add a markdown cell at the top of `01_exploration.ipynb` explaining the project goal
- Add brief markdown commentary above each major code block (what it does, what you found)
- Restart kernel → Run All → confirm no errors
- Update `README.md` with:
  - 2-sentence project description
  - Dataset source link
  - How to run the notebook
- Commit and push everything to GitHub

**Done when:** Your GitHub repo is live, notebook runs clean end-to-end.

---

## Day 9 — Saturday May 30 · 2 hours
### First ML Model — Random Forest Baseline

**Goal:** Train a classifier to distinguish Crohn's from healthy controls.

**Create `02_modeling.ipynb`:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

# Prep: keep only Crohn's vs healthy (binary classification)
mask = otu_meta["diagnosis"].isin(["CD", "nonIBD"])
X = clr_transform(otu[mask])
y = (otu_meta[mask]["diagnosis"] == "CD").astype(int)

# Cross-validated ROC-AUC
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

print(f"Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit on full data for SHAP (tomorrow)
rf.fit(X, y)
```

- Try also: Logistic Regression, Gradient Boosting — compare AUCs in a table
- Plot a ROC curve
- Note your baseline AUC (literature typically reports ~0.75–0.85 for this task)

**Done when:** You have AUC scores for at least two models and a ROC curve plot.

---

## Day 10 — Sunday May 31 · 2 hours
### SHAP Values — Find the Key Bacteria

**Goal:** Identify which bacterial taxa drive predictions. This is where the biology comes alive.

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Beeswarm plot — top 20 most important features
shap.summary_plot(shap_values[1], X, max_display=20,
                  show=False)
plt.savefig("../figures/shap_beeswarm.png", bbox_inches="tight")
```

**The exciting part:** Take your top 10 SHAP features (bacterial species names) and Google them + "Crohn's disease". You should find that bacteria like *Faecalibacterium prausnitzii* (a known anti-inflammatory species depleted in Crohn's) appear near the top. This means your model rediscovered real biology.

- Write a markdown cell for each top bacterium: what it is, what the literature says about it in Crohn's
- Save the beeswarm plot — this will be a key figure in your blog post

**Done when:** SHAP plot saved, at least 3 top bacteria Googled and annotated in the notebook.

---

## End of Week 1 Checkpoint

By May 31 you should have:
- ✅ Working GitHub repo with clean, documented notebooks
- ✅ EDA with alpha diversity, beta diversity, and UMAP visualizations
- ✅ Baseline Random Forest model with cross-validated AUC
- ✅ SHAP analysis identifying biologically meaningful bacterial features

**Next phase (Week 2–3):** Compare additional models, handle longitudinal data, start blog post draft.

---

## Key Resources

| Resource | Link |
|---|---|
| HMP2 Dataset Portal | https://ibdmdb.org |
| Original HMP2 Nature Paper | doi: 10.1038/s41586-019-1237-9 |
| SHAP Documentation | https://shap.readthedocs.io |
| Microbiome ML Tutorial | Search "microbiome machine learning python tutorial" on Towards Data Science |
| UMAP Documentation | https://umap-learn.readthedocs.io |
