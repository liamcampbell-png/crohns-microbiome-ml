# Project Context for Claude

## Project Overview
This is a **microbiome data science project** set up by Liam Campbell (liamc0004@gmail.com).

## Setup Status (as of 2026-05-22)
- GitHub repo created with README ✅
- Python virtual environment created ✅
- Required libraries installed ✅
- Folder structure created ✅
- Jupyter notebooks started (`01_exploration.ipynb`) ✅

## Progress
- **Weeks 1–2 — Complete ✅**: Setup, EDA, CLR, alpha/beta diversity, UMAP, RF baseline + SHAP, CoRF experiment (`01_exploration.ipynb`).
- **Week 3 — Complete ✅** (2026-06-11): Data-quality decisions (keep low-read samples, class-weight imbalance), compositional-methods writeup, 4-model comparison (`03_modeling.ipynb`).
- **Week 4 — Complete ✅** (2026-06-11): RF / XGBoost / L1-LogReg / Linear SVM compared on one split; ROC curves; RF best (`03_modeling.ipynb`).
- **Week 5 — Complete ✅** (2026-06-11): Longitudinal structure, diversity trajectories, disease-activity analysis, and the patient-leakage result — honest AUC ~0.68 via `GroupKFold` (`04_longitudinal.ipynb`).
- **Week 6 — Complete ✅** (2026-06-11): SHAP interpretation, direction-of-effect, biological narrative (`05_interpretation.ipynb`).
- **Week 7 — Complete ✅** (2026-06-11): `METHODS.md`, README rewrite, `requirements.txt`, all notebooks re-run clean.
- **Week 8 — Draft done ✅** (2026-06-11): `BLOG.md` written. External actions left for Liam: publish blog, LinkedIn post.

**Headline result:** a single 16S snapshot separates CD from healthy at AUC ~0.68 (honest, patient-grouped CV); the signal is depletion of health-associated commensals (significant diversity drop, p=0.007).

**Two documented limitations:** taxa can't be mapped to species (opaque Greengenes OTU ids), and the cohort is too small to power within-CD activity contrasts. See `problems_log.md`.

**Key files added this round:** `src/microbiome.py` (shared preprocessing), notebooks 03–05, `METHODS.md`, `BLOG.md`, `requirements.txt`.

## Environment
- **OS:** macOS
- **Virtual environment:** activate with `source venv/bin/activate` from the project root
- **Jupyter:** launch with `python -m jupyterlab` (the `venv/bin/jupyter` shebang broke when the repo was moved — see `problems_log.md` #6)

## Notebook Naming Convention
Notebooks are numbered sequentially, e.g.:
- `01_exploration.ipynb` — initial data exploration

## How to Start a Session
```bash
cd /path/to/this-project
source venv/bin/activate
jupyter notebook
```

## Reference
- The primary roadmap is `2month_roadmap.md` — check this for weekly goals and milestone checkpoints. The old 10-day plan is deprecated.

## Notes for Claude
- The user is early in the project — setup phase is complete, exploration is just beginning.
- When helping with notebooks, assume the venv is already activated unless stated otherwise.
- Refer to `microbiome_10day_plan.md` for project scope and day-by-day goals.
- User is comfortable with GitHub, virtual environments, and Jupyter — no need to over-explain basics.
