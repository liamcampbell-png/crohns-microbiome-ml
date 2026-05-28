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
- **Day 1 — Complete ✅** (2026-05-22): Environment setup, venv, libraries, folder structure, `01_exploration.ipynb` created and opened in JupyterLab.
- **Day 2 — Complete ✅** (2026-05-23): Biology crash course + downloaded HMP2 dataset (metadata CSV + taxonomic profiles TSV) from ibdmdb.org.
- **Day 3 — Complete ✅** (2026-05-24): Data exploration — answered 5 key questions (178 samples, 982 taxa, 91.7% sparsity, top taxa identified, 9 low-quality samples flagged). Bar chart saved to `figures/`.
- **Day 4 — Complete ✅** (2026-05-24): Compositional data explainer + CLR transform written in `01_exploration.ipynb`. `clr_transform()` function ready for use in modeling.
- **Day 5 — Up next (2026-05-26)**: Alpha diversity — Shannon diversity boxplot, Mann-Whitney test (Crohn's vs healthy).

## Environment
- **OS:** macOS
- **Virtual environment:** activate with `source venv/bin/activate` from the project root
- **Jupyter:** launch with `jupyter notebook` or `jupyter lab` after activating the venv

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
