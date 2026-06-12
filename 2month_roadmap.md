# Crohn's Microbiome ML — 2-Month Roadmap
**May 23 – July 16, 2026**

---

## Week 1 (May 23 – May 28) — Foundation & Exploration

- [x] Environment set up — venv, libraries, folder structure, Jupyter running
- [x] Biology crash course complete — understand OTU tables, 16S rRNA, compositional data
- [x] HMP2 dataset downloaded — metadata CSV + taxonomic profiles in `data/`
- [x] Data explored — 178 samples, 982 taxa, 91.7% sparsity, top taxa identified, 9 low-quality samples flagged, bar chart saved to `figures/`
- [x] Compositional data understood — know why raw counts break ML and what CLR solves
- [x] CLR transform written and tested in notebook — ready to use in modeling
- [x] Alpha diversity computed — Shannon diversity calculated per sample, boxplot (CD vs nonIBD) saved, Mann-Whitney p-value reported, biological finding noted (do Crohn's patients have lower diversity?)
- [x] Beta diversity computed — Bray-Curtis distance matrix calculated, saved as CSV
- [x] UMAP visualization done — 2D plot colored by diagnosis saved to `figures/`, visual separation between CD and healthy noted in a markdown cell

---

## Week 2 (May 29 – Jun 4) — First ML Pipeline

- [x] Notebook cleaned up — clear markdown commentary above every code block, runs top-to-bottom with no errors on a fresh kernel
- [x] GitHub updated — everything committed and pushed, README describes the project and how to run it
- [x] Random Forest baseline trained — CLR-transformed OTU table used as features, 5-fold cross-validated AUC reported, understand what the number means
- [x] Feature importance understood — top 20 most important taxa identified by the Random Forest, can explain what "importance" means in this context
- [x] SHAP values computed — beeswarm plot saved to `figures/`, top bacteria identified
- [x] Biological sanity check done — model recovers the known *commensal-depletion* signature (top-5 SHAP taxa all depleted in CD); species names can't be confirmed because the 16S table uses opaque Greengenes OTU ids (see notebook 05)

---

## Week 3 (Jun 5 – Jun 11) — Data Quality & Smarter Preprocessing

- [x] Low-quality sample decision made — tested drop vs keep (ΔAUC ~0.01); decided to **keep** all samples, rarefaction rejected as lossy (notebook 03)
- [x] Taxa filtering applied — prevalence >10% + variance + correlation filters, 982 → 179 features (notebook 03 / `src/microbiome.py`)
- [~] Taxonomic aggregation explored — **blocked by data:** the 16S table has no taxonomy/lineage to aggregate on (opaque OTU ids); concept documented, limitation logged (notebook 03)
- [x] Class imbalance understood — stratified folds + `class_weight='balanced'` adopted; SMOTE rejected for compositional data (notebook 03)
- [x] Read about ANCOM and Aitchison distance — written up in notebook 03 / METHODS.md

---

## Week 4 (Jun 12 – Jun 18) — Multiple Models & Proper Evaluation

- [x] Research phase complete — conceptual comparison written into notebook 03
- [x] All 4 models trained — one identical stratified 5-fold split for every model (notebook 03)
- [x] Model comparison table built — AUC / precision / recall / F1 per model (notebook 03)
- [x] ROC curves plotted — all 4 on one axis, `figures/roc_curves.png`
- [x] Best model identified — Random Forest (AUC ~0.83); reasoning documented
- [x] Target hit — 3 of 4 models >0.80 AUC under naive CV (but see Week 5 — the honest patient-level AUC is ~0.68)

---

## Week 5 (Jun 19 – Jun 25) — Longitudinal Analysis

- [x] Longitudinal structure understood — repeated-measures structure mapped (notebook 04)
- [x] Research done — patient-level splits via `GroupKFold` applied and explained
- [x] Patient trajectories plotted — Shannon over study week for participants with ≥4 visits, `figures/shannon_trajectories.png`
- [x] Flare periods identified — disease activity via HBI (remission vs active) + fecal calprotectin, joined by participant+week
- [x] Diversity change analysed — active CD trends to lower diversity (NS, underpowered); CD-vs-healthy diversity drop is significant (p=0.007)
- [x] Modeling decision made — keep all timepoints, evaluate with patient-grouped CV; **honest AUC ~0.68** (notebook 04)

---

## Week 6 (Jun 26 – Jul 2) — Deep Biological Interpretation

- [x] Top 20 SHAP features fully documented — importance + direction (Δ CLR, CD−healthy) per taxon (notebook 05)
- [~] Literature review complete — **limited by naming:** done at the *pattern* level (commensal depletion matches IBD literature); per-species lookup blocked by opaque OTU ids
- [x] Annotation table written into the notebook — rank, importance, direction, effect (notebook 05)
- [x] Headline result identified — "Crohn's is marked by depletion of health-associated commensals; the model's top-5 features are all such depletions"
- [x] Biological narrative written — notebook 05 + `BLOG.md`

---

## Week 7 (Jul 3 – Jul 9) — Polish & Reproducibility

- [x] Every notebook runs clean from top to bottom on a fresh kernel — 01/03/04/05 re-executed via nbclient, 0 errors
- [x] Every code block has a markdown cell above it explaining what it does and what was found
- [x] Methods section written — `METHODS.md`
- [x] README is complete — description, citation, reproduce steps, key findings (blog is `BLOG.md`, not yet published externally)
- [x] All figures are saved, clearly named, and referenced (notebooks + README)
- [x] GitHub repo is clean enough that a stranger could understand it in 5 minutes

---

## Week 8 (Jul 10 – Jul 16) — Blog Post & Showcase

- [x] Blog post outline drafted — and written in full (`BLOG.md`)
- [x] Hero figures selected — `cv_leakage.png` (leakage), `taxa_direction.png` (biology), plus model comparison
- [x] Full blog post written — `BLOG.md`, aimed at a Python-literate non-biologist
- [ ] Blog post published — **external action, left for Liam** (draft ready in `BLOG.md`)
- [ ] LinkedIn post written and shared — **external action, left for Liam**
- [ ] Stretch goal: Streamlit app prototyped — not attempted (optional stretch)

---

## End State (July 16)

- Clean, documented, reproducible GitHub repo
- Trained and interpreted ML model (>0.80 AUC target)
- Longitudinal analysis that most beginner projects skip
- Biological validation of model findings against the literature
- Published blog post and public GitHub presence
