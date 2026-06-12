# Problems Log

A running record of issues encountered during this project and how they were resolved.

---

## 1. UMAP produced uninformative scatter plot

**Date:** 2026-05-29  
**Problem:** Running UMAP directly on the full CLR-transformed OTU table (982 features) produced a plot with no visible structure or separation between diagnosis groups. Everything looked like random noise.  
**Root cause:** 91.7% of the OTU table is zeros. UMAP was trying to find manifold structure in a space dominated by near-zero noise across 982 features. Most taxa are present in almost no samples and carry no signal.  
**Fix (in progress):**
- Filter to **prevalent taxa** only — keep taxa present in >10% of samples (applied on raw counts before CLR)
- Remove **near-zero variance** features after CLR
- Collapse **highly correlated** features (|r| > 0.90)
- Redo CLR on the filtered table
- Add a **PCA step** (n_components=50) before UMAP to stabilise the projection
- Tune UMAP parameters: `n_neighbors=30`, `min_dist=0.05`

---

## 2. CLR implementation was incorrect (log1p, not CLR)

**Date:** 2026-05-29  
**Problem:** When rewriting the CLR cell after restructuring the notebook, used `np.log(otu_prevalent + 1)` followed by `StandardScaler`. This is a log1p transform, not CLR.  
**Root cause:** Forgot the "Centered" part of CLR — each sample's log-geometric-mean (the row mean after log) must be subtracted from every value in that row. Without this step, the sum-to-one compositional constraint is not removed.  
**Fix:** Correct three-step CLR:
```python
pseudo = otu_prevalent.astype(float) + 0.5
log_vals = np.log(pseudo)
otu_clr = log_vals.subtract(log_vals.mean(axis=1), axis=0)
```
Verify by checking that row means are all ~0.

---

## 3. `TypeError: float object has no callable log method`

**Date:** 2026-05-29  
**Problem:** `np.log(pseudo)` threw `TypeError: loop of ufunc does not support argument 0 of type float which has no callable log method`.  
**Root cause:** `otu_prevalent` had `object` dtype (Python floats instead of numpy float64), which `np.log` cannot handle via its ufunc path.  
**Fix:** Add `.astype(float)` when creating the pseudocount:
```python
pseudo = otu_prevalent.astype(float) + 0.5
```

---

## 4. Cross-validation AUC was inflated by patient leakage

**Date:** 2026-06-11
**Problem:** The Random Forest scored AUC ~0.83 with plain 5-fold CV, but that overstated real performance.
**Root cause:** HMP2 is longitudinal — participants contribute multiple visits. Shuffled sample-level folds put a patient's visits in both train and test, so the model partly learned to recognise *individuals*, not the disease.
**Fix:** Evaluate with `GroupKFold` by `Participant ID` so every test patient is unseen. Honest AUC drops to ~0.68. This is now the reported headline (notebook 04).

---

## 5. Taxa can't be mapped to species names

**Date:** 2026-06-11
**Problem:** Biological interpretation (Week 6) and genus-level aggregation (Week 3) both need taxon names; the SHAP top taxa are opaque ids like `Unc03y4v`.
**Root cause:** The HMP2 16S `taxonomic_profiles.tsv` labels rows with Greengenes-derived OTU ids and carries **no lineage column**. Most discriminative taxa are `Unc` (uncultured) clones with no validated species name. Resolving them needs the external Greengenes reference taxonomy, not in the downloaded file.
**Fix:** Interpret at the level the data supports — importance + direction of effect — and connect to literature at the *pattern* level (commensal depletion). Logged as a documented limitation rather than faked. Genus aggregation marked blocked.

---

## 6. Broken `venv/bin/jupyter` shebang after repo move

**Date:** 2026-06-11
**Problem:** `venv/bin/jupyter ...` fails with `bad interpreter: /Users/liamcampbell/Desktop/crohns-microbiome-ml/venv/bin/python3: no such file`.
**Root cause:** The project was moved/renamed; console-script shebangs in `venv/bin/` still point to the old absolute path. (Does not affect reproducibility — `venv/` is gitignored and the README rebuilds it.)
**Fix / workaround:** Launch via the module form `venv/bin/python -m jupyterlab`, or recreate the venv (`python -m venv venv && pip install -r requirements.txt`). Notebooks here were executed programmatically with `nbclient`.

---
