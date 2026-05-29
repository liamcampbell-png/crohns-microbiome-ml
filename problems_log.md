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
