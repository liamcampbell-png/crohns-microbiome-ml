# Methods

A compact record of the preprocessing and modeling decisions behind this project, and why each
was made. The runnable detail lives in the notebooks; this file is the prose a reviewer can read
without opening a kernel.

## Dataset

**HMP2 / iHMP IBD cohort** ([ibdmdb.org](https://ibdmdb.org/)). We use the 16S rRNA taxonomic
profiles and the clinical metadata.

- `data/taxonomic_profiles.tsv(.gz)` — taxa × samples count table (982 taxa × 179 samples).
- `data/hmp2_metadata_2018-08-20.csv` — per-sample metadata (diagnosis, participant, study week,
  disease-activity indices, fecal calprotectin, …).

Joining the taxonomic table to the 16S metadata rows yields **178 samples** with labels:
86 Crohn's disease (CD), 46 ulcerative colitis (UC), 46 healthy controls (nonIBD). The primary
classification task is **CD vs nonIBD** (132 samples).

## Preprocessing pipeline

Implemented once in `src/microbiome.py` and reused by every notebook.

1. **Prevalence filter** — keep taxa present (count > 0) in > 10% of samples. The raw table is
   91.7% zeros; rare taxa are noise, not signal. 982 → 184 taxa.
2. **CLR transform** — centered log-ratio with a 0.5 pseudocount. Microbiome counts are
   compositional (depth is an arbitrary total); CLR maps each sample onto log-ratios against its
   own geometric mean, removing the sum-to-constant constraint that breaks ordinary ML.
3. **Presence filter** — drop taxa above their geometric mean (CLR > 0) in fewer than 5% of
   samples, removing near-constant features.
4. **Correlation filter** — drop one taxon from each pair with |r| > 0.90 to limit redundancy and
   stabilise importances. Final feature set: **179 taxa**.

## Data-quality decisions (notebook 03)

- **Low-read samples — kept.** Nine samples fall below the 5th read-depth percentile; only four
  are in the CD/nonIBD task. Dropping them shifts AUC by ~0.01, well inside fold-to-fold noise, and
  with 132 samples discarding data is the worse trade. Rarefaction was rejected as lossy.
- **Class imbalance (86 CD / 46 nonIBD) — stratified folds + `class_weight='balanced'`.** AUC is
  already imbalance-robust; the weighting adds a small free gain and protects minority recall.
  Synthetic oversampling (e.g. SMOTE) was avoided — interpolating compositions in CLR space risks
  biologically implausible samples.
- **Compositional approach — CLR.** Aitchison distance (Euclidean on CLR) and ANCOM-BC
  (differential-abundance testing) are the relatives; CLR + a multivariate classifier is the right
  tool for a predictive, ranked-importance goal.

## Models (notebook 03)

Four models on one identical stratified 5-fold split (`shuffle=True, random_state=42`):
Random Forest (500 trees), XGBoost (300 trees, depth 3, lr 0.1, subsampled), L1 logistic
regression (standardised, `C=0.5`), and a linear SVM (standardised). Random Forest is carried
forward as the primary model (best AUC, native importances, clean SHAP support).

## Evaluation — the honest version (notebook 04)

HMP2 is longitudinal: participants contribute multiple visits. Sample-level cross-validation can
place a participant's visits in both train and test folds, so the model can recognise the *person*
rather than the *disease*. We therefore evaluate the final model with **`GroupKFold` by
participant**, which is the number this project reports as its headline. Naive 5-fold gives
AUC ≈ 0.83; patient-grouped CV gives **AUC ≈ 0.68** — the honest estimate for an unseen patient.

## Interpretation (notebook 05)

SHAP (TreeExplainer) on the Random Forest ranks taxa and gives each a direction (Δ CLR abundance,
CD − healthy). The five strongest features are all *depleted* in CD: the model's main signal is the
absence of health-associated commensals, consistent with the significant diversity reduction
(Shannon CD 2.44 vs healthy 2.78, Mann-Whitney p = 0.007).

## Known limitations

- **No taxonomy names.** The 16S table uses Greengenes-derived OTU ids (`Unc03y4v`, …) with no
  embedded lineage, and most discriminative taxa are `Unc` (uncultured) clones. Species-level
  literature mapping and genus-level aggregation would require the external Greengenes reference
  taxonomy, which is not in the downloaded file. Interpretation is therefore at the level of
  importance and direction, not Linnaean identity.
- **Small cohort.** Within-CD disease-activity contrasts (HBI remission vs active; diversity vs
  calprotectin) trend in the biologically-expected direction but are underpowered to reach
  significance.
