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

- [ ] Notebook cleaned up — clear markdown commentary above every code block, runs top-to-bottom with no errors on a fresh kernel
- [ ] GitHub updated — everything committed and pushed, README describes the project and how to run it
- [ ] Random Forest baseline trained — CLR-transformed OTU table used as features, 5-fold cross-validated AUC reported, understand what the number means
- [ ] Feature importance understood — top 20 most important taxa identified by the Random Forest, can explain what "importance" means in this context
- [ ] SHAP values computed — beeswarm plot saved to `figures/`, top bacteria identified
- [ ] Biological sanity check done — Google top 5 SHAP taxa + "Crohn's disease", note whether the model rediscovered known biology (look for *Faecalibacterium prausnitzii*, *Roseburia*, *Ruminococcus*)

---

## Week 3 (Jun 5 – Jun 11) — Data Quality & Smarter Preprocessing

- [ ] Low-quality sample decision made — research the two options (rarefaction vs. dropping), decide what to do with the 9 flagged samples, document the rationale in a markdown cell
- [ ] Taxa filtering applied — understand why rare taxa add noise, filter out taxa present in very few samples, note how this changes the feature count
- [ ] Taxonomic aggregation explored — understand the difference between species, genus, family, and phylum level; aggregate the table at genus level and compare it to species level
- [ ] Class imbalance understood — notice that 86 CD vs 46 nonIBD is imbalanced, research what this does to a classifier, decide whether to address it (e.g. stratified splits, class weights)
- [ ] Read about ANCOM and Aitchison distance — understand that CLR is one of several compositional approaches, know roughly when you'd use alternatives

---

## Week 4 (Jun 12 – Jun 18) — Multiple Models & Proper Evaluation

- [ ] Research phase complete — understand how Random Forest, XGBoost, Logistic Regression (L1), and Linear SVM differ conceptually; know which tend to work well on high-dimensional sparse biological data and why
- [ ] All 4 models trained — same train/test setup used for every model so comparisons are fair
- [ ] Model comparison table built — AUC, precision, recall, and F1 recorded for each model in a clean table
- [ ] ROC curves plotted — all 4 models on the same axis, can visually see which performs best
- [ ] Best model identified — understand why it outperforms the others (e.g. handles sparsity better, less prone to overfitting with 178 samples)
- [ ] Target hit — at least one model at >0.80 AUC for CD vs nonIBD

---

## Week 5 (Jun 19 – Jun 25) — Longitudinal Analysis

- [ ] Longitudinal structure understood — know that HMP2 has multiple timepoints per patient, understand why treating them as independent samples is a problem
- [ ] Research done — read about how repeated-measures data is handled in ML (patient-level train/test splits, mixed-effects models, trajectory features)
- [ ] Patient trajectories plotted — for a handful of patients, Shannon diversity plotted over time; can see how their microbiome changes across visits
- [ ] Flare periods identified — use the metadata to find which timepoints correspond to disease flares vs. remission
- [ ] Diversity change analysed — does diversity drop before, during, or after a flare? Note the finding in a markdown cell
- [ ] Modeling decision made — decide whether to treat timepoints as independent or aggregate per patient for the final model, document the reasoning

---

## Week 6 (Jun 26 – Jul 2) — Deep Biological Interpretation

- [ ] Top 20 SHAP features fully documented — taxon name, whether it is higher or lower in Crohn's patients, confidence in the finding
- [ ] Literature review complete — each top taxon Googled against "Crohn's disease" or "IBD", key findings noted (one or two sentences per taxon is enough)
- [ ] Annotation table written into the notebook — taxon, direction of effect, literature summary, all in a readable markdown table
- [ ] Headline result identified — can you state in one sentence what the model found and whether it aligns with known biology?
- [ ] Biological narrative written — a markdown section in the notebook that tells the story of the findings to someone who knows data science but not gut biology

---

## Week 7 (Jul 3 – Jul 9) — Polish & Reproducibility

- [ ] Every notebook runs clean from top to bottom on a fresh kernel — no hidden state, no errors
- [ ] Every code block has a markdown cell above it explaining what it does and what you found
- [ ] Methods section written — explains preprocessing decisions (CLR, filtering, sample handling) and why each choice was made
- [ ] README is complete — project description, dataset source and citation, how to reproduce results, summary of key findings, link to blog post
- [ ] All figures are saved, clearly named, and referenced in the notebook
- [ ] GitHub repo is clean enough that a stranger could understand it in 5 minutes

---

## Week 8 (Jul 10 – Jul 16) — Blog Post & Showcase

- [ ] Blog post outline drafted — structure: gut microbiome intro, what is Crohn's, the dataset, what you did, what you found, why it matters
- [ ] Hero figures selected — UMAP plot and SHAP beeswarm are the two must-haves
- [ ] Full blog post written — aimed at someone who knows Python but not biology, no jargon without explanation
- [ ] Blog post published — Towards Data Science or personal site
- [ ] LinkedIn post written and shared — link to the GitHub repo and blog post
- [ ] Stretch goal: Streamlit app prototyped — user uploads a metadata file, sees a diversity plot

---

## End State (July 16)

- Clean, documented, reproducible GitHub repo
- Trained and interpreted ML model (>0.80 AUC target)
- Longitudinal analysis that most beginner projects skip
- Biological validation of model findings against the literature
- Published blog post and public GitHub presence
