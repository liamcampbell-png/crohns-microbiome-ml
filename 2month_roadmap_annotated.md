# Crohn's Microbiome ML — Annotated Roadmap
**May 23 – July 16, 2026**

*This file is a companion to `2month_roadmap.md`. Every step here has the same goal, but instead of just telling you what to do, it explains why — what question we're trying to answer, what we'd learn from the result, and why that particular method was chosen over alternatives.*

---

## Week 1 — Foundation & Exploration

**The overarching question this week:** *Is this dataset actually usable, and does it contain a real biological signal worth modeling?*

Before writing a single ML model, you need to understand what you're working with. Skipping exploration is the most common reason ML projects fail — you build a model on data you don't understand, then can't explain why it works or doesn't.

---

**Why explore the dataset before anything else?**
You have 178 samples, 982 taxa, and 91.7% zeros. That sparsity number alone changes every modeling decision you'll make. If you'd gone straight to fitting a model, you'd be feeding it a near-empty matrix and wondering why it performs poorly. The exploration phase tells you whether the data is even in a condition to be modeled.

---

**Why learn about compositional data and apply CLR before modeling?**
The central statistical problem with microbiome data is that read counts aren't absolute measurements — they're relative shares of a sequencing budget. If you feed raw counts into a classifier, it will learn the total read depth of each sample as a confound, not the biology. A model trained this way might just learn "sample 206646 had 50,000 reads" rather than anything about Crohn's disease.

CLR removes this by expressing each taxon's abundance relative to the geometric mean of that sample. Now the feature values carry biological meaning: "this bacterium is above or below average for this patient," not "this bacterium had X raw reads." This is why CLR is the default preprocessing step in microbiome ML, not just a nice-to-have.

---

**Why compute Shannon diversity (alpha diversity)?**
Shannon diversity is your first real hypothesis test. The hypothesis is: *Crohn's disease patients have lower gut microbial diversity than healthy controls.* This is one of the most replicated findings in IBD research — a disrupted, less diverse microbiome is a hallmark of the disease. If your dataset doesn't show this signal, it raises a red flag about data quality or cohort selection before you invest weeks in modeling.

Shannon was chosen over simpler richness counts because it captures both the *number* of taxa and how *evenly* they're distributed. A gut dominated by one or two inflammation-tolerant bacteria (low evenness) is biologically different from one with many bacteria in rough balance, even if both technically contain the same number of species.

The Mann-Whitney test was chosen over a t-test because we cannot assume Shannon scores are normally distributed — with only 46 healthy controls and a hard floor at 0, parametric assumptions are unreliable. Mann-Whitney ranks all values together and tests whether one group's ranks are systematically higher: no normality needed.

---

**Why compute beta diversity and UMAP?**
Alpha diversity tells you about each sample in isolation. Beta diversity asks a different question: *are Crohn's microbiomes more similar to each other than to healthy microbiomes?* If yes, it means the disease has a coherent microbial signature that a classifier could potentially learn.

Bray-Curtis distance was chosen because it is the standard dissimilarity measure for ecological abundance data — it accounts for shared absence (two samples both missing a taxon doesn't make them more similar, which other distances would incorrectly imply).

UMAP then takes that distance matrix and projects it down to 2D so you can see visually whether CD and nonIBD samples cluster separately. If the two groups overlap completely in UMAP space, a classifier will struggle regardless of how sophisticated it is. Visual separation is a necessary (though not sufficient) signal that modeling is worthwhile.

---

## Week 2 — First ML Pipeline

**The overarching question this week:** *Can a simple model detect a Crohn's vs. healthy signal in the microbiome, and which bacteria is it using to do so?*

---

**Why Random Forest as the first model?**
Random Forest is the right starting point for high-dimensional, sparse biological data for three reasons. First, it handles the 982-feature, 178-sample ratio without needing explicit regularization — it naturally selects relevant features through random subsampling. Second, it is robust to the 91.7% sparsity because it splits on individual feature thresholds, not distances. Third, it provides feature importance scores essentially for free, so you can immediately ask "which bacteria matter?"

If you'd started with a neural network, you'd spend weeks tuning hyperparameters with no interpretability, on a dataset too small to benefit from deep learning. Random Forest gives you a reasonable answer quickly.

---

**Why 5-fold cross-validation rather than a single train/test split?**
With only 178 samples, a single 80/20 split gives you a test set of ~35 samples. The AUC you measure on 35 samples has enormous variance — you could get 0.85 one run and 0.72 the next depending on which 35 samples you happened to draw. Cross-validation averages the performance over 5 different splits, giving a much more stable estimate of how the model would generalize to new patients. This is standard practice whenever you have fewer than ~500 samples.

---

**Why compute SHAP values rather than just feature importances?**
Random Forest's built-in feature importance scores tell you which taxa the model relies on, but they have a known flaw: they overestimate the importance of high-cardinality or correlated features. SHAP (SHapley Additive exPlanations) is grounded in game theory — it asks "how much did each feature contribute to this specific prediction?" and distributes credit fairly across correlated features.

More importantly for this project, SHAP tells you *direction*: is this bacterium higher or lower in Crohn's patients? Standard importance scores don't tell you that. The biological sanity check — Googling your top SHAP taxa against the Crohn's literature — is how you validate whether your model rediscovered real biology or found a statistical artifact.

---

## Week 3 — Data Quality & Smarter Preprocessing

**The overarching question this week:** *Are we feeding the model the right representation of the data, and are we being honest about its quality?*

---

**Why revisit preprocessing after the first model?**
Most tutorials apply preprocessing once and move on. In research, preprocessing decisions are scientifically significant — they change what your model learns. This week is about making deliberate, defensible choices rather than accepting defaults.

---

**Why decide what to do with the 9 low-quality samples?**
Leaving them in risks training the model on noise. Dropping them arbitrarily risks introducing selection bias. The two main options — rarefaction (subsampling all samples to the same depth) and simply filtering — have different tradeoffs. Rarefaction discards real data; filtering discards samples. The goal isn't to pick the "correct" answer but to understand the tradeoff and document your reasoning. Any reviewer of your work will ask this question.

---

**Why filter rare taxa?**
A taxon detected in only 2 out of 178 samples is almost certainly noise, not biology. Including it as a feature doesn't help the model — it just adds dimensions for overfitting. The question "how rare is too rare?" is a judgment call, but the reasoning is: you're trying to find taxa that vary *systematically* between Crohn's and healthy patients. A taxon that appears in one or two samples can't be systematic.

---

**Why aggregate to genus level?**
Species-level OTUs are noisy — two reads that differ by a single base pair might be classified as different "species" even though they're effectively the same organism. Aggregating to genus smooths this noise and is more biologically interpretable. The comparison between genus and species level tells you whether that noise matters for your model. If performance is similar at genus level, you gain interpretability at no cost.

---

**Why address class imbalance?**
86 CD vs 46 nonIBD means your training data has nearly twice as many disease samples as healthy controls. A naive classifier could achieve decent accuracy by just predicting "Crohn's" for every sample. Stratified cross-validation (keeping the 86/46 ratio consistent across folds) and class weights (penalizing misclassification of the minority class more heavily) are the two standard fixes. Understanding this now prevents you from reporting misleadingly optimistic performance later.

---

## Week 4 — Multiple Models & Proper Evaluation

**The overarching question this week:** *Which type of model is actually best suited to this data, and how do we compare them fairly?*

---

**Why train four different models?**
No single algorithm is best for all problems. The four models (Random Forest, XGBoost, Logistic Regression L1, Linear SVM) test different assumptions about the data:

- **Random Forest** — assumes a small number of taxa combine non-linearly to predict disease. Good when the signal is carried by interactions.
- **XGBoost** — similar to Random Forest but builds trees sequentially, correcting previous errors. Often outperforms RF when the relationship between features and outcome is complex.
- **Logistic Regression (L1)** — assumes a linear relationship and automatically drives irrelevant feature weights to zero. With 982 features and 178 samples, this sparsity-inducing regularization is a real advantage. If this model performs similarly to Random Forest, it suggests the signal is largely linear — which would be a meaningful scientific finding.
- **Linear SVM** — finds the maximum-margin hyperplane between classes. Works well in high-dimensional spaces (like 982 taxa), especially when the number of features exceeds the number of samples.

Training all four with identical splits means any performance difference is due to the algorithm, not the data.

---

**Why track AUC rather than accuracy?**
Accuracy is misleading when classes are imbalanced (86 CD vs 46 nonIBD). A model that always predicts Crohn's would have 65% accuracy while being completely useless. AUC (Area Under the ROC Curve) measures how well the model *ranks* positive cases above negative ones, regardless of the decision threshold. An AUC of 0.80 means the model correctly ranks a randomly drawn Crohn's sample above a randomly drawn healthy sample 80% of the time — a meaningful, threshold-independent measure.

---

## Week 5 — Longitudinal Analysis

**The overarching question this week:** *Does the microbiome change over time in Crohn's patients, and can we detect a flare from a single sample?*

---

**Why does the longitudinal structure matter?**
The HMP2 dataset is not a simple cross-sectional study. Multiple stool samples were collected from each patient over months. If you treat each sample as independent (which most beginner projects do), you risk having the same patient appear in both training and test sets — the model memorizes that patient rather than learning generalizable biology. This is called data leakage, and it inflates your AUC.

Patient-level splits (all samples from a given patient go into either train or test, never both) give you an honest estimate of whether the model would generalize to a *new patient* it has never seen.

---

**Why plot individual patient trajectories?**
Before building anything sophisticated, you want to check whether diversity actually changes meaningfully over time for individual patients. If every patient's microbiome is stable regardless of their disease state, longitudinal modeling won't help. The trajectory plots are a sanity check — if diversity visibly drops around documented flare timepoints, that's your signal.

---

**Why look at flare periods specifically?**
The clinical goal that motivated the HMP2 dataset is: *can we predict a Crohn's flare before it happens?* If diversity drops not during a flare but in the weeks before it, that's a clinically actionable finding — you could in principle monitor a patient's microbiome and intervene early. This is why flare periods are the most interesting timepoints in the dataset, not just the cross-sectional diagnosis labels.

---

## Week 6 — Deep Biological Interpretation

**The overarching question this week:** *Does our model rediscover known biology, and what does it tell us that we didn't already know?*

---

**Why validate model findings against the literature?**
A model with 0.85 AUC is not inherently meaningful. The meaningful question is: *what did it learn?* If your top SHAP features include *Faecalibacterium prausnitzii* (a well-known anti-inflammatory commensal that is consistently depleted in IBD), that is evidence that your model found real biology. If your top features are obscure, poorly characterized taxa with no prior literature connection to IBD, you should be skeptical — it might be overfitting to dataset-specific noise.

This validation step is what separates a data science exercise from a piece of science.

---

**Why write a biological narrative?**
The ability to explain what your model found in plain language is how you demonstrate genuine understanding. A table of SHAP values is not a finding — it's an output. The narrative synthesizes the output into a claim: "our model primarily used depletion of butyrate-producing bacteria to distinguish Crohn's patients from healthy controls, consistent with the inflammation-microbiome feedback loop described in the literature." That sentence, if supported by your data, is something you can put in a portfolio, a blog post, or eventually a paper.

---

## Week 7 — Polish & Reproducibility

**The overarching question this week:** *Can someone else (or your future self in six months) reproduce and understand every decision you made?*

---

**Why does reproducibility matter for a personal project?**
Reproducibility is not just about satisfying reviewers. Practically: you will forget what you did. The markdown cells explaining each decision are written for your future self as much as anyone else. A notebook that runs clean from top to bottom with no hidden state is also a signal to potential employers that you understand the difference between exploratory hacking and production-quality analysis.

The methods section specifically exists because preprocessing choices (CLR vs. rarefaction, genus vs. species, how you handled low-quality samples) are the decisions most likely to be questioned. Writing them down forces you to justify them, which often reveals cases where you made a default choice you can't actually defend.

---

## Week 8 — Blog Post & Showcase

**The overarching question this week:** *How do you communicate what you found to someone who wasn't in the room?*

---

**Why write a blog post rather than just publish the GitHub repo?**
A GitHub repo demonstrates that you can code. A blog post demonstrates that you understand what you built. Hiring managers and collaborators who don't have time to read your notebook will read a well-written blog post. The two hero figures — the UMAP plot and the SHAP beeswarm — are chosen because each tells a complete story in a single image: "the groups separate" and "here are the bacteria responsible."

The Streamlit stretch goal is included because it forces you to think about your model as a tool someone else would use, not just a script. Designing for a user who doesn't know Python is a forcing function for clarity.

---

## End State (July 16)

By the end of this project you will have answered five questions:

1. **Does this dataset contain a detectable microbial signature of Crohn's disease?** (Weeks 1–2)
2. **Which preprocessing choices best preserve the biological signal?** (Week 3)
3. **Which type of model best captures the relationship between microbiome composition and disease?** (Week 4)
4. **Does the microbiome change predictably around disease flares?** (Week 5)
5. **Do the model's learned features align with known IBD biology?** (Week 6)

Answering these five questions cleanly and honestly, with reproducible code and clear writing, is what makes this project worth showing to anyone.
