# UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

**Paper:** McInnes, Healy & Melville (2020) — arXiv:1802.03426v3  
**Full title:** UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

---

## What is UMAP?

UMAP is a dimensionality reduction algorithm — it takes high-dimensional data (e.g., 982-dimensional microbiome taxa profiles) and projects it into a low-dimensional space (typically 2D or 3D) for visualization or downstream machine learning. Think of it as a smarter, faster alternative to t-SNE.

The core claim: UMAP produces embeddings competitive with t-SNE in visual quality, but better preserves **global structure**, runs significantly faster, and has no restriction on the output dimension (so it works as general-purpose preprocessing, not just visualization).

---

## The Core Idea (Non-Technical)

UMAP assumes your high-dimensional data lies on a lower-dimensional **manifold** — a curved surface embedded in high-dimensional space. For example, thousands of microbiome samples might actually vary along a handful of meaningful biological axes, even though you measured 982 taxa.

UMAP estimates the shape of that manifold from local neighborhood relationships, then finds a low-dimensional layout that preserves as much of that shape as possible.

The two-phase algorithm:

1. **Graph Construction** — Build a weighted graph where each data point connects to its k nearest neighbors. Edge weights represent how "likely" it is that two points are truly close on the underlying manifold.
2. **Graph Layout** — Use force-directed optimization to position points in 2D/3D such that the layout graph best matches the high-dimensional graph. Connected points attract; unconnected points repel.

---

## Theoretical Foundations

UMAP's design choices are grounded in **Riemannian geometry** and **algebraic topology** (specifically, the theory of fuzzy simplicial sets). This is what distinguishes it from t-SNE, which was derived more empirically.

### Key theoretical assumptions
1. Data lies (approximately) on a manifold.
2. The manifold is locally connected.
3. Preserving the topological structure of that manifold is the primary goal.

### Fuzzy simplicial sets
Rather than working directly with distances, UMAP converts local neighborhoods into **fuzzy simplicial sets** — a category-theoretic structure that encodes both the topology (which points are "near" each other) and the confidence of that proximity. This allows incompatible local views (each point has its own local distance scale) to be merged into a single global representation.

The key insight: each point $x_i$ gets its own local metric, normalized so that its nearest neighbor always has distance 0 (local connectivity) and the $k$-th neighbor is at a consistent normalized distance. These local metric spaces are converted to fuzzy topological objects and unioned together.

The algorithm then optimizes the 2D layout by minimizing **cross-entropy** between the fuzzy graph of the high-dimensional data and the fuzzy graph derived from the low-dimensional embedding.

---

## The Algorithm in Detail

### Phase 1: Graph Construction

For each point $x_i$:
- Find its $k$ nearest neighbors.
- Compute $\rho_i$ = distance to the nearest neighbor (sets the local scale; ensures local connectivity).
- Compute $\sigma_i$ via binary search so that the sum of edge weights equals $\log_2(k)$.
- Edge weight: $w(x_i, x_j) = \exp\!\left(-\frac{\max(0,\ d(x_i,x_j) - \rho_i)}{\sigma_i}\right)$

The directed graph is then symmetrized using a probabilistic union:

$$B = A + A^\top - A \circ A^\top$$

where $B_{ij}$ is interpreted as "probability that at least one of the two directed edges between $x_i$ and $x_j$ exists."

### Phase 2: Graph Layout (Optimization)

Initialize the embedding using a spectral decomposition of the normalized graph Laplacian (related to the Laplace-Beltrami operator of the manifold). Then run stochastic gradient descent to minimize cross-entropy, using:

- **Attractive force** along edges: pulls connected points together.
- **Repulsive force** via negative sampling: pushes unconnected points apart.

The attractive/repulsive forces are parameterized by $a$ and $b$, fit automatically from `min_dist`.

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_neighbors` | 15 | Number of neighbors used to approximate the local manifold. Small values: fine-grained local structure, but may fragment into many components. Large values: broader global structure, but local detail is averaged out. |
| `min_dist` | 0.1 | Minimum distance between points in the embedding. Low values: densely packed clusters that reveal internal structure. High values: points spread out, better for avoiding overplotting. Primarily aesthetic. |
| `n_epochs` | 200–500 | Training epochs for the SGD optimizer. More epochs = better optimization, slower runtime. |
| `n_components` | 2 | Dimensionality of the output embedding. Unlike t-SNE, UMAP works for any dimension, including higher-dimensional embeddings used as ML features. |

**Practical guidance:** For microbiome data with strong cluster structure (e.g., Crohn's vs. healthy), medium `n_neighbors` (10–30) is recommended. `min_dist` is mostly visual preference.

---

## Performance Results

UMAP was benchmarked against t-SNE, LargeVis, Laplacian Eigenmaps, and PCA on datasets including MNIST (70,000 × 784), Fashion-MNIST, COIL-20, and a 1M-point flow cytometry dataset.

### Quality
- UMAP matches t-SNE on local structure preservation (kNN classifier accuracy on the embedding is competitive).
- UMAP captures **more global structure** than t-SNE — class relationships in MNIST are spatially coherent (e.g., visually similar digits cluster near each other in the embedding).

### Speed
- UMAP is substantially faster than t-SNE, especially at scale. Empirical complexity: approximately $O(N^{1.14})$ driven by approximate nearest neighbor search.
- t-SNE becomes infeasible for very large datasets; UMAP ran successfully on 1,000,000-point flow cytometry data.

---

## Weaknesses

- **Constellation effect**: with small `n_neighbors`, noise gets interpreted as local manifold structure, producing spurious clusters.
- **Stochastic**: results vary across runs (random initialization + SGD). Use `random_state` for reproducibility.
- **Distance interpretation**: distances within clusters are meaningful; distances *between* clusters are less so (similar to t-SNE). Don't over-interpret inter-cluster spacing.
- **Supervised/semi-supervised** settings are possible but require additional care.

---

## Extensions and Future Work

The mathematical framework enables several extensions:
- **Supervised UMAP**: incorporate label information to guide the embedding.
- **Metric learning**: learn a custom distance metric during embedding.
- **Semi-supervised learning**: use partial labels.
- **Inverse transform**: map from the low-dimensional embedding back to the original space (useful for generative modeling).

---

## Relevance to This Project

For this Crohn's microbiome ML project, UMAP is used in **Week 2** as a visualization and feature-engineering tool:

- **Input**: CLR-transformed OTU matrix (samples × taxa, e.g., 178 × 982 after filtering).
- **Output**: 2D or 3D embedding where each point is a patient sample, colored by diagnosis (CD, UC, nonIBD).
- **Goal**: visually assess whether microbial composition separates disease states before fitting a classifier. A clean UMAP separation suggests the microbiome signal is strong; if groups overlap heavily, the classification problem is harder.
- **Follow-on**: UMAP embeddings can also serve as input features to downstream ML models (with the caveat that they encode a non-linear, stochastic transformation).

Key parameters to tune for microbiome data: `n_neighbors=15` (a reasonable default for 178 samples), `min_dist=0.1` for tight cluster visualization, `metric='euclidean'` on CLR-transformed data.
