# Abagen tiered gene selection (DS / co-expression clustering / Fulcher)

When merging AHBA gene expression (abagen cache) into the expanded training set, we cap the number of gene maps to avoid overweighting and redundancy. The literature has converged on a **three-stage pipeline** (DS, WGCNA, PCA); NeuroLab implements a **tiered selection** plus **dual-axis** training (gradients + residual-selected genes) informed by Fulcher et al.

**Pipeline order:** First, get a **list of informative genes** (e.g. rank by residual variance after regressing out top gene-expression PCs, or by residual correlation with PET receptor maps). Then **filter** that list with tiered selection: Tier 1 = pharmacological anchors (receptor genes), Tier 2 = WGCNA-style cluster medoids, Tier 3 = top residual-variance (or medoids) to fill remaining slots. Informative-first, then filter — so the genetic information chosen is both pharmacologically relevant and non-redundant.

## Literature background

- **Differential stability (DS):** Hawrylycz et al. (Nat Neurosci 2015) defined DS as the tendency for a gene's spatial pattern to be reproducible across donors. High-DS genes are kept; low-DS genes are filtered. abagen already selects the probe with highest DS per gene; further DS filtering would require per-donor expression (e.g. `return_donors=True` or gene_pca phase 1).
- **WGCNA modules:** Hawrylycz et al. applied weighted gene co-expression network analysis (WGCNA) to high-DS genes and identified ~32 modules; each module is summarized by a **module eigengene** or **hub gene** (the gene most correlated to the eigengene). That gives a small set of representative spatial patterns.
- **PCA:** Dear et al. (Nat Neurosci 2024) and others show that the effective dimensionality of AHBA is much lower than the number of genes (e.g. top 100–200 PCs capture most variance). The full gene × parcel matrix is kept in `data/gene_pca/` for the pharmacological projection pathway; the **merge** step only decides which gene maps go into the **text-to-brain training set**.

- **Fulcher et al. (Nat Commun 2021):** [Overcoming false-positive gene-category enrichment](https://www.nature.com/articles/s41467-021-22862-1). Enrichment on atlas data is biased by within-category coexpression and spatial autocorrelation. The same structure inflates raw correlation between predicted and PET maps. **Residual correlation** (regress top gene-expression PCs from both, then correlate residuals) measures pharmacologically specific agreement. For training, include gradient maps and genes with **high residual variance** (unique local patterns).

References: Hawrylycz et al. 2015, Arnatkevičiūtė et al. (abagen), Dear et al. 2024 (cortical gene architecture), Fulcher et al. 2021 (false positives, residual correlation).

## Dual-axis training set

| Content | Maps | Purpose |
|--------|------|--------|
| Gradient PCs | 5–10 | Dominant spatial axes; structural basis and what to subtract for specificity. |
| Receptor genes (raw) | ~250 | Pharmacological anchors; gradient + specific. |
| Residual-selected genes | 200–300 | High variance after regressing PCs; unique local patterns. |
| WGCNA hub genes | ~32 | One representative per coexpression module. |
| PET receptor maps | ~40 | Raw in training; use residual correlation when evaluating predictions. |

## Three-tier selection (implemented)

When `--max-abagen-terms N` is set in `build_expanded_term_maps.py`, gene selection follows **informative first, then filter**: Tier 1 and Tier 2 define anchors and module representatives; Tier 3 fills remaining slots from an **informative** ranking (residual variance, or medoids). The steps are:

1. **Tier 1 — Pharmacological anchors (always include):**  
   All genes in the receptor list (`receptor_kb.load_receptor_genes()`, ~250) that appear in the abagen cache. These are the bridge between PDSP pharmacology and the Generalizer's semantic space (e.g. DRD2, SLC6A4, GABRA1).

2. **Tier 2 — Co-expression cluster medoids (WGCNA-inspired):**  
   From **non-receptor** genes, compute gene–gene **spatial correlation** (across parcels). Hierarchical clustering (average linkage on distance = 1 − correlation) is cut to obtain **K clusters** (default K = 32, `--abagen-n-clusters`). For each cluster, the **medoid** (gene with highest mean correlation to others in the cluster) is selected. These ~32 genes represent the main independent spatial patterns (e.g. cortical hierarchy, striatal, glial). Note: this is a simplified implementation inspired by WGCNA (Hawrylycz et al.) but uses standard hierarchical clustering rather than the full WGCNA algorithm (soft-thresholded adjacency, topological overlap matrix, dynamic tree cut).

3. **Tier 3 — Remaining slots:**  
   - **`residual_variance`** (default): Rank genes (excluding Tier 1/2) by variance of residual after regressing out top 5 PCs; take top N − (Tier1 + Tier2). Fulcher-aligned.  
   - **`medoids`**: Cluster remaining genes; one medoid per cluster.

**Gradient maps:** With `--abagen-add-gradient-pcs K` (e.g. 5–10), the first K PCs of the full abagen matrix are added as `gene_expression_gradient_PC1`, … and saved as `abagen_gradient_components.npy` for PET residual-correlation evaluation.

**Total:** Up to N gene maps plus K gradient maps; pharmacological anchors always included.

## Script options

- `--max-abagen-terms N` (default 0 = no cap): When set, use tiered selection; recommend 300–500.
- `--abagen-n-clusters K` (default 32): Number of Tier-2 co-expression clusters.
- `--abagen-add-gradient-pcs K` (default 0): Add K gradient PC maps. Recommend 5–10.
- `--abagen-tier3-method` (`residual_variance` | `medoids`): Tier 3 selection. Default `residual_variance`.

## PET residual correlation (evaluation)

When evaluating predicted vs PET receptor maps, **raw** correlation can be inflated by shared spatial autocorrelation. Regress dominant gene-expression PCs from both maps, then correlate the residuals. Load `abagen_gradient_components.npy` from the merged cache; use `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)`.

## Where the full gene set lives

- The **full** gene × parcel matrix is produced by the gene PCA pipeline and stored under `data/gene_pca/` for the pharmacological projection pathway.
- The **merge** step selects which gene maps and gradient PCs go into the merged training set and writes `abagen_gradient_components.npy` when `--abagen-add-gradient-pcs > 0`.

**⚠️ Normalization dependency:** Gradient PCs in `abagen_gradient_components.npy` are computed from the abagen matrix, which uses **cortex/subcortex separate** z-scoring. The PET maps (neuromaps) also use cortex/subcortex separate normalization. The PET residual computation (regressing gradient PCs from PET maps) is only valid when both are in the same normalization space. If abagen normalization changes, **re-run gene PCA Phase 1–2** and rebuild the merge with `--abagen-add-gradient-pcs`.

## Optional future: DS pre-filter

Differential stability filtering (keep only genes above median or top-quartile DS) would require per-donor expression (e.g. abagen with `return_donors=True` or using gene_pca phase 1 donor-level data). If added, it would be a **pre-filter** before clustering: run DS filter first, then apply the same three-tier selection on the high-DS set.
