# Documentation Audit: Errors, Missing Items, and Improvements

**Date:** 2026-02-21  
**Scope:** All six core docs + accuracy-and-testing.md + CRITICAL_PATH_CACHES_SPEC.md  
**Method:** Cross-reference docs against each other, against our discussion specs, and internal consistency checks.

---

## 🔴 WILL BREAK (errors that affect correctness if docs are followed literally)

### 1. `abagen_gradient_components.npy` shape contradiction

**TRAINING_DATASET_BUILD_METHODS.md §10 (line 428):** Says shape is `(392, K)` — parcels × components.  
**TRAINING_READINESS_AUDIT.md §5.1 (lines 76, 93):** Says shape is `(5, 392)` — components × parcels.

**Verdict:** The audit's `(5, 392)` is correct. sklearn PCA convention stores `components_` as `(n_components, n_features)` = `(K, 392)`. The residual computation `residual = map - projection` needs to project a 392-D vector onto K components, which works with `(K, 392)` shape. The build methods doc has it transposed.

**Fix:** Change BUILD_METHODS line 428 from `(392, K)` to `(K, 392)`.

### 2. Merge command missing `--neurovault-pharma-cache-dir`

**BUILD_METHODS §9.7 (line 400-413):** The merge command includes `--neurovault-cache-dir`, `--neuromaps-cache-dir`, `--enigma-cache-dir`, `--abagen-cache-dir`, `--pharma-neurosynth-cache-dir` — but NOT `--neurovault-pharma-cache-dir`.

**AUDIT §4:** Lists `neurovault_pharma_cache (632 maps)` as present.  
**AUDIT §5.5:** Sources in merged_sources: `direct, neurovault, abagen, enigma, neuromaps, neuromaps_residual, pharma_neurosynth` — no `neurovault_pharma`.

**Verdict:** The 632 pharmacological NeuroVault maps (drug challenge fMRI studies — actual drug effects on brain activation) are **not** in merged_sources. These are among the highest-value training data for pharmacological prediction: real fMRI studies of drugs like psilocybin, ketamine, MDMA acting on the brain. They're built and sitting in the cache but never merged.

**Fix:** Add `--neurovault-pharma-cache-dir neurolab/data/neurovault_pharma_cache` to the merge command. Also add to `run_full_cache_build.py` step 12 description. This will add ~632 maps with weight 1.2 and source `neurovault_pharma`.

### 3. Decoder cache has only 4,983 terms (should be ~7K)

**AUDIT §3:** `decoder_cache: 4,983 maps`.  
**CRITICAL_PATH_CACHES §0.2:** Flags `--max-terms 5000` as a problem; says full vocab is ~7,548.  
**CRITICAL_PATH_CACHES §0.6:** "If n < 6000, the cache was likely built with a cap."

**Verdict:** The decoder cache was built with the old default cap. Missing ~2,500 terms (mostly rare/specialized neuroscience terms that NeuroQuery handles well via semantic smoothing). These are terms the model will never see during training, reducing coverage for rare queries.

**Impact:** Not a "break" per se (training works fine with 4,983), but a silent data loss of ~33% of available vocabulary. The audit says "OK" without flagging this.

**Fix:** Rebuild decoder cache with `--max-terms 0`. Update audit to flag if count < 6,000.

---

## 🟡 MISSING (content that should be in the docs but isn't)

### 4. No per-source term count breakdown in merged_sources

**AUDIT §5.3** says "7 unique sources" but never lists how many terms per source. Without this, you can't verify that abagen is actually capped at ~500, or that neurovault_pharma is missing (see #2 above).

**Fix:** Add table to audit:
```
| Source | Terms | % of total |
|--------|-------|------------|
| direct | ~8,200 | 57.8% |
| neurovault | ~5,300 | 37.4% |
| abagen | ~505 | 3.6% |
| enigma | ~49 | 0.3% |
| neuromaps | ~40 | 0.3% |
| neuromaps_residual | ~40 | 0.3% |
| pharma_neurosynth | ~26 | 0.2% |
| neurovault_pharma | 0 (MISSING) | 0% |
```

### 5. accuracy-and-testing.md is entirely legacy (400-D / Schaefer)

**Lines 13, 15, 22, 41, 159:** All reference "400-D" and "Schaefer" parcellation. Zero mentions of 392, Glasser, or Tian.

**Missing entirely:**
- Residual correlation (Fulcher) as evaluation metric for pharmacological predictions
- Source-weighted evaluation (reporting metrics per source type, not just overall)
- Gene head evaluation (how to measure quality of gene PCA predictions)
- Map-type-conditioned evaluation (fMRI vs structural vs PET accuracy separately)
- Any mention of the current atlas or pipeline

**Verdict:** This doc is from a previous pipeline generation and will mislead anyone running evaluation. It describes evaluation for a system that no longer exists.

**Fix:** Full rewrite needed. Update to 392 parcels, add residual_correlation metric, add per-source and per-map-type evaluation, remove Schaefer references.

### 6. PDSP processing pipeline not documented end-to-end

**BUILD_METHODS §8** describes `build_pdsp_cache.py` but:
- Doesn't describe the intermediate `process_pdsp_for_neurolab.py` step (Ki → pKi → affinity matrix)
- Doesn't document the RECEPTOR_TO_GENE mapping completeness (which PDSP receptor names map, which don't)
- Doesn't specify what happens when the automated download fails
- No validation step (how many compounds matched, how many receptors mapped)

**Fix:** Add subsection on PDSP preprocessing: download → process (parse Ki, filter species, map receptors, build pKi matrix) → gene PCA projection → spatial maps. Include expected output: ~3K-5K compounds with ≥3 receptors, ~60-80 gene symbols mapped.

### 7. No documentation of what happens when caches have overlapping terms

**BUILD_METHODS §9.3** says "first occurrence wins" for deduplication. But:
- What if "dopamine" appears in `direct` (from NeuroQuery — text-mined meta-analysis) AND in `pharma_neurosynth` (MKDA of studies mentioning dopamine) AND in `neuromaps` (PET dopamine receptor density)? These are completely different maps measuring different things.
- Currently, the NeuroQuery version wins because `direct` is merged first. The PET dopamine receptor map and the pharma meta-analysis map are silently dropped.
- This is especially problematic for receptor/neurotransmitter terms where the name collision is between fundamentally different data types.

**Impact:** PET receptor maps (source=neuromaps) with common names like "serotonin", "dopamine" may be lost if those terms already exist in the direct/neurovault cache. The audit can't detect this because it only counts totals.

**Fix:** Document the collision policy clearly. Consider: (a) namespace prefixing for non-fMRI sources (e.g. "5HT2A_PET" instead of "5HT2A"), (b) allowing multiple maps per term with different map_types, or (c) at minimum, logging which terms are dropped at merge time and verifying that precious PET maps aren't casualties.

### 8. Training command in audit doesn't match BUILD_METHODS merge command

**AUDIT §9** recommended training command: uses `--cache-dir neurolab/data/merged_sources` — correct.  
But it's missing `--max-abagen-terms 500` (which is a merge-step flag, not training).

More importantly, the training command doesn't specify:
- `--add-pet-residuals` (merge-step)
- Source-weighted sampling targets (what's the default in the code?)
- Gene head activation (automatic when `gene_pca.pkl` present? or needs flag?)

**Fix:** Add a clear two-step in the audit: (1) Merge command (with all flags), (2) Training command. Currently only step 2 is shown in §9.

---

## 🟢 IMPROVEMENTS (not wrong, but should be better)

### 9. Neuromaps weight should be 1.0, not 0.8

Both docs say neuromaps = 0.8. But there are only ~40 PET maps. With source-weighted sampling at 5% of batches AND loss weight 0.8, each PET map's effective contribution is tiny. These are the single most valuable ground-truth pharmacological training targets.

**Recommendation:** Bump to 1.0 (same as direct). With only 40 maps and 5% sampling, even 1.0 won't let them dominate. The 0.8 discount makes sense for abundant sources (neurovault) but not for scarce, precious ones.

### 10. PET residual maps: no re-z-score is potentially wrong

**PREPROCESSING §3.3 (line 68/82):** "No re-z-score is applied" to residual maps.

After regressing out gradient PCs, the residual has different variance than the original map. If the original was z-scored (cortex/subcortex separate), the residual will have reduced variance (some signal removed). Other training targets (fMRI) are z-scored to unit variance. The residual maps will have systematically lower variance → lower MSE → model learns to predict near-zero for residuals → loss for these terms is artificially low.

**Recommendation:** Consider z-scoring residual maps (cortex/subcortex separate) after computing them. Or at minimum, document why not re-z-scoring is deliberate and that the loss weight (0.6) partially compensates.

### 11. Tier 2 uses "cluster medoids" not actual WGCNA

**abagen_tiered_gene_selection.md §Three-tier selection:** Tier 2 is described as "WGCNA-style cluster medoids" but the implementation is hierarchical clustering on spatial correlation, not WGCNA (which uses a soft-thresholded adjacency matrix, topological overlap, and dynamic tree cut). 

The docs use "WGCNA" in the heading and throughout, which overstates what's implemented. The method is simpler (which is fine) but calling it WGCNA could confuse reviewers or collaborators.

**Fix:** Call it "co-expression cluster medoids" or "WGCNA-inspired clustering." Reserve "WGCNA" for if you actually implement the full algorithm.

### 12. `--max-abagen-terms 500` as a hard-coded recommendation

Both BUILD_METHODS and PREPROCESSING recommend 500. But the actual optimal number depends on the training set composition. With 632 neurovault_pharma maps added (fix #2), the non-gene portion grows and 500 abagen is fine. Without them, abagen at 500 is 3.6% of training — maybe too few.

**Recommendation:** Document the reasoning: "500 keeps abagen at ~3-5% of training set; adjust if other sources change significantly." The ratio matters more than the absolute number.

### 13. Gene PCA Phase 1 normalization vs merge normalization

**GENE_PCA_PIPELINE.md line 19:** Phase 1 uses `--separate-cortex-subcortex` (default) for standardization.  
**BUILD_METHODS §7.1:** abagen cache uses cortex/subcortex separate z-score.

These are consistent, which is good. But when the gene PCA basis (from Phase 2) is used to compute gradient components for PET residual subtraction, the PCs are in the cortex/subcortex-separate space. The PET maps (neuromaps) are ALSO cortex/subcortex-separate normalized. So the residual computation is consistent.

BUT: if someone changes abagen normalization without changing gene PCA Phase 1, the gradient components would be in a different space than the PET maps. This dependency isn't documented.

**Fix:** Add a note in abagen_tiered_gene_selection.md: "Gradient PCs must use the same normalization as the abagen cache and neuromaps cache (cortex/subcortex separate). If normalization changes, re-run gene PCA Phase 1-2."

### 14. No mention of text augmentation details

**PREPROCESSING §7.1 (line 192):** "Optional text augmentation (paraphrases / LLM variants)" — mentioned but never specified. What LLM? What prompt? How many variants per term? Are augmented terms in train only or also val/test?

This matters because text augmentation is the primary way to improve generalization for the semantic encoder. With 14K terms, each seen once, the encoder has limited data per concept.

**Fix:** Either remove the mention or add a subsection specifying the augmentation strategy.

### 15. No documentation of expected training metrics

The accuracy doc (which is legacy) mentions test ~0.64 with PubMedBERT, but that was on the old 400-D Schaefer decoder-only cache. With the new 392-parcel merged_sources (14K terms, multi-source, type-conditioned), expected metrics are unknown.

**Fix:** After first training run on the new pipeline, add expected baseline metrics to the audit: overall test correlation, per-source test correlation, residual correlation for PET maps.

---

## Summary table

| # | Severity | Doc | Issue |
|---|----------|-----|-------|
| 1 | 🔴 Error | BUILD_METHODS | gradient_components shape (392,K) should be (K,392) |
| 2 | 🔴 Missing data | BUILD_METHODS, AUDIT | neurovault_pharma (632 maps) not in merge command |
| 3 | 🔴 Silent loss | AUDIT, CRITICAL_PATH | Decoder cache has 4,983 terms, should be ~7K |
| 4 | 🟡 Missing | AUDIT | No per-source term count breakdown |
| 5 | 🟡 Legacy | accuracy-and-testing | Entirely 400-D/Schaefer, no 392, no Fulcher eval |
| 6 | 🟡 Missing | BUILD_METHODS | PDSP end-to-end processing not documented |
| 7 | 🟡 Missing | BUILD_METHODS | Term collision policy for different data types |
| 8 | 🟡 Missing | AUDIT | No merge command shown, only training command |
| 9 | 🟢 Improve | Both weight tables | neuromaps should be 1.0, not 0.8 |
| 10 | 🟢 Improve | PREPROCESSING | PET residual maps may need re-z-score |
| 11 | 🟢 Improve | abagen_tiered | "WGCNA" label overstates implementation |
| 12 | 🟢 Improve | BUILD_METHODS | 500 abagen should be ratio-justified, not magic number |
| 13 | 🟢 Improve | GENE_PCA_PIPELINE | Cross-normalization dependency not documented |
| 14 | 🟢 Improve | PREPROCESSING | Text augmentation mentioned but unspecified |
| 15 | 🟢 Improve | AUDIT | No expected baseline metrics for new pipeline |
