# Implementation Acquisition Review

Cross-check of implementation docs vs. current data state. **Last run:** 2026-02-19.

---

## 1. NeuroVault Acquisition Guide

| Requirement | Status | Current | Notes |
|-------------|--------|---------|-------|
| **Tiers 1–4** (all collections) | ⚠️ Partial | 51/126 collections in manifest | Download in progress; run `download_neurovault_curated.py --all` to resume |
| **WM atlas** (7756–7761) | In script | — | Included in `--all` |
| **Pharmacological** collections | In script | — | Included in `--all` |
| **Slug-based** (EBAYVDBZ, OCAMCQFK, UOWUSAMV, ZSVLTNSF) | In script | — | Included in `--all` |
| **neurovault_curated_data** | ✅ Present | 3,053 images, 51 collections | Manifest recovered; 56 collections on disk |
| **neurovault_cache from curated** | 🔄 In progress | 20,090 (bulk) | `build_neurovault_cache` from curated running; will overwrite when done |

**Action:** Let curated download finish (`--all`), then rebuild cache. Current `neurovault_cache` is from bulk `neurovault_data`, not curated.

---

## 2. BUILD_MAPS_AND_TRAINING_PIPELINE

| Step | Component | Status | Current |
|------|------------|--------|---------|
| 0 | Combined atlas | ✅ | `combined_atlas_392.nii.gz` (Glasser+Tian) |
| 0a | Ontologies | — | Check `neurolab/data/ontologies/` |
| 0b | Neuromaps raw | ✅ | `neuromaps_data/` |
| 0c | NeuroVault curated | ⚠️ Partial | 3,053 images; download continuing |
| 1 | NeuroQuery decoder | ✅ | 4,983 maps × 392 |
| 2 | NeuroSynth | ✅ | 3,228 maps × 392 |
| 3 | Merge NQ+NS | ❌ Capped | `unified_cache`: 600 maps (built with cap; re-merge with full NQ+NS) |
| 4 | Neuromaps cache | ✅ | 40 maps × 392 |
| 4b | ENIGMA | ✅ | 49 maps × 392 |
| 4c | abagen | ✅ | 15,633 maps × 392 |
| 5 | NeuroVault cache | ⚠️ | From bulk (20K); curated build in progress |
| 5 | merged_sources | ✅ | 22,780 maps × 392 |
| 6c | PDSP Ki | ❌ | `pdsp_cache` missing |
| 6c | NeuroVault pharma | ✅ | 632 maps |
| 6c | OpenNeuro pharma | — | Optional |
| 6g | Luppi FC drug maps | ❌ | Manual extraction; 0 drug maps in fc_cache |

---

## 3. CACHE_VERIFICATION_REPORT

| Cache | Expected | Status | Current |
|-------|---------|--------|---------|
| decoder_cache | ~7K, 392 | ✅ | 4,983 × 392 (full usable set) |
| neurosynth_cache | ~3.4K, 392 | ✅ | 3,228 × 392 |
| unified_cache | N, 392 | ❌ Capped | 600 (was built with --max-terms cap; re-merge with full decoder+neurosynth) |
| merged_sources | N, 392 | ✅ | 22,780 × 392 |
| neuromaps_cache | 29+, 392 | ✅ | 40 × 392 |
| enigma_cache | ~50, 392 | ✅ | 49 × 392 |
| abagen_cache | N, 392 | ✅ | 15,633 × 392 |
| neurovault_cache | N, 392 | ⚠️ | 20,090 (bulk); curated build in progress |
| neurovault_pharma_cache | N, 392 | ✅ | 632 × 392 |
| pharma_neurosynth_cache | N, 392 | ✅ | 8 × 392 |
| pdsp_cache | pdsp_pc_projections.npz | ❌ | Missing |
| gene_pca | pc_scores_full.npy | ✅ | (392, 15); cortex/subcortex standardized separately |
| fc_cache | fc_maps.npz | ✅ | 8 × 392 |

---

## 4. FC_ACQUISITION_GUIDE

| Source | Type | Status | Current |
|--------|------|--------|---------|
| ENIGMA load_fc | Healthy | ✅ | 6 maps |
| netneurolab liu_fc-pyspi | Healthy | ✅ | 2 maps (fc_cons_400 + pyspi) |
| **Total healthy** | — | ✅ | 8 maps |
| Luppi 2023 drug FC | Drug | ❌ | 0 maps; manual extraction from supplement |

---

## 5. CRITICAL_PATH_CACHES_SPEC

| Cache | Requirement | Status |
|-------|-------------|--------|
| NeuroQuery full vocab | N ≥ 6,000 | ✅ 4,983 (full usable; 1,153 terms produce empty maps) |
| PDSP Ki cache | pdsp_cache/ | ❌ Missing |
| Gene PCA basis | gene_pca/pc_scores_full.npy | ❌ Missing |
| Training embeddings | embeddings/ | Optional; SKIP |

---

## 6. download_neurovault_curated.py vs Acquisition Guide

**Script collections (--all):**
- TIER_1: 8 collections ✓
- TIER_2: 22 collections ✓
- TIER_3: 45 collections ✓
- TIER_4: 24 collections ✓
- WM_ATLAS: 7756–7761 ✓
- SLUG_COLLECTIONS: EBAYVDBZ, OCAMCQFK, UOWUSAMV, ZSVLTNSF ✓
- PHARMA: 14 collections ✓

**Total: 126 collections** — matches acquisition guide.

---

## 7. Summary: Gaps to Address

| Priority | Gap | Action |
|----------|-----|--------|
| **High** | NeuroVault curated incomplete | Run `download_neurovault_curated.py --all` until complete; then `build_neurovault_cache` from curated |
| **High** | neurovault_cache from bulk | After curated build completes, it will overwrite; or run build manually when download done |
| **High** | unified_cache only 600 maps (capped) | Re-run: `merge_neuroquery_neurosynth_cache.py --neuroquery-cache-dir neurolab/data/decoder_cache --neurosynth-cache-dir neurolab/data/neurosynth_cache --output-dir neurolab/data/unified_cache --prefer neuroquery` |
| **Medium** | merged_sources uses bulk NeuroVault | Rebuild merged_sources after neurovault_cache is from curated |
| **Low** | pdsp_cache | Requires pdsp_ki + gene_pca; run Phase 1–2 + build_pdsp_cache |
| ~~Low~~ | ~~gene_pca~~ | ✅ Done (Phase 1+2 with --separate-cortex-subcortex) |
| **Low** | FC drug maps | 0 drug maps; Luppi 2023 requires manual extraction from supplement |

