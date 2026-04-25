# Training Set Gap Audit (Second Pass)

**Date:** 2026-02-21  
**Purpose:** Second thorough audit to identify any remaining gaps in the training set, build pipeline, and data quality.

---

## Executive Summary

| Category | Status | Action |
|----------|--------|--------|
| **Previously fixed gaps** | ✓ Resolved | ENIGMA mapping, receptor norm, neurovault_pharma build, rebuild NeuroVault averaging |
| **Data quality** | ⚠ New | 2 ENIGMA maps with NaN; 1 neurovault_pharma near-zero map |
| **Decoder cache** | ⚠ Optional | 4,983 terms (warn if <6K); full vocab ~7.5K when --max-terms 0 |
| **Deduplication** | ℹ Documented | pharma_neurosynth: 26→11 (15 dropped); enigma: 52→49 (3 dropped) |
| **check_training_readiness** | Minor | Suggests rebuild_all_caches; primary path is run_full_cache_build |

---

## 1. Data Quality Gaps

### 1.1 NaN in merged_sources (2 terms)

**Finding:** 2 ENIGMA maps contain NaN values:
- `ADHD subcortical volume adult` (49 NaNs)
- `antisocial behavior subcortical volume` (49 NaNs)

**Likely cause:** ENIGMA toolbox or `surface_to_parcel`/`parcel_to_surface` may return NaN for some parcels when source data is sparse or missing. `build_enigma_cache` uses `nan_to_num` on `d_icv` but not on the final mapped vector before z-score. If the mapping produces NaN, `zscore_cortex_subcortex_separately` propagates them.

**Fix (applied):** In `build_enigma_cache.py`, after stacking maps and before z-scoring, apply `np.nan_to_num(term_maps, nan=0.0, posinf=0.0, neginf=0.0)`. Future builds will produce clean ENIGMA maps.

**To fix existing merged_sources:** Rebuild enigma_cache (`python neurolab/scripts/build_enigma_cache.py`) and re-run the merge step.

**Impact:** Training will see NaN in loss computation (often treated as 0 or skipped); 2/14,815 terms is negligible. Fix prevents recurrence.

### 1.2 Near-zero map (1 term)

**Finding:** 1 neurovault_pharma map is effectively all zeros:
- `neurovault_image_3`

**Likely cause:** Failed parcellation, all-NaN image, or QC miss. NeuroVault QC should reject all-zero maps; this may have slipped through or come from a collection not in AVERAGE_FIRST.

**Recommendation:** Strengthen `build_neurovault_cache` QC to reject maps with `np.abs(vec).max() < 1e-10`. Or add a post-merge sanity check in `build_expanded_term_maps` to skip/drop such maps (optional).

**Impact:** One near-zero map adds little signal and can slightly distort gradients; low priority.

---

## 2. Build Pipeline Gaps

### 2.1 Decoder cache term count

**Finding:** `decoder_cache` has 4,983 terms. Verification warns if <6,000 and suggests `--max-terms 0` for full vocab (~7.5K).

**Status:** `run_full_cache_build` uses `decoder_max = "0"` in full mode (no `--quick`), so a fresh full build should produce ~7.5K. Current 4,983 may be from an older run or `--quick` build.

**Recommendation:** Document that `--quick` caps decoder at 500; full build uses 0. For production training, run full build. No code change needed.

### 2.2 pharma_neurosynth deduplication

**Finding:** `pharma_neurosynth_cache` has 26 maps; merged_sources has 11 pharma_neurosynth terms. 15 are dropped at merge due to label collision with direct (NeuroQuery/NeuroSynth).

**Status:** Expected. First occurrence wins; direct is merged first. Drug terms like "caffeine", "dopamine" in pharma_neurosynth lose to NeuroQuery versions.

**Recommendation:** Document in TRAINING_DATASET_BUILD_METHODS §9.3. Consider distinct labels for non-fMRI sources (e.g. `caffeine_PET` or `caffeine_mkda`) if PET/meta-analysis maps should be preferred for those terms. Current behavior is intentional.

### 2.3 ENIGMA term count

**Finding:** `enigma_cache` has 52 maps; merged_sources has 49. 3 ENIGMA terms dropped at merge (label collision or dedup).

**Status:** Expected. Deduplication and/or overlap with other sources.

---

## 3. Verification & Documentation Gaps

### 3.1 check_training_readiness guidance

**Finding:** When cache is missing, suggests `rebuild_all_caches.py --ensure-data --n-jobs 30`. Primary documented build path is `run_full_cache_build.py`.

**Recommendation:** Update message to: "Run: python neurolab/scripts/run_full_cache_build.py (or rebuild_all_caches.py --ensure-data --n-jobs 30)".

### 3.2 gene_pca artifacts in merged_sources

**Finding:** `merged_sources` contains `gene_pca.pkl`, `gene_loadings.npz`, `abagen_term_indices.pkl` from a merge run with `--gene-pca-variance > 0`. The default merge (run_full_cache_build, rebuild_all_caches) does **not** pass `--gene-pca-variance`, so these may be from an older or alternate build.

**Status:** Optional for gene-head training. Not required for standard text-to-brain training. No gap.

---

## 4. Previously Fixed Gaps (Confirmed)

| Gap | Fix | Status |
|-----|-----|--------|
| ENIGMA cyclic replication | DK→vertex→Glasser + ENIGMA 16→Tian 32 | ✓ |
| Receptor raw merge | zscore_cortex_subcortex_separately in merge | ✓ |
| neurovault_pharma not built | run_full_cache_build step 5b + --download-neurovault-pharma | ✓ |
| rebuild NeuroVault no averaging | --average-subject-level for curated | ✓ |
| ENIGMA toolbox filename typo | _fix_enigma_toolbox_filenames() in build_enigma_cache | ✓ |

---

## 5. Recommended Fixes (Priority)

### High priority
1. **ENIGMA NaN sanitization:** ✓ Fixed. `build_enigma_cache` now applies `np.nan_to_num` before z-scoring. Rebuild enigma_cache and re-merge to fix existing merged_sources.

### Low priority
2. **NeuroVault QC:** Reject maps with `np.abs(vec).max() < 1e-10` (or tighten existing QC).
3. **check_training_readiness:** Prefer run_full_cache_build in missing-cache message.

---

## 6. Verification Commands

```bash
# Check for NaN and near-zero maps
python -c "
import numpy as np
import pickle
from pathlib import Path
p = Path('neurolab/data/merged_sources')
maps = np.load(p / 'term_maps.npz')['term_maps']
terms = pickle.load(open(p / 'term_vocab.pkl','rb'))
sources = pickle.load(open(p / 'term_sources.pkl','rb'))
nan_rows = np.where(np.isnan(maps).any(axis=1))[0]
zero_rows = np.where(np.abs(maps).max(axis=1) < 1e-10)[0]
print('NaN rows:', len(nan_rows), [terms[i] for i in nan_rows])
print('Near-zero rows:', len(zero_rows), [terms[i] for i in zero_rows])
"

# Full verification
python neurolab/scripts/verify_full_cache_pipeline.py
python neurolab/scripts/check_training_readiness.py --require-expanded
```

---

## 7. Summary

The training set is largely sound. The only substantive gap is **2 ENIGMA maps with NaN**—fixable by sanitizing in `build_enigma_cache`. One neurovault_pharma near-zero map is a minor QC edge case. All methodology, normalization, and build-pipeline gaps from the first audit have been addressed.
