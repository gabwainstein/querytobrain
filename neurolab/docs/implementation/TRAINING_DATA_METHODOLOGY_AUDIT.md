# Training Data & Preprocessing Methodology Audit

**Date:** 2026-02-21  
**Purpose:** Map each source in the training set to its methodology, verify preprocessing was applied as designed, and identify gaps.

**See also:**
- [TRAINING_DATA_ALIGNMENT_REVIEW.md](TRAINING_DATA_ALIGNMENT_REVIEW.md) — deep review (data engineer, methodology, training, domain personas).
- [TRAINING_SET_GAP_AUDIT_2026-02-21.md](TRAINING_SET_GAP_AUDIT_2026-02-21.md) — second-pass gap audit (data quality, build pipeline).

---

## 1. Current training set composition (merged_sources)

| Source | Terms | % | Build script | Data path |
|--------|------|---|--------------|-----------|
| direct | 8,211 | 55.4% | decoder_cache + neurosynth_cache → merge | unified_cache |
| neurovault | 5,327 | 35.9% | build_neurovault_cache | neurovault_curated_data |
| neurovault_pharma | 632 | 4.3% | build_neurovault_cache | neurovault_pharma_data |
| abagen | 505 | 3.4% | build_abagen_cache | abagen (AHBA) |
| enigma | 49 | 0.3% | build_enigma_cache | enigmatoolbox |
| neuromaps | 40 | 0.3% | build_neuromaps_cache | neuromaps_data |
| neuromaps_residual | 40 | 0.3% | **Merge** (from neuromaps + gradient PCs) | — |
| pharma_neurosynth | 11 | 0.1% | build_pharma_neurosynth_cache | neurosynth_data |
| **TOTAL** | **14,815** | — | — | — |

**Not in current merged_sources:** receptor (requires `--receptor-path`), receptor_residual, ontology, reference.

---

## 2. Per-source methodology vs implementation

### 2.1 direct (unified_cache)

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | NeuroQuery + NeuroSynth merged | decoder_cache + neurosynth_cache → merge_neuroquery_neurosynth_cache | ✓ |
| **Resampling** | Resample to Glasser+Tian 392 | `resample_to_atlas(img)` in both builds | ✓ |
| **Parcellation** | 392-D via NiftiLabelsMasker | masker.transform() → 392-D | ✓ |
| **Normalization** | Global z-score (fMRI) | decoder: `stats.zscore(term_maps, axis=1)`; neurosynth: `zscore_maps(axis=1)` | ✓ |
| **Merge** | Prefer NeuroQuery on overlap; no re-norm | merge_neuroquery_neurosynth_cache: --prefer neuroquery default; concat as-is | ✓ |

**Decoder build:** `build_term_maps_cache.py` — NeuroQuery model → brain map per term → resample_to_atlas → parcellate → global z.  
**NeuroSynth build:** `build_neurosynth_cache.py` — NiMARE MKDA → resample_to_atlas → parcellate → zscore_maps(axis=1).  
**Merge:** `merge_neuroquery_neurosynth_cache.py` — union of terms; first occurrence wins (NQ preferred).

---

### 2.2 neurovault

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Curated collections (neurovault_curated_data) | build_neurovault_cache --data-dir neurovault_curated_data | ✓ |
| **Resampling** | Resample to atlas | `resample_to_atlas(img)` | ✓ |
| **Parcellation** | 392-D | masker.transform() | ✓ |
| **Averaging** | AVERAGE_FIRST: mean by contrast; others as-is | neurovault_ingestion.ingest_collection; AVERAGE_FIRST logic | ✓ |
| **QC** | Reject all-zero, high-NaN, extreme | qc_filter; --no-qc to skip | ✓ |
| **Normalization** | Global z-score | Stage 4: `zscore_maps(term_maps, axis=1)` | ✓ |
| **Sample weights** | Meta 2×, group 1×, subject-averaged 0.8× | term_sample_weights.pkl from get_sample_weight() | ✓ |

**zscore_maps:** `neurovault_ingestion.zscore_maps` — per-row (axis=1) mean/std, nan-safe.

---

### 2.3 neurovault_pharma

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Drug-related NeuroVault collections | download_neurovault_pharma → neurovault_pharma_data | ✓ |
| **Build** | Same pipeline as neurovault | build_neurovault_cache --data-dir neurovault_pharma_data --output-dir neurovault_pharma_cache | ✓ |
| **Resample / parcellate / z-score** | Same as neurovault | Same script; resample_to_atlas, parcellate, zscore_maps | ✓ |

**Build path:** `run_full_cache_build.py` builds neurovault_pharma_cache when neurovault_pharma_data exists. Use `--download-neurovault-pharma` to fetch data first if missing. Also: `rebuild_all_caches.py` (when data exists) or manually: `download_neurovault_pharma.py` → `build_neurovault_cache.py --data-dir neurovault_pharma_data --output-dir neurovault_pharma_cache --from-downloads`.

---

### 2.4 pharma_neurosynth

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Curated drug terms from neurosynth_pharma_terms.json | build_pharma_neurosynth_cache --pharma-terms-key all_terms_sorted | ✓ |
| **Method** | MKDA meta-analysis (NiMARE) | Same as neurosynth; MKDA per term | ✓ |
| **Resampling** | Resample to atlas | `resample_to_atlas(img)` | ✓ |
| **Parcellation** | 392-D | masker.transform() | ✓ |
| **Normalization** | Global z-score | `zscore_maps(term_maps, axis=1)` | ✓ |

---

### 2.5 neuromaps

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Neuromaps annotations (PET, metabolism, etc.) | build_neuromaps_cache; neuromaps.datasets | ✓ |
| **Resampling** | Resample to atlas | `resample_to_atlas(img)` | ✓ |
| **Parcellation** | 392-D | masker.transform() | ✓ |
| **Normalization** | Cortex (0:360) and subcortex (360:392) z-scored separately | `zscore_cortex_subcortex_separately(parcellated)` (--zscore-separate default True) | ✓ |

**Merge:** Uses `annotation_maps.npz` key `"matrix"`, not `"term_maps"`. Labels from `annotation_labels.pkl`.

---

### 2.6 neuromaps_residual

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Derived at merge from neuromaps + abagen gradient PCs | build_expanded_term_maps.py, --add-pet-residuals | ✓ |
| **Computation** | residual = map − projection onto first K gradient PCs | `residual = x - G.T @ coef` | ✓ |
| **Re-z-score** | Cortex/subcortex separately (audit fix) | `zscore_cortex_subcortex_separately(residual)` | ✓ |

**Requires:** abagen merged with --abagen-add-gradient-pcs > 0; neuromaps merged.

---

### 2.7 abagen

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | AHBA via abagen | build_abagen_cache --all-genes | ✓ |
| **Parcellation** | 392-D (abagen pipeline) | abagen parcellation to pipeline atlas | ✓ |
| **Normalization** | Cortex and subcortex z-scored separately | `zscore_cortex_subcortex_separately(vec)` (--zscore-separate default True) | ✓ |
| **Merge selection** | Tiered: receptor + cluster medoids + residual-variance; max 500 | --max-abagen-terms 500; Tier 1/2/3 in build_expanded_term_maps | ✓ |
| **Gradient PCs** | Add gene_expression_gradient_PC1..K | --abagen-add-gradient-pcs 3; saved as abagen_gradient_components.npy | ✓ |

---

### 2.8 enigma

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | ENIGMA summary stats (enigmatoolbox) | load_summary_stats(disorder) | ✓ |
| **Parcel mapping** | DK → pipeline atlas (392) | Cortical (68): parcel_to_surface(aparc_fsa5)→surface_to_parcel(glasser_360_fsa5); Subcortical (16): ENIGMA 16→Tian 32 (duplicate per struct) | ✓ |
| **Normalization** | Cortex and subcortex z-scored separately | `zscore_cortex_subcortex_separately(term_maps[i])` | ✓ |

**Fixed (2026-02-21):** Replaced cyclic replication with proper spatial mapping via enigmatoolbox `parcel_to_surface` / `surface_to_parcel` (cortical) and ENIGMA 16→Tian 32 mapping (subcortical).

---

### 2.9 receptor (not in current merged_sources)

| Aspect | Methodology (docs) | Code implementation | Match? |
|--------|-------------------|---------------------|--------|
| **Source** | Hansen CSV or receptor atlas | --receptor-path | — |
| **Normalization** | Cortex/subcortex separate | Merge applies `zscore_cortex_subcortex_separately` to receptor maps | ✓ |

**Fixed (2026-02-21):** Merge now applies `zscore_cortex_subcortex_separately` to receptor maps before concatenation.

---

## 3. Merge step preprocessing

**build_expanded_term_maps.py:**

| Source type | Merge preprocessing |
|-------------|---------------------|
| direct, neurovault, neurovault_pharma, pharma_neurosynth, abagen, enigma, neuromaps | **None** — maps concatenated as-is |
| neuromaps_residual, receptor_residual | **Computed** — residual = map − gradient projection; then `zscore_cortex_subcortex_separately(residual)` |
| --zscore-renormalize | Deprecated; would re-z all cortex/subcortex (bad for fMRI) |

**Deduplication:** First occurrence wins by normalized label (lowercase, strip, collapse whitespace). Order: base → neuromaps → neurovault → neurovault_pharma → pharma_neurosynth → receptor → enigma → abagen → residuals.

---

## 4. Build pipeline coverage

| Script | Builds | run_full_cache_build | rebuild_all_caches |
|--------|--------|---------------------|-------------------|
| build_term_maps_cache | decoder_cache | ✓ | ✓ |
| build_neurosynth_cache | neurosynth_cache | ✓ | ✓ |
| merge_neuroquery_neurosynth | unified_cache | ✓ | ✓ |
| build_neurovault_cache | neurovault_cache | ✓ (curated) | ✓ |
| build_neurovault_cache | neurovault_pharma_cache | ✓ (if neurovault_pharma_data exists; use --download-neurovault-pharma to fetch) | ✓ (if neurovault_pharma_data exists) |
| build_neuromaps_cache | neuromaps_cache | ✓ | ✓ |
| build_enigma_cache | enigma_cache | ✓ | ✓ |
| build_pharma_neurosynth_cache | pharma_neurosynth_cache | ✓ | ✓ |
| build_abagen_cache | abagen_cache | ✓ | ✓ |
| build_expanded_term_maps | merged_sources | ✓ | ✓ |

**Action:** Use `--download-neurovault-pharma` when running `run_full_cache_build.py` to fetch neurovault_pharma_data first if missing; then build+merge runs automatically.

---

## 5. Summary: methodology vs implementation

| Source | Doc normalization | Code normalization | Applied to data? |
|--------|-------------------|--------------------|--------------------|
| direct (NQ+NS) | Global z | Global z (stats.zscore / zscore_maps) | ✓ Yes |
| neurovault | Global z | Global z (zscore_maps) | ✓ Yes |
| neurovault_pharma | Same as neurovault | Same (build_neurovault_cache) | ✓ Yes |
| pharma_neurosynth | Global z | Global z (zscore_maps) | ✓ Yes |
| neuromaps | Cortex/subcortex separate | zscore_cortex_subcortex_separately | ✓ Yes |
| neuromaps_residual | Re-z cortex/subcortex | zscore_cortex_subcortex_separately | ✓ Yes |
| abagen | Cortex/subcortex separate | zscore_cortex_subcortex_separately | ✓ Yes |
| enigma | Cortex/subcortex separate | zscore_cortex_subcortex_separately | ✓ Yes |
| receptor | Cortex/subcortex separate | zscore_cortex_subcortex_separately in merge | ✓ Yes |

---

## 6. Identified gaps

### 6.1 neurovault_pharma not built by run_full_cache_build — FIXED

**Issue (resolved):** `run_full_cache_build.py` previously did not build neurovault_pharma_cache.

**Fix:** run_full_cache_build now builds neurovault_pharma_cache when neurovault_pharma_data exists. Use `--download-neurovault-pharma` to fetch data first if missing.

### 6.2 ENIGMA replicate fallback — FIXED

**Issue (resolved):** ENIGMA previously used cyclic replication for DK→392. Now uses proper spatial mapping: cortical via DK→vertex→Glasser, subcortical via ENIGMA 16→Tian 32.

**Fix:** For production, provide a proper DK→Glasser+Tian 392 mapping (or DK→Schaefer 400 then map to 392). Document the limitation and the --dk-to-schaefer option.

### 6.3 Receptor normalization — FIXED

**Issue (resolved):** Receptor maps were merged raw when added via --receptor-path.

**Fix:** Merge now applies `zscore_cortex_subcortex_separately` to each receptor map before concatenation.

---

## 7. Verification commands

```bash
# Per-source counts in merged_sources
python -c "
import pickle
from pathlib import Path
from collections import Counter
p = Path('neurolab/data/merged_sources')
with open(p / 'term_sources.pkl','rb') as f: src = pickle.load(f)
for s, n in sorted(Counter(src).items(), key=lambda x: -x[1]):
    print(f'  {s}: {n}')
"

# Full pipeline verification
python neurolab/scripts/verify_full_cache_pipeline.py

# Training readiness
python neurolab/scripts/check_training_readiness.py --require-expanded
```
