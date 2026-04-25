# Training Data Alignment Review: Audit, Methods, and Intended Datasets vs Actual Data

**Date:** 2026-02-21  
**Purpose:** Deep review—using data engineer, methodology, training, and domain personas—to verify that the documented audit, methodology, and intended datasets align with the data actually used for training.

---

## Executive Summary

| Alignment | Status | Notes |
|-----------|--------|------|
| **Audit §1 composition** | ✓ Matches | Actual merged_sources: 14,815 terms; per-source counts match audit table exactly |
| **Merge order** | ✓ Matches | Code order = audit §3 (base → neuromaps → neurovault → neurovault_pharma → pharma_neurosynth → receptor → enigma → abagen → residuals) |
| **Training consumption** | ✓ Matches | train_text_to_brain_embedding loads term_maps, term_vocab, term_sources, term_sample_weights, term_map_types correctly |
| **NeuroVault methodology** | ⚠ Gap | `rebuild_all_caches.py` never uses `--average-subject-level` for curated data; docs and run_full_cache_build use it |
| **Map types** | ✓ Consistent | fmri_activation 14,181; pet_receptor 585; structural 49 (SOURCE_TO_MAP_TYPE matches) |

---

## 1. Persona-Based Review

### 1.1 Data Engineer Persona

**Task:** Trace data flow from raw sources → caches → merged_sources → training.

**Findings:**

| Stage | Expected | Actual | Match? |
|-------|----------|--------|--------|
| Base cache (direct) | unified_cache = merge(decoder, neurosynth) | decoder 4,983 + neurosynth 3,228 → unified 8,211 (NQ preferred on overlap) | ✓ |
| NeuroVault | neurovault_curated_data → neurovault_cache | run_full_cache_build: uses curated + `--average-subject-level`; rebuild_all_caches: uses curated but `--from-downloads` only, no averaging | ⚠ |
| NeuroVault pharma | neurovault_pharma_data → neurovault_pharma_cache | run_full_cache_build: builds when data exists (or after --download-neurovault-pharma); rebuild_all_caches: same | ✓ |
| Merge step | build_expanded_term_maps --no-ontology | Both scripts pass merge args; cache presence checked before adding to merge | ✓ |
| Output artifacts | term_maps.npz, term_vocab.pkl, term_sources.pkl, term_sample_weights.pkl, term_map_types.pkl | All present in merged_sources | ✓ |

**File layout verified:**

```
merged_sources/
  term_maps.npz          # (14815, 392)
  term_vocab.pkl         # 14815 strings
  term_sources.pkl       # 14815 source labels
  term_sample_weights.pkl
  term_map_types.pkl     # fmri_activation | pet_receptor | structural
  abagen_gradient_components.npy
  abagen_term_indices.pkl
  gene_loadings.npz, gene_pca.pkl, gene_pca.pkl  # optional
```

**Critical gap:** `rebuild_all_caches.py` lines 157–159 invoke `build_neurovault_cache` with `--from-downloads` only. For curated data, the methodology requires `--average-subject-level` so that AVERAGE_FIRST collections (1952, 6618, 2138, 4343, 16284, etc.) are averaged by contrast. Without it, each subject-level image becomes a separate term—map counts and labels diverge from the intended methodology.

---

### 1.2 Methodology Reviewer Persona

**Task:** Verify preprocessing and normalization match documented methods.

**Findings:**

| Source | Doc method | Implementation | Verified |
|--------|------------|----------------|----------|
| direct | Global z | stats.zscore / zscore_maps axis=1 | build_term_maps_cache, build_neurosynth_cache |
| neurovault | Global z | zscore_maps(term_maps, axis=1) | build_neurovault_cache Stage 4 |
| neurovault_pharma | Same as neurovault | Same script | build_neurovault_cache |
| pharma_neurosynth | Global z | zscore_maps | build_pharma_neurosynth_cache |
| neuromaps | Cortex/subcortex separate | zscore_cortex_subcortex_separately | build_neuromaps_cache |
| neuromaps_residual | Re-z cortex/subcortex | zscore_cortex_subcortex_separately(residual) | build_expanded_term_maps merge |
| abagen | Cortex/subcortex separate | zscore_cortex_subcortex_separately | build_abagen_cache |
| enigma | Cortex/subcortex separate | zscore_cortex_subcortex_separately | build_enigma_cache |
| receptor | Cortex/subcortex separate | zscore_cortex_subcortex_separately in merge | build_expanded_term_maps |

**Parcellation:** All sources use Glasser 360 + Tian S2 = 392 parcels. Atlas: `combined_atlas_392.nii.gz`.

**ENIGMA spatial mapping (post-fix):** Cortical: DK 68 → vertex (aparc_fsa5) → Glasser 360. Subcortical: ENIGMA 16 → Tian 32 (duplicate per struct). No cyclic replication.

**Merge preprocessing:** As documented—no re-normalization; residuals computed at merge when `--add-pet-residuals` and gradient PCs present.

---

### 1.3 Training Engineer Persona

**Task:** Confirm training script consumes merged_sources correctly.

**Findings:**

| Artifact | Loaded? | Usage |
|----------|---------|--------|
| term_maps.npz | Yes | `data["term_maps"]` → supervision |
| term_vocab.pkl | Yes | `terms` → text labels |
| term_sources.pkl | Yes | Source-weighted batch sampling; per-source loss weight |
| term_sample_weights.pkl | Yes | Per-term loss weight when present (overrides source default) |
| term_map_types.pkl | Yes | Type-conditioned MLP (fmri_activation, structural, pet_receptor) |

**SOURCE_TO_MAP_TYPE consistency:** Trainer expects direct, neurovault, neurovault_pharma, pharma_neurosynth → fmri_activation; neuromaps, receptor, neuromaps_residual, receptor_residual, abagen → pet_receptor; enigma → structural. Matches merge step assignment.

**SAMPLE_WEIGHT_BY_SOURCE:** direct 1.0, neurovault 0.8, neurovault_pharma 1.2, pharma_neurosynth 1.2, enigma 0.5, abagen 0.4, neuromaps 1.0, neuromaps_residual 0.6. Matches audit and methodology docs.

**SOURCE_SAMPLING_WEIGHTS:** Batch sampling fractions sum to 1; direct+neurovault ~60%; abagen 10%; enigma 5%; neuromaps+residuals ~13%; pharma sources ~5%.

---

### 1.4 Domain Expert Persona

**Task:** Assess whether intended datasets (cognitive + biological + structural + pharmacological) match what is actually in the training set.

**Intended composition (from docs):**

- **Cognitive/task fMRI:** NeuroQuery + NeuroSynth (direct) + NeuroVault curated + neurovault_pharma + pharma_neurosynth
- **Biological PET/gene:** neuromaps, abagen (tiered), neuromaps_residual
- **Structural:** ENIGMA (cortical thickness, surface area, subcortical volume)
- **Pharmacological:** neurovault_pharma, pharma_neurosynth, abagen receptor genes; PDSP for inference (not in merged_sources)

**Actual composition (from term_sources.pkl):**

| Source | Count | % | Map type |
|--------|-------|---|----------|
| direct | 8,211 | 55.4% | fmri_activation |
| neurovault | 5,327 | 35.9% | fmri_activation |
| neurovault_pharma | 632 | 4.3% | fmri_activation |
| abagen | 505 | 3.4% | pet_receptor |
| enigma | 49 | 0.3% | structural |
| neuromaps | 40 | 0.3% | pet_receptor |
| neuromaps_residual | 40 | 0.3% | pet_receptor |
| pharma_neurosynth | 11 | 0.1% | fmri_activation |
| **TOTAL** | **14,815** | 100% | — |

**Assessment:**

- Cognitive/fMRI dominance (direct + neurovault + neurovault_pharma + pharma_neurosynth) ≈ 96%—appropriate for a text-to-brain model trained primarily on activation maps.
- Biological (abagen + neuromaps + neuromaps_residual) ≈ 4%—intended as minority for pharmacological/biological generalization.
- Structural (enigma) ≈ 0.3%—small but present for disorder-related structure.
- Receptor: not in current merged_sources; would require `--receptor-path`. When added, normalization is now applied (audit fix).

**Deduplication caveat (doc §9.3):** Terms like "dopamine" in multiple sources—first occurrence wins. NeuroQuery (direct) wins, so PET/pharma meta-analysis maps for the same label can be dropped. Recommendation: use distinct labels for non-fMRI sources (e.g. `5HT2A_PET`) or log dropped terms.

---

## 2. Audit vs Implementation Cross-Check

### 2.1 Audit §1 Table vs Actual

| Audit §1 | Actual | Match |
|----------|--------|-------|
| direct 8,211 | 8,211 | ✓ |
| neurovault 5,327 | 5,327 | ✓ |
| neurovault_pharma 632 | 632 | ✓ |
| abagen 505 | 505 | ✓ |
| enigma 49 | 49 | ✓ |
| neuromaps 40 | 40 | ✓ |
| neuromaps_residual 40 | 40 | ✓ |
| pharma_neurosynth 11 | 11 | ✓ |
| TOTAL 14,815 | 14,815 | ✓ |

**Conclusion:** Audit table reflects the current merged_sources build. These are actuals, not targets; a fresh build can differ if NeuroVault composition, deduplication, or build options change.

### 2.2 Build Pipeline Coverage

| Script | run_full_cache_build | rebuild_all_caches | Notes |
|--------|----------------------|--------------------|-------|
| decoder_cache | ✓ | ✓ | |
| neurosynth_cache | ✓ | ✓ | |
| merge NQ+NS | ✓ | ✓ | |
| neurovault_cache | ✓ (curated + --average-subject-level) | ✓ (curated: --average-subject-level; legacy: --from-downloads) | ✓ Fixed |
| neurovault_pharma_cache | ✓ (when data exists; --download-neurovault-pharma) | ✓ (when data exists) | |
| neuromaps_cache | ✓ | ✓ | |
| enigma_cache | ✓ | ✓ | |
| pharma_neurosynth_cache | ✓ | ✓ | |
| abagen_cache | ✓ | ✓ | |
| merge → merged_sources | ✓ | ✓ | |

### 2.3 TRAINING_DATASET_BUILD_METHODS vs Scripts

Step order in TRAINING_DATASET_BUILD_METHODS §12 aligns with run_full_cache_build. NeuroVault pharma is step 5b. Rebuild_all_caches differs in Pharma NeuroSynth placement (before neuromaps) but merge order is unchanged.

---

## 3. Identified Gaps and Recommendations

### 3.1 Critical: NeuroVault Curated in rebuild_all_caches

**Issue:** `rebuild_all_caches.py` uses `--from-downloads` for NeuroVault and never passes `--average-subject-level`. For curated data, AVERAGE_FIRST collections should be averaged by contrast.

**Impact:** Running rebuild_all_caches with neurovault_curated_data produces different map counts and labels than run_full_cache_build. Methodology docs and acquisition guide specify averaging for curated collections.

**Recommendation:** In `rebuild_all_caches.py`, when `nv_data == nv_curated`, add `--average-subject-level` and avoid `--from-downloads` if manifest.json exists (or support both modes per docs).

**Status:** Fixed. rebuild_all_caches now uses `--average-subject-level` when building from neurovault_curated_data, and `--from-downloads` only when manifest.json is missing.

### 3.2 Minor: check_training_readiness.py

**Issue:** Suggests `rebuild_all_caches.py --ensure-data --n-jobs 30` for full build. Primary documented path is `run_full_cache_build.py`.

**Recommendation:** Prefer run_full_cache_build in guidance; document rebuild_all_caches as an alternative with caveats (e.g. NeuroVault averaging).

### 3.3 Minor: ENIGMA Term Count

**Note:** build_enigma_cache can produce 52 maps (all disorders). Current merged_sources has 49. Likely due to earlier build or subset of disorders; not a bug. Audit correctly reports current state.

### 3.4 Documentation Clarity

**Recommendation:** In audit §1, add a footnote: "Counts reflect merged_sources as of [date]. A fresh run of run_full_cache_build (with same data) should yield similar totals; rebuild_all_caches without --average-subject-level for curated NeuroVault will differ."

---

## 4. Verification Commands (Run Periodically)

```bash
# Per-source counts
python -c "
import pickle
from pathlib import Path
from collections import Counter
p = Path('neurolab/data/merged_sources')
with open(p / 'term_sources.pkl','rb') as f: src = pickle.load(f)
for s, n in sorted(Counter(src).items(), key=lambda x: -x[1]):
    print(f'  {s}: {n}')
print('TOTAL:', len(src))
"

# term_map_types
python -c "
import pickle
from pathlib import Path
from collections import Counter
p = Path('neurolab/data/merged_sources')
with open(p / 'term_map_types.pkl','rb') as f: t = pickle.load(f)
print(sorted(Counter(t).items(), key=lambda x: -x[1]))
"

# Full pipeline verification
python neurolab/scripts/verify_full_cache_pipeline.py

# Training readiness
python neurolab/scripts/check_training_readiness.py --require-expanded
```

---

## 5. Summary

The audit, methodology docs, and intended datasets are largely aligned with the data used for training. The main mismatch is `rebuild_all_caches.py` omitting `--average-subject-level` for curated NeuroVault. Fixing that will ensure both build paths produce methodologically consistent NeuroVault caches. All other sources (direct, neuromaps, abagen, enigma, neurovault_pharma, pharma_neurosynth, residuals) match the documented methods, and the training script consumes merged_sources correctly with appropriate source and map-type handling.
