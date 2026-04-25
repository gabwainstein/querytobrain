# Cache Verification Report

Run: `python neurolab/scripts/verify_full_cache_pipeline.py`

## Verification Checklist

### 1. Atlas (Glasser 360 + Tian S2 = 392)

- **Path:** `neurolab/data/combined_atlas_392.nii.gz`
- **Structure:** Labels 1–360 = Glasser cortical; 361–392 = Tian S2 subcortical
- **Resampling:** All volumetric maps are resampled to this atlas via `resample_to_atlas()` before parcellation

### 2. Build Scripts Using Correct Resampling

| Script | resample_to_atlas | get_masker | Notes |
|--------|-------------------|------------|-------|
| build_term_maps_cache.py | ✓ | ✓ | NeuroQuery decoder |
| build_neurosynth_cache.py | ✓ | ✓ | NeuroSynth meta-analysis |
| build_neurovault_cache.py | ✓ | ✓ | NeuroVault task maps |
| build_neuromaps_cache.py | ✓ | ✓ | Neuromaps annotations |
| build_pharma_neurosynth_cache.py | ✓ | ✓ | Pharma NeuroSynth |
| build_enigma_cache.py | — | — | Uses DK→parcels mapping (summary stats, not NIfTI) |
| build_abagen_cache.py | — | ✓ | Uses abagen + atlas path directly |

### 3. Critical-Path Caches (full sizes)

| Cache | File | Expected | Build Command |
|-------|------|----------|---------------|
| decoder_cache | term_maps.npz | (~7K, 392) | `build_term_maps_cache.py --max-terms 0 --n-jobs 1` |
| neurosynth_cache | term_maps.npz | (~3.4K, 392) | `build_neurosynth_cache.py --max-terms 0` |
| unified_cache | term_maps.npz | (N, 392) | `merge_neuroquery_neurosynth_cache.py` |
| merged_sources | term_maps.npz | (N, 392) | `build_expanded_term_maps.py` (after merge) |
| neuromaps_cache | annotation_maps.npz | (29+, 392) | `build_neuromaps_cache.py --max-annot 0` |
| enigma_cache | term_maps.npz | (~50, 392) | `build_enigma_cache.py` |

**Full build:** `run_full_cache_build.py` (no `--quick`) or `build_full_caches.ps1` / `build_full_caches.sh`
| abagen_cache | term_maps.npz | (N, 392) | `build_abagen_cache.py` |
| pdsp_cache | pdsp_pc_projections.npz | (n_compounds, 392) | `build_pdsp_cache.py` |
| gene_pca | pc_scores_full.npy | (392, 15) | `run_gene_pca_phase1.py` + phase 2 |
| fc_cache | fc_degree.npy | (392,) | `build_fc_cache.py` |

### 4. Gaps and Remediation

**If merged_sources is missing:**
```bash
python neurolab/scripts/run_full_cache_build.py
# Or manually: build_expanded_term_maps.py with unified_cache + neuromaps + enigma + abagen
```

**If pdsp_cache is missing:**
```bash
# Prerequisites: pdsp_ki/KiDatabase.csv, gene_pca/
python neurolab/scripts/download_pdsp_ki.py  # if needed
python neurolab/scripts/run_gene_pca_phase1.py
python neurolab/scripts/run_gene_pca_phase2.py
python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache
```

**If gene_pca is missing:**
```bash
python neurolab/scripts/run_gene_pca_phase1.py
python neurolab/scripts/run_gene_pca_phase2.py
```

**Legacy 400-parcel caches (rebuild for 392):**
- cache_brainpedia_plus_decoder
- neurovault_cache_brainpedia
- neurovault_cache_test
- smoke_decoder_cache

Run the corresponding build scripts with the Glasser+Tian atlas in place.
