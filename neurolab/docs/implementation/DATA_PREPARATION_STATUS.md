# Data preparation: what’s ready and what’s to do

Checklist for getting from raw data to a training-ready merged cache. “Ready” = implemented and (where known) data or scripts in place; “To do” = run these steps if not done yet.

**Quick check:** Run from repo root:
```bash
python neurolab/scripts/verify_full_cache_pipeline.py
```
Or inspect `neurolab/data/` for the paths below.

**Build the training dataset (one command):** See [BUILD_TRAINING_DATASET.md](BUILD_TRAINING_DATASET.md) for the full preprocessed, curated, normalized, averaged → merged pipeline and `run_full_cache_build.py`.

---

## Evaluation result (example)

Last run found:

| Step | Status | Path / note |
|------|--------|-------------|
| Atlas (392) | Done | `neurolab/data/combined_atlas_392.nii.gz` |
| Decoder cache | Done | `neurolab/data/decoder_cache/term_maps.npz` |
| NeuroSynth cache | Done | `neurolab/data/neurosynth_cache/term_maps.npz` |
| Unified cache | Done | `neurolab/data/unified_cache/term_maps.npz` |
| NeuroVault curated | Done | `neurolab/data/neurovault_curated_data/manifest.json` |
| NeuroVault cache | Done | `neurolab/data/neurovault_cache/term_maps.npz` |
| Neuromaps cache | Done | `neurolab/data/neuromaps_cache/annotation_maps.npz` |
| ENIGMA cache | Done | `neurolab/data/enigma_cache/term_maps.npz` |
| Pharma NeuroSynth cache | Done | `neurolab/data/pharma_neurosynth_cache/term_maps.npz` |
| Abagen cache | Done | `neurolab/data/abagen_cache/term_maps.npz` |
| Merged sources | Done | `neurolab/data/merged_sources/term_maps.npz` (11,714 terms × 392 parcels); has `term_sources.pkl`, `term_sample_weights.pkl`, `term_map_types.pkl` |
| Ontologies | Done | `neurolab/data/ontologies/` |

**Gap:** If `merged_sources/abagen_gradient_components.npy` is missing, the merge was run before `--abagen-add-gradient-pcs` was added. Re-run the merge step (step 12) to produce gradient PCs and PET residual terms; then PET residual-correlation evaluation will work.

---

## Ready (implemented / in place)

| Item | Status |
|------|--------|
| **Pipeline spec** | [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md): parcellation, per-source normalization, NeuroVault averaging/QC, merge (no re-norm), source-weighted sampling, PET residuals, neuromaps/receptor weight bump. |
| **NeuroVault curated download** | Script skips fetch when `manifest.json` + `downloads/neurovault/` exist. **Curated = 126 collections → ~2–4K maps** in neurovault_cache (after `build_neurovault_cache --average-subject-level`); the download stores all raw images in those collections. Re-running `download_neurovault_curated.py --all` does not call `fetch_neurovault_ids` when data is present. Use `--force-fetch` only to re-download. |
| **NeuroVault build (curated)** | `build_neurovault_cache.py` with `--data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level`. Full build uses `--average-subject-level` when using curated data. |
| **Abagen tiered selection** | Merge: Tier 1 receptor, Tier 2 WGCNA-style medoids (~32), Tier 3 residual-variance; gradient PCs (`--abagen-add-gradient-pcs 3`); `abagen_gradient_components.npy` for PET residual-correlation eval. |
| **PET residual training targets** | `--add-pet-residuals`: gradient-regressed neuromaps/receptor maps as `*_residual` terms. Full build passes this when abagen + neuromaps/receptor are merged. |
| **Merge options** | `--save-term-sources`, `term_sample_weights.pkl`, neuromaps/receptor default weight 0.8, residual sources 0.6; SOURCE_TO_MAP_TYPE for residual sources. |
| **Trainer** | Source-weighted batch sampling (abagen ~10%, direct+neurovault ~60%); per-sample loss weights; type-conditioned MLP; optional gene head. Neuromaps/receptor 0.8, residual 0.6; sampling weights for `neuromaps_residual` / `receptor_residual`. |
| **Evaluation** | `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)` for PET specificity (informed by Fulcher et al.). |

---

## To do (run these if not done yet)

Run from repo root (`querytobrain/`). Order matters where dependencies exist.

| Step | Command / check | Notes |
|------|------------------|--------|
| **1. Atlas** | `python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz` | One-off; required for all parcellated caches. |
| **2. Ontologies** (optional for merged_sources) | `python neurolab/scripts/download_ontologies.py --output-dir neurolab/data/ontologies` | Needed only if you build an *expanded* cache with ontology terms. |
| **3. Decoder cache** | `python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache [--max-terms 0]` | NeuroQuery term maps (392-D, global z). |
| **4. NeuroSynth cache** | `python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache [--max-terms 0]` | Meta-analysis maps. |
| **5. Unified cache** | `python neurolab/scripts/merge_neuroquery_neurosynth_cache.py --neuroquery-cache-dir neurolab/data/decoder_cache --neurosynth-cache-dir neurolab/data/neurosynth_cache --output-dir neurolab/data/unified_cache` | NQ+NS merged; required before merged_sources. |
| **6. NeuroVault cache** | `python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level` | Curated data is already present; this step builds the parcellated cache (resample, average-by-contrast, QC, z-score). |
| **7. Neuromaps cache** | `python neurolab/scripts/download_neuromaps_data.py` then `python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache --data-dir neurolab/data/neuromaps_data --max-annot 0` | PET/receptor annotations; needed for merge + PET residuals. |
| **8. ENIGMA cache** | `python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache` | Structural (cortical thickness, subcortical volume). |
| **9. Pharma NeuroSynth** | `python neurolab/scripts/build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache --data-dir neurolab/data/neurosynth_data --pharma-terms-key all_terms_sorted --min-studies 3` | Optional. Uses **curated** list from `neurosynth_pharma_terms.json` (194 terms). Only terms with ≥3 studies in NeuroSynth produce maps (~8–many). For uncurated expansion use `--all-drug-columns`. |
| **10. Abagen cache** | `python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --all-genes` | AHBA gene expression; needed for tiered merge + gradient PCs + PET residuals. |
| **11. PDSP Ki cache** | `python neurolab/scripts/download_pdsp_ki.py` then `run_gene_pca_phase1.py` + `run_gene_pca_phase2.py` (if gene_pca/ incomplete) then `python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache --gene-pca-dir neurolab/data/gene_pca --pdsp-csv neurolab/data/pdsp_ki/KiDatabase.csv` | **Required** for drug inference (compound→brain pharmacological pathway). Full build includes this. |
| **12. Receptor CSV** (optional) | Place receptor atlas CSV (e.g. Hansen) and pass `--receptor-path ...` to merge. | Merge can run without it; neuromaps gives PET maps. |
| **13. Merge (merged_sources)** | `python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/unified_cache --output-dir neurolab/data/merged_sources --no-ontology --save-term-sources --neurovault-cache-dir neurolab/data/neurovault_cache --neuromaps-cache-dir neurolab/data/neuromaps_cache --enigma-cache-dir neurolab/data/enigma_cache --abagen-cache-dir neurolab/data/abagen_cache --max-abagen-terms 500 --abagen-add-gradient-pcs 3 --add-pet-residuals` (+ pharma/receptor if built) | Produces training set: term_maps.npz, term_vocab.pkl, term_sources.pkl, term_sample_weights.pkl, term_map_types.pkl, abagen_gradient_components.npy. |
| **14. Verify** | `python neurolab/scripts/verify_parcellation_and_map_types.py` | Confirms all caches use 392 parcels and correct map types. |

**One-command option:**  
`python neurolab/scripts/run_full_cache_build.py` runs steps 1–13 including PDSP (with skips for missing inputs). PDSP is **required** for drug inference; use `--skip-pdsp` only to bypass. Use `--skip-neurovault` (or others) if a step is already done or you don’t want it. Curated NeuroVault data is used automatically when present; NeuroVault cache build uses `--average-subject-level` for curated.

---

## After data preparation

- **Training:** `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources ...`
- **Evaluation vs PET:** Load `abagen_gradient_components.npy` from the merged cache dir; use `residual_correlation(pred, target, gradient_components)` for specificity.

See [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) and [BUILD_MAPS_AND_TRAINING_PIPELINE.md](BUILD_MAPS_AND_TRAINING_PIPELINE.md) for full detail.
