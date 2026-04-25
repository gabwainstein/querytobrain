# Build the training dataset

This doc describes how to produce the **single dataset that goes into training**: fully preprocessed, curated, normalized, averaged (when needed), and merged into one (term, map) set. All commands assume you run from the **repository root** (the directory containing `neurolab/`).

---

## Goal and output

**Goal:** One cache directory that the trainer loads: **`neurolab/data/merged_sources/`**.

**Contents (training-ready):**

| File | Purpose |
|------|--------|
| `term_maps.npz` | `term_maps`: (N, 392) float — one 392-D map per term |
| `term_vocab.pkl` | List of N strings (text labels for each term) |
| `term_sources.pkl` | Source label per term (direct, neurovault, abagen, neuromaps, enigma, …) for source-weighted sampling |
| `term_sample_weights.pkl` | Per-term loss weight (used in weighted MSE) |
| `term_map_types.pkl` | Map type per term: `fmri_activation`, `structural`, `pet_receptor` (for type-conditioned MLP) |
| `abagen_gradient_components.npy` | (392, K) — gradient PCs when `--abagen-add-gradient-pcs` used; needed for PET residual-correlation evaluation |

**Preprocessing already applied before merge:**

- **Parcellation:** All maps are 392-D (Glasser 360 + Tian S2). Each source builder resamples to the pipeline atlas and parcellates.
- **Normalization:** Per source — global z for fMRI (NeuroQuery, NeuroSynth, NeuroVault); cortex/subcortex separate z for gene/PET/structural (abagen, neuromaps, ENIGMA). Merge does **not** re-normalize.
- **NeuroVault:** Curated 126 collections; subject-level collections are **averaged by contrast** (`--average-subject-level`); then global z. Result: ~2–4K maps (not raw image count).
- **PET residuals:** Optional gradient-regressed neuromaps/receptor terms (`_residual`) added at merge when `--add-pet-residuals` and abagen gradient PCs are used.

See [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) for full detail.

---

## One-command build (recommended)

From repo root:

```bash
python neurolab/scripts/run_full_cache_build.py
```

This runs, in order:

1. **Atlas** — `combined_atlas_392.nii.gz` (Glasser+Tian)
2. **Decoder cache** — NeuroQuery term maps (392-D, global z)
3. **NeuroSynth cache** — meta-analysis maps
4. **Unified cache** — merge NQ + NeuroSynth
5. **NeuroVault** — if `neurovault_curated_data/` exists: build with `--average-subject-level` → neurovault_cache (~2–4K maps); else skip or use bulk
6. **Neuromaps cache** — PET/receptor annotations (cortex/subcortex z)
7. **ENIGMA cache** — structural (cortical thickness, subcortical volume)
8. **Pharma NeuroSynth** (optional): uses **curated** term list from `neurolab/docs/implementation/neurosynth_pharma_terms.json` (`--pharma-terms-key all_terms_sorted` = 194 terms). Only terms with ≥3 studies in NeuroSynth produce maps. For uncurated expansion use `--all-drug-columns`.
9. **abagen cache** — gene expression (cortex/subcortex z)
10. **PDSP Ki cache** (required for drug inference): gene PCA Phase 1–2 (if needed), `download_pdsp_ki.py`, `build_pdsp_cache.py`. Compound→brain pharmacological pathway. Use `--skip-pdsp` to bypass.
11. **Merge** → `merged_sources/` with `--no-ontology`, `--save-term-sources`, `--abagen-add-gradient-pcs 3`, `--add-pet-residuals`, and all available cache dirs

If curated NeuroVault data is not present yet:

```bash
python neurolab/scripts/run_full_cache_build.py --download-neurovault-curated
```

This runs `download_neurovault_curated.py --all` first (126 collections; skips fetch if manifest + downloads already exist). Then the pipeline uses `neurovault_curated_data` and builds neurovault_cache with `--average-subject-level`.

**Skips (when steps are already done):**

```bash
python neurolab/scripts/run_full_cache_build.py --skip-decoder --skip-neurosynth --skip-neurovault --skip-neuromaps --skip-enigma --skip-abagen
```

Use only the skips you need; the merge step still runs if `unified_cache/term_maps.npz` exists and will include any cache dir that exists (neurovault, neuromaps, enigma, abagen, etc.).

---

## Verify before/after

Check that the training dataset and all upstream caches are present and 392-D:

```bash
python neurolab/scripts/verify_full_cache_pipeline.py
```

Strict mode (exit 1 on any failure):

```bash
python neurolab/scripts/verify_full_cache_pipeline.py --strict
```

Confirm parcellation and map types:

```bash
python neurolab/scripts/verify_parcellation_and_map_types.py
```

---

## Step-by-step (when you need fine control)

If you prefer to run each step yourself (e.g. different paths, caps, or order):

| Step | Command | Note |
|------|--------|-----|
| 1. Atlas | `python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz` | One-off; required for all parcellated caches. |
| 2. Decoder | `python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0` | NeuroQuery; 392-D, global z. |
| 3. NeuroSynth | `python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 0` | Meta-analysis; 392-D, global z. |
| 4. Unified | `python neurolab/scripts/merge_neuroquery_neurosynth_cache.py --neuroquery-cache-dir neurolab/data/decoder_cache --neurosynth-cache-dir neurolab/data/neurosynth_cache --output-dir neurolab/data/unified_cache` | NQ+NS merged. |
| 5. NeuroVault (curated) | `python neurolab/scripts/download_neurovault_curated.py --all --output-dir neurolab/data/neurovault_curated_data` (if needed) then `python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level` | Curated: 126 collections → ~2–4K maps after averaging. |
| 6. Neuromaps | `python neurolab/scripts/download_neuromaps_data.py` then `python neurolab/scripts/build_neuromaps_cache.py --cache-dir neurolab/data/neuromaps_cache --data-dir neurolab/data/neuromaps_data --max-annot 0` | PET/receptor; cortex/subcortex z. |
| 7. ENIGMA | `python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache` | Structural. |
| 8. Pharma NeuroSynth (optional) | `python neurolab/scripts/build_pharma_neurosynth_cache.py --output-dir neurolab/data/pharma_neurosynth_cache --data-dir neurolab/data/neurosynth_data --pharma-terms-key all_terms_sorted --min-studies 3` | Curated list from `neurosynth_pharma_terms.json` (194 terms); only terms with ≥3 studies produce maps. |
| 9. Abagen | `python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --all-genes` | Gene expression; needed for gradient PCs + PET residuals. |
| 10. PDSP Ki (required for drug inference) | `python neurolab/scripts/download_pdsp_ki.py` then `run_gene_pca_phase1.py` + `run_gene_pca_phase2.py` (if gene_pca/ incomplete) then `python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache --gene-pca-dir neurolab/data/gene_pca --pdsp-csv neurolab/data/pdsp_ki/KiDatabase.csv` | Compound→brain pharmacological pathway. Full build includes this. |
| 11. **Merge (training set)** | See below | Produces `merged_sources/`. |

**Merge command (full training set with PET residuals and gradient PCs):**

```bash
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir neurolab/data/unified_cache \
  --output-dir neurolab/data/merged_sources \
  --no-ontology --save-term-sources \
  --truncate-to-392 \
  --neurovault-cache-dir neurolab/data/neurovault_cache \
  --neuromaps-cache-dir neurolab/data/neuromaps_cache \
  --enigma-cache-dir neurolab/data/enigma_cache \
  --abagen-cache-dir neurolab/data/abagen_cache \
  --max-abagen-terms 500 \
  --abagen-add-gradient-pcs 3 \
  --gradient-pc-label-style distinct \
  --abagen-gene-info neurolab/data/gene_info.json \
  --abagen-pca-variance 0.95 \
  --add-pet-residuals
```

Add `--pharma-neurosynth-cache-dir neurolab/data/pharma_neurosynth_cache` if you built that cache. Add `--receptor-path path/to/receptor.csv` if you have a receptor CSV.

---

## After the build

- **Train:** `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources ...`
- **PET residual evaluation:** Load `abagen_gradient_components.npy` from `merged_sources/`; use `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)`.

See [DATA_PREPARATION_STATUS.md](DATA_PREPARATION_STATUS.md) for a compact checklist, [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) for normalization, weighting, and training behavior, and [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) for thorough step-by-step methods documentation.
