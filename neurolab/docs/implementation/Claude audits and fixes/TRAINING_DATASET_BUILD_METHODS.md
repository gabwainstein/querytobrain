# Training Dataset Build Methods

**This document is the canonical description of how the NeuroLab training data is built.** It matches the implementation in `build_expanded_term_maps.py`, `run_full_cache_build.py`, and the per-source cache builders. If you need to understand what the code does or reproduce the training dataset, use this document.

This document provides a **thorough, step-by-step** description of the methods used to build the NeuroLab training dataset: from raw data sources through preprocessing, normalization, averaging, curation, and merge into the final `merged_sources/` cache. It is intended for reproducibility, auditing, and handoff to other researchers or agents.

**Output:** `neurolab/data/merged_sources/` — a single (term, map) cache with ~14K terms × 392 parcels, fully preprocessed and ready for text-to-brain embedding training.

**Principle:** Normalization and grouping are done **per source at build time**. The merge step **does not** re-normalize; it concatenates terms and maps, records sources and sample weights, and optionally adds gradient PCs and PET residual maps.

---

## 1. Parcellation and atlas

### 1.1 Atlas definition

All brain maps use a single parcellation: **392 parcels** = Glasser 360 (cortex) + Tian S2 32 (subcortex). Defined in `neurolab/parcellation.py`; constants `N_CORTICAL_GLASSER`, `N_SUBCORTICAL_TIAN`, `N_PARCELS_392`.

### 1.2 Atlas build

**Script:** `build_combined_atlas.py`

**Command:**
```bash
python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz
```

**Output:** Single NIfTI with integer labels 1–360 (cortex) and 361–392 (subcortex). All subsequent cache builders use this atlas via `parcellation.get_combined_atlas_path()`, `get_n_parcels()`, and `get_masker()`.

### 1.3 Resampling

Every volumetric NIfTI (NeuroVault, neuromaps, etc.) is **resampled to the pipeline atlas** before parcellation. This ensures correct alignment regardless of source template, MNI space, or origin. Implemented in `parcellation.resample_to_atlas(img)`.

### 1.4 Parcellation

Each image is turned into a 392-D vector via `NiftiLabelsMasker` (or equivalent) using the combined atlas. All caches store maps as **392-D** float arrays.

---

## 2. Base cognitive maps: NeuroQuery and NeuroSynth

### 2.1 NeuroQuery decoder cache

**Source:** NeuroQuery predictive model (Dockès et al.) — term → brain map from coordinate meta-analysis.

**Script:** `build_term_maps_cache.py`

**Method:**
1. Load NeuroQuery decoder model and vocabulary.
2. For each term (or up to `--max-terms`): run decoder → volumetric map.
3. Resample to pipeline atlas.
4. Parcellate to 392-D.
5. **Normalization:** Global z-score across all 392 parcels per map (`zscore_maps(maps, axis=1)`).
6. Save `term_maps.npz`, `term_vocab.pkl`.

**Rationale for global z:** fMRI effect sizes (t/z) are comparable across cortex and subcortex; cross-compartment pattern (e.g. high striatum, low cortex) is meaningful and must be preserved.

**Command:**
```bash
python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0
```

### 2.2 NeuroSynth cache

**Source:** NeuroSynth database (Yarkoni et al.) — coordinate-based meta-analysis (e.g. MKDA) via NiMARE.

**Script:** `build_neurosynth_cache.py`

**Method:**
1. Fetch NeuroSynth data (version 7, abstract, terms) via NiMARE.
2. For each term with ≥ `--min-studies` (default 5) studies: run MKDA meta-analysis.
3. Resample result image to pipeline atlas.
4. Parcellate to 392-D.
5. **Normalization:** Global z-score across 392 parcels for each map.
6. Save `term_maps.npz`, `term_vocab.pkl`.

**Command:**
```bash
python neurolab/scripts/build_neurosynth_cache.py --cache-dir neurolab/data/neurosynth_cache --max-terms 0
```

### 2.3 Unified cache (merge NQ + NeuroSynth)

**Script:** `merge_neuroquery_neurosynth_cache.py`

**Method:**
1. Load decoder cache and NeuroSynth cache.
2. Merge term lists; prefer NeuroQuery when both have the same term (`--prefer neuroquery`).
3. Concatenate maps; no re-normalization.
4. Save to `unified_cache/`.

**Command:**
```bash
python neurolab/scripts/merge_neuroquery_neurosynth_cache.py \
  --neuroquery-cache-dir neurolab/data/decoder_cache \
  --neurosynth-cache-dir neurolab/data/neurosynth_cache \
  --output-dir neurolab/data/unified_cache --prefer neuroquery
```

**Result:** ~8K terms from NeuroQuery + NeuroSynth combined; used as the **base cache** for the merge step.

---

## 3. NeuroVault: task-contrast maps

### 3.1 Curated collection list

**Source:** 126 collections curated for task fMRI, meta-analyses, and structural atlases. Defined in `download_neurovault_curated.py` (Tiers 1–4 + pharmacological).

**Collection types:**
- **Use as-is:** Group-level or meta-analytic collections (one map per image or already consensus maps). No averaging.
- **Average first:** Subject-level collections (many images per contrast). These are **averaged by contrast** within collection before producing one map per contrast. IDs include: 1952, 6618, 2138, 4343, 16284 (high priority), plus 426, 445, 507, 504, … (medium). See [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md).

### 3.2 Download

**Script:** `download_neurovault_curated.py`

**Method:**
1. Fetch NeuroVault API for collection IDs.
2. Download NIfTI images for each collection.
3. Save to `neurovault_curated_data/downloads/neurovault/` with `manifest.json` (paths, `name`, `contrast_definition`, `task`, `collection_id` per image).
4. If `manifest.json` and `downloads/neurovault/` already exist with sufficient paths on disk, **skip** fetch (use `--force-fetch` to re-download).

**Command:**
```bash
python neurolab/scripts/download_neurovault_curated.py --all --output-dir neurolab/data/neurovault_curated_data
```

**Output:** Raw images in 126 collections. The **download** stores all raw images; the **curated training set** (~2–4K maps) is produced after averaging and cache build.

### 3.3 Cache build (with averaging)

**Script:** `build_neurovault_cache.py`

**Method (with `--average-subject-level`):**

1. **Load images** from curated data dir: manifest + NIfTIs.
2. **Resample** each image to pipeline atlas (`resample_to_atlas`).
3. **Parcellate** to 392-D.
4. **Per collection:**
   - If collection **in AVERAGE_FIRST:**
     - **QC** (optional): reject all-zero, high-NaN, extreme, or near-constant maps (`qc_filter`).
     - **Group by contrast:** key = `contrast_definition` or `cognitive_paradigm_cogatlas_id` or `name` or path stem (`get_contrast_key`).
     - **Average:** For each group with ≥ `min_subjects` (default 3, or 2 for heterogeneous collections), compute mean map. Optional **outlier removal:** drop maps with correlation to group mean < 0.2, then recompute mean if enough remain.
     - One **term** per contrast (label from key); one **map** per contrast; **sample weight** = 0.8 (subject-averaged).
   - If collection **not in AVERAGE_FIRST:**
     - **Use as-is:** one term per image (label from `contrast_definition` or name).
     - **QC** (optional): applied to full stack before writing cache.
     - **Sample weight** = 2.0 if collection in `META_ANALYTIC_COLLECTIONS`, else 1.0 (group).
5. **Z-score:** Global z-score across 392 parcels for all NeuroVault maps (`zscore_maps(term_maps, axis=1)`).
6. **Write** `term_maps.npz`, `term_vocab.pkl`, `term_collection_ids.pkl`, **`term_sample_weights.pkl`** (meta 2×, group 1×, subject-averaged 0.8×).

**Command:**
```bash
python neurolab/scripts/build_neurovault_cache.py \
  --data-dir neurolab/data/neurovault_curated_data \
  --output-dir neurolab/data/neurovault_cache \
  --average-subject-level
```

**Result:** ~2–4K maps (not raw image count); per-term sample weights carried forward to merge.

---

## 4. Pharmacological NeuroSynth

### 4.1 Curated term list

**Source:** `neurolab/docs/implementation/neurosynth_pharma_terms.json` — 194 pharmacologically relevant terms (neurotransmitters, drugs, receptors, systems). Categories: neurotransmitter names, drugs of abuse, pharmaceutical drugs, receptor terms, etc.

**Keys:**
- `all_terms_sorted`: 194 terms (full curated list).
- `high_confidence_terms`: ~80 terms (subset).

**Default:** Use `--pharma-terms-key all_terms_sorted` for the full curated list.

### 4.2 Term matching to NeuroSynth

**Method:**
1. Load NeuroSynth data (version 7, abstract, terms) via NiMARE.
2. For each term in the curated list: find NeuroSynth column whose normalized name contains the normalized term (e.g. "caffeine" → `terms_abstract_tfidf__caffeine`). Substring matching; spaces/underscores normalized.
3. Terms with no matching NeuroSynth column are skipped.
4. Terms with < `--min-studies` (default 3) studies (non-zero weight in annotations) are skipped.

**Alternative:** `--all-drug-columns` — include ALL NeuroSynth columns matching drug keywords (uncurated). Not used by default.

### 4.3 Meta-analysis and parcellation

**Method:**
1. For each matched term: run MKDA meta-analysis (NiMARE `MKDADensity`, kernel r=6, approximate null).
2. Resample result image to pipeline atlas.
3. Parcellate to 392-D.
4. **Normalization:** Global z-score across 392 parcels for each map.
5. Save `term_maps.npz`, `term_vocab.pkl`, `term_sample_weights.pkl` (weight 1.2 for all pharma terms).

**Script:** `build_pharma_neurosynth_cache.py`

**Command:**
```bash
python neurolab/scripts/build_pharma_neurosynth_cache.py \
  --output-dir neurolab/data/pharma_neurosynth_cache \
  --data-dir neurolab/data/neurosynth_data \
  --pharma-terms-key all_terms_sorted \
  --min-studies 3
```

**Result:** ~26 maps (number depends on NeuroSynth vocabulary and study counts; many curated terms have no matching column or < 3 studies).

---

## 5. Neuromaps (PET / receptor annotations)

### 5.1 Data source

**Source:** neuromaps (Markello et al.) — PET receptor maps, structural annotations, etc. in MNI space.

**Script:** `download_neuromaps_data.py` then `build_neuromaps_cache.py`

### 5.2 Preprocessing

**Method:**
1. Fetch neuromaps annotations (if not present).
2. For each annotation: load volumetric map, resample to pipeline atlas, parcellate to 392-D.
3. **Normalization:** **Cortex (0:360) and subcortex (360:392) z-scored separately.** Reason: PET binding and structural measures have different absolute scales by tissue; global z would let cortex dominate and distort subcortical pattern.
4. Save `annotation_maps.npz` (key `matrix`), `annotation_labels.pkl`.

**Command:**
```bash
python neurolab/scripts/download_neuromaps_data.py
python neurolab/scripts/build_neuromaps_cache.py \
  --cache-dir neurolab/data/neuromaps_cache \
  --data-dir neurolab/data/neuromaps_data \
  --max-annot 0
```

**Result:** ~40 maps (PET receptor, structural annotations).

---

## 6. ENIGMA (structural disorder maps)

### 6.1 Data source

**Source:** ENIGMA cortical thickness (CT), surface area (SA), subcortical volume (SubVol) effect sizes for psychiatric disorders.

**Script:** `build_enigma_cache.py`

### 6.2 Preprocessing

**Method:**
1. Load ENIGMA effect sizes (parcellated or volumetric).
2. Resample to pipeline atlas if needed; parcellate to 392-D.
3. **Normalization:** Cortex and subcortex z-scored separately (same rationale as neuromaps).
4. Save `term_maps.npz`, `term_vocab.pkl` (e.g. "schizophrenia cortical thickness", "major depression cortical thickness").

**Command:**
```bash
python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache
```

**Result:** ~49 maps (structural).

---

## 7. Abagen (gene expression)

### 7.1 Full gene cache

**Source:** Allen Human Brain Atlas (AHBA) via abagen — gene expression parcellated to Glasser+Tian.

**Script:** `build_abagen_cache.py`

**Method:**
1. Run abagen pipeline: probe selection (highest DS per gene), sample-to-parcel mapping.
2. Parcellate to 392-D per gene.
3. **Normalization:** Cortex (0:360) and subcortex (360:392) z-scored separately. Reason: gene expression scales and sampling density differ by compartment; global z would distort subcortical pattern.
4. Save `term_maps.npz`, `term_vocab.pkl` (gene symbols as labels).

**Command:**
```bash
python neurolab/scripts/build_abagen_cache.py --output-dir neurolab/data/abagen_cache --all-genes
```

**Result:** ~15K gene maps (full AHBA).

### 7.2 Tiered selection at merge (when `--max-abagen-terms N`)

When merging, we cap gene maps to avoid overweighting. Selection is **tiered** (literature-aligned): see [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md). **Recommended total: ~500** (not 2000). The full build uses `--max-abagen-terms 500`.

**Tier 1 — Pharmacological anchors (always include):**  
All genes in `receptor_kb.load_receptor_genes()` (~250) that appear in the abagen cache. These are the bridge between PDSP pharmacology and the Generalizer's semantic space (e.g. DRD2, SLC6A4, GABRA1).

**Tier 2 — WGCNA-style cluster medoids:**  
From **non-receptor** genes, compute gene–gene **spatial correlation** (across parcels). Hierarchical clustering (average linkage on distance = 1 − correlation) is cut to obtain **K clusters** (default K = 32, `--abagen-n-clusters`). For each cluster, the **medoid** (gene with highest mean correlation to others in the cluster) is selected. These ~32 genes represent the main independent spatial patterns (e.g. cortical hierarchy, striatal, glial).

**Tier 3 — Residual-variance selected genes:**  
- **`residual_variance`** (default): Rank genes (excluding Tier 1/2) by variance of residual after regressing out **top 3–5 PCs**; take top N − (Tier1 + Tier2). Fulcher-aligned. Typical Tier 3 count: **~200** when N = 500.  
- **`medoids`**: Cluster remaining genes; one medoid per cluster.

**Summary:** Tier 1 ≈ 250 receptor genes + Tier 2 ≈ 30 WGCNA hub genes + Tier 3 ≈ 200 residual-variance genes ≈ **~500 total**. This avoids abagen dominating; source-weighted sampling (abagen ~10%) and loss weight 0.4 then balance effectively. The target is ~3–5% of training set; if other sources change significantly (e.g. adding neurovault_pharma adds ~632 maps), adjust proportionally.

**Gradient maps (recommended default):** With **`--abagen-add-gradient-pcs K`** (e.g. **5**, recommended), the first K PCs are added as **separate synthetic terms** `gene_expression_gradient_PC1`, … and saved as `abagen_gradient_components.npy` for PET residual-correlation evaluation. **Use this by default** for merged_sources; the full build passes `--abagen-add-gradient-pcs 5`.

**Implementation:** `build_expanded_term_maps.py` (merge step); options `--max-abagen-terms` (recommend 500), `--abagen-n-clusters` (default 32), `--abagen-add-gradient-pcs` (recommend 5), `--abagen-tier3-method` (default residual_variance).

---

## 8. PDSP Ki cache (compound→brain pharmacological pathway)

**Source:** NIMH Psychoactive Drug Screening Program (PDSP) Ki database — drug × receptor binding affinities. Public domain.

**Purpose:** Compound→brain spatial maps for drug inference. Drug Ki profile → weight receptor gene expression → project through gene PCA → 392-D brain map. **Required** for the full pharmacological pipeline (drug queries, enrichment, similar-drug lookup).

### 8.1 Prerequisites

- **Gene PCA Phase 1–2:** `expression_scaled.npy`, `gene_names.json`, `gene_loadings_full.npy`, `pc_scores_full.npy` in `neurolab/data/gene_pca/`. Run `run_gene_pca_phase1.py` and `run_gene_pca_phase2.py` (or `run_gene_pca_pipeline.py --skip-phase3 --skip-phase4`).
- **PDSP Ki CSV:** `neurolab/data/pdsp_ki/KiDatabase.csv` from `download_pdsp_ki.py`. If direct download fails, manual download from pdsp.unc.edu.

### 8.2 PDSP preprocessing (Ki → affinity matrix)

**Script:** `process_pdsp_for_neurolab.py` (optional intermediate step; `build_pdsp_cache.py` can also parse Ki directly)

**Method:**
1. Parse Ki values from CSV: handle `>`, `<`, `~`, ranges (geometric mean); convert to **pKi** = −log₁₀(Ki in M).
2. Filter to human species (default; `--species human`).
3. Map PDSP receptor names → HUGO gene symbols via `RECEPTOR_TO_GENE` dictionary (~80+ mappings covering major targets: 5-HT1A→HTR1A, SERT→SLC6A4, D2→DRD2, etc.). Unmapped receptor names are logged and skipped.
4. For each (compound, receptor) pair with multiple Ki values: take geometric mean.
5. Filter compounds with ≥ `--min-ki-per-compound` (default 3) receptor measurements.

**Expected output:** ~3,000–5,000 compounds × ~60–80 gene symbols; matrix density ~5–15%. Top compounds (e.g. haloperidol, clozapine, risperidone) have 30+ receptors profiled.

**Validation:** Check unmapped receptors in output; extend `RECEPTOR_TO_GENE` if important targets are missing.

**If automated download fails:** Download manually from https://pdspdb.unc.edu/databases/kiDownload/ ("Download Whole Database" button) or use the Selenium fallback script (`download_pdsp_ki_selenium.py`).

### 8.3 Build

**Script:** `build_pdsp_cache.py`

**Method:**
1. Load PDSP CSV; detect columns: Ligand Name, Receptor, Ki (nM).
2. Filter to human species when available.
3. For each compound: build binding profile (receptor → 1/Ki, normalized).
4. Map PDSP receptor names → gene symbols via `RECEPTOR_TO_GENE`.
5. Project each drug profile through gene PCA: `gene_loadings @ weight_vector` → PC coords; `pc_scores @ pc_coords` → 392-D spatial map.
6. Save `pdsp_profiles.npz`, `pdsp_pc_projections.npz`, `pdsp_pc_coordinates.npy`, `compound_names.json`, `metadata.json`.

**Commands:**
```bash
python neurolab/scripts/download_pdsp_ki.py --output-dir neurolab/data/pdsp_ki
python neurolab/scripts/run_gene_pca_phase1.py --output-dir neurolab/data/gene_pca
python neurolab/scripts/run_gene_pca_phase2.py --output-dir neurolab/data/gene_pca
python neurolab/scripts/build_pdsp_cache.py --output-dir neurolab/data/pdsp_cache --gene-pca-dir neurolab/data/gene_pca --pdsp-csv neurolab/data/pdsp_ki/KiDatabase.csv
```

**Or:** `run_full_cache_build.py` includes PDSP by default (runs gene PCA Phase 1–2 if needed, download, build). Use `--skip-pdsp` to bypass.

**Result:** `pdsp_cache/pdsp_pc_projections.npz` — (n_compounds, 392) spatial maps; used at inference for drug queries.

---

## 9. Merge step: combining all sources

### 9.1 Script and purpose

**Script:** `build_expanded_term_maps.py` (used for **merged_sources** with `--no-ontology` and all cache dirs).

**Purpose:** Combine all per-source caches into one (term, map) set; no re-normalization; record sources, sample weights, map types; optionally add gradient PCs and PET residual maps.

### 9.2 Order of merging (matches `build_expanded_term_maps.py`)

1. **Base cache** (unified): term_maps + term_vocab; source = `direct`.
2. **Ontology expansion** (skipped with `--no-ontology`).
3. **Neuromaps:** add maps from `neuromaps_cache`; source = `neuromaps`.
4. **NeuroVault:** add maps from `neurovault_cache`; source = `neurovault`. If NeuroVault cache has `term_sample_weights.pkl`, those **per-term** weights are used.
5. **NeuroVault pharma** (if `--neurovault-pharma-cache-dir`): source = `neurovault_pharma`.
6. **Pharma NeuroSynth:** add maps from `pharma_neurosynth_cache`; source = `pharma_neurosynth`.
7. **Receptor** (CSV path, if `--receptor-path`): source = `receptor`.
8. **ENIGMA:** source = `enigma`.
9. **Abagen:** with tiered selection when `--max-abagen-terms N`; source = `abagen`. Add gradient PC maps when `--abagen-add-gradient-pcs` > 0; write `abagen_gradient_components.npy`.
10. **Receptor reference** (if `--receptor-reference-cache-dir`): source = `reference`.
11. **PET/receptor residual maps** (if `--add-pet-residuals` and gradient components available): For each neuromaps and receptor map, compute residual after projecting out `abagen_gradient_components`. **Re-z-score** residuals (cortex/subcortex separate) to restore comparable variance. Add as additional terms with `_residual` suffix (e.g. `5HT2A_PET_residual`). Source = `neuromaps_residual` or `receptor_residual`; sample weight = 0.6; map_type = `pet_receptor`. Purpose: train pharmacologically specific spatial patterns beyond dominant transcriptional gradients.

### 9.3 Deduplication and term collision policy

Terms are deduplicated by **normalized label** (lowercase, strip, collapse whitespace). If a term appears in multiple sources, the **first occurrence wins** (order = merge order in §9.2).

**Collision risk for pharmacological terms:** A term like "dopamine" may exist in `direct` (NeuroQuery meta-analytic map of studies mentioning dopamine), `pharma_neurosynth` (MKDA of dopamine studies), and `neuromaps` (PET dopamine receptor density). These are fundamentally different maps. With first-occurrence-wins, the NeuroQuery version is kept and the PET map is silently dropped.

**Mitigation:** Neuromaps and receptor maps typically have specific labels (e.g. `D2_beliveau2017`, `5HT2A_savli2012`) that don't collide with NeuroQuery terms. Pharma NeuroSynth terms (`dopamine`, `serotonin`) may collide with direct/NeuroSynth terms. When this happens, the NeuroQuery map is preferred because it has broader study coverage. The pharma-specific signal is captured by NeuroVault pharma collections (actual drug challenge studies) rather than pharma NeuroSynth (text-mined meta-analysis of studies mentioning the drug).

**To verify:** After a merge, inspect dropped terms:
```bash
python -c "
import pickle
from pathlib import Path
p = Path('neurolab/data/merged_sources')
with open(p / 'term_vocab.pkl','rb') as f: vocab = set(pickle.load(f))
# Check if any neuromaps labels are missing
# (load neuromaps cache vocab and diff)
"
```

### 9.4 What the merge step does **not** do

- **No re-normalization** of maps (each source already normalized in its builder).
- No change to parcel dimension (all maps already 392-D).

### 9.5 Default loss weights by source

Used when building `term_sample_weights` for terms that don't have a per-term weight (e.g. from NeuroVault's pkl):

| Source | Default weight | Note |
|--------|----------------|------|
| direct | 1.0 | |
| neurovault | 0.8 | |
| neuromaps | **1.0** | ~40 PET maps; scarce, high-value pharmacological ground truth. With only ~40 maps and ~5% sampling, 1.0 won't dominate. |
| receptor | **1.0** | Same rationale as neuromaps. |
| neuromaps_residual / receptor_residual | 0.6 | Residual (specificity) terms. |
| enigma | 0.5 | |
| abagen | 0.4 | ~500 gene maps; 0.4 + source-weighted sampling (~10%) keeps effective contribution balanced. |
| pharma_neurosynth / neurovault_pharma | 1.2 | |

### 9.6 Map types

Each term is assigned a **map type** for the type-conditioned MLP:
- `fmri_activation`: NeuroQuery, NeuroSynth, Pharma NeuroSynth, NeuroVault, ontology.
- `structural`: ENIGMA.
- `pet_receptor`: neuromaps, receptor, abagen, neuromaps_residual, receptor_residual, reference.

### 9.7 Merge command (full training set)

```bash
python neurolab/scripts/build_expanded_term_maps.py \
  --cache-dir neurolab/data/unified_cache \
  --output-dir neurolab/data/merged_sources \
  --no-ontology --save-term-sources \
  --neurovault-cache-dir neurolab/data/neurovault_cache \
  --neurovault-pharma-cache-dir neurolab/data/neurovault_pharma_cache \
  --neuromaps-cache-dir neurolab/data/neuromaps_cache \
  --enigma-cache-dir neurolab/data/enigma_cache \
  --abagen-cache-dir neurolab/data/abagen_cache \
  --max-abagen-terms 500 \
  --abagen-add-gradient-pcs 5 \
  --add-pet-residuals \
  --pharma-neurosynth-cache-dir neurolab/data/pharma_neurosynth_cache
```

---

## 10. Output format

**Directory:** `neurolab/data/merged_sources/`

| File | Format | Purpose |
|------|--------|---------|
| `term_maps.npz` | `term_maps`: (N, 392) float64 | One 392-D map per term |
| `term_vocab.pkl` | List of N strings | Text labels for each term |
| `term_sources.pkl` | List of N strings | Source label per term (direct, neurovault, abagen, neuromaps, enigma, pharma_neurosynth, neuromaps_residual, receptor_residual, …) |
| `term_sample_weights.pkl` | List of N floats | Per-term loss weight (used in weighted MSE during training) |
| `term_map_types.pkl` | List of N strings | Map type per term: `fmri_activation`, `structural`, `pet_receptor` |
| `abagen_gradient_components.npy` | (K, 392) float64 | Gradient PCs when `--abagen-add-gradient-pcs` used; for PET residual-correlation evaluation. Shape follows sklearn convention: (n_components, n_features). |

**Typical size:** ~14K terms × 392 parcels (exact count depends on deduplication and which caches are included).

---

## 11. Verification

**Script:** `verify_full_cache_pipeline.py`

**Checks:**
- Atlas exists and has 360 cortical + 32 subcortical labels.
- All critical caches (decoder, neurosynth, unified, merged_sources, neuromaps, enigma, abagen) exist and have 392 parcels.
- NeuroVault cache (optional) exists and has 392 parcels.
- Build scripts use resampling and masker correctly.

**Command:**
```bash
python neurolab/scripts/verify_full_cache_pipeline.py
```

**Script:** `verify_parcellation_and_map_types.py` — confirms parcellation and map types across caches.

---

## 12. One-command build

For a full build from scratch:

```bash
python neurolab/scripts/run_full_cache_build.py
```

**Order (matches `run_full_cache_build.py`):**
1. Atlas (`combined_atlas_392.nii.gz`)
2. Decoder cache (NeuroQuery)
3. NeuroSynth cache
4. Merge NQ+NS → unified_cache
5. NeuroVault (if `neurovault_curated_data/` exists: build with `--average-subject-level`; else use bulk or skip)
6. Neuromaps cache
7. ENIGMA cache
8. Pharma NeuroSynth (curated JSON, `--pharma-terms-key all_terms_sorted`)
9. FC cache (optional, `--skip-fc` to skip)
10. Abagen cache
11. PDSP Ki (gene PCA Phase 1–2 if needed, download, build_pdsp_cache; `--skip-pdsp` to bypass)
12. Merge → `merged_sources/` with `--no-ontology`, `--save-term-sources`, `--abagen-add-gradient-pcs 5`, `--add-pet-residuals`, and all available cache dirs (neuromaps, neurovault, **neurovault_pharma**, pharma_neurosynth, enigma, abagen)

If curated NeuroVault data is not present:

```bash
python neurolab/scripts/run_full_cache_build.py --download-neurovault-curated
```

---

## 13. References

| Topic | Document |
|-------|----------|
| Normalization rationale | [term_map_normalization_strategy.md](term_map_normalization_strategy.md) |
| NeuroVault averaging and collections | [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md) |
| Abagen tiered selection and gradients | [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md) |
| Preprocessing, weighting, training | [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) |
| Quick build commands | [BUILD_TRAINING_DATASET.md](BUILD_TRAINING_DATASET.md) |
| Data preparation checklist | [DATA_PREPARATION_STATUS.md](DATA_PREPARATION_STATUS.md) |
| NeuroVault acquisition | [NeuroVault acquisition guide for brain activation prediction training.md](NeuroVault%20acquisition%20guide%20for%20brain%20activation%20prediction%20training.md) |
