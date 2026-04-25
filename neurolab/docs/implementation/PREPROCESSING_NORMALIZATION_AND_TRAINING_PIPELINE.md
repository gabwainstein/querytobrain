# Preprocessing, Normalization, Averaging, Grouping, Weighting, and Training

This document is the **single reference** for the full pipeline: from raw data through preprocessing, normalization, averaging/grouping, weighting, merge, and what happens **during training**. Run all commands from the repository root (the directory containing `neurolab/`).

---

## 1. Plan overview

| Phase | What | Output |
|-------|------|--------|
| **Atlas** | One parcellation for the entire pipeline | `combined_atlas_392.nii.gz` (Glasser 360 + Tian S2 = 392 parcels) |
| **Per-source caches** | Each data source: resample → parcellate → normalize | `decoder_cache/`, `neurosynth_cache/`, `neurovault_cache/`, `abagen_cache/`, `neuromaps_cache/`, `enigma_cache/`, etc. |
| **Merge** | Combine caches into one (term, map) set; no re-normalization | `merged_sources/` or expanded cache dir: `term_maps.npz`, `term_vocab.pkl`, `term_sources.pkl`, `term_sample_weights.pkl` |
| **Training** | Train text encoder + MLP on merged set with source-weighted sampling and per-term loss weights | `embedding_model/` (checkpoint, split_info, etc.) |

**Principle:** Normalization and grouping are done **per source at build time**. The merge step **does not** re-normalize; it only concatenates terms and maps and records source/sample weights. Training then uses **source-weighted batch sampling** and **per-sample loss weights** so no single source dominates.

---

## 2. Parcellation and resampling (all sources)

- **Atlas:** Glasser 360 (cortex) + Tian S2 32 (subcortex) = **392 parcels**. Defined in `neurolab/parcellation.py`; built with `build_combined_atlas.py --atlas glasser+tian`.
- **Resampling:** Every NIfTI (NeuroVault, neuromaps, etc.) is **resampled to the pipeline atlas** before parcellation so alignment is correct regardless of source template/MNI/origin. `parcellation.resample_to_atlas(img)` handles this.
- **Parcellation:** Each image is turned into a 392-D vector via `NiftiLabelsMasker` (or equivalent) using the combined atlas. All caches store maps as **392-D**; legacy 400/414 is not used in the current pipeline.

---

## 3. Preprocessing and normalization **by source** (before merge)

Each cache builder is responsible for: (1) loading raw data, (2) resampling to atlas if needed, (3) parcellating to 392-D, (4) **normalizing** in a way that matches the data type. The merge step does **not** change these values.

### 3.1 fMRI activation (NeuroQuery, NeuroSynth, Pharma NeuroSynth, NeuroVault)

| Step | What | Where |
|------|------|--------|
| Parcellation | 392-D per term | Each builder |
| Normalization | **Global z-score** (across all 392 parcels) | `zscore_maps(maps, axis=1)` or equivalent |
| Reason | fMRI effect sizes (t/z) are comparable across cortex and subcortex; cross-compartment pattern (e.g. high striatum, low cortex) is meaningful and must be preserved. |

- **NeuroQuery:** `build_term_maps_cache.py` — decoder outputs; global z per map.
- **NeuroSynth:** `build_neurosynth_cache.py` — meta-analysis maps; global z.
- **Pharma NeuroSynth:** `build_pharma_neurosynth_cache.py` — global z.
- **NeuroVault:** `build_neurovault_cache.py` — after grouping/QC; global z (see §4 for NeuroVault-specific steps).

### 3.2 Gene expression (abagen)

| Step | What | Where |
|------|------|--------|
| Parcellation | 392-D per gene (abagen pipeline) | `build_abagen_cache.py` |
| Normalization | **Cortex (0:360) and subcortex (360:392) z-scored separately** | `build_abagen_cache.py` (cortex/subcortex separate) |
| Reason | Gene expression scales and sampling density differ by compartment; global z would let cortex dominate and distort subcortical pattern. |

At **merge** time (`build_expanded_term_maps.py`), when abagen is included with `--max-abagen-terms N`:
- **Gradient PCs** (optional): first K PCs of the full abagen matrix added as synthetic terms `gene_expression_gradient_PC1` …; saved as `abagen_gradient_components.npy` for PET residual-correlation evaluation.
- **Tiered selection:** Tier 1 = receptor genes (~250), Tier 2 = WGCNA-style cluster medoids (~32), Tier 3 = residual-variance (default) or cluster medoids to fill up to N. See [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md).

### 3.3 PET / receptor / structural (neuromaps, receptor reference, ENIGMA)

| Step | What | Where |
|------|------|--------|
| Normalization | **Cortex and subcortex z-scored separately** | Each builder |
| Reason | PET binding and structural measures have different absolute scales by tissue; same as abagen. |

- **neuromaps:** `build_neuromaps_cache.py` — cortex/subcortex separate.
- **receptor reference:** `build_receptor_reference_cache.py` — cortex/subcortex separate.
- **ENIGMA:** `build_enigma_cache.py` — cortical thickness / subcortical volume; cortex/subcortex separate.

**PET residual maps (at merge):** For each **neuromaps** PET map (and each **receptor** map when receptor is merged), the merge step computes: **residual = map − projection of map onto the first K abagen gradient PCs** (from `--abagen-add-gradient-pcs K`). Residuals are **re-z-scored (cortex and subcortex separately)** so their scale matches other training targets (the projection step reduces variance; re-z-scoring avoids artificially low MSE for these terms). Then added as **separate** terms with suffix `_residual` (e.g. `5HT2A_PET_residual`) and source = `neuromaps_residual` or `receptor_residual`. Purpose: train the model to predict **pharmacologically specific** spatial patterns beyond dominant transcriptional gradients. Sample weight = 0.6. Enable with `--add-pet-residuals`; requires abagen merged with `--abagen-add-gradient-pcs` &gt; 0.

### 3.4 Summary table (normalization by source)

| Source | Script | Normalization |
|--------|--------|---------------|
| NeuroQuery | `build_term_maps_cache.py` | Global z |
| NeuroSynth | `build_neurosynth_cache.py` | Global z |
| Pharma NeuroSynth | `build_pharma_neurosynth_cache.py` | Global z |
| NeuroVault | `build_neurovault_cache.py` | Global z |
| abagen | `build_abagen_cache.py` | Cortex/subcortex separate |
| neuromaps | `build_neuromaps_cache.py` | Cortex/subcortex separate |
| receptor / receptor_reference | respective builders | Cortex/subcortex separate |
| ENIGMA | `build_enigma_cache.py` | Cortex/subcortex separate |
| neuromaps_residual / receptor_residual | merge (from PET maps + gradient PCs) | Residual = map − gradient projection; then cortex/subcortex separate z-score |

See [term_map_normalization_strategy.md](term_map_normalization_strategy.md) for rationale. **Merge:** no re-normalization; `--zscore-renormalize` is deprecated and can distort fMRI patterns.

---

## 4. NeuroVault: averaging, grouping, QC, and sample weights

NeuroVault has two paths: **subject-level collections** (average first by contrast) and **group/meta-analytic collections** (use as-is). See [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md) for the full list.

### 4.1 Classification of collections

- **AVERAGE_FIRST:** Collections that are subject-level (many images per contrast). These are **averaged by contrast** within collection before producing one map per contrast. IDs include: 1952, 6618, 2138, 4343, 16284 (high priority), plus 426, 445, 507, 504, … (medium).
- **Use as-is:** Group-level or meta-analytic collections (one map per image or already consensus maps). No averaging; one term per image.

### 4.2 Preprocessing steps (in order)

1. **Load images** from curated data dir (`neurovault_curated_data/`): manifest + NIfTIs.
2. **Resample** each image to pipeline atlas (`resample_to_atlas`).
3. **Parcellate** to 392-D.
4. **Per collection:**
   - If collection **in AVERAGE_FIRST:**  
     - **QC** (optional): reject all-zero, high-NaN, extreme, or near-constant maps (`qc_filter`).  
     - **Group by contrast:** key = `contrast_definition` or `cognitive_paradigm_cogatlas_id` or `name` or path stem (`get_contrast_key`).  
     - **Average:** For each group with ≥ `min_subjects` (default 3, or 2 for heterogeneous collections), compute mean map. Optional **outlier removal:** drop maps with correlation to group mean &lt; 0.2, then recompute mean if enough remain.  
     - One **term** per contrast (label from key); one **map** per contrast; **sample weight** = 0.8 (subject-averaged).
   - If collection **not in AVERAGE_FIRST:**  
     - **Use as-is:** one term per image (label from `contrast_definition` or name).  
     - **QC** (optional): applied to full stack before writing cache.  
     - **Sample weight** = 2.0 if collection in `META_ANALYTIC_COLLECTIONS`, else 1.0 (group).
5. **Z-score:** Global z-score across 392 parcels for all NeuroVault maps (`zscore_maps(term_maps, axis=1)`).
6. **Write** `term_maps.npz`, `term_vocab.pkl`, `term_collection_ids.pkl`, **`term_sample_weights.pkl`** (meta 2×, group 1×, subject-averaged 0.8×).

**Recommended command (curated data):**
```bash
python neurolab/scripts/build_neurovault_cache.py \
  --data-dir neurolab/data/neurovault_curated_data \
  --output-dir neurolab/data/neurovault_cache \
  --average-subject-level
```

Optional: `--no-qc` to skip QC; `--no-zscore` to skip z-score (not recommended). Optional `--cluster-by-description` (non-average path): group by exact contrast_definition and average within group to get one map per unique description.

---

## 5. Merge step: what gets combined and how

**Script:** `build_expanded_term_maps.py` (used for **merged_sources** with `--no-ontology` and all cache dirs, or for expanded cache with ontology).

### 5.1 Order of merging (matches `build_expanded_term_maps.py`)

1. **Base cache** (decoder or unified): term_maps + term_vocab; source = `direct`.
2. **Ontology expansion** (optional; skipped with `--no-ontology` for merged_sources): add terms with derived maps (weighted average of related cache terms); source = `ontology`.
3. **Neuromaps:** add maps from `neuromaps_cache`; source = `neuromaps`.
4. **NeuroVault:** add maps from `neurovault_cache`; source = `neurovault`. If NeuroVault cache has `term_sample_weights.pkl`, those **per-term** weights are used.
5. **NeuroVault pharma** (if `--neurovault-pharma-cache-dir`): source = `neurovault_pharma`.
6. **Pharma NeuroSynth** (if `--pharma-neurosynth-cache-dir`): source = `pharma_neurosynth`.
7. **Receptor** (if `--receptor-path`): parcellated receptor maps; source = `receptor`.
8. **ENIGMA:** source = `enigma`.
9. **abagen:** with optional cap and tiered selection; source = `abagen`. Optionally add gradient PC maps; write `abagen_gradient_components.npy` when `--abagen-add-gradient-pcs` &gt; 0.
10. **Receptor reference** (if `--receptor-reference-cache-dir`): source = `reference`.
11. **PET/receptor residual maps** (if `--add-pet-residuals` and gradient components available): For each neuromaps and receptor map, compute residual after projecting out `abagen_gradient_components`. Add as additional terms with `_residual` suffix (e.g. `5HT2A_PET_residual`). Source = `neuromaps_residual` or `receptor_residual`; sample weight = 0.6; map_type = `pet_receptor`. Requires abagen to have been merged with `--abagen-add-gradient-pcs` &gt; 0.

### 5.2 What the merge step does **not** do

- **No re-normalization** of maps (each source already normalized in its builder).
- No change to parcel dimension (all maps already 392-D).

### 5.3 What the merge step writes

- `term_maps.npz`, `term_vocab.pkl`: one row per term, 392-D per map.
- **`term_sources.pkl`**: list of source labels (`direct`, `ontology`, `neurovault`, `abagen`, `enigma`, `neuromaps`, `receptor`, `reference`, etc.) for each term. Required for source-weighted sampling and for applying source-based default loss weights when per-term weights are missing.
- **`term_sample_weights.pkl`**: list of floats, one per term (loss weight). If a merged cache (e.g. NeuroVault) provides per-term weights, those are used for that subset; otherwise `SAMPLE_WEIGHT_BY_SOURCE[source]` is used. When present, the trainer uses these for **per-sample loss weighting**.
- **`term_map_types.pkl`**: one of `fmri_activation`, `structural`, `pet_receptor` per term (for type-conditioned MLP).
- **`abagen_gradient_components.npy`** (optional): when `--abagen-add-gradient-pcs K`; used for PET residual-correlation evaluation.

### 5.4 Default loss weights by source (merge)

Used when building `term_sample_weights` for terms that don't have a per-term weight (e.g. from NeuroVault's pkl). **neuromaps** and **receptor** are set to **0.8–1.0** (not 0.4): they are ~40 ground-truth PET receptor density maps and the highest-value pharmacological targets.

| Source | Default weight | Note |
|--------|----------------|------|
| direct | 1.0 | |
| neurovault | 0.8 | |
| ontology | 0.6 | |
| neuromaps | **1.0** | ~40 PET maps; high-value pharmacological ground truth; 1.0 so they are not underweighted (source-weighted sampling already limits batch share). |
| receptor | **1.0** | Same rationale as neuromaps. |
| neuromaps_residual | 0.6 | Gradient-regressed PET terms; pharmacologically specific signal. |
| receptor_residual | 0.6 | Same as neuromaps_residual. |
| enigma | 0.5 | |
| abagen | 0.4 | ~500 gene maps (with tiered selection); 0.4 + source-weighted sampling (~10%) keeps contribution balanced. |
| reference | 0.6 | |
| pharma_neurosynth / neurovault_pharma | 0.8–1.2 | |

---

## 6. Weighting: summary before training

After the merge you have:

1. **Per-term sample weights** (`term_sample_weights.pkl`): used in the **loss** (each sample's squared error is multiplied by this weight before summing). NeuroVault contributes meta 2×, group 1×, subject-averaged 0.8×; other sources get the defaults above.
2. **Source labels** (`term_sources.pkl`): used for **batch sampling** (see §7) so that each batch has a target mix of sources (e.g. abagen ~10%, direct+neurovault ~60%), avoiding gene-dominated gradients.

---

## 7. During training (train_text_to_brain_embedding.py)

### 7.1 Load and split

- Load merged cache: `term_maps.npz`, `term_vocab.pkl`, `term_sources.pkl`, `term_sample_weights.pkl`, `term_map_types.pkl`.
- **Text augmentation (not yet implemented):** Paraphrases or LLM-generated variants (same map for multiple text variants; sources and weights inherited) would improve generalization but are not currently implemented. When added, specify: which LLM/prompt, how many variants per term, and whether augmented terms are train-only or also val/test.
- **Train/val/test split:** Random split (seed fixed); typically 80/10/10 or as specified. Indices refer to the full term list.

### 7.2 Source-weighted batch sampling

- If `term_sources.pkl` exists and `--no-source-weighted-sampling` is **not** set:
  - For each source \(s\), target fraction = `SOURCE_SAMPLING_WEIGHTS[s]` (e.g. direct 0.30, neurovault 0.30, ontology 0.15, abagen 0.10, enigma 0.05, neuromaps 0.05, **neuromaps_residual 0.05**, **receptor_residual 0.03**, pharma 0.03+0.02, receptor/reference 0). Residual sources get small target fractions so PET/receptor presence in batches is not inflated.
  - Per-sample probability: \(P(i) \propto \texttt{SOURCE_SAMPLING_WEIGHTS}[s_i] / n_{s_i}\), so that over many batches the **proportion of samples from each source** matches the target (e.g. abagen ~10%).
  - Each **epoch**, training indices are drawn **with replacement** from the train set using these probabilities; then batches are formed from that permuted list.
- If disabled or no term_sources: standard random permutation (no replacement) each epoch.

### 7.3 Per-sample loss weights

- **Train** samples only: each sample \(i\) has weight `train_weights[i]` from `term_sample_weights` (or from source default when per-term weights are missing).
- Loss: for batch indices \(B\),  
  \(\mathcal{L} = \frac{\sum_{i \in B} w_i \, (y_i - \hat{y}_i)^2}{\sum_{i \in B} w_i}\)  
  (weighted MSE). So high-weight terms (e.g. meta-analytic NeuroVault) contribute more to the gradient. **Neuromaps and receptor** default to **0.8** (not 0.4) so the ~40 PET maps are not washed out despite low sampling fraction; abagen stays at 0.4 with ~10% batch share so gene maps don't dominate.

- **All training paths use these weights:** TF-IDF path passes `sample_weight=train_weights` to `MLPRegressor.fit()`; PyTorch path applies `train_weights` to both `loss_main` and `loss_gene` (gene head) via weighted MSE; sklearn fallback and retrain-on-train+val also pass `sample_weight`. No path ignores per-sample weights.

### 7.4 Model and targets

- **Encoder:** Text → embedding (e.g. sentence-transformers or TF-IDF). Embedding dimension is fixed by the encoder.
- **MLP head:** Embedding → 392-D (or PCA-reduced target dimension). **Type-conditioned:** input = concat(embedding, one-hot(map_type)); map_type = `fmri_activation` | `structural` | `pet_receptor` from `term_map_types.pkl`. So the model sees whether the target is fMRI, structural, or PET/gene and can learn different mappings.
- **Targets:** Usually the 392-D map (or PCA transform of it if `--pca-variance` / `--pca-components` is set). For **abagen** terms, if **gene PCA** is provided (`gene_pca.pkl`, `gene_loadings.npz`, `abagen_term_indices.pkl`), a **second head** (gene head) predicts PC loadings for those terms; loss for abagen is on loadings, and at inference the map is `gene_pca.inverse_transform(gene_head(embedding))`.

### 7.5 Training loop (per epoch)

1. Draw training indices **with replacement** using `sampling_probs` (source-weighted) or **without replacement** permutation if source-weighted sampling is off.
2. For each batch: forward (encoder → embedding → MLP/gene head), compute **weighted** MSE (and optional triad pairwise loss), backward, optimizer step.
3. Validation: predict on val set; compute correlation (and optionally PCA inverse if used). Early stopping on validation correlation if configured.

### 7.6 What is **not** done during training

- No further normalization of the 392-D targets (they are already normalized per source at cache build).
- No re-weighting of sources inside the loss beyond the per-sample weights and the batch composition (source-weighted sampling).

---

## 8. Step-by-step checklist (concise)

**Before training:**

1. Build atlas: `build_combined_atlas.py --atlas glasser+tian`.
2. Build per-source caches (each normalizes as in §3):
   - Decoder, NeuroSynth, unified (if needed).
   - NeuroVault: `build_neurovault_cache.py --data-dir ... --output-dir ... --average-subject-level`.
   - abagen, neuromaps, ENIGMA, receptor, pharma as needed.
3. Merge: `build_expanded_term_maps.py` with all cache dirs, `--save-term-sources`, and abagen options (e.g. `--max-abagen-terms 500 --abagen-add-gradient-pcs 3`). Output = merged cache dir (e.g. `merged_sources/`).
4. Confirm outputs: `term_maps.npz`, `term_vocab.pkl`, `term_sources.pkl`, `term_sample_weights.pkl`, `term_map_types.pkl` (and optionally `abagen_gradient_components.npy`).

**Training:**

1. Run `train_text_to_brain_embedding.py --cache-dir <merged_cache_dir>` (and encoder/epochs/batch_size as needed).
2. Source-weighted sampling is on by default (use `--no-source-weighted-sampling` to disable).
3. Per-sample weights come from `term_sample_weights.pkl` when present, else from source defaults.
4. For evaluation vs PET maps: use `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)` with `abagen_gradient_components.npy` to measure pharmacologically specific agreement (see [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md)).

---

## 9. References to other docs

- **Thorough methods documentation:** [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) — Step-by-step description of all build methods (parcellation, normalization, averaging, curation, merge).
- **Normalization rationale:** [term_map_normalization_strategy.md](term_map_normalization_strategy.md)
- **NeuroVault averaging and collections:** [neurovault_collections_averaging_guide.md](neurovault_collections_averaging_guide.md)
- **Abagen tiered selection and gradients:** [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md)
- **Full build order and commands:** [BUILD_MAPS_AND_TRAINING_PIPELINE.md](BUILD_MAPS_AND_TRAINING_PIPELINE.md)
- **Critical path caches:** [CRITICAL_PATH_CACHES_SPEC.md](CRITICAL_PATH_CACHES_SPEC.md)
