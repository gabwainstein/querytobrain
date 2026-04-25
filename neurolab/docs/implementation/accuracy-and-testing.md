# Accuracy and Testing

This document defines **how we measure accuracy** and **the process to run tests** for NeuroLab pipelines. Every pipeline that reports performance should align with these definitions and be reproducible using the steps below.

**Pipeline:** Atlas = Glasser 360 + Tian S2 = **392 parcels**. All training and evaluation use 392-D parcellated maps. Legacy 400-D (Schaefer) caches are not used for training.

---

## 1. How we define accuracy

Accuracy is **context-dependent** by component. We use the following definitions consistently.

### 1.1 Text-to-brain embedding (expandable term space)

**What we measure:** How well the model predicts a parcellated brain map (**392-D**, Glasser+Tian) for a given text.

- **Primary metric:** **Mean correlation** — For each (term, ground-truth map) pair, we compute the **Pearson correlation** between the predicted 392-D vector and the ground-truth 392-D vector; we then average over terms. Reported as *Train mean correlation*, *Val mean correlation*, *Test mean correlation*.
- **Secondary metric:** **MSE** — Mean squared error between predicted and ground-truth maps (reported as Val MSE, Test MSE).
- **Per-source and per-map-type:** When the cache is built with `--save-term-sources` and has `term_sources.pkl` and `term_map_types.pkl`, report mean correlation **per source** (direct, neurovault, abagen, neuromaps, enigma, neuromaps_residual, pharma_neurosynth, neurovault_pharma) and **per map type** (`fmri_activation`, `structural`, `pet_receptor`) so that performance on fMRI vs PET vs structural targets is visible.

**Interpretation:**

- **Test mean correlation** = unbiased estimate of generalization to **unseen terms** (held-out test set). This is the main number we use to compare models (e.g. different encoders, PCA sizes, regularization).
- **Train (or Train+val) mean correlation** = fit on the data the model was trained on. Used together with test to assess **overfitting**: the **gap** (train − test) should be small; a large gap indicates overfitting.
- All correlations and MSE are computed in **392-D parcel space** (after inverse PCA if PCA was used).

**Scope:** Applies to the model produced by `train_text_to_brain_embedding.py` and used by `TextToBrainEmbedding`, `query.py --use-embedding-model`, and `predict_map.py`.

**Residual correlation (PET / receptor evaluation):** When evaluating predicted maps against **PET receptor maps** (e.g. neuromaps, receptor), **raw** Pearson correlation can be inflated by shared spatial autocorrelation (Fulcher et al. document this for GCEA; the same structure applies to gene–PET associations). Use **residual correlation**: regress the first K gene-expression gradient PCs (from `abagen_gradient_components.npy` in the merged cache) from both the predicted and target maps, then correlate the residuals. Load gradient components from the merged cache; call `neurolab.evaluation_utils.residual_correlation(pred, target, gradient_components)`. See [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md).

**Generalization and encoder choice:** **Recommended for best test:** `NeuML/pubmedbert-base-embeddings` (PubMedBERT). On the **current 392-parcel merged_sources pipeline** (14K terms, multi-source, type-conditioned), expected baseline metrics are to be filled after the first full training run (see §4). On the **legacy 400-D decoder-only** setup, PubMedBERT gave test ~0.64; TF-IDF ~0.55. Compare encoders: `python neurolab/scripts/run_encoder_comparison.py` (when run on the same cache and split).

- **Sample weights:** When training on merged_sources (or any cache built with `--save-term-sources`), the trainer uses `term_sample_weights.pkl` and source-weighted sampling (e.g. direct 1.0, neurovault 0.8, neuromaps 1.0, abagen 0.4, pharma 1.2). See [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md).
- **Gene head:** When the cache includes `gene_pca.pkl`, `gene_loadings.npz`, and `abagen_term_indices.pkl`, the trainer can use a gene head (predict PC loadings, inverse to 392-D); activation is automatic when those files are present.

**Why did test used to be ~0.55 (legacy)?** Run: `python neurolab/scripts/diagnose_generalization_ceiling.py --cache-dir neurolab/data/decoder_cache`. It reports **(1) Best-map ceiling** and **(2) Nearest-neighbor (text) baseline**. A better text encoder (e.g. PubMedBERT) improves alignment.

- **More / better data:** Train on **merged_sources** (unified + neurovault + neuromaps + enigma + abagen + pharma, etc.) for full coverage. Build with `build_expanded_term_maps.py` as in [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) §9.7.
- **Target quality:** Ground-truth maps come from NeuroQuery/NeuroSynth/NeuroVault/neuromaps/abagen etc.; improving source maps and merge choices (e.g. including neurovault_pharma) improves ceiling.

**Ontology distance and brain maps:** When using ontology-expanded caches, relation-type weights (synonym, child, parent) determine blending; see `ontology_expansion.py`. Run `ontology_brain_correlation.py` to check whether ontology proximity correlates with brain-map similarity.

### 1.2 Cognitive decoder (term maps cache)

**What we measure:** How well an input parcellated map matches cached term maps.

- **Metric:** For a given map, we correlate it with each cached term map and return the top-N terms by correlation (e.g. `r` per term). No single “accuracy” number; quality is assessed by whether the top terms are semantically plausible for the map.
- **Verification:** `verify_decoder.py` loads the cache and runs decode on sample maps to confirm the pipeline runs and returns rankings.

**Scope:** `CognitiveDecoder`, `build_term_maps_cache.py`, `verify_decoder.py`. Use a **392-parcel** cache (e.g. decoder_cache or merged_sources built for 392 parcels).

### 1.3 Biological / receptor enrichment

**What we measure:** Spatial correlation between a parcellated map and receptor (or other biological) density maps.

- **Metric:** **Pearson r** per receptor (or annotation) between the input map and each receptor map. For **pharmacologically specific** agreement with PET receptor maps, use **residual correlation** (regress gradient PCs from both, then correlate residuals) as in §1.1.
- **Caveat:** If receptor data is **placeholder** (random), r values are uninterpretable. Use real data (neuromaps cache, abagen, or receptor reference) for meaningful numbers.

**Scope:** `ReceptorEnrichment`, `UnifiedEnrichment` biological layer, neuromaps cache (392 parcels).

### 1.4 Scope guardrail (in-scope vs out-of-scope)

**What we measure:** Whether a query is classified as in-scope (relevant for brain-map prediction) or out-of-scope.

- **Metric:** **Score** = max cosine similarity of the query embedding to the set of training-term embeddings. **in_scope** = score ≥ threshold (e.g. 0.25). No “accuracy” in the sense of correct/incorrect labels unless we have a labelled set of in/out-of-scope queries (optional future work).

**Scope:** `ScopeGuard`, `query.py --guardrail=on`.

### 1.5 NeuroQuery vs NeuroSynth (sources of term maps)

Our **decoder cache** (and thus the text-to-brain training data) can be built from **NeuroQuery** or **NeuroSynth**. Both provide “term → brain map,” but the methods and accuracy differ.

| | NeuroSynth | NeuroQuery |
|---|------------|------------|
| **Input** | Article **abstracts** | **Full-text** papers |
| **Method** | Coordinate-based meta-analysis (e.g. ALE, MKDA) | **Predictive model**: supervised regression from text to brain maps. |
| **Vocabulary** | ~1,000–1,300 terms. | ~7,500 terms (neuroscience vocabulary; includes rare terms). |
| **Rare / difficult terms** | **Weak** when term appears in few studies. | **Strong**: uses related terms to predict maps. |

**In our pipeline:** Default decoder cache is built from **NeuroQuery** (`build_term_maps_cache.py`). Use `--max-terms 0` for full vocabulary (~7.5K terms); if decoder cache has &lt;6K terms, rebuild with `--max-terms 0` (see [TRAINING_READINESS_AUDIT](Claude audits and fixes/TRAINING_READINESS_AUDIT.md) §3). Training uses **merged_sources** (unified + neurovault + neuromaps + enigma + abagen + pharma, etc.) for best coverage.

---

## 2. Train / val / test and evaluation protocol

For **any model that generalizes to new terms** (text-to-brain embedding), we use a fixed evaluation protocol so that numbers are comparable.

### 2.1 Data split

- **Source:** Merged training cache (`merged_sources/` or other cache with `term_maps.npz`, `term_vocab.pkl`, and optionally `term_sources.pkl`, `term_sample_weights.pkl`, `term_map_types.pkl`). All maps **392-D**.
- **Split:** Terms are partitioned into **train**, **val**, and **test** (e.g. 80% / 10% / 10%). The split is **deterministic** (fixed seed; `--seed 42` by default). The script saves **split_info.pkl** so you can see exactly which terms were held out.
- **Usage:** Train for fitting; val for monitoring (and early stopping when `--early-stopping`); **test** held out and used only for final reporting.

### 2.2 Optional: final model on train+val

- If `--final-retrain-on-train-and-val` is set, we **retrain** on **train ∪ val**, then save that model. **Test** is still used only once for final **Test mean correlation** (and per-source / per-map-type if implemented).

### 2.3 Reporting

- Every training run prints: **Train** (and optionally Train+val) mean correlation, **Val** mean correlation, **Test** mean correlation, and MSE. Report **per-source** and **per-map-type** test correlation when term_sources and term_map_types are available.
- For PET/receptor test terms, report **residual correlation** (with `abagen_gradient_components.npy`) in addition to raw correlation when relevant.

---

## 3. Process to run tests (reproducible)

### 3.1 Build merged_sources (training cache)

See [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) §9.7 and [TRAINING_READINESS_AUDIT](Claude audits and fixes/TRAINING_READINESS_AUDIT.md) §9.1. Example:

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
  --abagen-add-gradient-pcs 3 \
  --add-pet-residuals \
  --pharma-neurosynth-cache-dir neurolab/data/pharma_neurosynth_cache
```

Use `--max-terms 0` when building decoder cache so vocabulary is ~7.5K (see audit §3).

### 3.2 Train text-to-brain model

```bash
python neurolab/scripts/train_text_to_brain_embedding.py \
  --cache-dir neurolab/data/merged_sources \
  --output-dir neurolab/data/embedding_model \
  --encoder sentence-transformers \
  --encoder-model NeuML/pubmedbert-base-embeddings \
  --max-terms 0 \
  --epochs 50 \
  --dropout 0.2 \
  --weight-decay 1e-5 \
  --early-stopping
```

- **What you get:** Train/val/test mean correlation (and MSE); saved model and guardrail embeddings. Accuracy is in **392-D**; report per-source and per-map-type when cache has term_sources and term_map_types.

### 3.3 Verify pipelines (smoke tests)

- **Decoder:** `python neurolab/scripts/verify_decoder.py` — loads cache and runs decode.
- **Unified enrichment:** `python neurolab/scripts/verify_unified.py`
- **Training load:** `python neurolab/scripts/check_training_readiness.py --require-expanded` (and `verify_full_cache_pipeline.py`, `verify_parcellation_and_map_types.py` as in the audit).

### 3.4 Query with guardrail (optional)

```bash
python neurolab/scripts/query.py "dopamine" --use-embedding-model neurolab/data/embedding_model --guardrail on
```

---

## 4. Expected baseline metrics (392-parcel merged_sources pipeline)

After the **first full training run** on the current pipeline (merged_sources, 392 parcels, multi-source, source-weighted sampling, PubMedBERT), fill in:

- **Overall test mean correlation:** (e.g. ___)
- **Test mean correlation per source:** direct ___, neurovault ___, abagen ___, neuromaps ___, enigma ___, neuromaps_residual ___, pharma_neurosynth ___, neurovault_pharma ___ (if present).
- **Test mean correlation per map type:** fmri_activation ___, structural ___, pet_receptor ___.
- **Residual correlation (PET/receptor test terms):** ___ (when evaluated with `abagen_gradient_components.npy`).

These baselines allow comparison when changing encoders, regularization, or merge composition.

---

## 5. Implicit rule for new pipelines

Whenever we add a **new pipeline or model** that reports performance:

1. **Define accuracy** in the same way we do here: metric, unit (per-term, per-parcel), and what “good” means.
2. **Define the evaluation protocol:** split (train/val/test), no test leakage.
3. **Document the exact commands** to reproduce the numbers, including cache and atlas (392 parcels).
4. **Cross-link** this doc from the pipeline’s README or implementation slice.

---

## 6. References

- [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) — merge command, sources, weights.
- [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) — normalization, weighting, training.
- [abagen_tiered_gene_selection.md](abagen_tiered_gene_selection.md) — residual correlation, gradient PCs.
- [TRAINING_READINESS_AUDIT](Claude audits and fixes/TRAINING_READINESS_AUDIT.md) — verification commands and merge/training two-step.
