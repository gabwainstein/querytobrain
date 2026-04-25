# Minimal context for analyzing the text-to-brain pipeline

Give these files to Claude (or another agent) to understand the pipeline with minimal reading.

---

## 1. Minimal file set (in order)

| # | Path | Purpose |
|---|------|--------|
| 1 | **neurolab/docs/implementation/BUILD_MAPS_AND_TRAINING_PIPELINE.md** | Single source of truth: build order (ontologies → NQ → NS → merge → neuromaps → expand → train), file formats, risks, KG-regularized option (§14b). Read this first. |
| 2 | **neurolab/scripts/ontology_expansion.py** | Ontology layer: loads OBO/OWL, `expand_term(term, vocab)` → list of (cache_term, weight, relation_type). `RELATION_WEIGHTS`; `DIRECTION_SCALE` (parent 0.7, child 0.9) for blending. Direct map always wins. |
| 3 | **neurolab/scripts/build_expanded_term_maps.py** | Expands (term, map) set: base cache + forward expansion (ontology label → weighted avg of related cache maps) + reverse expansion (parent concepts get blend of children). Optional: min-cache-matches, min-pairwise-correlation, term_sources.pkl, neuromaps/receptor merge. |
| 4 | **neurolab/scripts/train_text_to_brain_embedding.py** | Training: load term_maps.npz + term_vocab.pkl (and optional term_sources.pkl for sample weights). Encode terms (TF-IDF / sentence-transformers / openai) → MLP head → 400-D (or PCA space). Loss = MSE; optional deeper MLP (--head-hidden2), PCA, early stopping. |
| 5 | **neurolab/scripts/build_all_maps.py** | Orchestrator: NQ cache → NS cache → merge → neuromaps cache → optional expand (with --save-term-sources). One entry point for “build everything.” |
| 6 | **neurolab/scripts/merge_neuroquery_neurosynth_cache.py** | Merges decoder_cache (NQ) + neurosynth_cache (NS) → unified_cache; --prefer neurosynth | neuroquery for conflicts. |
| 7 | **neurolab/scripts/ontology_brain_correlation.py** | Validation: mean brain-map correlation for ontology-related term pairs vs random pairs; --output-weights for data-driven weights. Run before relying on ontology expansion (need r > 0.3). |
| 8 | **neurolab/scripts/inspect_expanded_maps.py** | Visual sanity check: glass brains for expanded terms; quality flags (sparsity, peak, r vs global mean). Run before training on expanded cache. |

**Optional for “how do we use the model at inference”:**

| Path | Purpose |
|------|--------|
| **neurolab/enrichment/text_to_brain.py** | Loads trained model (config.pkl, head_weights.pt or head_mlp.pkl, pca.pkl); `embed(text)` → vector; `predict_map(text)` → 400-D parcellated map. |

---

## 2. Architecture in one paragraph

**Data flow:** (1) NeuroQuery and/or NeuroSynth provide (term, brain map) pairs parcellated to Schaefer 400 → saved as `term_maps.npz` + `term_vocab.pkl`. (2) Optional merge combines NQ+NS into one cognitive cache. (3) Ontology expansion adds more terms: for each ontology label not in the cache, we take a weighted average of related cache maps (weights from relation type); we also add “reverse” terms (e.g. parent “executive function” = blend of child maps). Quality filters: min-cache-matches ≥ 2, optional min-pairwise-correlation. (4) Optional neuromaps and receptor maps are merged into the same (label, map) set. (5) Training: text encoder (e.g. PubMedBERT) + MLP head; MSE(predicted_map, target_map); optional sample weights by source (direct > ontology > neuromaps/receptor). Output: any text → predicted 400-D map. **Key design choice:** Ontology expansion creates *hard* targets (one average map per term). The doc §14b describes an alternative: *KG-regularized contrastive loss* (soft constraint: “neighbors in KG should have similar predicted maps”) — validate with `ontology_brain_correlation.py` first (r > 0.3).

---

## 3. File formats (reference)

- **Cache:** `term_maps.npz` → `data["term_maps"]` shape (N, 400); `term_vocab.pkl` → list of N strings. Same format for decoder_cache, unified_cache, unified_cache_expanded.
- **Expanded only:** Optional `term_sources.pkl` → list of "direct" | "ontology" | "neuromaps" | "neurovault" | "receptor" for sample weighting.
- **Model:** `config.pkl` (encoder type, dim, head_hidden, head_hidden2, pca_components), `head_weights.pt` or `head_mlp.pkl`, `pca.pkl` if PCA used.

---

## 4. How neuromaps and NeuroVault are used

**Neuromaps**
- **Download:** `download_neuromaps_data.py` → `neurolab/data/neuromaps_data/` (raw NIfTIs).
- **Cache:** `build_neuromaps_cache.py` → `neuromaps_cache/` (`annotation_maps.npz`, `annotation_labels.pkl`).
- **Training:** When you run `build_all_maps.py --expand`, if `neuromaps_cache` exists, the expand step merges neuromaps labels + maps into `unified_cache_expanded` (labels like "5HT2A", "beliveau2017 cimbi36" become training examples; sample weight 0.4).
- **Query time:** `query.py` and `term_to_map.py` use neuromaps for **biological term→map**: if the user query matches a neuromaps label (e.g. "myelin", "5HT2A"), the map comes from neuromaps; otherwise cognitive cache/ontology. Enrichment can correlate a query map against all neuromaps maps.

**NeuroVault**
- **Fetch:** `download_neurovault_data.py` is what’s “still fetching” → `neurolab/data/neurovault_data/` (manifest.json + NIfTIs under `downloads/neurovault/`). Wait for it to finish.
- **Cache:** Once the fetch completes, run `build_neurovault_cache.py --data-dir neurolab/data/neurovault_data --output-dir neurolab/data/neurovault_cache` → `neurovault_cache/` (term_maps.npz, term_vocab.pkl; labels = contrast_definition / task / name per image).
- **Training:** To include NeuroVault in the training set, run the expand step with `--neurovault-cache-dir neurolab/data/neurovault_cache`. Those task-contrast (text, map) pairs are merged into the expanded set with source "neurovault" (sample weight 0.8). `build_all_maps.py` does **not** yet call NeuroVault automatically; add that step manually or pass `--neurovault-cache-dir` when you run `build_expanded_term_maps.py`.

---

## 5. Distance decay (hierarchy / embedding vs brain r)

**Observed:** Mean brain-map Pearson r decreases as distance (ontology hop or embedding distance) increases — clear distance-decay effect.

**Parametric form:** The relationship does **not** appear to be well described by a single simple form (e.g. exponential \(r = a e^{-d/k}\) or inverse \(r = a/(d+c)\)). Scripts still fit these for exploratory comparison and report R²; the **primary summary is the empirical binned mean trend** (mean/median over terms per distance bin). Do not interpret the fits as the “true” functional form; use the binned mean for interpretation and for any downstream use (e.g. expansion cutoff).

**Both regressors + interaction:** To test hierarchy (hop) and embedding distance together, run `regress_brain_r_on_hop_and_embedding.py`. It fits OLS at **pair level**: brain r ~ 1 + hop + emb_dist + hop×emb_dist. This gives main effects of hop and embedding distance and whether the effect of one depends on the other (interaction). Output: coefficients, p-values, R²; optional `--output-json` for downstream use.

**Using the triad in training:** The triad (embedding distance, KG hop, interaction) can improve MLP term→brain training by adding a **pairwise auxiliary loss**: target brain r from the regression, penalize when predicted pairwise r deviates. See **BUILD_MAPS_AND_TRAINING_PIPELINE.md §14c** for options (pairwise loss, contrastive, sample weighting) and implementation notes.

---

## 6. Commands to run (from repo root)

```bash
# Full build + expansion (then train)
python neurolab/scripts/build_all_maps.py --expand
python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/unified_cache_expanded --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings --pca-components 80 --early-stopping

# Validate ontology ↔ brain correlation before trusting expansion or KG regularizer
python neurolab/scripts/ontology_brain_correlation.py --cache-dir neurolab/data/unified_cache --ontology-dir neurolab/data/ontologies

# Distance decay: both hierarchy (hop) and embedding distance + interaction (pair-level OLS)
python neurolab/scripts/regress_brain_r_on_hop_and_embedding.py --embedding-dir neurolab/data/embedding_model --output-json neurolab/data/brain_r_hop_embedding_regression.json

# NeuroVault: build cache. For curated (126 collections → ~2–4K maps): use neurovault_curated_data + --average-subject-level. For bulk/legacy: neurovault_data + --from-downloads if manifest not yet written.
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level
python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_data --output-dir neurolab/data/neurovault_cache --from-downloads   # when manifest not yet written (legacy/bulk)
python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/unified_cache --ontology-dir neurolab/data/ontologies --output-dir neurolab/data/unified_cache_expanded --neuromaps-cache-dir neurolab/data/neuromaps_cache --neurovault-cache-dir neurolab/data/neurovault_cache --save-term-sources
```
