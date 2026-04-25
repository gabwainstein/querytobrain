# Enrichment Pipeline: Step-by-Step Build Plan

This plan ensures we **correctly fetch data, install packages, and prepare datasets/models** before and while building the cognitive decoding and unified enrichment pipeline. Each step has exit criteria and an optional owner from [expert-personas](../product/expert-personas.md).

**Reference:** [cognitive_decoding_addendum](../../../docs/external-specs/cognitive_decoding_addendum.md).

---

## Overview: Build Order

| Phase | Goal | Owner (suggested) | Exit criteria |
|-------|------|-------------------|---------------|
| **0** | Environment & packages | Coding | All imports work; versions pinned |
| **1** | Fetch & verify data/model | Coding + AI | NeuroQuery + parcellation work in a notebook |
| **2** | Build cognitive term maps cache | AI + Coding | `term_maps.npz` + `term_vocab.pkl` exist; shape correct |
| **3** | CognitiveDecoder class + tests | AI + Science | `decode()` returns sensible terms for known maps |
| **4** | Biological layer (Hansen or minimal) | Science + Coding | Receptor enrichment works for a test map |
| **5** | UnifiedEnrichment + summary | AI + Application | One map → cognitive + biological + summary |
| **6** | API / plugin integration | Coding + Application | Endpoint or tool returns manifest + Evidence Tier |

---

## Phase 0: Environment & Packages

**Goal:** Reproducible environment with all dependencies. No pipeline code yet.

### 0.1 Python and venv

- Python **3.9+** (3.10 or 3.11 recommended).
- Create and use a virtual environment (e.g. `python -m venv .venv`).
- Document in README or `requirements.txt` / `pyproject.toml`.

### 0.2 Core packages (cognitive pipeline)

Install and **pin versions** in `requirements.txt` or `pyproject.toml`:

| Package | Purpose | Min version (suggested) |
|---------|---------|--------------------------|
| `neuroquery` | Text→brain map + vocabulary for term maps | latest (check PyPI) |
| `nilearn` | Parcellation (Schaefer), maskers, datasets | >= 0.10 |
| `nibabel` | NIfTI I/O for brain maps | >= 3.2 |
| `numpy` | Arrays, correlation | >= 1.21 |
| `scipy` | stats (pearsonr, zscore) | >= 1.7 |
| `pandas` | Optional: result tables | — |

### 0.3 Optional packages (biological layer)

- **Option A — Hansen only:** No extra install; use a Hansen receptor CSV (e.g. parcellated densities). Lightweight.
- **Option B — neuromaps:** `neuromaps` for 80+ annotations (receptors, MEG, metabolism, hierarchy). Adds ~2–5 GB download on first run. Install only when starting Phase 4 with full biological layer.

### 0.4 Verify imports

Run once and confirm no errors:

```python
import neuroquery
from neuroquery import fetch_neuroquery_model
import nilearn
from nilearn import datasets as nilearn_datasets
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
from scipy import stats
import nibabel as nib
```

**Exit criteria:** All imports succeed; versions logged (e.g. `pip freeze > requirements-lock.txt`).  
**Owner:** Coding Lead.

---

## Phase 1: Fetch Data & Model — Verify Before Coding

**Goal:** Confirm we can fetch the NeuroQuery model and the parcellation atlas; no pipeline logic yet.

### 1.1 Fetch NeuroQuery model

- **NeuroQuery 1.1.x:** `fetch_neuroquery_model()` returns a path (str); load with `NeuroQueryModel.from_data_dir(path)`. Vocabulary: `model.vectorizer.get_feature_names()`. Use `model.transform([document])` (list of strings); returns dict with `"brain_map"` (list of NIfTI images) and `"z_map"`. Vocabulary size ~6308.
- **Check:** `result["brain_map"][0]` is a NIfTI image; shape may be (46, 55, 46) or similar; parcellation masker will resample as needed.
- **Note:** First run may download ~500 MB–1 GB. Cache typically in `neuroquery_data/` under cwd or package default.

### 1.2 Fetch parcellation atlas

- `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)`.
- **Check:** `atlas['maps']` is a NIfTI image; number of non-zero labels = 400.
- **Check:** Build `NiftiLabelsMasker(labels_img=atlas['maps'], standardize=False)` and `fit_transform` on a small test volume → output shape `(1, 400)`.

### 1.3 One end-to-end test (notebook or script)

- Get one term from NeuroQuery, e.g. `result = model("motor")`; get `brain_map`.
- Parcellate: `parcellated = masker.fit_transform(brain_map).ravel()` → shape `(400,)`.
- **Check:** No all-zero or all-NaN; values look plausible.

**Exit criteria:** NeuroQuery model and Schaefer 400 parcellation work; one term → parcellated vector (400,) with valid numbers.  
**Owner:** Coding Lead + AI Lead.

---

## Phase 2: Build Cognitive Term Maps Cache

**Goal:** Precompute parcellated brain maps for all NeuroQuery terms and save to disk. This is the slow step (1–2 hours); afterwards decode is fast.

### 2.1 Design cache layout

- **Cache directory (local only, no backend):** e.g. `neurolab/data/decoder_cache` (configurable). All pipeline scripts use this local folder; no API or server.
- **Files:**  
  - `term_maps.npz` with key `term_maps` → array shape `(n_terms, n_parcels)` (e.g. ~5000×400).  
  - `term_vocab.pkl` → list of term strings (same order as rows of `term_maps`).

### 2.2 Build script or CognitiveDecoder._load_or_build_term_maps()

- Load NeuroQuery model and parcellation (as in Phase 1).
- Filter vocabulary: exclude short terms, generic words (e.g. "participants", "activation", "brain"), numerics — per addendum `EXCLUDE_TERMS` and `min_term_length`.
- For each term: `model(term)["brain_map"]` → parcellate → append to list. Skip empty maps.
- Optional: cap to top `max_terms` by variance (e.g. 5000) to keep cache smaller.
- Z-score each row (optional but useful for Pearson in decode): `term_maps_z = stats.zscore(term_maps, axis=1)`; handle NaN.
- Save `term_maps.npz` (compressed) and `term_vocab.pkl`.

### 2.3 Validate cache

- Load `term_maps.npz` and `term_vocab.pkl`.
- **Check:** `term_maps.shape[1] == 400` (or your n_parcels).
- **Check:** No row is all NaN or all zero.
- **Check:** Vocabulary length matches `term_maps.shape[0]`.

**Exit criteria:** Cache files exist; shape and content validated; build time and disk size documented.  
**Owner:** AI Lead + Coding Lead.

**Implemented:** `neurolab/scripts/build_term_maps_cache.py` — run with `--cache-dir neurolab/data/decoder_cache` (default), `--max-terms 5000` (or 0 for full vocab). Saves `term_maps.npz` (with `term_maps_z`) and `term_vocab.pkl`. Cache dir is in `neurolab/data/.gitignore` (not committed).

**NeuroSynth (NiMARE) — optional:** `neurolab/scripts/build_neurosynth_cache.py` — same output format (`term_maps.npz`, `term_vocab.pkl`) so `CognitiveDecoder(cache_dir=neurosynth_cache)` works. Requires `nimare`; first run downloads NeuroSynth; MKDA per term is slow (use `--max-terms 100` for a test). ODbL applies to NeuroSynth data.

**Build all maps:** `neurolab/scripts/build_all_maps.py` — runs NeuroQuery (full vocab), NeuroSynth (all terms), and neuromaps (all MNI152 annotations) in sequence. Use for a complete enrichment system. `--quick` uses small caps for testing; `--skip-neurosynth` / `--skip-neuromaps` to build only some caches.

### Expandable term space (train embeddings)

To **expand beyond the fixed vocabularies** (NeuroQuery ~6.3k, NeuroSynth ~1k), we train a **text → 400-D** model so arbitrary text maps into the same parcellated space. Then we can decode to nearest known terms or add synthetic terms.

- **Design:** [expandable-term-space-embeddings.md](expandable-term-space-embeddings.md)
- **Train:** `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model` (optional `--encoder sentence-transformers` for better semantics).
- **Use (local only):** Predict maps for text not in any DB: `predict_map(text)` returns (400,) predicted map; `predict_map_to_nifti(text, path)` saves 3D NIfTI. Script: `python neurolab/scripts/predict_map.py "phrase" [--save-nifti out.nii.gz]`. Query: `query.py --use-embedding-model ... "any phrase"`. Verify: `python neurolab/scripts/verify_embedding.py`.
- **Compare encoders:** `python neurolab/scripts/compare_encoders.py --max-terms 500` trains tfidf, MiniLM, and mpnet on the same subset and prints val metrics so you can pick the best encoder for a full run.

---

## Phase 3: CognitiveDecoder Class + Tests

**Goal:** Implement the decoder class that loads the cache and exposes `decode(parcellated_activation)`.

### 3.1 Implement CognitiveDecoder

- **Init:** Load parcellation; set cache paths; call `_load_or_build_term_maps()` (or load existing cache).
- **decode(parcellated_activation, method="pearson", top_n=30):**  
  - Compute correlation of input with each term map (vectorized: e.g. `term_maps_z @ zscore(activation)` for Pearson).  
  - Return dict: `top_terms`, `all_correlations`, `category_scores` (if you add categories), `word_cloud_data`.

### 3.2 Unit tests

- **Test 1:** Decode a synthetic vector (e.g. random) → returns top_terms list; no crash.
- **Test 2:** Decode a known map: e.g. parcellated map for term "motor" from NeuroQuery → top terms should include "motor", "movement", etc. (Science Lead can define expected terms.)
- **Test 3:** Wrong shape (e.g. 200 instead of 400) → clear error.

**Exit criteria:** CognitiveDecoder.decode() returns sensible results for at least one known map; tests pass.  
**Owner:** AI Lead + Science Lead (interpretation check).

**Implemented:** `neurolab/enrichment/cognitive_decoder.py` — `CognitiveDecoder(cache_dir)` loads cache; `decode(parcellated_activation, method="pearson"|"spearman"|"cosine", top_n=30)` returns `top_terms`, `all_correlations`, `category_scores`, `word_cloud_data`. Verification: `python neurolab/scripts/verify_decoder.py`.

---

## Phase 4: Biological Layer (Receptor Enrichment)

**Goal:** Correlate a parcellated map with receptor (and optionally other) maps. Start minimal (Hansen CSV or a few neuromaps), then expand if needed.

### 4.1 Option A — Hansen receptor atlas only

- Obtain Hansen parcellated receptor densities (e.g. CSV or pre-parcellated matrix) for 19 receptors in the same parcellation (Schaefer 400).
- Implement a small `ReceptorEnrichment` (or `HansenEnrichment`): load matrix `(n_receptors, n_parcels)`, correlate input with each row, return sorted list with receptor name and r.
- **Check:** For a test map (e.g. "serotonin" from NeuroQuery parcellated), 5-HT receptors should rank high.

### 4.2 Option B — neuromaps (subset or full)

**Implemented:** `neurolab/scripts/build_neuromaps_cache.py` — fetches annotations via `neuromaps.datasets` (filter by `--tags` or `--space MNI152`), parcellates to Schaefer 400, saves `annotation_maps.npz` and `annotation_labels.pkl`. `NeuromapsEnrichment(cache_dir)` loads and correlates; `UnifiedEnrichment(neuromaps_cache_dir=...)` runs receptor + neuromaps and merges biological results.

### 4.3 Validate biological layer

- Run enrichment on a test parcellated map.
- **Check:** Results are ordered by |r|; receptor names and r values look plausible.

**Exit criteria:** Biological enrichment runs and returns at least receptor (and optionally other) correlations; one sanity check passes.  
**Owner:** Science Lead + Coding Lead.

**Implemented:** `neurolab/enrichment/receptor_enrichment.py` — `ReceptorEnrichment(receptor_matrix_path=None, n_parcels=400)` loads CSV or NPZ (matrix shape `n_receptors x n_parcels`); if path missing, uses placeholder data (19 Hansen receptor names). Hansen receptor atlas is **also available via neuromaps** (build neuromaps cache; then use `NeuromapsEnrichment` or `UnifiedEnrichment(neuromaps_cache_dir=...)` for receptor + other biological maps). Direct Hansen CSV/NPZ path remains supported for pre-parcellated files from [netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors).

---

## Phase 5: UnifiedEnrichment + Summary

**Goal:** One entry point that runs cognitive decode + biological enrichment and produces a short text summary.

### 5.1 Implement UnifiedEnrichment

- **Init:** Optional `CognitiveDecoder`, optional biological (Hansen or MultimodalEnrichment); parcellation aligned.
- **enrich(parcellated_activation, cognitive_top_n=20, biological_method="pearson"):**  
  - Call cognitive.decode() if enabled.  
  - Call biological.enrich() if enabled.  
  - Build `summary` string from top cognitive terms + top biological hits (e.g. for LLM or UI).

### 5.2 Test end-to-end

- Generate one parcellated map (e.g. from NeuroQuery for "psilocybin" or "attention").
- Call `UnifiedEnrichment().enrich(map)`.
- **Check:** Keys `cognitive`, `biological`, `summary` present; summary is readable and consistent with the map.

**Exit criteria:** enrich() returns cognitive + biological + summary; one full example logged.  
**Owner:** AI Lead + Application Lead.

**Implemented:** `neurolab/enrichment/unified_enrichment.py` — `UnifiedEnrichment(cache_dir, receptor_path=None, enable_cognitive=True, enable_biological=True, n_parcels=400)`. `enrich(parcellated_activation)` returns `cognitive` (if decoder cache present), `biological` (receptor enrichment), and `summary` (short text from top terms + top receptor hits). Verification: `python neurolab/scripts/verify_unified.py`.

---

## Phase 6: API / Plugin Integration

**Goal:** Expose the pipeline via an API or ResearchAgent tool so that data is fetched correctly, responses include Evidence Tier, and the contract matches [interfaces](../architecture/interfaces.md).

### 6.1 Request/response contract

- **Input:** Query text (and/or precomputed parcellated map if BYOE).  
- **Output:** At least: brain map or map ref, cognitive terms, biological enrichment, **summary**, **manifest** (provenance, Evidence Tier, attribution).  
- Evidence Tier: Label Tier C for NeuroQuery-derived maps and cognitive decode; Tier B if CBMA was used; Tier A if IBMA.

### 6.2 Implement endpoint or tool

- **Option A:** FastAPI route, e.g. `POST /api/enrich` or `POST /api/query` that: (1) optional text→map, (2) parcellate, (3) UnifiedEnrichment.enrich(), (4) build manifest, (5) return JSON.
- **Option B:** ResearchAgent plugin tool, e.g. `neuro.enrich_map(parcellated_map)` or `neuro.query_and_enrich(text)` returning artifacts + manifest.

### 6.3 Validate

- Call endpoint/tool with a test query.
- **Check:** Response has manifest with Evidence Tier and provenance; attribution present when using ODbL/CC-BY data.

**Exit criteria:** One working endpoint or tool; response matches contract; Evidence Tier in manifest.  
**Owner:** Coding Lead + Application Lead.

**Status:** Deferred. Focus is **local testing** (no API connections).

### Local testing (no API)

Run from **querytobrain** repo root:

| Script | Purpose |
|--------|--------|
| `python neurolab/scripts/verify_environment.py` | Phase 0+1: imports, NeuroQuery + Schaefer fetch, one term → (400,) |
| `python neurolab/scripts/verify_decoder.py` | Phase 3: CognitiveDecoder (builds minimal cache if missing) |
| `python neurolab/scripts/verify_unified.py` | Phase 4+5: ReceptorEnrichment + UnifiedEnrichment |
| `python neurolab/scripts/test_enrichment_e2e.py` | E2E: text → parcellated map → `UnifiedEnrichment.enrich()`; requires decoder cache |
| `python neurolab/scripts/verify_embedding.py` | Train small text→brain embedding; run query with `--use-embedding-model` (local, no API) |

E2E uses NeuroQuery to get a map for a known term (e.g. "attention"), then runs full enrichment and asserts on `cognitive`, `biological`, `summary`.

---

## Checklist Summary (copy-paste for tracking)

```
[ ] Phase 0: venv, packages (neuroquery, nilearn, nibabel, numpy, scipy), imports OK
[ ] Phase 1: NeuroQuery model fetch OK; Schaefer 400 parcellation OK; one term → (400,) vector
[ ] Phase 2: term_maps.npz + term_vocab.pkl built and validated
[ ] Phase 3: CognitiveDecoder.decode() implemented and tested on known map
[ ] Phase 4: Biological layer (Hansen or neuromaps) implemented and tested
[ ] Phase 5: UnifiedEnrichment.enrich() returns cognitive + biological + summary
[ ] Phase 6: API or plugin tool returns manifest + Evidence Tier
```

---

## Dependencies Between Steps

- **Phase 1** must complete before **Phase 2** (we need the model and parcellation to build the cache).
- **Phase 2** must complete before **Phase 3** (decoder needs the cache).
- **Phase 3** and **Phase 4** can be done in parallel once Phase 2 (and optionally 4.1/4.2 data) are done.
- **Phase 5** depends on Phase 3 and Phase 4.
- **Phase 6** depends on Phase 5.

---

**See also:** [pipeline-slices.md](pipeline-slices.md), [expert-personas.md](../product/expert-personas.md), [cognitive_decoding_addendum](../../../docs/external-specs/cognitive_decoding_addendum.md).
