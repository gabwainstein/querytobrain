# NeuroLab Repo Analysis (querytobrain)

**Branch:** `feature/neurolab-enrichment-pipeline`  
**Canonical repo:** this **querytobrain** repository (use a single working tree; avoid duplicating work across multiple legacy integration folders).

This doc summarizes what the neurolab repo contains, how the branch differs from main, and how the enrichment pipeline, query-to-map methods, and production setup fit together.

---

## 1. What the neurolab repo has

### 1.1 Enrichment pipeline (existing, on branch)

- **`neurolab/enrichment/`** — Core classes used by the product:
  - **CognitiveDecoder** — Decodes a brain map to cognitive terms using a **decoder cache** (term_maps.npz + term_vocab.pkl, Schaefer 400).
  - **ReceptorEnrichment** — Biological layer (e.g. Hansen receptor atlas).
  - **NeuromapsEnrichment** — Optional neuromaps annotations.
  - **UnifiedEnrichment** — One map → cognitive + biological + summary.
  - **TextToBrainEmbedding** — Optional text→brain embedding.
  - **ScopeGuard** — Guardrails.

- **Cache format:** Same across NeuroQuery and NeuroSynth caches: `term_maps.npz` (parcellated maps) + `term_vocab.pkl` (list of term strings). Parcellation: Schaefer 400.

### 1.2 Scripts: build & verify (branch-original)

| Script | Purpose |
|--------|---------|
| **build_term_maps_cache.py** | NeuroQuery model → encode each vocab term → parcellate → decoder cache. Uses `EXCLUDE_TERMS`, `--cache-dir`, `--max-terms`, `--min-term-length`. |
| **build_neurosynth_cache.py** | NeuroSynth (NiMARE) → MKDADensity per term → Schaefer 400. Accepts **--cache-dir** and **--output-dir** (for `build_all_maps.py` compatibility). |
| **build_neuromaps_cache.py** | Neuromaps annotations → biological cache; `--cache-dir`, optional `--max-annot`. |
| **build_all_maps.py** | Orchestrator: runs the three build scripts in order; `--quick` caps terms/annotations for testing. |
| **verify_decoder.py** | Verify decoder cache (shape, vocab). |
| **verify_unified.py** | Verify unified enrichment. |
| **verify_environment.py** | Check imports and data paths. |
| **test_enrichment_e2e.py** | End-to-end enrichment test. |

### 1.3 Scripts: query-to-map & pipeline (added on the enrichment branch)

| Script | Purpose |
|--------|---------|
| **option_a.py** | OptionAMapper: cache → ontology → similarity fallback for OOV/low-similarity terms. |
| **ontology_expansion.py** | Load ontology index (mf.owl, uberon.owl), `expand_term`, `get_map_for_term` (cache → ontology → similarity). |
| **download_ontologies.py** | Download ontologies to `neurolab/data/ontologies`. |
| **query_to_map.py** | `get_map_neuroquery_style`, `get_map_neuroquery_style_with_ontology`, `get_map_decompose_avg`, `correlation()`. |
| **compare_query_methods.py** | Compare Method A (neuroquery_style), A+ontology, B (decompose→avg); uses `data/unified_cache` when present. |
| **pipeline.py** | `run_pipeline(query, mapper, combine=..., use_ontology=..., decomposer=..., neuromaps_receptor=...)`; CLI; default decoder = unified_cache. |
| **build_neuroquery_cache.py** | NeuroQuery → vocabulary.csv + encode → Schaefer 400 cache (same format; used by merge/setup_production). |
| **merge_neuroquery_neurosynth_cache.py** | Union vocab; prefer NeuroSynth on overlap → unified decoder cache. |
| **setup_production.py** | Full local setup: ontologies → NQ cache → NS cache → merge → unified_cache; `--quick` for small test caps. |
| **neuromaps_fetch.py** | List/fetch/describe; **fetch-all** for all annotations (with nilearn Path workaround). |

### 1.4 Two ways to get “decoder cache”

- **decoder_cache (branch):** From **build_term_maps_cache.py** (NeuroQuery only). Used by **CognitiveDecoder** and **build_all_maps.py**.
- **unified_cache (setup_production):** From **setup_production.py**: NQ cache + NS cache → **merge** → one cache. Used by **pipeline.py**, **compare_query_methods.py**, and optional ontology path.

So: **decoder_cache** = NeuroQuery-only; **unified_cache** = merged NeuroQuery + NeuroSynth. Both use the same file format.

### 1.5 Two map-combine methods (query → map)

1. **NeuroQuery-style:** Query → weights over cache terms → weighted sum of term maps.
2. **Decompose → average:** Query → concepts (e.g. simple_rule_decomposer) → one map per concept → average maps.

Comparison and correlation: **compare_query_methods.py**.

### 1.6 Ontology path (optional)

- When a term is **OOV** or similarity to cache is **low**, ontology (mf.owl, uberon.owl) can expand to related cache terms and map via those.
- **Option A** = cache → ontology → similarity in **option_a.py** / **ontology_expansion.py**; used optionally in **query_to_map** and **pipeline**.

### 1.7 Docs

- **implementation/** — Build plan, pipeline slices, accuracy/testing, expandable-term-space, custom-prompt-to-map.
- **NEUROLAB_NEUROSYNTH_VALIDATION_AND_SEMANTIC_EXPANSION.md** — Ontologies, Option A, two combine methods, optional ontology, full local cache, validation (reference for pipeline/compare/setup_production).

---

## 2. Branch vs main

- **Modified (tracked):** e.g. `docs/external-specs/neurolab_implementation_guide.md`, `neurolab/README.md`, `neurolab/docs/*`.
- **Untracked (on branch):** e.g. `neurolab/__init__.py`, `neurolab/enrichment/`, `neurolab/scripts/`, `neurolab/data/`, implementation docs, `requirements-enrichment.txt`, `neuroquery_data/`.

All new work for this feature should stay in **querytobrain** (this repo), branch **feature/neurolab-enrichment-pipeline** when applicable; avoid scattering copies in unrelated checkout paths.

---

## 3. Reconciliations and next steps

| Item | Status / action |
|------|-----------------|
| **build_neurosynth_cache.py** | Branch’s `build_all_maps.py` passes `--cache-dir`. The copied script now accepts both **--output-dir** and **--cache-dir** so the orchestrator works. |
| **decoder_cache vs unified_cache** | Document in README or implementation: decoder_cache = NQ-only (CognitiveDecoder); unified_cache = merged NQ+NS (pipeline, compare_query_methods, setup_production). Optionally wire UnifiedEnrichment to use unified_cache. |
| **Option A / pipeline in enrichment** | OptionAMapper and pipeline live in scripts; CognitiveDecoder uses decoder cache. Decide: use unified_cache in enrichment and/or expose pipeline as an alternative “query → map” path. |
| **Implementation README** | This analysis and **NEUROLAB_NEUROSYNTH_VALIDATION_AND_SEMANTIC_EXPANSION.md** are linked from [implementation/README.md](README.md). |
| **Run from repo root** | All neurolab commands are intended from **querytobrain** root, e.g. `python neurolab/scripts/pipeline.py ...`, `python neurolab/scripts/setup_production.py --quick`. |

---

## 4. Quick reference: run from querytobrain root

```bash
# Full production data (ontologies + NQ + NS + merge)
python neurolab/scripts/setup_production.py
python neurolab/scripts/setup_production.py --quick

# Branch build (decoder + neurosynth + neuromaps)
python neurolab/scripts/build_all_maps.py
python neurolab/scripts/build_all_maps.py --quick

# Query → map comparison (uses data/unified_cache if present)
python neurolab/scripts/compare_query_methods.py "working memory"

# Pipeline
python neurolab/scripts/pipeline.py "attention" --combine decompose_avg
```
