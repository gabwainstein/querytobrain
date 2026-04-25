# NeuroLab Repo: Coherence and Consistency Review

**Date:** 2025-02-14  
**Scope:** Full neurolab repo (docs spine, enrichment code, scripts, READMEs, cross-references).

---

## Executive summary

The repo is **largely coherent**: the docs spine (product → architecture → implementation → process) is well structured, the enrichment pipeline code matches the build plan and interfaces, and scripts align with the phased verification story. A number of **fixable inconsistencies** (wrong CLI flag in README, outdated repo layout, atlas access style) and **documentation gaps** (decoder_cache vs unified_cache, pipeline/setup_production vs main README) should be addressed so code and docs stay in sync.

---

## 1. Documentation spine

### 1.1 Structure and navigation

- **docs/README.md** — Clear map: product truth → architecture → implementation → process. Reading order and “Where do I put X?” are useful.
- **docs/product/README.md** — Normative invariants (Evidence Tier, research-first, terminology) are stated; traceability and conflict rule (product wins) are clear.
- **docs/architecture/architecture.md** — Plugin boundary, tool contract, Evidence Tier (A/B/C) match **interfaces.md** and product scope.
- **docs/architecture/interfaces.md** — Viewer manifest, map registry entry, tool response, Evidence Tier, and run audit contracts are defined and referenced from implementation.

**Verdict:** Coherent. Cross-links between product, architecture, and implementation are consistent.

### 1.2 Canonical references

- Implementation docs correctly point to parent repo: `../../../docs/external-specs/` for cognitive_decoding_addendum, neurolab_implementation_guide, NeuroLab spec.
- **cognitive_decoding_addendum.md** exists at `docs/external-specs/cognitive_decoding_addendum.md`; references resolve.

---

## 2. Implementation docs vs code

### 2.1 Enrichment pipeline build plan

- **Phases 0–6** in `enrichment-pipeline-build-plan.md` match what the code and verify scripts do:
  - Phase 0+1: `verify_environment.py` (imports, NeuroQuery, Schaefer, one term → (400,)).
  - Phase 2: `build_term_maps_cache.py` → `term_maps.npz`, `term_vocab.pkl`; optional NeuroSynth/neuromaps build scripts documented.
  - Phase 3: `CognitiveDecoder` in `cognitive_decoder.py`; `verify_decoder.py`.
  - Phase 4: `ReceptorEnrichment`, `NeuromapsEnrichment`; `verify_unified.py`.
  - Phase 5: `UnifiedEnrichment.enrich()`; same verify script.
  - Phase 6: Correctly marked deferred (local testing only).

- **UnifiedEnrichment** signature in the build plan (`cache_dir`, `receptor_path`, `enable_cognitive`, `enable_biological`, `n_parcels`; optional `neuromaps_cache_dir`) matches `unified_enrichment.py`.
- **NeuromapsEnrichment** expects `annotation_maps.npz` with key `matrix` and `annotation_labels.pkl`; `build_neuromaps_cache.py` writes exactly that. Coherent.

### 2.2 Accuracy and testing

- **accuracy-and-testing.md** defines metrics (mean correlation, MSE, r per receptor, scope guardrail) and train/val/test protocol. Script names and flags referenced there match the scripts (`train_text_to_brain_embedding.py`, `verify_decoder.py`, `verify_unified.py`, `verify_embedding.py`, `query.py --guardrail on`). Coherent.

### 2.3 Expandable term space and custom prompt

- **expandable-term-space-embeddings.md** and **custom-prompt-to-map.md** match `TextToBrainEmbedding`, `train_text_to_brain_embedding.py`, and `predict_map.py`. Steps (build cache → train embedding → predict) are consistent.

---

## 3. Enrichment module coherence

### 3.1 Exports and dependencies

- **neurolab/enrichment/__init__.py** exports: CognitiveDecoder, ReceptorEnrichment, NeuromapsEnrichment, UnifiedEnrichment, TextToBrainEmbedding, ScopeGuard. All exist and are used.
- **UnifiedEnrichment** composes CognitiveDecoder, ReceptorEnrichment, NeuromapsEnrichment; merges biological `by_layer` / `top_hits` / `layer_summary`; builds a text summary. Behavior matches the docstring and build plan.

### 3.2 Shared constants and contracts

- **EXCLUDE_TERMS** is duplicated in `cognitive_decoder.py` and `build_term_maps_cache.py` with the same set. Consider moving to a shared constant (e.g. `enrichment/constants.py` or doc-only) to avoid drift.
- **n_parcels = 400** (Schaefer) is consistent across decoder, receptor, neuromaps, unified, and scripts.
- **ReceptorEnrichment** and **NeuromapsEnrichment** return the same shape of result (`by_layer`, `top_hits`, `layer_summary`); UnifiedEnrichment merges them correctly. Receptor entries include `system`; neuromaps do not; unified summary handles both.

### 3.3 Minor code note

- **UnifiedEnrichment**: `biological_parts` is a list of tuples `(name, res)`; iteration and merge logic are correct. No bug found.

---

## 4. Scripts: CLI, paths, and behavior

### 4.1 Run-from root convention

- Scripts that need repo root (query, verify_*, test_enrichment_e2e, predict_map, etc.) use the same pattern: derive `repo_root` from `__file__`, insert into `sys.path`, `os.chdir(repo_root)`. Coherent.

### 4.2 Cache directory semantics

- **decoder_cache** — Produced by `build_term_maps_cache.py` (NeuroQuery). Used by CognitiveDecoder, query.py, train_text_to_brain_embedding, verify_decoder, test_enrichment_e2e.
- **unified_cache** — Produced by setup_production (merge of NeuroQuery + NeuroSynth). Used by pipeline.py, compare_query_methods.py. Same file format as decoder_cache.
- **NEUROLAB_REPO_ANALYSIS.md** describes this split. The main **neurolab/README.md** does not explain unified_cache or when to use pipeline/setup_production vs build_term_maps_cache + query. New users may be unsure which path to use.

**Recommendation:** Add a short “Two ways to get a decoder cache” (or “Production vs branch workflow”) to the main README, pointing to NEUROLAB_REPO_ANALYSIS.md or implementation README.

### 4.3 Inconsistencies found

| Location | Issue | Fix |
|---------|--------|-----|
| **neurolab/README.md** (line ~83) | Says `query.py --cognitive-cache-dir neurolab/data/neurosynth_cache` | **query.py** only has **--cache-dir**. Use: `query.py --cache-dir neurolab/data/neurosynth_cache` (and optionally document that the same cache format works for NeuroSynth output). |
| **repo-layout.md** | “Current layout (docs-only)” and tree shows only README + docs under neurolab | Layout is outdated; neurolab now has `enrichment/`, `scripts/`, `data/`. Update the “Current layout” section to include code and data dirs (and optionally .gitignore for data). |

### 4.4 Atlas access style

- Most code uses **atlas["maps"]** (verify_environment, build_term_maps_cache, build_neuromaps_cache, query, test_enrichment_e2e, text_to_brain).
- **build_neurosynth_cache.py** and **build_neuroquery_cache.py** use **atlas.maps**.
- Nilearn’s `fetch_atlas_schaefer_2018` returns a Bunch; both work. For consistency and future-proofing, prefer **atlas["maps"]** everywhere.

---

## 5. Cross-references and broken links

- **docs/README.md** links to `../../docs/external-specs/` for canonical specs; from `neurolab/docs/` that resolves to repo root `docs/`. Correct.
- **implementation/README.md** references cognitive_decoding_addendum, accuracy-and-testing, pipeline-slices, NEUROLAB_REPO_ANALYSIS, NEUROLAB_NEUROSYNTH_VALIDATION_AND_SEMANTIC_EXPANSION; all targets exist.
- **enrichment-pipeline-build-plan.md** references expert-personas, pipeline-slices, interfaces, cognitive_decoding_addendum; paths checked and valid.

No broken internal links were found; only the README CLI flag and repo layout need edits.

---

## 6. Requirements and environment

- **requirements-enrichment.txt** lists neuroquery, nilearn, nibabel, numpy, scipy, pandas; optional neuromaps, nimare. **verify_environment.py** imports match (including pandas). Coherent.
- Phase 0 “Verify imports” in the build plan matches the packages and the verify script.

---

## 7. Summary of recommendations

### High priority (correctness / UX)

1. **Fix README CLI example:** In `neurolab/README.md`, change `query.py --cognitive-cache-dir neurolab/data/neurosynth_cache` to `query.py --cache-dir neurolab/data/neurosynth_cache`.
2. **Update repo-layout.md:** Extend “Current layout” to include `neurolab/enrichment/`, `neurolab/scripts/`, `neurolab/data/` (and note data is gitignored).

### Medium priority (consistency and clarity)

3. **Decoder vs unified cache:** In main README, add one or two sentences (and a link to NEUROLAB_REPO_ANALYSIS or implementation README) explaining decoder_cache (NeuroQuery, default for enrichment) vs unified_cache (merged NQ+NS for pipeline/compare).
4. **Atlas access:** In `build_neurosynth_cache.py` and `build_neuroquery_cache.py`, use `atlas["maps"]` instead of `atlas.maps` for consistency with the rest of the repo and build plan.

### Low priority (maintainability)

5. **EXCLUDE_TERMS:** Consider a single source of truth (e.g. shared module or doc) so build script and decoder cannot diverge.
6. **Phase 6:** When API/plugin is implemented, update enrichment-pipeline-build-plan.md “Status: Deferred” and add a short implementation note or link.

---

## 8. Conclusion

The neurolab repo is **coherent**: documentation spine, build plan, accuracy definitions, and code (enrichment modules and scripts) align. The main issues are a **wrong CLI flag in the README**, an **outdated repo layout** in docs, and **missing high-level guidance** on decoder_cache vs unified_cache. Addressing the high- and medium-priority items above will keep the repo consistent and easier to onboard.
