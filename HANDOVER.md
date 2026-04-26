# NeuroLab Handover: Audit and Run Guide

This document is a **handover audit** for a new agent or team: what the repository contains, what is required to run the system, and a step-by-step record of the implementation. All paths are relative to the **repository root** (the directory containing `neurolab/`). No personal identifiers or machine-specific paths are used.

---

## 1. Repository layout and branching

### 1.1 Top-level structure

```
<repo_root>/
├── .gitignore              # Ignore generated data, .env, .venv, caches (this handover)
├── HANDOVER.md             # This document
├── docs/                   # Reference specs and integration notes
├── neurolab/               # Core NeuroLab code and data config
├── neuroquery_data/        # Downloaded by neuroquery (ignored)
├── nilearn_cache/          # Nilearn cache (ignored)
└── twitter-agent-kit/      # Separate agent kit (if present)
```

### 1.2 What to use for “run the system”

- **Single entry for “run everything”:** `neurolab/scripts/run_full_cache_build.py`
- **Query entry (after build):** `neurolab/scripts/query.py`
- **Main pipeline doc:** `neurolab/docs/implementation/BUILD_MAPS_AND_TRAINING_PIPELINE.md`
- **Quick start:** `neurolab/README.md`

### 1.3 Branching

- **main** (or **master**): primary branch. All “run the system” code and the handover live here.
- No feature branches are required to run the pipeline; optional branches may exist for experiments.

---

## 2. What is needed to run (minimal set)

### 2.1 Code and config (must be in repo)

| Path | Purpose |
|------|--------|
| `neurolab/parcellation.py` | Atlas and masker; used by all caches and inference |
| `neurolab/enrichment/` | `text_to_brain.py`, `unified_enrichment.py`, `cognitive_decoder.py`, `scope_guard.py`, `term_expansion.py` |
| `neurolab/scripts/run_full_cache_build.py` | Full pipeline runner |
| `neurolab/scripts/query.py` | Query CLI (map + retrieval + enrichment) |
| `neurolab/scripts/train_text_to_brain_embedding.py` | Train text→brain model |
| `neurolab/scripts/build_combined_atlas.py` | Build parcellation |
| `neurolab/scripts/build_term_maps_cache.py` | NeuroQuery decoder cache |
| `neurolab/scripts/build_neurosynth_cache.py` | NeuroSynth cache |
| `neurolab/scripts/merge_neuroquery_neurosynth_cache.py` | NQ+NS → unified |
| `neurolab/scripts/download_ontologies.py` | Ontologies for expansion |
| `neurolab/scripts/build_neurovault_cache.py` | NeuroVault task cache |
| `neurolab/scripts/download_neurovault_curated.py` | NeuroVault curated download |
| `neurolab/scripts/improve_neurovault_labels.py` | NeuroVault label cleanup |
| `neurolab/scripts/build_neuromaps_cache.py` | Neuromaps biological cache |
| `neurolab/scripts/build_enigma_cache.py` | ENIGMA structural cache |
| `neurolab/scripts/build_abagen_cache.py` | abagen gene expression cache |
| `neurolab/scripts/build_pharma_neurosynth_cache.py` | Pharma NeuroSynth cache |
| `neurolab/scripts/build_expanded_term_maps.py` | Merge all → merged_sources (training set) |
| `neurolab/scripts/relabel_pharma_terms.py` | Pharma schema-based relabeling |
| `neurolab/scripts/relabel_pharma_semantic.py` | Pharma natural-language labels |
| `neurolab/scripts/ontology_expansion.py` | Term expansion from ontologies |
| `neurolab/scripts/ontology_meta_graph.py` | Meta-graph for retrieval augmentation |
| `neurolab/scripts/term_to_map.py` | Term→map resolution (cache + neuromaps) |
| `neurolab/scripts/query_to_map.py` | Query→weights→map (NeuroQuery-style) |
| `neurolab/data/neurovault_pharma_schema.json` | Pharma collections and semantic labels |
| `neurolab/data/receptor_knowledge_base.json` | Receptor KB (source path inside is relative) |
| `neurolab/requirements-enrichment.txt` | Python dependencies |

### 2.2 Data produced by scripts (not in repo; see .gitignore)

- **Parcellation:** `neurolab/data/combined_atlas_392.nii.gz` (or 427; from `build_combined_atlas.py`)
- **Caches:** `decoder_cache/`, `neurosynth_cache/`, `unified_cache/`, `neurovault_cache/`, `neurovault_pharma_cache_semantic/`, `neuromaps_cache/`, `enigma_cache/`, `abagen_cache/`, `merged_sources/`, etc.
- **Model:** `neurolab/data/embedding_model/` (encoder + head + training_embeddings, training_terms, config)
- **Ontologies:** `neurolab/data/ontologies/*.owl|obo|ttl` (from `download_ontologies.py`)

### 2.3 Environment

- **Python:** 3.9+ (3.10+ recommended)
- **Install:** `pip install -r neurolab/requirements-enrichment.txt`
- **Secrets:** `.env` in repo root with `OPENAI_API_KEY` if using OpenAI encoder (not committed; in .gitignore)
- **Run from:** repository root

---

## 3. Pipeline: step-by-step (what the code does)

### 3.1 One-command build

From repo root:

```bash
python neurolab/scripts/run_full_cache_build.py
```

This runs (in order): atlas → decoder → neurosynth → merge NQ+NS → NeuroVault (curated) → NeuroVault pharma (improved → relabeled → semantic) → neuromaps → ENIGMA → pharma NeuroSynth → abagen → merged_sources. Optional: `--quick` (small caps), `--skip-*` to skip steps.

### 3.2 Step-by-step (for reference)

1. **Atlas**  
   - Script: `neurolab/scripts/build_combined_atlas.py`  
   - Output: `neurolab/data/combined_atlas_392.nii.gz` (Glasser+Tian 392 parcels).  
   - Used by all cache builds and inference.

2. **Ontologies**  
   - Script: `neurolab/scripts/download_ontologies.py`  
   - Output: `neurolab/data/ontologies/` (MF, UBERON, Cognitive Atlas, MONDO, HPO, ChEBI, etc.).  
   - Used for term expansion and optional KG context.

3. **NeuroQuery decoder**  
   - Script: `neurolab/scripts/build_term_maps_cache.py`  
   - Output: `neurolab/data/decoder_cache/term_maps.npz`, `term_vocab.pkl`.

4. **NeuroSynth**  
   - Script: `neurolab/scripts/build_neurosynth_cache.py`  
   - Output: `neurolab/data/neurosynth_cache/`.

5. **Merge NQ + NS**  
   - Script: `neurolab/scripts/merge_neuroquery_neurosynth_cache.py`  
   - Output: `neurolab/data/unified_cache/`.

6. **NeuroVault curated**  
   - Download: `neurolab/scripts/download_neurovault_curated.py --all`  
   - Build: `neurolab/scripts/build_neurovault_cache.py` with `--average-subject-level`  
   - Improve: `improve_neurovault_labels.py` → `neurovault_cache_improved` (and optionally relabeled).

7. **NeuroVault pharma**  
   - Build from curated data + `neurolab/data/neurovault_pharma_schema.json` (collections, label rules).  
   - Improve → relabel (schema) → semantic: `relabel_pharma_terms.py` → `relabel_pharma_semantic.py`  
   - Output: `neurovault_pharma_cache_semantic/` (preferred for merge).

8. **Neuromaps**  
   - Script: `neurolab/scripts/build_neuromaps_cache.py`  
   - Output: `neurolab/data/neuromaps_cache/`.

9. **ENIGMA**  
   - Script: `neurolab/scripts/build_enigma_cache.py`  
   - Output: `neurolab/data/enigma_cache/`.

10. **Pharma NeuroSynth**  
    - Script: `neurolab/scripts/build_pharma_neurosynth_cache.py`  
    - Output: `neurolab/data/pharma_neurosynth_cache/`.

11. **abagen**  
    - Script: `neurolab/scripts/build_abagen_cache.py`  
    - Output: `neurolab/data/abagen_cache/` (and optional residual/denoised caches).

12. **Merged training set**  
    - Script: `neurolab/scripts/build_expanded_term_maps.py`  
    - Inputs: unified_cache, neurovault (improved/relabeled), neurovault_pharma (semantic preferred), neuromaps, enigma, abagen, pharma_neurosynth.  
    - Output: `neurolab/data/merged_sources/` (term_maps.npz, term_vocab.pkl, term_sources.pkl).

13. **Train text→brain model**  
    - Script: `neurolab/scripts/train_text_to_brain_embedding.py`  
    - Input: `--cache-dir neurolab/data/merged_sources`  
    - Output: `neurolab/data/embedding_model/` (config, encoder weights, head, training_embeddings.npy, training_terms.pkl).  
    - Optional: `--encoder openai`, `--ontology-retrieval-cache-dir`, KG context, etc.

14. **Query (inference)**  
    - Script: `neurolab/scripts/query.py "working memory" --use-embedding-model neurolab/data/embedding_model --cache-dir neurolab/data/merged_sources`  
    - Loads `TextToBrainEmbedding`; runs retrieval-first (top-k evidence + provenance) then predicted map; then `UnifiedEnrichment` (cognitive + biological).  
    - Use `--guardrail off` if ScopeGuard fails (e.g. when encoder is OpenAI and guard expects sentence-transformers).

---

## 4. Retrieval-first and pharma semantics (recent work)

### 4.1 Retrieval-first output

- **Where:** `neurolab/enrichment/text_to_brain.py`  
  - `retrieve(query, top_k=10)`  
  - `predict_map_with_retrieval(text, top_k=10)` → (map, {retrieval, confidence, map})  
  - Uses `training_embeddings.npy` and `training_terms.pkl` from the model dir; optional `cache_dir` for `term_sources.pkl` and `term_maps.npz` for provenance and evidence maps.
- **Query CLI:** `neurolab/scripts/query.py` uses `predict_map_with_retrieval` when `--use-embedding-model` is set and prints top evidence terms with similarity and source.
- **Trainer:** Saves `cache_dir` in `config.pkl` for provenance at inference.

### 4.2 Pharma semantic labels

- **Schema:** `neurolab/data/neurovault_pharma_schema.json`  
  - `semantic_label_by_collection`: natural-language description per collection.  
  - `label_prefix_by_collection`, `abbreviation_expansions`, `source_buckets`, `exclude_from_pharma`.
- **Script:** `neurolab/scripts/relabel_pharma_semantic.py`  
  - Reads schema; maps collection ID → semantic label; produces `neurovault_pharma_cache_semantic/`.
- **Merge:** `run_full_cache_build.py` and `build_expanded_term_maps.py` prefer `neurovault_pharma_cache_semantic` when present.

### 4.3 Anonymization

- `neurolab/data/receptor_knowledge_base.json`: `_source` set to relative path `neurolab/docs/implementation/receptor_gene_list_v2.csv` (no machine or user paths).
- No personal identifiers (e.g. usernames, real names) in the handover or in the minimal runnable set above.

---

## 5. Documents: what to read first

| Document | Audience | Purpose |
|----------|----------|---------|
| `neurolab/README.md` | Everyone | Quick start, verify, query, train |
| `neurolab/docs/implementation/BUILD_MAPS_AND_TRAINING_PIPELINE.md` | Dev/ops | Full pipeline, commands, options |
| `HANDOVER.md` (this file) | New agent / team | Audit, paths, run set, branching |
| `neurolab/docs/implementation/expandable-term-space-embeddings.md` | Dev | Text→brain training and inference |
| `neurolab/docs/implementation/accuracy-and-testing.md` | Dev | How accuracy and tests are defined |
| `neurolab/docs/implementation/TESTING_RUNBOOK.md` | Dev | How to run tests |

Analytical or one-off audit docs (e.g. TRAINING_GAP_ANALYSIS, CACHE_AUDIT_*, REPO_REVIEW, various “AUDIT” and “ANALYSIS” files) are for internal review only; not required to run the system.

---

## 6. Scripts: core vs optional/legacy

### 6.1 Core (needed to build and run)

- `run_full_cache_build.py` – full pipeline  
- `query.py` – query with retrieval + enrichment  
- `train_text_to_brain_embedding.py` – train embedding model  
- `build_combined_atlas.py`, `build_term_maps_cache.py`, `build_neurosynth_cache.py`, `merge_neuroquery_neurosynth_cache.py`  
- `download_ontologies.py`, `build_neurovault_cache.py`, `download_neurovault_curated.py`, `improve_neurovault_labels.py`  
- `build_neuromaps_cache.py`, `build_enigma_cache.py`, `build_abagen_cache.py`, `build_pharma_neurosynth_cache.py`  
- `build_expanded_term_maps.py`  
- `relabel_pharma_terms.py`, `relabel_pharma_semantic.py`  
- `ontology_expansion.py`, `ontology_meta_graph.py`  
- `term_to_map.py`, `query_to_map.py`  
- Verification: `verify_environment.py`, `verify_decoder.py`, `verify_unified.py`, `verify_parcellation_and_map_types.py`

### 6.2 Supporting (used by core or useful for ops)

- `build_training_embeddings.py`, `embed_ontology_labels.py`  
- `report_train_correlation_by_collection.py`  
- `check_training_readiness.py`, `ensure_all_brain_map_data_local.py`  
- `composite_distance_utils.py` (used by graph/composite scripts)

### 6.3 Optional / experimental

Experimental sweep, diagnostic, and analysis scripts were removed in the
production cleanup. The shipping surface is the core/supporting set above
plus `report_train_correlation_by_collection.py`. To re-enable
experimental flags such as `--triad-pairwise-loss` in
`train_text_to_brain_embedding.py`, restore the supporting scripts from
git history.

---

## 7. .gitignore summary

Root `.gitignore` ensures the following are not committed:

- `.env`, `.cursor/`, `.venv/`, Python bytecode and caches  
- All generated caches under `neurolab/data/` (decoder, neurosynth, unified, neurovault*, neuromaps, enigma, abagen, merged_sources, embedding_model*, ontology files, etc.)  
- `neuroquery_data/`, `nilearn_cache/`  
- Logs and OS cruft  

Small schema/config files in `neurolab/data/` (e.g. `neurovault_pharma_schema.json`, `receptor_knowledge_base.json`) remain trackable; large or generated data stays ignored.

---

## 8. Quick run checklist

1. Clone repo; from repo root create venv and install:  
   `pip install -r neurolab/requirements-enrichment.txt`
2. Add `.env` with `OPENAI_API_KEY` if using OpenAI encoder (do not commit).
3. Run full build:  
   `python neurolab/scripts/run_full_cache_build.py`  
   (Use `--quick` for a smaller test.)
4. Train embedding model (if not already present):  
   `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources --output-dir neurolab/data/embedding_model`
5. Query with retrieval:  
   `python neurolab/scripts/query.py "working memory" --use-embedding-model neurolab/data/embedding_model --cache-dir neurolab/data/merged_sources --guardrail off`

This handover is sufficient for another agent or team to understand the layout, run the pipeline, and continue work (e.g. on another agent) without relying on legacy or analytical-only docs.
