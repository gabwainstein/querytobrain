# Term Label Quality: Subject IDs, Placeholders, ICA

## Subject IDs — average by experiment

**Problem:** Labels like `Study01Subject01`, `sub001 positive` are per-subject maps. For training we want one map per experimental condition (averaged across subjects).

**What we have:**
- **Build time:** `build_neurovault_cache.py --average-subject-level` averages subject-level maps for collections in `AVERAGE_FIRST` (503, 504, 16284, etc.). Grouping uses `get_contrast_key()` in `neurovault_ingestion.py`.
- **Post-process:** `average_neurovault_cache.py` can average an existing cache. Only processes collections in `AVERAGE_FIRST`; uses `get_cognitive_condition()` to extract condition from labels.

**Collection 3324 (Kragel 2018):** 15 subjects × 18 studies. Labels: `Study01Subject01`, `Study01Subject02`, … `Study18Subject15`. **Now supported:** added to `AVERAGE_FIRST`; `Study01Subject01` → `Study 1` (263 maps → 18 averaged maps).

**To fix an existing cache:** Run `average_neurovault_cache.py` — it averages AVERAGE_FIRST collections in-place (or `--output-dir` for a new cache). Example: 263 Kragel maps → 18 averaged "Study 1" … "Study 18".

**Other subject-ID collections:** Add to `AVERAGE_FIRST` and implement `get_cognitive_condition()` / `get_contrast_key()` for that collection in `neurovault_ingestion.py` and `average_neurovault_cache.py`.

---

## Placeholders — discard

**Examples:** `neurovault_image_123`, `test`, `collection 456 image 7` — fallbacks when no contrast/task metadata exists.

**Action:** **DISCARD** — no semantic content. The relabel pipeline (`relabel_terms_with_llm.py`) excludes them. Current curated cache: 0 placeholders.

---

## ICA loadings — keep, relabel in plain language

**What they are (non-technical):** Each ICA component is a **brain network pattern** — a set of regions that tend to activate together. Think of it as "brain regions that work as a team."

**Examples:** `z-value voxel loadings NeuroSynth IC14 Hippocampus` → the hippocampus network; `IC3 DefaultMode` → the default mode network.

**Action:** **KEEP and RELABEL** — the maps are useful (real brain activation patterns). The LLM relabeler turns technical labels into plain language (e.g. "Hippocampus network", "Default mode network"). Moved from DISCARD to RELABEL in `relabel_terms_with_llm.py`.

---

## Collection metadata for dose/drug-aware labels

**Problem:** Dose collections (e.g. ibuprofen 200/600 mg) have vague image labels (regparam, main experiment). Drug/dose info lives in collection metadata, not per-image fields.

**Solution (implemented):**
1. **Manifest:** Download scripts now store `collection_description` in `collections_meta` (pharma, curated, data).
2. **Cache:** `build_neurovault_cache.py` writes `collection_metadata.json` (id → name, description) when manifest has `collections_meta`.
3. **Relabel:** `relabel_terms_with_llm.py` passes collection name + description to the LLM prompt. For vague labels, the model infers drug, dose, task from context (e.g. "200 mg ibuprofen emotion task").

**Reproducible:** LLM runs with `temperature=0`. Use `--model gpt-4.1-mini` for stronger biomedical summarization.
