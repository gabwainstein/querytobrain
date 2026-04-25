# NeuroLab: NeuroSynth Sanity Check & Expanding Semantic Space

This document describes (1) how to validate decoder and text-to-brain output against NeuroSynth maps for well-studied terms, and (2) how to expand the semantic space beyond NeuroQuery’s vocabulary while staying as accurate as possible.

---

## 0. How this works: ontologies vs SciBERT

Two different levers are often confused. Here’s how they fit together.

### Ontology expansion = query-time fallback (we do **not** expand the training vocabulary)

- **What it does:** When a **query term is not in the decoder cache** (OOV), or is **far in semantic space** from any cache term (embedding/TF-IDF similarity low), we look it up in **canonical ontologies** (MF, UBERON, Cognitive Atlas, NIFSTD). The ontology gives us **related terms** (synonyms, parents, children). We then **resolve those to terms that *are* in the cache** and return a **weighted average of their maps**.
- **The idea:** Query terms may be **not close in semantic space** (model embeddings, NeuroQuery vocabulary) to cache terms, but **close in ontology space** (same concept, synonym, parent/child). We use **ontology proximity** to find cache terms that are related in concept, and use *those* cache terms’ maps to answer the query. So we do the query **via ontology-related cache terms**, not via the raw query term in semantic space. In other words: it **is** a query in the **semantic space of the ontology-related terms** (the cache terms we resolved to)—we return the map(s) of those terms (e.g. weighted average), i.e. we query in *their* semantic space, not the query term’s.
- **So:** We do **not** add new (term, map) pairs to the training set from ontologies. We expand **effective coverage at query time**: OOV or “semantically far” term → ontology → related cache terms (close in ontology space) → average map. The **training data** stays the same (decoder cache from NeuroQuery/NeuroSynth).
- **Use case:** Fallback when the user asks for a concept that never appears in the cache or is far in embedding space (e.g. a rare disorder name); we still return a plausible map by **using ontology-related cache terms to do the query**.

### SciBERT = encoder for text-to-brain (same training data, better text representation)

- **What it does:** The **text-to-brain** model has two parts: (1) a **text encoder** (e.g. all-mpnet-base-v2 or **SciBERT**) that turns any text into a vector, and (2) a **head** (e.g. MLP) that maps that vector to a 400-D brain map. **Training** uses (term, map) pairs from the **decoder cache only**: for each term we get its embedding from the encoder and train the head to predict the cache map. We do **not** fine-tune SciBERT “on the ontologies”; we use SciBERT as the encoder and train the **head** on the same (term, map) cache.
- **Optional:** You can **fine-tune SciBERT** (or another domain model) on neuroscience text (e.g. abstracts) **before** training the head, so the encoder’s embedding space is better aligned with neuroscience terminology. Then train the head on (term, map) as before.
- **So:** “Fine-tune with SciBERT” means: **encoder = SciBERT** (pretrained or fine-tuned on neuro text), **head = embedding → map** trained on **(term, map) from the decoder cache**. Ontologies are not part of that training; they are a separate, query-time fallback.

### How they fit together

| Step | Ontology expansion | SciBERT (encoder) |
|------|--------------------|--------------------|
| **Training data** | Not used. Training = (term, map) from decoder cache only. | Same. Encoder turns each term into an embedding; head is trained to predict the cache map from that embedding. |
| **Training** | — | Optional: fine-tune SciBERT on neuro text; then train head on (term, map). Or use pretrained SciBERT and train only the head. |
| **Query time** | If term is **OOV** (not in cache): ontology → related cache terms → **weighted average map** (fallback). | For **any** text: text → SciBERT embedding → head → map. |
| **Combined** | Use **SciBERT + head** for in-vocabulary and in-distribution text; use **ontology fallback** when the term is OOV or the model is low-confidence. |

So: **we expand effective vocabulary at query time using ontologies** (fallback map from related cache terms), and **we improve the semantic space of the model using SciBERT** (better encoder, same cache for training). We do not “expand the vocabulary and then fine-tune SciBERT on it”—we use the same cache to train the (SciBERT → head) model, and use ontologies only as a fallback when a query is OOV.

### Do we need better training to expand (term → brain map)?

We **do not have to** retrain a NeuroQuery-like model (full text + coordinates → maps) to get more coverage. We already use NeuroQuery's (and optionally NeuroSynth's) outputs as our decoder cache. Choices:

| Option | What | When |
|--------|------|------|
| **A. No new training** | Keep current cache; train text-to-brain (SciBERT + head) on it; use **ontology fallback** at query time for OOV. Coverage = cache terms + ontology-resolved cache terms. | Default. Enough if ontology fallback gives good answers for your OOV terms. |
| **B. Expand training set with ontology-derived pseudo (term, map)** | For **ontology terms not in the cache**, assign map = weighted average of ontology-related cache terms' maps. Add these as **(term, map)** training pairs. Train text-to-brain on **cache + pseudo pairs**. So we "expand" the training vocabulary using the **same** brain-map source (cache), via ontology. | Use if you want the **model** (not only the fallback) to predict maps for more terms; validate on held-out terms and NeuroSynth overlap (pseudo-labels can propagate errors). |
| **C. NeuroQuery-like pipeline from scratch** | Train a model from **full-text papers + coordinates** → brain maps (TF-IDF + semantic smoothing + regression), like NeuroQuery. | Only if you need maps from a **different corpus** or method; large effort. |

**Recommendation:** Start with **A** (ontology fallback, no new training). If OOV performance is still weak, try **B** (ontology-expanded training set) with validation. **C** is only if you need your own "NeuroQuery" from different data.

### Implementation: Option A (chosen)

We implement **Option A**: no new training; keep current decoder cache; use **ontology fallback at query time** for OOV.

**Steps:**

1. **Decoder cache**  
   Use your existing NeuroQuery (and optionally NeuroSynth) decoder cache: `term_maps.npz`, `term_vocab.pkl` in a directory (e.g. `path/to/decoder_cache`).

2. **Ontologies (one-time)**  
   ```bash
   python scripts/download_ontologies.py
   ```  
   Saves `mf.owl` and `uberon.owl` to `data/ontologies/`.

3. **Query with ontology fallback**  
   For any term:
   - **In cache:** return that term’s map.
   - **OOV:** ontology → related cache terms → weighted average of their maps.

   **Entry point:** `scripts/option_a.py` (Option A in code and CLI).  
   ```bash
   python scripts/option_a.py --decoder-cache-dir path/to/decoder_cache "working memory" "prosopagnosia"
   ```  
   In code:
   ```python
   from option_a import OptionAMapper
   mapper = OptionAMapper(ontology_dir="data/ontologies", decoder_cache_dir="path/to/decoder_cache")
   map_400 = mapper.get_map("working memory")   # cache or ontology fallback
   ```
   Alternative: `scripts/query_with_ontology_fallback.py` or `get_map_for_term(...)` from `ontology_expansion`.

4. **Optional: text-to-brain**  
   If you have a trained text-to-brain model (SciBERT + head on same cache), use it first for arbitrary text; use ontology fallback when the term is OOV or low-confidence.

5. **Optional: cosine-similarity fallback**  
   When ontology returns no cache terms, pass an **encoder** and **cache_embeddings** (same encoder used for cache terms) to `get_map_for_term` or `OptionAMapper`. The query term is embedded; top-K cache terms by cosine similarity are used to blend their maps. So OOV → ontology first; if none, then similarity fallback.

No new training; no ontology-derived pseudo-labels. Coverage = cache terms + ontology-resolved cache terms + optional similarity fallback at query time.

### Query flow: step-by-step example

**Example query:** *"Show me the brain activity map when I see my dog after I took ritalin."*

| Step | What happens |
|------|----------------|
| **1. Parse / decompose query** | Turn the sentence into **concepts** the system can map to brain maps. E.g. "see my dog" → *visual perception*, *object recognition*, *reward* (or *face/object recognition*); "ritalin" → *attention*, *dopamine* (drug), *stimulant*. This can be done with an LLM or a simple rule/NER layer that outputs a short list of terms (e.g. `["visual perception", "object recognition", "attention"]`). |
| **2. Get a cognitive map per concept** | For each concept (e.g. "visual perception", "attention"): **(a)** Is it in the **unified cache** (NeuroQuery + NeuroSynth)? → return that term’s map. **(b)** If **OOV**, use **ontology**: look up the concept in MF/UBERON etc. → get related cache terms → **weighted average** of their maps. **(c)** If ontology returns nothing, use **cosine-similarity fallback**: embed the concept, find top-K cache terms by similarity → weighted average of their maps. Result: one 400-D (or N-parcel) map per concept. |
| **3. Combine cognitive maps** | The scenario is "see dog + ritalin" → combine the maps for the chosen concepts (e.g. visual perception + object recognition + attention). **Simple rule:** e.g. **average** or **weighted sum** of the per-concept maps (same parcellation). That gives one **composite brain activity map** for “brain activity when you see your dog after ritalin” (functional/cognitive part). |
| **4. Optional: add receptor / drug map** | "Ritalin" (methylphenidate) mainly affects **dopamine** (and norepinephrine). Optionally fetch a **dopamine receptor** (or related) map from **neuromaps** (receptors). That map shows where ritalin is more likely to act. You can **return it alongside** the cognitive map (e.g. “cognitive map” + “receptor map”) or **overlay** it (e.g. weight the cognitive map by receptor density) for “activity map modulated by where ritalin acts.” |
| **5. Return** | Return the **composite cognitive map** (400-D or back-projected to voxel space for visualization). Optionally also return the **receptor map** and/or a short explanation (which concepts were used, cache vs ontology vs similarity). |

So end-to-end: **query → concepts → (cache / ontology / similarity) per concept → combine maps → optional receptor map → one (or two) brain map(s) for the user.**

**Full pipeline implementation:** `scripts/pipeline.py` — single entry point that runs the above steps.

- **Decompose:** `simple_rule_decomposer(query)` splits on " and ", ", ", " after ", " when ", " + " (or pass a custom `decomposer`).
- **Map per concept:** OptionAMapper (cache → ontology when OOV/low similarity → cosine-similarity fallback).
- **Combine:** `--combine neuroquery_style` (weights over cache terms → one map) or `--combine decompose_avg` (one map per concept → average).
- **Optional receptor:** `--receptor dopamine` fetches a receptor map from neuromaps and returns it alongside the cognitive map.

**CLI:**  
`python scripts/pipeline.py "working memory and attention" --decoder-cache-dir path/to/cache [--combine decompose_avg|neuroquery_style] [--use-ontology] [--receptor dopamine]`

**In code:**  
`from pipeline import run_pipeline, decompose_query, simple_rule_decomposer`  
`result = run_pipeline(query, mapper, combine="decompose_avg", decomposer=simple_rule_decomposer, neuromaps_receptor="dopamine")`  
`result.cognitive_map`, `result.concepts_used`, `result.receptor_info`.

### Two ways to combine, and how to compare them

We support **two** ways to get one map from a full query; you can run **both** and compare.

| Method | How | When to use |
|--------|-----|-------------|
| **NeuroQuery-style** | Query → **weights over cache terms** (embedding similarity or simple TF-IDF overlap) → **one** map = `weights @ term_maps`. Combination in **term space**, then one linear map to brain space. | Aligns with NeuroQuery; uses **all** cache terms the query relates to (with weights). |
| **Decompose → average** | Query → **concepts** (e.g. LLM or rules) → one map per concept (cache/ontology/similarity) → **average** maps in brain space. | Interpretable (explicit concepts); uses a **few** concepts. |

**Ontology:** Ontology is **not** used by default in NeuroQuery-style. It is an **optional** path when terms are **OOV** (not in cache) or **similarity to the cache is low**: we then use ontology to find ontologically related terms that *are* in the cache and do the mapping via those cache terms. Use `get_map_neuroquery_style_with_ontology(...)` when OOV or low similarity; otherwise use `get_map_neuroquery_style` (default, no ontology).

**Compare:** For the same query, compute both maps and report **correlation(A, B)**.  
**Script:** `scripts/compare_query_methods.py` — takes `--decoder-cache-dir`, optional test queries; prints Method A (NeuroQuery-style, default no ontology), Method A+onto (optional ontology), Method B (decompose→avg) and correlations.  
**Module:** `scripts/query_to_map.py` — `get_map_neuroquery_style(...)` (default), `get_map_neuroquery_style_with_ontology(...)` (optional ontology), `get_map_decompose_avg(...)`.

---

## 0.2 NeuroQuery+ and neuromaps (expanded vocabulary + other maps)

**Goal:** (1) Give users more capabilities by expanding the effective vocabulary (unified NeuroQuery + NeuroSynth cache, OOV → ontology/similarity). (2) Let users fetch **other** brain maps from **neuromaps** (receptors, structure, gene expression, etc.), not only cognitive (term→map) from NeuroQuery/NeuroSynth.

### Full local cache: what is included and how to get it

**The pipeline does not run with a full local cache by default.** Here is what is local vs on-demand:

| Data | In repo / local? | Notes |
|------|------------------|--------|
| **Ontologies** | Yes | `data/ontologies/` (mf.owl, uberon.owl) after `python scripts/download_ontologies.py`. |
| **Decoder cache (NeuroQuery + NeuroSynth)** | No | Repo has no `term_maps.npz` / `term_vocab.pkl`. If `--decoder-cache-dir` is missing or the dir is empty, the pipeline uses a **dummy** 5-term cache (working memory, attention, memory, language, vision) with random maps. |
| **Neuromaps** | On demand | Receptor/structure/gene maps are fetched via the `neuromaps` package on first use (downloads then caches). No "all neuromaps" pre-downloaded in this repo. |

**To run with a full local decoder cache (production):**

**Option A — One-command setup (recommended):**  
`python scripts/setup_production.py [--data-dir data]`  
This downloads ontologies, builds NeuroQuery cache (~7547 terms), NeuroSynth cache (~1300 terms), and merges to `data/unified_cache`. Use `--quick` for a fast test (200 NQ + 100 NS terms). Full build can take hours.

**Option B — Manual steps:**  
1. **Ontologies:** `python scripts/download_ontologies.py [--output-dir data/ontologies]`  
2. **NeuroQuery cache:** `python scripts/build_neuroquery_cache.py --output-dir data/neuroquery_cache [--max-terms 0]`  
3. **NeuroSynth cache:** `python scripts/build_neurosynth_cache.py --output-dir data/neurosynth_cache [--max-terms 0]`  
4. **Merge:**  
   `python scripts/merge_neuroquery_neurosynth_cache.py --neuroquery-cache-dir data/neuroquery_cache --neurosynth-cache-dir data/neurosynth_cache --output-dir data/unified_cache [--prefer neurosynth]`  
5. **Pipeline** uses `data/unified_cache` by default when present; no need to pass `--decoder-cache-dir`.

**Neuromaps:** Use `--receptor dopamine` (or similar) in the pipeline to fetch that map on demand; or use `scripts/neuromaps_fetch.py list` / `fetch` to pre-download and cache annotations locally (neuromaps stores them in its data dir after first fetch).

### Unified (term, map) cache: NeuroQuery + NeuroSynth

- **Problem:** NeuroQuery vocabulary is ~7.5k terms; NeuroSynth ~1.3k. Using only one limits coverage or meta-analytic quality.
- **Solution:** Build a **unified** term→map cache that includes **both** NeuroQuery and NeuroSynth.
  - **Vocabulary:** Union of NeuroQuery and NeuroSynth terms (one entry per normalized term).
  - **Map per term:** For terms in **both**, choose one source (e.g. **prefer NeuroSynth** for well-studied terms; else NeuroQuery). For terms in only one, use that source.
  - **Output:** Same format as decoder cache: `term_maps.npz`, `term_vocab.pkl` (parcellation, e.g. Schaefer 400, must match).
- **Script:** `scripts/merge_neuroquery_neurosynth_cache.py` — takes `--neuroquery-cache-dir`, `--neurosynth-cache-dir`, `--output-dir`, `--prefer neurosynth|neuroquery`; writes unified `term_maps.npz` and `term_vocab.pkl`. Use that output as the decoder cache for Option A / NeuroQuery+.

### Expanded vocabulary (NeuroQuery+)

- **Problem:** Even with unified cache, users may query terms **not** in the 7.5k+ vocab.
- **Solution:** At query time, represent OOV terms in the **same** vocab space and apply the same logic:
  1. **Ontology:** OOV term → related concepts → keep only in-vocab terms → weights → vector over vocab → lookup/blend cache maps.
  2. **Cosine similarity (optional):** OOV term → embed → similarity to in-vocab term embeddings → weights → blend cache maps.
- So users get **more capabilities**: same (term, map) cache, but OOV queries still return a map (ontology or similarity fallback).

### Neuromaps: receptors, structure, gene expression, etc.

- **Problem:** Users want not only **cognitive** maps (term→map from NeuroQuery/NeuroSynth) but also **receptor**, **structural**, **gene expression**, and other annotation maps.
- **Solution:** Integrate **neuromaps** ([netneurolab.github.io/neuromaps](https://netneurolab.github.io/neuromaps)) so those maps can be fetched by name/identifier.
  - **neuromaps** provides 80+ annotations: receptors (e.g. Hansen), structure, gene expression, metabolism, oscillations, cognitive specialization; multiple spaces (MNI-152, fsaverage, fsLR, CIVET).
  - **API:** `neuromaps.datasets.available_annotations()` — list by source/tags/space/format; `neuromaps.datasets.fetch_annotation()` — fetch by (source, desc, space, den).
  - **Script:** `scripts/neuromaps_fetch.py` — list (`list [--tags receptors]`), fetch one (`fetch --source <source> [--desc <desc>]`), or **fetch all** (`fetch-all [--output-dir path] [--format volumetric] [--space MNI152]`). The full set is manageable (~80–90 annotations; typically a few hundred MB to ~1 GB). Use `--format volumetric` or `--space MNI152` to limit size. Requires `pip install neuromaps`. Optionally parcellate fetched maps to Schaefer 400 for comparison with cognitive maps (separate step).
- **User flow:** "Get dopamine receptor map" → neuromaps fetch; "Get working memory map" → unified cache (or ontology fallback). So **cognitive** maps come from unified NeuroQuery+NeuroSynth cache; **receptor/structure/gene** maps come from neuromaps.

### Summary

| What | Source | How |
|------|--------|-----|
| **Cognitive (term→map)** | Unified cache (NeuroQuery + NeuroSynth) | One cache; prefer NeuroSynth on overlap. OOV → ontology/similarity → in-vocab → cache lookup. |
| **Receptors, structure, gene, etc.** | neuromaps | List via `available_annotations()`; fetch via `fetch_annotation()`. |

### Next steps checklist

1. **Install deps** (from this repo):  
   `pip install -r neurolab/requirements-enrichment.txt`  
   (numpy, obonet, networkx, rdflib, neuromaps, setuptools).

2. **Ontologies (one-time):**  
   `python scripts/download_ontologies.py`  
   → `data/ontologies/` (mf.owl, uberon.owl)

3. **Build NeuroQuery cache** (in your neurolab pipeline or with neuroquery): term_maps.npz, term_vocab.pkl (e.g. Schaefer 400).

4. **Build NeuroSynth cache** (same format, same parcellation): term_maps.npz, term_vocab.pkl in a separate dir.

5. **Merge into unified cache:**  
   `python scripts/merge_neuroquery_neurosynth_cache.py --neuroquery-cache-dir path/to/nq --neurosynth-cache-dir path/to/ns --output-dir path/to/unified [--prefer neurosynth]`

6. **Cognitive maps (Option A):**  
   `python scripts/option_a.py --decoder-cache-dir path/to/unified "term1" "term2"`  
   Or in code: `OptionAMapper(ontology_dir=..., decoder_cache_dir=path/to/unified).get_map(term)`.  
   Optional: pass `encoder` and `cache_embeddings` for cosine-similarity fallback when ontology returns no cache terms.

7. **Receptor/structure/gene maps (neuromaps):**  
   `python scripts/neuromaps_fetch.py list [--tags receptors]`  
   `python scripts/neuromaps_fetch.py fetch --source <source> [--desc <desc>] [--output-dir path]`

8. **Optional: NeuroSynth sanity check** (when both caches exist):  
   `python scripts/compare_decoder_to_neurosynth.py --decoder-cache-dir path/to/unified --neurosynth-cache-dir path/to/ns [--output results.csv]`

---

## 1. NeuroSynth sanity check (decoder / text-to-brain vs NeuroSynth)

NeuroQuery and NeuroSynth are complementary: NeuroQuery is better for rare terms and long text; NeuroSynth gives classic meta-analytic maps for well-studied terms. Comparing our pipeline to NeuroSynth on **overlapping terms** is a good sanity check.

### 1.1 What to compare

| Source A | Source B | Metric |
|----------|----------|--------|
| **Decoder cache** (NeuroQuery term → map) | **NeuroSynth** (same term → association map) | Pearson r between the two maps (same parcellation, e.g. Schaefer 400). |
| **Text-to-brain** predicted map for term | **NeuroSynth** map for same term | Pearson r. |
| **Decoder** (NeuroQuery map) | **NeuroSynth** map | Pearson r. |

For each term that appears in **both** NeuroQuery (decoder cache) and NeuroSynth:

- Load the decoder map and the NeuroSynth map in the **same parcellation** (e.g. Schaefer 400 parcels).
- Compute **Pearson correlation** between the two 400-D vectors (after optional z-scoring per map).
- Report: term, r(decoder, NeuroSynth), r(text-to-brain, NeuroSynth) if text-to-brain is used, and summary stats (mean r, median r, count of terms).

### 1.2 Requirements

- **Decoder cache** (NeuroQuery): `term_maps.npz` (N_terms × N_parcels), `term_vocab.pkl` (list of term strings).
- **NeuroSynth cache** in the **same format**: parcellated maps (e.g. Schaefer 400) for each NeuroSynth term, saved as `term_maps.npz` and `term_vocab.pkl` in a separate directory. Building this requires either:
  - A script that uses the `neurosynth` Python package to fetch association test maps (NIfTI) per term, then parcellates them to the same atlas (e.g. with `nilearn` or `neuromaps`), or
  - Pre-downloaded NeuroSynth maps parcellated to the same space.
- **Term matching**: Normalize term strings (e.g. lowercased, strip, replace `_` with space) so that "Working Memory" (NeuroQuery) matches "working memory" (NeuroSynth). Overlapping set = terms that appear in both caches after normalization.

### 1.3 How to run

Use the script `compare_decoder_to_neurosynth.py` under `neurolab/scripts/` (if present in your branch):

```bash
# From repo root
python neurolab/scripts/compare_decoder_to_neurosynth.py \
  --decoder-cache-dir path/to/decoder_cache \
  --neurosynth-cache-dir path/to/neurosynth_cache \
  [--text-to-brain-dir path/to/trained_model] \
  [--output results.csv]
```

The script expects each cache dir to contain `term_maps.npz` and `term_vocab.pkl` (same layout as the NeuroQuery decoder cache). Parcellation (e.g. Schaefer 400) must match between decoder and NeuroSynth caches.

Output: table of overlapping terms with r(decoder, NS), optional r(text-to-brain, NS), and summary statistics.

### 1.4 Interpretation

- **High mean r** (e.g. > 0.5) on overlapping terms: decoder/text-to-brain and NeuroSynth agree well for well-studied terms; pipeline is coherent.
- **Low mean r**: check parcellation alignment, normalization (z-score vs raw), and that NeuroSynth maps are association (or uniformity) maps in the same space.
- Use this as a **sanity check**, not a single metric: NeuroSynth and NeuroQuery can differ somewhat even for the same term (different methods: meta-analysis vs predictive model).

---

## 2. Expanding semantic space while staying accurate

NeuroQuery uses a **fixed vocabulary** of ~7,547 neuroscience terms and TF-IDF (+ NMF smoothing). That limits:

- **Out-of-vocabulary** concepts (terms not in the 7,547).
- **Nuance** beyond what co-occurrence in that corpus captures.

Our **text-to-brain** model already extends the *query* side (any text → embedding → map), but the **training targets** (maps) are still (term, map) pairs from the decoder cache. To expand semantic space **and** stay accurate, consider the following (in order of impact vs effort).

### 2.1 Use full NeuroQuery vocabulary and merge NeuroSynth terms (recommended first step)

- **Full NeuroQuery vocab:** Build the decoder cache with **all** NeuroQuery terms (e.g. `--max-terms 0` or 7547), not a cap of 2k–5k. That maximizes the number of (term, map) pairs for training and decoding.
- **Merge NeuroSynth into the cache:** For terms that appear in **both** NeuroQuery and NeuroSynth, you can:
  - **Option A:** Keep NeuroQuery only (simplest).
  - **Option B:** Prefer **NeuroSynth** for well-studied terms (higher meta-analytic validity) and **NeuroQuery** for the rest. Build a **unified cache**: term → map, where map is NeuroSynth if available, else NeuroQuery. Then train text-to-brain on this unified cache. That expands coverage (NeuroQuery’s 7.5k terms) and can **improve accuracy** for the overlapping terms (NeuroSynth as ground truth where it’s strong).
- **Term normalization:** Use the same normalization when merging (lowercase, strip, maybe replace underscores with spaces) so "working_memory" and "working memory" map to the same key.

**Accuracy:** High. You’re only adding more (term, map) pairs from trusted sources; for overlaps you choose the better source (e.g. NeuroSynth for well-studied terms).

### 2.2 Domain-adapted or larger text encoder (moderate effort)

- **Current:** Generic sentence encoder (e.g. `all-mpnet-base-v2`) maps any text to an embedding; text-to-brain maps embedding → brain map.
- **Expand:** Use a **biomedical/neuroscience** encoder (e.g. SciBERT, BioBERT, or a model fine-tuned on neuroscience abstracts) so the **embedding space** is better aligned with neuroscience terminology. That can improve generalization for domain terms not seen in training.
- **Caution:** Validate on held-out terms and on the NeuroSynth sanity check; a domain encoder can help or hurt depending on data size and domain shift.

**Accuracy:** Validate empirically. Often a small gain on domain terms; sometimes little change if the generic encoder already covers the training terms well.

### 2.3 Ontology-based expansion using canonical open-source ontologies (recommended)

- **Idea:** Map novel terms to NeuroQuery (or NeuroSynth) terms via **canonical, open-source ontologies** that can be **downloaded and used locally**. For a query term not in the cache, look up related terms in the ontology (synonyms, parents, children) and use the **average** of their maps (or the closest term’s map) as a proxy. That expands the *effective* vocabulary without new (term, map) training.
- **Accuracy:** Depends on ontology quality and how you aggregate (mean map vs single closest term). Use as a fallback; prefer trained text-to-brain when the query is in-distribution. See **§3** for which ontologies to use and how to run them locally (SOTA, open, downloadable).

### 2.4 Semi-supervised or pseudo-labeling (higher risk)

- **Idea:** Use the trained text-to-brain model to generate maps for a **larger set of terms** (e.g. from Cognitive Atlas or MeSH), then add (term, predicted_map) as soft targets and retrain with confidence weighting. That expands the training set but can **propagate errors**.
- **Accuracy:** Risky. Only consider with a strong validation protocol (e.g. hold out a subset with known ground truth, or use NeuroSynth overlap as a gate).

### 2.5 Summary: recommended order

1. **Merge NeuroSynth + NeuroQuery** into one cache (NeuroSynth for overlapping well-studied terms, NeuroQuery for the rest); use **full NeuroQuery vocabulary** for the decoder cache. Re-train text-to-brain on the unified cache. Run the **NeuroSynth sanity check** to confirm agreement.
2. **Run the sanity-check script** regularly (e.g. after every cache or model change) on overlapping terms.
3. Optionally try a **domain encoder** and re-run validation.
4. Use **ontology-based** expansion (§2.3, §3) as a fallback for out-of-vocabulary terms; avoid aggressive pseudo-labeling unless you have a clear validation strategy.

---

## 3. Canonical open-source ontologies (local, SOTA)

Use **open-source ontologies** that are **downloadable locally**, **canonical** in neuroscience/cognitive domains, and **SOTA** (actively maintained, OBO Foundry or equivalent). All of the following can be downloaded as OWL/OBO and used offline.

### 3.1 Recommended ontologies

| Ontology | What it covers | Download (local) | Format | License |
|----------|----------------|------------------|--------|--------|
| **Cognitive Atlas** | Cognitive concepts, tasks, disorders, contrasts; purpose-built for cognitive neuroscience. | [GitHub: CognitiveAtlas/ontology](https://github.com/CognitiveAtlas/ontology), [poldracklab/cogat](https://github.com/poldracklab/cogat) (OWL generator). API: [cognitiveatlas.org](https://cognitiveatlas.org). | OWL, JSON (API) | MIT |
| **Mental Functioning Ontology (MF)** | Mental functioning, cognitive processes, traits, mental disorders; BFO-aligned. | [purl.obolibrary.org/obo/mf.owl](http://purl.obolibrary.org/obo/mf.owl), [GitHub: jannahastings/mental-functioning-ontology](https://github.com/jannahastings/mental-functioning-ontology) | OBO, OWL | CC BY 3.0 |
| **NIFSTD (NIF Standard Ontology)** | Neuroscience standard: brain regions (nlx.br), cells, molecules, NeuroLex. | [GitHub: SciCrunch/NIF-Ontology](https://github.com/SciCrunch/NIF-Ontology) (TTL, OWL), [BioPortal NIFSTD](https://bioportal.bioontology.org/ontologies/NIFSTD) | OWL, TTL, CSV | CC-BY-4.0 |
| **UBERON** | Cross-species anatomy including brain regions; widely used. | [purl.obolibrary.org/obo/uberon.owl](http://purl.obolibrary.org/obo/uberon.owl), [uberon.github.io/downloads](https://uberon.github.io/downloads.html) | OBO, OWL | CC-BY-4.0 |

**Why these:** Cognitive Atlas and MF are the main **cognitive/mental** ontologies; NIFSTD and UBERON cover **brain anatomy and neuroscience** terminology. All are open, downloadable, and used in practice (NeuroSynth/NeuroQuery papers cite Cognitive Atlas; OBO Foundry lists MF and UBERON).

### 3.2 Optional: EBI Ontology Lookup Service (OLS) for local deployment

- **OLS4** ([ebi.ac.uk/ols4](https://www.ebi.ac.uk/ols4), [GitHub: EBISPOT/ols4](https://github.com/EBISPOT/ols4)) hosts many biomedical ontologies and supports **downloads** (OWL) and **local deployment** (Docker) for offline lookup.
- Use OLS **downloads** to fetch OWL files for the ontologies above (or others) and load them locally, or run OLS in Docker for a full local API.

### 3.3 Quick start (download + load once, then use fallback)

1. **Download** ontologies once (into `neurolab/data/ontologies`):

   ```bash
   python scripts/download_ontologies.py
   # Saves mf.owl and uberon.owl to data/ontologies/
   # Use --no-uberon to skip UBERON (~93 MB). Use --output-dir path to change location.
   ```

2. **Load index once at startup** and use **ontology fallback** when a term is not in the decoder cache:

   ```python
   from pathlib import Path
   from ontology_expansion import load_ontology_index, get_map_for_term
   import numpy as np

   # Paths (adjust to your layout)
   ONTOLOGY_DIR = Path(__file__).resolve().parent / "data" / "ontologies"
   DECODER_CACHE_DIR = Path("path/to/decoder_cache")  # term_maps.npz, term_vocab.pkl

   # Load once
   ontology_index = load_ontology_index(ONTOLOGY_DIR)
   term_maps = np.load(DECODER_CACHE_DIR / "term_maps.npz")["term_maps"]  # or key from your cache
   with open(DECODER_CACHE_DIR / "term_vocab.pkl", "rb") as f:
       import pickle; term_vocab = pickle.load(f)

   def get_map(term: str):
       return get_map_for_term(term, term_maps, term_vocab, ontology_index)

   # For any query term: cache hit or ontology-expanded map
   map_400 = get_map("working memory")   # from cache if present
   map_400 = get_map("prosopagnosia")    # from ontology-related cache terms if OOV
   ```

   See also `scripts/query_with_ontology_fallback.py` for a runnable example.

### 3.4 How to use them locally (manual download)

1. **Download** the OWL/OBO files manually (e.g. to `data/ontologies/`) if not using `download_ontologies.py`:

   | Ontology | Direct download (save as in `data/ontologies/`) |
   |----------|--------------------------------------------------|
   | **Mental Functioning (MF)** | `http://purl.obolibrary.org/obo/mf.owl` → `mf.owl` |
   | **UBERON** | `http://purl.obolibrary.org/obo/uberon.owl` → `uberon.owl` (or use `uberon.obo` from [downloads](https://uberon.github.io/downloads.html)) |
   | **NIFSTD** | Clone [SciCrunch/NIF-Ontology](https://github.com/SciCrunch/NIF-Ontology) and copy TTL/OWL, or download from [BioPortal NIFSTD](https://bioportal.bioontology.org/ontologies/NIFSTD) |
   | **Cognitive Atlas** | Export from [cognitiveatlas.org](https://cognitiveatlas.org) or build OWL from [CognitiveAtlas/ontology](https://github.com/CognitiveAtlas/ontology) / [poldracklab/cogat](https://github.com/poldracklab/cogat) |

2. **Load in Python** using the provided script (recommended):
   - **Script:** `neurolab/scripts/ontology_expansion.py`
   - **Libraries:** `obonet` and `networkx` for OBO; optional `rdflib` for OWL.
   - **Usage:** `load_ontology_index(ontology_dir)` → index; `expand_term(term, decoder_vocab, index)` → list of (cache_term, weight); `get_map_for_term(term, decoder_maps, term_vocab, index)` → 400-D map (average of related cache maps) when term is out-of-vocabulary.

3. **Ontology-based expansion (fallback):**
   - User query term **T** not in decoder cache.
   - Look up **T** in the ontologies (exact or fuzzy match on labels/synonyms).
   - If found, get **related terms** (synonyms, parent/child labels, or same-branch terms).
   - Resolve related terms to **decoder cache terms** (normalize and intersect with `term_vocab`).
   - Return **average map** of those cache terms (or weighted by ontology distance), or the single closest cache term’s map.

4. **Term normalization:** Use the same normalization as decoder/NeuroSynth (lowercase, strip, `_` → space) when matching ontology labels to cache terms.

### 3.5 Implementation

- **Script:** `neurolab/scripts/ontology_expansion.py`
- **Inputs:** Path to directory of downloaded OWL/OBO files; decoder cache (`term_maps.npz`, `term_vocab.pkl` or term list).
- **Outputs:** `expand_term(term, decoder_vocab, index)` → list of (cache_term, weight); `get_map_for_term(term, decoder_maps, term_vocab, index)` → 400-D map (weighted average of related cache maps) when term is not in cache.
- **Libraries:** `obonet` and `networkx` for OBO (recommended); optional `rdflib` for OWL. No running OLS server required when files are local.
- **Download script:** `scripts/download_ontologies.py` — downloads MF and UBERON to `data/ontologies/` (use `--no-uberon` to skip UBERON; `--output-dir path` to change location).
- **CLI:** `python ontology_expansion.py path/to/ontologies [term]` prints number of loaded labels and optionally related terms. `python query_with_ontology_fallback.py [--decoder-cache-dir path] term1 term2` runs the full flow (load index once, get_map with ontology fallback).

### 3.6 Summary

- **Cognitive Atlas** + **MF** + **NIFSTD** + **UBERON** give broad coverage of cognitive, mental, and brain-anatomy terms; all are open, downloadable, and SOTA.
- Download OWL/OBO once; load locally with `obonet`/`rdflib`/`owlready2`; use for **term → related terms → decoder cache terms → average map** as a fallback when the query is out-of-vocabulary.

---

## 4. References

- NeuroSynth: [neurosynth.org](https://neurosynth.org), [GitHub](https://github.com/neurosynth/neurosynth)
- NeuroQuery: [neuroquery.org](https://neuroquery.org), Dockès et al. (2020) eLife
- Cognitive Atlas: [cognitiveatlas.org](https://cognitiveatlas.org), [GitHub CognitiveAtlas/ontology](https://github.com/CognitiveAtlas/ontology)
- Mental Functioning Ontology: [OBO Foundry MF](https://obofoundry.org/ontology/mf.html), [purl.obolibrary.org/obo/mf.owl](http://purl.obolibrary.org/obo/mf.owl)
- NIFSTD: [GitHub SciCrunch/NIF-Ontology](https://github.com/SciCrunch/NIF-Ontology), [NIF Vocabularies](https://neuinfo.org/about/nifvocabularies)
- UBERON: [purl.obolibrary.org/obo/uberon](http://purl.obolibrary.org/obo/uberon), [downloads](https://uberon.github.io/downloads.html)
- OLS: [EBI OLS4](https://www.ebi.ac.uk/ols4), [OLS4 downloads](https://www.ebi.ac.uk/ols4/downloads)
- This doc links to the **accuracy-and-testing** and **implementation guide** in the NeuroLab repo (when present) for metrics and reproduction steps.
