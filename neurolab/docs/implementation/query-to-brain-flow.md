# Query → Brain map: flow and paths

How a text query becomes a 400-D parcellated brain map, and how **semantic similarity** (to the cache vocabulary) and **biological term matching** (neuromaps) decide which path is taken.

---

## Top-level: two entry modes

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  Query (e.g. "attention", "myelin density", "alpha2a")   │
                    └───────────────────────────┬─────────────────────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │  --use-embedding-model DIR set?                │
                        └───────────────────────┬───────────────────────┘
                                  YES │                    │ NO
                                      ▼                    ▼
                    ┌─────────────────────────┐  ┌─────────────────────────────────────────┐
                    │  EMBEDDING PATH         │  │  COMBINED TERM → MAP (default)            │
                    │  TextToBrainEmbedding   │  │  1) Neuromaps by name (receptors,       │
                    │  (expandable term space)│  │     myelin, RNA, PET, etc.)               │
                    │  → 400-D map directly   │  │  2) Cognitive (cache + ontology)          │
                    └─────────────────────────┘  └─────────────────────────────────────────┘
```

- **Embedding path**: With `--use-embedding-model`, the query is passed through the trained text→brain model only. No cache weights, no ontology. Optional scope guardrail can block out-of-scope terms.
- **Combined path** (default when not using embedding): First try **neuromaps by name**: if the query matches a biological annotation label (e.g. "myelin density", "alpha2a PET receptor expression", "RNA receptor map"), return that map. Otherwise use the **cognitive path** (cache + ontology). Use `--no-combined` to skip neuromaps-by-name and use cognitive only.
- **Cognitive path**: Map from **cognitive cache** (term_maps + term_vocab). Similarity to cache terms is computed; below a threshold, **ontology** can be used to expand the query to related cache terms.

---

## Combined term → map: neuromaps + cognitive

When **not** using the embedding model and **combined** mode is on (default), the pipeline resolves the map in two steps:

1. **Neuromaps by name**: Match the query to neuromaps cache labels (e.g. `hansen_5ht2a`, `bigbrain_myelin`). Query terms like "alpha2a PET receptor expression", "myelin density", "RNA receptor map" are matched via token overlap and biological aliases (e.g. alpha2a, 5-HT2a, PET, receptor, myelin, RNA). If a label matches above a minimum score, return that annotation map (400-D) and **source = neuromaps**.
2. **Cognitive path**: If no neuromaps match, proceed as in **Cache path** below (cache weights + ontology on low similarity).

So you can **query biological terms by text** and get the corresponding map (receptors, myelin, gene expression, etc.) without going through the cognitive decoder. Script: `neurolab/scripts/term_to_map.py`; `--list-neuromaps` lists available labels.

---

## Cache path: how similarity chooses the route

When **not** using the embedding model, the pipeline does:

1. Load **term_maps** (n_terms × 400) and **term_vocab** from the cache (e.g. `unified_cache`).
2. Compute **weights** over cache terms for the query:
   - If an encoder + cache_embeddings were provided: **cosine similarity** (query embedding vs each cache term embedding), then normalized to sum to 1.
   - Otherwise (default in `query.py`): **TF-IDF-style** overlap (query tokens vs each term’s tokens), normalized.
3. **max_sim = max(weights)** (direct similarity to cache terms).
4. Compare **max_sim** to **similarity_threshold** (threshold A; default **0.15**).

When direct similarity is low, ontology expansion is tried. A second threshold (**threshold B**, `--similarity-threshold-ontology`, default same as A) ensures we only use ontology when the **ontology-derived** cache terms are still "close enough" to the query: we require that the query's similarity to at least one of those terms is **≥ threshold B**. So: direct similarity low, but ontology terms are stronger (above B) → use ontology with those weights.

Resulting behavior:

```
  max_sim ≥ threshold A (e.g. ≥ 0.15)
       → Cache path:  map = weights @ term_maps
       → “Enough” similarity to cache terms; no ontology.

  max_sim < threshold A  AND  ontology available  AND  expand_term(query) returns cache terms
       → Let max_sim_ontology = max(weights[i] for i in ontology-derived term indices).
       → If max_sim_ontology ≥ threshold B:  Ontology path:  map = ontology_weights @ term_maps
       → “Used ontology fallback (low similarity to cache terms).”

  max_sim < threshold A  AND  (no ontology OR expand_term returns nothing)
       → Cache path:  map = weights @ term_maps
       → No ontology used; map may be weak/noisy.
```

So **semantic similarity** (here: max weight over cache terms) decides:

- **Direct high** (≥ threshold A): use cache-term weights → **cache path**.
- **Direct low** (< A): try **ontology**. Use **ontology path** only if the query’s similarity to the ontology-derived cache terms is **≥ threshold B**; otherwise keep the original cache weights.

---

## Flowchart (cache path only)

```
  Query
    │
    ▼
  Load term_maps, term_vocab (unified_cache / decoder_cache)
    │
    ▼
  Compute weights over cache terms
  (TF-IDF token overlap, or encoder cosine similarity if provided)
    │
    ▼
  max_sim = max(weights)
    │
    ├── max_sim ≥ similarity_threshold (0.15) ──────────────────►  map = weights @ term_maps
    │                                                              (cache path; no ontology)
    │
    └── max_sim < similarity_threshold
            │
            ▼
        Ontology dir present and --no-ontology not set?
            │
            ├── No ─────────────────────────────────────────────►  map = weights @ term_maps
            │                                                       (cache path; no ontology)
            │
            └── Yes
                    │
                    ▼
                expand_term(query, term_vocab, ontology_index)
                    │
                    ├── Returns related (cache_term, weight) list
                    │       │
                    │       ▼
                    │   max_sim_ontology = max(weights[i] for i in ontology term indices)
                    │       │
                    │       ├── max_sim_ontology ≥ similarity_threshold_ontology (B) ──►  map = ontology_weights @ term_maps
                    │       │   (ontology path; "Used ontology fallback (low similarity to cache terms).")
                    │       │
                    │       └── max_sim_ontology < B ──►  map = weights @ term_maps
                    │           (ontology terms not close enough; cache path)
                    │
                    └── Returns nothing
                            │
                            ▼
                        map = weights @ term_maps
                        (fallback to cache path with original low weights)
```

---

## After the map: enrichment

Once the 400-D parcellated map is obtained (by any path), it is passed to **UnifiedEnrichment**:

- **Cognitive**: correlate with cache terms (and/or other cognitive maps) → top-N terms, summary.
- **Biological**: if neuromaps cache is present, correlate with receptor/annotation maps → top biological hits.

So “query → brain” is: **query → (embedding or cache/ontology) → 400-D map → enrichment**.

---

## Summary table

| Condition | Path | Map construction |
|-----------|------|-------------------|
| `--use-embedding-model` set | Embedding | TextToBrainEmbedding(term) → 400-D |
| No embedding; **combined**; query matches neuromaps label | Neuromaps | annotation map from neuromaps_cache (e.g. receptor, myelin, RNA) |
| No embedding; (combined with no neuromaps match, or --no-combined); max_sim ≥ threshold A | Cache | weights @ term_maps (weights from TF-IDF or encoder) |
| No embedding; …; max_sim < A; ontology; expand_term gives terms; max_sim_ontology ≥ threshold B | Ontology | ontology_weights @ term_maps |
| No embedding; …; max_sim < A; (no ontology, or expand_term empty, or max_sim_ontology < B) | Cache (or fallback) | weights @ term_maps |

If the cache path fails (e.g. missing cache files), `query.py` falls back to **NeuroQuery** (online model) to produce the brain map, then runs enrichment on that.
