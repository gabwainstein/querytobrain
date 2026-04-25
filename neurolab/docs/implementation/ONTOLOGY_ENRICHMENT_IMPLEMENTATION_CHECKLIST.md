# Ontology Enrichment + Training Implementation Checklist

**Design principle:** *Ontology = semantics; maps = evidence.*  
Use ontologies to organize meaning, normalize language, and regularize similarity. Anchor ground-truth maps to real sources (NeuroVault, NeuroSynth, NeuroQuery, ENIGMA, neuromaps, abagen). Do not fabricate synthetic targets.

---

## 1. Current State

| Component | Status | Location |
|-----------|--------|----------|
| Ontology loading | ✅ | `download_ontologies.py`, `ontology_expansion.py` |
| Cognitive Atlas, CogPO | ✅ | `--cognitive-atlas`, `--cogpo` in download |
| MONDO, HPO, ChEBI | ✅ | `--clinical` in download |
| OBO Relations (RO) | ✅ | `--extra` or `--ro` in download |
| Term expansion (label → related cache terms) | ✅ | `expand_term` in ontology_expansion |
| KG context augmentation (Role 1) | ✅ | `get_kg_context`, `get_kg_augmentation` in train |
| Ontology meta-graph (cross-ontology) | ✅ | `ontology_meta_graph.py` |
| Ontology retrieval augmentation | ✅ | `use_ontology_retrieval_augmentation` in text_to_brain |
| **Term → ontology ID resolution** | ⚠️ Partial | Label substring match; no explicit ID mapping |
| **Ontology similarity regularizer (Role 2)** | ❌ | Not implemented |

---

## 2. Canonicalization: "n-back → working memory"

### 2.1 Goal

Resolve training labels and queries to ontology nodes (IDs) so "n-back", "2-back", "WM updating" all map to the same concept (e.g. Cognitive Atlas `working_memory` or CogPO task).

### 2.2 Injection Points

| Where | What to add |
|-------|-------------|
| **Cache builders** | `term_ontology_ids.pkl` (optional): term_id → list of (ontology_id, label, source) |
| **build_expanded_term_maps** | When building term_vocab, resolve each term to ontology nodes via label match; store in `term_ontology_ids.pkl` (optional, for regularization) |
| **relabel_terms_with_llm** | LLM output can be canonicalized post-hoc: resolve to Cognitive Atlas / CogPO when possible |

### 2.3 Resolution Logic

```python
# Pseudocode: resolve term to ontology IDs
def resolve_to_ontology_ids(term: str, ontology_index: dict) -> list[tuple[str, str, str]]:
    """Return [(ontology_id, label, source), ...] for matching nodes."""
    # 1. Substring match: term in ontology label or label in term
    # 2. Synonym match: term matches synonym of any node
    # 3. Return normalized label + source for each match
    ...
```

**Use OBO Relations:** When traversing relations, use `is_a`, `part_of`, `has_synonym` from RO. `ontology_expansion` already uses relation types; ensure they align with RO where possible.

### 2.4 Checklist

- [ ] Add `resolve_term_to_ontology_ids(term, index)` in ontology_expansion
- [ ] Optionally write `term_ontology_ids.pkl` in build_expanded_term_maps when `--save-term-sources`
- [ ] Use resolution in `get_kg_context` / `get_kg_augmentation` so "n-back" → "working memory" context is included

---

## 3. Role 1: Retrieval / Context Augmentation (Already Implemented)

**Current:** `--kg-context-hops N` appends `| measures: working memory | related: executive function` to the term before encoding.

**Enhancement:** Ensure Cognitive Atlas + CogPO are in ontology_dir when building for cognitive/psychopharmacology tasks. Run:

```bash
python neurolab/scripts/download_ontologies.py --cognitive-atlas --cogpo --clinical
```

**Checklist:** No code changes; verify ontology set includes Cognitive Atlas and CogPO for "n-back → working memory" coverage.

---

## 4. Role 2: Ontology Similarity Regularizer (New)

**Idea:** Instead of fabricating maps for ontology neighbors, add a soft constraint:  
*If two training terms are close in the ontology graph, their predicted maps should be more similar than two distant terms.*

### 4.1 Loss Term

```
L_reg = λ * Σ_{(i,j) in ontology_pairs} w_ij * (1 - corr(pred_i, pred_j))
```

Where `w_ij` = weight from ontology distance (e.g. `gamma^path_length` or relation type weight).  
Minimize `L_reg` so close ontology terms get similar predictions.

### 4.2 Implementation

1. **Build ontology pairs:** For each training term, get its ontology neighbors (up to 2 hops). Pairs = (term_i, term_j) with weight w_ij.
2. **Sample pairs per batch:** Add a subset of ontology pairs to each batch (or a separate regularization pass).
3. **Compute loss:** For each pair, `pred_i = model(embed_i)`, `pred_j = model(embed_j)`. Loss = `w_ij * (1 - corr(pred_i, pred_j))`.
4. **Total loss:** `L = L_mse + λ * L_reg`.

### 4.3 Checklist

- [ ] Add `get_ontology_pairs(terms, term_ontology_ids, index, max_pairs_per_term)` returning `[(i, j, w_ij), ...]`
- [ ] Add `--ontology-regularizer-lambda` (default 0) to train_text_to_brain_embedding
- [ ] When λ > 0: each batch, sample ontology pairs; compute L_reg; add to total loss
- [ ] Start with λ = 0.01–0.1; tune via validation

---

## 5. Metadata in Term Creation (Pharmacology / Dose)

**Already done:** `build_neurovault_cache` injects `drug=X | dose=Y | placebo-controlled` for dose-related collections.

**Enhancement:** Ensure `relabel_terms_with_llm` receives full context:

- [x] `collection_name`, `collection_description` (done)
- [x] `map_kind`, `dose_related` (done)
- [ ] Optional: pass parsed `drug`, `dose_mg`, `design` as structured fields to LLM prompt

**Checklist:** Consider adding explicit structured fields to the relabel prompt for pharmacology collections.

---

## 6. Typed Heads (ROI / Structural / Biology)

**From NOOTROPICS_BIOHACKER_DESIGN.md:**

- [ ] Add separate structural head; route ENIGMA to it
- [ ] Add separate biology head; route neuromaps + receptor to it
- [ ] Keep ROI/mask (2508, 437) excluded from regression; retrieval only

---

## 7. Evaluation: Canonical Query Set

**Human sanity panel:** 30–50 canonical queries scored by domain experts.

**Suggested queries:**  
`V1`, `language comprehension`, `reward`, `theory of mind`, `amygdala fear`, `working memory`, `n-back`, `dopamine`, `serotonin receptor`, `cortical thickness schizophrenia`, `ibuprofen 600 mg emotion task`, etc.

**Checklist:**

- [ ] Create `neurolab/data/canonical_queries.json` with query + expected map_type
- [ ] Add script `eval_canonical_queries.py` that runs each query, computes similarity to reference maps (when available), and reports by map_type

---

## 8. Priority Order

1. **Verify ontology set** includes Cognitive Atlas + CogPO for cognitive coverage.
2. **Add ontology regularizer** (Role 2) — highest impact for "n-back ↔ working memory" without fabricating targets.
3. **Add term resolution** — `resolve_term_to_ontology_ids` for explicit canonicalization.
4. **Separate heads** — structural, biology (from NOOTROPICS_BIOHACKER_DESIGN).
5. **Canonical query eval** — fixed set for reproducibility.
