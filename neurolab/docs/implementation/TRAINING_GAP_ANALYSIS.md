# Training Gap Analysis: Master Plan vs Current Implementation

**Date:** 2026-02-14  
**Purpose:** Deep read of master plan and docs to identify what training *should* do vs what it *does*, and what must be fixed before training is ready.

---

## 1. How Training Is Supposed to Work (Master Plan)

### 1.1 Data Preparation (Section 5.2)

**Ontology-based data multiplication** — for each (term, brain_map) pair, generate multiple training pairs:

| Version | What | Purpose |
|---------|------|---------|
| A | Raw embedding | Base term → map |
| B | KG-augmented embedding | Term + ontology triples (parent, child, synonym) → same map |
| C | Synonym embeddings | Each synonym from ontology → same map |
| D | Related-term embeddings | Parent/child, 1-hop relations → same map |

**Effective training set:** ~2–3× base count = **~50K–80K pairs** (not 14K raw terms).

**Critical:** ALL embeddings are **pre-computed and cached** as numpy arrays. **Zero API calls during training.**

### 1.2 Embedding Pipeline (Section 3.3)

```
Build time (once):
  1. Embed all ~60K ontology terms → label_embeddings.npz (~350MB)
  2. For each training term:
     a. Find top-k ontology neighbors via cosine sim against label_embeddings
     b. Format augmented text (term + KG triples from neighbors)
     c. Embed augmented text
     d. Embed all synonyms and close ontological relations
     e. Save everything → training_embeddings.npz (all variants per term)
  3. Train Generalizer on all embedding variants (zero API calls during training)
  4. Train Memorizer on all embedding variants (zero API calls during training)
```

### 1.3 Model Architecture (Section 5.3–5.4)

**Two separate networks**, not one MLP:

| Network | Purpose | Architecture | Training |
|---------|---------|--------------|----------|
| **Generalizer** | Generalize to novel queries | PC-constrained dual-head MLP (PC head + residual head + gate) | Regularized: dropout 0.3, weight decay 1e-4, LayerNorm. Phased schedule: λ_residual 1.0 → 0.1 → 0.01 |
| **Memorizer** | Perfect reproduction + interpolation | Overfit MLP (1536→1024→1024→392) | NO dropout, NO weight decay. 500+ epochs until train loss ≈ 0 |

**Inference:** Cosine sim vs training embeddings → routing α = sigmoid(20×(best_sim − 0.75)) → blend `(1−α)×gen_map + α×mem_map`.

### 1.4 Weight System (Docs)

- **Per-term sample weights** (`term_sample_weights.pkl`): meta 2×, group 1×, subject-averaged 0.8× (NeuroVault); pharma 1.2×; source defaults (direct 1.0, neurovault 0.8, ontology 0.6, abagen 0.4, etc.).
- **Source-weighted batch sampling** (`SOURCE_SAMPLING_WEIGHTS`): target fraction per source so batches have ~direct 30%, neurovault 30%, ontology 15%, abagen 10%, etc.
- **Loss:** Each sample’s MSE is multiplied by its weight before summing.

### 1.5 Ontology Usage

- **Training-time:** Data multiplication (synonyms, related terms, KG-augmented text) + optional KG context appended to text before encoding (`--kg-context-hops 1` or `2`).
- **merged_sources:** Can be built with `--no-ontology` (current) or WITH ontology (adds ontology-derived terms).
- **decoder_cache_expanded:** Built with `--ontology-dir` to add ontology terms with derived maps.

---

## 2. Current Implementation vs Master Plan

| Aspect | Master Plan | Current State | Gap |
|--------|-------------|---------------|-----|
| **Model** | Dual network (Generalizer + Memorizer) | Single MLP (or MLP + gene head) | No dual network; no routing; no PC-constrained head with gate |
| **Data multiplication** | Raw + KG-augmented + synonyms + related (2–3× terms) | Raw terms only (~14K) | No ontology expansion in merged_sources; no embedding variants |
| **Pre-computed embeddings** | All variants cached; zero API during training | On-the-fly or `--use-cached-embeddings` (base terms only) | No ontology-augmented or synonym/related variants in cache |
| **Ontology in training** | Ontology terms + KG context in text | merged_sources built with `--no-ontology`; KG context off by default | Ontologies present but not used for expansion or KG context |
| **Weight system** | Per-term + source-weighted sampling in loss | PyTorch: weights used when no gene head; TF-IDF: **ignored**; gene head: **ignored** | TF-IDF and gene-head paths do not apply sample weights |
| **PC-constrained head** | Generalizer predicts 15 PC coefs + residual + gate | Gene head predicts abagen PC loadings (separate); no PC head for main terms | Different architecture |
| **Phased training** | λ_residual schedule (1.0 → 0.1 → 0.01) | None | Not implemented |
| **Memorizer** | Separate overfit network, 500+ epochs | Simulated by low-reg hyperparams in single MLP | No dedicated Memorizer network |

---

## 3. What Must Be Done for Training to Match the Plan

### 3.1 Immediate Fixes (Blockers) — **DONE (2026-02-14)**

1. **Weight system** — **DONE**
   - TF-IDF path: pass `sample_weight` to `MLPRegressor.fit()` (sklearn supports it).
   - Gene-head path: apply `train_weights` to `loss_main` and `loss_gene` (weighted MSE for both).
   - PyTorch retrain-on-train+val: also uses weighted loss.
   - sklearn fallback: both initial fit and retrain use sample_weight.

2. **Ontology integration** — **DONE**
   - `--ontology-dir` now defaults to `neurolab/data/ontologies`.
   - KG context: `--kg-context-hops 1` (or `2`) uses the default ontology dir when available.
   - To enable: `python train_text_to_brain_embedding.py ... --kg-context-hops 1`

### 3.2 Full Master Plan Alignment (Larger Effort)

1. **Data multiplication**
   - Script to generate all embedding variants (raw, KG-augmented, synonyms, related) per term.
   - Save to `training_embeddings.npz` + metadata (which variant maps to which base term).
   - `build_training_embeddings.py` exists but does not produce ontology-augmented variants.

2. **Dual-network architecture**
   - Implement `Generalizer` (PC head + residual head + gate, LayerNorm, phased λ_residual).
   - Implement `Memorizer` (overfit MLP, no reg).
   - Train both on same (or variant-expanded) data.
   - Implement inference: cosine sim → α → blend.

3. **PC-constrained Generalizer**
   - Load gene PCA from `gene_pca/` (pc_scores, 15 PCs).
   - PC head: predict 15-dim coefs → `map = pc_scores @ coefs`.
   - Residual head: predict 392-dim correction.
   - Gate: learned blend of PC vs residual.

4. **Phased training schedule**
   - Phase A (epochs 1–50): λ_residual=1.0 (force PC pathway).
   - Phase B (50–100): λ_residual=0.1.
   - Phase C (100+): λ_residual=0.01.

---

## 4. Recommended Path Forward

### Option A: Minimal (Get Current Pipeline Correct)

1. Fix weight system (TF-IDF + gene head).
2. Add `--kg-context-hops 1` to training when ontology dir exists.
3. Optionally build expanded cache with ontology and train on it.
4. Document that current model is a single-MLP baseline, not the full dual network.

### Option B: Full Master Plan

1. Implement data multiplication (embedding variants).
2. Implement Generalizer + Memorizer + routing.
3. Implement PC-constrained head with phased schedule.
4. Build expanded cache with ontology.
5. Pre-compute all embeddings (including variants).
6. Train both networks; tune routing on held-out set.

---

## 5. Verification Commands (After Fixes)

```bash
# 1. Weights: confirm term_sample_weights used in all paths
python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources --encoder tfidf --max-terms 100 --epochs 2
# Expect: "Using per-term sample weights" and loss reflects weights (compare to run with --no-source-weighted-sampling)

# 2. KG context: confirm ontology triples appended
python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources --ontology-dir neurolab/data/ontologies --kg-context-hops 1 --encoder tfidf --max-terms 50 --epochs 1
# Expect: "KG context: 1 hop" or similar; terms have appended triples

# 3. Expanded cache with ontology
python neurolab/scripts/build_expanded_term_maps.py --cache-dir neurolab/data/unified_cache --ontology-dir neurolab/data/ontologies --output-dir neurolab/data/merged_sources_expanded --save-term-sources --neurovault-cache-dir ... [full merge args]
# Then train on merged_sources_expanded
```

---

## 6. Summary

**Training is not ready** because:

1. **Weight system** is incomplete (TF-IDF and gene head ignore weights).
2. **Ontology** is not used (merged_sources has no ontology terms; KG context is off).
3. **Architecture** is a single MLP, not the dual Generalizer + Memorizer with routing.
4. **Data multiplication** (embedding variants) is not implemented.

**Minimum to be "ready" in the documented sense:** Fix weights and enable ontology (KG context or expanded cache). The full dual-network architecture is a larger implementation effort described in the master plan.
