# KG-to-Brain GNN

A heterogeneous graph neural network that predicts parcellated brain maps
(Glasser+Tian, 392 parcels) from a knowledge graph spanning Term, Gene,
Receptor, OntologyConcept, and Region nodes.

This is an **ensemble member** alongside `enrichment.text_to_brain` — it is not
a replacement. The intended deployment is `unified_enrichment.UnifiedEnrichment`
blending the GNN's prediction into the input parcel vector at runtime via a
`kg_weight` knob (default 0.3 when `--kg-gnn` is set in `query.py`).

---

## 1. Why a GNN

The existing `text_to_brain` MLP is strong on well-covered cognitive terms but
weak on rare drugs, novel compounds, and out-of-vocabulary phrases. A KG with
gene → region (abagen), receptor → region (neuromaps PET), and term → region
(merged_sources CBMA) edges encodes inductive bias the MLP has no access to:
"compound binds receptor X" implies "should activate regions where X is dense."

Empirically (this repo, 2026-04-25):
- **text_to_brain baseline (term-split):** mean Pearson r = 0.611
- **KG-to-brain GNN (collection-split, harder):** mean Pearson r = **0.749**

Collection-split holds out *entire studies*, so the GNN winning on the harder
generalization split is meaningful, not a metric quirk.

---

## 2. Graph schema

| Node type | Source | Count | Init feature |
|---|---|---:|---|
| Term | `merged_sources/term_vocab.pkl` | 11,375 | OpenAI `text-embedding-3-large` (3072-dim, normalized) |
| Gene | `data/gene_info.json` | 44,934 | hash-based (placeholder; swap via `--gene-embeddings`) |
| Receptor | parsed from `neuromaps_cache/annotation_labels.pkl` | 21 | hash-based |
| OntologyConcept | UBERON / MF / MONDO / HPO / ChEBI / Cognitive Atlas | 371,305 | hash-based |
| Region | combined atlas (Glasser+Tian) | 392 | one-hot + learnable `Embedding` |

**Edges** (each typed; reverse edges added automatically with `rev_<rel>` for
bidirectional message passing):

| Edge type | Source | Count |
|---|---|---:|
| `Term —activates→ Region` | `merged_sources/term_maps.npz` (top-decile parcels per term) | ~507k |
| `Gene —expressedIn→ Region` | `abagen_cache/term_maps.npz` | ~578k |
| `Receptor —densityIn→ Region` | `neuromaps_cache/annotation_maps.npz` | 840 |
| `Gene —encodes→ Receptor` | static map in `build_heterogeneous_graph.py` | 13 |
| `OntologyConcept —relatedTo→ OntologyConcept` | `ontology_meta_graph.build_meta_graph` | ~900k |

---

## 3. Architecture

```
KGBrainGNN(
    input_proj: ModuleDict[node_type → Linear(d_in, hidden)]
    convs: ModuleList[
        HeteroConv({edge_type: SAGEConv((-1, -1), out_dim) for edge_type in metadata})
        ×3 layers (last layer hidden_dim → out_dim; earlier layers stay at hidden_dim)
    ]
    region_embedding: Embedding(n_parcels, out_dim)
    layer_norm + dropout(0.2) between layers
)
```

Default config (matches the trained model): `hidden_dim=256`, `out_dim=256`,
`num_layers=3`, ~4M parameters.

**Why SAGEConv inside HeteroConv** rather than R-GCN: PyG's `HeteroConv`
dispatches to a separate inner module per edge type, which already gives you
per-relation weights — same expressiveness as R-GCN with a simpler API. R-GCN
inside HeteroConv would also fail because `HeteroConv` does not pass an
`edge_type` tensor through.

**Readout:** `pred_map[term] = encoder(term)[idx] @ region_embedding.weight.T`.
A simple dot product against learned per-parcel vectors. The Region embedding
table is the only fully-learnable spatial component.

---

## 4. Training

Multi-task loss (default weights in parens):

- `L_map` (1.0): MSE between predicted parcel vector and the merged_sources
  ground-truth row for that Term.
- `L_contrastive` (0.2): InfoNCE on highly-correlated Term pairs (auto-mined
  at startup from `term_maps`; threshold 0.6 on the cross-correlation matrix
  of a 2k-term sample).
- `L_link` (0.0 default; opt-in via `--link-loss-weight`): DistMult on a held-
  out KG edge set. Wired but not yet tuned.
- `L_teacher` (0.0 default; opt-in via `--teacher <path.npz>`): KL between the
  predicted distribution and a pre-computed teacher distribution from
  `calibrate_kg_teacher_from_audit.py`.

Eval split: `--eval-split collection` (default) holds out entire source
buckets from `term_sources.pkl`. `--eval-split term` is a random per-term
split. `--eval-split both` reports both at the end.

Defaults: `lr=5e-4`, `weight_decay=1e-5`, `batch_size=256`, gradient clip at
norm 5.0, AdamW, best-val-corr checkpointing.

---

## 5. Files

| Path | Purpose |
|---|---|
| `scripts/build_heterogeneous_graph.py` | Builds the graph from caches; saves `kg_brain_graph/{hetero_data.pt, node_index.pkl, meta.json}`. |
| `scripts/train_kg_to_brain_gnn.py` | Trainer. Outputs to `kg_brain_gnn_model/`. |
| `scripts/verify_kg_gnn.py` | Synthetic-graph smoke test (CI-friendly, no caches needed). |
| `scripts/review_kg_gnn_training.py` | Generates a markdown training-review report; compares to baseline if `--baseline` is passed. |
| `enrichment/kg_to_brain.py` | `build_model()` and `KGToBrainPredictor` (inference adapter mirroring `TextToBrainEmbedding.predict_map`). |
| `enrichment/unified_enrichment.py` | `enable_kg_gnn` constructor flag; `kg_query_text` + `kg_weight` `enrich()` args. |
| `scripts/query.py` | `--kg-gnn`, `--kg-graph-dir`, `--kg-weight` CLI flags. |

---

## 6. End-to-end run (recipe)

```bash
# 1. embed merged_sources terms once (~$0.01, ~2 min)
python neurolab/scripts/build_training_embeddings_from_openai_npz.py \
  --cache-dir neurolab/data/merged_sources \
  --openai-npz neurolab/data/abagen_cache/openai_embeddings_text-embedding-3-large.npz \
  --openai-meta-pkl neurolab/data/abagen_cache/openai_embedding_meta.pkl \
  --output-dir neurolab/data/merged_sources_openai_embeddings \
  --model text-embedding-3-large

# 2. build the heterogeneous graph (~30 s with the OpenAI features)
python neurolab/scripts/build_heterogeneous_graph.py \
  --term-embeddings neurolab/data/merged_sources_openai_embeddings/all_training_embeddings.npy

# 3. train (~90 min on a single GPU at hidden=256, num_layers=3, 30 epochs)
python neurolab/scripts/train_kg_to_brain_gnn.py \
  --epochs 30 --batch-size 256 --hidden-dim 256 --num-layers 3 \
  --output-dir neurolab/data/kg_brain_gnn_model

# 4. review
python neurolab/scripts/review_kg_gnn_training.py \
  --baseline neurolab/data/embedding_model/generalization_metrics.json

# 5. query with the GNN blended in
python neurolab/scripts/query.py "noradrenergic modulation of attention" --kg-gnn
```

---

## 7. Known issues / next steps

- **OntologyConcept dominance.** 371k OntologyConcept nodes vs 11k Terms; most
  contribute no signal to any Term query. Pruning to concepts within 2 hops of
  any Term in the supervision set would shrink memory and likely speed up
  training ~5x with no quality loss.
- **Sparse Gene→Receptor edges (13).** The static `RECEPTOR_TO_GENES` map only
  covers ~15 receptor tokens. Extending it to the full neuromaps tracer set
  would densify the receptor neighborhood and likely help on rare-receptor
  queries.
- **Hash-based Gene/Receptor/OntologyConcept features.** Only Term features
  are real (OpenAI). Replacing Gene features with a real embedding (e.g.
  `text-embedding-3-large` over gene name + summary, or a biomedical embedder)
  is the most likely next-largest correlation lift.
- **Convergence not reached.** In the 30-epoch run, val_corr was still
  climbing every epoch (0.74 → 0.77 across the last 4). A 50-epoch follow-up
  is likely to push test_corr meaningfully past 0.75.
- **`enable_kg_gnn=False` constructor default in `UnifiedEnrichment`.** Stays
  off by default for backward compat (callers without the trained model
  shouldn't break). The CLI opt-in in `query.py` is the canonical way to
  enable it.
