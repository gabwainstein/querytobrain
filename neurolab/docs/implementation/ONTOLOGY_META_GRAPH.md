# Ontology Meta-Graph: Interconnected Ontologies for Retrieval Augmentation

## Why a meta-graph (not a merged ontology)

Ontologies are kept as separate silos; we do **not** merge them into one schema. Instead we build a **meta-graph** where:

- Each ontology’s concepts stay intact (nodes keyed by normalized label, with `source` = which file they came from).
- **Within-ontology edges** come from existing relations (parent, child, synonym) in `label_to_related`.
- **Bridge edges** connect concepts **across** ontologies (e.g. Cognitive Atlas ↔ MONDO, MONDO ↔ HPO) via:
  - **Embedding similarity**: cosine between ontology label embeddings above a threshold (e.g. 0.85).
  - (Future) **Curated xrefs**: MONDO→HPO, ChEBI→DrugBank, etc.

This gives:

- **Term expansion**: e.g. "memory problems in Alzheimer's" → traverse to Alzheimer's, episodic memory, hippocampal atrophy and retrieve/combine their maps.
- **Cross-domain paths**: e.g. anxiety (Cognitive Atlas) → GAD (MONDO) → SSRIs (ChEBI) → 5-HT1A (receptors).
- **Training / inference**: use graph expansion to get related terms and blend their brain maps with the MLP prediction (retrieval augmentation).

## Implementation

### Building the graph

- **Script**: `neurolab/scripts/ontology_meta_graph.py`
- **Inputs**: `load_ontology_index(ontology_dir)` plus optional `label_embeddings` and `label_list` (from `build_ontology_label_embeddings`).
- **Output**: NetworkX `DiGraph` with:
  - **Nodes**: one per normalized label; attributes `label`, `source`, `node_type` (disease | phenotype | chemical | cognitive_concept | …), optional `embedding`.
  - **Edges**: `relation`, `weight`, `source` (ontology | embedding_similarity).

```python
from ontology_expansion import load_ontology_index
from ontology_meta_graph import build_meta_graph, expand_query_via_graph

index = load_ontology_index("neurolab/data/ontologies")
G = build_meta_graph(index, label_embeddings=emb, label_list=labels,
                     similarity_threshold=0.85, max_bridges_per_node=5)
```

### Query expansion

- **`expand_query_via_graph(query_embedding, G, max_hops=2, ...)`**  
  Finds seed nodes by cosine similarity to the query embedding, then BFS up to `max_hops`. Returns `seeds` and `expanded_terms` (id, label, source, node_type, relevance).

### Retrieval-augmented prediction

- **`augmented_prediction(query_text, query_embedding, predicted_map, G, training_maps_db, drug_spatial_maps=None, ...)`**  
  1. Runs query expansion.  
  2. Looks up brain maps for expanded terms in `training_maps_db` (and optionally drug maps from gene PCA Phase 4).  
  3. Blends: `final_map = (1 - alpha)*predicted_map + alpha*retrieval_map`, with `alpha` capped (e.g. 0.3).

### Wiring in the pipeline

- **Inference**  
  `TextToBrainEmbedding` supports:
  - `use_ontology_retrieval_augmentation`: bool in config.
  - `ontology_retrieval_cache_dir`: path to expanded cache (term_maps.npz + term_vocab.pkl).
  - `ontology_retrieval_alpha`, `ontology_retrieval_max_hops`.

  When these are set (and KG index + ontology label embeddings are loaded), the embedder builds the meta-graph and loads the training maps; in `embed()` it blends the MLP output with the retrieval map as above.

- **Training**  
  Save these in `config.pkl` by passing:
  - `--use-ontology-retrieval-augmentation`
  - `--ontology-retrieval-cache-dir neurolab/data/decoder_cache_expanded`
  - `--ontology-retrieval-alpha 0.3` (optional)
  - `--ontology-retrieval-max-hops 2` (optional)

  Requires `--kg-context-hops > 0` and semantic KG mode so ontology embeddings exist.

## Priority connections (for future curation)

- **Tier 1**: Cognitive Atlas ↔ MONDO; PDSP/ChEBI ↔ receptor genes ↔ gene PCA; MONDO ↔ HPO.
- **Tier 2**: Cognitive Atlas ↔ NeuroSynth terms; ChEBI ↔ PDSP drug names; GO ↔ gene PCA loadings.
- **Tier 3**: UBERON ↔ atlas parcels; DisGeNET disease–gene ↔ gene expression.

## Graph embedding as MLP input (future)

The plan describes a **graph embedding** (e.g. node2vec or mean neighbor embedding) concatenated with the text embedding as extra input to the MLP. That would require:

- Computing a graph embedding per query (e.g. from the query’s neighborhood in the meta-graph).
- Changing the model to accept `(text_dim + graph_dim)` and training with that.

Recommendation: get retrieval augmentation and the meta-graph solid first, then add graph embedding as an optional feature.

## Dependencies

- `networkx`, `scikit-learn` (for cosine_similarity), `numpy`.
- Ontology index and (for bridges) ontology label embeddings from the same encoder as the text-to-brain model.
