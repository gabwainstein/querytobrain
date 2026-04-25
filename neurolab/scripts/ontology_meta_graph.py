#!/usr/bin/env python3
"""
Meta-ontology graph: one graph where each ontology stays intact and bridge edges
connect related concepts across ontologies. Used for graph-aware query expansion
and retrieval-augmented prediction.

Do NOT merge ontologies into one schema. Nodes = ontology concepts (by normalized
label; source = which file they came from). Within-ontology edges from
label_to_related; bridge edges from embedding similarity (and optionally curated
xrefs). See docs/implementation/ONTOLOGY_META_GRAPH.md.

Usage:
  from ontology_expansion import load_ontology_index
  from ontology_meta_graph import build_meta_graph, expand_query_via_graph

  index = load_ontology_index("neurolab/data/ontologies")
  G = build_meta_graph(index, label_embeddings=emb, label_list=labels,
                       similarity_threshold=0.85)
  expansion = expand_query_via_graph(query_embedding, G, max_hops=2)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

from ontology_expansion import _normalize_term, load_ontology_index


def _source_to_node_type(source: str) -> str:
    """Infer node_type from ontology filename."""
    s = (source or "").lower()
    if "mondo" in s or "doid" in s:
        return "disease"
    if "hpo" in s or "hp." in s:
        return "phenotype"
    if "chebi" in s:
        return "chemical"
    if "cogat" in s or "cogpo" in s or "nbo" in s or "mf." in s or "mf " in s:
        return "cognitive_concept"
    if "go" in s:
        return "biological_process"
    if "uberon" in s:
        return "anatomy"
    return "concept"


def _filename_stem(path_or_name: str) -> str:
    """e.g. mondo.owl -> mondo, cogat.v2.owl -> cogat."""
    name = Path(path_or_name).stem
    # cogat.v2 -> cogat
    if "." in name:
        name = name.split(".")[0]
    return name or "ontology"


def build_meta_graph(
    index: dict[str, Any],
    label_embeddings: np.ndarray | None = None,
    label_list: list[str] | None = None,
    similarity_threshold: float = 0.85,
    max_bridges_per_node: int = 5,
) -> "nx.DiGraph":
    """
    Build a single DiGraph from the merged ontology index.

    Nodes: one per normalized label. Attributes: label (display name), source
    (filename), node_type (disease|phenotype|chemical|cognitive_concept|...), embedding (optional).
    Edges: from label_to_related (relation, weight); plus bridge edges between
    nodes from different sources when embedding similarity >= threshold.

    Returns NetworkX DiGraph. Requires networkx.
    """
    if nx is None:
        raise ImportError("ontology_meta_graph requires networkx: pip install networkx")

    G = nx.DiGraph()
    label_to_related = index.get("label_to_related") or {}
    label_to_source = index.get("label_to_source") or {}

    # Embedding index: norm_label -> vector (only for labels in label_list)
    emb_by_label: dict[str, np.ndarray] = {}
    if label_embeddings is not None and label_list is not None:
        for i, lab in enumerate(label_list):
            if i < label_embeddings.shape[0]:
                emb_by_label[lab] = np.asarray(label_embeddings[i], dtype=np.float32).ravel()

    # 1. Nodes
    for norm_label in label_to_related:
        related = label_to_related[norm_label]
        display = norm_label
        for item in related:
            if len(item) >= 3 and item[2] == "self":
                display = (item[0] or norm_label).strip()
                break
        source_fname = label_to_source.get(norm_label, "")
        source_stem = _filename_stem(source_fname)
        node_type = _source_to_node_type(source_fname)
        attrs = {
            "label": display,
            "source": source_stem,
            "source_file": source_fname,
            "node_type": node_type,
        }
        if norm_label in emb_by_label:
            attrs["embedding"] = emb_by_label[norm_label]
        G.add_node(norm_label, **attrs)

    # 2. Within-ontology (and cross-ontology) edges from label_to_related
    for norm_label, related in label_to_related.items():
        if norm_label not in G:
            continue
        for item in related:
            name = (item[0] or "").strip()
            weight = float(item[1]) if len(item) > 1 else 1.0
            rtype = item[2] if len(item) >= 3 else "related_to"
            other_norm = _normalize_term(name)
            if not other_norm or other_norm == norm_label:
                continue
            if other_norm in G:
                G.add_edge(
                    norm_label,
                    other_norm,
                    relation=rtype,
                    weight=weight,
                    source="ontology",
                )
                # Bidirectional for synonym/self
                if rtype in ("synonym", "self"):
                    G.add_edge(
                        other_norm,
                        norm_label,
                        relation=rtype,
                        weight=weight,
                        source="ontology",
                    )

    # 3. Bridge edges by embedding similarity (only across different sources)
    if emb_by_label and similarity_threshold > 0:
        by_source: dict[str, list[tuple[str, np.ndarray]]] = {}
        for nid in G:
            if nid not in emb_by_label:
                continue
            src = G.nodes[nid].get("source", "")
            by_source.setdefault(src, []).append((nid, emb_by_label[nid]))

        from sklearn.metrics.pairwise import cosine_similarity

        sources = list(by_source.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                src_a, src_b = sources[i], sources[j]
                list_a = by_source[src_a]
                list_b = by_source[src_b]
                if not list_a or not list_b:
                    continue
                emb_a = np.array([x[1] for x in list_a])
                emb_b = np.array([x[1] for x in list_b])
                sims = cosine_similarity(emb_a, emb_b)
                for idx_a in range(len(list_a)):
                    top = np.argsort(sims[idx_a])[::-1][:max_bridges_per_node]
                    for idx_b in top:
                        if sims[idx_a, idx_b] < similarity_threshold:
                            continue
                        id_a = list_a[idx_a][0]
                        id_b = list_b[idx_b][0]
                        sim = float(sims[idx_a, idx_b])
                        G.add_edge(
                            id_a,
                            id_b,
                            relation="semantic_bridge",
                            source="embedding_similarity",
                            similarity=sim,
                        )
                        G.add_edge(
                            id_b,
                            id_a,
                            relation="semantic_bridge",
                            source="embedding_similarity",
                            similarity=sim,
                        )

    return G


def expand_query_via_graph(
    query_embedding: np.ndarray,
    G: "nx.DiGraph",
    max_hops: int = 2,
    max_neighbors: int = 15,
    min_similarity: float = 0.7,
    min_relevance: float = 0.3,
    hop_decay: float = 0.7,
) -> dict[str, Any]:
    """
    Find ontology nodes relevant to the query and traverse the graph to collect
    related concepts across ontologies.

    Returns dict with:
      - seeds: list of (node_id, similarity)
      - expanded_terms: list of {id, label, source, node_type, relevance}
    """
    if nx is None or G.number_of_nodes() == 0:
        return {"seeds": [], "expanded_terms": []}

    query_embedding = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    # Nodes that have embeddings
    nodes_with_emb = []
    embs = []
    for nid, data in G.nodes(data=True):
        emb = data.get("embedding")
        if emb is not None:
            nodes_with_emb.append(nid)
            embs.append(np.asarray(emb, dtype=np.float32).ravel())

    if not nodes_with_emb:
        return {"seeds": [], "expanded_terms": []}

    from sklearn.metrics.pairwise import cosine_similarity

    E = np.array(embs)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    q = query_embedding.reshape(1, -1)
    if q.shape[1] != E.shape[1]:
        return {"seeds": [], "expanded_terms": []}
    sims = cosine_similarity(q, E)[0]

    # Seeds: top by similarity above min_similarity
    top_k = min(10, len(nodes_with_emb))
    order = np.argsort(sims)[::-1][:top_k]
    seeds = [(nodes_with_emb[i], float(sims[i])) for i in order if sims[i] >= min_similarity]

    expanded: dict[str, float] = {}
    for seed_id, seed_sim in seeds:
        expanded[seed_id] = seed_sim
        visited = {seed_id}
        frontier: list[tuple[str, int]] = [(seed_id, 0)]

        while frontier:
            current_id, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            neighbors = list(G.successors(current_id)) + list(G.predecessors(current_id))
            for neighbor_id in neighbors[:max_neighbors]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                data = G.nodes.get(neighbor_id, {})
                emb = data.get("embedding")
                if emb is not None:
                    emb = np.asarray(emb).reshape(1, -1)
                    s = cosine_similarity(q, emb)[0, 0]
                else:
                    s = seed_sim * (hop_decay ** (hop + 1))
                weight = s * (hop_decay ** (hop + 1))
                if weight >= min_relevance:
                    expanded[neighbor_id] = max(expanded.get(neighbor_id, 0), weight)
                    frontier.append((neighbor_id, hop + 1))

    result_terms = []
    for nid, rel in sorted(expanded.items(), key=lambda x: -x[1]):
        data = G.nodes[nid]
        result_terms.append({
            "id": nid,
            "label": data.get("label", nid),
            "source": data.get("source", ""),
            "node_type": data.get("node_type", "concept"),
            "relevance": round(rel, 4),
        })

    return {
        "seeds": seeds,
        "expanded_terms": result_terms,
    }


def get_training_maps_db(cache_dir: str | Path) -> tuple[dict[str, np.ndarray], int]:
    """
    Load term_maps.npz and term_vocab.pkl from an expanded (or decoder) cache.
    Returns (label_to_map, n_parcels) where label_to_map keys are term strings
    (we use normalized for lookup when possible). If term_vocab is list of str,
    keys are those strings; we also key by normalized for expansion.
    """
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "term_maps.npz"
    vocab_path = cache_dir / "term_vocab.pkl"
    if not npz_path.exists() or not vocab_path.exists():
        return {}, 0

    import pickle
    data = np.load(npz_path)
    if "term_maps" in data:
        maps = data["term_maps"]
    else:
        keys = [k for k in data.files if k != "term_maps" and not k.startswith("_")]
        if not keys:
            return {}, 0
        maps = data[keys[0]]
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    if not isinstance(vocab, (list, tuple)) or len(vocab) != maps.shape[0]:
        return {}, 0
    n_parcels = maps.shape[1]
    label_to_map = {}
    for i, term in enumerate(vocab):
        t = term.strip() if isinstance(term, str) else str(term)
        label_to_map[t] = maps[i].astype(np.float64)
        norm = _normalize_term(t)
        if norm and norm not in label_to_map:
            label_to_map[norm] = maps[i].astype(np.float64)
    return label_to_map, n_parcels


def augmented_prediction(
    query_text: str,
    query_embedding: np.ndarray,
    predicted_map: np.ndarray,
    G: "nx.DiGraph",
    training_maps_db: dict[str, np.ndarray],
    drug_spatial_maps: dict[str, np.ndarray] | None = None,
    max_hops: int = 2,
    alpha_retrieval_cap: float = 0.3,
    min_relevance: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Blend MLP prediction with maps retrieved via ontology graph expansion.

    training_maps_db: label -> (n_parcels,) map (from get_training_maps_db).
    drug_spatial_maps: optional drug name -> (n_parcels,) from gene PCA Phase 4.

    Returns (final_map, enrichment). enrichment has keys: related_diseases,
    related_phenotypes, related_concepts, related_drugs, related_receptors.
    """
    expansion = expand_query_via_graph(
        query_embedding,
        G,
        max_hops=max_hops,
        min_relevance=min_relevance,
    )
    retrieved_maps = []
    retrieved_weights = []

    for term_info in expansion.get("expanded_terms", []):
        label = term_info.get("label", "")
        relevance = term_info.get("relevance", 0)
        node_type = term_info.get("node_type", "")

        if label in training_maps_db:
            retrieved_maps.append(training_maps_db[label])
            retrieved_weights.append(relevance)
        norm = _normalize_term(label)
        if norm != label and norm in training_maps_db:
            retrieved_maps.append(training_maps_db[norm])
            retrieved_weights.append(relevance)

        if drug_spatial_maps and node_type in ("chemical", "drug", "concept"):
            if label in drug_spatial_maps:
                retrieved_maps.append(drug_spatial_maps[label])
                retrieved_weights.append(relevance * 0.8)

    if not retrieved_maps:
        return predicted_map.astype(np.float64), {
            "related_diseases": [],
            "related_phenotypes": [],
            "related_concepts": [],
            "related_drugs": [],
            "related_receptors": [],
            "expansion": expansion,
        }

    retrieved_weights = np.array(retrieved_weights, dtype=np.float64)
    retrieved_weights /= retrieved_weights.sum()
    retrieval_map = np.zeros_like(predicted_map, dtype=np.float64)
    for m, w in zip(retrieved_maps, retrieved_weights):
        retrieval_map += w * np.asarray(m, dtype=np.float64).ravel()[: retrieval_map.size]

    alpha = min(alpha_retrieval_cap, 0.05 * len(retrieved_maps))
    final_map = (1.0 - alpha) * np.asarray(predicted_map, dtype=np.float64).ravel() + alpha * retrieval_map

    terms = expansion.get("expanded_terms", [])
    enrichment = {
        "related_diseases": [t for t in terms if t.get("node_type") == "disease"][:5],
        "related_phenotypes": [t for t in terms if t.get("node_type") == "phenotype"][:5],
        "related_concepts": [t for t in terms if t.get("node_type") == "cognitive_concept"][:5],
        "related_drugs": [t for t in terms if t.get("node_type") in ("chemical", "drug")][:5],
        "related_receptors": [t for t in terms if t.get("node_type") == "receptor"][:5],
        "expansion": expansion,
    }
    return final_map, enrichment


if __name__ == "__main__":
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    ont_dir = repo / "neurolab" / "data" / "ontologies"
    if not ont_dir.exists():
        print("No ontology dir; run download_ontologies.py --clinical --extra", file=sys.stderr)
        sys.exit(1)
    index = load_ontology_index(ont_dir)
    G = build_meta_graph(index, similarity_threshold=0.0)
    print(f"Meta-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    sys.exit(0)
