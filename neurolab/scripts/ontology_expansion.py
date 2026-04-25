#!/usr/bin/env python3
"""
Ontology-based term expansion using canonical open-source ontologies (local, SOTA).

Loads OBO/OWL ontologies from a local directory and exposes:
  - expand_term(term, decoder_vocab) -> list of (cache_term, weight, relation_type)
  - get_map_for_term(term, decoder_maps, term_vocab) -> 400-D map (average of related cache maps)

**When a term has a direct map (e.g. "executive function" in NQ/NS cache):** We always use that
map. Expansion only adds terms we don't have; we never overwrite a cache term with an
ontology-derived blend.

**Ontology distance → weight:** We map relationship type to a weight used when blending brain
maps (closer in ontology = higher weight). Current mapping (RELATION_WEIGHTS): self 1.0,
synonym 0.95, child 0.85, parent 0.8. Whether ontological proximity correlates with brain-map
similarity is an empirical question; run ontology_brain_correlation.py to check on your cache.

Recommended ontologies (download once to data/ontologies/):
  - Cognitive Atlas: https://github.com/CognitiveAtlas/ontology (OWL/export)
  - Mental Functioning (MF): http://purl.obolibrary.org/obo/mf.owl
  - NIFSTD: https://github.com/SciCrunch/NIF-Ontology (TTL/OWL)
  - UBERON: http://purl.obolibrary.org/obo/uberon.owl

Usage:
  from ontology_expansion import load_ontology_index, expand_term, get_map_for_term
  index = load_ontology_index("path/to/ontologies")
  related = expand_term("working memory", decoder_term_list, index)
  map_400 = get_map_for_term("prosopagnosia", decoder_maps, term_vocab, index)

For KG context at encoding (text-to-brain): get_kg_context(term, index, max_hops) appends triples for a single term;
get_kg_context_for_query(query, index, max_hops) finds every ontology concept that appears in the query (substring match)
and appends their triples so the encoder sees one connected block. Use Cognitive Atlas (and CogPO) in ontology_dir for
task concepts (n-back, working memory, executive function, etc.).
"""
from __future__ import annotations

# Weights by relation type when blending related terms' brain maps (ontology distance → weight).
# These are HEURISTIC: "closer in ontology = higher weight", not derived from data or graph theory.
# Better approaches:
#   (1) Data-driven: run ontology_brain_correlation.py, then use observed mean brain-map r per
#       relation type as weights (e.g. --relation-weights-file). Weights then reflect actual
#       brain-map similarity of synonym/parent/child pairs in your cache.
#   (2) Graph-theoretic: weight = gamma^path_length (shortest path in ontology graph). Would
#       require building a unified term graph and computing distances; not yet implemented.
#       Caveat: when merging Cognitive Atlas + UBERON + MF etc., there are no cross-ontology
#       edges — "working memory" (CogAt) and "prefrontal cortex" (UBERON) are disconnected.
#       If you implement graph distance, run per-ontology and take min path across ontologies.
#   (3) Data-driven weights caveat: when you run ontology_brain_correlation on NQ/NS cache,
#       you measure correlation of *model outputs*, not real brain activations. NQ already
#       uses semantic smoothing, so related terms get correlated maps by construction. The
#       measured r values are upper bounds; if you later use real meta-analytic maps (e.g.
#       NeuroSynth MKDA), optimal weights will likely be lower.
# Until you run validation, these defaults are a plausible guess (synonym > child > parent).
RELATION_WEIGHTS = {
    "self": 1.0,
    "synonym": 0.95,
    "child": 0.85,
    "parent": 0.8,
}
# When blending: parent's map used to approximate child = too broad → downweight; child's
# map used to approximate parent = subset → slight downweight. Synonym/self symmetric.
DIRECTION_SCALE = {
    "self": 1.0,
    "synonym": 1.0,
    "child": 0.9,
    "parent": 0.7,
}

import re
from pathlib import Path
from typing import Any

# Optional: obonet for OBO, rdflib for OWL
try:
    import obonet
    import networkx as nx
    HAS_OBONET = True
except ImportError:
    HAS_OBONET = False

try:
    import rdflib
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False


def _normalize_term(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ")


def _load_obo(path: Path, relation_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Load OBO file into id->name, id->synonyms, id->parents, name/synonym->ids."""
    w = relation_weights or RELATION_WEIGHTS
    if not HAS_OBONET:
        return {}
    try:
        graph = obonet.read_obo(path)
    except Exception:
        return {}
    id_to_name: dict[str, str] = {}
    id_to_synonyms: dict[str, list[str]] = {}
    id_to_parents: dict[str, list[str]] = {}
    for nid, data in graph.nodes(data=True):
        raw = data.get("name")
        name = (raw[0] if isinstance(raw, list) and raw else raw) if raw else None
        if isinstance(name, str):
            id_to_name[nid] = name
        syns = data.get("synonym") or []
        if syns:
            id_to_synonyms[nid] = [s.split('"')[1] if '"' in s else s for s in syns]
        parents = list(graph.predecessors(nid)) if graph.number_of_nodes() else []
        if parents:
            id_to_parents[nid] = parents
    # Build label/synonym -> list of (related_term_string, weight, relation_type)
    label_to_related: dict[str, list[tuple[str, float, str]]] = {}
    for nid, name in id_to_name.items():
        key = _normalize_term(name)
        if not key:
            continue
        related: list[tuple[str, float, str]] = [(name, w.get("self", 1.0), "self")]
        for syn in id_to_synonyms.get(nid, []):
            if syn and _normalize_term(syn) != key:
                related.append((syn, w.get("synonym", 0.95), "synonym"))
        for pid in id_to_parents.get(nid, []):
            pname = id_to_name.get(pid)
            if pname:
                related.append((pname, w.get("parent", 0.8), "parent"))
        if HAS_OBONET and graph:
            for cid in graph.successors(nid):
                cname = id_to_name.get(cid)
                if cname:
                    related.append((cname, w.get("child", 0.85), "child"))
        label_to_related[key] = related
    for nid, syns in id_to_synonyms.items():
        for s in syns:
            key = _normalize_term(s)
            if key and key not in label_to_related:
                name = id_to_name.get(nid)
                if name:
                    related = [(name, w.get("synonym", 0.95), "synonym")]
                    for pid in id_to_parents.get(nid, []):
                        pname = id_to_name.get(pid)
                        if pname:
                            related.append((pname, w.get("parent", 0.8), "parent"))
                    label_to_related[key] = related
    # norm_to_ids: normalized label -> list of node IDs (for graph path-length weighting)
    norm_to_ids: dict[str, list[str]] = {}
    for nid, name in id_to_name.items():
        key = _normalize_term(name)
        if key:
            if key not in norm_to_ids:
                norm_to_ids[key] = []
            if nid not in norm_to_ids[key]:
                norm_to_ids[key].append(nid)
    for nid, syns in id_to_synonyms.items():
        for s in syns:
            key = _normalize_term(s)
            if key:
                if key not in norm_to_ids:
                    norm_to_ids[key] = []
                if nid not in norm_to_ids[key]:
                    norm_to_ids[key].append(nid)
    return {
        "id_to_name": id_to_name,
        "id_to_synonyms": id_to_synonyms,
        "id_to_parents": id_to_parents,
        "label_to_related": label_to_related,
        "graph": graph if HAS_OBONET else None,
        "norm_to_ids": norm_to_ids,
    }


def _load_owl_rdf(path: Path, relation_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Load OWL/RDF: rdfs:label, skos:prefLabel, skos:altLabel -> labels; rdfs:subClassOf -> hierarchy.
    Cognitive Atlas and similar ontologies use skos:prefLabel for concept names and subClassOf for
    parent/child, so we need both to get good decoder overlap and synonym expansion."""
    if not HAS_RDFLIB:
        return {}
    try:
        g = rdflib.Graph()
        suf = path.suffix.lower()
        if suf in (".owl", ".rdf"):
            g.parse(path, format="xml")
        elif suf == ".ttl":
            g.parse(path, format="turtle")
        else:
            g.parse(path)
    except Exception:
        return {}
    RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    label_to_ids: dict[str, list[str]] = {}
    id_to_name: dict[str, str] = {}
    # Collect labels: rdfs:label, skos:prefLabel (Cognitive Atlas), skos:altLabel
    for s, p, o in g:
        if p in (RDFS.label, SKOS.prefLabel) and isinstance(o, rdflib.Literal):
            label = str(o).strip()
            key = _normalize_term(label)
            if key:
                uri = str(s)
                id_to_name[uri] = label
                if uri not in (label_to_ids.get(key) or []):
                    label_to_ids.setdefault(key, []).append(uri)
        if p == SKOS.altLabel and isinstance(o, rdflib.Literal):
            label = str(o).strip()
            key = _normalize_term(label)
            if key:
                uri = str(s)
                if id_to_name.get(uri) is None:
                    id_to_name[uri] = label
                if uri not in (label_to_ids.get(key) or []):
                    label_to_ids.setdefault(key, []).append(uri)
    # Parent/child: rdfs:subClassOf (subject subclass of object; object is parent)
    id_to_parents: dict[str, list[str]] = {}
    for s, p, o in g:
        if p == RDFS.subClassOf and isinstance(o, rdflib.term.Node):
            subj_uri = str(s)
            parent_uri = str(o)
            if parent_uri != subj_uri and not parent_uri.endswith("#Thing") and "owl#" not in parent_uri:
                id_to_parents.setdefault(subj_uri, []).append(parent_uri)
    w = relation_weights or RELATION_WEIGHTS
    label_to_related: dict[str, list[tuple[str, float, str]]] = {}
    for label, ids in label_to_ids.items():
        seen: dict[str, tuple[str, float, str]] = {}  # norm -> (display_name, weight, type)
        for uri in ids:
            name = id_to_name.get(uri)
            if name:
                nkey = _normalize_term(name)
                if nkey and (nkey not in seen or seen[nkey][1] < w.get("self", 1.0)):
                    seen[nkey] = (name, w.get("self", 1.0), "self")
            for parent_uri in id_to_parents.get(uri, []):
                pname = id_to_name.get(parent_uri)
                if pname:
                    pkey = _normalize_term(pname)
                    if pkey and (pkey not in seen or seen[pkey][1] < w.get("parent", 0.8)):
                        seen[pkey] = (pname, w.get("parent", 0.8), "parent")
        if seen:
            label_to_related[label] = list(seen.values())
    return {
        "id_to_name": id_to_name,
        "label_to_related": label_to_related,
        "graph": g,
    }


def _load_json_ontology(path: Path) -> dict[str, Any]:
    """Load JSON ontology index (e.g. nootropics_ontology_index.json from build_nootropics_knowledge_graph)."""
    try:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    label_to_related_raw = data.get("label_to_related") or {}
    label_to_related: dict[str, list[tuple[str, float, str]]] = {}
    for label, related in label_to_related_raw.items():
        if not related:
            continue
        converted = []
        for item in related:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                term = item[0]
                weight = float(item[1]) if len(item) > 1 else 0.8
                rel_type = item[2] if len(item) > 2 else "other"
                converted.append((term, weight, rel_type))
        if converted:
            label_to_related[_normalize_term(label)] = converted
    return {
        "id_to_name": {},
        "label_to_related": label_to_related,
        "graph": None,
        "norm_to_ids": {},
    }


def load_ontology_index(
    ontology_dir: str | Path,
    relation_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Load all OBO/OWL files in ontology_dir into a single index.
    relation_weights: optional dict (self, synonym, child, parent) -> weight; if None, uses RELATION_WEIGHTS.
    Use data-driven weights from ontology_brain_correlation.py --output-weights for better blending.
    Returns a dict with:
      - label_to_related: normalized_label -> list of (related_term_string, weight, relation_type)
      - label_to_source: normalized_label -> filename (first file that contributed it, for reporting)
      - id_to_name: (from OBO/OWL)
    """
    ontology_dir = Path(ontology_dir)
    if not ontology_dir.exists():
        return {"label_to_related": {}, "label_to_source": {}, "id_to_name": {}, "ontology_graphs": []}

    all_label_to_related: dict[str, list[tuple[str, float, str]]] = {}
    all_id_to_name: dict[str, str] = {}
    label_to_source: dict[str, str] = {}
    ontology_graphs: list[tuple[Any, dict[str, list[str]]]] = []
    w = relation_weights

    for path in sorted(ontology_dir.glob("*")):
        if path.suffix.lower() in (".obo", ".obo.gz", ".obo.xz"):
            data = _load_obo(path, relation_weights=w)
        elif path.suffix.lower() in (".owl", ".rdf", ".ttl"):
            data = _load_owl_rdf(path, relation_weights=w)
        elif path.suffix.lower() == ".json":
            data = _load_json_ontology(path)
        else:
            continue
        if not data:
            continue
        id_to_name = data.get("id_to_name") or {}
        label_to_related = data.get("label_to_related") or {}
        all_id_to_name.update(id_to_name)
        graph = data.get("graph")
        norm_to_ids = data.get("norm_to_ids") or {}
        if graph is not None and norm_to_ids:
            ontology_graphs.append((graph, norm_to_ids))
        fname = path.name
        for label, related in label_to_related.items():
            if not related:
                continue
            if label not in all_label_to_related:
                all_label_to_related[label] = list(related)
                label_to_source[label] = fname
            else:
                # Merge: same label in multiple ontologies -> union by norm name, max weight
                def _item(n, weight, t="other"):
                    return (_normalize_term(n), (n, weight, t))
                existing = {}
                for item in all_label_to_related[label]:
                    n, weight = item[0], item[1]
                    t = item[2] if len(item) >= 3 else "other"
                    k, v = _item(n, weight, t)
                    if k:
                        existing[k] = v
                for item in related:
                    n, weight = item[0], item[1]
                    t = item[2] if len(item) >= 3 else "other"
                    k, v = _item(n, weight, t)
                    if k and (k not in existing or weight > existing[k][1]):
                        existing[k] = v
                all_label_to_related[label] = list(existing.values())

    return {
        "label_to_related": all_label_to_related,
        "label_to_source": label_to_source,
        "id_to_name": all_id_to_name,
        "ontology_graphs": ontology_graphs,
    }


def _expand_term_by_graph_distance(
    term: str,
    decoder_vocab: list[str],
    index: dict[str, Any],
    gamma: float = 0.8,
) -> list[tuple[str, float, str]]:
    """
    Return list of (cache_term, weight, "other") using weight = gamma^path_length.
    Per-ontology shortest path; take min path across ontologies that contain both terms.
    Only OBO graphs are used (OWL not supported for path length).
    """
    if not HAS_OBONET:
        return []
    ontology_graphs = index.get("ontology_graphs") or []
    if not ontology_graphs:
        return []
    key = _normalize_term(term)
    if not key:
        return []
    decoder_set = {_normalize_term(t): t for t in decoder_vocab}
    out: dict[str, float] = {}
    for n in decoder_set:
        if n == key:
            out[decoder_set[n]] = 1.0
            continue
        min_path = float("inf")
        for graph, norm_to_ids in ontology_graphs:
            if key not in norm_to_ids or n not in norm_to_ids:
                continue
            try:
                g_undir = graph.to_undirected() if graph.is_directed() else graph
            except Exception:
                continue
            for tid in norm_to_ids[key]:
                for rid in norm_to_ids[n]:
                    try:
                        path_len = nx.shortest_path_length(g_undir, tid, rid)
                        min_path = min(min_path, path_len)
                    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                        pass
        if min_path != float("inf"):
            orig = decoder_set[n]
            out[orig] = max(out.get(orig, 0.0), gamma ** min_path)
    ranked = sorted(out.items(), key=lambda x: -x[1])[:20]
    return [(orig, w, "other") for orig, w in ranked]


def expand_term(
    term: str,
    decoder_vocab: list[str],
    index: dict[str, Any],
    use_graph_distance: bool = False,
    gamma: float = 0.8,
) -> list[tuple[str, float, str]]:
    """
    For a query term, return list of (decoder_cache_term, weight, relation_type) that are related via ontology.
    Only returns terms that exist in decoder_vocab (after normalization). When same term appears from
    multiple relations, keeps the one with highest weight (and its type, for direction scaling).
    use_graph_distance: if True and index has ontology_graphs (from OBO), use weight = gamma^path_length
    (min path across ontologies that contain both terms). Per-ontology graphs have no cross-ontology edges.
    """
    key = _normalize_term(term)
    if not key:
        return []
    if use_graph_distance and (index.get("ontology_graphs") or []):
        return _expand_term_by_graph_distance(term, decoder_vocab, index, gamma=gamma)
    decoder_set = {_normalize_term(t) for t in decoder_vocab}
    label_to_related = index.get("label_to_related") or {}
    # orig -> (best_weight, best_relation_type)
    out: dict[str, tuple[float, str]] = {}
    if key in label_to_related:
        for item in label_to_related[key]:
            related_name = item[0]
            w = item[1]
            rtype = item[2] if len(item) >= 3 else "other"
            n = _normalize_term(related_name)
            if n in decoder_set:
                orig = next((v for v in decoder_vocab if _normalize_term(v) == n), related_name)
                if orig not in out or w > out[orig][0]:
                    out[orig] = (w, rtype)
    if key in decoder_set:
        orig = next((v for v in decoder_vocab if _normalize_term(v) == key), term)
        if orig not in out or 1.0 > out[orig][0]:
            out[orig] = (1.0, "self")
    ranked = sorted(out.items(), key=lambda x: -x[1][0])[:20]
    return [(orig, w, rtype) for orig, (w, rtype) in ranked]


def get_kg_context(
    term: str,
    index: dict[str, Any],
    max_hops: int = 1,
    max_per_relation: int = 15,
    sep: str = " | ",
) -> str:
    """
    Return a short string of KG triples (relation_type: object1, object2, ...) for the term.
    Used to append ontology context to the term before encoding so the encoder sees hierarchy.
    max_hops=1: direct relations only (parent, child, synonym). max_hops=2: include relations of those.
    """
    label_to_related = index.get("label_to_related") or {}
    key = _normalize_term(term)
    if not key or key not in label_to_related:
        return ""

    parts: list[str] = []
    seen: set[str] = set()

    def add_triples_for_label(label_norm: str, hop: int, prefix: str = "") -> None:
        if label_norm not in label_to_related or hop > max_hops:
            return
        by_type: dict[str, list[str]] = {}
        for item in label_to_related[label_norm]:
            related_name = item[0]
            rtype = item[2] if len(item) >= 3 else "other"
            obj_norm = _normalize_term(related_name)
            if not obj_norm or obj_norm in seen:
                continue
            seen.add(obj_norm)
            by_type.setdefault(rtype, []).append(related_name.strip())
        for rtype in ("synonym", "child", "parent", "other"):
            if rtype not in by_type:
                continue
            objs = by_type[rtype][:max_per_relation]
            if objs:
                parts.append(f"{prefix}{rtype}: {', '.join(objs)}")

    # 1-hop: direct relations of the term
    add_triples_for_label(key, 1)
    if max_hops >= 2:
        # 2-hop: for each related label that is in the index, add its relations (with prefix "2hop: label -> ")
        if key in label_to_related:
            for item in label_to_related[key]:
                related_name = item[0]
                rtype1 = item[2] if len(item) >= 3 else "other"
                related_norm = _normalize_term(related_name)
                if not related_norm or related_norm == key:
                    continue
                add_triples_for_label(related_norm, 2, prefix=f"2hop {related_name} -> ")

    if not parts:
        return ""
    return sep + sep.join(parts)


# Query phrases users type that map to ontology nodes (e.g. Cognitive Atlas has "n-back", not "2-back").
# When the normalized query contains the key, we also fetch triples for the ontology label(s) in the value.
QUERY_VARIANT_TO_ONTOLOGY_LABELS: dict[str, list[str]] = {
    "2-back": ["n-back"],
    "0-back": ["n-back"],
    "1-back": ["n-back"],
    "3-back": ["n-back"],
}


def get_kg_context_for_query(
    query: str,
    index: dict[str, Any],
    max_hops: int = 1,
    max_per_relation: int = 15,
    sep: str = " | ",
) -> str:
    """
    KG context for free-text queries: triples for every ontology concept that appears in the query.

    So the encoder sees one connected block, e.g.:
      "image doing a 2-back task | parent: working memory, n-back | child: ... | 2hop working memory -> parent: executive function"

    - If the full query (normalized) is an ontology label, returns get_kg_context(query, ...).
    - Else finds all ontology labels that appear as a substring of the normalized query
      (e.g. "2-back" in "image doing a 2-back task") and concatenates their triple strings.
    - Also maps query variants to ontology nodes: e.g. "2-back" -> "n-back" so we get triples for the
      n-back node even when the ontology only has "n-back" (see QUERY_VARIANT_TO_ONTOLOGY_LABELS).
    - Labels are processed longest-first so "working memory" is used before "memory".
    - Use Cognitive Atlas (and CogPO) in ontology_dir for task concepts (n-back, working memory, etc.).
    """
    label_to_related = index.get("label_to_related") or {}
    key = _normalize_term(query)
    if not key:
        return ""
    # Full match: query is exactly an ontology concept
    if key in label_to_related:
        return get_kg_context(query, index, max_hops=max_hops, max_per_relation=max_per_relation, sep=sep)
    # Substring match: every ontology label that appears in the query (longest first)
    matches = sorted([label for label in label_to_related if label and label in key], key=lambda x: -len(x))
    chosen: list[str] = []
    covered: set[int] = set()
    for label in matches:
        start = key.find(label)
        if start == -1:
            continue
        end = start + len(label)
        if any(i in covered for i in range(start, end)):
            continue
        chosen.append(label)
        for i in range(start, end):
            covered.add(i)
    # Variant expansion: e.g. "2-back" in query -> also fetch triples for "n-back" if in ontology
    for variant_substr, ontology_labels in QUERY_VARIANT_TO_ONTOLOGY_LABELS.items():
        if variant_substr not in key:
            continue
        for ont_label in ontology_labels:
            if ont_label in label_to_related and ont_label not in chosen:
                chosen.append(ont_label)
    if not chosen:
        return ""
    parts = [get_kg_context(label, index, max_hops=max_hops, max_per_relation=max_per_relation, sep=sep) for label in chosen]
    combined = "".join(p for p in parts if p).strip(sep).strip()
    return (sep + combined) if combined else ""


def build_embedding_text_for_label(
    label: str,
    index: dict[str, Any],
    max_synonyms: int = 5,
    max_parents: int = 3,
    max_children: int = 3,
    max_other: int = 3,
    max_def_words: int = 0,
    sep: str = " | ",
) -> str:
    """
    Build rich text for a single ontology term so the embedding captures its neighborhood.

    Instead of embedding just "n-back", embed e.g.:
      "n-back | Also known as: n-back task; working memory n-back | Type of: working memory task |
       Related: working memory; executive function"

    Gives the encoder more semantic surface for informal-to-formal matching. Use with
    build_ontology_label_embeddings(..., use_rich_text=True).
    """
    label_to_related = index.get("label_to_related") or {}
    related = label_to_related.get(label, [])
    if not related:
        return label
    by_type: dict[str, list[str]] = {}
    for item in related:
        name = (item[0] or "").strip()
        rtype = item[2] if len(item) >= 3 else "other"
        if not name or name == label:
            continue
        by_type.setdefault(rtype, []).append(name)
    parts = [label]
    syns = by_type.get("synonym", [])[:max_synonyms]
    if syns:
        parts.append("Also known as: " + "; ".join(syns))
    parents = by_type.get("parent", [])[:max_parents]
    if parents:
        parts.append("Type of: " + ", ".join(parents))
    children = by_type.get("child", [])[:max_children]
    if children:
        parts.append("Includes: " + ", ".join(children))
    other = (by_type.get("other", []) + [x for k, v in by_type.items() if k not in ("synonym", "parent", "child", "self") for x in v])[:max_other]
    if other:
        parts.append("Related: " + ", ".join(other))
    return sep.join(parts)


def build_ontology_label_embeddings(
    index: dict[str, Any],
    encode_fn: Any,
    batch_size: int = 64,
    use_rich_text: bool = True,
    embed_sources: set[str] | list[str] | None = None,
) -> tuple[Any, list[str]]:
    """
    Embed ontology labels with the given encoder for cosine-similarity retrieval.

    encode_fn(texts: list[str]) -> np.ndarray of shape (n, dim). Returns (embeddings, label_list)
    where label_list[i] corresponds to embeddings[i]. Use the same encoder as text-to-brain for
    consistent semantics.

    If use_rich_text is True (default), each label is expanded with synonyms, parents, children,
    and related terms (build_embedding_text_for_label) so the embedding captures ontological
    neighborhood — better informal-to-formal matching (e.g. "2-back" → "n-back").

    embed_sources: If set, only embed labels that come from these ontology filenames (e.g.
    {"cogat.v2.owl", "mf.owl", "nbo.owl", "CogPOver1.owl"}). Use to avoid embedding large
    clinical ontologies (MONDO, HPO, ChEBI) with the API; they remain in the graph for
    substring matching and expansion. If None, embed all labels.
    """
    import numpy as np
    label_to_related = index.get("label_to_related") or {}
    label_to_source = index.get("label_to_source") or {}
    label_list = sorted(label_to_related.keys())
    if embed_sources is not None:
        sources_set = set(embed_sources)
        label_list = [l for l in label_list if label_to_source.get(l, "") in sources_set]
    if not label_list:
        return np.zeros((0, 1), dtype=np.float32), []
    if use_rich_text:
        texts = [build_embedding_text_for_label(l, index) for l in label_list]
    else:
        texts = label_list
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = encode_fn(batch)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings.append(emb)
    return np.vstack(embeddings), label_list


def get_kg_context_for_query_semantic(
    query: str,
    index: dict[str, Any],
    encode_fn: Any,
    label_embeddings: Any,
    label_list: list[str],
    top_k: int = 5,
    max_hops: int = 1,
    max_per_relation: int = 15,
    sep: str = " | ",
) -> str:
    """
    KG context by embedding similarity: embed the query, find the top-k closest ontology nodes
    by cosine similarity, and concatenate their triple strings.

    So "image doing a 2-back task" will pull in triples for n-back, working memory, etc. even
    when those words don't appear literally. Use the same encoder as the text-to-brain model.
    """
    import numpy as np
    if not label_list or label_embeddings is None or getattr(label_embeddings, "shape", (0,))[0] == 0:
        return ""
    q_emb = encode_fn([query])
    q_emb = np.asarray(q_emb, dtype=np.float32).reshape(1, -1)
    L = np.asarray(label_embeddings, dtype=np.float32)
    if q_emb.shape[1] != L.shape[1]:
        return ""
    norms = np.linalg.norm(L, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    L_norm = L / norms
    sims = np.dot(L_norm, q_emb.ravel())
    k = min(top_k, len(label_list))
    top_idx = np.argsort(-sims.ravel())[:k]
    chosen = [label_list[i] for i in top_idx]
    parts = [get_kg_context(label, index, max_hops=max_hops, max_per_relation=max_per_relation, sep=sep) for label in chosen]
    combined = "".join(p for p in parts if p).strip(sep).strip()
    return (sep + combined) if combined else ""


def format_triple(subject: str, relation: str, obj: str) -> str:
    """
    Format (subject, relation, obj) as natural language for the encoder.
    OpenAI embeddings handle structured text well; these templates give consistent semantics.
    CogAt uses parent/child (is_a), synonym, and optionally measures / is_measured_by.
    """
    subject = (subject or "").strip()
    obj = (obj or "").strip()
    if not subject or not obj:
        return ""
    relation = (relation or "other").strip().lower()
    templates: dict[str, str] = {
        "synonym": f"{subject} ({obj})",
        "child": f"{obj} is a type of {subject}",
        "parent": f"{subject} is a type of {obj}",
        "self": f"{subject}",
        "measures": f"{subject} measures {obj}",
        "is_measured_by": f"{obj} measures {subject}",
        "part_of": f"{subject} is part of {obj}",
        "has_part": f"{subject} includes {obj}",
        "is_a": f"{subject} is a type of {obj}",
    }
    if relation in templates:
        out = templates[relation]
        return out if out != subject else ""
    return f"{subject} {relation} {obj}"


def get_kg_augmentation(
    query_text: str,
    query_embedding: Any,
    label_embeddings: Any,
    label_names: list[str],
    index: dict[str, Any],
    top_k: int = 5,
    sim_floor: float = 0.4,
    max_triples: int = 15,
) -> str:
    """
    Select top-k ontology nodes by embedding similarity (with sim_floor), gather their
    depth-1 triples, format as natural language, and return the top max_triples by
    (query_sim × relation_weight). Prepended to the query before encoding.

    Depth 1 only: no 2-hop expansion, so we don't dilute with generic concepts.
    """
    import numpy as np
    label_to_related = index.get("label_to_related") or {}
    if not label_names or label_embeddings is None or getattr(label_embeddings, "shape", (0,))[0] == 0:
        return ""
    q = np.asarray(query_embedding, dtype=np.float32).ravel()
    L = np.asarray(label_embeddings, dtype=np.float32)
    if q.shape[0] != L.shape[1]:
        return ""
    norms = np.linalg.norm(L, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    L_norm = L / norms
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return ""
    sims = np.dot(L_norm, q / q_norm)
    # Filter by sim_floor, take top_k nodes (oversample then filter)
    order = np.argsort(-sims.ravel())
    candidates: list[tuple[str, float]] = []
    for i in order[: top_k * 2]:
        if sims.flat[i] >= sim_floor:
            candidates.append((label_names[i], float(sims.flat[i])))
        if len(candidates) >= top_k:
            break
    candidates = candidates[:top_k]
    if not candidates:
        return ""
    gathered: list[tuple[str, float]] = []
    for label, sim in candidates:
        related = label_to_related.get(label, [])
        for item in related:
            rel_name = item[0] if len(item) >= 1 else ""
            weight = float(item[1]) if len(item) >= 2 else 1.0
            rtype = item[2] if len(item) >= 3 else "other"
            if not rel_name or rtype == "self":
                continue
            triple = format_triple(label, rtype, rel_name.strip())
            if triple:
                gathered.append((triple, sim * weight))
    gathered.sort(key=lambda x: -x[1])
    selected = [t[0] for t in gathered[:max_triples]]
    if not selected:
        return ""
    return ". ".join(selected) + ". "


def get_map_for_term(
    term: str,
    decoder_maps: Any,
    term_vocab: list[str],
    index: dict[str, Any],
    encoder: Any = None,
    cache_embeddings: Any = None,
    top_k_similarity: int = 10,
    direction_scale: dict[str, float] | None = None,
) -> Any | None:
    """
    Return a 400-D (or N_parcel) map for term: cache hit, else ontology expansion,
    else cosine-similarity fallback (if encoder + cache_embeddings provided).
    Returns None if no map available.
    """
    import numpy as np
    if hasattr(decoder_maps, "shape"):
        pass
    else:
        decoder_maps = np.asarray(decoder_maps)
    vocab_to_idx = {t: i for i, t in enumerate(term_vocab)}
    norm_to_idx = {}
    for i, v in enumerate(term_vocab):
        k = _normalize_term(v)
        if k and k not in norm_to_idx:
            norm_to_idx[k] = i
    # Direct hit in cache
    key = _normalize_term(term)
    if key in norm_to_idx:
        return decoder_maps[norm_to_idx[key]].copy()
    # Ontology expansion (with optional direction scaling: parent too broad → downweight)
    scale = direction_scale or DIRECTION_SCALE
    related = expand_term(term, term_vocab, index)
    if related:
        pairs: list[tuple[int, float]] = []
        for item in related:
            t = item[0]
            w = item[1]
            rtype = item[2] if len(item) >= 3 else "other"
            adj = scale.get(rtype, 0.8)
            i = vocab_to_idx.get(t)
            if i is not None:
                pairs.append((i, w * adj))
        if pairs:
            indices = [p[0] for p in pairs]
            weights = np.array([p[1] for p in pairs], dtype=float)
            weights = weights / weights.sum()
            return np.average(decoder_maps[indices], axis=0, weights=weights)
    # Cosine-similarity fallback: query embedding vs cache embeddings -> top_k -> blend maps
    if encoder is not None and cache_embeddings is not None:
        try:
            q_emb = np.asarray(encoder(term) if callable(encoder) else encoder.encode(term)).ravel()
            cache_emb = np.asarray(cache_embeddings)
            if cache_emb.shape[0] != len(term_vocab):
                return None
            # cosine sim: (q @ cache_emb.T) / (||q|| ||cache_emb||)
            q_norm = np.linalg.norm(q_emb)
            if q_norm <= 0:
                return None
            sims = np.dot(cache_emb, q_emb) / (q_norm * (np.linalg.norm(cache_emb, axis=1) + 1e-12))
            top = np.argsort(-sims)[:top_k_similarity]
            weights = np.maximum(sims[top], 0.0)
            if weights.sum() <= 0:
                return None
            weights = weights / weights.sum()
            return np.average(decoder_maps[top], axis=0, weights=weights)
        except Exception:
            pass
    return None


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/ontologies"
    idx = load_ontology_index(d)
    label_to_related = idx.get("label_to_related") or {}
    n = len(label_to_related)
    print(f"Loaded {n} ontology labels from {d}")
    if len(sys.argv) > 2:
        term = " ".join(sys.argv[2:])
        key = _normalize_term(term)
        if key in label_to_related:
            print(f"Ontology related terms for {term!r}: {label_to_related[key][:15]}")
        else:
            related = expand_term(term, [], idx)
            print(f"expand_term({term!r}) (no decoder vocab) -> {related[:15]}")
