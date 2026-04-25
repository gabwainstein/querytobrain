#!/usr/bin/env python3
"""
Combined term → map: resolve a text query to a parcellated map from
the cognitive cache (NeuroQuery-style + ontology), neuromaps biological cache,
or abagen gene expression (retrieval only; Option B).

Enables querying biological terms by text, e.g.:
  "alpha2a PET receptor expression", "5-HT2a", "myelin density", "HTR2A", "DRD2 gene"

Usage (from scripts or query.py):
  from term_to_map import get_map_from_neuromaps, get_map_from_abagen, get_parcellated_map_combined
  map_nm, label = get_map_from_neuromaps("myelin density", neuromaps_cache_dir)
  map_ab, label = get_map_from_abagen("HTR2A", abagen_cache_dir)  # Option B: retrieval only
  map_400, source = get_parcellated_map_combined(term, cache_dir, neuromaps_dir, ..., abagen_cache_dir=...)
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Repo root for imports
def _repo_root():
    return Path(__file__).resolve().parent.parent.parent


def _normalize_for_match(s: str) -> str:
    """Lowercase, collapse punctuation and spaces, for matching."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_.,;:]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _token_set(s: str) -> set[str]:
    return set(_normalize_for_match(s).split()) - {""}


# Gene/abagen aliases: query-friendly → substrings that may appear in abagen labels (Gene: ... (SYMBOL))
_GENE_ALIASES = {
    "htr2a": ["htr2a", "5ht2a", "5-ht2a", "serotonin 2a"],
    "drd2": ["drd2", "d2", "dopamine d2"],
    "slc6a4": ["slc6a4", "sert", "serotonin transporter"],
    "bdnf": ["bdnf", "brain derived neurotrophic"],
    "comt": ["comt", "catechol o methyltransferase"],
    "maoa": ["maoa", "monoamine oxidase"],
    "gad1": ["gad1", "gad67", "gaba"],
    "pvalb": ["pvalb", "parvalbumin"],
    "sst": ["sst", "somatostatin"],
}

# Common aliases: query-friendly → substrings that may appear in neuromaps labels
_BIOLOGICAL_ALIASES = {
    "alpha2a": ["a2a", "alpha2a", "alpha 2a", "adra2a", "noradrenaline"],
    "alpha2": ["a2a", "alpha2", "adra2"],
    "5ht1a": ["5ht1a", "5-ht1a", "htr1a", "serotonin"],
    "5ht2a": ["5ht2a", "5-ht2a", "htr2a"],
    "dopamine": ["dopamine", "d1", "d2", "dat"],
    "pet": ["pet", "receptor", "binding"],
    "rna": ["rna", "gene", "expression", "ahba"],
    "myelin": ["myelin", "bigbrain"],
    "receptor": ["receptor", "binding", "pet", "hansen"],
    "density": ["density", "expression", "level"],
}


def _expand_query_tokens(query: str) -> set[str]:
    """Expand query with biological aliases to improve match to neuromaps labels."""
    qnorm = _normalize_for_match(query)
    tokens = _token_set(query) | set(qnorm.split())
    expanded = set(tokens)
    for alias, subs in _BIOLOGICAL_ALIASES.items():
        if alias in tokens or any(a in qnorm for a in [alias]):
            expanded.update(subs)
    return expanded


def match_query_to_abagen_labels(
    query: str,
    labels: list[str],
    min_score: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Rank abagen labels by match to query. Returns list of (index, score).
    Handles gene symbols (HTR2A, DRD2), aliases, and substring match.
    """
    q_norm = _normalize_for_match(query)
    q_tokens = _token_set(query) | set(q_norm.split())
    # Expand with gene aliases
    for alias, subs in _GENE_ALIASES.items():
        if alias in q_tokens or any(a in q_norm for a in [alias]):
            q_tokens.update(subs)
    scored: list[tuple[int, float]] = []
    for i, label in enumerate(labels):
        label_norm = _normalize_for_match(label)
        label_tokens = _token_set(label)
        overlap = len(q_tokens & label_tokens) / max(len(q_tokens), 1)
        overlap += len(q_tokens & label_tokens) / max(len(label_tokens), 1)
        if q_norm in label_norm or label_norm in q_norm:
            overlap += 1.0
        for t in q_tokens:
            if t in label_norm:
                overlap += 0.5
        for t in label_tokens:
            if t in q_norm:
                overlap += 0.5
        if overlap >= min_score:
            scored.append((i, min(overlap, 10.0)))
    scored.sort(key=lambda x: -x[1])
    return scored


def get_map_from_abagen(
    query: str,
    abagen_cache_dir: str | Path,
    n_parcels: int = 427,
    min_match_score: float = 0.3,
) -> tuple[np.ndarray | None, str | None]:
    """
    Resolve a text query to a single parcellated map from the abagen cache
    by matching query to gene labels (e.g. "HTR2A", "DRD2 gene expression").

    Returns (map, label) if a match is found, else (None, None).
    Use for retrieval/annotation; abagen is NOT in the regression training set (Option B).
    """
    abagen_cache_dir = Path(abagen_cache_dir)
    npz_path = abagen_cache_dir / "term_maps.npz"
    pkl_path = abagen_cache_dir / "term_vocab.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        return None, None
    data = np.load(npz_path)
    maps = np.asarray(data["term_maps"], dtype=np.float64)
    with open(pkl_path, "rb") as f:
        labels = pickle.load(f)
    labels = list(labels)
    if maps.shape[0] != len(labels) or maps.shape[1] != n_parcels:
        return None, None
    ranked = match_query_to_abagen_labels(query, labels, min_score=min_match_score)
    if not ranked:
        return None, None
    best_idx, _ = ranked[0]
    return np.asarray(maps[best_idx], dtype=np.float64).ravel(), labels[best_idx]


def match_query_to_neuromaps_labels(
    query: str,
    labels: list[str],
    min_score: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Rank neuromaps labels by match to query. Returns list of (index, score).
    Score: token overlap and substring match; higher = better match.
    """
    q_tokens = _expand_query_tokens(query)
    q_norm = _normalize_for_match(query)
    scored: list[tuple[int, float]] = []
    for i, label in enumerate(labels):
        label_norm = _normalize_for_match(label)
        label_tokens = _token_set(label)
        # Token overlap (query in label, label in query)
        overlap = len(q_tokens & label_tokens) / max(len(q_tokens), 1)
        overlap += len(q_tokens & label_tokens) / max(len(label_tokens), 1)
        # Substring: query contained in label or label in query
        if q_norm in label_norm or label_norm in q_norm:
            overlap += 1.0
        for t in q_tokens:
            if t in label_norm:
                overlap += 0.5
        for t in label_tokens:
            if t in q_norm:
                overlap += 0.5
        if overlap >= min_score:
            scored.append((i, min(overlap, 10.0)))
    scored.sort(key=lambda x: -x[1])
    return scored


def get_map_from_neuromaps(
    query: str,
    neuromaps_cache_dir: str | Path,
    n_parcels: int = 400,
    min_match_score: float = 0.3,
) -> tuple[np.ndarray | None, str | None]:
    """
    Resolve a text query to a single parcellated map from the neuromaps cache
    by matching query to annotation labels (e.g. "myelin", "alpha2a PET receptor", "RNA").

    Returns (map_400, label) if a match is found, else (None, None).
    If multiple labels match, returns the best-matching map (or mean of top 2 if very close scores).
    """
    neuromaps_cache_dir = Path(neuromaps_cache_dir)
    npz_path = neuromaps_cache_dir / "annotation_maps.npz"
    pkl_path = neuromaps_cache_dir / "annotation_labels.pkl"
    if not npz_path.exists() or not pkl_path.exists():
        return None, None
    data = np.load(npz_path)
    matrix = data["matrix"] if "matrix" in data.files else data[data.files[0]]
    matrix = np.asarray(matrix, dtype=np.float64)
    with open(pkl_path, "rb") as f:
        labels = pickle.load(f)
    labels = list(labels)
    if matrix.shape[0] != len(labels) or matrix.shape[1] != n_parcels:
        return None, None

    ranked = match_query_to_neuromaps_labels(query, labels, min_score=min_match_score)
    if not ranked:
        return None, None
    best_idx, best_score = ranked[0]
    # Optional: blend top 2 if scores are close
    if len(ranked) >= 2 and ranked[1][1] >= best_score * 0.8:
        idx2 = ranked[1][0]
        map_400 = (matrix[best_idx] + matrix[idx2]) / 2.0
        label = f"{labels[best_idx]}+{labels[idx2]}"
    else:
        map_400 = np.asarray(matrix[best_idx], dtype=np.float64).ravel()
        label = labels[best_idx]
    return map_400, label


def get_parcellated_map_combined(
    term: str,
    cache_dir: str,
    neuromaps_cache_dir: str | None,
    ontology_dir: str | None,
    similarity_threshold: float,
    similarity_threshold_ontology: float | None,
    use_ontology: bool,
    prefer_neuromaps_if_matched: bool = True,
    n_parcels: int = 400,
    abagen_cache_dir: str | Path | None = None,
    # Optional: get_parcellated_map_from_cache_and_ontology and fallbacks
    _get_cognitive: Any = None,
    _get_neuroquery: Any = None,
) -> tuple[np.ndarray, str]:
    """
    Combined term → map: try neuromaps by name first; then abagen (gene retrieval); then cognitive path
    (cache + ontology). So biological queries like "myelin density", "HTR2A", "alpha2a receptor"
    return the stored map; cognitive queries like "attention" use cache/ontology.

    Returns (map_400, source) where source is "neuromaps", "abagen", "cognitive", "ontology", or "neuroquery".
    """
    # 1) Try neuromaps by label match
    if neuromaps_cache_dir and prefer_neuromaps_if_matched:
        map_nm, label_nm = get_map_from_neuromaps(term, neuromaps_cache_dir, n_parcels=n_parcels)
        if map_nm is not None and label_nm:
            return map_nm, "neuromaps"

    # 2) Try abagen by name (gene retrieval; Option B: not in regression, retrieval only)
    if abagen_cache_dir:
        map_ab, label_ab = get_map_from_abagen(term, abagen_cache_dir, n_parcels=n_parcels)
        if map_ab is not None and label_ab:
            return map_ab, "abagen"

    # 3) Cognitive path (cache + ontology)
    if _get_cognitive is not None:
        map_cog, used_ontology = _get_cognitive(
            term,
            cache_dir,
            ontology_dir,
            similarity_threshold,
            similarity_threshold_ontology,
            use_ontology,
        )
        if map_cog is not None:
            return map_cog, "ontology" if used_ontology else "cognitive"

    # 4) NeuroQuery fallback
    if _get_neuroquery is not None:
        try:
            map_nq = _get_neuroquery(term)
            if map_nq is not None:
                return map_nq, "neuroquery"
        except Exception:
            pass

    # 5) Return zeros if nothing worked (caller may check)
    return np.zeros(n_parcels, dtype=np.float64), "none"


def main() -> int:
    """CLI: resolve term to map and print source (neuromaps label or cognitive)."""
    parser = __import__("argparse").ArgumentParser(
        description="Combined term → map (cognitive + neuromaps by name)"
    )
    parser.add_argument("term", help="Query term (e.g. attention, myelin density, alpha2a receptor)")
    parser.add_argument("--cache-dir", default=None, help="Cognitive cache dir")
    parser.add_argument("--neuromaps-cache-dir", default=None, help="Neuromaps cache dir")
    parser.add_argument("--ontology-dir", default=None, help="Ontology dir for low-similarity")
    parser.add_argument("--no-neuromaps-first", action="store_true", help="Do not try neuromaps by name first")
    parser.add_argument("--abagen-cache-dir", default=None, help="Abagen cache dir for gene retrieval (Option B)")
    parser.add_argument("--list-neuromaps", action="store_true", help="List neuromaps labels and exit")
    args = parser.parse_args()

    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, str(repo))
    data_dir = repo / "neurolab" / "data"
    cache_dir = args.cache_dir or str(data_dir / "unified_cache")
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(repo, cache_dir)
    nm_dir = args.neuromaps_cache_dir or str(data_dir / "neuromaps_cache")
    if nm_dir and not os.path.isabs(nm_dir):
        nm_dir = os.path.join(repo, nm_dir)
    ab_dir = args.abagen_cache_dir or str(data_dir / "abagen_cache")
    if ab_dir and not os.path.isabs(ab_dir):
        ab_dir = os.path.join(repo, ab_dir)
    if not (Path(ab_dir) / "term_maps.npz").exists():
        ab_dir = None

    if args.list_neuromaps:
        pkl_path = Path(nm_dir) / "annotation_labels.pkl"
        if not pkl_path.exists():
            print("Neuromaps cache not found.", file=sys.stderr)
            return 1
        with open(pkl_path, "rb") as f:
            labels = pickle.load(f)
        for i, L in enumerate(labels):
            print(f"  {i}: {L}")
        return 0

    # Resolve map
    def _get_cognitive(t, c, o, st, sto, uo):
        scripts = repo / "neurolab" / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from query import get_parcellated_map_from_cache_and_ontology
        return get_parcellated_map_from_cache_and_ontology(t, c, o, st, sto, uo)

    def _get_nq(t):
        from query import get_parcellated_map_for_term
        return get_parcellated_map_for_term(t)

    n_parcels = 427
    try:
        from neurolab.parcellation import get_n_parcels
        n_parcels = get_n_parcels()
    except Exception:
        pass
    map_400, source = get_parcellated_map_combined(
        args.term.strip(),
        cache_dir=cache_dir,
        neuromaps_cache_dir=nm_dir if os.path.exists(os.path.join(nm_dir, "annotation_maps.npz")) else None,
        ontology_dir=args.ontology_dir or (str(data_dir / "ontologies") if (data_dir / "ontologies").is_dir() else None),
        similarity_threshold=0.15,
        similarity_threshold_ontology=None,
        use_ontology=True,
        prefer_neuromaps_if_matched=not args.no_neuromaps_first,
        n_parcels=n_parcels,
        abagen_cache_dir=ab_dir,
        _get_cognitive=_get_cognitive,
        _get_neuroquery=_get_nq,
    )
    print(f"Source: {source}")
    print(f"Map shape: {map_400.shape}, norm: {float(np.linalg.norm(map_400)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
