#!/usr/bin/env python3
"""
Two ways to go from a full query string to one brain map:

1. NeuroQuery-style (default): query -> weights over cache terms -> map = weights @ term_maps
   (combination in term space, then one linear map to brain space). No ontology.

2. Decompose -> average: query -> concepts (e.g. LLM or rules) -> one map per concept -> average maps
   (combination in brain space).

Ontology is NOT used by default. It is an optional path when terms are OOV (not in cache)
or similarity to the cache is low: we then use ontology to find ontologically related terms
that are in the cache and do the mapping via those cache terms (e.g. weighted average).
Fall back to (1) when ontology returns no cache terms.

Both can be compared with compare_query_methods.py.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from ontology_expansion import expand_term
except ImportError:
    expand_term = None


def _normalize_term(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ")


def _compute_weights_neuroquery_style(
    query: str,
    term_vocab: list[str],
    encoder: Any = None,
    cache_embeddings: np.ndarray | None = None,
    use_tfidf_fallback: bool = True,
) -> np.ndarray | None:
    """Compute weights over cache terms (cosine similarity or TF-IDF). Returns (n_terms,) or None."""
    n_terms = len(term_vocab)
    weights = None
    if encoder is not None and cache_embeddings is not None:
        try:
            q_emb = np.asarray(
                encoder(query) if callable(encoder) else encoder.encode(query),
                dtype=np.float64,
            ).ravel()
            cache_emb = np.asarray(cache_embeddings, dtype=np.float64)
            if cache_emb.shape[0] == n_terms:
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 1e-12:
                    sims = np.dot(cache_emb, q_emb) / (
                        q_norm * (np.linalg.norm(cache_emb, axis=1) + 1e-12)
                    )
                    weights = np.maximum(sims, 0.0)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
        except Exception:
            weights = None

    if weights is None and use_tfidf_fallback:
        query_tokens = set(_normalize_term(query).split())
        weights = np.zeros(n_terms, dtype=np.float64)
        for i, t in enumerate(term_vocab):
            term_tokens = set(_normalize_term(t).split())
            overlap = len(query_tokens & term_tokens) / max(len(term_tokens), 1)
            if overlap > 0 or query_tokens & term_tokens:
                weights[i] = overlap
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_terms) / n_terms

    return weights


def get_map_neuroquery_style(
    query: str,
    term_maps: np.ndarray,
    term_vocab: list[str],
    encoder: Any = None,
    cache_embeddings: np.ndarray | None = None,
    use_tfidf_fallback: bool = True,
) -> np.ndarray | None:
    """
    Default path: one map from query via weights over cache terms -> weighted sum of term_maps.
    No ontology. Weights from (a) encoder + cache_embeddings (cosine similarity), or
    (b) simple TF-IDF-style: overlap of query words with each term, normalized.

    term_maps: (n_terms, n_parcels). term_vocab: list of n_terms strings.
    """
    term_maps = np.asarray(term_maps, dtype=np.float64)
    n_terms = len(term_vocab)
    if term_maps.shape[0] != n_terms:
        return None

    weights = _compute_weights_neuroquery_style(
        query, term_vocab,
        encoder=encoder,
        cache_embeddings=cache_embeddings,
        use_tfidf_fallback=use_tfidf_fallback,
    )
    if weights is None:
        return None
    return np.dot(weights, term_maps)


def get_map_neuroquery_style_with_ontology(
    query: str,
    term_maps: np.ndarray,
    term_vocab: list[str],
    ontology_index: dict[str, Any] | None,
    encoder: Any = None,
    cache_embeddings: np.ndarray | None = None,
    use_tfidf_fallback: bool = True,
) -> np.ndarray | None:
    """
    Optional path: when terms are OOV or similarity to the cache is low, use ontology to
    find ontologically related terms that are in the cache, then do the mapping via those
    cache terms (weighted sum of their maps). Resolves query to cache terms via ontology;
    if ontology returns related cache terms, uses their weights and weighted sum; otherwise
    falls back to get_map_neuroquery_style (default path). Use when OOV or low similarity.
    """
    term_maps = np.asarray(term_maps, dtype=np.float64)
    n_terms = len(term_vocab)
    if term_maps.shape[0] != n_terms:
        return None

    if ontology_index is not None and expand_term is not None:
        related = expand_term(query, term_vocab, ontology_index)
        if related:
            weights = np.zeros(n_terms, dtype=np.float64)
            for item in related:
                t, w = item[0], item[1]
                try:
                    i = term_vocab.index(t)
                    weights[i] = weights[i] + w
                except ValueError:
                    continue
            if weights.sum() > 0:
                weights = weights / weights.sum()
                return np.dot(weights, term_maps)

    return get_map_neuroquery_style(
        query,
        term_maps,
        term_vocab,
        encoder=encoder,
        cache_embeddings=cache_embeddings,
        use_tfidf_fallback=use_tfidf_fallback,
    )


def get_map_with_ontology_on_low_similarity(
    query: str,
    term_maps: np.ndarray,
    term_vocab: list[str],
    ontology_index: dict[str, Any] | None,
    similarity_threshold: float = 0.15,
    similarity_threshold_ontology: float | None = None,
    encoder: Any = None,
    cache_embeddings: np.ndarray | None = None,
    use_tfidf_fallback: bool = True,
) -> tuple[np.ndarray | None, bool]:
    """
    Get map from query: use cache-term weights first; if max similarity < threshold,
    use ontology to find related cache terms and map via those (for OOV / low-similarity).
    Ontology is only used if the query's similarity to those ontology-derived cache terms
    is above similarity_threshold_ontology (threshold B); otherwise fall back to cache weights.

    Returns (map_400, used_ontology). used_ontology is True when ontology fallback was used.
    """
    if similarity_threshold_ontology is None:
        similarity_threshold_ontology = similarity_threshold

    term_maps = np.asarray(term_maps, dtype=np.float64)
    n_terms = len(term_vocab)
    if term_maps.shape[0] != n_terms:
        return None, False

    weights = _compute_weights_neuroquery_style(
        query, term_vocab,
        encoder=encoder,
        cache_embeddings=cache_embeddings,
        use_tfidf_fallback=use_tfidf_fallback,
    )
    if weights is None:
        return None, False

    max_sim = float(np.max(weights))
    if max_sim >= similarity_threshold:
        return np.dot(weights, term_maps), False

    # Low direct similarity: try ontology expansion to related cache terms
    if ontology_index is not None and expand_term is not None:
        related = expand_term(query, term_vocab, ontology_index)
        if related:
            ont_weights = np.zeros(n_terms, dtype=np.float64)
            ont_indices = []
            for item in related:
                t, w = item[0], item[1]
                try:
                    i = term_vocab.index(t)
                    ont_weights[i] = ont_weights[i] + w
                    ont_indices.append(i)
                except ValueError:
                    continue
            if ont_weights.sum() > 0:
                # Only use ontology if query similarity to ontology-derived terms is above threshold B
                max_sim_ontology_terms = float(np.max(weights[ont_indices])) if ont_indices else 0.0
                if max_sim_ontology_terms >= similarity_threshold_ontology:
                    ont_weights = ont_weights / ont_weights.sum()
                    return np.dot(ont_weights, term_maps), True
                # else: ontology terms exist but are not "close enough"; fall back to cache weights
    return np.dot(weights, term_maps), False


def get_map_decompose_avg(
    query: str,
    mapper: Any,
    decomposer: Callable[[str], list[str]] | None = None,
) -> np.ndarray | None:
    """
    One map from query: decompose into concepts → get_map per concept → average maps.

    decomposer(query) returns list of concept strings. If None, use [query] as single concept.
    """
    if decomposer is not None:
        concepts = decomposer(query)
    else:
        concepts = [query]
    maps = []
    for c in concepts:
        m = mapper.get_map(c)
        if m is not None:
            maps.append(np.asarray(m).ravel())
    if not maps:
        return None
    return np.mean(maps, axis=0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1D arrays."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    if aa <= 0 or bb <= 0:
        return float("nan")
    return float(np.dot(a, b) / np.sqrt(aa * bb))
