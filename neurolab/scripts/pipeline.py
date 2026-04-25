#!/usr/bin/env python3
"""
Full NeuroLab pipeline: query -> concepts -> (cache/ontology/similarity) per concept
-> combine maps -> optional receptor map.

  - Step 1: Decompose query into concepts (rule-based or custom decomposer).
  - Step 2: Get a cognitive map per concept via OptionAMapper (cache, then ontology
    when OOV/low similarity, then cosine-similarity fallback).
  - Step 3: Combine maps: either "neuroquery_style" (weights over cache terms -> one map)
    or "decompose_avg" (one map per concept -> average).
  - Step 4 (optional): Fetch receptor/structure map from neuromaps (e.g. dopamine).
  - Step 5: Return composite cognitive map + optional receptor info + metadata.

Usage:
  from pipeline import run_pipeline, decompose_query, simple_rule_decomposer
  result = run_pipeline("see my dog after ritalin", mapper, combine="decompose_avg", decomposer=simple_rule_decomposer)
  # result["cognitive_map"], result["concepts_used"], result["receptor_info"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_scripts))

from option_a import OptionAMapper, _default_ontology_dir
from query_to_map import (
    get_map_decompose_avg,
    get_map_neuroquery_style,
    get_map_neuroquery_style_with_ontology,
)


def simple_rule_decomposer(query: str) -> list[str]:
    """
    Rule-based concept extraction: split on ' and ', ', ', ' after ', ' when ', ' + '.
    Normalize and dedupe. Use when no LLM is available.
    """
    if not (query or query.strip()):
        return []
    # Split on common conjunctions and punctuation
    parts = re.split(r"\s+and\s+|\s*,\s*|\s+after\s+|\s+when\s+|\s*\+\s*", query, flags=re.IGNORECASE)
    concepts = []
    seen: set[str] = set()
    for p in parts:
        t = p.strip()
        if not t:
            continue
        # Normalize for dedupe (lowercase, collapse spaces)
        key = " ".join(t.lower().split())
        if key not in seen:
            seen.add(key)
            concepts.append(t)
    return concepts if concepts else [query.strip()]


def decompose_query(
    query: str,
    decomposer: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """
    Turn query into a list of concept strings.
    If decomposer is None, use simple_rule_decomposer.
    """
    if decomposer is not None:
        return decomposer(query)
    return simple_rule_decomposer(query)


def _fetch_receptor_info(receptor_name: str) -> dict[str, Any] | None:
    """
    Optional: fetch receptor (or tag) map from neuromaps.
    Returns dict with keys like "paths", "source", "desc", or None if neuromaps unavailable.
    """
    try:
        from neuromaps.datasets import fetch_annotation, available_annotations
    except ImportError:
        return None
    name_lower = receptor_name.strip().lower()
    try:
        ann = available_annotations()
        if hasattr(ann, "to_dict"):
            rows = ann.to_dict(orient="records")
        elif hasattr(ann, "iterrows"):
            rows = [r for _, r in ann.iterrows()]
        else:
            rows = []
        row = None
        for r in rows:
            desc = (r.get("desc") or r.get("description") or "").lower()
            source = (r.get("source") or "").lower()
            if name_lower in desc or name_lower in source:
                row = r
                break
        if row is None:
            result = fetch_annotation(desc=receptor_name)
            return {"paths": result, "desc": receptor_name}
        src = row.get("source") or row.get("name")
        desc = row.get("desc") or row.get("description") or ""
        result = fetch_annotation(source=src, desc=desc or None)
        return {"paths": result, "source": src, "desc": desc}
    except Exception:
        return None


@dataclass
class PipelineResult:
    """Result of run_pipeline."""

    cognitive_map: np.ndarray | None = None
    receptor_info: dict[str, Any] | None = None
    concepts_used: list[str] = field(default_factory=list)
    combine_method: str = ""
    sources: list[str] = field(default_factory=list)  # "cache" | "ontology" | "similarity" per concept (for decompose_avg)


def run_pipeline(
    query: str,
    mapper: OptionAMapper,
    *,
    combine: str = "decompose_avg",
    use_ontology: bool = False,
    decomposer: Callable[[str], list[str]] | None = None,
    neuromaps_receptor: str | None = None,
) -> PipelineResult:
    """
    Full pipeline: query -> concepts -> map(s) -> combined cognitive map + optional receptor.

    - combine: "neuroquery_style" (weights over cache terms -> one map) or "decompose_avg"
      (concepts -> get_map each -> average).
    - use_ontology: for NeuroQuery-style only; when True, use ontology when OOV/low similarity.
    - decomposer: function query -> list of concept strings; default simple_rule_decomposer.
    - neuromaps_receptor: e.g. "dopamine"; if set, try to fetch that receptor map from neuromaps.

    Returns PipelineResult with cognitive_map, receptor_info (if requested), concepts_used, combine_method.
    """
    concepts = decompose_query(query, decomposer)
    cognitive_map = None
    sources: list[str] = []

    if combine == "neuroquery_style":
        term_maps = mapper._decoder_maps
        term_vocab = mapper._term_vocab
        ontology_index = getattr(mapper, "_index", None)
        encoder = getattr(mapper, "_encoder", None)
        cache_embeddings = getattr(mapper, "_cache_embeddings", None)
        if use_ontology and ontology_index is not None:
            cognitive_map = get_map_neuroquery_style_with_ontology(
                query,
                term_maps,
                term_vocab,
                ontology_index,
                encoder=encoder,
                cache_embeddings=cache_embeddings,
                use_tfidf_fallback=True,
            )
        else:
            cognitive_map = get_map_neuroquery_style(
                query,
                term_maps,
                term_vocab,
                encoder=encoder,
                cache_embeddings=cache_embeddings,
                use_tfidf_fallback=True,
            )
        # Per-concept source not tracked in neuroquery_style path
    else:
        # decompose_avg
        cognitive_map = get_map_decompose_avg(query, mapper, decomposer=decomposer)
        # Optionally track source per concept (cache vs ontology vs similarity)
        # For now we don't have get_map_for_term return source; leave sources empty or infer from concepts
        sources = []  # could extend OptionAMapper.get_map to return source

    receptor_info = None
    if neuromaps_receptor:
        receptor_info = _fetch_receptor_info(neuromaps_receptor)

    return PipelineResult(
        cognitive_map=np.asarray(cognitive_map).ravel() if cognitive_map is not None else None,
        receptor_info=receptor_info,
        concepts_used=concepts,
        combine_method=combine,
        sources=sources,
    )


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Full NeuroLab pipeline: query -> concepts -> maps -> combined map (+ optional receptor)."
    )
    parser.add_argument("query", type=str, nargs="?", default="working memory and attention")
    parser.add_argument(
        "--decoder-cache-dir",
        type=Path,
        default=None,
        help="Decoder cache dir (default: data/unified_cache if present)",
    )
    parser.add_argument("--ontology-dir", type=Path, default=None)
    parser.add_argument(
        "--combine",
        choices=("neuroquery_style", "decompose_avg"),
        default="decompose_avg",
        help="How to combine: neuroquery_style (weights over cache) or decompose_avg (map per concept -> average)",
    )
    parser.add_argument(
        "--use-ontology",
        action="store_true",
        help="For neuroquery_style: use ontology when OOV/low similarity",
    )
    parser.add_argument("--receptor", type=str, default=None, help="Fetch receptor map from neuromaps (e.g. dopamine)")
    args = parser.parse_args()

    ontology_dir = args.ontology_dir or _default_ontology_dir()
    if not ontology_dir.exists():
        print(f"Ontology dir not found: {ontology_dir}", file=sys.stderr)
        print("Run: python scripts/download_ontologies.py", file=sys.stderr)
        return 1

    decoder_cache = args.decoder_cache_dir
    if decoder_cache is None:
        default_cache = _scripts.parent / "data" / "unified_cache"
        if (default_cache / "term_maps.npz").exists():
            decoder_cache = default_cache
    mapper = OptionAMapper(ontology_dir=ontology_dir, decoder_cache_dir=decoder_cache)
    result = run_pipeline(
        args.query,
        mapper,
        combine=args.combine,
        use_ontology=args.use_ontology,
        decomposer=simple_rule_decomposer,
        neuromaps_receptor=args.receptor,
    )

    print("Query:", args.query)
    print("Concepts:", result.concepts_used)
    print("Combine:", result.combine_method)
    if result.cognitive_map is not None:
        arr = np.asarray(result.cognitive_map)
        print("Cognitive map: shape", arr.shape, "mean", float(np.mean(arr)))
    else:
        print("Cognitive map: None")
    if result.receptor_info:
        print("Receptor info:", result.receptor_info)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
