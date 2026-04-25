#!/usr/bin/env python3
"""
Compare two ways to get a brain map from a full query:

1. NeuroQuery-style (default, no ontology): query -> weights over cache terms -> map
2. NeuroQuery-style + optional ontology: when terms are OOV or similarity to cache is low,
   resolve via ontologically related terms that are in the cache, then map from those (not default)
3. Decompose -> average: query -> concepts -> one map per concept -> average maps

For each test query, compute all maps and report correlations.

Usage:
  python compare_query_methods.py --decoder-cache-dir path/to/cache [query1 query2 ...]
  Default test queries: "working memory", "visual attention", "see my dog after ritalin"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from option_a import OptionAMapper, _default_ontology_dir
from query_to_map import (
    get_map_neuroquery_style,
    get_map_neuroquery_style_with_ontology,
    get_map_decompose_avg,
    correlation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare NeuroQuery-style vs decompose-average query->map.")
    parser.add_argument(
        "queries",
        nargs="*",
        default=["working memory", "visual attention", "see my dog after ritalin"],
        help="Test queries (default: working memory, visual attention, see my dog after ritalin)",
    )
    parser.add_argument(
        "--decoder-cache-dir",
        type=Path,
        default=None,
        help="Decoder cache (default: data/unified_cache if present)",
    )
    parser.add_argument("--ontology-dir", type=Path, default=None)
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
    term_maps = mapper._decoder_maps
    term_vocab = mapper._term_vocab
    ontology_index = mapper._index
    encoder = getattr(mapper, "_encoder", None)
    cache_embeddings = getattr(mapper, "_cache_embeddings", None)

    print("Method A:       NeuroQuery-style (default, no ontology)")
    print("Method A+onto:  NeuroQuery-style + optional ontology (OOV/low sim -> ontology -> cache terms)")
    print("Method B:       Decompose -> average (concepts -> get_map each -> average)")
    print()

    for query in args.queries:
        map_a = get_map_neuroquery_style(
            query,
            term_maps,
            term_vocab,
            encoder=encoder,
            cache_embeddings=cache_embeddings,
            use_tfidf_fallback=True,
        )
        map_a_onto = get_map_neuroquery_style_with_ontology(
            query,
            term_maps,
            term_vocab,
            ontology_index,
            encoder=encoder,
            cache_embeddings=cache_embeddings,
            use_tfidf_fallback=True,
        )
        map_dec = get_map_decompose_avg(query, mapper, decomposer=None)

        r_aa = float("nan")
        r_ab = float("nan")
        r_a_onto_b = float("nan")
        if map_a is not None and map_a_onto is not None:
            r_aa = correlation(map_a, map_a_onto)
        if map_a is not None and map_dec is not None:
            r_ab = correlation(map_a, map_dec)
        if map_a_onto is not None and map_dec is not None:
            r_a_onto_b = correlation(map_a_onto, map_dec)

        print(f"Query: {query!r}")
        print(f"  Method A (no ontology):   {'OK' if map_a is not None else 'None'}")
        print(f"  Method A+onto:            {'OK' if map_a_onto is not None else 'None'}")
        print(f"  Method B (decompose->avg): {'OK' if map_dec is not None else 'None'}")
        fmt = "  Correlation({0}, {1}): {2:.4f}"
        if r_aa == r_aa:
            print(fmt.format("A", "A+onto", r_aa))
        if r_ab == r_ab:
            print(fmt.format("A", "B", r_ab))
        if r_a_onto_b == r_a_onto_b:
            print(fmt.format("A+onto", "B", r_a_onto_b))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
