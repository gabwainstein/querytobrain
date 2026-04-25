#!/usr/bin/env python3
"""
Phase 4+5 verification: ReceptorEnrichment and UnifiedEnrichment.

- ReceptorEnrichment: uses placeholder data if no CSV/NPZ provided.
- UnifiedEnrichment: runs cognitive (if cache) + biological; checks summary.

Run from querytobrain root: python neurolab/scripts/verify_unified.py
"""
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np

DATA_DIR = os.path.join(repo_root, "neurolab", "data")
UNIFIED_CACHE = os.path.join(DATA_DIR, "unified_cache")
DECODER_CACHE = os.path.join(DATA_DIR, "decoder_cache")
NEUROMAPS_CACHE = os.path.join(DATA_DIR, "neuromaps_cache")


def _cognitive_cache_dir():
    if os.path.exists(os.path.join(UNIFIED_CACHE, "term_maps.npz")):
        return UNIFIED_CACHE
    return DECODER_CACHE


def _neuromaps_cache_dir():
    if os.path.exists(os.path.join(NEUROMAPS_CACHE, "annotation_maps.npz")):
        return NEUROMAPS_CACHE
    return None


def main():
    cache_dir = _cognitive_cache_dir()
    neuromaps_dir = _neuromaps_cache_dir()
    print("Phase 4: ReceptorEnrichment")
    from neurolab.enrichment.receptor_enrichment import ReceptorEnrichment

    rec = ReceptorEnrichment(receptor_matrix_path=None, n_parcels=400)
    print(f"  Loaded {len(rec.receptor_names)} receptors (placeholder)")

    rng = np.random.default_rng(99)
    test_map = rng.standard_normal(400)
    out = rec.enrich(test_map, method="pearson")
    assert "by_layer" in out and "receptors" in out["by_layer"]
    assert "top_hits" in out and len(out["top_hits"]) > 0
    print(f"  top_hits sample: {out['top_hits'][0]}")

    print("\nPhase 5: UnifiedEnrichment")
    from neurolab.enrichment.unified_enrichment import UnifiedEnrichment

    unified = UnifiedEnrichment(
        cache_dir=cache_dir,
        receptor_path=None,
        neuromaps_cache_dir=neuromaps_dir,
        enable_cognitive=os.path.exists(os.path.join(cache_dir, "term_maps.npz")),
        enable_biological=True,
        n_parcels=400,
    )
    print(f"  Cognitive cache: {cache_dir}")
    if neuromaps_dir:
        print(f"  Neuromaps cache: {neuromaps_dir}")
    result = unified.enrich(test_map, cognitive_top_n=10)
    assert "summary" in result
    assert "biological" in result
    if unified.cognitive:
        assert "cognitive" in result
    print("  summary:")
    for line in result["summary"].split("\n"):
        print(f"    {line}")
    print("\nPhase 4+5 passed. ReceptorEnrichment and UnifiedEnrichment are ready.")


if __name__ == "__main__":
    main()
