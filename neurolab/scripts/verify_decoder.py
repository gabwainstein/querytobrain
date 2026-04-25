#!/usr/bin/env python3
"""
Phase 3 verification: load CognitiveDecoder from cache and run decode().

If cache is missing, builds a minimal cache (10 terms) then runs tests.
Run from querytobrain root: python neurolab/scripts/verify_decoder.py
"""
import os
import sys

# Add repo root so neurolab.enrichment is importable
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np

DATA_DIR = os.path.join(repo_root, "neurolab", "data")
UNIFIED_CACHE = os.path.join(DATA_DIR, "unified_cache")
DECODER_CACHE = os.path.join(DATA_DIR, "decoder_cache")


def _cognitive_cache_dir():
    """Use unified_cache (NQ+NS merged) if present, else decoder_cache."""
    if os.path.exists(os.path.join(UNIFIED_CACHE, "term_maps.npz")):
        return UNIFIED_CACHE
    return DECODER_CACHE


def ensure_minimal_cache():
    if os.path.exists(os.path.join(UNIFIED_CACHE, "term_maps.npz")) or os.path.exists(os.path.join(DECODER_CACHE, "term_maps.npz")):
        return
    print("Cache missing. Building minimal cache (10 terms) into decoder_cache...")
    import subprocess
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "neurolab", "scripts", "build_term_maps_cache.py"),
            "--cache-dir", DECODER_CACHE,
            "--max-terms", "10",
        ],
        cwd=repo_root,
        timeout=300,
    )
    if r.returncode != 0:
        print("Build failed.", file=sys.stderr)
        sys.exit(1)


def main():
    ensure_minimal_cache()
    CACHE_DIR = _cognitive_cache_dir()
    from neurolab.enrichment.cognitive_decoder import CognitiveDecoder

    print("Phase 3: Verifying CognitiveDecoder")
    print(f"  Cache: {CACHE_DIR}")
    decoder = CognitiveDecoder(cache_dir=CACHE_DIR)
    print(f"  Loaded {len(decoder.vocabulary)} terms x {decoder.n_parcels} parcels")

    # Test 1: synthetic vector
    print("\n  Test 1: decode(random vector)...")
    rng = np.random.default_rng(42)
    random_map = rng.standard_normal(decoder.n_parcels)
    out = decoder.decode(random_map, top_n=5)
    assert "top_terms" in out and len(out["top_terms"]) == 5
    assert out["n_terms_evaluated"] == len(decoder.vocabulary)
    print(f"    top_terms: {[t[0] for t in out['top_terms']]}")

    # Test 2: wrong shape
    print("  Test 2: decode(wrong shape) -> ValueError...")
    try:
        decoder.decode(np.zeros(200))
    except ValueError as e:
        assert "400" in str(e) or "200" in str(e)
        print("    OK: ValueError raised")
    else:
        print("    FAIL: expected ValueError", file=sys.stderr)
        sys.exit(1)

    print("\nPhase 3 passed. CognitiveDecoder is ready.")


if __name__ == "__main__":
    main()
