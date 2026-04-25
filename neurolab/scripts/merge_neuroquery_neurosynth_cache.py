#!/usr/bin/env python3
"""
Merge NeuroQuery and NeuroSynth (term, map) caches into one unified cache.

Vocabulary = union of NeuroQuery and NeuroSynth terms.
Map per term: for overlap, prefer one source (--prefer neurosynth | neuroquery); else use the only source.

Input: two cache dirs, each with term_maps.npz and term_vocab.pkl (same parcellation, Glasser+Tian 392).
Output: unified term_maps.npz and term_vocab.pkl in --output-dir.

Usage:
  python merge_neuroquery_neurosynth_cache.py \\
    --neuroquery-cache-dir path/to/neuroquery_cache \\
    --neurosynth-cache-dir path/to/neurosynth_cache \\
    --output-dir path/to/unified_cache \\
    [--prefer neurosynth]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _normalize(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ")


def load_cache(cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / "term_maps.npz")
    key = "term_maps" if "term_maps" in data else data.files[0]
    maps = np.asarray(data[key])
    with open(cache_dir / "term_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab = list(vocab.keys() if isinstance(vocab, dict) else vocab)
    if len(vocab) != maps.shape[0]:
        raise ValueError(f"Cache shape mismatch in {cache_dir}: {len(vocab)} terms vs {maps.shape[0]} rows")
    return maps, vocab


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge NeuroQuery + NeuroSynth caches into one.")
    parser.add_argument("--neuroquery-cache-dir", type=Path, required=True, help="NeuroQuery cache dir")
    parser.add_argument("--neurosynth-cache-dir", type=Path, required=True, help="NeuroSynth cache dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output unified cache dir")
    parser.add_argument(
        "--prefer",
        choices=("neurosynth", "neuroquery"),
        default="neuroquery",
        help="For overlapping terms, which source to use (default: neuroquery; smoother NQ maps often better for MSE training)",
    )
    args = parser.parse_args()

    nq_dir = Path(args.neuroquery_cache_dir)
    ns_dir = Path(args.neurosynth_cache_dir)
    out_dir = Path(args.output_dir)
    if not nq_dir.exists():
        print(f"NeuroQuery cache not found: {nq_dir}", file=sys.stderr)
        return 1
    if not ns_dir.exists():
        print(f"NeuroSynth cache not found: {ns_dir}", file=sys.stderr)
        return 1

    nq_maps, nq_vocab = load_cache(nq_dir)
    ns_maps, ns_vocab = load_cache(ns_dir)
    if nq_maps.shape[1] != ns_maps.shape[1]:
        print("Warning: parcel count mismatch; output will use NeuroQuery parcel count.", file=sys.stderr)

    n_parcels = nq_maps.shape[1]
    nq_norm = {_normalize(t): (i, t) for i, t in enumerate(nq_vocab)}
    ns_norm = {_normalize(t): (i, t) for i, t in enumerate(ns_vocab)}
    all_terms_norm = sorted(set(nq_norm) | set(ns_norm))
    unified_vocab = []
    unified_maps = []

    for norm in all_terms_norm:
        in_nq = norm in nq_norm
        in_ns = norm in ns_norm
        if in_ns and in_nq:
            if args.prefer == "neurosynth":
                i_ns, orig = ns_norm[norm]
                unified_maps.append(ns_maps[i_ns])
                unified_vocab.append(orig)
            else:
                i_nq, orig = nq_norm[norm]
                unified_maps.append(nq_maps[i_nq])
                unified_vocab.append(orig)
        elif in_ns:
            i_ns, orig = ns_norm[norm]
            unified_maps.append(ns_maps[i_ns])
            unified_vocab.append(orig)
        else:
            i_nq, orig = nq_norm[norm]
            unified_maps.append(nq_maps[i_nq])
            unified_vocab.append(orig)

    out_dir.mkdir(parents=True, exist_ok=True)
    maps_out = np.stack(unified_maps).astype(np.float32)
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=maps_out)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(unified_vocab, f)
    print(f"Unified cache: {len(unified_vocab)} terms, shape {maps_out.shape}, written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
