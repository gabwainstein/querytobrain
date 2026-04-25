#!/usr/bin/env python3
"""
Parameter sweep: optimize similarity_threshold (A) and similarity_threshold_ontology (B).

For a set of test terms, gets the map from cache+ontology (get_map_with_ontology_on_low_similarity)
for each (A, B) and compares to a reference map (NeuroQuery). Reports mean Pearson r and best (A, B).

Run from querytobrain root:
  python neurolab/scripts/parameter_sweep_thresholds.py
  python neurolab/scripts/parameter_sweep_thresholds.py --terms "attention" "memory" --grid 0.1 0.15 0.2
  python neurolab/scripts/parameter_sweep_thresholds.py --output results_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

DATA_DIR = repo_root / "neurolab" / "data"
N_PARCELS = 400

# Default test terms: mix of high-similarity (cache) and possible ontology (low similarity)
DEFAULT_TERMS = [
    "attention",
    "memory",
    "working memory",
    "pain",
    "emotion",
    "language",
    "vision",
    "executive function",
]


def _cognitive_cache_dir():
    unified = DATA_DIR / "unified_cache"
    decoder = DATA_DIR / "decoder_cache"
    if (unified / "term_maps.npz").exists():
        return str(unified)
    return str(decoder)


def _load_cache(cache_dir: str):
    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        return None, None, None
    data = np.load(npz_path)
    key = "term_maps" if "term_maps" in data.files else data.files[0]
    term_maps = np.asarray(data[key], dtype=np.float64)
    with open(pkl_path, "rb") as f:
        term_vocab = pickle.load(f)
    if isinstance(term_vocab, dict):
        term_vocab = list(term_vocab.keys()) if term_vocab else []
    term_vocab = list(term_vocab)
    if term_maps.shape[0] != len(term_vocab):
        return None, None, None
    return term_maps, term_vocab, cache_dir


def _load_ontology_index(ontology_dir: str | None):
    if not ontology_dir or not os.path.isdir(ontology_dir):
        return None
    scripts_dir = repo_root / "neurolab" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from ontology_expansion import load_ontology_index
        idx = load_ontology_index(ontology_dir)
        if idx.get("label_to_related"):
            return idx
    except Exception:
        pass
    return None


def get_reference_map_neuroquery(term: str) -> np.ndarray | None:
    """NeuroQuery term -> 400-D parcellated map (reference)."""
    scripts_dir = repo_root / "neurolab" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from query import get_parcellated_map_for_term
        return get_parcellated_map_for_term(term)
    except Exception:
        return None


def correlation(a: np.ndarray, b: np.ndarray) -> float:
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


def run_sweep(
    cache_dir: str,
    ontology_dir: str | None,
    terms: list[str],
    grid_a: list[float],
    grid_b: list[float] | None,
    use_ontology: bool,
) -> tuple[dict[tuple[float, float], float], list[tuple[float, float, float]]]:
    """
    For each (A, B) in grid, compute mean Pearson r between cache+ontology map and NeuroQuery map.
    Returns (results_dict, list of (A, B, mean_r)).
    """
    term_maps, term_vocab, _ = _load_cache(cache_dir)
    if term_maps is None:
        return {}, []

    scripts_dir = repo_root / "neurolab" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from query_to_map import get_map_with_ontology_on_low_similarity

    ontology_index = _load_ontology_index(ontology_dir) if use_ontology else None
    if grid_b is None:
        grid_b = grid_a  # B = A by default

    # Precompute reference maps (NeuroQuery) once per term
    ref_maps = {}
    for term in terms:
        ref = get_reference_map_neuroquery(term)
        if ref is not None:
            ref_maps[term] = ref
    if not ref_maps:
        print("No reference maps (NeuroQuery) could be computed.", file=sys.stderr)
        return {}, []

    results: dict[tuple[float, float], float] = {}
    rows: list[tuple[float, float, float]] = []

    for A in grid_a:
        for B in grid_b:
            rs = []
            for term, ref in ref_maps.items():
                map_ab, _ = get_map_with_ontology_on_low_similarity(
                    term,
                    term_maps,
                    term_vocab,
                    ontology_index=ontology_index,
                    similarity_threshold=A,
                    similarity_threshold_ontology=B,
                    encoder=None,
                    cache_embeddings=None,
                    use_tfidf_fallback=True,
                )
                if map_ab is not None:
                    r = correlation(map_ab, ref)
                    if not np.isnan(r):
                        rs.append(r)
            mean_r = float(np.mean(rs)) if rs else float("nan")
            results[(A, B)] = mean_r
            rows.append((A, B, mean_r))

    return results, rows


def main():
    parser = argparse.ArgumentParser(description="Sweep similarity thresholds A and B")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cognitive cache dir (default: unified_cache or decoder_cache)",
    )
    parser.add_argument(
        "--ontology-dir",
        default=None,
        help="Ontology dir for low-similarity path (default: neurolab/data/ontologies)",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=DEFAULT_TERMS,
        help="Test terms for sweep",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help="Grid values for A (and B if --grid-b not set)",
    )
    parser.add_argument(
        "--grid-b",
        nargs="+",
        type=float,
        default=None,
        help="Grid values for B (default: same as --grid)",
    )
    parser.add_argument(
        "--no-ontology",
        action="store_true",
        help="Do not use ontology in sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to CSV (columns: threshold_a, threshold_b, mean_r)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or _cognitive_cache_dir()
    ontology_dir = args.ontology_dir or str(DATA_DIR / "ontologies")
    if not os.path.isdir(ontology_dir):
        ontology_dir = None
    use_ontology = not args.no_ontology and ontology_dir is not None

    if not os.path.exists(os.path.join(cache_dir, "term_maps.npz")):
        print("Cache missing. Build first: python neurolab/scripts/build_all_maps.py --quick", file=sys.stderr)
        sys.exit(1)

    print(f"Cache: {cache_dir}")
    print(f"Ontology: {ontology_dir or 'none'}")
    print(f"Terms: {args.terms}")
    print(f"Grid A: {args.grid}")
    print(f"Grid B: {args.grid_b or args.grid}")

    results, rows = run_sweep(
        cache_dir=cache_dir,
        ontology_dir=ontology_dir,
        terms=args.terms,
        grid_a=args.grid,
        grid_b=args.grid_b,
        use_ontology=use_ontology,
    )

    if not rows:
        print("No results.", file=sys.stderr)
        sys.exit(1)

    # Best (A, B) by mean r
    valid = [(a, b, r) for a, b, r in rows if not np.isnan(r)]
    if valid:
        best = max(valid, key=lambda x: x[2])
        print(f"\nBest: threshold_a={best[0]:.2f}, threshold_b={best[1]:.2f}, mean_r={best[2]:.4f}")
    else:
        print("\nNo valid mean_r (all NaN).")

    # Table
    print("\nResults (threshold_a, threshold_b, mean_r):")
    for a, b, r in sorted(rows, key=lambda x: (-x[2] if not np.isnan(x[2]) else -999)):
        print(f"  {a:.2f}, {b:.2f}, {r:.4f}" if not np.isnan(r) else f"  {a:.2f}, {b:.2f}, nan")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold_a", "threshold_b", "mean_r"])
            for a, b, r in rows:
                w.writerow([a, b, r if not np.isnan(r) else ""])
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
