#!/usr/bin/env python3
"""
Check whether ontology distance/relationship correlates with brain-map similarity.

For term pairs that are related in the ontology and both have maps in the cache,
we compute the Pearson correlation of their 400-D brain maps. If ontology proximity
maps to brain similarity, we'd expect: synonym pairs > parent/child > unrelated.
Results inform how we set RELATION_WEIGHTS in ontology_expansion.py.

Usage (from repo root):
  python neurolab/scripts/ontology_brain_correlation.py --cache-dir neurolab/data/decoder_cache --ontology-dir neurolab/data/ontologies
  python neurolab/scripts/ontology_brain_correlation.py --cache-dir neurolab/data/unified_cache_expanded --ontology-dir neurolab/data/ontologies
"""
import argparse
import json
import os
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

N_PARCELS = 400


def main():
    parser = argparse.ArgumentParser(description="Ontology relationship vs brain map correlation")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="Cache (term_maps.npz, term_vocab.pkl)")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies", help="Ontology directory")
    parser.add_argument("--max-pairs", type=int, default=5000, help="Max (label, related) pairs to sample (default 5000)")
    parser.add_argument("--output-weights", default=None, help="Write relation_type -> mean_r (and random_mean) to JSON for use as --relation-weights-file in build_expanded_term_maps")
    args = parser.parse_args()

    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    ontology_dir = args.ontology_dir if os.path.isabs(args.ontology_dir) else os.path.join(repo_root, args.ontology_dir)

    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print("Cache not found. Build first: build_term_maps_cache.py or build_expanded_term_maps.py", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    term_maps = np.asarray(data["term_maps"])
    with open(pkl_path, "rb") as f:
        import pickle
        term_vocab = pickle.load(f)
    term_vocab = list(term_vocab)
    if term_maps.shape[0] != len(term_vocab) or term_maps.shape[1] != N_PARCELS:
        print("Cache shape mismatch.", file=sys.stderr)
        sys.exit(1)

    from ontology_expansion import load_ontology_index, _normalize_term

    if not os.path.isdir(ontology_dir):
        print("Ontology dir not found.", file=sys.stderr)
        sys.exit(1)
    index = load_ontology_index(ontology_dir)
    label_to_related = index.get("label_to_related") or {}
    if not label_to_related:
        print("No ontology labels.", file=sys.stderr)
        sys.exit(1)

    cache_norm_to_idx = {}
    for i, t in enumerate(term_vocab):
        k = _normalize_term(t)
        if k:
            cache_norm_to_idx[k] = i

    # Bucket relation type from weight (matches RELATION_WEIGHTS in ontology_expansion.py)
    def _weight_to_relation_type(w: float) -> str:
        if 0.99 <= w <= 1.01:
            return "self"
        if 0.93 <= w <= 0.97:
            return "synonym"
        if 0.83 <= w <= 0.87:
            return "child"
        if 0.78 <= w <= 0.82:
            return "parent"
        return "other"

    # Map to directional buckets: parent_of = label is parent of related (related is child); child_of = label is child of related (related is parent)
    def _to_directional_bucket(rtype: str) -> str:
        if rtype == "child":
            return "parent_of"
        if rtype == "parent":
            return "child_of"
        return rtype

    # Collect (label_norm, related_norm) by directional relation type
    # parent_of = label is parent of related (expansion: use child's map for parent → subset, plausible)
    # child_of = label is child of related (expansion: use parent's map for child → too broad, penalize)
    pairs_by_type = {}  # relation_type -> list of (a, b)
    for label_norm, related_list in label_to_related.items():
        if label_norm not in cache_norm_to_idx:
            continue
        for item in related_list:
            rel_name = item[0]
            w = item[1]
            rtype = item[2] if len(item) >= 3 else _weight_to_relation_type(w)
            bucket = _to_directional_bucket(rtype)
            rel_norm = _normalize_term(rel_name)
            if not rel_norm or rel_norm == label_norm:
                continue
            if rel_norm not in cache_norm_to_idx:
                continue
            pairs_by_type.setdefault(bucket, []).append((label_norm, rel_norm))
            if sum(len(v) for v in pairs_by_type.values()) >= args.max_pairs:
                break
        if sum(len(v) for v in pairs_by_type.values()) >= args.max_pairs:
            break

    if not pairs_by_type or sum(len(v) for v in pairs_by_type.values()) == 0:
        print("No ontology-related pairs found in cache. Try --cache-dir with expanded cache.")
        sys.exit(0)

    # Compute correlations per relation type (directional: parent_of vs child_of)
    type_order = ["self", "synonym", "parent_of", "child_of", "other"]
    corrs_by_type = {}
    for rtype in type_order:
        pairs = pairs_by_type.get(rtype, [])
        if not pairs:
            continue
        corrs = []
        for a, b in pairs:
            i, j = cache_norm_to_idx[a], cache_norm_to_idx[b]
            r = np.corrcoef(term_maps[i], term_maps[j])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
        if corrs:
            corrs_by_type[rtype] = corrs

    if not corrs_by_type:
        print("No finite correlations.")
        sys.exit(0)

    # Random pairs for comparison
    rng = np.random.default_rng(42)
    n_random = min(5000, sum(len(c) for c in corrs_by_type.values()))
    random_corrs = []
    for _ in range(n_random):
        i, j = rng.integers(0, len(term_vocab), 2)
        if i == j:
            continue
        r = np.corrcoef(term_maps[i], term_maps[j])[0, 1]
        if np.isfinite(r):
            random_corrs.append(r)
    random_mean = np.mean(random_corrs) if random_corrs else 0.0

    print("Ontology distance vs brain map correlation (per relation type)")
    print("=" * 60)
    for rtype in type_order:
        corrs = corrs_by_type.get(rtype, [])
        if not corrs:
            print(f"  {rtype:10s}  n=    0  (no pairs in cache)")
            continue
        m, s = np.mean(corrs), np.std(corrs)
        diff = m - random_mean
        print(f"  {rtype:10s}  n={len(corrs):5d}  mean r={m:.4f}  std={s:.4f}  (vs random: {diff:+.4f})")
    if random_corrs:
        print(f"  {'random':10s}  n={len(random_corrs):5d}  mean r={random_mean:.4f}")
    print("\nDirectional: parent_of = label is parent of related (child's map used for parent); child_of = label is child of related (parent's map used for child).")
    print("Use these per-type means to tune RELATION_WEIGHTS; if child_of mean r < parent_of, parent's map is too broad for child -> keep DIRECTION_SCALE['parent'] low.")
    print("Run with unified_cache_expanded for more term coverage.")

    if args.output_weights:
        out = {rtype: float(np.mean(corrs)) for rtype, corrs in corrs_by_type.items()}
        out["_random_mean"] = float(random_mean)
        # Expansion expects "parent" and "child"; map directional stats so expansion can use them
        if "child_of" in out:
            out["parent"] = out["child_of"]
        if "parent_of" in out:
            out["child"] = out["parent_of"]
        out_path = args.output_weights if os.path.isabs(args.output_weights) else os.path.join(repo_root, args.output_weights)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote mean r per relation type -> {out_path}")
        print("Use in expansion: build_expanded_term_maps.py --relation-weights-file ...")


if __name__ == "__main__":
    main()
