#!/usr/bin/env python3
"""
Pre-training sanity checks for merged_sources and training pipeline.

1. Abagen tiered selection — Verify 505 genes: Tier1 receptor, Tier2 medoids, Tier3 residual-variance
2. Train/val/test split balance — Per-source counts; flag rare sources with 0-5 test
3. Decoder cache size — Note if decoder_cache < unified_cache potential
4. NaN/zero check — No all-zero or NaN maps in merged_sources
5. Term collision audit — Neuromaps PET labels dropped due to collision with direct/neurovault

Run from repo root:
  python neurolab/scripts/pre_training_checks.py
  python neurolab/scripts/pre_training_checks.py --cache-dir neurolab/data/merged_sources
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("  ", " ") if s else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-training sanity checks")
    ap.add_argument("--cache-dir", default="neurolab/data/merged_sources", help="Merged sources dir")
    ap.add_argument("--seed", type=int, default=42, help="Match trainer --seed for split simulation")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / args.cache_dir

    npz = cache_dir / "term_maps.npz"
    pkl = cache_dir / "term_vocab.pkl"
    src_pkl = cache_dir / "term_sources.pkl"

    if not npz.exists() or not pkl.exists():
        print(f"Cache not found: {cache_dir}", file=sys.stderr)
        return 1

    data = np.load(npz)
    maps = np.asarray(data["term_maps"])
    with open(pkl, "rb") as f:
        terms = pickle.load(f)
    term_sources = None
    if src_pkl.exists():
        with open(src_pkl, "rb") as f:
            term_sources = pickle.load(f)

    n = len(terms)
    n_parcels = maps.shape[1]

    print("=" * 70)
    print("PRE-TRAINING CHECKS")
    print("=" * 70)
    print(f"\nCache: {n} terms × {n_parcels} parcels")

    # --- 1. Abagen tiered selection ---
    print("\n" + "-" * 70)
    print("1. ABAGEN TIERED SELECTION")
    print("-" * 70)
    receptor_genes = set()
    try:
        from neurolab.receptor_kb import load_receptor_genes
        receptor_genes = set(g.upper() for g in load_receptor_genes())
        print(f"Receptor gene list: {len(receptor_genes)} genes")
    except Exception as e:
        print(f"  Could not load receptor genes: {e}")

    abagen_terms = []
    if term_sources:
        abagen_idx = [i for i, s in enumerate(term_sources) if s == "abagen"]
        abagen_terms = [terms[i] for i in abagen_idx]
        # Exclude gradient PCs (gene_expression_gradient_PC1..5)
        gene_terms = [t for t in abagen_terms if "gradient_PC" not in t]
        gradient_terms = [t for t in abagen_terms if "gradient_PC" in t]
        n_abagen_genes = len(gene_terms)
        n_gradient = len(gradient_terms)

        # Infer Tier1: "gene expression (SYMBOL)" matches receptor list
        tier1_count = 0
        tier2_medoid_hints = 0  # cluster medoids often have no obvious pattern
        for t in gene_terms:
            if "(" in t and ")" in t:
                symbol = t.split("(")[-1].rstrip(")").strip().upper()
                if symbol in receptor_genes:
                    tier1_count += 1

        print(f"Abagen terms: {n_abagen_genes} genes + {n_gradient} gradient PCs = {n_abagen_genes + n_gradient}")
        print(f"  Tier 1 (receptor overlap): {tier1_count} of {len(receptor_genes)} receptor genes in merged")
        print(f"  Expected: ~250 Tier1, ~32 Tier2, ~200 Tier3 (build_expanded reports Tier1/Tier2/Tier3)")

    # --- 2. Train/val/test split balance ---
    print("\n" + "-" * 70)
    print("2. TRAIN/VAL/TEST SPLIT BALANCE (simulated, stratified when term_sources)")
    print("-" * 70)
    rng = np.random.default_rng(args.seed)
    if term_sources:
        from collections import defaultdict
        import math
        idx_by_source = defaultdict(list)
        for i in range(n):
            idx_by_source[term_sources[i]].append(i)
        test_idx_list, val_idx_list, train_idx_list = [], [], []
        for src, indices in sorted(idx_by_source.items()):
            n_s = len(indices)
            perm_s = rng.permutation(n_s)
            idx_arr = np.array(indices)[perm_s]
            if n_s < 50:
                n_test_s = max(1, math.ceil(n_s * 0.2))
                n_val_s = max(0, min(n_s - n_test_s, math.ceil(n_s * args.val_frac)))
            else:
                n_test_s = max(0, int(n_s * args.test_frac))
                n_val_s = max(0, min(n_s - n_test_s, int(n_s * args.val_frac)))
            test_idx_list.extend(idx_arr[:n_test_s].tolist())
            val_idx_list.extend(idx_arr[n_test_s : n_test_s + n_val_s].tolist())
            train_idx_list.extend(idx_arr[n_test_s + n_val_s :].tolist())
        test_idx = np.array(test_idx_list, dtype=int)
        val_idx = np.array(val_idx_list, dtype=int)
        train_idx = np.array(train_idx_list, dtype=int)
        print("Stratified: sources <50 terms get >=20% in test")
    else:
        perm = rng.permutation(n)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if term_sources:
        test_sources = [term_sources[i] for i in test_idx]
        val_sources = [term_sources[i] for i in val_idx]
        train_sources = [term_sources[i] for i in train_idx]
        c_test = Counter(test_sources)
        c_val = Counter(val_sources)
        c_train = Counter(train_sources)
        print("\nPer-source test counts:")
        for src in sorted(set(term_sources)):
            n_test_s = c_test.get(src, 0)
            n_val_s = c_val.get(src, 0)
            n_train_s = c_train.get(src, 0)
            total = n_test_s + n_val_s + n_train_s
            flag = " [LOW]" if n_test_s < 5 and total > 0 else ""
            print(f"  {src:22s}: train={n_train_s:5d}  val={n_val_s:4d}  test={n_test_s:4d}{flag}")
        low_test = [s for s in set(term_sources) if c_test.get(s, 0) < 5 and (c_test.get(s, 0) + c_val.get(s, 0) + c_train.get(s, 0)) > 0]
        if low_test:
            print(f"\n  [!] Rare sources with <5 test: {low_test}")
            print("  Consider --stratify-by-source if trainer supports it, or accept noisy per-source test metrics.")

    # --- 3. Decoder cache size ---
    print("\n" + "-" * 70)
    print("3. DECODER CACHE SIZE")
    print("-" * 70)
    dec_dir = repo_root / "neurolab" / "data" / "decoder_cache"
    unif_dir = repo_root / "neurolab" / "data" / "unified_cache"
    for name, p in [("decoder_cache", dec_dir), ("unified_cache", unif_dir)]:
        vp = p / "term_vocab.pkl"
        if vp.exists():
            with open(vp, "rb") as f:
                v = pickle.load(f)
            print(f"  {name}: {len(v)} terms")
        else:
            print(f"  {name}: not found")

    # --- 4. NaN/zero check ---
    print("\n" + "-" * 70)
    print("4. NaN / ZERO CHECK")
    print("-" * 70)
    all_zero_mask = (np.abs(maps).sum(axis=1) == 0)
    all_zero = int(all_zero_mask.sum())
    has_nan = np.isnan(maps).any()
    print(f"  All-zero rows: {all_zero}")
    print(f"  Any NaN: {has_nan}")
    if all_zero > 0:
        zero_idx = np.where(all_zero_mask)[0]
        for i in zero_idx[:5]:
            src = term_sources[i] if term_sources else "?"
            print(f"    All-zero: '{terms[i]}' (source={src}, idx={i})")
        if all_zero > 5:
            print(f"    ... and {all_zero - 5} more")
    if all_zero > 0 or has_nan:
        print("  [!] Fix before training.")

    # --- 5. Term collision audit (neuromaps) ---
    print("\n" + "-" * 70)
    print("5. TERM COLLISION AUDIT (neuromaps vs merged)")
    print("-" * 70)
    nm_dir = repo_root / "neurolab" / "data" / "neuromaps_cache"
    if (nm_dir / "annotation_labels.pkl").exists():
        with open(nm_dir / "annotation_labels.pkl", "rb") as f:
            nm_labels = pickle.load(f)
        merged_norm_to_idx = {}
        for i, t in enumerate(terms):
            n = _norm(t)
            if n and n not in merged_norm_to_idx:
                merged_norm_to_idx[n] = i
        nm_kept = 0
        nm_collision = []  # (nm_label, merged_term, merged_source) — neuromaps dropped, direct/neurovault won
        nm_missing = []    # norm not in merged at all (shouldn't happen if merge is correct)
        for lb in nm_labels:
            nlb = _norm(lb)
            if not nlb:
                continue
            if nlb in merged_norm_to_idx:
                idx = merged_norm_to_idx[nlb]
                src = term_sources[idx] if term_sources else None
                if src == "neuromaps" or src == "neuromaps_residual":
                    nm_kept += 1
                else:
                    nm_collision.append((lb, terms[idx], src or "?"))
            else:
                nm_missing.append(lb)
        n_nm = len(nm_labels)
        print(f"  Neuromaps cache labels: {n_nm}")
        print(f"  Kept (source=neuromaps): {nm_kept}")
        print(f"  Collision (dropped neuromaps; direct/neurovault won): {len(nm_collision)}")
        print(f"  Missing (norm not in merged): {len(nm_missing)}")
        if nm_collision:
            print(f"  Collision examples: {nm_collision[:5]}")
        if nm_missing:
            print(f"  Missing examples: {nm_missing[:5]}")
    else:
        print("  neuromaps_cache/annotation_labels.pkl not found")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
