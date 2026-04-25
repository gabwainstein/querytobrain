#!/usr/bin/env python3
"""
Final maps × terms validation before training: READY / NOT READY verdict.

Runs verify_term_labels and adds:
  - Maps × terms alignment (shape match, NaN, all-zero counts)
  - Per-source summary (from term_sources.pkl)
  - Map stats (mean, std, parcel coverage)
  - Clear READY / NOT READY verdict

Run after build_mercedes_training_set.py or build_expanded_term_maps.py merge:
  python neurolab/scripts/final_training_review.py
  python neurolab/scripts/final_training_review.py --cache-dir neurolab/data/merged_sources
  python neurolab/scripts/final_training_review.py --strict  # exit 1 if NOT READY
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import check_cache from sibling script
_scripts = _repo_root / "neurolab" / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from verify_term_labels import check_cache


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Final maps × terms validation: READY / NOT READY verdict"
    )
    ap.add_argument(
        "--cache-dir",
        default="neurolab/data/merged_sources",
        help="Merged training cache directory",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if NOT READY",
    )
    ap.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip map validity checks (all-zero, NaN)",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _repo_root / args.cache_dir

    print("=" * 70)
    print("  FINAL TRAINING REVIEW: maps × terms validation")
    print(f"  Cache: {cache_dir}")
    print("=" * 70)

    issues = []

    # 1. Label verification
    all_ok, report = check_cache(cache_dir, load_maps=not args.no_maps, min_length=3)
    print("\n1. LABEL VERIFICATION")
    print(f"   Terms: {report['n_terms']}")
    for e in report.get("errors", []):
        print(f"   ERROR: {e}")
        issues.append(e)
    for w in report.get("warnings", []):
        print(f"   WARN:  {w}")
    if report.get("poor_labels"):
        for i, t, r in report["poor_labels"][:5]:
            print(f"   Poor: [{i}] {repr(t)[:60]} ({r})")
        if len(report["poor_labels"]) > 5:
            print(f"   ... and {len(report['poor_labels']) - 5} more")
    if all_ok and not report.get("errors"):
        print("   OK")

    # 2. Maps × terms alignment
    print("\n2. MAPS × TERMS ALIGNMENT")
    npz = cache_dir / "term_maps.npz"
    pkl = cache_dir / "term_vocab.pkl"
    if not pkl.exists():
        pkl = cache_dir / "annotation_labels.pkl"
    if not npz.exists():
        print("   ERROR: term_maps.npz not found")
        issues.append("term_maps.npz missing")
    elif not pkl.exists():
        print("   ERROR: term_vocab.pkl not found")
        issues.append("term_vocab.pkl missing")
    else:
        with open(pkl, "rb") as f:
            terms = pickle.load(f)
        terms = list(terms)
        data = np.load(npz)
        key = "term_maps" if "term_maps" in data.files else data.files[0]
        maps = np.asarray(data[key])
        n_terms, n_parcels = len(terms), maps.shape[1]
        if maps.shape[0] != n_terms:
            print(f"   ERROR: shape mismatch: maps {maps.shape[0]} vs terms {n_terms}")
            issues.append("maps/terms count mismatch")
        else:
            print(f"   Shape: {maps.shape[0]} terms × {maps.shape[1]} parcels")
        if np.isnan(maps).any():
            n_nan = int(np.isnan(maps).sum())
            print(f"   ERROR: {n_nan} NaN values in maps")
            issues.append("NaN in maps")
        else:
            print("   NaN: none")
        n_zero = int((np.abs(maps).sum(axis=1) == 0).sum())
        if n_zero > 0:
            print(f"   ERROR: {n_zero} all-zero maps")
            issues.append(f"{n_zero} all-zero maps")
        else:
            print("   All-zero maps: none")
        # Parcel coverage
        finite = np.isfinite(maps)
        covered = (finite & (maps != 0)).sum(axis=0)
        n_parcels_covered = (covered > 0).sum()
        print(f"   Parcels with data: {n_parcels_covered} / {n_parcels}")

    # 3. Per-source summary
    print("\n3. PER-SOURCE SUMMARY")
    src_pkl = cache_dir / "term_sources.pkl"
    if src_pkl.exists():
        with open(src_pkl, "rb") as f:
            sources = pickle.load(f)
        from collections import Counter
        counts = Counter(sources)
        for src, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"   {src}: {n}")
    else:
        print("   term_sources.pkl not found (optional)")

    # 4. Required files
    print("\n4. REQUIRED FILES")
    required = ["term_maps.npz", "term_vocab.pkl"]
    optional = ["term_sources.pkl", "term_sample_weights.pkl", "term_map_types.pkl"]
    for name in required:
        p = cache_dir / name
        if p.exists():
            print(f"   [OK] {name}")
        else:
            print(f"   [MISSING] {name}")
            issues.append(f"Missing {name}")
    for name in optional:
        p = cache_dir / name
        print(f"   {'[OK]' if p.exists() else '[--]'} {name}")

    # Verdict
    ready = len(issues) == 0
    print("\n" + "=" * 70)
    if ready:
        print("  READY TO TRAIN")
        print("  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/merged_sources ...")
    else:
        print("  NOT READY")
        for i in issues:
            print(f"  - {i}")
    print("=" * 70)

    if args.strict and not ready:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
