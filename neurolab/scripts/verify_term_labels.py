#!/usr/bin/env python3
"""
Verify term labels in a cache: no broken or useless labels.

Checks:
  1. Poor/generic labels — neurovault_image_N, empty, too short
  2. Missing map type prefix — fMRI:, PET:, Structural:, Gene:, etc.
  3. Duplicate labels — normalized form collisions
  4. Suspicious patterns — R0, collection_X_map_Y, placeholder-like
  5. Map validity — all-zero or NaN (when term_maps available)

Run from repo root:
  python neurolab/scripts/verify_term_labels.py
  python neurolab/scripts/verify_term_labels.py --cache-dir neurolab/data/merged_sources
  python neurolab/scripts/verify_term_labels.py --strict  # exit 1 on any failure
  python neurolab/scripts/verify_term_labels.py --all-caches  # check multiple caches

Note: Intermediate caches (decoder, neurovault, abagen, etc.) may show "missing type prefix"
and "suspicious" — those get prefixed at merge. Focus on merged_sources for final quality.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _normalize(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ").replace("  ", " ") if t else ""


# Known map type prefixes (labels should have one so model learns modality)
_KNOWN_TYPE_PREFIXES = ("fMRI:", "PET:", "Structural:", "Gene:", "Cognitive:", "Perfusion:", "DTI:")

# Patterns that indicate poor/placeholder labels
_POOR_PATTERNS = [
    (re.compile(r"^neurovault_image_\d+$"), "neurovault_image_N placeholder"),
    (re.compile(r"^collection_\d+_map_\d+$"), "generic collection_X_map_Y"),
    (re.compile(r"^R\d+$"), "generic receptor fallback R0, R1, ..."),
    (re.compile(r"^\d+$"), "numeric-only label"),
]


def _is_poor_label(label: str, min_length: int = 3) -> tuple[bool, str | None]:
    """Return (is_poor, reason). min_length=3 allows 'acc', 'age' etc. in decoder vocab."""
    if not label or not isinstance(label, str):
        return True, "empty or non-string"
    s = label.strip()
    if not s:
        return True, "empty after strip"
    if len(s) < min_length:
        return True, f"too short ({len(s)} chars)"
    for pat, reason in _POOR_PATTERNS:
        if pat.search(_normalize(s).replace(" ", "")):
            return True, reason
    return False, None


def _has_type_prefix(label: str) -> bool:
    s = (label or "").strip()
    return any(s.startswith(p) for p in _KNOWN_TYPE_PREFIXES)


def _is_suspicious(label: str) -> tuple[bool, str | None]:
    """Labels that look placeholder-like but aren't in _POOR_PATTERNS."""
    s = (label or "").strip()
    if not s:
        return False, None
    # Very short non-prefixed labels that might be truncated
    if len(s) < 8 and not _has_type_prefix(s):
        return True, "very short and no type prefix"
    # Backslash suggests a path (forward slash is common in e.g. D2/D3)
    if "\\" in s:
        return True, "contains backslash (path?)"
    return False, None


def check_cache(cache_dir: Path, load_maps: bool = True, min_length: int = 3) -> tuple[bool, dict]:
    """
    Run all label checks on a cache. Returns (all_ok, report_dict).
    Supports term_vocab.pkl or annotation_labels.pkl (neuromaps).
    """
    import pickle

    report = {
        "cache_dir": str(cache_dir),
        "n_terms": 0,
        "errors": [],
        "warnings": [],
        "poor_labels": [],
        "missing_type_prefix": [],
        "duplicates": [],
        "suspicious": [],
        "all_zero": 0,
        "has_nan": False,
    }

    pkl = cache_dir / "term_vocab.pkl"
    if not pkl.exists():
        pkl = cache_dir / "annotation_labels.pkl"
    if not pkl.exists():
        report["errors"].append("term_vocab.pkl or annotation_labels.pkl not found")
        return False, report

    with open(pkl, "rb") as f:
        terms = pickle.load(f)
    terms = list(terms)
    report["n_terms"] = len(terms)

    # 1. Poor labels
    for i, t in enumerate(terms):
        ok, reason = _is_poor_label(t, min_length=min_length)
        if ok:
            report["poor_labels"].append((i, t, reason))

    # 2. Missing map type prefix
    for i, t in enumerate(terms):
        if _is_poor_label(t, min_length=min_length)[0]:
            continue
        if not _has_type_prefix(t):
            report["missing_type_prefix"].append((i, t))

    # 3. Duplicates (by normalized form)
    norm_to_indices = {}
    for i, t in enumerate(terms):
        n = _normalize(t)
        if n:
            norm_to_indices.setdefault(n, []).append(i)
    for n, indices in norm_to_indices.items():
        if len(indices) > 1:
            report["duplicates"].append((n, [terms[i] for i in indices]))

    # 4. Suspicious
    for i, t in enumerate(terms):
        if _is_poor_label(t, min_length=min_length)[0]:
            continue
        ok, reason = _is_suspicious(t)
        if ok:
            report["suspicious"].append((i, t, reason))

    # 5. Map validity
    if load_maps:
        npz = cache_dir / "term_maps.npz"
        key = "term_maps"
        if not npz.exists():
            npz = cache_dir / "annotation_maps.npz"
            key = "matrix"
        if npz.exists():
            import numpy as np
            data = np.load(npz)
            arr = np.asarray(data[key] if key in data.files else data[data.files[0]])
            if arr.ndim == 2 and arr.shape[0] == len(terms):
                report["all_zero"] = int((np.abs(arr).sum(axis=1) == 0).sum())
                report["has_nan"] = bool(np.isnan(arr).any())

    # Summary
    if report["poor_labels"]:
        report["errors"].append(f"{len(report['poor_labels'])} poor/placeholder labels")
    if report["duplicates"]:
        report["errors"].append(f"{len(report['duplicates'])} duplicate label groups (by normalized form)")
    if report["all_zero"] > 0:
        report["errors"].append(f"{report['all_zero']} all-zero maps")
    if report["has_nan"]:
        report["errors"].append("NaN values in maps")

    if report["missing_type_prefix"]:
        report["warnings"].append(f"{len(report['missing_type_prefix'])} labels missing map type prefix")
    if report["suspicious"]:
        report["warnings"].append(f"{len(report['suspicious'])} suspicious labels")

    all_ok = len(report["errors"]) == 0
    return all_ok, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify term labels: no broken or useless labels in cache"
    )
    ap.add_argument(
        "--cache-dir",
        default="neurolab/data/merged_sources",
        help="Cache directory with term_vocab.pkl (and term_maps.npz)",
    )
    ap.add_argument(
        "--all-caches",
        action="store_true",
        help="Check merged_sources, neurovault_cache, neurovault_pharma_cache, neuromaps_cache, abagen_cache, enigma_cache",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any errors (poor labels, duplicates, zero maps, NaN)",
    )
    ap.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip map validity check (all-zero, NaN)",
    )
    ap.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum label length for 'too short' check (default 3; 4 flags 'acc', 'age')",
    )
    args = ap.parse_args()

    data_dir = _repo_root / "neurolab" / "data"
    caches_to_check = []

    if args.all_caches:
        for name in [
            "merged_sources",
            "unified_cache",
            "neurovault_cache",
            "neurovault_pharma_cache",
            "neuromaps_cache",
            "abagen_cache",
            "enigma_cache",
            "pharma_neurosynth_cache",
            "decoder_cache",
            "receptor_reference_cache",
        ]:
            d = data_dir / name
            if (d / "term_vocab.pkl").exists() or (d / "annotation_labels.pkl").exists():
                caches_to_check.append((name, d))
        # neuromaps uses annotation_labels.pkl
        if (data_dir / "neuromaps_cache" / "annotation_labels.pkl").exists():
            p = data_dir / "neuromaps_cache"
            if ("neuromaps_cache", p) not in caches_to_check:
                caches_to_check.append(("neuromaps_cache", p))
    else:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = _repo_root / args.cache_dir
        caches_to_check = [(cache_dir.name, cache_dir)]

    all_ok = True
    for name, cache_dir in caches_to_check:
        if not (cache_dir / "term_vocab.pkl").exists() and not (cache_dir / "annotation_labels.pkl").exists():
            print(f"[SKIP] {name}: no term_vocab.pkl or annotation_labels.pkl")
            continue

        ok, report = check_cache(
            cache_dir,
            load_maps=not args.no_maps,
            min_length=getattr(args, "min_length", 3),
        )
        if not ok:
            all_ok = False

        print("=" * 70)
        print(f"LABEL CHECK: {name}")
        print("=" * 70)
        print(f"Terms: {report['n_terms']}")

        if report["errors"]:
            print("\nERRORS:")
            for e in report["errors"]:
                print(f"  [X] {e}")
            if report["poor_labels"]:
                print("\n  Poor labels (first 10):")
                for i, t, r in report["poor_labels"][:10]:
                    print(f"    idx={i}: {repr(t)[:60]} ({r})")
            if report["duplicates"]:
                print("\n  Duplicate groups (first 5):")
                for norm, labels in report["duplicates"][:5]:
                    print(f"    norm={repr(norm)[:40]}: {labels[:3]}")

        if report["warnings"]:
            print("\nWARNINGS:")
            for w in report["warnings"]:
                print(f"  [!] {w}")
            if report["missing_type_prefix"] and len(report["missing_type_prefix"]) <= 15:
                print("\n  Missing type prefix:")
                for i, t in report["missing_type_prefix"][:15]:
                    print(f"    idx={i}: {repr(t)[:60]}")
            elif report["missing_type_prefix"]:
                print(f"\n  Missing type prefix (first 10 of {len(report['missing_type_prefix'])}):")
                for i, t in report["missing_type_prefix"][:10]:
                    print(f"    idx={i}: {repr(t)[:60]}")
            if report["suspicious"]:
                print("\n  Suspicious (first 5):")
                for i, t, r in report["suspicious"][:5]:
                    print(f"    idx={i}: {repr(t)[:50]} ({r})")

        if not report["errors"] and not report["warnings"]:
            print("\n  OK — no broken or useless labels detected.")
        print()

    if args.strict and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
