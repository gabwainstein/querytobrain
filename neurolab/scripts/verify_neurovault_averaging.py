#!/usr/bin/env python3
"""
Verify NeuroVault averaging: compare raw image counts vs output term counts per collection.

For AVERAGE_FIRST collections, output should be << raw (substantial reduction from averaging).
Flags collections where output ≈ raw (averaging likely failed) or output is unexpectedly high/low.

Usage:
  python neurolab/scripts/verify_neurovault_averaging.py
  python neurolab/scripts/verify_neurovault_averaging.py --cache-dir neurolab/data/neurovault_cache --data-dir neurolab/data/neurovault_curated_data
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

# Expected output ranges for single-subject collections (min, max) after averaging
# None = no specific expectation; (a,b) = expect output in [a,b]
EXPECTED_OUTPUT = {
    503: (1, 1),        # PINES: 1 map (negative vs neutral)
    6618: (100, 250),   # IBC 2nd: ~205 contrasts
    2138: (50, 150),    # IBC 1st: ~59 contrasts (can vary with sub-splits)
    1952: (100, 250),   # BrainPedia: ~196 conditions
    4343: (20, 100),    # UCLA LA5C
    16284: (2, 10),     # IAPS valence: few conditions
    426: (5, 30),
    445: (5, 30),
    507: (5, 50),
    504: (5, 30),
    2503: (5, 30),
    4804: (5, 30),
    13042: (2, 50),     # Can be sparse
    13705: (5, 50),
    2108: (5, 30),
    4683: (5, 30),
    3887: (5, 30),
    1516: (5, 30),
    13474: (5, 30),
    20510: (5, 30),
    11646: (5, 30),
    437: (5, 30),
    12992: (5, 30),
    19012: (5, 30),
    6825: (5, 30),
    1620: (5, 30),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify NeuroVault cache averaging")
    parser.add_argument("--cache-dir", default="neurolab/data/neurovault_cache")
    parser.add_argument("--data-dir", default="neurolab/data/neurovault_curated_data")
    args = parser.parse_args()

    cache_dir = REPO_ROOT / args.cache_dir
    data_dir = REPO_ROOT / args.data_dir

    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    # Load AVERAGE_FIRST from ingestion
    from neurolab.neurovault_ingestion import AVERAGE_FIRST

    # Raw image counts from manifest or downloads
    manifest_path = data_dir / "manifest.json"
    downloads_dir = data_dir / "downloads" / "neurovault"

    raw_by_col: dict[int, int] = defaultdict(int)
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for img in data.get("images") or []:
            cid = img.get("collection_id")
            if cid is not None:
                raw_by_col[cid] += 1
        print(f"Loaded manifest: {sum(raw_by_col.values())} images, {len(raw_by_col)} collections")
    elif downloads_dir.exists():
        for cdir in downloads_dir.iterdir():
            if not cdir.is_dir() or not cdir.name.startswith("collection_"):
                continue
            try:
                cid = int(cdir.name.replace("collection_", ""))
            except ValueError:
                continue
            raw_by_col[cid] = len(list(cdir.glob("image_*.*")))
        print(f"Scanned downloads: {sum(raw_by_col.values())} images, {len(raw_by_col)} collections")
    else:
        print("No manifest or downloads found; cannot compute raw counts.")
        raw_by_col = {}

    # Output counts from cache
    if not (cache_dir / "term_collection_ids.pkl").exists():
        print(f"Cache not found: {cache_dir / 'term_collection_ids.pkl'}")
        print("Run build_neurovault_cache.py --average-subject-level first.")
        return 1

    cids = pickle.load(open(cache_dir / "term_collection_ids.pkl", "rb"))
    terms = pickle.load(open(cache_dir / "term_vocab.pkl", "rb"))
    out_by_col = dict(Counter(cids))

    print(f"Cache: {len(terms)} terms, {len(out_by_col)} collections")
    print()

    # Report
    print("=" * 85)
    print(f"{'Collection':<10} {'Raw':>8} {'Output':>8} {'Ratio':>8} {'Status':<40}")
    print("-" * 85)

    issues = []
    all_cols = sorted(set(raw_by_col) | set(out_by_col))

    for cid in all_cols:
        raw = raw_by_col.get(cid, 0)
        out = out_by_col.get(cid, 0)
        ratio = out / raw if raw > 0 else 0
        should_avg = cid in AVERAGE_FIRST

        if raw == 0 and out > 0:
            status = "OK (output only; raw not in manifest?)"
        elif raw > 0 and out == 0:
            status = "WARN: no output (all skipped or failed?)" if should_avg else "OK (excluded?)"
            if should_avg:
                issues.append((cid, f"AVERAGE_FIRST but 0 output from {raw} raw"))
        elif should_avg and raw > 0:
            # Averaging should reduce substantially
            if ratio > 0.5:
                status = f"FAIL: ratio {ratio:.2f} - averaging likely NOT applied"
                issues.append((cid, f"Output {out} ≈ raw {raw} (ratio {ratio:.2f}); expected << raw"))
            else:
                exp = EXPECTED_OUTPUT.get(cid)
                if exp:
                    lo, hi = exp
                    if out < lo or out > hi:
                        status = f"CHECK: expected {lo}-{hi}, got {out}"
                        issues.append((cid, f"Expected {lo}-{hi} terms, got {out}"))
                    else:
                        status = "OK"
                else:
                    status = "OK" if ratio < 0.3 else f"CHECK: ratio {ratio:.2f}"
        else:
            status = "OK (use-as-is)" if ratio <= 1.01 else "CHECK"

        print(f"{cid:<10} {raw:>8} {out:>8} {ratio:>7.2%} {status:<40}")

    print("=" * 85)

    if issues:
        print()
        print("ISSUES TO INVESTIGATE:")
        for cid, msg in issues:
            print(f"  col{cid}: {msg}")
        print()
        print("To fix: rebuild neurovault_cache with --average-subject-level:")
        print("  python neurolab/scripts/build_neurovault_cache.py --data-dir neurolab/data/neurovault_curated_data --output-dir neurolab/data/neurovault_cache --average-subject-level")
    else:
        print()
        print("No issues detected.")

    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
