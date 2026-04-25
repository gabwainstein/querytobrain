#!/usr/bin/env python3
"""
Report how many maps each cache contains (terms/annotations x parcels).
Run after rebuild to see actual counts. Uses neurolab/data by default.

  python neurolab/scripts/report_map_counts.py
  python neurolab/scripts/report_map_counts.py --data-dir /path/to/neurolab/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy required")
    exit(1)

_repo_root = Path(__file__).resolve().parent.parent.parent

CACHES = [
    ("decoder_cache", "term_maps", "NeuroQuery"),
    ("neurosynth_cache", "term_maps", "NeuroSynth"),
    ("unified_cache", "term_maps", "NQ+NS merged"),
    ("merged_sources", "term_maps", "Merged (NQ+NS+neuromaps+neurovault+enigma+abagen)"),
    ("decoder_cache_expanded", "term_maps", "Expanded (ontology + all) [optional]"),
    ("neurovault_cache", "term_maps", "NeuroVault"),
    ("enigma_cache", "term_maps", "ENIGMA"),
    ("abagen_cache", "term_maps", "abagen"),
]
NEUROMAPS = ("neuromaps_cache", "annotation_maps.npz", "matrix", "Neuromaps")


def main() -> int:
    ap = argparse.ArgumentParser(description="Report map counts per cache")
    ap.add_argument("--data-dir", type=Path, default=None, help="Override neurolab/data")
    args = ap.parse_args()
    data_dir = args.data_dir or _repo_root / "neurolab" / "data"

    print(f"Data dir: {data_dir}")
    print()
    for rel_path, npz_key, label in CACHES:
        npz = data_dir / rel_path / "term_maps.npz"
        if not npz.exists():
            print(f"  {label:30} (not found)")
            continue
        try:
            data = np.load(npz)
            arr = np.asarray(data["term_maps"])
            n_maps, n_parcels = arr.shape[0], arr.shape[1]
            print(f"  {label:30} {n_maps:>6} maps  x  {n_parcels} parcels")
        except Exception as e:
            print(f"  {label:30} error: {e}")

    nm_path = data_dir / NEUROMAPS[0] / NEUROMAPS[1]
    if nm_path.exists():
        try:
            data = np.load(nm_path)
            arr = np.asarray(data[NEUROMAPS[2]])
            n_maps = arr.shape[0] if arr.ndim >= 2 else 1
            n_parcels = arr.shape[1] if arr.ndim >= 2 else arr.size
            print(f"  {NEUROMAPS[3]:30} {n_maps:>6} maps  x  {n_parcels} parcels")
        except Exception as e:
            print(f"  {NEUROMAPS[3]:30} error: {e}")
    else:
        print(f"  {NEUROMAPS[3]:30} (not found)")

    print()
    print("  For training use merged_sources (NQ+NS+neuromaps+neurovault+enigma+abagen, no ontology).")
    return 0


if __name__ == "__main__":
    exit(main())
