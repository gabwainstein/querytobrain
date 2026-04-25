#!/usr/bin/env python3
"""
Build ENIGMA DTI white matter tract cache from Kochunov et al. (2022) cross-disorder data.
7 disorders × FA (and optionally MD, RD, AD) across 25 JHU tract ROIs.
Requires: CSV with extracted Cohen's d from Kochunov 2022 Table 2 (supplementary).

Source: Kochunov et al. 2022, Human Brain Mapping, DOI 10.1002/hbm.24998
Disorders: schizophrenia, bipolar, MDD, OCD, PTSD, TBI, 22q

Usage:
  python neurolab/scripts/build_enigma_dti_cache.py --csv path/to/kochunov2022_table2.csv
  python neurolab/scripts/build_enigma_dti_cache.py --list-only  # print acquisition steps

Output: enigma_dti_cache/ with term_maps.npz (N, n_parcels) — JHU 25 ROIs mapped to
pipeline parcels via a simple spatial mapping (or replicate if no mapping provided).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

KOCHUNOV_2022_DOI = "10.1002/hbm.24998"
ACQUISITION_URL = "https://onlinelibrary.wiley.com/doi/10.1002/hbm.24998"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ENIGMA DTI cache from Kochunov 2022")
    parser.add_argument("--csv", type=Path, default=None, help="Path to extracted Table 2 CSV")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list-only", action="store_true", help="Print acquisition steps")
    args = parser.parse_args()

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "enigma_dti_cache"

    if args.list_only:
        print("ENIGMA DTI — Kochunov et al. 2022 cross-disorder")
        print("=" * 60)
        print(f"Paper: {ACQUISITION_URL}")
        print("Table 2: Cohen's d for 24 JHU tract ROIs × 7 disorders (FA, MD, RD, AD)")
        print()
        print("Manual steps:")
        print("  1. Download supplementary from Wiley")
        print("  2. Extract Table 2 to CSV: columns = disorder, tract_roi, metric, d_value")
        print("  3. Run: python build_enigma_dti_cache.py --csv path/to/table2.csv")
        print()
        print("Expected CSV format (header): disorder,tract_roi,metric,d_value")
        return 0

    csv_path = Path(args.csv) if args.csv else root / "data" / "enigma_dti" / "kochunov2022_table2.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run with --list-only for acquisition steps.", file=sys.stderr)
        return 1

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("pandas and numpy required", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    required = ["disorder", "tract_roi", "metric", "d_value"]
    if not all(c in df.columns for c in required):
        print(f"CSV must have columns: {required}", file=sys.stderr)
        return 1

    from neurolab.parcellation import get_n_parcels
    n_parcels = get_n_parcels()

    # JHU has 25 ROIs; we replicate to n_parcels (no spatial mapping for now)
    n_jhu = 25
    terms = []
    maps_list = []

    for (disorder, metric), grp in df.groupby(["disorder", "metric"]):
        d_vals = grp.set_index("tract_roi")["d_value"].reindex(range(n_jhu)).fillna(0).values
        if len(d_vals) < n_jhu:
            d_vals = np.pad(d_vals, (0, n_jhu - len(d_vals)), constant_values=0)
        # Replicate to n_parcels (placeholder; proper mapping would use JHU→Glasser overlap)
        map_392 = np.zeros(n_parcels, dtype=np.float64)
        for i in range(n_parcels):
            map_392[i] = d_vals[i % n_jhu]
        terms.append(f"{disorder} DTI {metric}")
        maps_list.append(map_392)

    if not terms:
        print("No rows in CSV.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    term_maps = np.stack(maps_list)
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    import pickle
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"source": "Kochunov 2022", "doi": KOCHUNOV_2022_DOI, "n_maps": len(terms)}, f, indent=2)

    print(f"ENIGMA DTI cache: {len(terms)} maps -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
