#!/usr/bin/env python3
"""
Rebuild NeuroVault cache (with P-map transform, min_subjects override, ROI fallback)
and merged_sources training set. Use after code fixes to neurovault_ingestion or build_neurovault_cache.

  python neurolab/scripts/rebuild_neurovault_and_merge.py
  python neurolab/scripts/rebuild_neurovault_and_merge.py --n-jobs 8 --no-fetch-metadata  # faster
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_scripts = _repo_root / "neurolab" / "scripts"
_data = _repo_root / "neurolab" / "data"


def run(script: str, args: list[str], desc: str) -> bool:
    path = _scripts / script
    if not path.exists():
        print(f"  Skip {desc}: {path.name} not found")
        return True
    cmd = [sys.executable, str(path)] + args
    print(f"\n{'='*60}\n  {desc}\n  {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(_repo_root))
    return r.returncode == 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild NeuroVault cache + merged_sources")
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel parcellation jobs (default 4)")
    ap.add_argument("--no-fetch-metadata", action="store_true", help="Skip NeuroVault API fetch for missing labels")
    args = ap.parse_args()

    ok = True
    nv_data = _data / "neurovault_curated_data"
    nv_cache = _data / "neurovault_cache"

    if not (nv_data / "manifest.json").exists():
        print(f"NeuroVault curated data not found: {nv_data / 'manifest.json'}")
        print("Run: python neurolab/scripts/download_neurovault_curated.py --all")
        return 1

    # 1. NeuroVault cache
    nv_args = [
        "--data-dir", str(nv_data),
        "--output-dir", str(nv_cache),
        "--average-subject-level",
        "--n-jobs", str(args.n_jobs),
    ]
    if args.no_fetch_metadata:
        nv_args += ["--no-fetch-missing-metadata"]
    ok &= run("build_neurovault_cache.py", nv_args, "NeuroVault cache (P-map, min_subjects, ROI fixes)")

    # 2. Improve labels
    if ok and (nv_cache / "term_maps.npz").exists():
        imp_args = ["--cache-dir", str(nv_cache)]
        if args.no_fetch_metadata:
            imp_args += ["--no-fetch-metadata"]
        ok &= run("improve_neurovault_labels.py", imp_args, "NeuroVault label improvement")

    # 3. Merge
    if ok and (_data / "unified_cache" / "term_maps.npz").exists():
        merge_args = [
            "--cache-dir", str(_data / "unified_cache"),
            "--output-dir", str(_data / "merged_sources"),
            "--neurovault-cache-dir", str(nv_cache),
            "--no-ontology", "--save-term-sources",
        ]
        if (_data / "neuromaps_cache" / "annotation_maps.npz").exists():
            merge_args += ["--neuromaps-cache-dir", str(_data / "neuromaps_cache")]
        if (_data / "neurovault_pharma_cache" / "term_maps.npz").exists():
            merge_args += ["--neurovault-pharma-cache-dir", str(_data / "neurovault_pharma_cache")]
        if (_data / "pharma_neurosynth_cache" / "term_maps.npz").exists():
            merge_args += ["--pharma-neurosynth-cache-dir", str(_data / "pharma_neurosynth_cache")]
        if (_data / "enigma_cache" / "term_maps.npz").exists():
            merge_args += ["--enigma-cache-dir", str(_data / "enigma_cache")]
        if (_data / "abagen_cache" / "term_maps.npz").exists():
            merge_args += ["--abagen-cache-dir", str(_data / "abagen_cache"), "--max-abagen-terms", "500",
                          "--abagen-add-gradient-pcs", "3", "--add-pet-residuals"]
        ok &= run("build_expanded_term_maps.py", merge_args, "Merged sources (training set)")

    print("\n" + "="*60)
    print("  Rebuild complete." if ok else "  Rebuild failed.")
    print("="*60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
