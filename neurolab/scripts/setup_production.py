#!/usr/bin/env python3
"""
One-shot production setup: download ontologies, build NeuroQuery cache, build NeuroSynth cache,
merge into unified decoder cache. Pipeline and compare scripts use this cache by default.

Usage:
  python setup_production.py [--data-dir data] [--quick]
  --data-dir: base dir for data/, ontologies/, caches (default: neurolab/data)
  --quick: build with --max-terms 200 (NQ) and 100 (NS) for a fast test; omit for full build.

Full build: ~7547 NeuroQuery terms + ~1300 NeuroSynth terms, then merge. Takes time (hours).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_repo = _scripts.parent


def run(cmd: list[str], cwd: Path | None = None) -> int:
    r = subprocess.run(cmd, cwd=cwd or _scripts)
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Production setup: ontologies + NQ + NS caches + merge.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Base data dir (default: repo/data)")
    parser.add_argument("--quick", action="store_true", help="Limit terms for fast test (NQ 200, NS 100)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or _repo / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    ontologies_dir = data_dir / "ontologies"
    nq_cache = data_dir / "neuroquery_cache"
    ns_cache = data_dir / "neurosynth_cache"
    unified = data_dir / "unified_cache"

    print("Step 1/4: Download ontologies")
    dl_cmd = [sys.executable, str(_scripts / "download_ontologies.py"), "--output-dir", str(ontologies_dir)]
    if run(dl_cmd, cwd=_repo) != 0:
        return 1
    if not (ontologies_dir / "mf.owl").exists() and not (ontologies_dir / "uberon.owl").exists():
        print("Ontologies not found; ensure download_ontologies.py writes to data/ontologies.", file=sys.stderr)
        return 1

    print("Step 2/4: Build NeuroQuery cache")
    nq_cmd = [sys.executable, str(_scripts / "build_neuroquery_cache.py"), "--output-dir", str(nq_cache)]
    if args.quick:
        nq_cmd += ["--max-terms", "200"]
    if run(nq_cmd, cwd=_repo) != 0:
        return 1
    if not (nq_cache / "term_maps.npz").exists():
        print("NeuroQuery cache not produced.", file=sys.stderr)
        return 1

    print("Step 3/4: Build NeuroSynth cache")
    ns_cmd = [
        sys.executable,
        str(_scripts / "build_neurosynth_cache.py"),
        "--output-dir", str(ns_cache),
        "--data-dir", str(data_dir / "neurosynth_data"),
    ]
    if args.quick:
        ns_cmd += ["--max-terms", "100"]
    if run(ns_cmd, cwd=_repo) != 0:
        return 1
    if not (ns_cache / "term_maps.npz").exists():
        print("NeuroSynth cache not produced.", file=sys.stderr)
        return 1

    print("Step 4/4: Merge into unified cache")
    if run([
        sys.executable,
        str(_scripts / "merge_neuroquery_neurosynth_cache.py"),
        "--neuroquery-cache-dir", str(nq_cache),
        "--neurosynth-cache-dir", str(ns_cache),
        "--output-dir", str(unified),
        "--prefer", "neurosynth",
    ], cwd=_repo) != 0:
        return 1

    print("Production setup complete.")
    print(f"  Ontologies:    {ontologies_dir}")
    print(f"  Unified cache: {unified}")
    print("Run pipeline with (from querytobrain repo root):")
    print(f"  python neurolab/scripts/pipeline.py \"your query\" --decoder-cache-dir {unified}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
