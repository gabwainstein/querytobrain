#!/usr/bin/env python3
"""
Build all maps for the enrichment system: NeuroQuery + NeuroSynth (cognitive)
and neuromaps (biological; includes Hansen receptor atlas). Run from querytobrain repo root.

By default builds ALL available maps (full NeuroQuery vocab, all NeuroSynth terms,
all neuromaps MNI152 annotations). This can take many hours; use --quick for a test.

  python neurolab/scripts/build_all_maps.py
  python neurolab/scripts/build_all_maps.py --quick
  python neurolab/scripts/build_all_maps.py --skip-neurosynth --skip-neuromaps
  python neurolab/scripts/build_all_maps.py --quick --skip-neuromaps --expand   # + ontology-expanded cache (NQ+NS, all ontologies)

With --expand: builds unified_cache_expanded (NQ+NS+ontology+neuromaps+receptor if available)
for text-to-brain training. Neuromaps is built before expand when not --skip-neuromaps so the
full set includes biological labels. Optional --receptor-path adds receptor atlas maps to the set.
"""
import argparse
import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_script(script_name: str, args: list, description: str) -> bool:
    script_path = os.path.join(repo_root, "neurolab", "scripts", script_name)
    if not os.path.exists(script_path):
        print(f"  Skip {description}: {script_path} not found", file=sys.stderr)
        return True
    cmd = [sys.executable, script_path] + args
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print("="*60)
    r = subprocess.run(cmd, cwd=repo_root)
    if r.returncode != 0:
        print(f"  FAILED: {script_name} exited {r.returncode}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build all enrichment maps (NeuroQuery, NeuroSynth, neuromaps)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small caps for a fast test (NeuroQuery 500, NeuroSynth 50, neuromaps 30)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(repo_root, "neurolab", "data"),
        help="Base data directory (default: neurolab/data)",
    )
    parser.add_argument("--skip-neuroquery", action="store_true", help="Skip NeuroQuery cache")
    parser.add_argument("--skip-neurosynth", action="store_true", help="Skip NeuroSynth cache")
    parser.add_argument("--skip-neuromaps", action="store_true", help="Skip neuromaps cache")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Build full (label, map) set: NQ+NS+ontology+neuromaps+receptor → unified_cache_expanded",
    )
    parser.add_argument("--receptor-path", default=None, help="Receptor atlas path (e.g. Hansen CSV) to merge into --expand output")
    args = parser.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(repo_root, args.data_dir)
    decoder_cache = os.path.join(data_dir, "decoder_cache")
    neurosynth_cache = os.path.join(data_dir, "neurosynth_cache")
    unified_cache = os.path.join(data_dir, "unified_cache")
    neuromaps_cache = os.path.join(data_dir, "neuromaps_cache")
    os.makedirs(decoder_cache, exist_ok=True)
    os.makedirs(neurosynth_cache, exist_ok=True)
    os.makedirs(unified_cache, exist_ok=True)
    os.makedirs(neuromaps_cache, exist_ok=True)

    if args.quick:
        nq_max = 500
        ns_max = 50
        nm_max = 30
        print("Quick mode: building limited maps for testing.")
    else:
        nq_max = 0   # 0 = all terms (no cap)
        ns_max = 0   # 0 = all terms
        nm_max = 0   # 0 = all annotations
        print("Full mode: building ALL available maps (can take many hours).")

    ok = True

    if not args.skip_neuroquery:
        ok &= run_script(
            "build_term_maps_cache.py",
            ["--cache-dir", decoder_cache, "--max-terms", str(nq_max)],
            "NeuroQuery cognitive term maps (decoder_cache)",
        )

    if not args.skip_neurosynth:
        ok &= run_script(
            "build_neurosynth_cache.py",
            ["--cache-dir", neurosynth_cache, "--max-terms", str(ns_max)],
            "NeuroSynth cognitive term maps via NiMARE (neurosynth_cache)",
        )

    # Merge NQ + NS into one cognitive cache so the whole system is connected (one decoder cache).
    if not args.skip_neuroquery and not args.skip_neurosynth and ok:
        ok &= run_script(
            "merge_neuroquery_neurosynth_cache.py",
            [
                "--neuroquery-cache-dir", decoder_cache,
                "--neurosynth-cache-dir", neurosynth_cache,
                "--output-dir", unified_cache,
                "--prefer", "neuroquery",
            ],
            "Merge NeuroQuery + NeuroSynth → unified_cache (connected cognitive cache)",
        )

    # Build neuromaps before expand so the full set can include biological labels.
    # Uses neurolab/data/neuromaps_data by default (run download_neuromaps_data.py if missing).
    if not args.skip_neuromaps:
        neuromaps_data = os.path.join(data_dir, "neuromaps_data")
        nm_args = ["--cache-dir", neuromaps_cache, "--data-dir", neuromaps_data]
        if nm_max > 0:
            nm_args.extend(["--max-annot", str(nm_max)])
        ok &= run_script(
            "build_neuromaps_cache.py",
            nm_args,
            "Neuromaps biological annotations (neuromaps_cache)",
        )

    # Full (label, map) set: cognitive + ontology + neuromaps + receptor → unified_cache_expanded.
    if args.expand and ok and not args.skip_neuroquery and not args.skip_neurosynth:
        ontology_dir = os.path.join(data_dir, "ontologies")
        unified_expanded = os.path.join(data_dir, "unified_cache_expanded")
        expand_args = [
            "--cache-dir", unified_cache,
            "--ontology-dir", ontology_dir,
            "--output-dir", unified_expanded,
            "--min-cache-matches", "2",
            "--min-pairwise-correlation", "0.3",
        ]
        if os.path.exists(os.path.join(neuromaps_cache, "annotation_maps.npz")):
            expand_args += ["--neuromaps-cache-dir", neuromaps_cache]
        if args.receptor_path and os.path.exists(args.receptor_path if os.path.isabs(args.receptor_path) else os.path.join(repo_root, args.receptor_path)):
            expand_args += ["--receptor-path", args.receptor_path]
        expand_args += ["--save-term-sources"]
        ok &= run_script(
            "build_expanded_term_maps.py",
            expand_args,
            "Full set: NQ+NS+ontology+neuromaps+receptor → unified_cache_expanded",
        )

    if not ok:
        sys.exit(1)
    print("\nAll requested caches built successfully. Connected system:")
    if not args.skip_neuroquery:
        print(f"  Cognitive (NQ):     {decoder_cache}")
    if not args.skip_neurosynth:
        print(f"  Cognitive (NS):    {neurosynth_cache}")
    if not args.skip_neuroquery and not args.skip_neurosynth:
        print(f"  Cognitive (merged): {unified_cache}  ← use this as default decoder cache")
    if not args.skip_neuromaps:
        print(f"  Biological:        {neuromaps_cache}")
    if args.expand and not args.skip_neuroquery and not args.skip_neurosynth:
        print(f"  Full set (train):  {os.path.join(data_dir, 'unified_cache_expanded')}  ← for text-to-brain training")
    print("  Query/verify scripts default to unified_cache + neuromaps_cache when present.")


if __name__ == "__main__":
    main()
