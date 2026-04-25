#!/usr/bin/env python3
"""
Run audit-recommended data rebuild steps in order.

Applies the Claude-audit fixes on data:
  1. Rebuild decoder_cache with --max-terms 0 (full vocab ~7.5K) if currently < 6K.
  2. Rebuild unified_cache (merge decoder + neurosynth).
  3. Rebuild merged_sources with neurovault_pharma and all other caches.

Run from repo root (directory containing neurolab/). Each step only runs if its
inputs exist; step 3 adds cache dirs only when present.

Usage:
  python neurolab/scripts/rebuild_training_data_steps.py --step 1   # decoder full vocab
  python neurolab/scripts/rebuild_training_data_steps.py --step 2   # NQ+NS -> unified
  python neurolab/scripts/rebuild_training_data_steps.py --step 3   # merged_sources (with neurovault_pharma)
  python neurolab/scripts/rebuild_training_data_steps.py             # all steps 1–3
  python neurolab/scripts/rebuild_training_data_steps.py --dry-run   # print commands only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_scripts = _repo_root / "neurolab" / "scripts"
_data = _repo_root / "neurolab" / "data"


def _run(cmd: list[str], desc: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"[dry-run] {desc}")
        print("  " + " ".join(cmd))
        return True
    print(f"\n>>> {desc}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_repo_root))
    return result.returncode == 0


def step1_decoder_full_vocab(dry_run: bool) -> bool:
    """Rebuild decoder_cache with --max-terms 0 (full NeuroQuery vocab ~7.5K)."""
    script = _scripts / "build_term_maps_cache.py"
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return False
    cmd = [
        sys.executable,
        str(script),
        "--cache-dir", str(_data / "decoder_cache"),
        "--max-terms", "0",
    ]
    return _run(cmd, "Step 1: Rebuild decoder_cache (full vocab)", dry_run)


def step2_unified_cache(dry_run: bool) -> bool:
    """Merge NeuroQuery + NeuroSynth -> unified_cache."""
    script = _scripts / "merge_neuroquery_neurosynth_cache.py"
    dec = _data / "decoder_cache" / "term_maps.npz"
    ns = _data / "neurosynth_cache" / "term_maps.npz"
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return False
    if not dec.exists():
        print("Step 2 skipped: decoder_cache not found. Run step 1 first.", file=sys.stderr)
        return True  # skip, not fail
    if not ns.exists():
        print("Step 2 skipped: neurosynth_cache not found. Build with build_neurosynth_cache.py first.", file=sys.stderr)
        return True
    cmd = [
        sys.executable,
        str(script),
        "--neuroquery-cache-dir", str(_data / "decoder_cache"),
        "--neurosynth-cache-dir", str(_data / "neurosynth_cache"),
        "--output-dir", str(_data / "unified_cache"),
    ]
    return _run(cmd, "Step 2: Merge NQ+NS -> unified_cache", dry_run)


def step3_merged_sources(dry_run: bool) -> bool:
    """Rebuild merged_sources with neurovault_pharma and all available caches."""
    script = _scripts / "build_expanded_term_maps.py"
    unified = _data / "unified_cache" / "term_maps.npz"
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return False
    if not unified.exists():
        print("Step 3 skipped: unified_cache not found. Run step 2 first.", file=sys.stderr)
        return True
    cmd = [
        sys.executable,
        str(script),
        "--cache-dir", str(_data / "unified_cache"),
        "--output-dir", str(_data / "merged_sources"),
        "--no-ontology",
        "--save-term-sources",
    ]
    if (_data / "neurovault_cache" / "term_maps.npz").exists():
        cmd += ["--neurovault-cache-dir", str(_data / "neurovault_cache")]
    if (_data / "neurovault_pharma_cache" / "term_maps.npz").exists():
        cmd += ["--neurovault-pharma-cache-dir", str(_data / "neurovault_pharma_cache")]
    if (_data / "neuromaps_cache" / "annotation_maps.npz").exists():
        cmd += ["--neuromaps-cache-dir", str(_data / "neuromaps_cache")]
    if (_data / "enigma_cache" / "term_maps.npz").exists():
        cmd += ["--enigma-cache-dir", str(_data / "enigma_cache")]
    if (_data / "abagen_cache" / "term_maps.npz").exists():
        cmd += [
            "--abagen-cache-dir", str(_data / "abagen_cache"),
            "--max-abagen-terms", "500",
            "--abagen-add-gradient-pcs", "3",
            "--add-pet-residuals",
        ]
    if (_data / "pharma_neurosynth_cache" / "term_maps.npz").exists():
        cmd += ["--pharma-neurosynth-cache-dir", str(_data / "pharma_neurosynth_cache")]
    return _run(cmd, "Step 3: Rebuild merged_sources (neurovault_pharma + all caches)", dry_run)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run audit data-rebuild steps")
    ap.add_argument("--step", type=int, default=0, choices=(0, 1, 2, 3), help="Run only this step (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    steps = []
    if args.step == 0:
        steps = [1, 2, 3]
    else:
        steps = [args.step]

    ok = True
    for s in steps:
        if s == 1:
            ok &= step1_decoder_full_vocab(args.dry_run)
        elif s == 2:
            ok &= step2_unified_cache(args.dry_run)
        elif s == 3:
            ok &= step3_merged_sources(args.dry_run)

    if ok and not args.dry_run and steps:
        print("\nDone. Verify: python neurolab/scripts/check_training_readiness.py --require-expanded")
        print("             python neurolab/scripts/verify_full_cache_pipeline.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
