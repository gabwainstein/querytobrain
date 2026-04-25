#!/usr/bin/env python3
"""
Build the Mercedes of the training set: highest-quality merged_sources.

Rebuilds source caches with full quality (AHBA labels, expanded PET descriptors,
contrast definitions, poor-term filter) then merges with map-type prefixes and
PET residuals. Optionally runs final validation.

Quality settings:
  - abagen: AHBA labels, 500 tiered terms, 5 gradient PCs
  - neuromaps: expanded PET/Cognitive/Perfusion labels (no authors)
  - neurovault: contrast_definition (API fetch when missing)
  - neurovault_pharma: same, poor terms filtered at merge
  - receptor_reference: AHBA labels (when available)
  - Merge: map type prefixes (fMRI:, PET:, Gene:, Structural:), poor-term filter

Usage:
  python neurolab/scripts/build_mercedes_training_set.py
  python neurolab/scripts/build_mercedes_training_set.py --merge-only   # use existing caches
  python neurolab/scripts/build_mercedes_training_set.py --validate     # run final review after merge
  python neurolab/scripts/build_mercedes_training_set.py --receptor-path path/to/hansen.csv
"""
from __future__ import annotations

import argparse
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
    print(f"\n{'='*70}\n  {desc}\n  {' '.join(cmd)}\n{'='*70}")
    r = subprocess.run(cmd, cwd=str(_repo_root))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build Mercedes training set: highest-quality merged_sources"
    )
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip cache rebuilds; only run merge (use existing caches)")
    ap.add_argument("--validate", action="store_true",
                    help="Run final_training_review.py after merge")
    ap.add_argument("--receptor-path", default=None,
                    help="Hansen receptor CSV/NPZ to merge (optional)")
    ap.add_argument("--skip-neurovault-pharma", action="store_true")
    ap.add_argument("--skip-abagen", action="store_true")
    ap.add_argument("--skip-neuromaps", action="store_true")
    ap.add_argument("--skip-enigma", action="store_true")
    args = ap.parse_args()

    ok = True

    if not args.merge_only:
        # 1. Main NeuroVault (task contrasts, contrast_definition)
        nv_curated = _data / "neurovault_curated_data"
        nv_legacy = _data / "neurovault_data"
        has_curated = (nv_curated / "manifest.json").exists() or (nv_curated / "downloads" / "neurovault").exists()
        has_legacy = (nv_legacy / "manifest.json").exists() or (nv_legacy / "downloads" / "neurovault").exists()
        nv_data = nv_curated if has_curated else (nv_legacy if has_legacy else None)
        if nv_data and ((nv_data / "manifest.json").exists() or (nv_data / "downloads" / "neurovault").exists()):
            nv_args = ["--data-dir", str(nv_data), "--output-dir", str(_data / "neurovault_cache")]
            if nv_data == nv_curated:
                nv_args += ["--average-subject-level"]
            if not (nv_data / "manifest.json").exists():
                nv_args += ["--from-downloads"]
            ok &= run("build_neurovault_cache.py", nv_args, "neurovault task (contrast definitions)")
        else:
            print("\n  [SKIP] NeuroVault data not found (neurovault_data or neurovault_curated_data)")

        # 2. abagen (AHBA labels)
        if not args.skip_abagen:
            ok &= run("build_abagen_cache.py",
                      ["--output-dir", str(_data / "abagen_cache"), "--all-genes"],
                      "abagen (AHBA labels, all genes)")

        # 3. neuromaps (expanded PET/Cognitive/Perfusion labels)
        if not args.skip_neuromaps:
            ok &= run("build_neuromaps_cache.py",
                      ["--cache-dir", str(_data / "neuromaps_cache"),
                       "--data-dir", str(_data / "neuromaps_data"), "--max-annot", "0"],
                      "neuromaps (expanded labels, no authors)")

        # 4. receptor_reference (AHBA labels)
        ref_dir = _data / "receptor_reference_cache"
        if (ref_dir / "term_maps.npz").exists():
            ok &= run("build_receptor_reference_cache.py",
                      ["--output-dir", str(ref_dir)],
                      "receptor_reference (AHBA labels)")
        else:
            print("\n  [SKIP] receptor_reference_cache not found; run build_receptor_reference_cache.py first")

        # 5. neurovault_pharma (contrast_definition, API fetch)
        nv_pharma_data = _data / "neurovault_pharma_data"
        has_pharma = (nv_pharma_data / "manifest.json").exists() or (nv_pharma_data / "downloads" / "neurovault").exists()
        if not args.skip_neurovault_pharma and has_pharma:
            ok &= run("build_neurovault_cache.py",
                      ["--data-dir", str(nv_pharma_data),
                       "--output-dir", str(_data / "neurovault_pharma_cache"), "--from-downloads", "--pharma-add-drug", "--cluster-by-description"],
                      "neurovault_pharma (contrast definitions)")

    # 5. Merge: Mercedes quality (poor-term filter, map type prefixes, PET residuals)
    unified = _data / "unified_cache" / "term_maps.npz"
    if not unified.exists():
        print("\n  [ERROR] unified_cache not found. Run merge_neuroquery_neurosynth_cache first.", file=sys.stderr)
        return 1

    merge_args = [
        "--cache-dir", str(_data / "unified_cache"),
        "--output-dir", str(_data / "merged_sources"),
        "--no-ontology",
        "--save-term-sources",
    ]
    if (_data / "neurovault_cache" / "term_maps.npz").exists():
        merge_args += ["--neurovault-cache-dir", str(_data / "neurovault_cache")]
    if (_data / "neurovault_pharma_cache" / "term_maps.npz").exists():
        merge_args += ["--neurovault-pharma-cache-dir", str(_data / "neurovault_pharma_cache")]
    if (_data / "neuromaps_cache" / "annotation_maps.npz").exists():
        merge_args += ["--neuromaps-cache-dir", str(_data / "neuromaps_cache")]
    if (_data / "enigma_cache" / "term_maps.npz").exists() and not args.skip_enigma:
        merge_args += ["--enigma-cache-dir", str(_data / "enigma_cache")]
    if (_data / "abagen_cache" / "term_maps.npz").exists() and not args.skip_abagen:
        merge_args += [
            "--abagen-cache-dir", str(_data / "abagen_cache"),
            "--max-abagen-terms", "500",
            "--abagen-add-gradient-pcs", "3",
            "--add-pet-residuals",
        ]
    if (_data / "pharma_neurosynth_cache" / "term_maps.npz").exists():
        merge_args += ["--pharma-neurosynth-cache-dir", str(_data / "pharma_neurosynth_cache")]
    if (_data / "receptor_reference_cache" / "term_maps.npz").exists():
        merge_args += ["--receptor-reference-cache-dir", str(_data / "receptor_reference_cache")]
    if args.receptor_path and Path(args.receptor_path).exists():
        merge_args += ["--receptor-path", args.receptor_path]

    ok &= run("build_expanded_term_maps.py", merge_args,
              "Merge Mercedes (poor-term filter, map type prefixes, PET residuals)")

    # 6. Final review
    if ok and args.validate:
        print("\n" + "=" * 70)
        print("  FINAL TRAINING REVIEW")
        print("=" * 70)
        ok &= run("final_training_review.py",
                  ["--cache-dir", str(_data / "merged_sources"), "--strict"],
                  "Final maps × terms validation")

    if ok:
        print("\n" + "=" * 70)
        print("  Mercedes training set ready.")
        print("  Output: neurolab/data/merged_sources")
        if not args.validate:
            print("  Run final review: python neurolab/scripts/final_training_review.py")
        print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
