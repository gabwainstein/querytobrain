#!/usr/bin/env python3
"""
Download OpenNeuro pharmacological fMRI datasets (raw data, CC0).
Key datasets: psilocybin (ds006072), LSD/psilocybin (ds003059), and others.
Raw timeseries require preprocessing (drug vs placebo contrast → parcellate) for training maps.

Usage:
  python neurolab/scripts/download_openneuro_pharma.py --output-dir neurolab/data/openneuro_pharma
  python neurolab/scripts/download_openneuro_pharma.py --datasets ds006072 ds003059  # only these
  python neurolab/scripts/download_openneuro_pharma.py --list-only  # print dataset list and exit

Requires: openneuro-py (pip install openneuro-py) or datalad for download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Curated pharmacological / drug-challenge OpenNeuro dataset IDs (CC0).
# ds006072: Psilocybin precision fMRI (Nature); ds003059: LSD and psilocybin resting-state (Carhart-Harris).
# Add more as needed: ketamine, MDMA, etc. (search openneuro.org for "ketamine", "pharmacological").
OPENNEURO_PHARMA_DATASETS = [
    "ds006072",   # Psilocybin precision functional mapping (psilocybin vs methylphenidate)
    "ds003059",   # LSD and psilocybin resting-state fMRI (Carhart-Harris)
    "ds003747",   # Ketamine (example; verify on openneuro.org)
    "ds002790",   # Nicotine (example; verify)
]
# Optional: extend with more IDs from https://openneuro.org (search: drug, ketamine, MDMA, pharmacological)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OpenNeuro pharmacological fMRI datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base dir for dataset clones (default: neurolab/data/openneuro_pharma)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset IDs to download (default: built-in OPENNEURO_PHARMA_DATASETS)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print dataset list and manual/datalad instructions; do not download.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "openneuro_pharma"
    datasets = args.datasets or OPENNEURO_PHARMA_DATASETS

    if args.list_only:
        print("OpenNeuro pharmacological dataset IDs (use with --datasets or download all):")
        for d in datasets:
            print(f"  {d}  https://openneuro.org/datasets/{d}")
        print("\nDownload via datalad (recommended):")
        print(f"  mkdir -p {out_dir}")
        for d in datasets:
            print(f"  datalad clone https://github.com/OpenNeuroDatasets/{d}.git {out_dir}/{d}")
        print("  cd {out_dir} && datalad get .  # or get specific subdirs to save space")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try openneuro-py first
    try:
        import openneuro
        for ds in datasets:
            dest = out_dir / ds
            if dest.exists():
                print(f"  {ds} already present at {dest}; skip or remove to re-download.")
                continue
            print(f"Downloading {ds} -> {dest} ...")
            openneuro.download(dataset=ds, target_dir=str(dest))
        print(f"Done. Data under {out_dir}")
        return 0
    except ImportError:
        pass

    # Fallback: datalad
    try:
        for ds in datasets:
            dest = out_dir / ds
            if dest.exists():
                print(f"  {ds} already present; skipping.")
                continue
            url = f"https://github.com/OpenNeuroDatasets/{ds}.git"
            subprocess.run(["datalad", "clone", url, str(dest)], check=True)
        print(f"Cloned. Run 'datalad get <path>' to fetch file contents. Base: {out_dir}")
        return 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("datalad not found or clone failed.", file=sys.stderr)

    print("Install openneuro-py (pip install openneuro-py) or datalad to download.", file=sys.stderr)
    print("Or run with --list-only and use the printed datalad commands manually.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
