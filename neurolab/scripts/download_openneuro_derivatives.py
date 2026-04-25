#!/usr/bin/env python3
"""
Download OpenNeuro fMRIPrep derivatives for pharmacological datasets.
OpenNeuroDerivatives hosts preprocessed outputs; avoids running fMRIPrep on raw data.

Checks: https://github.com/OpenNeuroDerivatives/{dataset_id}-fmriprep
If derivative exists, clones via datalad. Otherwise prints manual steps.

Usage:
  python neurolab/scripts/download_openneuro_derivatives.py --output-dir neurolab/data/openneuro_derivatives
  python neurolab/scripts/download_openneuro_derivatives.py --datasets ds006072 ds003059
  python neurolab/scripts/download_openneuro_derivatives.py --list-only

Requires: datalad (pip install datalad) or git for clone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Pharmacological OpenNeuro datasets — check OpenNeuroDerivatives for fMRIPrep
OPENNEURO_PHARMA_DATASETS = [
    "ds006072",   # Psilocybin precision fMRI
    "ds003059",   # LSD and psilocybin resting-state
    "ds003747",   # Ketamine (verify)
    "ds002790",   # Nicotine (verify)
]

DERIVATIVES_ORG = "OpenNeuroDerivatives"


def _check_derivative_exists(dataset_id: str) -> bool:
    """Check if OpenNeuroDerivatives has fMRIPrep for this dataset."""
    try:
        url = f"https://github.com/{DERIVATIVES_ORG}/{dataset_id}-fmriprep"
        req = __import__("urllib.request").request.Request(url, headers={"User-Agent": "NeuroLab/1.0"})
        with __import__("urllib.request").request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro fMRIPrep derivatives for pharma datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/openneuro_derivatives)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset IDs (default: OPENNEURO_PHARMA_DATASETS)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print availability and datalad commands; do not download.",
    )
    args = parser.parse_args()

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "openneuro_derivatives"
    datasets = args.datasets or OPENNEURO_PHARMA_DATASETS

    if args.list_only:
        print("OpenNeuro fMRIPrep derivatives (OpenNeuroDerivatives org)")
        print("=" * 60)
        for ds in datasets:
            url = f"https://github.com/{DERIVATIVES_ORG}/{ds}-fmriprep"
            print(f"  {ds}: {url}")
        print()
        print("Download via datalad:")
        print(f"  mkdir -p {out_dir}")
        for ds in datasets:
            print(f"  datalad clone https://github.com/{DERIVATIVES_ORG}/{ds}-fmriprep.git {out_dir}/{ds}-fmriprep")
        print("  cd {out_dir} && datalad get .  # fetch file contents")
        print()
        print("If derivative repo does not exist, use download_openneuro_pharma.py for raw data")
        print("and run fMRIPrep manually (or via cloud job).")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        dest = out_dir / f"{ds}-fmriprep"
        if dest.exists():
            print(f"  {ds}-fmriprep already present at {dest}; skipping.")
            continue
        url = f"https://github.com/{DERIVATIVES_ORG}/{ds}-fmriprep.git"
        print(f"Cloning {ds}-fmriprep ...")
        try:
            subprocess.run(["datalad", "clone", url, str(dest)], check=True)
        except FileNotFoundError:
            try:
                subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
            except subprocess.CalledProcessError:
                print(f"  Clone failed. Try: datalad clone {url} {dest}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}", file=sys.stderr)

    print(f"Derivatives under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
