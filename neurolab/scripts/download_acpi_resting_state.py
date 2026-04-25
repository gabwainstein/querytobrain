#!/usr/bin/env python3
"""
Download ACPI (Addiction Connectome Preprocessed Initiative) resting-state data.
192 subjects, cannabis + cocaine, 8 preprocessing pipelines.
Preprocessed to multiple atlases (AAL, Craddock 200, Harvard-Oxford, Random parcels).

Source: NITRC https://fcon_1000.projects.nitrc.org/indi/ACPI/

Usage:
  python neurolab/scripts/download_acpi_resting_state.py --output-dir neurolab/data/acpi_resting_state
  python neurolab/scripts/download_acpi_resting_state.py --list-only

Note: NITRC may require registration. Script prints download URLs and manual steps.
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

ACPI_NITRC_BASE = "https://fcon_1000.projects.nitrc.org/indi/ACPI"
ACPI_DOCS = "https://fcon_1000.projects.nitrc.org/indi/ACPI/html/"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ACPI resting-state preprocessed data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/acpi_resting_state)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print NITRC URLs and manual steps; do not download.",
    )
    args = parser.parse_args()

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "acpi_resting_state"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_only:
        print("ACPI — Addiction Connectome Preprocessed Initiative")
        print("=" * 60)
        print(f"Documentation: {ACPI_DOCS}")
        print(f"NITRC base: {ACPI_NITRC_BASE}")
        print()
        print("Content: 192 subjects, cannabis + cocaine, 8 preprocessing pipelines")
        print("Parcellations: AAL, Craddock 200, Harvard-Oxford, Random parcels")
        print()
        print("Manual steps:")
        print("  1. Visit NITRC, register if required")
        print("  2. Navigate to ACPI project, download preprocessed data")
        print("  3. Place ROI timeseries / connectivity matrices in:", out_dir)
        print("  4. Resample to Glasser+Tian 392 if needed for pipeline")
        return 0

    with open(out_dir / "acquisition_guide.json", "w") as f:
        json.dump({
            "name": "ACPI",
            "description": "Addiction Connectome Preprocessed Initiative",
            "subjects": 192,
            "substances": ["cannabis", "cocaine"],
            "preprocessing_pipelines": 8,
            "parcellations": ["AAL", "Craddock 200", "Harvard-Oxford", "Random parcels"],
            "nitrc_url": ACPI_NITRC_BASE,
            "docs_url": ACPI_DOCS,
        }, f, indent=2)

    print(f"Wrote acquisition guide to {out_dir}")
    print("ACPI data requires manual download from NITRC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
