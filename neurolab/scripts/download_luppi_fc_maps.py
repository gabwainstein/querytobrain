#!/usr/bin/env python3
"""
Download Luppi et al. 2023 (Science Advances) pharmacologically induced FC maps.
10 drugs, 15 contrasts; Schaefer 100 parcellation. Resample to Glasser+Tian 392 for pipeline.

Paper: "In vivo mapping of pharmacologically induced functional reorganization onto the
human brain's neurotransmitter landscape" (Science Advances, eadf8332)
Data: Cambridge repository, Science Advances supplementary.

Usage:
  python neurolab/scripts/download_luppi_fc_maps.py --output-dir neurolab/data/luppi_fc_maps
  python neurolab/scripts/download_luppi_fc_maps.py --list-only  # print URLs and manual steps

Note: Supplementary FC matrices may be in the paper's SI or Cambridge/Figshare.
If not publicly downloadable, script prints manual acquisition steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
except ImportError:
    urlopen = Request = HTTPError = URLError = None

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Luppi 2023: 10 drugs, multiple contrasts each
# Drugs: propofol, sevoflurane, ketamine, LSD, psilocybin, DMT, ayahuasca, MDMA, modafinil, methylphenidate
LUPPI_CAMBRIDGE_URI = "https://www.repository.cam.ac.uk/handle/1810/354086"
LUPPI_DOI = "10.1126/sciadv.adf8332"
# Science Advances supplementary often on separate URL
SCIADV_SUPP_BASE = "https://www.science.org/doi/suppl/10.1126/sciadv.adf8332"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Luppi et al. 2023 pharmacologically induced FC maps."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/luppi_fc_maps)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print data URLs and manual acquisition steps; do not download.",
    )
    args = parser.parse_args()

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "luppi_fc_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_only:
        print("Luppi et al. 2023 (Science Advances) — Pharmacological FC maps")
        print("=" * 60)
        print(f"Paper: {LUPPI_DOI}")
        print(f"Cambridge repository: {LUPPI_CAMBRIDGE_URI}")
        print(f"Science Advances supplementary: {SCIADV_SUPP_BASE}")
        print()
        print("Data format: Schaefer 100 parcellation (FC matrices).")
        print("Manual steps:")
        print("  1. Open Cambridge repository, download supplementary files (if available)")
        print("  2. Or: Science Advances → Supplementary Materials → download FC/data files")
        print("  3. Place files in:", out_dir)
        print("  4. Run build script to resample Schaefer 100 → Glasser+Tian 392")
        return 0

    # Try direct download of Cambridge published version (PDF) - FC data may be in supplement
    try:
        pdf_url = "https://www.repository.cam.ac.uk/bitstreams/9bd755c5-b78b-400a-a927-060887ed3327/download"
        pdf_path = out_dir / "Luppi2023_SciAdv_published.pdf"
        if not pdf_path.exists():
            print("Downloading Luppi 2023 published version (PDF)...")
            req = Request(pdf_url, headers={"User-Agent": "NeuroLab/1.0"})
            with urlopen(req, timeout=60) as resp:
                pdf_path.write_bytes(resp.read())
            print(f"  Saved {pdf_path.name}")
        else:
            print(f"  {pdf_path.name} exists")
    except Exception as e:
        print(f"Cambridge download: {e}", file=sys.stderr)

    # FC matrices: paper supplement may have CSV/Excel; extract or manual
    # Create placeholder instructions

    with open(out_dir / "acquisition_guide.json", "w") as f:
        json.dump({
            "paper": "Luppi et al. 2023, Science Advances eadf8332",
            "drugs": [
                "propofol", "sevoflurane", "ketamine", "LSD", "psilocybin",
                "DMT", "ayahuasca", "MDMA", "modafinil", "methylphenidate",
            ],
            "parcellation_source": "Schaefer 100",
            "target_parcellation": "Glasser+Tian 392",
            "cambridge_uri": LUPPI_CAMBRIDGE_URI,
            "sciadv_supp": SCIADV_SUPP_BASE,
        }, f, indent=2)

    print(f"Wrote acquisition guide to {out_dir}")
    print("Luppi FC data may require manual download from Cambridge/Science Advances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
