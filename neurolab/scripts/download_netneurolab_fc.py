#!/usr/bin/env python3
"""
Download netneurolab FC matrices (e.g. liu_fc-pyspi: Schaefer 400 group-average rsfMRI FC).
Data hosted on OSF; script provides acquisition guide and optional download via osfclient.

Usage:
  python neurolab/scripts/download_netneurolab_fc.py --output-dir neurolab/data/netneurolab_fc
  python neurolab/scripts/download_netneurolab_fc.py --list-only  # print URLs and manual steps

Requires for download: pip install osfclient (optional)
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

# liu_fc-pyspi: Liu et al. benchmarking FC methods; OSF project
LIU_FC_OSF = "https://osf.io/75je2"
LIU_FC_REPO = "https://github.com/netneurolab/liu_fc-pyspi"

# luppi-cognitive-matching: anesthesia FC (propofol, sevoflurane, ketamine)
LUPPI_COGNITIVE_REPO = "https://github.com/netneurolab/luppi-cognitive-matching"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download netneurolab FC matrices (liu_fc-pyspi, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: neurolab/data/netneurolab_fc)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print data URLs and manual acquisition steps; do not download.",
    )
    parser.add_argument(
        "--use-osfclient",
        action="store_true",
        help="Attempt download via osfclient (pip install osfclient)",
    )
    args = parser.parse_args()

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "netneurolab_fc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_only:
        print("netneurolab FC datasets")
        print("=" * 60)
        print("liu_fc-pyspi (Liu et al. FC benchmarking):")
        print(f"  OSF: {LIU_FC_OSF}")
        print(f"  Repo: {LIU_FC_REPO}")
        print("  Content: fc_cons_400.npy (Schaefer 400 group-average rsfMRI FC)")
        print()
        print("luppi-cognitive-matching (anesthesia FC):")
        print(f"  Repo: {LUPPI_COGNITIVE_REPO}")
        print("  Content: FC matrices from propofol, sevoflurane, ketamine vs wakefulness")
        print()
        print("Manual steps:")
        print("  1. Open OSF project, download data/derivatives (or data/raw)")
        print("  2. Place fc_cons_400.npy (or equivalent) in:", out_dir)
        print("  3. Reparcellate Schaefer 400 → Glasser+Tian 392 for pipeline")
        return 0

    if args.use_osfclient:
        # Direct HTTP download (osfclient Python API changed)
        try:
            from urllib.request import urlretrieve
            # liu_fc-pyspi OSF: derivatives with FC/SC matrices
            OSF_FILES = [
                ("https://osf.io/download/w8dp3/", "sc_cons_wei.npy"),  # structural
                ("https://osf.io/download/btm3d/", "x_comm_mats.npy"),  # comm matrices
                ("https://osf.io/download/enygf/", "pyspi_hcp_schaefer100x7gsr_term_profile_mean.npy"),  # FC term mean
            ]
            for url, fname in OSF_FILES:
                target = out_dir / fname
                if not target.exists() or target.stat().st_size == 0:
                    print(f"Downloading {fname}...")
                    urlretrieve(url, target)
                else:
                    print(f"  {fname} exists, skip")
            # Build fc_cons from term profile mean if we have it
            term_mean = out_dir / "pyspi_hcp_schaefer100x7gsr_term_profile_mean.npy"
            fc_cons = out_dir / "fc_cons_400.npy"
            if term_mean.exists() and not fc_cons.exists():
                try:
                    import numpy as np
                    arr = np.load(term_mean)
                    if arr.ndim == 3:  # (n_terms, n, n)
                        fc_mat = np.nanmean(arr, axis=0)
                    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                        fc_mat = arr
                    else:
                        fc_mat = None
                    if fc_mat is not None and fc_mat.shape[0] >= 100:
                        n = min(400, fc_mat.shape[0])
                        fc_small = fc_mat[:n, :n] if fc_mat.shape[0] >= n else fc_mat
                        np.save(fc_cons, fc_small)
                        print(f"  Saved fc_cons from term mean -> {fc_cons.name}")
                except Exception as e:
                    print(f"  Could not derive fc_cons: {e}", file=sys.stderr)
        except Exception as e:
            print(f"OSF download failed: {e}", file=sys.stderr)
            return 1
    else:
        # Write acquisition guide
        with open(out_dir / "acquisition_guide.json", "w") as f:
            json.dump({
                "liu_fc_pyspi": {
                    "osf": LIU_FC_OSF,
                    "repo": LIU_FC_REPO,
                    "parcellation": "Schaefer 400",
                    "files": ["fc_cons_400.npy", "data/derivatives/*"],
                },
                "luppi_cognitive_matching": {
                    "repo": LUPPI_COGNITIVE_REPO,
                    "content": "anesthesia FC (propofol, sevoflurane, ketamine)",
                },
                "target_parcellation": "Glasser+Tian 392",
            }, f, indent=2)
        print(f"Wrote acquisition guide to {out_dir}")
        print("Run with --use-osfclient to attempt OSF download, or download manually from OSF.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
