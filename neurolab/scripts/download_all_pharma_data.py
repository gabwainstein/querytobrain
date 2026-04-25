#!/usr/bin/env python3
"""
Download all open pharmacological data used in the pipeline (no license filter).
Calls in sequence:
  1. PDSP Ki database (CSV) -> neurolab/data/pdsp_ki/
  2. OpenNeuro pharma datasets (raw fMRI) -> neurolab/data/openneuro_pharma/
  3. NeuroVault pharma-related collections (contrast maps) -> neurolab/data/neurovault_pharma_data/
  4. NeuroSynth data (if not present) for build_pharma_neurosynth_cache.py

Does not re-download ontologies or neuromaps; use download_ontologies.py --clinical and
download_neuromaps_data.py separately. Pharmacological meta-analysis maps are generated
by build_pharma_neurosynth_cache.py (uses NeuroSynth/NiMARE).

Usage:
  python neurolab/scripts/download_all_pharma_data.py
  python neurolab/scripts/download_all_pharma_data.py --skip-pdsp --skip-openneuro  # only NeuroVault pharma
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_root = _scripts.parent.parent


def _run(script: str, *extra_args: str) -> bool:
    cmd = [sys.executable, str(_scripts / script)] + list(extra_args)
    r = subprocess.run(cmd, cwd=str(_root))
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all open pharmacological data.")
    parser.add_argument("--skip-pdsp", action="store_true", help="Skip PDSP Ki download")
    parser.add_argument("--skip-openneuro", action="store_true", help="Skip OpenNeuro pharma datasets")
    parser.add_argument("--skip-neurovault-pharma", action="store_true", help="Skip NeuroVault pharma search/download")
    args = parser.parse_args()

    ok = True
    if not args.skip_pdsp:
        print("[1/3] PDSP Ki database ...")
        if not _run("download_pdsp_ki.py"):
            print("  PDSP download failed (may require manual download from pdsp.unc.edu).", file=sys.stderr)
            ok = False
    else:
        print("[1/3] PDSP Ki (skipped)")

    if not args.skip_openneuro:
        print("[2/3] OpenNeuro pharma datasets ...")
        if not _run("download_openneuro_pharma.py"):
            print("  OpenNeuro: install openneuro-py or datalad, or run with --list-only for manual steps.", file=sys.stderr)
            ok = False
    else:
        print("[2/3] OpenNeuro (skipped)")

    if not args.skip_neurovault_pharma:
        print("[3/3] NeuroVault pharma collections ...")
        if not _run("download_neurovault_pharma.py"):
            print("  NeuroVault pharma download failed.", file=sys.stderr)
            ok = False
    else:
        print("[3/3] NeuroVault pharma (skipped)")

    print("\nDone. Next:")
    print("  - build_pharma_neurosynth_cache.py  (meta-analysis maps for drug terms)")
    print("  - build_neurovault_cache.py --data-dir neurolab/data/neurovault_pharma_data  (if you downloaded NeuroVault pharma)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
