#!/usr/bin/env python3
"""
Download Brainstem Navigator atlas for Glasser+Tian+brainstem atlas.

NITRC requires manual download. This script:
1. Opens the NITRC project page in your browser
2. Prints where to extract the zip

After manual download, extract BrainstemNavigatorv1.0.zip to:
  neurolab/data/atlas_cache/

So that you have: neurolab/data/atlas_cache/BrainstemNavigatorv1.0/... with
subdirs containing nucleus *.nii.gz files.

Then run: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

NITRC_URL = "https://www.nitrc.org/projects/brainstemnavig"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir.parent / "data" / "atlas_cache"

    print("Brainstem Navigator (Hansen et al. 2024 Nat Neurosci)")
    print("=" * 60)
    print(f"1. Opening NITRC page: {NITRC_URL}")
    webbrowser.open(NITRC_URL)
    print()
    print("2. Click 'Download Now' and get BrainstemNavigatorv1.0.zip (~87 MB)")
    print()
    print(f"3. Extract to: {cache_dir}")
    print("   Result should be: atlas_cache/BrainstemNavigatorv1.0/<atlas subdirs>/*.nii.gz")
    print()
    print("4. Then run:")
    print("   python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
