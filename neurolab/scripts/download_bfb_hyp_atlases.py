#!/usr/bin/env python3
"""
Download instructions for Zaborszky basal forebrain and Neudorfer hypothalamus atlases.

Required for: --atlas glasser+tian+brainstem+bfb+hyp

Zaborszky (Ch1-2, Ch4):
  - JuBrain Anatomy Toolbox: https://github.com/inm7/jubrain-anatomy-toolbox
  - Or siibra: pip install siibra  (will auto-fetch on first build)
  - Extract NIfTI masks to: neurolab/data/atlas_cache/zaborszky/
  - Need: Ch1-2 (medial septum/diagonal band) and Ch4 (nucleus basalis)

Neudorfer (LH, TM, PA):
  - Zenodo (auto-fetch): https://zenodo.org/records/3903588  (build_combined_atlas downloads automatically)
  - Or Lead-DBS: https://www.lead-dbs.org/
  - Or Scientific Data: doi 10.1038/s41597-020-00644-6
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

JUBRAIN_URL = "https://github.com/inm7/jubrain-anatomy-toolbox"
NEUDORFER_ZENODO = "https://zenodo.org/records/3903588"
NEUDORFER_PAPER = "https://www.nature.com/articles/s41597-020-00644-6"
LEAD_DBS_URL = "https://www.lead-dbs.org/"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir.parent / "data" / "atlas_cache"

    print("Basal Forebrain (Zaborszky) + Hypothalamus (Neudorfer)")
    print("=" * 60)
    print()
    print("1. Zaborszky basal forebrain (Ch1-2, Ch4)")
    print(f"   - JuBrain: {JUBRAIN_URL}")
    print("   - Or: pip install siibra  (build_combined_atlas will fetch via EBRAINS)")
    print(f"   - Manual: extract Ch1-2 and Ch4 NIfTI to {cache_dir / 'zaborszky'}")
    print()
    print("2. Neudorfer hypothalamus (LH, TM, PA)")
    print("   - Auto-fetch from Zenodo on first build (no manual step needed)")
    print("   - Zenodo: https://zenodo.org/records/3903588")
    print(f"   - Or manual: {LEAD_DBS_URL} or {NEUDORFER_PAPER}")
    print()
    print("3. Then run:")
    print("   python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem+bfb+hyp")
    print()
    webbrowser.open(JUBRAIN_URL)
    webbrowser.open(NEUDORFER_PAPER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
