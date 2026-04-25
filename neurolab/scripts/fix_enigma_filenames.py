#!/usr/bin/env python3
"""Fix ENIGMA toolbox filename typos (e.g. case-control vs case-controls).

Run from repo root. Patches the enigmatoolbox summary_statistics dir by copying
files that have the wrong name to the expected name. Idempotent.
"""
from __future__ import annotations

import os
import shutil
import sys

# Known typos: (actual_filename, expected_by_toolbox)
FIXES = [
    # Schizophrenia: case-control (singular) -> case-controls (plural)
    ("Schizophrenia_case-control_SubVol.csv", "Schizophrenia_case-controls_SubVol.csv"),
]


def main() -> int:
    try:
        import enigmatoolbox
    except ImportError:
        print("enigmatoolbox not installed", file=sys.stderr)
        return 1

    base = os.path.dirname(enigmatoolbox.__file__)
    stats_dir = os.path.join(base, "datasets", "summary_statistics")
    if not os.path.isdir(stats_dir):
        print(f"Not found: {stats_dir}", file=sys.stderr)
        return 1

    fixed = 0
    for actual, expected in FIXES:
        src = os.path.join(stats_dir, actual)
        dst = os.path.join(stats_dir, expected)
        if os.path.exists(src) and (not os.path.exists(dst) or os.path.samefile(src, dst) is False):
            shutil.copy2(src, dst)
            print(f"Fixed: {actual} -> {expected}")
            fixed += 1
        elif not os.path.exists(src) and os.path.exists(dst):
            pass  # already fixed
        elif not os.path.exists(src):
            print(f"Skip (missing): {actual}", file=sys.stderr)

    if fixed:
        print(f"Fixed {fixed} file(s)")
    else:
        print("No fixes needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
