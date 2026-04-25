#!/usr/bin/env python3
"""
Fetch all neuromaps annotations needed for the cache into the repo.

Saves to neurolab/data/neuromaps_data by default so build_neuromaps_cache.py
can use local data (no download at build time). Run once if the repo does not
include the data; if you already fetched, this dir is used automatically.

Requires: pip install neuromaps [setuptools]

  python neurolab/scripts/download_neuromaps_data.py
  python neurolab/scripts/download_neuromaps_data.py --output-dir /path/to/neuromaps_data
  python neurolab/scripts/download_neuromaps_data.py --space MNI152 --tags receptors
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTPUT_DIR = os.path.join(_repo_root, "neurolab", "data", "neuromaps_data")


def _annotations_to_rows(ann):
    """Turn available_annotations() into list of (source, desc, space, den_res)."""
    if hasattr(ann, "columns"):
        cols = [c for c in ("source", "desc", "space", "den", "res") if c in ann.columns]
        if not cols:
            return []
        return [tuple(getattr(row, c) for c in cols) for row in ann.itertuples(index=False)]
    try:
        return list(ann)
    except TypeError:
        return list(iter(ann))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch all neuromaps annotations into repo (for build_neuromaps_cache)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save (default: neurolab/data/neuromaps_data)",
    )
    parser.add_argument("--space", default="MNI152", help="Filter by space (default: MNI152)")
    parser.add_argument("--tags", default=None, help="Filter by tags, comma-separated (default: all)")
    parser.add_argument("--format", default=None, help="Filter by format (e.g. volumetric)")
    args = parser.parse_args()

    try:
        from neuromaps.datasets import available_annotations, fetch_annotation
        import neuromaps.datasets.annotations as _ann
    except ImportError:
        print("Install neuromaps: pip install neuromaps", file=sys.stderr)
        return 1

    # neuromaps passes data_dir as str to nilearn which expects Path; patch so .mkdir() works
    _orig_fetch = getattr(_ann, "_fetch_file", None)
    if _orig_fetch:
        def _wrap_fetch(url, data_dir, **kw):
            return _orig_fetch(url, Path(data_dir) if not isinstance(data_dir, Path) else data_dir, **kw)
        _ann._fetch_file = _wrap_fetch

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = Path(_repo_root) / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    filters = {"space": args.space}
    if args.tags:
        filters["tags"] = [t.strip() for t in args.tags.split(",")]
    if args.format:
        filters["format"] = args.format

    ann = available_annotations(**filters)
    items = _annotations_to_rows(ann)
    if not items:
        print("No annotations match filters.", file=sys.stderr)
        return 1

    print(f"Fetching {len(items)} annotation(s) to {out} ...")
    ok = 0
    for i, row in enumerate(items):
        if len(row) < 3:
            continue
        src, desc, sp = row[0], row[1], row[2]
        den_res = row[3] if len(row) > 3 else None
        try:
            # Pass Path so neuromaps internals that call .mkdir() work (they fail on str in some versions)
            kwargs = {"source": src, "desc": desc, "space": sp, "data_dir": out}
            if den_res is not None and str(den_res) != "nan":
                kwargs["den"] = den_res
            result = fetch_annotation(**kwargs)
            if result:
                ok += 1
        except (TypeError, KeyError):
            try:
                kwargs = {"source": src, "desc": desc, "space": sp, "data_dir": out}
                if den_res is not None and str(den_res) != "nan":
                    kwargs["res"] = den_res
                result = fetch_annotation(**kwargs)
                if result:
                    ok += 1
            except Exception as e:
                print(f"  Skip {src}/{desc}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  Skip {src}/{desc}: {e}", file=sys.stderr)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(items)} {src}/{desc}")
    print(f"Fetched {ok}/{len(items)} annotations to {out}")
    print("Run build_neuromaps_cache.py; it will use this dir by default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
