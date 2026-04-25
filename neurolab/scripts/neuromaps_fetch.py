#!/usr/bin/env python3
"""
List and fetch brain maps from neuromaps (receptors, structure, gene expression, etc.).

Requires: pip install neuromaps setuptools (neuromaps 0.0.5 needs pkg_resources from setuptools)

Usage:
  python neuromaps_fetch.py list [--tags receptors] [--format volumetric]
  python neuromaps_fetch.py fetch --source <source> --desc <desc> [--space MNI152] [--output-dir path]
  python neuromaps_fetch.py fetch-all [--output-dir path] [--format volumetric] [--space MNI152] [--tags ...]
  python neuromaps_fetch.py describe

fetch-all: Downloads every annotation (or filtered subset) into --output-dir. Total size is manageable
(~86 annotations; typically a few hundred MB to ~1 GB for all, or less if filtered by --format volumetric).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _has_neuromaps() -> bool:
    try:
        import neuromaps.datasets
        return True
    except ImportError:
        return False


def cmd_list(tags: str | None, space: str | None, fmt: str | None) -> int:
    if not _has_neuromaps():
        print("Install neuromaps: pip install neuromaps", file=sys.stderr)
        return 1
    from neuromaps.datasets import available_annotations
    filters = {}
    if tags:
        filters["tags"] = tags.split(",")
    if space:
        filters["space"] = space
    if fmt:
        filters["format"] = fmt
    ann = available_annotations(**filters)
    if hasattr(ann, "to_dict"):
        ann = ann.to_dict(orient="records") if hasattr(ann, "to_dict") else ann
    print(json.dumps(ann, indent=2, default=str))
    return 0


def cmd_fetch(source: str, desc: str | None, space: str | None, den: str | None, output_dir: Path | None) -> int:
    if not _has_neuromaps():
        print("Install neuromaps: pip install neuromaps", file=sys.stderr)
        return 1
    from neuromaps.datasets import fetch_annotation
    kwargs = {"source": source}
    if desc:
        kwargs["desc"] = desc
    if space:
        kwargs["space"] = space
    if den:
        kwargs["den"] = den
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["data_dir"] = str(output_dir)
    try:
        result = fetch_annotation(**kwargs)
        print(json.dumps({str(k): v for k, v in result.items()}, indent=2, default=str))
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1
    return 0


def _annotations_to_rows(ann) -> list[tuple]:
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


def cmd_fetch_all(
    output_dir: Path | None,
    tags: str | None,
    space: str | None,
    fmt: str | None,
) -> int:
    if not _has_neuromaps():
        print("Install neuromaps: pip install neuromaps", file=sys.stderr)
        return 1
    from neuromaps.datasets import available_annotations, fetch_annotation
    import neuromaps.datasets.annotations as _ann
    # nilearn.fetch_single_file expects Path; neuromaps passes str(fn.parent). Wrap so Path is passed.
    _orig_fetch = _ann._fetch_file
    def _wrap_fetch(url, data_dir, **kw):
        return _orig_fetch(url, Path(data_dir) if not isinstance(data_dir, Path) else data_dir, **kw)
    _ann._fetch_file = _wrap_fetch
    filters = {}
    if tags:
        filters["tags"] = tags.split(",") if isinstance(tags, str) else tags
    if space:
        filters["space"] = space
    if fmt:
        filters["format"] = fmt
    ann = available_annotations(**filters)
    items = _annotations_to_rows(ann)
    if not items:
        print("No annotations match filters.", file=sys.stderr)
        return 1
    out = Path(output_dir) if output_dir else Path.home() / "neuromaps-data"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(items)} annotation(s) to {out} ...")
    ok = 0
    for i, row in enumerate(items):
        if len(row) < 3:
            continue
        src, desc, sp = row[0], row[1], row[2]
        den_res = row[3] if len(row) > 3 else None
        try:
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
    return 0


def cmd_describe() -> int:
    if not _has_neuromaps():
        print("Install neuromaps: pip install neuromaps", file=sys.stderr)
        return 1
    from neuromaps.datasets import describe_annotations
    try:
        df = describe_annotations()
        print(df.to_string() if hasattr(df, "to_string") else str(df))
    except Exception as e:
        print(f"Describe failed: {e}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="List/fetch neuromaps annotations (receptors, structure, etc.).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # list
    p_list = sub.add_parser("list", help="List available annotations")
    p_list.add_argument("--tags", type=str, default=None, help="Filter by tags (comma-separated, e.g. receptors)")
    p_list.add_argument("--space", type=str, default=None, help="Filter by space (e.g. MNI152, fsaverage)")
    p_list.add_argument("--format", type=str, default=None, help="Filter by format (e.g. volumetric)")
    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch annotation by source/desc")
    p_fetch.add_argument("--source", type=str, required=True, help="Annotation source")
    p_fetch.add_argument("--desc", type=str, default=None, help="Description/identifier")
    p_fetch.add_argument("--space", type=str, default=None)
    p_fetch.add_argument("--den", type=str, default=None)
    p_fetch.add_argument("--output-dir", type=Path, default=None)
    # fetch-all
    p_fetch_all = sub.add_parser("fetch-all", help="Fetch all annotations (or filtered subset) to a local dir")
    p_fetch_all.add_argument("--output-dir", type=Path, default=None, help="Where to save (default: ~/neuromaps-data)")
    p_fetch_all.add_argument("--tags", type=str, default=None, help="Filter by tags (comma-separated)")
    p_fetch_all.add_argument("--space", type=str, default=None, help="Filter by space (e.g. MNI152)")
    p_fetch_all.add_argument("--format", type=str, default=None, help="Filter by format (e.g. volumetric)")
    # describe
    sub.add_parser("describe", help="Describe available annotations (dataframe)")
    args = parser.parse_args()

    if args.cmd == "list":
        return cmd_list(getattr(args, "tags", None), getattr(args, "space", None), getattr(args, "format", None))
    if args.cmd == "fetch":
        return cmd_fetch(
            args.source,
            getattr(args, "desc", None),
            getattr(args, "space", None),
            getattr(args, "den", None),
            getattr(args, "output_dir", None),
        )
    if args.cmd == "fetch-all":
        return cmd_fetch_all(
            getattr(args, "output_dir", None),
            getattr(args, "tags", None),
            getattr(args, "space", None),
            getattr(args, "format", None),
        )
    if args.cmd == "describe":
        return cmd_describe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
