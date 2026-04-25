#!/usr/bin/env python3
"""
Pre-fetch contrast_definition for all NeuroVault manifest entries that are missing it.
Updates manifest in place (or --output to a new file). After this, build_neurovault_cache
will not need to fetch per-image in the loop, so cache build runs much faster.

  python neurolab/scripts/enrich_neurovault_manifest.py --data-dir neurolab/data/neurovault_data
  python neurolab/scripts/enrich_neurovault_manifest.py --data-dir neurolab/data/neurovault_data --output manifest_enriched.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

NEUROVAULT_IMAGE_API = "https://neurovault.org/api/images/{id}/"


# NeuroVault returns 403 for default Python User-Agent; use a browser-like one
REQUEST_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (compatible; NeuroVaultCache/1.0)"}


def fetch_contrast_definition(image_id: int, timeout: float = 15.0) -> str:
    try:
        req = urllib.request.Request(
            NEUROVAULT_IMAGE_API.format(id=image_id),
            headers=REQUEST_HEADERS,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, dict) and data.get("contrast_definition"):
            return (data["contrast_definition"] or "").strip()
    except Exception:
        pass
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch contrast_definition for manifest entries missing it")
    parser.add_argument("--data-dir", default="neurolab/data/neurovault_data", help="Dir containing manifest.json")
    parser.add_argument("--output", default=None, help="Output manifest path (default: overwrite manifest.json)")
    parser.add_argument("--interval", type=int, default=200, help="Print progress every N images")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save manifest every N images (0 = only at end)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir) if os.path.isabs(args.data_dir) else repo_root / args.data_dir
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images") or []
    n_total = len(images)
    need_fetch = [
        i for i, e in enumerate(images)
        if not (e.get("contrast_definition") or "").strip() and e.get("image_id")
    ]
    print(f"Manifest: {n_total} images, {len(need_fetch)} need contrast_definition from API.", flush=True)
    if not need_fetch:
        print("Nothing to do.", flush=True)
        return 0

    for k, i in enumerate(need_fetch):
        entry = images[i]
        iid = entry.get("image_id")
        if not iid:
            continue
        definition = fetch_contrast_definition(iid)
        if definition:
            entry["contrast_definition"] = definition
        time.sleep(0.2)
        if (k + 1) % args.interval == 0 or k == 0:
            print(f"  Fetched {k + 1}/{len(need_fetch)} (image index {i})", flush=True)
        if args.save_interval > 0 and (k + 1) % args.save_interval == 0:
            out_path = Path(args.output) if args.output else manifest_path
            if not out_path.is_absolute():
                out_path = repo_root / out_path
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved checkpoint -> {out_path}", flush=True)

    out_path = Path(args.output) if args.output else manifest_path
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Done. Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
