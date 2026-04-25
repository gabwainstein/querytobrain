#!/usr/bin/env python3
"""
Download curated NeuroVault collections for task-contrast maps + descriptions.

Saves images and metadata to neurolab/data/neurovault_data by default. Each image
has a name/description from NeuroVault that can be used as a text label for
text-to-brain training (e.g. "fearful faces > neutral faces, N=24").

**Curated collections (default):**
- **1952** — BrainPedia/IBC: SPMs from OpenfMRI, HCP, Neurospin (~6,573 images). Task-contrast maps.
- **4337** — Human Connectome Project (HCP) “data acquisition perspective”: volumetric z-stat maps (~18,070 images). Large collection.

With both collections, the script fetches **all** images (no cap); total is ~24,600+ images. Default is BrainPedia (1952) only. Use **--collections 1952 4337** to fetch both. Use **--max-images N** (e.g. 500) to cap images for a quick test.

Requires: nilearn (already in requirements-enrichment.txt).

  python neurolab/scripts/download_neurovault_data.py
  python neurolab/scripts/download_neurovault_data.py --output-dir /path/to/neurovault_data
  python neurolab/scripts/download_neurovault_data.py --collections 1952 4337 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BrainPedia only (1952). Use --collections 1952 4337 to add HCP.
DEFAULT_COLLECTION_IDS = (1952,)
DEFAULT_OUTPUT_DIR = os.path.join(_repo_root, "neurolab", "data", "neurovault_data")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download curated NeuroVault collections (IBC, NARPS) for task-contrast maps"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save images and manifest (default: neurolab/data/neurovault_data)",
    )
    parser.add_argument(
        "--collections",
        type=int,
        nargs="+",
        default=list(DEFAULT_COLLECTION_IDS),
        help=f"Collection IDs to fetch (default: {DEFAULT_COLLECTION_IDS})",
    )
    parser.add_argument(
        "--mode",
        choices=("download_new", "overwrite", "offline"),
        default="download_new",
        help="download_new = skip existing; overwrite = re-download; offline = disk only",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default 30)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, cap total images to this many (fetches image IDs from API first, then downloads). Use e.g. 500 for a quick test.",
    )
    args = parser.parse_args()

    try:
        from nilearn.datasets import fetch_neurovault_ids
    except ImportError:
        print("Install nilearn: pip install nilearn", file=sys.stderr)
        return 1

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = Path(_repo_root) / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching NeuroVault collections: {args.collections}")
    print(f"  Output: {out}")
    print(f"  Mode: {args.mode}")
    if args.max_images > 0:
        print(f"  Cap: {args.max_images} images (quick test)")

    image_ids_to_fetch = ()
    if args.max_images > 0:
        try:
            import urllib.request
            image_ids_to_fetch = []
            for cid in args.collections:
                if len(image_ids_to_fetch) >= args.max_images:
                    break
                offset = 0
                while len(image_ids_to_fetch) < args.max_images:
                    need = args.max_images - len(image_ids_to_fetch)
                    limit = min(500, need)
                    req_url = f"https://neurovault.org/api/collections/{cid}/images/?limit={limit}&offset={offset}"
                    with urllib.request.urlopen(req_url, timeout=int(args.timeout)) as resp:
                        data = json.loads(resp.read().decode())
                    results = data.get("results") or []
                    if not results:
                        break
                    for r in results:
                        if "id" in r:
                            image_ids_to_fetch.append(r["id"])
                            if len(image_ids_to_fetch) >= args.max_images:
                                break
                    if len(image_ids_to_fetch) >= args.max_images:
                        break
                    offset += len(results)
                    if offset >= data.get("count", offset + 1):
                        break
            image_ids_to_fetch = tuple(image_ids_to_fetch[: args.max_images])
            print(f"  Selected {len(image_ids_to_fetch)} image IDs from API")
        except Exception as e:
            print(f"Could not get image IDs from API: {e}; falling back to full collection fetch.", file=sys.stderr)
            image_ids_to_fetch = ()

    try:
        if image_ids_to_fetch:
            bunch = fetch_neurovault_ids(
                collection_ids=(),
                image_ids=image_ids_to_fetch,
                mode=args.mode,
                data_dir=str(data_dir.resolve().as_posix()),
                fetch_neurosynth_words=False,
                resample=False,
                timeout=args.timeout,
                verbose=2,
            )
        else:
            bunch = fetch_neurovault_ids(
                collection_ids=args.collections,
                image_ids=(),
                mode=args.mode,
                data_dir=str(data_dir.resolve().as_posix()),
                fetch_neurosynth_words=False,
                resample=False,
                timeout=args.timeout,
                verbose=2,
            )
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        return 1

    images = getattr(bunch, "images", [])
    images_meta = getattr(bunch, "images_meta", [])
    collections_meta = getattr(bunch, "collections_meta", [])

    if not images:
        print("No images returned. Check collection IDs and network.", file=sys.stderr)
        return 1

    # Build manifest: one entry per image (path, label-friendly name, collection_id, contrast_definition, etc.)
    manifest = []
    for i, path in enumerate(images):
        meta = images_meta[i] if i < len(images_meta) else {}
        if hasattr(meta, "items"):
            pass
        else:
            meta = {}
        name = meta.get("name") or meta.get("full_name") or f"image_{i}"
        desc = meta.get("description") or meta.get("contrast_definition") or ""
        if isinstance(desc, str):
            desc = desc[:500]
        else:
            desc = ""
        entry = {
            "index": i,
            "path": str(Path(path).resolve()) if path else None,
            "name": name,
            "contrast_definition": (meta.get("contrast_definition") or "")[:300],
            "task": meta.get("task"),
            "collection_id": meta.get("collection_id"),
            "image_id": meta.get("id"),
            "map_type": meta.get("map_type"),
            "description": desc,
        }
        manifest.append(entry)

    def _json_safe(obj):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        return str(obj)

    # Ensure collection description is present (for dose/drug-aware term enrichment)
    collections_meta = list(collections_meta or [])
    for c in collections_meta:
        if not (c.get("description") or "").strip():
            cid = c.get("id")
            if cid is not None:
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        f"https://neurovault.org/api/collections/{cid}/",
                        headers={"Accept": "application/json", "User-Agent": "NeuroLab/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        full = json.loads(resp.read().decode())
                    c["description"] = (full.get("description") or "")[:2000]
                except Exception:
                    c["description"] = ""

    collections_meta_safe = [_json_safe(c) for c in collections_meta]
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "collections": args.collections,
                "n_images": len(images),
                "images": manifest,
                "collections_meta": collections_meta_safe,
            },
            f,
            indent=2,
        )
    print(f"Downloaded {len(images)} images -> {data_dir}")
    print(f"Manifest -> {manifest_path}")
    print("Next: run build_neurovault_cache.py to parcellate to Glasser+Tian (392) and produce (label, map) cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
