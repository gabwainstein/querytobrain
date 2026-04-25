#!/usr/bin/env python3
"""
Download NeuroVault collections that match pharmacological/drug-related search terms.
All NeuroVault data is open (CC0). Searches collection names/descriptions, then fetches
all images from matching collections. Use for drug challenge contrast maps.

Usage:
  python neurolab/scripts/download_neurovault_pharma.py --output-dir neurolab/data/neurovault_pharma_data
  python neurolab/scripts/download_neurovault_pharma.py --search drug ketamine pain  # extra keywords
  python neurolab/scripts/download_neurovault_pharma.py --max-collections 20  # cap number of collections

Requires: nilearn (for fetch_neurovault_ids).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = _repo_root / "neurolab" / "data" / "neurovault_pharma_data"

# Search terms to find pharmacological / drug-related collections on NeuroVault.
PHARMA_SEARCH_TERMS = [
    "drug",
    "pharmacolog",
    "ketamine",
    "LSD",
    "psilocybin",
    "nicotine",
    "caffeine",
    "pain",      # pain-related meta-analyses often include drug studies
    "opioid",
    "cannabis",
    "antidepressant",
    "antipsychotic",
]


def search_collections(keywords: list[str], max_per_keyword: int = 50, timeout: float = 15.0) -> list[dict]:
    """Query NeuroVault API for collections whose name contains any keyword. Returns list of collection dicts."""
    seen = set()
    results = []
    for kw in keywords:
        url = f"https://neurovault.org/api/collections/?name={urllib.parse.quote(kw)}&limit={max_per_keyword}"
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroLab-Pharma-Download/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Search '{kw}': {e}", file=sys.stderr)
            continue
        for c in data.get("results") or []:
            cid = c.get("id")
            if cid is None or cid in seen:
                continue
            n_img = c.get("number_of_images") or 0
            if n_img > 0:
                seen.add(cid)
                results.append(c)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NeuroVault pharmacological/drug-related collections.")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Output dir (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--search",
        nargs="*",
        default=None,
        help="Extra search terms (default: use built-in PHARMA_SEARCH_TERMS)",
    )
    parser.add_argument(
        "--max-collections",
        type=int,
        default=0,
        help="Max number of collections to fetch (0 = all found)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images total (0 = no cap)",
    )
    parser.add_argument(
        "--mode",
        choices=("download_new", "overwrite", "offline"),
        default="download_new",
        help="Download mode for nilearn",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    args = parser.parse_args()

    try:
        from nilearn.datasets import fetch_neurovault_ids
    except ImportError:
        print("Install nilearn: pip install nilearn", file=sys.stderr)
        return 1

    keywords = args.search if args.search is not None else PHARMA_SEARCH_TERMS
    print(f"Searching NeuroVault for: {keywords}")
    collections = search_collections(keywords)
    if not collections:
        print("No collections found. Try --search with different terms.", file=sys.stderr)
        return 1

    if args.max_collections > 0:
        collections = collections[: args.max_collections]
    cids = [c["id"] for c in collections]
    print(f"Found {len(cids)} collections with images: {cids[:10]}{'...' if len(cids) > 10 else ''}")

    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)

    image_ids_to_fetch = ()
    if args.max_images > 0:
        image_ids_to_fetch = []
        for cid in cids:
            offset = 0
            while len(image_ids_to_fetch) < args.max_images:
                req_url = f"https://neurovault.org/api/collections/{cid}/images/?limit=500&offset={offset}"
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
                offset += len(results)
                if offset >= data.get("count", offset + 1):
                    break
            if len(image_ids_to_fetch) >= args.max_images:
                break
        image_ids_to_fetch = tuple(image_ids_to_fetch[: args.max_images])
        print(f"  Capped to {len(image_ids_to_fetch)} image IDs")

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
                collection_ids=cids,
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
    if not images:
        print("No images downloaded. Check collection IDs and network.", file=sys.stderr)
        return 1

    manifest = []
    for i, path in enumerate(images):
        meta = images_meta[i] if i < len(images_meta) else {}
        if not hasattr(meta, "items"):
            meta = {}
        entry = {
            "index": i,
            "path": str(Path(path).resolve()) if path else None,
            "name": meta.get("name") or meta.get("full_name") or f"image_{i}",
            "contrast_definition": (meta.get("contrast_definition") or "")[:300],
            "collection_id": meta.get("collection_id"),
            "image_id": meta.get("id"),
        }
        manifest.append(entry)

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "collections": cids,
                "collections_meta": [
                    {"id": c["id"], "name": c.get("name"), "description": (c.get("description") or "")[:2000],
                     "number_of_images": c.get("number_of_images")}
                    for c in collections
                ],
                "n_images": len(images),
                "images": manifest,
            },
            f,
            indent=2,
        )
    print(f"Downloaded {len(images)} images -> {data_dir}")
    print(f"Manifest -> {manifest_path}")
    print("Next: run build_neurovault_cache.py with --data-dir to parcellate and build (label, map) cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
