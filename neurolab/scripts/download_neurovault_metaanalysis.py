#!/usr/bin/env python3
"""
Download NeuroVault images suitable for meta-analysis: quality-filtered group-level T/Z maps
from published collections, plus targeted pharmacological collections.

**Strategy:**
1. **Meta-analysis quality pull**: Query /api/images/ with filters for group-level T/Z maps,
   unthresholded, MNI space, from collections with DOI (published). Optionally filter by
   brain_coverage, Cognitive Atlas annotations.
2. **Targeted pharmacological collections**: Always add by collection ID (LSD, ketamine,
   amphetamine, oxytocin, cannabis, etc.) — these may not appear in the generic quality pull.

**Output:** Same structure as download_neurovault_data.py (manifest.json + downloads/neurovault/)
for use with build_neurovault_cache.py.

Usage:
  python neurolab/scripts/download_neurovault_metaanalysis.py --output-dir neurolab/data/neurovault_metaanalysis_data
  python neurolab/scripts/download_neurovault_metaanalysis.py --meta-only --max-images 5000   # quality pull only, capped
  python neurolab/scripts/download_neurovault_metaanalysis.py --pharma-only   # pharmacological collections only
  python neurolab/scripts/download_neurovault_metaanalysis.py --skip-pharma   # skip pharmacological (meta pull only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = _repo_root / "neurolab" / "data" / "neurovault_metaanalysis_data"

# Pharmacological collections (drug challenge, PET, meta-analyses)
# From inventory: LSD CBF, ketamine, amphetamine PET, ibuprofen, methylphenidate, L-DOPA, oxytocin, sulpiride, haloperidol, addiction, cannabis, etc.
PHARMA_COLLECTION_IDS = [
    1083,   # LSD CBF + RSFC
    12212,  # ketamine thalamic dysconnectivity
    4040, 4041,  # d-amphetamine PET
    9246,   # ibuprofen emotion
    8306,   # methylphenidate + sulpiride FDOPA
    13665,  # L-DOPA + oxytocin
    1186,   # sulpiride gambling
    3902,   # haloperidol
    5488,   # oxytocin amygdala FC
    3713,   # oxytocin meta-analysis
    1501,   # addiction meta-analysis (nicotine/cannabis/alcohol/cocaine)
    2508,   # cannabis ALE
    3264,   # pharmacological fMRI control task
]

API_BASE = "https://neurovault.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "NeuroLab-MetaAnalysis/1.0"}


def _get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_collections_with_doi(limit: int = 20000, timeout: float = 30.0) -> set[int]:
    """Fetch collection IDs that have a DOI (published). Used for client-side filtering when API filter fails."""
    ids = set()
    offset = 0
    while offset < limit:
        url = f"{API_BASE}/collections/?DOI__isnull=false&limit=500&offset={offset}"
        try:
            data = _get(url, timeout=timeout)
        except Exception:
            break
        for c in data.get("results") or []:
            cid = c.get("id")
            if cid is not None and c.get("DOI"):
                ids.add(cid)
        offset += 500
        if offset >= data.get("count", 0) or not data.get("results"):
            break
        time.sleep(0.1)
    return ids


def fetch_meta_quality_image_ids(
    *,
    map_types: tuple[str, ...] = ("T map", "Z map"),
    is_thresholded: bool = False,
    not_mni: bool = False,
    min_brain_coverage: float = 40.0,
    collection_has_doi: bool = True,
    max_images: int = 0,
    limit_per_request: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Fetch image metadata matching meta-analysis quality criteria.
    Strategy: get collections with DOI first, then fetch images from those collections
    (avoids paginating through 650k+ images). Filter by map_type, is_thresholded, brain_coverage.
    """
    if collection_has_doi:
        collection_ids = list(fetch_collections_with_doi(timeout=timeout))
        print(f"  Collections with DOI: {len(collection_ids)}")
    else:
        # No DOI filter: paginate collections with images (cap to avoid huge runtime)
        collection_ids = []
        for offset in range(0, 10000, 500):
            url = f"{API_BASE}/collections/?limit=500&offset={offset}"
            try:
                data = _get(url, timeout=timeout)
            except Exception:
                break
            for c in data.get("results") or []:
                if c.get("id") and (c.get("number_of_images") or 0) > 0:
                    collection_ids.append(c["id"])
            if offset + 500 >= data.get("count", 0) or not data.get("results"):
                break
            time.sleep(0.05)
        print(f"  Collections with images: {len(collection_ids)}")

    images = []
    for i, cid in enumerate(collection_ids):
        if max_images > 0 and len(images) >= max_images:
            break
        offset = 0
        while True:
            url = f"{API_BASE}/collections/{cid}/images/?limit=500&offset={offset}"
            try:
                data = _get(url, timeout=timeout)
            except Exception:
                break
            results = data.get("results") or []
            for r in results:
                if not is_thresholded and r.get("is_thresholded", False):
                    continue
                if not not_mni and r.get("not_mni", False):
                    continue
                mt = r.get("map_type") or ""
                if map_types and mt not in map_types:
                    continue
                cov = r.get("brain_coverage")
                if cov is not None and min_brain_coverage > 0 and cov < min_brain_coverage:
                    continue
                images.append(r)
                if max_images > 0 and len(images) >= max_images:
                    break
            if max_images > 0 and len(images) >= max_images:
                break
            offset += len(results)
            if offset >= data.get("count", offset + 1) or not results:
                break
            time.sleep(0.05)
        if (i + 1) % 100 == 0:
            print(f"  Meta-quality: {len(images)} images from {i + 1} collections", flush=True)
        time.sleep(0.05)
    return images[:max_images] if max_images > 0 else images


def fetch_collection_image_ids(collection_ids: list[int], max_images: int = 0, timeout: float = 30.0) -> list[dict]:
    """Fetch image metadata for specific collections."""
    images = []
    for cid in collection_ids:
        offset = 0
        while True:
            url = f"{API_BASE}/collections/{cid}/images/?limit=500&offset={offset}"
            try:
                data = _get(url, timeout=timeout)
            except Exception as e:
                print(f"  Collection {cid} error: {e}", file=sys.stderr)
                break
            results = data.get("results") or []
            for r in results:
                images.append(r)
                if max_images > 0 and len(images) >= max_images:
                    break
            if max_images > 0 and len(images) >= max_images:
                break
            offset += len(results)
            if offset >= data.get("count", offset + 1) or not results:
                break
            time.sleep(0.1)
        if max_images > 0 and len(images) >= max_images:
            break
    return images[:max_images] if max_images > 0 else images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NeuroVault meta-analysis-quality + pharmacological images"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Output dir (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Only run meta-analysis quality pull (skip pharmacological collections)",
    )
    parser.add_argument(
        "--pharma-only",
        action="store_true",
        help="Only download pharmacological collections (skip meta-quality pull)",
    )
    parser.add_argument(
        "--skip-pharma",
        action="store_true",
        help="Skip pharmacological collections (same as --meta-only)",
    )
    parser.add_argument(
        "--pharma-collections",
        type=int,
        nargs="+",
        default=None,
        help=f"Override pharmacological collection IDs (default: {PHARMA_COLLECTION_IDS[:5]}...)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap total images (0 = no cap). Applies to meta pull and pharma combined.",
    )
    parser.add_argument(
        "--max-meta-images",
        type=int,
        default=0,
        help="Cap meta-quality images only (0 = no cap)",
    )
    parser.add_argument(
        "--min-brain-coverage",
        type=float,
        default=40.0,
        help="Min brain_coverage %% for meta-quality (default 40)",
    )
    parser.add_argument(
        "--no-collection-doi",
        action="store_true",
        help="Include images from collections without DOI (less quality gate)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument(
        "--mode",
        choices=("download_new", "overwrite", "offline"),
        default="download_new",
        help="Download mode for nilearn fetch_neurovault_ids",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_images_meta: list[dict] = []
    source_labels: list[str] = []

    # 1. Meta-analysis quality pull
    if not args.pharma_only:
        print("Fetching meta-analysis-quality image IDs (T/Z maps, unthresholded, MNI, collection has DOI)...")
        meta_images = fetch_meta_quality_image_ids(
            is_thresholded=False,
            not_mni=False,
            min_brain_coverage=args.min_brain_coverage,
            collection_has_doi=not args.no_collection_doi,
            max_images=args.max_meta_images or (args.max_images if args.meta_only else 0),
            timeout=args.timeout,
        )
        print(f"  Found {len(meta_images)} meta-quality images")
        # Dedupe by id
        seen_ids = set()
        for r in meta_images:
            iid = r.get("id")
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
                all_images_meta.append(r)
                source_labels.append("meta")
        if args.meta_only and args.max_images > 0 and len(all_images_meta) > args.max_images:
            all_images_meta = all_images_meta[: args.max_images]
            source_labels = source_labels[: args.max_images]

    # 2. Pharmacological collections
    if not (args.meta_only or args.skip_pharma):
        pharma_ids = args.pharma_collections if args.pharma_collections is not None else PHARMA_COLLECTION_IDS
        print(f"Fetching pharmacological collections: {pharma_ids[:8]}{'...' if len(pharma_ids) > 8 else ''}")
        pharma_images = fetch_collection_image_ids(
            pharma_ids,
            max_images=args.max_images if args.pharma_only else 0,
            timeout=args.timeout,
        )
        print(f"  Found {len(pharma_images)} pharma images")
        seen = {r.get("id") for r in all_images_meta}
        remaining = args.max_images - len(all_images_meta) if args.max_images > 0 else 0
        for r in pharma_images:
            iid = r.get("id")
            if iid and iid not in seen:
                if args.max_images > 0 and remaining <= 0:
                    break
                seen.add(iid)
                all_images_meta.append(r)
                source_labels.append("pharma")
                if args.max_images > 0:
                    remaining -= 1

    if not all_images_meta:
        print("No images to download.", file=sys.stderr)
        return 1

    image_ids = [r["id"] for r in all_images_meta if r.get("id")]
    print(f"Total unique images to download: {len(image_ids)}")

    try:
        from nilearn.datasets import fetch_neurovault_ids
    except ImportError:
        print("Install nilearn: pip install nilearn", file=sys.stderr)
        return 1

    try:
        bunch = fetch_neurovault_ids(
            collection_ids=(),
            image_ids=image_ids,
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
    images_meta_fetched = getattr(bunch, "images_meta", [])
    if not images:
        print("No images downloaded.", file=sys.stderr)
        return 1

    # Build manifest
    id_to_source = {r["id"]: source_labels[j] for j, r in enumerate(all_images_meta) if r.get("id")}
    manifest = []
    for i, path in enumerate(images):
        meta = images_meta_fetched[i] if i < len(images_meta_fetched) else {}
        if not hasattr(meta, "items"):
            meta = {}
        iid = meta.get("id")
        label = id_to_source.get(iid, "unknown")
        entry = {
            "index": i,
            "path": str(Path(path).resolve()) if path else None,
            "name": meta.get("name") or meta.get("full_name") or f"image_{i}",
            "contrast_definition": (meta.get("contrast_definition") or "")[:300],
            "task": meta.get("task"),
            "collection_id": meta.get("collection_id"),
            "image_id": meta.get("id"),
            "map_type": meta.get("map_type"),
            "source": label,
        }
        manifest.append(entry)

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_images": len(images),
                "sources": {"meta": sum(1 for s in source_labels if s == "meta"), "pharma": sum(1 for s in source_labels if s == "pharma")},
                "images": manifest,
            },
            f,
            indent=2,
        )
    print(f"Downloaded {len(images)} images -> {data_dir}")
    print(f"Manifest -> {manifest_path}")
    print("Next: run build_neurovault_cache.py --data-dir", out, "--output-dir neurolab/data/neurovault_cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
