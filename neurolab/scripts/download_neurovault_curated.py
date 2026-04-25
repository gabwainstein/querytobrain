#!/usr/bin/env python3
"""
Download NeuroVault collections from the curated acquisition guide.

Implements the full collection list from:
  neurolab/docs/implementation/NeuroVault acquisition guide for brain activation prediction training.md

Tiers:
  Tier 1: Multi-domain compilations (HCP, BrainPedia, IBC, Cognitive Atlas decoding)
  Tier 2: Meta-analyses (consensus maps, highest information density)
  Tier 3: Domain-specific (WM, social cognition, emotion, pain, memory, reward, motor)
  Tier 4: Clinical, structural, connectivity, pharmacological, NARPS

Usage:
  python neurolab/scripts/download_neurovault_curated.py --tiers 1
  python neurolab/scripts/download_neurovault_curated.py --tiers 1 2 3
  python neurolab/scripts/download_neurovault_curated.py --all
  python neurolab/scripts/download_neurovault_curated.py --max-images 5000

Skip fetch when data present: If manifest.json and downloads/neurovault/ already exist
  (e.g. from a previous run), the script skips fetch_neurovault_ids and exits. Use
  --force-fetch to re-download or refresh.

Resume: If interrupted (restart, network failure), run the same command again.
  mode=download_new (default) skips already-downloaded images and continues.

Why fetch can take hours: nilearn's fetch_neurovault_ids downloads images sequentially (one
  HTTP request per image). With --all (126 collections) the raw image count can be large; expect
  several hours. After build_neurovault_cache --average-subject-level you get ~2–4K maps (curated
  training set). Use --max-images N for a quicker test. Increase --timeout if you see timeouts.

Recover: If manifest.json is missing (interrupted before completion), run:
  python neurolab/scripts/download_neurovault_curated.py --recover-manifest
  to build manifest from existing downloads/neurovault/. Then run --all to continue.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = _repo_root / "neurolab" / "data" / "neurovault_curated_data"

# From NeuroVault acquisition guide
TIER_1 = [
    457,    # HCP group-level (7 domains, ~1200 subjects)
    1274,   # Cognitive Atlas × NeuroSynth decoding (399 concepts)
    1952,   # BrainPedia (30 protocols)
    3324,   # Pain + CogControl + Emotion (270 maps)
    6618,   # IBC 2nd release (25 tasks, 205 contrasts)
    2138,   # IBC 1st release (12 tasks, 59 contrasts)
    4343,   # UCLA LA5C (healthy + clinical)
    20820,  # HCP-YA task-evoked network atlas
]

TIER_2 = [
    18197, 844, 833, 830, 825, 839, 1425, 1432, 1501, 2462,
    3884, 5070, 5377, 5943, 6262, 7793, 8448, 11343, 20036, 555,
    3822, 15965,
]

TIER_3 = [
    2884, 2621, 3085, 5623, 3192, 13042, 3158, 6009, 13656,
    426, 445, 507, 2503, 4804,
    503, 6221, 6237, 15274, 16284, 1541, 4146, 16266,
    504, 6126, 10410, 12874, 13924, 15030, 9244,
    6088, 5673, 2814,
    3340, 8676, 3960, 12480,
    63, 834, 11584, 315,
    13705, 2108, 4683, 3887, 1516,
]

TIER_4 = [
    # Clinical
    13474, 20510, 11646, 437, 12992, 19012, 6825, 1620,
    # Structural / atlases
    262, 264, 550, 3145, 1625, 2981, 6074, 7114, 5662, 9357, 8461,
    # Connectivity
    1057, 1598, 3434, 3245, 8076, 2485, 109,
    # Pharmacological
    1206, 15237, 17228,
    # NARPS
    6047, 6051,
]

WM_ATLAS = list(range(7756, 7762))

# Slug-based; resolved to numeric ID via API
SLUG_COLLECTIONS = ["EBAYVDBZ", "OCAMCQFK", "UOWUSAMV", "ZSVLTNSF"]

# Pharmacological: curated drug/placebo/task-on-drug collections (see neurovault_pharma_schema.json).
# Excludes 3264 (control task battery—not drug manipulation) and 2508 (cannabis meta—excluded from cache).
# Includes: drug challenge (1083 LSD, 12212 ketamine, 17403 ketamine touch, etc.), placebo/expectancy (9206, 9244, 20308),
# meta-analyses (3713 oxytocin ALE, 1501 addiction reward SDM, 19291 psychedelics MKDA), task-on-drug-null (1186 sulpiride gambling).
PHARMA = [
    1083, 12212, 17403, 8306, 3902, 4040, 4041, 9246, 5488, 3666, 3808, 13312, 13665,
    4414, 12992, 9206, 9244, 20308, 3713, 1501, 1186, 19291,
]

API_BASE = "https://neurovault.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "NeuroLab-Curated/1.0"}


def fetch_collection_metadata(cid: int, timeout: float = 15.0) -> dict:
    """Fetch collection name and description from NeuroVault API."""
    try:
        req = urllib.request.Request(f"{API_BASE}/collections/{cid}/", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def resolve_slug_to_id(slug: str, timeout: float = 15.0) -> int | None:
    """Resolve slug (e.g. EBAYVDBZ) to numeric collection ID."""
    try:
        url = f"{API_BASE}/collections/?id={slug}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("results") or []
        if results:
            return results[0].get("id")
        url = f"{API_BASE}/collections/{slug}/"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            c = json.loads(resp.read().decode())
        return c.get("id")
    except Exception:
        return None


def get_collections_for_tiers(tiers: list[int], include_pharma: bool, include_wm_atlas: bool, include_slugs: bool, timeout: float = 15.0) -> list[int]:
    all_ids = []
    tier_lists = {1: TIER_1, 2: TIER_2, 3: TIER_3, 4: TIER_4}
    for t in tiers:
        all_ids.extend(tier_lists.get(t, []))
    if include_wm_atlas:
        all_ids.extend(WM_ATLAS)
    if include_pharma:
        all_ids.extend(PHARMA)
    if include_slugs:
        for slug in SLUG_COLLECTIONS:
            cid = resolve_slug_to_id(slug, timeout=timeout)
            if cid:
                all_ids.append(cid)
            else:
                print(f"  Could not resolve slug {slug}; skipping", file=sys.stderr)
    return list(dict.fromkeys(all_ids))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download curated NeuroVault collections from acquisition guide"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Output dir (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[1],
        help="Tiers to download (1-4). Default: 1",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all tiers (1-4) + WM atlas + pharma + slugs",
    )
    parser.add_argument(
        "--include-pharma",
        action="store_true",
        help="Include pharmacological collections (also in download_neurovault_pharma)",
    )
    parser.add_argument(
        "--include-wm-atlas",
        action="store_true",
        help="Include WM function atlas (7756-7761)",
    )
    parser.add_argument(
        "--include-slugs",
        action="store_true",
        help="Include slug-based collections (EBAYVDBZ, OCAMCQFK, UOWUSAMV, ZSVLTNSF)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap total images (0 = no cap)",
    )
    parser.add_argument(
        "--mode",
        choices=("download_new", "overwrite", "offline"),
        default="download_new",
        help="Download mode",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--recover-manifest",
        action="store_true",
        help="If manifest.json missing, build it from existing downloads/neurovault/ (for interrupted downloads)",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Call fetch_neurovault_ids even when manifest.json and downloads already exist (default: skip fetch if data present)",
    )
    parser.add_argument(
        "--collections",
        type=int,
        nargs="+",
        default=None,
        help="Fetch only these collection IDs (overrides tiers; merges into existing manifest when present)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "downloads"
    nv_dir = data_dir / "neurovault"
    manifest_path = out / "manifest.json"

    # Recover manifest from existing downloads when interrupted before manifest was written
    if args.recover_manifest and not manifest_path.exists() and nv_dir.exists():
        images = []
        for cdir in sorted(nv_dir.iterdir()):
            if not cdir.is_dir() or not cdir.name.startswith("collection_"):
                continue
            try:
                cid = int(cdir.name.replace("collection_", ""))
            except ValueError:
                continue
            for f in cdir.glob("image_*.*"):
                stem = f.stem
                if f.suffix == ".gz" and stem.endswith(".nii"):
                    stem = Path(stem).stem
                if stem.startswith("image_"):
                    try:
                        iid = int(stem.replace("image_", "").split(".")[0])
                    except ValueError:
                        continue
                    images.append({
                        "path": str(f.resolve()),
                        "collection_id": cid,
                        "image_id": iid,
                        "name": f.name,
                        "contrast_definition": "",
                    })
        if images:
            collection_ids = sorted(set(e["collection_id"] for e in images))
            manifest = [{"index": i, "path": e["path"], "name": e["name"], "contrast_definition": e["contrast_definition"],
                        "collection_id": e["collection_id"], "image_id": e["image_id"]} for i, e in enumerate(images)]
            collections_meta = []
            for cid in collection_ids:
                meta = fetch_collection_metadata(cid)
                collections_meta.append({
                    "id": cid, "name": meta.get("name"), "description": (meta.get("description") or "")[:2000],
                    "number_of_images": meta.get("number_of_images"),
                })
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"collections": collection_ids, "collections_meta": collections_meta, "n_images": len(images), "images": manifest}, f, indent=2)
            print(f"Recovered manifest: {len(images)} images from {len(collection_ids)} collections -> {manifest_path}")
            print("Next: build_neurovault_cache.py --data-dir", out, "--output-dir neurolab/data/neurovault_cache")
            return 0
        else:
            print("No images found in downloads/neurovault/", file=sys.stderr)
            return 1
    elif args.recover_manifest and manifest_path.exists():
        print("Manifest already exists:", manifest_path)
        return 0
    elif args.recover_manifest:
        print("Cannot recover: manifest missing and downloads/neurovault/ not found. Run --all first.", file=sys.stderr)
        return 1

    tiers = [1, 2, 3, 4] if args.all else args.tiers
    include_pharma = args.all or args.include_pharma
    include_wm_atlas = args.all or args.include_wm_atlas
    include_slugs = args.all or args.include_slugs

    if args.collections:
        collection_ids = list(dict.fromkeys(args.collections))
        print(f"Collections to fetch (--collections): {len(collection_ids)}")
    else:
        collection_ids = get_collections_for_tiers(tiers, include_pharma, include_wm_atlas, include_slugs, args.timeout)
        print(f"Collections to fetch: {len(collection_ids)}")
    print(f"  Tiers: {tiers}")
    if include_pharma:
        print("  + pharmacological")
    if include_wm_atlas:
        print("  + WM atlas (7756-7761)")
    if include_slugs:
        print("  + slug-based")

    data_dir.mkdir(parents=True, exist_ok=True)
    nv_dir = data_dir / "neurovault"

    # If we already have manifest + downloads, skip fetch unless --force-fetch or --collections (avoid re-calling fetch_neurovault_ids)
    force_this_run = getattr(args, "force_fetch", False) or args.collections is not None
    if not force_this_run and manifest_path.exists() and nv_dir.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
            n_images = manifest_data.get("n_images", 0)
            images_list = manifest_data.get("images") or []
            n_exist = sum(1 for e in images_list if e.get("path") and Path(e["path"]).exists())
            if n_images > 0 and n_exist >= min(n_images, 10):
                print(f"NeuroVault curated data already present: manifest.json ({n_images} raw images on disk, {n_exist} verified). Skipping fetch.")
                print("  Curated training set: run build_neurovault_cache.py --average-subject-level to get ~2–4K maps.")
                print("  Next: build_neurovault_cache.py --data-dir", out, "--output-dir neurolab/data/neurovault_cache [--average-subject-level]")
                print("  To re-download or refresh: run with --force-fetch")
                return 0
        except (json.JSONDecodeError, KeyError):
            pass

    image_ids_to_fetch = ()
    if args.max_images > 0:
        image_ids_to_fetch = []
        for cid in collection_ids:
            if len(image_ids_to_fetch) >= args.max_images:
                break
            offset = 0
            while len(image_ids_to_fetch) < args.max_images:
                url = f"{API_BASE}/collections/{cid}/images/?limit=500&offset={offset}"
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=int(args.timeout)) as resp:
                    data = json.loads(resp.read().decode())
                results = data.get("results") or []
                for r in results:
                    if "id" in r:
                        image_ids_to_fetch.append(r["id"])
                        if len(image_ids_to_fetch) >= args.max_images:
                            break
                offset += len(results)
                if offset >= data.get("count", offset + 1) or not results:
                    break
        image_ids_to_fetch = tuple(image_ids_to_fetch[: args.max_images])
        print(f"  Capped to {len(image_ids_to_fetch)} image IDs")

    try:
        from nilearn.datasets import fetch_neurovault_ids
    except ImportError:
        print("Install nilearn: pip install nilearn", file=sys.stderr)
        return 1

    n_to_fetch = len(image_ids_to_fetch) if image_ids_to_fetch else None
    if n_to_fetch is None:
        n_to_fetch = len(collection_ids)
        if n_to_fetch > 10:
            print(f"fetch_neurovault_ids: {n_to_fetch} collections — downloads are sequential (no parallelism); can take hours. Curated training set after averaging: ~2–4K maps. Re-run same command to resume (mode=download_new skips existing).")
    elif n_to_fetch > 1000:
        print(f"fetch_neurovault_ids: {n_to_fetch} raw images — sequential download; can take hours. Re-run to resume (mode=download_new skips existing).")

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
                collection_ids=collection_ids,
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
        print("No images downloaded.", file=sys.stderr)
        return 1

    manifest = []
    for i, path in enumerate(images):
        meta = images_meta[i] if i < len(images_meta) else {}
        if not hasattr(meta, "items"):
            meta = {}
        desc = (meta.get("description") or meta.get("contrast_definition") or "")[:500]
        manifest.append({
            "index": i,
            "path": str(Path(path).resolve()) if path else None,
            "name": meta.get("name") or meta.get("full_name") or f"image_{i}",
            "contrast_definition": (meta.get("contrast_definition") or "")[:300],
            "description": desc,
            "collection_id": meta.get("collection_id"),
            "image_id": meta.get("id"),
            "map_type": meta.get("map_type"),
        })

    manifest_path = out / "manifest.json"
    # Merge into existing manifest when --collections was used and manifest exists
    if args.collections and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                existing = json.load(f)
            existing_images = existing.get("images") or []
            existing_paths = {e.get("path") for e in existing_images if e.get("path")}
            merged = list(existing_images)
            next_idx = len(merged)
            for m in manifest:
                p = m.get("path")
                if p and p not in existing_paths:
                    m = dict(m)
                    m["index"] = next_idx
                    next_idx += 1
                    merged.append(m)
                    existing_paths.add(p)
            all_collections = sorted(set(existing.get("collections") or []) | set(e.get("collection_id") for e in merged if e.get("collection_id")))
            collections_meta = existing.get("collections_meta") or []
            seen_cids = {c.get("id") for c in collections_meta if c.get("id") is not None}
            for cid in all_collections:
                if cid not in seen_cids:
                    meta = fetch_collection_metadata(cid)
                    collections_meta.append({"id": cid, "name": meta.get("name"), "description": (meta.get("description") or "")[:2000], "number_of_images": meta.get("number_of_images")})
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "collections": all_collections,
                        "collections_meta": collections_meta,
                        "tiers": existing.get("tiers", tiers),
                        "n_images": len(merged),
                        "images": merged,
                    },
                    f,
                    indent=2,
                )
            print(f"Downloaded {len(images)} images, merged into manifest ({len(merged)} total) -> {data_dir}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Merge failed ({e}), writing new manifest", file=sys.stderr)
            collections_meta = []
            for cid in collection_ids:
                m = fetch_collection_metadata(cid)
                collections_meta.append({"id": cid, "name": m.get("name"), "description": (m.get("description") or "")[:2000], "number_of_images": m.get("number_of_images")})
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"collections": collection_ids, "collections_meta": collections_meta, "tiers": tiers, "n_images": len(images), "images": manifest},
                    f, indent=2,
                )
            print(f"Downloaded {len(images)} images -> {data_dir}")
    else:
        collections_meta = []
        for cid in collection_ids:
            m = fetch_collection_metadata(cid)
            collections_meta.append({"id": cid, "name": m.get("name"), "description": (m.get("description") or "")[:2000], "number_of_images": m.get("number_of_images")})
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "collections": collection_ids,
                    "collections_meta": collections_meta,
                    "tiers": tiers,
                    "n_images": len(images),
                    "images": manifest,
                },
                f,
                indent=2,
            )
        print(f"Downloaded {len(images)} images -> {data_dir}")
    print(f"Manifest -> {manifest_path}")
    print("Next: build_neurovault_cache.py --data-dir", out, "--output-dir neurolab/data/neurovault_cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
