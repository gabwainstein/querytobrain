#!/usr/bin/env python3
"""
Diagnose why a NeuroVault collection produces 0 output terms.

Run parcellation on one collection's images and report what fails (load, resample, mask, QC).

Usage:
  python neurolab/scripts/diagnose_parcellation_failure.py --collection 555
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "neurolab" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose parcellation failure for a NeuroVault collection")
    ap.add_argument("--collection", "-c", type=int, default=555, help="Collection ID (default: 555)")
    ap.add_argument("--data-dir", default=str(BASE / "neurovault_curated_data"), help="Curated data dir")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    downloads_dir = data_dir / "downloads" / "neurovault"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = [e for e in manifest.get("images", []) if e.get("collection_id") == args.collection]
    if not images:
        print(f"No images for collection {args.collection} in manifest", file=sys.stderr)
        return 1

    print(f"Collection {args.collection}: {len(images)} images")
    print()

    try:
        import nibabel as nib
        from neurolab.parcellation import get_masker, get_n_parcels, resample_to_atlas
        from neurolab.neurovault_ingestion import qc_filter
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    n_parcels = get_n_parcels()
    masker = get_masker(memory="nilearn_cache", verbose=0)
    masker.fit()

    load_ok, load_fail = 0, 0
    resample_ok, resample_fail = 0, 0
    parcel_ok, parcel_fail = 0, 0
    qc_ok, qc_fail = 0, 0
    errors = []

    for i, entry in enumerate(images[:10]):  # First 10 only
        cid = entry.get("collection_id")
        iid = entry.get("image_id")
        path = entry.get("path")
        if not path or not Path(path).is_file():
            cand = downloads_dir / f"collection_{cid}" / f"image_{iid}.nii.gz"
            if cand.exists():
                path = str(cand)
            else:
                for ext in (".nii.gz", "_resampled.nii.gz", ".nii"):
                    c = downloads_dir / f"collection_{cid}" / f"image_{iid}{ext}"
                    if c.exists():
                        path = str(c)
                        break
        if not path or not Path(path).is_file():
            load_fail += 1
            errors.append((i, "path", "File not found"))
            continue
        load_ok += 1

        try:
            img = nib.load(path)
        except Exception as e:
            load_fail += 1
            load_ok -= 1
            errors.append((i, "load", str(e)))
            continue

        try:
            img_rs = resample_to_atlas(img)
        except Exception as e:
            resample_fail += 1
            errors.append((i, "resample", str(e)))
            if len([x for x in errors if x[1] == "resample"]) <= 2:
                print(f"  Resample fail image {i} ({path}): {e}")
            continue
        resample_ok += 1

        try:
            vec = masker.transform(img_rs).ravel().astype("float64")
        except Exception as e:
            parcel_fail += 1
            errors.append((i, "parcellate", str(e)))
            continue
        parcel_ok += 1

        if vec.shape[0] != n_parcels:
            parcel_fail += 1
            parcel_ok -= 1
            errors.append((i, "parcellate", f"shape {vec.shape[0]} != {n_parcels}"))
            continue

        if not np.isfinite(vec).all():
            parcel_fail += 1
            parcel_ok -= 1
            errors.append((i, "parcellate", "Non-finite values"))
            continue

        arr = np.asarray(vec).reshape(1, -1)
        keep = qc_filter(arr, n_parcels)
        if not keep[0]:
            qc_fail += 1
            v = vec
            zeros = (np.abs(v) < 1e-10).sum() / len(v)
            nans = np.isnan(v).sum() / len(v)
            reason = []
            if np.sum(np.abs(v)) < 1e-10:
                reason.append("all-zero")
            if zeros > 0.95:
                reason.append(f"zeros={zeros:.2f}")
            if nans > 0.1:
                reason.append(f"nans={nans:.2f}")
            if np.nanmax(np.abs(v)) > 50:
                reason.append(f"max={np.nanmax(np.abs(v)):.1f}")
            if np.nanstd(v) < 0.01:
                reason.append(f"std={np.nanstd(v):.2e}")
            errors.append((i, "qc", ", ".join(reason) or "unknown"))
        else:
            qc_ok += 1

    print("Results (first 10 images):")
    print(f"  Load:      {load_ok} ok, {load_fail} fail")
    print(f"  Resample:  {resample_ok} ok, {resample_fail} fail")
    print(f"  Parcellate:{parcel_ok} ok, {parcel_fail} fail")
    print(f"  QC:        {qc_ok} ok, {qc_fail} fail")
    if errors:
        print("\nSample errors:")
        for idx, stage, msg in errors[:5]:
            print(f"  [{idx}] {stage}: {msg[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
