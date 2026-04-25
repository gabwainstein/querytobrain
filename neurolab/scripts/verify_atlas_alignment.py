#!/usr/bin/env python3
"""
Verify that all brain images align to the pipeline atlas (Glasser+Tian, 392).

Checks:
  - Atlas affine, shape, origin
  - Source spaces: NeuroQuery, NeuroSynth (NiMARE), NeuroVault, neuromaps
  - Affine differences (origin, voxel size, direction)
  - Recommendations for resampling if mismatches found

Atlas sensitivity: origin, direction, sym vs asym (MNI152NLin2009cAsym vs Sym),
3T vs 7T templates can differ. NiftiLabelsMasker resamples the atlas to each
image at transform time—correct if image affine is accurate, but misalignment
if metadata is wrong or template differs.

Usage:
  python neurolab/scripts/verify_atlas_alignment.py
  python neurolab/scripts/verify_atlas_alignment.py --sample-neurovault 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent.parent
_data = _repo_root / "neurolab" / "data"


def _describe_affine(affine: np.ndarray, name: str = "") -> dict:
    """Extract origin, voxel size, and direction from affine."""
    origin = affine[:3, 3]
    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    # Check if RAS (positive diagonal) or flipped
    diag = np.diag(affine[:3, :3])
    return {
        "name": name,
        "origin_mm": origin.tolist(),
        "voxel_size_mm": voxel_size.tolist(),
        "diag_signs": np.sign(diag).astype(int).tolist(),
        "shape": None,  # filled by caller if available
    }


def _affine_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two affines; return max abs diff in origin and voxel size."""
    oa, ob = a[:3, 3], b[:3, 3]
    vs_a = np.sqrt(np.sum(a[:3, :3] ** 2, axis=0))
    vs_b = np.sqrt(np.sum(b[:3, :3] ** 2, axis=0))
    return {
        "origin_diff_mm": np.abs(oa - ob).tolist(),
        "origin_max_diff_mm": float(np.max(np.abs(oa - ob))),
        "voxel_size_diff_mm": np.abs(vs_a - vs_b).tolist(),
        "voxel_max_diff_mm": float(np.max(np.abs(vs_a - vs_b))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify atlas alignment across data sources")
    ap.add_argument("--sample-neurovault", type=int, default=5, help="Sample N NeuroVault images (0=skip)")
    ap.add_argument("--verbose", action="store_true", help="Print full affines")
    args = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        print("Install nibabel: pip install nibabel", file=sys.stderr)
        return 1

    try:
        from neurolab.parcellation import get_combined_atlas_path
        atlas_path = get_combined_atlas_path(_data)
    except ImportError:
        for name in ("combined_atlas_427.nii.gz", "combined_atlas_450.nii.gz", "combined_atlas_392.nii.gz"):
            p = _data / name
            if p.exists():
                atlas_path = p
                break
        else:
            atlas_path = _data / "combined_atlas_427.nii.gz"
    if not atlas_path.exists():
        print(f"Atlas not found: {atlas_path}", file=sys.stderr)
        print("Run: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem", file=sys.stderr)
        return 1

    atlas_img = nib.load(str(atlas_path))
    atlas_affine = atlas_img.affine
    atlas_shape = atlas_img.shape

    print("=" * 70)
    print("  Atlas alignment verification")
    print("=" * 70)
    print(f"\nPipeline atlas: {atlas_path.name}")
    print(f"  Shape: {atlas_shape}")
    print(f"  Origin (mm): [{atlas_affine[0,3]:.2f}, {atlas_affine[1,3]:.2f}, {atlas_affine[2,3]:.2f}]")
    vs = np.sqrt(np.sum(atlas_affine[:3, :3] ** 2, axis=0))
    print(f"  Voxel size (mm): [{vs[0]:.2f}, {vs[1]:.2f}, {vs[2]:.2f}]")
    if args.verbose:
        print("  Affine:\n", atlas_affine)

    ref = _describe_affine(atlas_affine, "atlas")
    ref["shape"] = list(atlas_shape)
    sources = []

    # 1. Schaefer 2mm (reference used to build atlas)
    try:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        schaefer = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
        schaefer_img = nib.load(schaefer["maps"])
        s = _describe_affine(schaefer_img.affine, "Schaefer 2mm (atlas ref)")
        s["shape"] = list(schaefer_img.shape)
        diff = _affine_diff(atlas_affine, schaefer_img.affine)
        s["vs_atlas"] = diff
        sources.append(s)
        print(f"\n[OK] Schaefer 2mm (atlas reference)")
        print(f"  Origin diff vs atlas: max {diff['origin_max_diff_mm']:.3f} mm")
        if diff["origin_max_diff_mm"] > 0.1:
            print(f"  WARNING: Origin mismatch > 0.1 mm")
    except Exception as e:
        print(f"\n[SKIP] Schaefer: {e}")

    # 2. NeuroQuery (one sample)
    try:
        from neuroquery import fetch_neuroquery_model, NeuroQueryModel
        model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())
        result = model.transform(["working memory"])
        bm = result["brain_map"][0]
        if hasattr(bm, "affine"):
            nq_affine = bm.affine
        else:
            nq_img = nib.load(bm) if isinstance(bm, (str, Path)) else nib.Nifti1Image(np.asarray(bm), np.eye(4))
            nq_affine = nq_img.affine
        s = _describe_affine(nq_affine, "NeuroQuery")
        diff = _affine_diff(atlas_affine, nq_affine)
        s["vs_atlas"] = diff
        sources.append(s)
        print(f"\n[OK] NeuroQuery (sample: 'working memory')")
        print(f"  Origin diff vs atlas: max {diff['origin_max_diff_mm']:.3f} mm")
        print(f"  Voxel size diff: max {diff['voxel_max_diff_mm']:.3f} mm")
        if diff["origin_max_diff_mm"] > 2.0:
            print(f"  NOTE: Different affine convention (e.g. corner vs center); masker resamples at transform time.")
    except Exception as e:
        print(f"\n[SKIP] NeuroQuery: {e}")

    # 3. NeuroSynth / NiMARE (one sample)
    try:
        from nimare.extract import fetch_neurosynth
        from nimare.io import convert_neurosynth_to_dataset
        from nimare.meta.cbma.mkda import MKDADensity
        files = fetch_neurosynth(data_dir=str(_data / "neurosynth_data"), version="7", overwrite=False, source="abstract", vocab="terms")
        dset = convert_neurosynth_to_dataset(
            coordinates_file=files[0]["coordinates"],
            metadata_file=files[0]["metadata"],
            annotations_files=files[0]["features"],
        )
        # Get first term with enough studies
        for col in dset.annotations.columns:
            if col in ("id", "study_id", "contrast_id", "pmid", "doi") or col.startswith("_"):
                continue
            ids = dset.annotations[dset.annotations[col] > 0.001]["id"].tolist()
            if len(ids) >= 5:
                sub = dset.slice(ids)
                est = MKDADensity(kernel__r=6, null_method="approximate")
                res = est.fit(sub)
                img = res.get_map("z", return_type="image")
                if img is not None:
                    ns_affine = img.affine
                    s = _describe_affine(ns_affine, f"NeuroSynth/NiMARE ({col})")
                    diff = _affine_diff(atlas_affine, ns_affine)
                    s["vs_atlas"] = diff
                    sources.append(s)
                    print(f"\n[OK] NeuroSynth/NiMARE (sample: {col})")
                    print(f"  Origin diff vs atlas: max {diff['origin_max_diff_mm']:.3f} mm")
                    if diff["origin_max_diff_mm"] > 2.0:
                        print(f"  NOTE: Different affine; masker resamples at transform time.")
                break
    except Exception as e:
        print(f"\n[SKIP] NeuroSynth: {e}")

    # 4. NeuroVault (sample)
    nv_manifest = _data / "neurovault_data" / "manifest.json"
    nv_curated = _data / "neurovault_curated_data" / "manifest.json"
    manifest_path = nv_curated if nv_curated.exists() else (nv_manifest if nv_manifest.exists() else None)
    if manifest_path and args.sample_neurovault > 0:
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            images = data.get("images") or []
            n_sample = min(args.sample_neurovault, len(images))
            origins = []
            for i, ent in enumerate(images[: n_sample * 3]):  # oversample in case paths missing
                if len(origins) >= n_sample:
                    break
                path = ent.get("path")
                if not path or not Path(path).exists():
                    cand = _data / "neurovault_data" / "downloads" / "neurovault" / f"collection_{ent.get('collection_id')}" / f"image_{ent.get('image_id')}.nii.gz"
                    if not cand.exists():
                        cand = _data / "neurovault_curated_data" / "downloads" / "neurovault" / f"collection_{ent.get('collection_id')}" / f"image_{ent.get('image_id')}.nii.gz"
                    path = str(cand) if cand.exists() else None
                if path:
                    try:
                        img = nib.load(path)
                        origins.append(_describe_affine(img.affine, f"NeuroVault #{ent.get('image_id', i)}"))
                        origins[-1]["vs_atlas"] = _affine_diff(atlas_affine, img.affine)
                    except Exception:
                        pass
            if origins:
                max_orig = max(o["vs_atlas"]["origin_max_diff_mm"] for o in origins)
                max_vs = max(o["vs_atlas"]["voxel_max_diff_mm"] for o in origins)
                print(f"\n[OK] NeuroVault (sampled {len(origins)} images)")
                print(f"  Max origin diff vs atlas: {max_orig:.3f} mm")
                print(f"  Max voxel size diff: {max_vs:.3f} mm")
                if max_orig > 5.0:
                    print(f"  WARNING: NeuroVault has mixed spaces - some images may misalign")
                sources.extend(origins)
        except Exception as e:
            print(f"\n[SKIP] NeuroVault: {e}")
    elif args.sample_neurovault > 0:
        print(f"\n[SKIP] NeuroVault: no manifest at neurovault_data or neurovault_curated_data")

    # 5. Glasser + Tian source atlases
    try:
        glasser_path = _data / "atlas_cache" / "glasser360MNI.nii.gz"
        tian_path = _data / "atlas_cache" / "Tian_Subcortex_S2_3T_1mm.nii.gz"
        for p, label in [(glasser_path, "Glasser 360"), (tian_path, "Tian S2 3T 1mm")]:
            if p.exists():
                img = nib.load(str(p))
                s = _describe_affine(img.affine, label)
                s["shape"] = list(img.shape)
                diff = _affine_diff(atlas_affine, img.affine)
                s["vs_atlas"] = diff
                sources.append(s)
                print(f"\n[OK] {label} (source)")
                print(f"  Origin diff vs combined atlas: max {diff['origin_max_diff_mm']:.3f} mm")
    except Exception as e:
        print(f"\n[SKIP] Atlas sources: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("\nPipeline atlas uses Schaefer 2mm as reference (typically MNI152 2mm).")
    print("NiftiLabelsMasker resamples the atlas to each image at transform time.")
    print("If image affine is correct, parcellation is correct regardless of template.")
    print("\nPotential misalignment causes:")
    print("  - Different MNI variants (NLin2009cAsym vs Sym vs NLin6)")
    print("  - 3T vs 7T templates (different brain shape)")
    print("  - Wrong or missing metadata in NeuroVault images")
    print("  - Origin offset > 1 voxel (~2mm) can shift parcel boundaries")
    print("\nRecommendation: For critical alignment, resample all images to the")
    print("atlas before parcellation (resample_to_img(img, atlas_img)) instead of")
    print("relying on masker's transform-time resampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
