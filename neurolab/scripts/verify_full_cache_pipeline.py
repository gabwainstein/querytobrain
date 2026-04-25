#!/usr/bin/env python3
"""
Comprehensive verification of NeuroLab cache pipeline:
- Atlas (Glasser 360 + Tian S2 = 392 parcels)
- All map caches use correct parcellation and resampling
- Critical-path datasets present
- Resampling usage in build scripts

Usage:
  python neurolab/scripts/verify_full_cache_pipeline.py
  python neurolab/scripts/verify_full_cache_pipeline.py --strict  # exit 1 on any failure
  python neurolab/scripts/verify_full_cache_pipeline.py --json  # machine-readable report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

N_CORTICAL_GLASSER = 360
N_SUBCORTICAL_TIAN = 32
N_PARCELS_392 = 392
N_PARCELS_450 = 450  # Glasser+Tian+brainstem (Brainstem Navigator)

# Critical-path caches (required for training/inference)
# (rel_path, file, key, desc, expect_parcels_first)
CRITICAL_CACHES = [
    ("decoder_cache", "term_maps.npz", "term_maps", "NeuroQuery full vocabulary", False),
    ("neurosynth_cache", "term_maps.npz", "term_maps", "NeuroSynth meta-analysis", False),
    ("unified_cache", "term_maps.npz", "term_maps", "Merged NQ+NS", False),
    ("merged_sources", "term_maps.npz", "term_maps", "Training set (NQ+NS+neuromaps+enigma+abagen)", False),
    ("neuromaps_cache", "annotation_maps.npz", "matrix", "Neuromaps annotations", False),
    ("enigma_cache", "term_maps.npz", "term_maps", "ENIGMA disorder maps", False),
    ("abagen_cache", "term_maps.npz", "term_maps", "AHBA gene expression", False),
    ("pdsp_cache", "pdsp_pc_projections.npz", "projections", "PDSP compound spatial maps", False),
    ("gene_pca", "pc_scores_full.npy", None, "Gene PCA basis (392×N)", True),
    ("fc_cache", "fc_maps.npz", "fc_maps", "FC maps (ENIGMA + Luppi + netneurolab)", False),
]

# Optional / secondary caches
OPTIONAL_CACHES = [
    ("neurovault_cache", "term_maps.npz", "term_maps", "NeuroVault task maps", False),
    ("neurovault_pharma_cache", "term_maps.npz", "term_maps", "NeuroVault pharma maps", False),
    ("pharma_neurosynth_cache", "term_maps.npz", "term_maps", "Pharma NeuroSynth", False),
    ("decoder_cache_expanded", "term_maps.npz", "term_maps", "Expanded decoder", False),
    ("embeddings", "all_training_embeddings.npy", None, "Precomputed training embeddings", "any"),  # no parcel check
]


def check_atlas(data_dir: Path) -> dict:
    """Verify combined atlas exists and has correct structure (Glasser 360 + Tian 32 [+ brainstem])."""
    out = {"ok": False, "path": None, "n_parcels": 0, "cortical_labels": 0, "subcortical_labels": 0, "error": None}
    try:
        from neurolab.parcellation import get_combined_atlas_path
        atlas_path = get_combined_atlas_path(data_dir)
    except ImportError:
        atlas_path = data_dir / "combined_atlas_410.nii.gz"
        if not atlas_path.exists():
            atlas_path = data_dir / "combined_atlas_392.nii.gz"
    out["path"] = str(atlas_path)
    if not atlas_path.exists():
        out["error"] = "Atlas not found. Run: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem"
        return out
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(str(atlas_path))
        data = np.round(img.get_fdata()).astype(int)
        uniq = np.unique(data)
        uniq = uniq[uniq > 0]
        n_parcels = len(uniq)
        cortical = sum(1 for u in uniq if 1 <= u <= 360)
        subcortical = sum(1 for u in uniq if 361 <= u <= 392)
        brainstem = sum(1 for u in uniq if u >= 393)
        out["n_parcels"] = n_parcels
        out["cortical_labels"] = cortical
        out["subcortical_labels"] = subcortical
        ok_392 = n_parcels == N_PARCELS_392 and cortical == N_CORTICAL_GLASSER and subcortical == N_SUBCORTICAL_TIAN
        ok_450 = n_parcels >= 400 and cortical == N_CORTICAL_GLASSER and subcortical == N_SUBCORTICAL_TIAN and brainstem >= 1
        out["ok"] = ok_392 or ok_450
        if not out["ok"]:
            out["error"] = f"Expected 392 (360+32) or ~450 (360+32+brainstem); got {cortical}+{subcortical}+{brainstem}={n_parcels}"
    except Exception as e:
        out["error"] = str(e)
    return out


def check_cache(data_dir: Path, rel_path: str, npz_file: str, key: str | None, desc: str, expect_parcels_first: bool | str = False) -> dict:
    """
    Check a single cache for correct parcellation.
    expect_parcels_first: True for gene_pca (392,15); "any" for embeddings (no parcel check).
    """
    expected = N_PARCELS_450  # prefer brainstem atlas
    try:
        from neurolab.parcellation import get_n_parcels
        expected = get_n_parcels()
    except ImportError:
        pass
    out = {"ok": False, "path": None, "n_maps": 0, "n_parcels": 0, "expected": expected, "error": None}
    base = data_dir / rel_path
    path = base / npz_file
    out["path"] = str(path)
    if not path.exists():
        out["error"] = "not found"
        return out
    try:
        import numpy as np
        if path.suffix == ".npy":
            arr = np.load(path)
        else:
            data = np.load(path)
            arr = data[key] if key and key in data.files else data[data.files[0]]
        arr = np.asarray(arr)
        if expect_parcels_first == "any":
            out["ok"] = arr.size > 0
            out["n_maps"] = arr.shape[0] if arr.ndim >= 1 else 0
            out["n_parcels"] = arr.shape[1] if arr.ndim >= 2 else 0
            return out
        if arr.ndim == 1:
            n_parcels = arr.size
            n_maps = 1
        elif expect_parcels_first:
            n_parcels = arr.shape[0]
            n_maps = arr.shape[1] if arr.ndim > 1 else 1
        else:
            n_maps, n_parcels = arr.shape[0], arr.shape[1]
        out["n_maps"] = int(n_maps)
        out["n_parcels"] = int(n_parcels)
        out["ok"] = n_parcels == expected
        if not out["ok"]:
            out["error"] = f"expected {expected} parcels, got {n_parcels}"
    except Exception as e:
        out["error"] = str(e)
    return out


def check_resampling_usage() -> dict:
    """Verify build scripts use resample_to_atlas and get_masker."""
    scripts_dir = _repo_root / "neurolab" / "scripts"
    # ENIGMA uses DK→parcels mapping (summary stats), not NIfTI resample
    scripts_that_parcellate = [
        "build_term_maps_cache.py",
        "build_neurosynth_cache.py",
        "build_neurovault_cache.py",
        "build_neuromaps_cache.py",
        "build_pharma_neurosynth_cache.py",
    ]
    out = {"ok": True, "scripts": []}
    for name in scripts_that_parcellate:
        path = scripts_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        has_resample = "resample_to_atlas" in text
        has_masker = "get_masker" in text or "NiftiLabelsMasker" in text
        ok = has_resample and has_masker
        out["scripts"].append({
            "name": name,
            "resample_to_atlas": has_resample,
            "get_masker": has_masker,
            "ok": ok,
        })
        if not ok:
            out["ok"] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify full cache pipeline")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any failure")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _repo_root / "neurolab" / "data"
    report = {
        "atlas": {},
        "critical_caches": [],
        "optional_caches": [],
        "resampling": {},
        "summary": {"critical_ok": 0, "critical_total": 0, "optional_ok": 0, "optional_total": 0},
    }

    # 1. Atlas
    report["atlas"] = check_atlas(data_dir)
    if not args.json:
        print("=" * 60)
        print("1. Atlas (Glasser 360 + Tian S2 = 392)")
        print("=" * 60)
        a = report["atlas"]
        status = "OK" if a["ok"] else "FAIL"
        print(f"  [{status}] {a['path']}")
        if a.get("n_parcels"):
            print(f"       Cortical: {a.get('cortical_labels', '?')}, Subcortical: {a.get('subcortical_labels', '?')}")
        if a.get("error"):
            print(f"       Error: {a['error']}")
        print()

    # 2. Critical caches
    if not args.json:
        print("2. Critical-path caches")
        print("=" * 60)
    for item in CRITICAL_CACHES:
        rel_path, npz_file, key, desc = item[:4]
        parcels_first = item[4] if len(item) > 4 else False
        r = check_cache(data_dir, rel_path, npz_file, key, desc, expect_parcels_first=parcels_first)
        report["critical_caches"].append({"path": rel_path, "desc": desc, **r})
        report["summary"]["critical_total"] += 1
        if r["ok"]:
            report["summary"]["critical_ok"] += 1
        if not args.json:
            status = "OK" if r["ok"] else ("SKIP" if r["error"] == "not found" else "MISMATCH")
            extra = f" - {r['n_maps']} maps x {r['n_parcels']} parcels" if r.get("n_parcels") else ""
            print(f"  [{status}] {rel_path}{extra}")
            if r.get("error") and r["error"] != "not found":
                print(f"       {r['error']}")
            # Decoder cache: flag if < 6000 terms (likely built with --max-terms 5000)
            if rel_path == "decoder_cache" and r.get("ok") and r.get("n_maps", 0) > 0 and r["n_maps"] < 6000:
                print(f"       WARNING: decoder has {r['n_maps']} terms (< 6000). Rebuild with --max-terms 0 for full vocab (~7.5K).")
    if not args.json:
        print()

    # 3. Optional caches
    if not args.json:
        print("3. Optional caches")
        print("=" * 60)
    for item in OPTIONAL_CACHES:
        rel_path, npz_file, key, desc = item[:4]
        parcels_first = item[4] if len(item) > 4 else False
        r = check_cache(data_dir, rel_path, npz_file, key, desc, expect_parcels_first=parcels_first)
        report["optional_caches"].append({"path": rel_path, "desc": desc, **r})
        report["summary"]["optional_total"] += 1
        if r["ok"]:
            report["summary"]["optional_ok"] += 1
        if not args.json:
            status = "OK" if r["ok"] else ("SKIP" if r["error"] == "not found" else "MISMATCH")
            extra = f" - {r['n_maps']} maps x {r['n_parcels']} parcels" if r.get("n_parcels") else ""
            print(f"  [{status}] {rel_path}{extra}")
    if not args.json:
        print()

    # 4. Resampling usage
    report["resampling"] = check_resampling_usage()
    if not args.json:
        print("4. Resampling (build scripts use resample_to_atlas + get_masker)")
        print("=" * 60)
        for s in report["resampling"]["scripts"]:
            status = "OK" if s["ok"] else "MISSING"
            print(f"  [{status}] {s['name']}: resample={s['resample_to_atlas']}, masker={s['get_masker']}")
        print()

    # Summary
    if not args.json:
        print("Summary")
        print("=" * 60)
        print(f"  Atlas: {'OK' if report['atlas']['ok'] else 'FAIL'}")
        print(f"  Critical caches: {report['summary']['critical_ok']}/{report['summary']['critical_total']} OK")
        print(f"  Optional caches: {report['summary']['optional_ok']}/{report['summary']['optional_total']} OK")
        print(f"  Resampling: {'OK' if report['resampling']['ok'] else 'Some scripts missing resample/masker'}")
        print()
        if report["atlas"]["ok"] and report["summary"]["critical_ok"] == report["summary"]["critical_total"]:
            print("  All critical checks passed.")
        else:
            print("  Gaps:")
            if not report["atlas"]["ok"]:
                print("    - Build atlas: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian")
            for c in report["critical_caches"]:
                if not c["ok"] and c.get("error") == "not found":
                    print(f"    - {c['path']}: run corresponding build script")
                elif not c["ok"] and c.get("error") and "parcels" in str(c.get("error", "")):
                    print(f"    - {c['path']}: rebuild with Glasser+Tian atlas (got {c.get('n_parcels')} parcels)")

    if args.json:
        print(json.dumps(report, indent=2))

    # Exit code
    all_ok = (
        report["atlas"]["ok"]
        and report["summary"]["critical_ok"] == report["summary"]["critical_total"]
        and report["resampling"]["ok"]
    )
    return 0 if all_ok or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
