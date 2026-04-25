#!/usr/bin/env python3
"""
Build functional connectivity cache from multiple sources (FC_DATASET_LANDSCAPE):
1. ENIGMA Toolbox (HCP group-average) — normative + multiple parcellations
2. Luppi 2023 pharmacological FC — when neurolab/data/luppi_fc_maps/ has data
3. netneurolab (liu_fc-pyspi) — when neurolab/data/netneurolab_fc/ has fc_cons_400.npy

Outputs: fc_degree.npy (normative), fc_maps.npz + fc_labels.pkl (all maps, N×392).

Usage:
  python neurolab/scripts/build_fc_cache.py --output-dir neurolab/data/fc_cache
  python neurolab/scripts/build_fc_cache.py --all-sources  # include Luppi, netneurolab when available

Requires: enigmatoolbox (pip install enigmatoolbox or git+https://github.com/MICA-MNI/ENIGMA.git)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# FreeSurfer aseg labels (same order as ENIGMA funcLabels_sctx)
ASEG_LABELS = [9, 10, 11, 12, 17, 18, 26, 48, 49, 50, 51, 52, 53, 58]
N_ASEG = 14
N_TIAN_S2 = 32


def _compute_aseg_to_tian_mapping(data_dir: Path) -> np.ndarray | None:
    """
    Compute mapping: for each Tian S2 parcel (0..31), which aseg region (0..13) has max overlap.
    Returns (32,) int array tian_to_aseg, or None if atlases unavailable.
    """
    try:
        from nilearn.image import resample_to_img
        from nilearn.datasets import fetch_atlas_schaefer_2018
        import nibabel as nib
    except ImportError:
        return None

    # Reuse build_combined_atlas fetch logic (same dir)
    import importlib.util
    bca_path = _scripts / "build_combined_atlas.py"
    spec = importlib.util.spec_from_file_location("build_combined_atlas", bca_path)
    bca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bca)
    _fetch_tian_s2 = bca._fetch_tian_s2
    _fetch_aseg_mni = bca._fetch_aseg_mni
    _remap_tian_to_361_392 = bca._remap_tian_to_361_392

    cache = data_dir / "atlas_cache"
    tian_img = _fetch_tian_s2(cache)
    if tian_img is None:
        return None

    aseg_path = _fetch_aseg_mni()
    if aseg_path is None:
        return None

    ref_bunch = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    ref_img = nib.load(ref_bunch["maps"])
    aseg_img = nib.load(str(aseg_path))

    tian_r = resample_to_img(tian_img, ref_img, interpolation="nearest")
    aseg_r = resample_to_img(aseg_img, ref_img, interpolation="nearest")

    tian_data = np.asarray(tian_r.get_fdata(), dtype=np.float32)
    aseg_data = np.asarray(aseg_r.get_fdata(), dtype=np.float32)

    tian_remap = _remap_tian_to_361_392(tian_data)  # 361..392
    aseg_remap = np.zeros_like(aseg_data, dtype=np.int32)
    for i, lab in enumerate(ASEG_LABELS):
        aseg_remap[aseg_data == lab] = i + 1  # 1..14

    tian_to_aseg = np.zeros(N_TIAN_S2, dtype=np.int32)
    for tian_idx in range(N_TIAN_S2):
        parcel_id = 361 + tian_idx
        mask = tian_remap == parcel_id
        if not np.any(mask):
            tian_to_aseg[tian_idx] = 0
            continue
        aseg_in_parcel = aseg_remap[mask]
        aseg_in_parcel = aseg_in_parcel[aseg_in_parcel > 0]
        if len(aseg_in_parcel) == 0:
            tian_to_aseg[tian_idx] = 0
            continue
        counts = np.bincount(aseg_in_parcel.astype(int), minlength=N_ASEG + 1)
        best_aseg = int(np.argmax(counts[1:]))  # 0..13
        tian_to_aseg[tian_idx] = best_aseg

    return tian_to_aseg


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FC cache from ENIGMA HCP group-average")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parcellation", default="glasser_360",
                        help="aparc, schaefer_100/200/300/400, glasser_360")
    parser.add_argument("--no-aseg-mapping", action="store_true",
                        help="Skip aseg→Tian overlap; pad subcortical with zeros")
    parser.add_argument("--save-mapping", action="store_true",
                        help="Save tian_to_aseg_mapping.npy when computed (for sharing)")
    parser.add_argument("--all-sources", action="store_true",
                        help="Include Luppi, netneurolab FC when data dirs exist")
    parser.add_argument("--all-enigma-parcellations", action="store_true",
                        help="Add ENIGMA FC from aparc, schaefer_100/200/300/400 (remap to 392)")
    args = parser.parse_args()

    try:
        from enigmatoolbox.datasets import load_fc
    except ImportError:
        print("Install enigmatoolbox: pip install enigmatoolbox", file=sys.stderr)
        print("Or: pip install git+https://github.com/MICA-MNI/ENIGMA.git", file=sys.stderr)
        return 1

    root = _scripts.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "fc_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FC (parcellation={args.parcellation})...")
    funcMatrix_ctx, funcLabels_ctx, funcMatrix_sctx, funcLabels_sctx = load_fc(parcellation=args.parcellation)

    # Cortico-cortical: FC degree = row-sum (or mean) of connectivity per region
    fc_ctx = np.asarray(funcMatrix_ctx, dtype=np.float64)
    fc_degree_ctx = np.sum(fc_ctx, axis=1)  # or np.mean for normalized
    fc_degree_ctx = fc_degree_ctx / (np.max(np.abs(fc_degree_ctx)) + 1e-1)  # normalize

    # Subcortico-cortical: 14 regions
    fc_sctx = np.asarray(funcMatrix_sctx, dtype=np.float64)
    fc_degree_sctx = np.sum(fc_sctx, axis=1)

    # Pipeline uses 392 parcels (Glasser 360 + Tian S2). ENIGMA subcortical is aseg 14.
    # Reparcellate aseg→Tian via volumetric overlap when atlases available.
    n_ctx = fc_degree_ctx.size
    n_sctx = fc_degree_sctx.size

    from neurolab.parcellation import get_n_parcels
    n_target = get_n_parcels()

    tian_to_aseg = None
    if not args.no_aseg_mapping:
        # Try precomputed mapping first (place in neurolab/data/fc_cache/)
        precomputed = root / "data" / "fc_cache" / "tian_to_aseg_mapping.npy"
        if precomputed.exists():
            loaded = np.load(precomputed)
            if loaded.shape == (N_TIAN_S2,):
                tian_to_aseg = loaded
                print("  Using precomputed aseg→Tian mapping")
        if tian_to_aseg is None:
            tian_to_aseg = _compute_aseg_to_tian_mapping(root / "data")
        if tian_to_aseg is not None and args.save_mapping:
            np.save(out_dir / "tian_to_aseg_mapping.npy", tian_to_aseg)
            print("  Saved tian_to_aseg_mapping.npy (commit to share)")
    if tian_to_aseg is not None:
        # Map aseg FC degree to Tian parcels via overlap
        fc_degree_tian = np.zeros(N_TIAN_S2, dtype=np.float64)
        for t in range(N_TIAN_S2):
            a = tian_to_aseg[t]
            if 0 <= a < n_sctx:
                fc_degree_tian[t] = fc_degree_sctx[a]
            else:
                fc_degree_tian[t] = 0.0
        # Normalize subcortical to match cortical scale
        if np.max(np.abs(fc_degree_tian)) > 1e-10:
            fc_degree_tian = fc_degree_tian / (np.max(np.abs(fc_degree_tian)) + 1e-10)
        fc_degree = np.concatenate([fc_degree_ctx, fc_degree_tian])
        subcort_method = "aseg_to_tian_overlap"
    else:
        # Fallback: concat 360+14, pad to 392 with zeros
        fc_degree = np.zeros(n_target, dtype=np.float64)
        fc_degree[:n_ctx] = fc_degree_ctx
        fc_degree[n_ctx:n_ctx + n_sctx] = fc_degree_sctx
        subcort_method = "aseg_pad_zeros"

    np.save(out_dir / "fc_degree.npy", fc_degree)
    np.savez_compressed(
        out_dir / "fc_cache.npz",
        fc_degree=fc_degree,
        fc_matrix_ctx=fc_ctx,
        fc_matrix_sctx=fc_sctx,
    )
    with open(out_dir / "fc_labels_ctx.json", "w") as f:
        json.dump(list(funcLabels_ctx) if hasattr(funcLabels_ctx, "__iter__") else funcLabels_ctx.tolist(), f)

    # Aggregate fc_maps from all sources (longer list)
    # map_type: "healthy" = normative/control FC; "drug" = pharmacological ΔFC
    fc_maps_list = [fc_degree.copy()]
    fc_labels_list = ["ENIGMA_normative_glasser360"]
    fc_map_types_list = ["healthy"]

    # Optional: ENIGMA other parcellations (remap to 392)
    # Landscape: aparc, schaefer_100/200/300/400, glasser_360
    if args.all_enigma_parcellations:
        for parc in ("aparc", "schaefer_100", "schaefer_200", "schaefer_300", "schaefer_400"):
            try:
                fm_ctx, _, fm_sctx, _ = load_fc(parcellation=parc)
                deg_ctx = np.sum(np.asarray(fm_ctx, dtype=np.float64), axis=1)
                deg_sctx = np.sum(np.asarray(fm_sctx, dtype=np.float64), axis=1)
                n_c = deg_ctx.size
                n_s = deg_sctx.size
                # Remap cortical via interpolation; subcortical 14→32
                deg_360 = np.interp(np.linspace(0, n_c - 1, 360), np.arange(n_c), deg_ctx) if n_c > 1 else np.zeros(360)
                deg_32 = np.interp(np.linspace(0, n_s - 1, 32), np.arange(n_s), deg_sctx) if n_s > 1 else np.zeros(32)
                combined = np.concatenate([deg_360, deg_32])
                if np.max(np.abs(combined)) > 1e-10:
                    combined = combined / (np.max(np.abs(combined)) + 1e-10)
                fc_maps_list.append(combined)
                fc_labels_list.append(f"ENIGMA_{parc}")
                fc_map_types_list.append("healthy")
            except Exception as e:
                print(f"  Skip {parc}: {e}", file=sys.stderr)

    # Optional: Luppi 2023 pharmacological FC (when data in luppi_fc_maps/)
    if args.all_sources:
        luppi_dir = root / "data" / "luppi_fc_maps"
        for p in luppi_dir.glob("*.npy") if luppi_dir.exists() else []:
            try:
                arr = np.load(p)
                arr = np.asarray(arr).ravel()
                if arr.size == 100:  # Schaefer 100
                    remap = np.interp(np.linspace(0, 99, 360), np.arange(100), arr)
                    sub = np.zeros(32, dtype=np.float64)
                    combined = np.concatenate([remap, sub])
                    fc_maps_list.append(combined)
                    fc_labels_list.append(f"Luppi_{p.stem}")
                    fc_map_types_list.append("drug")
            except Exception as e:
                print(f"  Skip Luppi {p.name}: {e}", file=sys.stderr)

    # Optional: netneurolab (liu_fc-pyspi, luppi-cognitive-matching, luppi-neurosynth-control)
    if args.all_sources:
        nn_dir = root / "data" / "netneurolab_fc"
        if nn_dir.exists():
            for p in nn_dir.glob("*.npy"):
                if "sc_cons" in p.name.lower() or "x_comm" in p.name.lower():
                    continue  # structural/comm matrices, not FC
                try:
                    data = np.load(p)
                    data = np.asarray(data)
                    if data.ndim == 2 and data.shape[0] == data.shape[1]:
                        deg = np.sum(data, axis=1)
                    else:
                        deg = data.ravel()
                    n_roi = deg.size
                    if n_roi < 2:
                        continue
                    label = "netneurolab_liu_fc_pyspi_Schaefer400" if p.name == "fc_cons_400.npy" else f"netneurolab_{p.stem}_n{n_roi}"
                    deg = np.nan_to_num(deg, nan=0.0, posinf=0.0, neginf=0.0)
                    remap = np.interp(np.linspace(0, n_roi - 1, 360), np.arange(n_roi), deg)
                    sub = np.zeros(32, dtype=np.float64)
                    combined = np.concatenate([remap, sub])
                    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.max(np.abs(combined)) > 1e-10:
                        combined = combined / (np.max(np.abs(combined)) + 1e-10)
                    fc_maps_list.append(combined)
                    fc_labels_list.append(label)
                    # liu_fc = healthy; luppi_cognitive = drug (anesthesia)
                    mt = "drug" if "luppi" in p.stem.lower() or "cognitive" in p.stem.lower() else "healthy"
                    fc_map_types_list.append(mt)
                except Exception as e:
                    print(f"  Skip netneurolab {p.name}: {e}", file=sys.stderr)

    # Save aggregated fc_maps
    import pickle
    fc_maps_arr = np.array(fc_maps_list, dtype=np.float64)
    np.savez_compressed(out_dir / "fc_maps.npz", fc_maps=fc_maps_arr)
    with open(out_dir / "fc_labels.pkl", "wb") as f:
        pickle.dump(fc_labels_list, f)
    with open(out_dir / "fc_map_types.pkl", "wb") as f:
        pickle.dump(fc_map_types_list, f)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "source": "ENIGMA Toolbox load_fc",
            "parcellation": args.parcellation,
            "n_ctx": n_ctx,
            "n_sctx": n_sctx,
            "n_target": n_target,
            "subcort_method": subcort_method,
            "n_fc_maps": len(fc_maps_list),
            "fc_labels": fc_labels_list,
            "fc_map_types": fc_map_types_list,
            "n_healthy": sum(1 for t in fc_map_types_list if t == "healthy"),
            "n_drug": sum(1 for t in fc_map_types_list if t == "drug"),
        }, f, indent=2)

    n_healthy = sum(1 for t in fc_map_types_list if t == "healthy")
    n_drug = sum(1 for t in fc_map_types_list if t == "drug")
    print(f"FC cache: {out_dir}")
    print(f"  fc_degree: {fc_degree.shape} (normative FC degree per parcel)")
    print(f"  fc_maps: {len(fc_maps_list)} total ({n_healthy} healthy, {n_drug} drug)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
