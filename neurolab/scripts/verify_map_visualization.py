#!/usr/bin/env python3
"""
Visual sanity check for merged_sources training maps (392 parcels, Glasser+Tian).

Samples terms per source, converts parcellated vectors to 3D NIfTI, plots glass brains,
and flags suspicious maps (diffuse, flat, non-specific).

Usage (from repo root):
  # Sample ~2 terms per source, save PNGs:
  python neurolab/scripts/verify_map_visualization.py --output-dir neurolab/data/map_verification

  # Inspect specific terms:
  python neurolab/scripts/verify_map_visualization.py --terms "schizophrenia cortical thickness" "HTR2A" "5-HT2A receptor" --output-dir neurolab/data/map_verification

  # Save NIfTIs for external viewers:
  python neurolab/scripts/verify_map_visualization.py --save-nifti --output-dir neurolab/data/map_verification

  # Quality flags only (no plots):
  python neurolab/scripts/verify_map_visualization.py --flag-only

  # Parcellation views (ortho slices, mosaic, surface projection):
  python neurolab/scripts/verify_map_visualization.py --view ortho --n-per-source 1
  python neurolab/scripts/verify_map_visualization.py --view all  # glass + ortho + mosaic + surf

Requires: nilearn, nibabel, matplotlib, numpy
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neurolab.parcellation import N_PARCELS_392, get_combined_atlas_path


def parcellated_to_nifti(parcellated_map: np.ndarray, atlas_img) -> "nib.Nifti1Image":
    """Convert 392-D parcellated vector to NIfTI volume for nilearn plotting."""
    import nibabel as nib

    atlas_data = atlas_img.get_fdata()
    out = np.zeros_like(atlas_data, dtype=np.float64)
    parcellated_map = np.asarray(parcellated_map).ravel()[:N_PARCELS_392]
    for i in range(N_PARCELS_392):
        mask = np.round(atlas_data).astype(int) == i + 1
        if np.any(mask):
            out[mask] = parcellated_map[i] if i < len(parcellated_map) else 0.0
    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)


def compute_quality_flags(
    parcellated: np.ndarray,
    global_mean: np.ndarray,
    n_parcels: int = N_PARCELS_392,
    threshold_nonzero: float = 0.95,
    threshold_peak: float = 0.01,
    threshold_global_r: float = 0.8,
) -> list[str]:
    """Return list of warning strings for a single map."""
    flags = []
    parcellated = np.asarray(parcellated).ravel()[:n_parcels]

    nonzero_frac = np.count_nonzero(np.abs(parcellated) > 1e-10) / n_parcels
    if nonzero_frac > threshold_nonzero:
        flags.append(f"DIFFUSE: {nonzero_frac:.0%} non-zero parcels — likely washed-out average")

    peak = float(np.nanmax(np.abs(parcellated)))
    if peak < threshold_peak:
        flags.append(f"FLAT: peak = {peak:.4f} — nearly flat map")

    if len(global_mean) == len(parcellated) and np.isfinite(parcellated).all() and np.isfinite(global_mean).all():
        r_global = float(np.corrcoef(parcellated, global_mean)[0, 1])
        if np.isfinite(r_global) and r_global > threshold_global_r:
            flags.append(f"NON-SPECIFIC: r={r_global:.3f} with global mean — not spatially specific")

    return flags


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual sanity check for merged_sources brain maps (392 parcels)"
    )
    parser.add_argument(
        "--cache-dir",
        default="neurolab/data/merged_sources",
        help="Merged sources or other cache dir (default: merged_sources)",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="Specific terms to inspect",
    )
    parser.add_argument(
        "--n-per-source",
        type=int,
        default=2,
        help="Sample N terms per source when auto-selecting (default 2)",
    )
    parser.add_argument(
        "--output-dir",
        default="neurolab/data/map_verification",
        help="Save PNGs and optional NIfTIs here",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Also save NIfTI files for external viewers",
    )
    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="Only print quality flags, no plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="plt.show() instead of saving",
    )
    parser.add_argument(
        "--view",
        choices=["glass", "ortho", "mosaic", "surf", "all"],
        default="glass",
        help="Plot style: glass (default), ortho (anatomical slices), mosaic (grid), surf (cortical projection), all (all views)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if Path(args.cache_dir).is_absolute() else repo_root / args.cache_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else repo_root / args.output_dir

    if not (cache_dir / "term_maps.npz").exists():
        print(f"Cache not found: {cache_dir}", file=sys.stderr)
        return 1

    # Load merged_sources
    data = np.load(cache_dir / "term_maps.npz")
    term_maps = np.asarray(data["term_maps"])
    with open(cache_dir / "term_vocab.pkl", "rb") as f:
        term_vocab = list(pickle.load(f))
    term_sources = None
    if (cache_dir / "term_sources.pkl").exists():
        with open(cache_dir / "term_sources.pkl", "rb") as f:
            term_sources = pickle.load(f)

    n_parcels = term_maps.shape[1]
    if n_parcels != N_PARCELS_392:
        print(f"Expected {N_PARCELS_392} parcels, got {n_parcels}. Use Glasser+Tian atlas.", file=sys.stderr)
        return 1

    global_mean = np.nanmean(term_maps, axis=0)
    print(f"Cache: {len(term_vocab)} terms × {n_parcels} parcels")
    if term_sources:
        print(f"Sources: {dict(sorted(Counter(term_sources).items(), key=lambda x: -x[1]))}")

    # Select terms to inspect
    if args.terms:
        norm_to_idx = {t.strip().lower().replace("_", " "): i for i, t in enumerate(term_vocab) if t}
        inspect_indices = []
        for t in args.terms:
            key = t.strip().lower().replace("_", " ")
            if key in norm_to_idx:
                inspect_indices.append(norm_to_idx[key])
            else:
                print(f"  Term not found: '{t}'")
    else:
        # Sample ~n_per_source per source
        rng = np.random.default_rng(42)
        inspect_indices = []
        if term_sources:
            for src in sorted(set(term_sources)):
                idx_src = [i for i, s in enumerate(term_sources) if s == src]
                n_pick = min(args.n_per_source, len(idx_src))
                pick = rng.choice(idx_src, n_pick, replace=False)
                inspect_indices.extend(pick.tolist())
        else:
            n_pick = min(20, len(term_vocab))
            inspect_indices = rng.choice(len(term_vocab), n_pick, replace=False).tolist()

    print(f"\nInspecting {len(inspect_indices)} terms:")
    for idx in inspect_indices:
        src = term_sources[idx] if term_sources else "?"
        print(f"  '{term_vocab[idx]}' (source: {src})")

    # Quality flags
    print("\n" + "=" * 70)
    print("QUALITY FLAGS")
    print("=" * 70)
    n_flagged = 0
    all_flags: dict[int, list[str]] = {}
    for idx in inspect_indices:
        flags = compute_quality_flags(term_maps[idx], global_mean, n_parcels=n_parcels)
        all_flags[idx] = flags
        if flags:
            n_flagged += 1
            src = term_sources[idx] if term_sources else "?"
            print(f"\n  [!] '{term_vocab[idx]}' (source: {src}):")
            for f in flags:
                print(f"      {f}")
    if n_flagged == 0:
        print("  [ok] No quality issues detected in inspected terms.")
    else:
        print(f"\n  Summary: {n_flagged}/{len(inspect_indices)} terms flagged")

    if args.flag_only:
        print("\n(--flag-only: skipping plots)")
        return 0

    # Load atlas
    atlas_path = get_combined_atlas_path()
    if not atlas_path.exists():
        print(f"Atlas not found: {atlas_path}", file=sys.stderr)
        return 1

    try:
        import nibabel as nib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from nilearn import plotting
    except ImportError as e:
        print(f"Cannot plot: {e}. Use --flag-only for text-only check.", file=sys.stderr)
        return 1

    atlas_img = nib.load(str(atlas_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in inspect_indices:
        term = term_vocab[idx]
        parcellated = term_maps[idx]
        src = term_sources[idx] if term_sources else "?"
        flags = all_flags.get(idx, [])

        nifti_img = parcellated_to_nifti(parcellated, atlas_img)

        if args.save_nifti:
            safe = term.replace(" ", "_").replace("/", "_")[:60]
            nifti_path = output_dir / f"{safe}.nii.gz"
            nib.save(nifti_img, str(nifti_path))
            print(f"  Saved NIfTI: {nifti_path}")

        nonzero_frac = np.count_nonzero(np.abs(parcellated) > 1e-10) / n_parcels
        peak = float(np.nanmax(np.abs(parcellated)))
        r_global = float(np.corrcoef(parcellated, global_mean)[0, 1]) if np.isfinite(parcellated).all() else 0
        flag_str = " [!]" if flags else " [ok]"

        title = (
            f"'{term}' (src={src}){flag_str}\n"
            f"sparsity={nonzero_frac:.2f}, peak={peak:.3f}, r_global={r_global:.3f}"
        )
        safe = term.replace(" ", "_").replace("/", "_")[:50]

        views = ["glass", "ortho", "mosaic", "surf"] if args.view == "all" else [args.view]

        for view in views:
            suffix = f"_{view}" if args.view == "all" else ""
            if view == "glass":
                fig = plt.figure(figsize=(12, 4))
                plotting.plot_glass_brain(
                    nifti_img,
                    title=title,
                    colorbar=True,
                    plot_abs=False,
                    threshold="auto",
                    figure=fig,
                )
            elif view == "ortho":
                fig = plt.figure(figsize=(10, 8))
                plotting.plot_stat_map(
                    nifti_img,
                    title=f"{title} [parcellation: ortho]",
                    colorbar=True,
                    display_mode="ortho",
                    cut_coords=None,
                    figure=fig,
                )
            elif view == "mosaic":
                fig = plt.figure(figsize=(14, 10))
                plotting.plot_stat_map(
                    nifti_img,
                    title=f"{title} [parcellation: mosaic]",
                    colorbar=True,
                    display_mode="mosaic",
                    figure=fig,
                )
            elif view == "surf":
                try:
                    out_path_surf = None if args.show else str(output_dir / f"{safe}{suffix}.png")
                    plotting.plot_img_on_surf(
                        nifti_img,
                        views=["lateral", "medial"],
                        hemispheres=["left", "right"],
                        colorbar=True,
                        title=title + " [parcellation: surface]",
                        output_file=out_path_surf,
                    )
                    if args.show:
                        fig = plt.gcf()
                    else:
                        print(f"  Saved: {out_path_surf}")
                        continue  # plot_img_on_surf already saved and closed
                except Exception as e:
                    print(f"  Skip surf for '{term}': {e}")
                    continue

            if args.show:
                plt.show()
            else:
                out_path = output_dir / f"{safe}{suffix}.png"
                plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
                print(f"  Saved: {out_path}")
            plt.close()

    print(f"\nDone. Review PNGs in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
