#!/usr/bin/env python3
"""
Visual sanity check: plot glass brains for ontology-expanded terms
and flag suspicious maps (blobby averages, flat maps, non-specific activations).

Run this BEFORE training on expanded maps to catch bad expansions.

Three automated quality flags:
  1. Non-zero fraction > 95%  -> map is diffuse (averaging destroyed spatial structure)
  2. Peak activation < 0.01  -> map is flat (cancellation or near-zero sources)
  3. Correlation with global mean > 0.8  -> map is not specific (looks like average brain)

Usage (from repo root):
  # Inspect random ontology-derived terms:
  python neurolab/scripts/inspect_expanded_maps.py \
    --cache-dir neurolab/data/decoder_cache_expanded \
    --n-random 20 \
    --output-dir neurolab/data/map_inspections

  # Inspect specific terms:
  python neurolab/scripts/inspect_expanded_maps.py \
    --cache-dir neurolab/data/decoder_cache_expanded \
    --terms "prosopagnosia" "executive function" "nociception" "reward prediction error" \
    --output-dir neurolab/data/map_inspections

  # Compare expanded vs direct maps for terms in both:
  python neurolab/scripts/inspect_expanded_maps.py \
    --cache-dir neurolab/data/decoder_cache_expanded \
    --compare-cache neurolab/data/decoder_cache \
    --n-random 10 \
    --output-dir neurolab/data/map_inspections

Requires: nilearn, nibabel, matplotlib, numpy
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

N_PARCELS = 400


def parcellated_to_nifti(parcellated_map: np.ndarray, atlas_img) -> "nib.Nifti1Image":
    """Convert 400-D parcellated vector back to a NIfTI volume for nilearn plotting."""
    import nibabel as nib

    atlas_data = atlas_img.get_fdata()
    out = np.zeros_like(atlas_data, dtype=np.float64)
    for i in range(1, N_PARCELS + 1):
        mask = atlas_data == i
        if i - 1 < len(parcellated_map):
            out[mask] = parcellated_map[i - 1]
    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)


def compute_quality_flags(
    parcellated: np.ndarray,
    global_mean: np.ndarray,
    threshold_nonzero: float = 0.95,
    threshold_peak: float = 0.01,
    threshold_global_r: float = 0.8,
) -> list[str]:
    """Return list of warning strings for a single map."""
    flags = []

    nonzero_frac = np.count_nonzero(parcellated) / N_PARCELS
    if nonzero_frac > threshold_nonzero:
        flags.append(
            f"DIFFUSE: {nonzero_frac:.0%} non-zero parcels — likely a washed-out average"
        )

    peak = float(np.max(np.abs(parcellated)))
    if peak < threshold_peak:
        flags.append(f"FLAT: peak activation = {peak:.4f} — nearly flat map")

    r_global = float(np.corrcoef(parcellated, global_mean)[0, 1])
    if np.isfinite(r_global) and r_global > threshold_global_r:
        flags.append(
            f"NON-SPECIFIC: r={r_global:.3f} with global mean — not spatially specific"
        )

    return flags


def load_cache(cache_dir: str) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Load term_maps, term_vocab, and optionally term_sources."""
    data = np.load(os.path.join(cache_dir, "term_maps.npz"))
    term_maps = np.asarray(data["term_maps"])

    with open(os.path.join(cache_dir, "term_vocab.pkl"), "rb") as f:
        term_vocab = pickle.load(f)
    term_vocab = list(term_vocab)

    sources = None
    src_path = os.path.join(cache_dir, "term_sources.pkl")
    if os.path.exists(src_path):
        with open(src_path, "rb") as f:
            sources = pickle.load(f)

    assert term_maps.shape[0] == len(term_vocab)
    assert term_maps.shape[1] == N_PARCELS
    return term_maps, term_vocab, sources


def main():
    parser = argparse.ArgumentParser(
        description="Visual sanity check for expanded brain maps"
    )
    parser.add_argument("--cache-dir", required=True, help="Expanded cache dir")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="Specific terms to inspect (if not set, picks random ontology terms)",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=20,
        help="Number of random ontology-derived terms to inspect (default 20)",
    )
    parser.add_argument(
        "--compare-cache",
        default=None,
        help="Optional: a second cache (e.g. base unified_cache) to compare maps side-by-side",
    )
    parser.add_argument(
        "--output-dir",
        default="neurolab/data/map_inspections",
        help="Save PNGs here",
    )
    parser.add_argument(
        "--show", action="store_true", help="plt.show() instead of saving to files"
    )
    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="Only print quality flags, don't generate plots (fast check)",
    )
    parser.add_argument(
        "--source-filter",
        choices=["ontology", "neuromaps", "receptor", "direct", "all"],
        default="ontology",
        help="Which source type to sample from (default: ontology)",
    )
    args = parser.parse_args()

    cache_dir = (
        args.cache_dir
        if os.path.isabs(args.cache_dir)
        else os.path.join(repo_root, args.cache_dir)
    )
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(repo_root, args.output_dir)
    )

    # Load primary cache
    term_maps, term_vocab, sources = load_cache(cache_dir)
    global_mean = term_maps.mean(axis=0)
    print(f"Cache: {len(term_vocab)} terms × {N_PARCELS} parcels")

    if sources:
        from collections import Counter
        counts = Counter(sources)
        print(f"Sources: {dict(counts)}")

    # Build norm lookup
    norm_to_idx = {}
    for i, t in enumerate(term_vocab):
        key = t.strip().lower().replace("_", " ")
        if key:
            norm_to_idx[key] = i

    # Select terms to inspect
    if args.terms:
        inspect_indices = []
        for t in args.terms:
            key = t.strip().lower().replace("_", " ")
            if key in norm_to_idx:
                inspect_indices.append(norm_to_idx[key])
            else:
                print(f"  Term not found in cache: '{t}'")
    else:
        # Pick random terms of the requested source type
        if sources and args.source_filter != "all":
            candidate_idx = [
                i for i, s in enumerate(sources) if s == args.source_filter
            ]
        else:
            candidate_idx = list(range(len(term_vocab)))

        if not candidate_idx:
            print(
                f"No terms with source='{args.source_filter}' found. "
                f"Available sources: {set(sources) if sources else 'unknown'}"
            )
            sys.exit(0)

        rng = np.random.default_rng(42)
        n_pick = min(args.n_random, len(candidate_idx))
        inspect_indices = list(rng.choice(candidate_idx, n_pick, replace=False))

    print(f"\nInspecting {len(inspect_indices)} terms:")
    for idx in inspect_indices:
        src = sources[idx] if sources else "?"
        print(f"  '{term_vocab[idx]}' (source: {src})")

    # Optionally load comparison cache
    compare_maps = None
    compare_vocab = None
    compare_norm_to_idx = None
    if args.compare_cache:
        comp_dir = (
            args.compare_cache
            if os.path.isabs(args.compare_cache)
            else os.path.join(repo_root, args.compare_cache)
        )
        if os.path.exists(os.path.join(comp_dir, "term_maps.npz")):
            compare_maps, compare_vocab, _ = load_cache(comp_dir)
            compare_norm_to_idx = {}
            for i, t in enumerate(compare_vocab):
                key = t.strip().lower().replace("_", " ")
                if key:
                    compare_norm_to_idx[key] = i
            print(f"Comparison cache: {len(compare_vocab)} terms")
        else:
            print(f"Comparison cache not found at {comp_dir}; skipping comparison.")

    # Quality flags pass (always runs)
    print("\n" + "=" * 70)
    print("QUALITY FLAGS")
    print("=" * 70)
    n_flagged = 0
    flag_counts = {"DIFFUSE": 0, "FLAT": 0, "NON-SPECIFIC": 0}
    all_flags: dict[int, list[str]] = {}

    for idx in inspect_indices:
        term = term_vocab[idx]
        flags = compute_quality_flags(term_maps[idx], global_mean)
        all_flags[idx] = flags
        if flags:
            n_flagged += 1
            src = sources[idx] if sources else "?"
            print(f"\n  [!] '{term}' (source: {src}):")
            for f in flags:
                print(f"      {f}")
                for key in flag_counts:
                    if f.startswith(key):
                        flag_counts[key] += 1

    if n_flagged == 0:
        print("  [ok] No quality issues detected in inspected terms.")
    else:
        print(f"\n  Summary: {n_flagged}/{len(inspect_indices)} terms flagged")
        for key, count in flag_counts.items():
            if count > 0:
                print(f"    {key}: {count}")

    # Full cache flag scan (quick summary)
    print("\n" + "-" * 70)
    print("FULL CACHE QUALITY SCAN")
    print("-" * 70)
    full_flags = {"DIFFUSE": 0, "FLAT": 0, "NON-SPECIFIC": 0, "clean": 0}
    for i in range(len(term_vocab)):
        flags = compute_quality_flags(term_maps[i], global_mean)
        if not flags:
            full_flags["clean"] += 1
        for f in flags:
            for key in ("DIFFUSE", "FLAT", "NON-SPECIFIC"):
                if f.startswith(key):
                    full_flags[key] += 1

    for key, count in full_flags.items():
        pct = count / len(term_vocab) * 100
        print(f"  {key:15s}: {count:6d} ({pct:5.1f}%)")

    if sources:
        # Break down flags by source
        print("\n  Flags by source type:")
        source_types = sorted(set(sources))
        for src in source_types:
            src_indices = [i for i, s in enumerate(sources) if s == src]
            n_src = len(src_indices)
            n_flagged_src = 0
            for i in src_indices:
                if compute_quality_flags(term_maps[i], global_mean):
                    n_flagged_src += 1
            pct = n_flagged_src / n_src * 100 if n_src else 0
            print(f"    {src:12s}: {n_flagged_src}/{n_src} flagged ({pct:.1f}%)")

    if args.flag_only:
        print("\n(--flag-only: skipping plots)")
        return

    # Plot glass brains
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from nilearn import datasets as nilearn_datasets
        from nilearn import plotting
        import nibabel as nib
    except ImportError as e:
        print(f"\nCannot plot: {e}. Use --flag-only for text-only quality check.")
        return

    atlas = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    os.makedirs(output_dir, exist_ok=True)

    for idx in inspect_indices:
        term = term_vocab[idx]
        parcellated = term_maps[idx]
        src = sources[idx] if sources else "?"
        flags = all_flags.get(idx, [])

        nifti_img = parcellated_to_nifti(parcellated, atlas_img)

        # Compute diagnostics for title
        nonzero_frac = np.count_nonzero(parcellated) / N_PARCELS
        peak = float(np.max(np.abs(parcellated)))
        r_global = float(np.corrcoef(parcellated, global_mean)[0, 1])
        flag_str = " [!]" if flags else " [ok]"

        # Check if we have a comparison map
        has_compare = False
        if compare_norm_to_idx is not None:
            key = term.strip().lower().replace("_", " ")
            if key in compare_norm_to_idx:
                has_compare = True

        if has_compare:
            # Side-by-side: expanded vs base
            comp_idx = compare_norm_to_idx[key]
            comp_map = compare_maps[comp_idx]
            comp_nifti = parcellated_to_nifti(comp_map, atlas_img)
            r_diff = float(np.corrcoef(parcellated, comp_map)[0, 1])

            fig, axes = plt.subplots(2, 1, figsize=(12, 6))

            plotting.plot_glass_brain(
                comp_nifti,
                title=f"BASE: '{term}'",
                colorbar=True,
                plot_abs=False,
                threshold="auto",
                axes=axes[0],
            )
            plotting.plot_glass_brain(
                nifti_img,
                title=f"EXPANDED: '{term}' (src={src}){flag_str}  [r_vs_base={r_diff:.3f}]",
                colorbar=True,
                plot_abs=False,
                threshold="auto",
                axes=axes[1],
            )
            plt.tight_layout()
        else:
            # Single glass brain
            title = (
                f"'{term}' (src={src}){flag_str}\n"
                f"sparsity={nonzero_frac:.2f}, peak={peak:.3f}, r_global={r_global:.3f}"
            )
            fig = plt.figure(figsize=(12, 4))
            plotting.plot_glass_brain(
                nifti_img,
                title=title,
                colorbar=True,
                plot_abs=False,
                threshold="auto",
                figure=fig,
            )

        if args.show:
            plt.show()
        else:
            safe_name = term.replace(" ", "_").replace("/", "_")[:50]
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out_path}")
            plt.close()

    print(f"\nDone. Review PNGs in {output_dir}")
    if n_flagged > 0:
        print(
            f"\n  Recommendation: {n_flagged} flagged maps detected.\n"
            f"  Consider tightening --min-pairwise-correlation or "
            f"raising --min-cache-matches in build_expanded_term_maps.py."
        )


if __name__ == "__main__":
    main()
