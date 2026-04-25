#!/usr/bin/env python3
"""
Verify all map caches use the pipeline parcellation (Glasser+Tian, 392 parcels) and report map types (fMRI, structural, PET).
Run from repo root. Any cache in Schaefer or other atlases must be reparcellated to Glasser+Tian first.

Usage:
  python neurolab/scripts/verify_parcellation_and_map_types.py
  python neurolab/scripts/verify_parcellation_and_map_types.py --strict  # exit 1 if any cache != 392
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Cache dirs under neurolab/data that contain parcellated maps.
# Format: (relative_path, map_type_category, description)
# map_type: "fmri" (task/activation), "structural" (thickness, volume), "pet" (receptor, gene expression, neurotransmitter)
CACHES = [
    ("decoder_cache", "fmri", "NeuroQuery term->map (decoder); task/activation"),
    ("neurosynth_cache", "fmri", "NeuroSynth meta-analysis; task/activation"),
    ("neurovault_cache", "fmri", "NeuroVault task-contrast maps"),
    ("neurovault_pharma_cache", "fmri", "NeuroVault pharmacological contrast maps"),
    ("pharma_neurosynth_cache", "fmri", "Pharmacological NeuroSynth meta-analysis"),
    ("merged_sources", "fmri", "Merged (NQ+NS+neuromaps+neurovault+enigma+abagen, no ontology)"),
    ("decoder_cache_expanded", "fmri", "Expanded (NQ/NS + ontology + optional neuromaps/neurovault/enigma/abagen)"),
    ("cache_brainpedia_plus_decoder", "fmri", "BrainPedia + decoder"),
    ("cache_brainpedia_plus_decoder_full_ontology", "fmri", "BrainPedia + decoder + full ontology"),
    ("neurovault_cache_brainpedia", "fmri", "NeuroVault BrainPedia"),
    ("neurovault_cache_test", "fmri", "NeuroVault test"),
    ("smoke_decoder_cache", "fmri", "Smoke test decoder cache"),
    ("neuromaps_cache", "pet", "Neuromaps annotations (receptors, metabolism, etc.); annotation_maps.npz"),
    ("enigma_cache", "structural", "ENIGMA disorder maps (cortical thickness, subcortical volume)"),
    ("abagen_cache", "pet", "AHBA gene expression (abagen)"),
]

# Receptor path is often a CSV/NPZ with (N, n_parcels); checked separately if path given
RECEPTOR_DEFAULT = "neurolab/data/receptor_atlas/hansen_400.csv"  # reparcellate to Glasser+Tian (392) if in Schaefer


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify parcellation (Glasser+Tian, 392) and list map types across caches.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any cache is not 392 parcels")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override neurolab/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _repo_root / "neurolab" / "data"
    try:
        from neurolab.parcellation import get_n_parcels
        expected = get_n_parcels()
    except Exception:
        expected = 392

    print(f"Expected parcellation: {expected} parcels (Glasser+Tian)")
    print()

    issues = []
    by_type = {"fmri": [], "structural": [], "pet": []}

    for rel_path, map_type, desc in CACHES:
        base = data_dir / Path(rel_path).name if "/" in rel_path or "\\" in rel_path else data_dir / rel_path
        if "neuromaps_cache" in rel_path:
            npz = base / "annotation_maps.npz"
            key = "matrix"  # neuromaps uses "matrix"
        else:
            npz = base / "term_maps.npz"
            key = "term_maps"

        if not npz.exists():
            print(f"  [SKIP] {rel_path}: not found")
            continue

        try:
            import numpy as np
            data = np.load(npz)
            if key in data.files:
                arr = data[key]
            else:
                arr = data[data.files[0]]
            arr = np.asarray(arr)
            if arr.ndim == 1:
                n_parcels = arr.size
                n_maps = 1
            else:
                n_maps, n_parcels = arr.shape[0], arr.shape[1]
        except Exception as e:
            print(f"  [FAIL] {rel_path}: {e}")
            issues.append((rel_path, f"load error: {e}"))
            continue

        ok = n_parcels == expected
        status = "OK" if ok else f"MISMATCH (got {n_parcels})"
        print(f"  [{status}] {rel_path}: {n_maps} maps x {n_parcels} parcels - {desc} ({map_type})")
        if not ok:
            issues.append((rel_path, f"expected {expected}, got {n_parcels}"))
        by_type[map_type].append((rel_path, n_maps, n_parcels))

    # Optional: receptor CSV/NPZ (often 400 or 414 columns)
    receptor_path = _repo_root / RECEPTOR_DEFAULT
    if receptor_path.exists():
        try:
            import numpy as np
            if receptor_path.suffix.lower() == ".csv":
                arr = np.genfromtxt(receptor_path, delimiter=",", skip_header=1)
            else:
                arr = np.load(receptor_path)
                arr = arr[arr.files[0]] if hasattr(arr, "files") else arr
            arr = np.asarray(arr)
            if arr.ndim == 2:
                n_maps, n_parcels = arr.shape[0], arr.shape[1]
            else:
                n_maps, n_parcels = 0, 0
            ok = n_parcels == expected
            status = "OK" if ok else f"MISMATCH (got {n_parcels})"
            print(f"  [{status}] receptor_atlas (e.g. Hansen): {n_maps} maps x {n_parcels} parcels - PET/receptor (pet)")
            if not ok:
                issues.append(("receptor_atlas", f"expected {expected}, got {n_parcels}"))
            by_type["pet"].append(("receptor_atlas", n_maps, n_parcels))
        except Exception as e:
            print(f"  [SKIP] receptor_atlas: {e}")
    else:
        print(f"  [SKIP] receptor_atlas: path not found ({receptor_path})")

    print()
    print("Summary by type:")
    print(f"  fMRI (task/activation): {len(by_type['fmri'])} caches")
    print(f"  Structural (thickness/volume): {len(by_type['structural'])} caches")
    print(f"  PET / gene / receptor: {len(by_type['pet'])} caches")
    print()
    print("Parcel space: All caches above are stored locally and transformed into the same")
    print(f"{expected}-D parcel space (Glasser+Tian). This includes NeuroQuery,")
    print("NeuroSynth, NeuroVault (task + pharma), neuromaps, and optionally ENIGMA,")
    print("abagen (gene expression), and receptor atlas when built/added.")
    if issues:
        print()
        print("Issues (parcellation mismatch or load error):")
        for name, msg in issues:
            print(f"  - {name}: {msg}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
