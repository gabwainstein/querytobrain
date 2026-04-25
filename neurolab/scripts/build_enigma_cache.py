#!/usr/bin/env python3
"""
Build a (term, map) cache from ENIGMA Toolbox disorder summary statistics.

Loads case-control effect size maps (cortical thickness, surface area, subcortical volume)
from ENIGMA Working Groups via enigmatoolbox, resamples from Desikan-Killiany (DK) to
Pipeline atlas (Glasser+Tian, 392), and saves term_maps.npz + term_vocab.pkl.

**Spatial mapping (no cyclic replication):**
- Cortical (68 DK): parcel_to_surface(aparc_fsa5) -> surface_to_parcel(glasser_360_fsa5)
- Subcortical (16 ENIGMA): each structure -> 2 Tian S2 parcels (same Cohen's d)

**Requirements:** pip install enigmatoolbox. ENIGMA summary stats CSVs must be present
(enigmatoolbox datasets/summary_statistics/). Optional: --dk-to-schaefer path/to/mapping.npy
(shape n_dk x 392) for a custom mapping.

**Output:** term_maps.npz (N, 392), term_vocab.pkl. Merge with:
  build_expanded_term_maps.py --enigma-cache-dir neurolab/data/enigma_cache --save-term-sources

Usage (from repo root):
  python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache
  python neurolab/scripts/build_enigma_cache.py --output-dir neurolab/data/enigma_cache --disorders schizophrenia depression bipolar adhd
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


# Disorder -> list of (table_key, label_suffix). Table key must exist in load_summary_stats(disorder).
# Expanded per ENIGMA audit: CT + SA + SubVol + age/subtype splits. See Complete-audit-of-ENIGMA-consortium-datasets.
DISORDER_TABLES = {
    "22q": [
        ("CortThick_case_vs_controls", "22q11.2 deletion cortical thickness"),
        ("CortSurf_case_vs_controls", "22q11.2 deletion cortical surface area"),
        ("SubVol_case_vs_controls", "22q11.2 deletion subcortical volume"),
    ],
    "adhd": [
        ("CortThick_case_vs_controls_adult", "ADHD cortical thickness adult"),
        ("CortSurf_case_vs_controls_adult", "ADHD cortical surface area adult"),
        ("CortThick_case_vs_controls_pediatric", "ADHD cortical thickness pediatric"),
        ("CortSurf_case_vs_controls_pediatric", "ADHD cortical surface area pediatric"),
        ("SubVol_case_vs_controls_adult", "ADHD subcortical volume adult"),
    ],
    "anorexia": [
        ("CortThick_anorexia_case_controls", "anorexia nervosa cortical thickness"),
        ("CortSurf_anorexia_case_controls", "anorexia nervosa cortical surface area"),
        ("SubVol_anorexia_case_controls", "anorexia nervosa subcortical volume"),
    ],
    "antisocial": [
        ("CortThick_case_controls_antisocial", "antisocial behavior cortical thickness"),
        ("CortSurf_case_controls_antisocial", "antisocial behavior cortical surface area"),
        ("SubVol_case_controls_antisocial", "antisocial behavior subcortical volume"),
    ],
    "asd": [
        ("CortThick_case_vs_controls_meta_analysis", "autism spectrum cortical thickness"),
        ("SubVol_case_vs_controls_meta_analysis", "autism spectrum subcortical volume"),
    ],
    "bipolar": [
        ("CortThick_case_vs_controls_adult", "bipolar disorder cortical thickness adult"),
        ("CortSurf_case_vs_controls_adult", "bipolar disorder cortical surface area adult"),
        ("CortThick_case_vs_controls_adolescent", "bipolar disorder cortical thickness adolescent"),
        ("CortSurf_case_vs_controls_adolescent", "bipolar disorder cortical surface area adolescent"),
        ("SubVol_case_vs_controls_typeI", "bipolar disorder subcortical volume type I"),
    ],
    "depression": [
        ("CortThick_case_vs_controls_adult", "major depression cortical thickness adult"),
        ("CortSurf_case_vs_controls_adult", "major depression cortical surface area adult"),
        ("CortThick_case_vs_controls_adolescent", "major depression cortical thickness adolescent"),
        ("CortSurf_case_vs_controls_adolescent", "major depression cortical surface area adolescent"),
        ("SubVol_case_vs_controls", "major depression subcortical volume"),
    ],
    "epilepsy": [
        ("CortThick_case_vs_controls_ltle", "temporal lobe epilepsy left cortical thickness"),
        ("SubVol_case_vs_controls_ltle", "temporal lobe epilepsy left subcortical volume"),
        ("CortThick_case_vs_controls_rtle", "temporal lobe epilepsy right cortical thickness"),
        ("SubVol_case_vs_controls_rtle", "temporal lobe epilepsy right subcortical volume"),
        ("CortThick_case_vs_controls_gge", "generalized epilepsy cortical thickness"),
        ("SubVol_case_vs_controls_gge", "generalized epilepsy subcortical volume"),
        ("CortThick_case_vs_controls_allepilepsy", "epilepsy all cortical thickness"),
        ("SubVol_case_vs_controls_allepilepsy", "epilepsy all subcortical volume"),
    ],
    "ocd": [
        ("CortThick_case_vs_controls_adult", "OCD cortical thickness adult"),
        ("CortSurf_case_vs_controls_adult", "OCD cortical surface area adult"),
        ("CortThick_case_vs_controls_pediatric", "OCD cortical thickness pediatric"),
        ("CortSurf_case_vs_controls_pediatric", "OCD cortical surface area pediatric"),
        ("SubVol_case_vs_controls_adult", "OCD subcortical volume adult"),
    ],
    "parkinsons": [
        ("CortThick_PDvsCN", "Parkinson disease cortical thickness"),
        ("CortSurf_PDvsCN", "Parkinson disease cortical surface area"),
        ("Subvol_PDvsCN", "Parkinson disease subcortical volume"),  # note: lowercase 'vol' in toolbox
    ],
    "psychosis": [
        ("CortThick_CHR_vs_HC_postCombat_mega", "clinical high risk psychosis cortical thickness"),
        ("CortSurf_CHR_vs_HC_postCombat_mega", "clinical high risk psychosis cortical surface area"),
        ("SubVol_CHR_vs_HC_postCombat_mega", "clinical high risk psychosis subcortical volume"),
    ],
    "schizophrenia": [
        ("CortThick_case_vs_controls", "schizophrenia cortical thickness"),
        ("CortSurf_case_vs_controls", "schizophrenia cortical surface area"),
        ("SubVol_case_vs_controls", "schizophrenia subcortical volume"),
    ],
    "schizotypy": [
        ("CortThick_ThicknessCovariate", "schizotypy cortical thickness"),
        ("CortSurf_SurfAreaCovariate", "schizotypy cortical surface area"),
    ],
    "asymmetry": [
        ("CortThick_asymm_population_level", "cortical thickness asymmetry population"),
        ("CortSurf_asymm_population_level", "cortical surface area asymmetry population"),
    ],
}


def _dk_cortical_to_glasser360(dk_68: np.ndarray) -> np.ndarray:
    """Map DK 68 cortical parcel values to Glasser 360 via fsaverage5 vertex space.

    Uses enigmatoolbox: parcel_to_surface(dk, aparc_fsa5) -> vertices;
    surface_to_parcel(vertices, glasser_360_fsa5) -> 360.
    Preserves spatial structure (unlike cyclic replication).
    """
    from enigmatoolbox.utils.parcellation import parcel_to_surface, surface_to_parcel

    dk_68 = np.asarray(dk_68, dtype=np.float64).ravel()
    if dk_68.size != 68:
        raise ValueError(f"Expected 68 cortical DK values, got {dk_68.size}")

    # DK 68 -> fsaverage5 vertices (aparc has 71 regions; enigmatoolbox handles 68->71)
    vertices = parcel_to_surface(dk_68, "aparc_fsa5", fill=0)
    # Vertices -> Glasser 360 (API returns 361; index 0 = medial wall, 1-360 = parcels)
    glasser = surface_to_parcel(vertices, "glasser_360_fsa5")
    glasser = np.asarray(glasser, dtype=np.float64).ravel()
    if glasser.size == 361:
        glasser = glasser[1:361]  # drop medial wall
    return glasser[:360]


def _enigma_subcortical_16_to_tian32(sub_16: np.ndarray) -> np.ndarray:
    """Map ENIGMA 16 subcortical structures to Tian S2 32 parcels.

    ENIGMA order (from enigmatoolbox subcorticalvertices): L/R accumbens, amygdala,
    caudate, hippocampus, pallidum, putamen, thalamus, ventricles.
    Tian S2 subdivides each structure into 2 parcels. Assign same Cohen's d to both
    subdivisions (imperfect but preserves effect direction and magnitude).
    """
    sub_16 = np.asarray(sub_16, dtype=np.float64).ravel()
    if sub_16.size != 16:
        raise ValueError(f"Expected 16 ENIGMA subcortical values, got {sub_16.size}")

    out = np.zeros(32, dtype=np.float64)
    for i in range(16):
        out[2 * i] = sub_16[i]
        out[2 * i + 1] = sub_16[i]
    return out


def _dk_to_parcels_replicate(dk_values: np.ndarray, n_parcels: int) -> np.ndarray:
    """Placeholder: spread n_dk values to n_parcels by simple replication (no spatial mapping)."""
    n = len(dk_values)
    out = np.zeros(n_parcels, dtype=np.float64)
    for i in range(n_parcels):
        out[i] = dk_values[i % n]
    return out


def _fix_enigma_toolbox_filenames() -> None:
    """Patch enigmatoolbox summary_statistics for known filename typos (e.g. case-control vs case-controls)."""
    try:
        import enigmatoolbox
    except ImportError:
        return
    base = os.path.dirname(enigmatoolbox.__file__)
    stats_dir = os.path.join(base, "datasets", "summary_statistics")
    if not os.path.isdir(stats_dir):
        return
    fixes = [
        ("Schizophrenia_case-control_SubVol.csv", "Schizophrenia_case-controls_SubVol.csv"),
    ]
    for actual, expected in fixes:
        src = os.path.join(stats_dir, actual)
        dst = os.path.join(stats_dir, expected)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)


# Direction suffix for signed effect sizes (patients - controls); helps encoder learn sign
ENIGMA_DIRECTION_SUFFIX = " (patients - controls)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ENIGMA disorder (term, map) cache, Glasser+Tian 392.")
    parser.add_argument("--output-dir", default="neurolab/data/enigma_cache", help="Output dir for term_maps.npz, term_vocab.pkl")
    parser.add_argument("--disorders", nargs="*", default=None, help="Disorder names (default: all in DISORDER_TABLES)")
    parser.add_argument("--add-direction", action="store_true", default=True, help="Append '(patients - controls)' to labels for sign clarity (default: True)")
    parser.add_argument("--no-add-direction", action="store_false", dest="add_direction", help="Do not add direction suffix")
    parser.add_argument("--dk-to-schaefer", default=None, help="Path to .npy mapping matrix (n_dk x 400). If not set, use replicate fallback.")
    parser.add_argument("--no-replicate-fallback", action="store_true", help="Fail if DK->Schaefer mapping not available (no simple replication)")
    args = parser.parse_args()

    try:
        from enigmatoolbox.datasets import load_summary_stats
    except ImportError:
        print("Install enigmatoolbox: pip install enigmatoolbox", file=sys.stderr)
        return 1

    _fix_enigma_toolbox_filenames()

    from neurolab.parcellation import get_n_parcels, N_CORTICAL_GLASSER, N_SUBCORTICAL_TIAN
    n_parcels = get_n_parcels()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(repo_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DK->parcels mapping if provided
    dk_to_schaefer = None
    if args.dk_to_schaefer:
        p = Path(args.dk_to_schaefer)
        if not p.is_absolute():
            p = Path(repo_root) / args.dk_to_schaefer
        if p.exists():
            dk_to_schaefer = np.load(p)
            if dk_to_schaefer.ndim != 2 or dk_to_schaefer.shape[1] != n_parcels:
                print(f"Expected mapping shape (n_dk, {n_parcels}), got {dk_to_schaefer.shape}", file=sys.stderr)
                return 1

    disorders = args.disorders or list(DISORDER_TABLES.keys())
    terms = []
    maps_list = []

    for dis in disorders:
        if dis not in DISORDER_TABLES:
            print(f"Unknown disorder {dis!r}; skipping.", file=sys.stderr)
            continue
        try:
            sum_stats = load_summary_stats(dis)
        except Exception as e:
            print(f"load_summary_stats({dis!r}) failed: {e}", file=sys.stderr)
            continue
        for table_key, label_suffix in DISORDER_TABLES[dis]:
            if table_key not in sum_stats:
                continue
            df = sum_stats[table_key]
            if "d_icv" not in df.columns:
                continue
            d_icv = np.asarray(df["d_icv"], dtype=np.float64)
            if d_icv.size == 0:
                continue
            d_icv = np.nan_to_num(d_icv.ravel(), nan=0.0)

            # Resample to 392-D (Glasser 360 + Tian 32) via proper spatial mapping
            if dk_to_schaefer is not None:
                n_dk = dk_to_schaefer.shape[0]
                d_use = d_icv[:n_dk].astype(np.float64)
                if d_use.size < n_dk:
                    d_use = np.pad(d_use, (0, n_dk - d_use.size), constant_values=0.0)
                map_392 = (d_use @ dk_to_schaefer).ravel()
                w = dk_to_schaefer.sum(axis=0)
                if np.any(w > 0):
                    map_392 = map_392 / np.where(w > 0, w, 1.0)
            elif d_icv.size == 68:
                # Cortical table (CortThick/CortSurf): DK 68 -> Glasser 360 via vertex space
                cortex = _dk_cortical_to_glasser360(d_icv)
                subcortex = np.zeros(N_SUBCORTICAL_TIAN, dtype=np.float64)
                map_392 = np.concatenate([cortex, subcortex])
            elif d_icv.size == 16:
                # Subcortical table (SubVol): ENIGMA 16 -> Tian 32 (duplicate each struct to 2 parcels)
                cortex = np.zeros(N_CORTICAL_GLASSER, dtype=np.float64)
                subcortex = _enigma_subcortical_16_to_tian32(d_icv)
                map_392 = np.concatenate([cortex, subcortex])
            else:
                if args.no_replicate_fallback:
                    print(f"Unsupported d_icv size {d_icv.size} for table {table_key}. Expected 68 (cortical) or 16 (subcortical).", file=sys.stderr)
                    return 1
                map_392 = _dk_to_parcels_replicate(d_icv, n_parcels)
                print(f"Using replicate fallback for d_icv size {d_icv.size} (table {table_key}).", file=sys.stderr)

            if map_392.size < n_parcels:
                map_392 = np.pad(np.ravel(map_392), (0, n_parcels - map_392.size), constant_values=0.0)
            if map_392.size != n_parcels:
                continue
            lab = label_suffix + (ENIGMA_DIRECTION_SUFFIX if args.add_direction else "")
            terms.append(lab)
            maps_list.append(map_392.ravel()[:n_parcels].astype(np.float64))

    if not terms:
        print("No ENIGMA maps produced.", file=sys.stderr)
        return 1

    term_maps = np.stack(maps_list, axis=0)
    assert term_maps.shape[1] == n_parcels
    # Sanitize NaN/Inf (enigmatoolbox mapping can produce NaN for sparse data)
    term_maps = np.nan_to_num(term_maps, nan=0.0, posinf=0.0, neginf=0.0)
    # Z-score cortex and subcortex separately (CT vs subcortical volume have different scales)
    from neurolab.parcellation import zscore_cortex_subcortex_separately
    for i in range(term_maps.shape[0]):
        term_maps[i] = zscore_cortex_subcortex_separately(term_maps[i])
    np.savez_compressed(out_dir / "term_maps.npz", term_maps=term_maps)
    with open(out_dir / "term_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)
    print(f"Saved {len(terms)} ENIGMA disorder maps x {n_parcels} parcels -> {out_dir}")
    print("Merge with: build_expanded_term_maps.py --enigma-cache-dir", str(out_dir), "--save-term-sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
