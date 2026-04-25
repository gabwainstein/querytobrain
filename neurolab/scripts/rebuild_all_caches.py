#!/usr/bin/env python3
"""
Rebuild all map caches to the pipeline atlas: Glasser 360 + Tian S2 + Brainstem Navigator + BFB/Hyp (~427 parcels).

All maps are parcellated (or reparcellated) to the pipeline atlas. Builds combined_atlas_427.nii.gz
(Ch1-2, Ch4, LH, TM, PA) then runs each build script. Use --n-jobs 30 to parallelize decoder term processing.

  python neurolab/scripts/rebuild_all_caches.py --ensure-data --n-jobs 30
  python neurolab/scripts/rebuild_all_caches.py --quick
  python neurolab/scripts/rebuild_all_caches.py --skip-decoder --skip-neurosynth

NeuroVault options:
  --max-neurovault-maps 5000   Cap main task cache (default: all ~20k). Use for faster builds.
  --neurovault-tier 1|12|123|1234|all   By acquisition priority (see acquisition guide): 1=multi-domain, 12=+meta-analyses, 123=+domain-specific, 1234=+clinical/pharma. Default: all.
  --neurovault-collections ID[,ID,...]  Explicit collection IDs (overrides --neurovault-tier).
  --skip-neurovault-pharma     Skip drug-related NeuroVault cache (requires download_neurovault_pharma.py first).
  --skip-pharma-neurosynth    Skip pharmacological NeuroSynth meta-analysis (ketamine, caffeine, etc.).

  For quality-filtered meta-analysis data (T/Z maps, unthresholded, from published collections) instead of
  BrainPedia/HCP, run download_neurovault_metaanalysis.py first, then point build_neurovault_cache at that dir.

First time: use --ensure-data to download NeuroQuery, NeuroSynth, NeuroVault, neuromaps,
ontologies, and atlas cache before building (or run ensure_all_brain_map_data_local.py --download separately).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "neurolab" / "data"

# NeuroVault acquisition guide tiers (see download_neurovault_curated.py and acquisition guide)
# Tier 2: meta-analyses = highest information density. Tier 3: domain-specific (social, pain, emotion).
# Tier 4: clinical, structural, connectivity, pharma. Do not skip smaller collections.
def _load_nv_tiers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dvc", repo_root / "neurolab" / "scripts" / "download_neurovault_curated.py"
    )
    dvc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dvc)
    return dvc.TIER_1, dvc.TIER_2, dvc.TIER_3, dvc.TIER_4, dvc.WM_ATLAS

# Atlas collections excluded from training cache (NEUROVAULT_CURATED_CACHE_FORMATION.md)
NV_ATLAS_EXCLUDED = {262, 264, 1625, 6074, 9357}

try:
    from neurolab.parcellation import get_combined_atlas_path, get_n_parcels
except ImportError:
    get_combined_atlas_path = lambda d=None: (d or data_dir) / "combined_atlas_427.nii.gz"
    get_n_parcels = lambda: 427


def run(script: str, args: list[str], desc: str, required: bool = True) -> bool:
    path = repo_root / "neurolab" / "scripts" / script
    if not path.exists():
        print(f"  Skip {desc}: {path.name} not found")
        return not required
    cmd = [sys.executable, str(path)] + args
    print(f"\n{'='*60}\n  {desc}\n  {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        print(f"  FAILED: {script} exited {r.returncode}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild all caches to pipeline atlas (~427 parcels)")
    ap.add_argument("--n-jobs", type=int, default=16, help="Parallel jobs for decoder/neurosynth/neurovault (default 16)")
    ap.add_argument("--quick", action="store_true", help="Smaller caps: decoder 500, neurosynth 100, neuromaps 20")
    ap.add_argument("--skip-atlas", action="store_true", help="Skip building combined atlas (use existing)")
    ap.add_argument("--skip-decoder", action="store_true", help="Skip NeuroQuery decoder cache")
    ap.add_argument("--skip-neurosynth", action="store_true", help="Skip NeuroSynth cache")
    ap.add_argument("--skip-neurovault", action="store_true", help="Skip main NeuroVault cache")
    ap.add_argument("--skip-neurovault-pharma", action="store_true", help="Skip NeuroVault pharmacological cache")
    ap.add_argument("--skip-pharma-neurosynth", action="store_true", help="Skip pharmacological NeuroSynth meta-analysis cache")
    ap.add_argument("--max-neurovault-maps", type=int, default=0, help="Cap main NeuroVault task cache (0 = all ~20k). Use e.g. 5000 for faster build.")
    ap.add_argument("--neurovault-tier", choices=("1", "12", "123", "1234", "all"), default="all",
        help="Collections by acquisition priority (see acquisition guide): 1=multi-domain, 12=+meta-analyses, 123=+domain-specific, 1234=+clinical/pharma, all=full curated")
    ap.add_argument("--neurovault-collections", type=str, default=None, help="Comma-separated collection IDs (overrides --neurovault-tier)")
    ap.add_argument("--skip-neuromaps", action="store_true", help="Skip neuromaps cache")
    ap.add_argument("--skip-enigma", action="store_true", help="Skip ENIGMA structural cache")
    ap.add_argument("--skip-abagen", action="store_true", help="Skip abagen gene expression cache")
    ap.add_argument("--skip-pdsp", action="store_true", help="Skip PDSP Ki cache (compound→brain pharmacological pathway). PDSP is required for full pharmacological inference.")
    ap.add_argument("--build-expanded", action="store_true", help="Also build decoder_cache_expanded (ontology + merges); default: merged_sources only for training")
    ap.add_argument("--relabel-neurovault", action="store_true",
                    help="Run LLM relabeling on NeuroVault caches (requires OPENAI_API_KEY; discards junk, fixes vague labels)")
    ap.add_argument("--receptor-path", default=None, help="Receptor CSV/NPZ path to merge into expanded (optional)")
    ap.add_argument("--ensure-data", action="store_true", help="Run ensure_all_brain_map_data_local.py --download before rebuild (recommended first time)")
    args = ap.parse_args()

    if getattr(args, "ensure_data", False):
        ensure_script = repo_root / "neurolab" / "scripts" / "ensure_all_brain_map_data_local.py"
        if ensure_script.exists():
            print("Ensuring all brain map data is local before reparcellation...")
            r = subprocess.run([sys.executable, str(ensure_script), "--download"], cwd=str(repo_root))
            if r.returncode != 0:
                print("Ensure-data failed; fix missing data and re-run.")
                return 1
        else:
            print("ensure_all_brain_map_data_local.py not found; skipping --ensure-data.")

    n_parcels = get_n_parcels()
    if args.quick:
        decoder_max, neurosynth_max, neuromaps_max = 500, 100, 20
        print("Quick mode: limited term/annotation caps.")
    else:
        decoder_max, neurosynth_max, neuromaps_max = 0, 0, 0  # 0 = no cap
        print("Full mode: no caps (decoder/neurosynth may take 1-2 hours).")

    ok = True

    # 1. Combined atlas (Glasser+Tian+brainstem+BFB+Hyp, ~427 parcels)
    atlas_path = data_dir / "combined_atlas_427.nii.gz"
    if not args.skip_atlas:
        if atlas_path.exists():
            print(f"  Combined atlas exists: {atlas_path}")
        build_args = ["--atlas", "glasser+tian+brainstem+bfb+hyp", "--output", str(atlas_path)]
        ok &= run(
            "build_combined_atlas.py",
            build_args,
            "Build combined atlas (Glasser+Tian+brainstem+BFB+Hyp, ~427 parcels)",
            required=True,
        )
    else:
        has_atlas = (data_dir / "combined_atlas_427.nii.gz").exists() or (data_dir / "combined_atlas_450.nii.gz").exists() or (data_dir / "combined_atlas_392.nii.gz").exists()
        if not has_atlas:
            print(f"  ERROR: --skip-atlas but no combined atlas found. Pipeline requires combined atlas.")
            return 1

    n_jobs = max(1, int(getattr(args, "n_jobs", 1)))
    decoder_args = ["--cache-dir", str(data_dir / "decoder_cache"), "--max-terms", str(decoder_max) if decoder_max else "0"]
    if n_jobs > 1:
        decoder_args += ["--n-jobs", str(n_jobs)]

    # 2. Decoder (NeuroQuery)
    if not args.skip_decoder:
        ok &= run(
            "build_term_maps_cache.py",
            decoder_args,
            f"NeuroQuery decoder cache ({atlas_path.stem}, n_jobs={n_jobs})",
            required=True,
        )

    # 3. NeuroSynth (n_jobs>1 can speed up; NiMARE may have occasional failures under parallel)
    neurosynth_args = ["--cache-dir", str(data_dir / "neurosynth_cache"), "--max-terms", str(neurosynth_max) if neurosynth_max else "0"]
    if n_jobs > 1:
        neurosynth_args += ["--n-jobs", str(n_jobs)]
    if not args.skip_neurosynth:
        ok &= run(
            "build_neurosynth_cache.py",
            neurosynth_args,
            f"NeuroSynth cache (n_jobs={n_jobs})",
            required=True,
        )

    # 4. Merge NQ + NS -> unified
    if ok and not args.skip_decoder and not args.skip_neurosynth:
        ok &= run(
            "merge_neuroquery_neurosynth_cache.py",
            [
                "--neuroquery-cache-dir", str(data_dir / "decoder_cache"),
                "--neurosynth-cache-dir", str(data_dir / "neurosynth_cache"),
                "--output-dir", str(data_dir / "unified_cache"),
                "--prefer", "neuroquery",
            ],
            "Merge NQ+NS -> unified_cache",
            required=True,
        )

    # 5. Main NeuroVault (task contrasts) - prefer curated (acquisition guide) if present
    if not args.skip_neurovault:
        nv_curated = data_dir / "neurovault_curated_data"
        nv_legacy = data_dir / "neurovault_data"
        has_curated = (nv_curated / "manifest.json").exists() or (nv_curated / "downloads" / "neurovault").exists()
        has_legacy = (nv_legacy / "manifest.json").exists() or (nv_legacy / "downloads" / "neurovault").exists()
        nv_data = nv_curated if has_curated else (nv_legacy if has_legacy else None)
        if nv_data and ((nv_data / "manifest.json").exists() or (nv_data / "downloads" / "neurovault").exists()):
            nv_args = ["--data-dir", str(nv_data), "--output-dir", str(data_dir / "neurovault_cache")]
            if nv_data == nv_curated:
                nv_args += ["--average-subject-level"]
            if not (nv_data / "manifest.json").exists():
                nv_args += ["--from-downloads"]
            if n_jobs > 1:
                nv_args += ["--n-jobs", str(n_jobs)]
            if getattr(args, "max_neurovault_maps", 0) > 0:
                nv_args += ["--max-maps", str(args.max_neurovault_maps)]
            # Restrict to collections when requested (acquisition guide tiers, not size-based)
            nv_collections = None
            if getattr(args, "neurovault_collections", None):
                nv_collections = [int(x.strip()) for x in args.neurovault_collections.split(",") if x.strip()]
            elif getattr(args, "neurovault_tier", "all") != "all":
                t1, t2, t3, t4, wm = _load_nv_tiers()
                tiers = args.neurovault_tier
                ids = []
                if "1" in tiers:
                    ids.extend(t1)
                if "2" in tiers:
                    ids.extend(t2)
                if "3" in tiers:
                    ids.extend(t3)
                if "4" in tiers:
                    ids.extend(t4)
                if tiers in ("1234", "all"):
                    ids.extend(wm)
                nv_collections = [c for c in dict.fromkeys(ids) if c not in NV_ATLAS_EXCLUDED]
            if nv_collections:
                nv_args += ["--collections"] + [str(c) for c in nv_collections]
            nv_desc_suffix = ""
            if getattr(args, "max_neurovault_maps", 0) > 0:
                nv_desc_suffix = f" [capped {args.max_neurovault_maps}]"
            if nv_collections:
                tier_info = f"tier={args.neurovault_tier}" if getattr(args, "neurovault_collections", None) is None else "custom"
                nv_desc_suffix += f" [{tier_info}, {len(nv_collections)} collections]"
            ok &= run(
                "build_neurovault_cache.py",
                nv_args,
                f"NeuroVault task cache ({n_parcels}){nv_desc_suffix}",
                required=False,
            )
            if ok and (data_dir / "neurovault_cache" / "term_maps.npz").exists():
                ok &= run(
                    "improve_neurovault_labels.py",
                    ["--cache-dir", str(data_dir / "neurovault_cache"), "--no-fetch-metadata"],
                    "NeuroVault label improvement",
                    required=False,
                )
                if ok and getattr(args, "relabel_neurovault", False):
                    ok &= run(
                        "relabel_terms_with_llm.py",
                        ["--cache-dir", str(data_dir / "neurovault_cache"),
                         "--output-dir", str(data_dir / "neurovault_cache_relabeled")],
                        "NeuroVault LLM relabeling",
                        required=False,
                    )
        else:
            print("\n  Skip NeuroVault: neurovault_data or neurovault_curated_data not found. Run download_neurovault_data.py or download_neurovault_curated.py first.")

    # 5b. NeuroVault pharma (drug-related collections) - optional
    if not args.skip_neurovault_pharma:
        nv_pharma_data = data_dir / "neurovault_pharma_data"
        if (nv_pharma_data / "manifest.json").exists() or (nv_pharma_data / "downloads" / "neurovault").exists():
            nv_pharma_args = ["--data-dir", str(nv_pharma_data), "--output-dir", str(data_dir / "neurovault_pharma_cache"), "--from-downloads", "--pharma-add-drug", "--cluster-by-description"]
            if n_jobs > 1:
                nv_pharma_args += ["--n-jobs", str(n_jobs)]
            ok &= run(
                "build_neurovault_cache.py",
                nv_pharma_args,
                f"NeuroVault pharma cache ({n_parcels})",
                required=False,
            )
            if ok and (data_dir / "neurovault_pharma_cache" / "term_maps.npz").exists():
                ok &= run(
                    "improve_neurovault_labels.py",
                    ["--cache-dir", str(data_dir / "neurovault_pharma_cache"), "--no-fetch-metadata"],
                    "NeuroVault pharma label improvement",
                    required=False,
                )
                if ok and getattr(args, "relabel_neurovault", False):
                    ok &= run(
                        "relabel_terms_with_llm.py",
                        ["--cache-dir", str(data_dir / "neurovault_pharma_cache"),
                         "--output-dir", str(data_dir / "neurovault_pharma_cache_relabeled")],
                        "NeuroVault pharma LLM relabeling",
                        required=False,
                    )
        else:
            print("\n  Skip NeuroVault pharma: neurovault_pharma_data not found. Run download_neurovault_pharma.py first.")

    # 5c. Pharmacological NeuroSynth (drug meta-analyses) - optional; uses curated neurosynth_pharma_terms.json (all_terms_sorted = 194)
    if not args.skip_pharma_neurosynth:
        ok &= run(
            "build_pharma_neurosynth_cache.py",
            ["--output-dir", str(data_dir / "pharma_neurosynth_cache"), "--data-dir", str(data_dir / "neurosynth_data"),
             "--exclude-generic-terms",
             "--pharma-terms-key", "all_terms_sorted", "--min-studies", "3"],
            f"Pharmacological NeuroSynth cache ({n_parcels}, curated JSON)",
            required=False,
        )

    # 6. Neuromaps (PET / biological)
    if not args.skip_neuromaps:
        nm_args = ["--cache-dir", str(data_dir / "neuromaps_cache"), "--data-dir", str(data_dir / "neuromaps_data")]
        if neuromaps_max:
            nm_args += ["--max-annot", str(neuromaps_max)]
        if n_jobs > 1:
            nm_args += ["--n-jobs", str(n_jobs)]
        ok &= run("build_neuromaps_cache.py", nm_args, f"Neuromaps cache ({n_parcels})", required=False)

    # 7. ENIGMA (structural)
    if not args.skip_enigma:
        ok &= run(
            "build_enigma_cache.py",
            ["--output-dir", str(data_dir / "enigma_cache")],
            f"ENIGMA structural cache ({n_parcels})",
            required=False,
        )

    # 8. abagen (PET / gene expression) — all genes per master plan
    if not args.skip_abagen:
        abagen_args = ["--output-dir", str(data_dir / "abagen_cache"), "--all-genes"]
        ok &= run(
            "build_abagen_cache.py",
            abagen_args,
            f"abagen gene expression cache ({n_parcels}, all genes)",
            required=False,
        )

    # 8b. PDSP Ki cache (compound→brain pharmacological pathway; required for drug inference)
    if not args.skip_pdsp:
        gene_pca_dir = data_dir / "gene_pca"
        if not (gene_pca_dir / "pc_scores_full.npy").exists():
            ok &= run("run_gene_pca_phase1.py", ["--output-dir", str(gene_pca_dir)], "Gene PCA Phase 1", required=False)
            ok &= run("run_gene_pca_phase2.py", ["--output-dir", str(gene_pca_dir)], "Gene PCA Phase 2", required=False)
        if ok:
            ok &= run("download_pdsp_ki.py", ["--output-dir", str(data_dir / "pdsp_ki")], "PDSP Ki database", required=False)
        if ok and (data_dir / "pdsp_ki" / "KiDatabase.csv").exists():
            ok &= run(
                "build_pdsp_cache.py",
                ["--output-dir", str(data_dir / "pdsp_cache"), "--gene-pca-dir", str(gene_pca_dir),
                 "--pdsp-csv", str(data_dir / "pdsp_ki" / "KiDatabase.csv")],
                f"PDSP cache ({n_parcels})",
                required=False,
            )

    # 9. Merged sources (unified + neuromaps + neurovault + enigma + abagen, NO ontology)
    # Use this for training first; expanded (with ontology) built separately when ready.
    if ok and (data_dir / "unified_cache" / "term_maps.npz").exists():
        merge_args = [
            "--cache-dir", str(data_dir / "unified_cache"),
            "--output-dir", str(data_dir / "merged_sources"),
            "--no-ontology",
            "--save-term-sources",
        ]
        if (data_dir / "neuromaps_cache" / "annotation_maps.npz").exists():
            merge_args += ["--neuromaps-cache-dir", str(data_dir / "neuromaps_cache")]
        nv_cache = data_dir / "neurovault_cache_relabeled" if (data_dir / "neurovault_cache_relabeled" / "term_maps.npz").exists() else data_dir / "neurovault_cache"
        if (nv_cache / "term_maps.npz").exists():
            merge_args += ["--neurovault-cache-dir", str(nv_cache)]
        nv_pharma = data_dir / "neurovault_pharma_cache_relabeled" if (data_dir / "neurovault_pharma_cache_relabeled" / "term_maps.npz").exists() else data_dir / "neurovault_pharma_cache"
        if (nv_pharma / "term_maps.npz").exists():
            merge_args += ["--neurovault-pharma-cache-dir", str(nv_pharma)]
        if (data_dir / "pharma_neurosynth_cache" / "term_maps.npz").exists():
            merge_args += ["--pharma-neurosynth-cache-dir", str(data_dir / "pharma_neurosynth_cache")]
        if (data_dir / "enigma_cache" / "term_maps.npz").exists():
            merge_args += ["--enigma-cache-dir", str(data_dir / "enigma_cache")]
        if (data_dir / "abagen_cache" / "term_maps.npz").exists():
            # Per plan (gene_expression_pca_plan.md): gene PCA basis from run_gene_pca_phase1/2 → data/gene_pca/, not merge.
            merge_args += ["--abagen-cache-dir", str(data_dir / "abagen_cache"), "--max-abagen-terms", "500", "--abagen-add-gradient-pcs", "3", "--add-pet-residuals"]
        if args.receptor_path and Path(args.receptor_path).exists():
            merge_args += ["--receptor-path", args.receptor_path]
        ok &= run("build_expanded_term_maps.py", merge_args, "Merged sources (NQ+NS+neuromaps+neurovault+enigma+abagen, no ontology)", required=False)

    # 10. Expanded cache (optional; only when --build-expanded)
    if ok and getattr(args, "build_expanded", False) and (data_dir / "unified_cache" / "term_maps.npz").exists():
        expand_args = [
            "--cache-dir", str(data_dir / "unified_cache"),
            "--ontology-dir", str(data_dir / "ontologies"),
            "--output-dir", str(data_dir / "decoder_cache_expanded"),
            "--min-cache-matches", "2",
            "--min-pairwise-correlation", "0.3",
            "--save-term-sources",
        ]
        if (data_dir / "neuromaps_cache" / "annotation_maps.npz").exists():
            expand_args += ["--neuromaps-cache-dir", str(data_dir / "neuromaps_cache")]
        nv_exp = data_dir / "neurovault_cache_relabeled" if (data_dir / "neurovault_cache_relabeled" / "term_maps.npz").exists() else data_dir / "neurovault_cache"
        if (nv_exp / "term_maps.npz").exists():
            expand_args += ["--neurovault-cache-dir", str(nv_exp)]
        if (data_dir / "enigma_cache" / "term_maps.npz").exists():
            expand_args += ["--enigma-cache-dir", str(data_dir / "enigma_cache")]
        if (data_dir / "abagen_cache" / "term_maps.npz").exists():
            expand_args += ["--abagen-cache-dir", str(data_dir / "abagen_cache"), "--max-abagen-terms", "2000"]
        if args.receptor_path and Path(args.receptor_path).exists():
            expand_args += ["--receptor-path", args.receptor_path]
        ok &= run("build_expanded_term_maps.py", expand_args, "Expanded cache (ontology + all sources) [optional]", required=False)

    if not ok:
        return 1
    print("\n" + "="*60)
    print(f"  Run verify_term_labels.py to check for broken/placeholder labels:")
    print("  python neurolab/scripts/verify_term_labels.py --cache-dir neurolab/data/merged_sources")
    print("  Run verify_parcellation_and_map_types.py to confirm {n_parcels} parcels and map types.")
    print("  python neurolab/scripts/verify_parcellation_and_map_types.py")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
