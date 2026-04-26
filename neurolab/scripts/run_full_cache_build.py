#!/usr/bin/env python3
"""
Run the full cache build pipeline: atlas, decoder, neurosynth, neurovault, neuromaps,
expanded ENIGMA, pharma, abagen, merged_sources. Optionally starts Semantic Scholar
compound fetch in background (rate-limited, no API key).

Usage:
  python neurolab/scripts/run_full_cache_build.py
  python neurolab/scripts/run_full_cache_build.py --quick
  python neurolab/scripts/run_full_cache_build.py --start-semantic-scholar-background  # compound literature
  python neurolab/scripts/run_full_cache_build.py --skip-decoder --skip-neurosynth  # faster partial build
  python neurolab/scripts/run_full_cache_build.py --download-neurovault-pharma  # ensure curated data (includes pharma) if missing

Run from repo root (querytobrain/). For background Semantic Scholar: starts fetch in a
separate process so you can keep working; logs to neurolab/data/compound_literature/compound_literature_fetch.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_scripts = _repo_root / "neurolab" / "scripts"
_data = _repo_root / "neurolab" / "data"
_pharma_schema = _data / "neurovault_pharma_schema.json"


def _pharma_collection_ids() -> list[int]:
    """Curated pharma collection IDs from schema (single source of truth)."""
    if not _pharma_schema.exists():
        return []
    with open(_pharma_schema, encoding="utf-8") as f:
        schema = json.load(f)
    return [c["id"] for c in schema.get("collections", [])]


def run(script: str, args: list[str], desc: str) -> bool:
    path = _scripts / script
    if not path.exists():
        print(f"  Skip {desc}: {path.name} not found")
        return True
    cmd = [sys.executable, str(path)] + args
    print(f"\n{'='*60}\n  {desc}\n  {' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(_repo_root))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Full cache build pipeline")
    ap.add_argument("--quick", action="store_true", help="Caps: decoder 500, neurosynth 100")
    ap.add_argument("--start-semantic-scholar-background", action="store_true",
                    help="Start fetch_semantic_scholar_compounds in background (rate-limited, no API key)")
    ap.add_argument("--skip-atlas", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--skip-neurosynth", action="store_true")
    ap.add_argument("--skip-neurovault", action="store_true")
    ap.add_argument("--skip-neurovault-pharma", action="store_true", help="Skip NeuroVault pharma cache (drug-related collections)")
    ap.add_argument("--download-neurovault-curated", action="store_true",
                    help="Run download_neurovault_curated.py --all (all tiers) if curated data missing")
    ap.add_argument("--download-neurovault-pharma", action="store_true",
                    help="Run download_neurovault_curated.py --all if curated data missing (curated includes pharma collections)")
    ap.add_argument("--skip-neuromaps", action="store_true")
    ap.add_argument("--skip-enigma", action="store_true")
    ap.add_argument("--skip-pharma-neurosynth", action="store_true")
    ap.add_argument("--skip-abagen", action="store_true")
    ap.add_argument("--skip-fc", action="store_true", help="Skip FC cache (ENIGMA load_fc)")
    ap.add_argument("--relabel-neurovault", action="store_true",
                    help="Run LLM relabeling on NeuroVault caches (requires OPENAI_API_KEY; discards junk, fixes vague labels)")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for decoder")
    args = ap.parse_args()

    ok = True

    # Optional: start Semantic Scholar in background first (so it runs while we build)
    if args.start_semantic_scholar_background:
        print("\nStarting Semantic Scholar compound fetch in background (5s delay, no API key)...")
        r = subprocess.run(
            [sys.executable, str(_scripts / "fetch_semantic_scholar_compounds.py"),
             "--output-dir", str(_data / "compound_literature"),
             "--delay", "5",
             "--resume",
             "--background"],
            cwd=str(_repo_root),
        )
        if r.returncode != 0:
            print("  Failed to start background process.", file=sys.stderr)
        else:
            print("  Background fetch started. Log: neurolab/data/compound_literature/compound_literature_fetch.log")

    # 1. Atlas (392 parcels: Glasser 360 + Tian S2; use glasser+tian for training consistency)
    if not args.skip_atlas:
        # Remove 427/450 atlases so parcellation uses 392 for all downstream caches
        for old in ("combined_atlas_427.nii.gz", "combined_atlas_450.nii.gz"):
            p = _data / old
            if p.exists():
                p.unlink()
                print(f"  Removed {old} so pipeline uses 392-parcel atlas")
        ok &= run("build_combined_atlas.py", ["--atlas", "glasser+tian", "--output", str(_data / "combined_atlas_392.nii.gz")],
                  "Combined atlas (Glasser+Tian 392 parcels)")

    # 2. Decoder (NeuroQuery)
    decoder_max = "500" if args.quick else "0"
    if not args.skip_decoder:
        decoder_args = ["--cache-dir", str(_data / "decoder_cache"), "--max-terms", decoder_max]
        if args.n_jobs > 1:
            decoder_args += ["--n-jobs", str(args.n_jobs)]
        ok &= run("build_term_maps_cache.py", decoder_args, "NeuroQuery decoder cache")

    # 3. NeuroSynth
    neurosynth_max = "100" if args.quick else "0"
    if not args.skip_neurosynth:
        ok &= run("build_neurosynth_cache.py",
                  ["--cache-dir", str(_data / "neurosynth_cache"), "--max-terms", neurosynth_max],
                  "NeuroSynth cache")

    # 4. Merge NQ + NS
    if ok and not args.skip_decoder and not args.skip_neurosynth:
        ok &= run("merge_neuroquery_neurosynth_cache.py",
                  ["--neuroquery-cache-dir", str(_data / "decoder_cache"),
                   "--neurosynth-cache-dir", str(_data / "neurosynth_cache"),
                   "--output-dir", str(_data / "unified_cache"), "--prefer", "neuroquery"],
                  "Merge NQ+NS -> unified_cache")

    # 5. NeuroVault (curated per acquisition guide: ~2–4K maps, not bulk ~20K)
    # Prefer neurovault_curated_data (download_neurovault_curated.py --all)
    # over neurovault_data (bulk BrainPedia/HCP). See NeuroVault acquisition guide.
    nv_curated = _data / "neurovault_curated_data"
    nv_data = nv_curated
    if not (nv_curated / "manifest.json").exists():
        if args.download_neurovault_curated and not args.skip_neurovault:
            ok &= run("download_neurovault_curated.py",
                      ["--all", "--output-dir", str(nv_curated)],
                      "NeuroVault curated (all tiers 1–4, ~2.7–5K maps)")
        if not (nv_curated / "manifest.json").exists():
            nv_data = _data / "neurovault_data"
    if not args.skip_neurovault and (nv_data / "manifest.json").exists():
        nv_args = ["--data-dir", str(nv_data), "--output-dir", str(_data / "neurovault_cache")]
        if nv_data == nv_curated:
            nv_args += ["--average-subject-level"]
        else:
            nv_args += ["--from-downloads"]
        ok &= run("build_neurovault_cache.py", nv_args, "NeuroVault cache")
        # Improve term labels (exclude atlas, strip [colN], clean WM atlas) — output to _improved for merge
        if ok and (_data / "neurovault_cache" / "term_maps.npz").exists():
            ok &= run("improve_neurovault_labels.py",
                      ["--cache-dir", str(_data / "neurovault_cache"),
                       "--output-dir", str(_data / "neurovault_cache_improved"),
                       "--no-fetch-metadata"],
                      "NeuroVault label improvement")
            if ok and args.relabel_neurovault:
                ok &= run("relabel_terms_with_llm.py",
                          ["--cache-dir", str(_data / "neurovault_cache_improved"),
                           "--output-dir", str(_data / "neurovault_cache_relabeled")],
                          "NeuroVault LLM relabeling")

    # 5b. NeuroVault pharma (curated drug/placebo collections from neurovault_curated_data)
    # Uses curated collection list + schema-based relabeling (see neurovault_pharma_schema.json).
    # Excludes neurovault_pharma_data (was polluted); pharma comes from curated --include-pharma.
    nv_curated_for_pharma = _data / "neurovault_curated_data"
    pharma_ids = _pharma_collection_ids()
    has_pharma = (
        (nv_curated_for_pharma / "manifest.json").exists()
        or (nv_curated_for_pharma / "downloads" / "neurovault").exists()
    ) and pharma_ids
    if not has_pharma and args.download_neurovault_pharma and not args.skip_neurovault_pharma:
        ok &= run("download_neurovault_curated.py",
                  ["--all", "--output-dir", str(nv_curated_for_pharma)],
                  "NeuroVault curated (includes pharma collections)")
        has_pharma = (nv_curated_for_pharma / "manifest.json").exists() and pharma_ids
    if not args.skip_neurovault_pharma and has_pharma:
        coll_args = [str(c) for c in pharma_ids]
        nv_pharma_args = [
            "--data-dir", str(nv_curated_for_pharma),
            "--output-dir", str(_data / "neurovault_pharma_cache"),
            "--collections", *coll_args,
            "--average-subject-level", "--pharma-add-drug", "--cluster-by-description",
        ]
        ok &= run("build_neurovault_cache.py", nv_pharma_args, "NeuroVault pharma cache (curated collections)")
        if ok and (_data / "neurovault_pharma_cache" / "term_maps.npz").exists():
            ok &= run("improve_neurovault_labels.py",
                      ["--cache-dir", str(_data / "neurovault_pharma_cache"),
                       "--output-dir", str(_data / "neurovault_pharma_cache_improved"),
                       "--no-fetch-metadata"],
                      "NeuroVault pharma label improvement")
            if ok:
                ok &= run("relabel_pharma_terms.py",
                         ["--cache-dir", str(_data / "neurovault_pharma_cache_improved"),
                          "--output-dir", str(_data / "neurovault_pharma_cache_relabeled")],
                         "NeuroVault pharma schema relabeling (drug/control/measure prefixes)")
                if ok:
                    ok &= run("relabel_pharma_semantic.py",
                             ["--cache-dir", str(_data / "neurovault_pharma_cache_improved"),
                              "--output-dir", str(_data / "neurovault_pharma_cache_semantic")],
                             "NeuroVault pharma semantic labels (natural language for OpenAI embeddings)")
            # Note: --relabel-neurovault (LLM) applies to main neurovault_cache only; pharma uses schema-based relabeling.

    # 6. Neuromaps (full: all MNI152 annotations; --max-annot 0)
    if not args.skip_neuromaps:
        nm_args = ["--cache-dir", str(_data / "neuromaps_cache"), "--data-dir", str(_data / "neuromaps_data"), "--max-annot", "0"]
        ok &= run("build_neuromaps_cache.py", nm_args, "Neuromaps cache (full)")

    # 7. ENIGMA (expanded: CT + SA + SubVol + splits)
    if not args.skip_enigma:
        ok &= run("build_enigma_cache.py", ["--output-dir", str(_data / "enigma_cache")],
                  "ENIGMA structural cache (expanded)")

    # 8. Pharma NeuroSynth
    # Pharma NeuroSynth: use curated term list from neurosynth_pharma_terms.json (all_terms_sorted = 194 terms)
    if not args.skip_pharma_neurosynth:
        ok &= run("build_pharma_neurosynth_cache.py",
                  ["--output-dir", str(_data / "pharma_neurosynth_cache"), "--data-dir", str(_data / "neurosynth_data"),
                   "--exclude-generic-terms",
                   "--pharma-terms-key", "all_terms_sorted", "--min-studies", "3"],
                  "Pharmacological NeuroSynth cache (curated JSON)")

    # 9. FC cache (ENIGMA + Luppi + netneurolab when available)
    if not args.skip_fc:
        fc_args = ["--output-dir", str(_data / "fc_cache"), "--parcellation", "glasser_360",
                   "--all-sources", "--all-enigma-parcellations"]
        ok &= run("build_fc_cache.py", fc_args, "FC cache (ENIGMA + all sources)")

    # 10. abagen
    if not args.skip_abagen:
        ok &= run("build_abagen_cache.py", ["--output-dir", str(_data / "abagen_cache"), "--all-genes"],
                  "abagen gene expression cache")


    # 12. Merged sources (training set)
    if ok and (_data / "unified_cache" / "term_maps.npz").exists():
        merge_args = [
            "--cache-dir", str(_data / "unified_cache"),
            "--output-dir", str(_data / "merged_sources"),
            "--no-ontology", "--save-term-sources",
            "--truncate-to-392",
        ]
        if (_data / "neuromaps_cache" / "annotation_maps.npz").exists():
            merge_args += ["--neuromaps-cache-dir", str(_data / "neuromaps_cache")]
        # Prefer label-improved caches: relabeled (LLM) > improved > base
        nv_cache = _data / "neurovault_cache"
        for cand in ("neurovault_cache_relabeled", "neurovault_cache_improved", "neurovault_cache"):
            if (_data / cand / "term_maps.npz").exists():
                nv_cache = _data / cand
                break
        if (nv_cache / "term_maps.npz").exists():
            merge_args += ["--neurovault-cache-dir", str(nv_cache)]
        nv_pharma = _data / "neurovault_pharma_cache"
        # Prefer semantic: natural-language labels for OpenAI embeddings; then relabeled; then improved
        for cand in ("neurovault_pharma_cache_semantic", "neurovault_pharma_cache_relabeled", "neurovault_pharma_cache_improved", "neurovault_pharma_cache"):
            if (_data / cand / "term_maps.npz").exists():
                nv_pharma = _data / cand
                break
        if (nv_pharma / "term_maps.npz").exists():
            merge_args += ["--neurovault-pharma-cache-dir", str(nv_pharma)]
        if (_data / "pharma_neurosynth_cache" / "term_maps.npz").exists():
            merge_args += ["--pharma-neurosynth-cache-dir", str(_data / "pharma_neurosynth_cache")]
        if (_data / "enigma_cache" / "term_maps.npz").exists():
            merge_args += ["--enigma-cache-dir", str(_data / "enigma_cache")]
        if (_data / "abagen_cache" / "term_maps.npz").exists():
            # Gradient PC labels: distinct style (PC1 brain_context, PC2/3 maximally different phrasing)
            # Gene enrichment: gene_info.json, 95% PCA denoising, tiered selection
            abagen_args = [
                "--abagen-cache-dir", str(_data / "abagen_cache"),
                "--max-abagen-terms", "500",
                "--abagen-add-gradient-pcs", "3",
                "--gradient-pc-label-style", "distinct",
                "--add-pet-residuals",
            ]
            if (_data / "gene_info.json").exists():
                abagen_args += ["--abagen-gene-info", str(_data / "gene_info.json")]
            abagen_args += ["--abagen-pca-variance", "0.95", "--gene-pca-variance", "0.95"]
            if (_data / "abagen_cache_receptor_residual_selected_denoised" / "term_maps.npz").exists():
                abagen_args += ["--additional-abagen-cache-dir", str(_data / "abagen_cache_receptor_residual_selected_denoised")]
            merge_args += abagen_args
        ok &= run("build_expanded_term_maps.py", merge_args, "Merged sources (training set)")

    print("\n" + "="*60)
    print("  Full cache build complete. Verify: python neurolab/scripts/verify_parcellation_and_map_types.py")
    if args.start_semantic_scholar_background:
        print("  Semantic Scholar fetch running in background. Check compound_literature_fetch.log")
    print("="*60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
