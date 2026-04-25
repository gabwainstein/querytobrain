#!/usr/bin/env python3
"""
Check that all artifacts are in place for training and next phases (gene PCA, ontology expansion).

Verifies: atlas, decoder/expanded cache with correct n_parcels, optional gene PCA and ontologies.
Prints next steps (train, verify, gene PCA) and exits 0 if ready for training, 1 if something is missing.

Usage:
  python neurolab/scripts/check_training_readiness.py
  python neurolab/scripts/check_training_readiness.py --require-expanded  # require decoder_cache_expanded
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
_data_dir = _repo_root / "neurolab" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check readiness for training and next phases")
    ap.add_argument("--require-expanded", action="store_true", help="Require merged_sources or decoder_cache_expanded (not just decoder_cache)")
    ap.add_argument("--data-dir", type=Path, default=None, help="Override neurolab/data")
    args = ap.parse_args()

    data_dir = args.data_dir or _data_dir
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    try:
        from neurolab.parcellation import get_n_parcels, get_combined_atlas_path
        n_parcels = get_n_parcels()
        atlas_path = get_combined_atlas_path(data_dir)
    except Exception:
        n_parcels = 392
        atlas_path = data_dir / "combined_atlas_392.nii.gz"

    issues = []
    ready = True

    # 1. Atlas
    if not (atlas_path and atlas_path.exists()):
        issues.append(f"Atlas not found. Run: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian --output neurolab/data/combined_atlas_392.nii.gz")
        ready = False
    else:
        print(f"  [OK] Atlas: {atlas_path.name} ({n_parcels} parcels)")

    # 2. Merged sources (preferred) or decoder/expanded cache
    merged_npz = data_dir / "merged_sources" / "term_maps.npz"
    decoder_npz = data_dir / "decoder_cache" / "term_maps.npz"
    expanded_npz = data_dir / "decoder_cache_expanded" / "term_maps.npz"
    cache_ok = merged_npz.exists() if args.require_expanded else (merged_npz.exists() or decoder_npz.exists() or expanded_npz.exists())
    cache_dir = "merged_sources" if merged_npz.exists() else ("decoder_cache_expanded" if expanded_npz.exists() else ("decoder_cache" if decoder_npz.exists() else None))

    if not cache_ok:
        issues.append("No cache. Run rebuild: python neurolab/scripts/rebuild_all_caches.py --ensure-data --n-jobs 30")
        ready = False
    else:
        import numpy as np
        npz = np.load(merged_npz if merged_npz.exists() else (expanded_npz if expanded_npz.exists() else decoder_npz))
        shape = npz["term_maps"].shape
        if shape[1] != n_parcels:
            issues.append(f"Cache has {shape[1]} parcels but pipeline requires {n_parcels} (atlas). Re-run rebuild to reparcellate.")
            ready = False
        else:
            print(f"  [OK] Cache: {cache_dir} ({shape[0]} terms x {shape[1]} parcels)")
        # Decoder cache: warn if < 6000 terms (rebuild with --max-terms 0 for full vocab)
        if decoder_npz.exists():
            dec_npz = np.load(decoder_npz)
            n_dec = dec_npz["term_maps"].shape[0]
            if 0 < n_dec < 6000:
                print(f"  [WARN] decoder_cache has {n_dec} terms (< 6000). For full vocab (~7.5K) rebuild: python neurolab/scripts/build_term_maps_cache.py --cache-dir neurolab/data/decoder_cache --max-terms 0")

    # 3. Optional: gene PCA (for gene-head training)
    expanded_dir = data_dir / "merged_sources"
    gene_pca = expanded_dir / "gene_pca.pkl"
    gene_loadings = expanded_dir / "gene_loadings.npz"
    if (merged_npz.exists() or expanded_npz.exists()) and (gene_pca.exists() and gene_loadings.exists()):
        print(f"  [OK] Gene PCA: gene_pca.pkl, gene_loadings.npz (for gene-head training)")
    elif args.require_expanded and (merged_npz.exists() or expanded_npz.exists()):
        print("  [--] Gene PCA: optional. Re-run build_expanded_term_maps.py with --gene-pca-variance 0.95 to add gene head support.")

    # 4. Optional: ontologies (for expansion / data multiplication)
    ontologies = data_dir / "ontologies"
    if ontologies.exists() and (any(ontologies.glob("*.owl")) or any(ontologies.glob("*.obo"))):
        print(f"  [OK] Ontologies: {ontologies}")
    else:
        print("  [--] Ontologies: optional. Run: python neurolab/scripts/download_ontologies.py --clinical --extra --output-dir neurolab/data/ontologies")

    # 5. Next steps
    print()
    if not ready:
        print("Not ready. Fix the following:")
        for i in issues:
            print(f"  - {i}")
        return 1

    print("Ready for training. Next steps:")
    print()
    print("  1. Verify term labels (no broken/placeholder labels):")
    print("     python neurolab/scripts/verify_term_labels.py --cache-dir neurolab/data/merged_sources")
    print()
    print("  2. Verify parcellation and map types:")
    print("     python neurolab/scripts/verify_parcellation_and_map_types.py")
    print()
    print("  3. Train text-to-brain embedding (use merged_sources for NQ+NS+neuromaps+neurovault+enigma+abagen):")
    cache_for_train = cache_dir or "merged_sources"
    print(f"     python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/{cache_for_train} --output-dir neurolab/data/embedding_model --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings --max-terms 0 --epochs 50 --dropout 0.2 --weight-decay 1e-5")
    print()
    print("  4. (Optional) Gene PCA: run build_expanded_term_maps with --gene-pca-variance 0.95 when merging abagen.")
    print("     python neurolab/scripts/run_gene_pca_phase1.py   # then phase 2–4; or ensure build_expanded_term_maps.py was run with --gene-pca-variance 0.95")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
