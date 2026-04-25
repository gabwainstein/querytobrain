#!/usr/bin/env python3
"""
Phase 0 + Phase 1 verification for the enrichment pipeline.
Run from repo root: python neurolab/scripts/verify_environment.py
Or from neurolab/: python scripts/verify_environment.py

Exit 0 = all checks passed. Exit 1 = failure (message to stderr).
"""
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def phase0_imports():
    """Phase 0: Verify all required packages import."""
    print("Phase 0: Checking imports...")
    try:
        import neuroquery
        from neuroquery import fetch_neuroquery_model
        import nilearn
        from nilearn import datasets as nilearn_datasets
        from nilearn.maskers import NiftiLabelsMasker
        import numpy as np
        from scipy import stats
        import nibabel as nib
        import pandas as pd
    except ImportError as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        print("  Install with: pip install -r neurolab/requirements-enrichment.txt", file=sys.stderr)
        return False
    print("  OK: neuroquery, nilearn, nibabel, numpy, scipy, pandas")
    # Log versions
    print(f"  neuroquery: {getattr(neuroquery, '__version__', '?')}")
    print(f"  nilearn: {nilearn.__version__}")
    print(f"  nibabel: {nib.__version__}")
    print(f"  numpy: {np.__version__}")
    return True


def phase1_fetch():
    """Phase 1: Fetch NeuroQuery model and pipeline atlas; one term -> parcellated vector."""
    print("\nPhase 1: Fetching data and model (first run may download ~500MB–1GB)...")
    from neuroquery import fetch_neuroquery_model, NeuroQueryModel
    import numpy as np
    import nibabel as nib

    # 1.1 NeuroQuery model (1.1.x: fetch returns path; load with NeuroQueryModel.from_data_dir)
    try:
        model_path = fetch_neuroquery_model()
        model = NeuroQueryModel.from_data_dir(model_path)
    except Exception as e:
        print(f"  FAIL loading NeuroQuery model: {e}", file=sys.stderr)
        return False
    vocab = list(model.vectorizer.get_feature_names())
    print(f"  OK: NeuroQuery model loaded, vocabulary size {len(vocab)}")

    # transform() expects list of documents; returns dict with 'brain_map' (list of images)
    try:
        result = model.transform(["attention"])
    except Exception as e:
        print(f"  FAIL model.transform(['attention']): {e}", file=sys.stderr)
        return False
    if "brain_map" not in result or not result["brain_map"]:
        print("  FAIL: result has no brain_map", file=sys.stderr)
        return False
    brain_map = result["brain_map"][0]
    print(f"  OK: transform(['attention']) returned brain_map")

    # brain_map may be path or image-like; get to Nifti1Image for masker
    if hasattr(brain_map, "get_fdata"):
        brain_img = brain_map
    elif isinstance(brain_map, (str, os.PathLike)):
        brain_img = nib.load(brain_map)
    else:
        brain_img = nib.Nifti1Image(np.asarray(brain_map), np.eye(4))
    data = brain_img.get_fdata()
    print(f"  brain_map shape: {data.shape}")

    # 1.2 Pipeline atlas (Glasser+Tian 392)
    try:
        from neurolab.parcellation import get_masker, get_n_parcels, resample_to_atlas
        n_parcels = get_n_parcels()
        masker = get_masker(memory="nilearn_cache", verbose=0)
        masker.fit()
    except Exception as e:
        print(f"  FAIL pipeline atlas: {e}", file=sys.stderr)
        print("  Build atlas: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian", file=sys.stderr)
        return False
    print(f"  OK: Pipeline atlas loaded ({n_parcels} parcels)")

    # 1.3 Resample and parcellate one term
    try:
        brain_img = resample_to_atlas(brain_img)
        parcellated = masker.transform(brain_img).ravel()
    except Exception as e:
        print(f"  FAIL parcellate: {e}", file=sys.stderr)
        return False
    if parcellated.shape != (n_parcels,):
        print(f"  FAIL: parcellated shape {parcellated.shape} != ({n_parcels},)", file=sys.stderr)
        return False
    valid = np.isfinite(parcellated) & (np.abs(parcellated) > 1e-12)
    if valid.sum() < 10:
        print("  FAIL: parcellated vector almost empty or NaN", file=sys.stderr)
        return False
    print(f"  OK: parcellated shape ({n_parcels},), {valid.sum()} non-near-zero values")
    print("  Phase 1 passed.")
    return True


def main():
    if not phase0_imports():
        sys.exit(1)
    if not phase1_fetch():
        sys.exit(1)
    print("\nAll checks passed. Ready for Phase 2 (build term maps cache).")


if __name__ == "__main__":
    main()
