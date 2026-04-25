"""
Central parcellation config: pipeline uses Glasser 360 + Tian S2 + Brainstem Navigator (~450 parcels),
optionally + Zaborszky basal forebrain + Neudorfer hypothalamus (~427 parcels).

All brain maps must be parcellated (or reparcellated) to this atlas. Build with:
  python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem
  python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem+bfb+hyp  # + Ch1-2, Ch4, LH, TM, PA
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Pipeline atlas: Glasser 360 + Tian S2 (32) + Brainstem Navigator (~66) = ~450 parcels
N_CORTICAL_GLASSER = 360
N_SUBCORTICAL_TIAN = 32
N_PARCELS_392 = N_CORTICAL_GLASSER + N_SUBCORTICAL_TIAN
N_BRAINSTEM_NAVIGATOR = 66  # 58 brainstem + 8 diencephalic
N_PARCELS_450 = N_PARCELS_392 + N_BRAINSTEM_NAVIGATOR

# Legacy / reference only
N_CORTICAL_SCHAEFER = 400
N_SUBCORTICAL_ASEG = 14
N_PARCELS_414 = N_CORTICAL_SCHAEFER + N_SUBCORTICAL_ASEG

_NEUROLAB_DATA = Path(__file__).resolve().parent / "data"
_ATLAS_427 = "combined_atlas_427.nii.gz"
_ATLAS_450 = "combined_atlas_450.nii.gz"
_ATLAS_392 = "combined_atlas_392.nii.gz"
N_CORTICAL = N_CORTICAL_GLASSER
N_SUBCORTICAL = N_SUBCORTICAL_TIAN
N_PARCELS = N_PARCELS_450


def get_combined_atlas_path(data_dir: str | Path | None = None) -> Path:
    """
    Path to the pipeline atlas NIfTI. Prefers 427 (with BFB/Hyp), then 450 (brainstem), then 392.
    """
    data_dir = Path(data_dir) if data_dir else _NEUROLAB_DATA
    p427 = data_dir / _ATLAS_427
    p450 = data_dir / _ATLAS_450
    p392 = data_dir / _ATLAS_392
    if p427.exists():
        return p427.resolve()
    if p450.exists():
        return p450.resolve()
    return p392.resolve()


def get_n_parcels(atlas_path: str | Path | None = None, data_dir: str | Path | None = None) -> int:
    """
    Number of parcels in the pipeline atlas.
    If the atlas file exists, returns its parcel count; otherwise returns 410.
    """
    path = Path(atlas_path) if atlas_path else get_combined_atlas_path(data_dir)
    if path.exists():
        import nibabel as nib
        data = nib.load(str(path)).get_fdata()
        uniq = np.unique(np.round(data).astype(int))
        uniq = uniq[uniq > 0]
        return int(len(uniq))
    return N_PARCELS_450


def resample_to_atlas(img, interpolation: str = "continuous"):
    """
    Resample a NIfTI image to the pipeline atlas space (Glasser+Tian 392).
    Use before parcellation when source images may have different origin, direction,
    or template (MNI sym vs asym, 3T vs 7T). Guarantees alignment regardless of source.
    """
    import warnings
    import nibabel as nib
    from nilearn.image import resample_to_img
    path = get_combined_atlas_path()
    if not path.exists():
        raise FileNotFoundError(f"Atlas not found: {path}")
    atlas_img = nib.load(str(path))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*binary images with continuous.*")
        return resample_to_img(img, atlas_img, interpolation=interpolation)


def zscore_cortex_subcortex_separately(vec: np.ndarray) -> np.ndarray:
    """
    Z-score cortex (parcels 0:360) and subcortex (parcels 360:N) separately.
    Subcortex includes Tian subcortical + Edlow brainstem nuclei.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    n = len(vec)
    if n <= N_CORTICAL_GLASSER:
        return vec
    cortex = vec[:N_CORTICAL_GLASSER]
    subcort = vec[N_CORTICAL_GLASSER:n]
    out = vec.copy()
    for i, part in enumerate([cortex, subcort]):
        m = np.nanmean(part)
        s = np.nanstd(part)
        if s > 1e-12:
            if i == 0:
                out[:N_CORTICAL_GLASSER] = (cortex - m) / s
            else:
                out[N_CORTICAL_GLASSER:n] = (subcort - m) / s
    return out


def get_masker(
    atlas_path: str | Path | None = None,
    n_parcels: int | None = None,
    memory: str = "nilearn_cache",
    verbose: int = 0,
    strategy: str = "mean",
):
    """
    NiftiLabelsMasker for the pipeline parcellation (Glasser+Tian, 392/427/450 parcels).
    Requires combined_atlas_*.nii.gz; build with build_combined_atlas.py.
    strategy: 'mean' (default) or 'sum' (for ROI/mask images that parcellate to zeros with mean).
    """
    from nilearn.maskers import NiftiLabelsMasker

    n_parcels = n_parcels if n_parcels is not None else N_PARCELS
    path = Path(atlas_path) if atlas_path else get_combined_atlas_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline atlas not found: {path}. "
            "Build with: python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem"
        )
    return NiftiLabelsMasker(
        labels_img=str(path),
        standardize=False,
        memory=memory,
        verbose=verbose,
        strategy=strategy,
    )
