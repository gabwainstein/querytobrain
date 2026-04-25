#!/usr/bin/env python3
"""
Build combined cortical + subcortical atlas in MNI 2mm.

Pipeline uses glasser+tian (Glasser 360 + Tian S2 = 392 parcels). All maps must be
parcellated or reparcellated to this atlas; data in Schaefer space must be reprojected to Glasser+Tian.

  glasser+tian: Glasser 360 (cortex) + Tian S2 (32 subcortical) -> 392 parcels.
  glasser+tian+brainstem: + Brainstem Navigator (58 brainstem + 8 diencephalic nuclei, Hansen 2024) -> ~450 parcels.
  glasser+tian+brainstem+bfb+hyp: + Zaborszky basal forebrain (Ch1-2, Ch4) + Neudorfer hypothalamus (LH, TM, PA) -> ~456 parcels.
  schaefer+aseg: Schaefer 400 + aseg (14) -> 414 parcels; legacy only, not the pipeline target.

Usage (from repo root):
  python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem   # default, ~410 parcels
  python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian             # 392 parcels, no brainstem

Brainstem Navigator requires manual download from NITRC; run download_brainstem_navigator.py for instructions.
Basal forebrain (Zaborszky) and hypothalamus (Neudorfer) require manual download; run download_bfb_hyp_atlases.py for instructions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

# Schaefer + aseg
ASEG_LABELS = [9, 10, 11, 12, 17, 18, 26, 48, 49, 50, 51, 52, 53, 58]
N_SUBCORTICAL_ASEG = 14
N_CORTICAL_SCHAEFER = 400
N_PARCELS_414 = N_CORTICAL_SCHAEFER + N_SUBCORTICAL_ASEG

# Glasser + Tian S2
N_CORTICAL_GLASSER = 360
N_SUBCORTICAL_TIAN_S2 = 32
N_PARCELS_392 = N_CORTICAL_GLASSER + N_SUBCORTICAL_TIAN_S2

# Brainstem Navigator (Hansen et al. 2024 Nat Neurosci) - 58 brainstem + 8 diencephalic nuclei
# Same atlas used in "Integrating brainstem and cortical functional architectures"
# https://www.nitrc.org/projects/brainstemnavig
BRAINSTEM_NAVIGATOR_NUCLEI = 66  # 58 brainstem + 8 diencephalic (HTH may be excluded)

# Defaults for legacy build_combined_atlas (schaefer+aseg)
N_CORTICAL = N_CORTICAL_SCHAEFER
N_SUBCORTICAL = N_SUBCORTICAL_ASEG
N_PARCELS = N_PARCELS_414


def _fetch_schaefer_2mm():
    from nilearn.datasets import fetch_atlas_schaefer_2018
    bunch = fetch_atlas_schaefer_2018(n_rois=N_CORTICAL, resolution_mm=2)
    return nib.load(bunch["maps"])


def _fetch_aseg_mni():
    """Fetch aseg in MNI space (TemplateFlow). Returns NIfTI path or None."""
    try:
        from templateflow import api as tflow
    except ImportError:
        return None
    # Try MNI152NLin2009cAsym 2mm with atlas=aseg
    for template in ("MNI152NLin2009cAsym", "MNI152NLin6Asym"):
        for res in (2, 1):
            try:
                out = tflow.get(template, resolution=res, atlas="aseg", suffix="dseg")
                if out is not None and (isinstance(out, (list, tuple)) and len(out) > 0 or not isinstance(out, (list, tuple))):
                    path = Path(out[0]) if isinstance(out, (list, tuple)) else Path(out)
                    if path.exists() and (path.suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".gz"):
                        return path
            except Exception:
                continue
    return None


def _fetch_harvard_oxford_sub_2mm():
    """Fallback: Harvard-Oxford subcortical 2mm. Returns NIfTI image.
    HO sub has 48 regions; we remap the first 14 non-zero labels to 401..414.
    """
    try:
        from nilearn.datasets import fetch_atlas_harvard_oxford
    except ImportError:
        return None
    bunch = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    if not bunch or "maps" not in bunch:
        return None
    maps = bunch["maps"]
    if hasattr(maps, "get_fdata"):
        return maps  # already a nibabel image
    return nib.load(str(maps))


def _remap_ho_sub_to_401_414(ho_data: np.ndarray) -> np.ndarray:
    """Map first 14 non-zero Harvard-Oxford sub labels to 401..414. Other voxels 0."""
    uniq = np.unique(np.round(ho_data).astype(int))
    uniq = uniq[uniq > 0]
    if len(uniq) < 14:
        return np.zeros_like(ho_data, dtype=np.int32)
    # Use first 14 labels (by value)
    labels_to_use = sorted(uniq)[:14]
    out = np.zeros_like(ho_data, dtype=np.int32)
    for i, lab in enumerate(labels_to_use):
        out[ho_data == lab] = 401 + i
    return out


def _remap_aseg_to_401_414(aseg_data: np.ndarray) -> np.ndarray:
    """Map aseg label values to 401..414. Non-matching voxels stay 0."""
    out = np.zeros_like(aseg_data, dtype=np.int32)
    for i, label in enumerate(ASEG_LABELS):
        out[aseg_data == label] = 401 + i
    return out


# --- Glasser 360 + Tian S2 ---
GLASSER_URL = "https://github.com/brainspaces/glasser360/raw/master/glasser360MNI.nii.gz"
TIAN_S2_URL = "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_1mm.nii.gz"


def _download_to(path: Path, url: str, desc: str = "", headers: dict | None = None) -> bool:
    """Download url to path. Returns True on success."""
    try:
        from urllib.request import urlopen, Request
    except ImportError:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = headers or {"User-Agent": "NeuroLab-Atlas/1.0"}
    req = Request(url, headers=hdr)
    try:
        with urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                return False
            path.write_bytes(resp.read())
        if desc:
            print(f"  Downloaded {desc} -> {path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  Download failed {url}: {e}", file=sys.stderr)
        return False




def _fetch_glasser_mni(cache_dir: Path) -> nib.Nifti1Image | None:
    """Fetch Glasser 360 MNI volumetric atlas (brainspaces/glasser360)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "glasser360MNI.nii.gz"
    if not out.exists():
        if not _download_to(out, GLASSER_URL, "Glasser 360"):
            return None
    return nib.load(str(out))


def _fetch_tian_s2(cache_dir: Path) -> nib.Nifti1Image | None:
    """Fetch Tian subcortex S2 (32 parcels) 1mm NIfTI from yetianmed/subcortex."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "Tian_Subcortex_S2_3T_1mm.nii.gz"
    if not out.exists():
        if not _download_to(out, TIAN_S2_URL, "Tian S2"):
            return None
    return nib.load(str(out))


def _find_brainstem_navigator_nuclei(cache_dir: Path) -> list[Path]:
    """
    Find nucleus NIfTI files in an extracted Brainstem Navigator zip.
    Uses MNI space only; labels_thresholded_binary_0.35 (Hansen et al. 2024).
    Excludes IIT (different template) and probabilistic (use binary for parcellation).
    """
    cache_dir = Path(cache_dir)
    skip_names = {"template", "t1", "t2", "mean", "mask", "mni152", "fsl", "brain", "ref"}
    nuclei = []
    for p in cache_dir.rglob("*.nii.gz"):
        path_str = str(p).replace("\\", "/").lower()
        if "__macosx" in path_str or "/._" in path_str or p.name.startswith("._"):
            continue
        name_lower = p.stem.lower()
        if any(s in name_lower for s in skip_names):
            continue
        if p.stat().st_size < 1000:
            continue
        if "brainstem" not in path_str and "navigator" not in path_str and "diencephalic" not in path_str:
            continue
        if "iit" in path_str:
            continue  # Must be MNI (Hansen); IIT is different template
        if "mni" not in path_str:
            continue
        if "template" in path_str or "images_" in path_str:
            continue
        if "labels_thresholded_binary_0.35" not in path_str:
            continue  # Hansen uses binary 0.35; avoid probabilistic/other thresholds
        nuclei.append(p)
    # Prefer whole nuclei; exclude subregions when whole exists (RN1/RN2 if RN, etc.)
    paths = sorted(set(nuclei), key=lambda x: (x.parent.name, x.name))

    def _base(p: Path) -> str:
        name = p.name.removesuffix(".gz").removesuffix(".nii")
        for s in ("_l", "_r", "_L", "_R"):
            if name.endswith(s):
                return name[: -len(s)]
        return name

    bases = {_base(p) for p in paths}
    subregions_to_whole = {"RN1": "RN", "RN2": "RN", "SN1": "SN", "SN2": "SN", "STh1": "STh", "STh2": "STh"}
    keep = []
    for p in paths:
        base = _base(p)
        whole = subregions_to_whole.get(base)
        if whole and whole in bases:
            continue  # Skip RN1/RN2 when RN exists, etc.
        keep.append(p)
    return keep


def _fetch_brainstem_navigator(cache_dir: Path) -> list[Path] | None:
    """
    Locate Brainstem Navigator nucleus files (manual download required).
    Download BrainstemNavigatorv1.0.zip from https://www.nitrc.org/projects/brainstemnavig
    Extract to atlas_cache/ so that atlas_cache/BrainstemNavigatorv1.0/ contains the atlas.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    nuclei = _find_brainstem_navigator_nuclei(cache_dir)
    if len(nuclei) < 10:
        print(
            "  Brainstem Navigator not found. Extract BrainstemNavigatorv1.0.zip to:\n"
            f"    {cache_dir}\n"
            "  Download from: https://www.nitrc.org/projects/brainstemnavig\n"
            "  (Same atlas as Hansen et al. 2024 Nat Neurosci)",
            file=sys.stderr,
        )
        return None
    print(f"  Found Brainstem Navigator: {len(nuclei)} nucleus masks", file=sys.stderr)
    return nuclei


JUBRAIN_GITHUB_RAW = "https://github.com/inm7/jubrain-anatomy-toolbox/raw/master"


def _load_jubrain_mat73(mat_path: str) -> dict:
    """Load JuBrain_Data v7.3 .mat (HDF5) with h5py. Returns dict with 'JuBrain' key."""
    import h5py

    out: dict = {}
    with h5py.File(mat_path, "r") as f:
        # v7.3: variable names as top-level; JuBrain is a group/struct
        for key in ("JuBrain", "juBrain", "jubrain"):
            if key in f:
                g = f[key]
                break
        else:
            # Find first group that has Namen
            for k in f.keys():
                if k.startswith("__"):
                    continue
                if "Namen" in f[k]:
                    g = f[k]
                    break
            else:
                raise KeyError("No JuBrain struct in .mat")

        def _read(ds):
            if ds is None:
                return None
            d = np.asarray(ds)
            if d.dtype == np.object_ or (d.dtype.kind == "O" and d.size == 1):
                return d
            return d

        def _get(h, name):
            if name not in h:
                return None
            v = h[name]
            if isinstance(v, h5py.Dataset):
                arr = np.asarray(v)
                # MATLAB stores references; dereference if needed
                if arr.dtype.kind == "O" or arr.dtype == np.object_:
                    pass
                return arr
            return v

        Namen_ref = _get(g, "Namen")
        idx_arr = _get(g, "idx")
        mpm_arr = _get(g, "mpm")
        Vo_ref = _get(g, "Vo")

        if Namen_ref is None or idx_arr is None or mpm_arr is None or Vo_ref is None:
            raise KeyError("JuBrain missing Namen/idx/mpm/Vo")

        # Namen: in v7.3 often cell array of strings - refs to char arrays
        names_list = []
        namen = np.atleast_1d(Namen_ref)
        for i in range(namen.size):
            ref = namen.flat[i]
            if hasattr(ref, "__iter__") and not isinstance(ref, (str, bytes)):
                ref = ref.flat[0] if hasattr(ref, "flat") else ref
            if isinstance(ref, h5py.h5r.Reference):
                try:
                    obj = f[ref]
                    if isinstance(obj, h5py.Dataset):
                        chars = np.asarray(obj)
                        s = "".join(chr(c) for c in chars.flatten() if c != 0)
                        names_list.append(s.strip())
                    else:
                        names_list.append(str(ref))
                except Exception:
                    names_list.append("")
            elif isinstance(ref, (np.ndarray, np.generic)):
                try:
                    if ref.size == 1:
                        r = ref.item()
                        if isinstance(r, h5py.h5r.Reference):
                            obj = f[r]
                            chars = np.asarray(obj)
                            s = "".join(chr(c) for c in chars.flatten() if c != 0)
                            names_list.append(s.strip())
                        else:
                            names_list.append(str(r))
                    else:
                        names_list.append("")
                except Exception:
                    names_list.append("")
            else:
                names_list.append(str(ref) if ref is not None else "")

        idx = np.atleast_1d(idx_arr).ravel()
        mpm = np.atleast_1d(mpm_arr).ravel()

        # Vo: struct with dim, mat (may be ref to group)
        Vo = {}
        try:
            vo_g = None
            if isinstance(Vo_ref, h5py.Group):
                vo_g = Vo_ref
            elif isinstance(Vo_ref, np.ndarray) and Vo_ref.size > 0:
                r = Vo_ref.flat[0]
                if isinstance(r, h5py.h5r.Reference):
                    vo_g = f[r]
            if vo_g is not None:
                if "dim" in vo_g:
                    Vo["dim"] = np.asarray(vo_g["dim"]).ravel()
                if "mat" in vo_g:
                    Vo["mat"] = np.asarray(vo_g["mat"])
        except Exception:
            pass
        if "dim" not in Vo:
            Vo["dim"] = np.array([91, 109, 91])
        if "mat" not in Vo:
            Vo["mat"] = np.eye(4)

        # Build a simple namespace that _field() can use
        class _JB:
            pass

        jb = _JB()
        jb.Namen = names_list
        jb.idx = idx
        jb.mpm = mpm
        jb.Vo = type("Vo", (), Vo)()
        jb.lr = _get(g, "lr")

        out["JuBrain"] = jb
    return out


def _fetch_zaborszky_from_github(cache_dir: Path, ref_img: nib.Nifti1Image) -> dict[str, list[Path]] | None:
    """
    Fetch Zaborszky basal forebrain from JuBrain (GitHub). Downloads JuBrain_Data_v30.mat.zip,
    parses the JuBrain struct (idx, mpm, Namen, Vo) and builds Ch1-2 and Ch4 masks. JuBrain
    stores voxel indices and region IDs; we match Namen for Ch1/Ch2/Ch4 and reconstruct 3D masks.
    """
    import zipfile
    from nilearn.image import resample_to_img

    cache_dir = Path(cache_dir)
    zab_dir = cache_dir / "zaborszky"
    zab_dir.mkdir(parents=True, exist_ok=True)

    # JuBrain_Map_v30.nii provides correct MNI affine for mask resampling
    mpm_path = zab_dir / "JuBrain_Map_v30.nii"
    if not mpm_path.exists():
        if _download_to(mpm_path, f"{JUBRAIN_GITHUB_RAW}/JuBrain_Map_v30.nii", "JuBrain MPM"):
            pass  # will use for affine

    mat_zip_path = zab_dir / "JuBrain_Data_v30.mat.zip"
    if not mat_zip_path.exists():
        if not _download_to(mat_zip_path, f"{JUBRAIN_GITHUB_RAW}/JuBrain_Data_v30.mat.zip", "JuBrain Data"):
            return None

    mat_path = zab_dir / "JuBrain_Data_v30.mat"
    if not mat_path.exists():
        try:
            with zipfile.ZipFile(mat_zip_path, "r") as z:
                for name in z.namelist():
                    if name.endswith(".mat"):
                        z.extract(name, zab_dir)
                        src = zab_dir / name
                        if src != mat_path:
                            src.rename(mat_path)
                        break
        except Exception as e:
            print(f"  JuBrain unzip: {e}", file=sys.stderr)
            return None

    data = None
    try:
        from scipy.io import loadmat

        data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        if "v7.3" in str(e).lower() or "hdf" in str(e).lower():
            try:
                data = _load_jubrain_mat73(str(mat_path))
            except Exception as e2:
                print(f"  JuBrain h5py: {e2}", file=sys.stderr)
                return None
        else:
            print(f"  JuBrain loadmat: {e}", file=sys.stderr)
            return None

    if data is None:
        return None

    # JuBrain struct: top-level key is usually variable name from save()
    jb = None
    for key in ("JuBrain", "juBrain", "jubrain"):
        if key in data:
            jb = data[key]
            break
    if jb is None:
        for k, v in data.items():
            if k.startswith("__"):
                continue
            if hasattr(v, "dtype") and hasattr(v.dtype, "names") and v.dtype.names:
                if "Namen" in v.dtype.names and "idx" in v.dtype.names:
                    jb = v
                    break
    if jb is None:
        print("  JuBrain: no JuBrain struct found in .mat", file=sys.stderr)
        return None
    if isinstance(jb, np.ndarray) and jb.ndim == 0:
        jb = jb.item()

    def _field(obj, name: str):
        if hasattr(obj, name):
            return getattr(obj, name)
        if hasattr(obj, "dtype") and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
        return None

    Namen = _field(jb, "Namen")
    idx = _field(jb, "idx")
    mpm = _field(jb, "mpm")
    Vo = _field(jb, "Vo")

    if Namen is None or idx is None or mpm is None or Vo is None:
        print("  JuBrain: missing Namen/idx/mpm/Vo", file=sys.stderr)
        return None

    # Namen can be object array, list, or cell array of strings
    def _get_names(namen):
        out = []
        if isinstance(namen, (list, tuple)):
            for v in namen:
                out.append(str(v).strip() if v is not None else "")
            return out
        arr = np.atleast_1d(namen)
        for i in range(arr.size):
            v = arr.flat[i]
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            out.append(str(v).strip() if v is not None else "")
        return out

    names = _get_names(Namen)
    idx = np.atleast_1d(idx).ravel()
    mpm = np.atleast_1d(mpm).ravel()
    if idx.size != mpm.size:
        print("  JuBrain: idx/mpm size mismatch", file=sys.stderr)
        return None

    # Vo.dim, Vo.mat (affine) — Vo.mat from .mat can be wrong; prefer JuBrain_Map affine
    dim = _field(Vo, "dim")
    if dim is not None:
        dim = np.atleast_1d(dim).ravel()
        if len(dim) >= 3:
            dim = (int(dim[0]), int(dim[1]), int(dim[2]))
        else:
            dim = None
    if dim is None:
        dim = (193, 229, 193)  # JuBrain default
    mpm_path = zab_dir / "JuBrain_Map_v30.nii"
    if mpm_path.exists():
        try:
            mpm_img = nib.load(str(mpm_path))
            aff = np.asarray(mpm_img.affine)
            if dim != mpm_img.shape[:3]:
                dim = mpm_img.shape[:3]
        except Exception:
            aff = np.eye(4)
    else:
        mat = _field(Vo, "mat")
        if mat is not None and np.asarray(mat).size >= 16:
            aff = np.array(mat).reshape(4, 4)
        else:
            # JuBrain MNI 1mm default (fallback if MPM not present)
            aff = np.eye(4)
            aff[:3, 3] = [-96.0, -132.0, -78.0]

    # Find region IDs for Ch1-2, Ch4 (mpm uses 1-based indices into Namen)
    # JuBrain uses "BF (Ch 4)" and "BF (Ch 1-3)" - normalize for matching
    ch12_ids = []
    ch4_ids = []
    for i, n in enumerate(names):
        region_id = i + 1  # MATLAB 1-based
        nlo = n.lower().replace(" ", "")
        if "ch4" in nlo and "ch1" not in nlo and "ch2" not in nlo:
            ch4_ids.append(region_id)
        elif "ch1" in nlo or "ch2" in nlo or "ch1-3" in nlo or "ch1-2" in nlo:
            ch12_ids.append(region_id)

    if not ch12_ids and not ch4_ids:
        print("  JuBrain: no Ch1/Ch2/Ch4 in Namen", file=sys.stderr)
        return None

    found: dict[str, Path] = {}
    lr = _field(jb, "lr")

    def _build_mask(region_ids: list[int]) -> np.ndarray:
        mask = np.zeros(dim, dtype=np.float32)
        sel = np.isin(mpm, region_ids)
        if lr is not None:
            lr_flat = np.atleast_1d(lr).ravel()
            sel = sel  # use both L and R
        lin = idx[sel].astype(int)
        for li in lin:
            li1 = int(li) - 1  # MATLAB 1-based
            if 0 <= li1 < np.prod(dim):
                x, y, z = np.unravel_index(li1, dim)
                mask[x, y, z] = 1.0
        return mask

    if ch12_ids:
        m12 = _build_mask(ch12_ids)
        if np.sum(m12) > 50:
            img = nib.Nifti1Image(m12, aff)
            r = resample_to_img(img, ref_img, interpolation="nearest")
            out_img = nib.Nifti1Image(
                (np.asarray(r.get_fdata()) > 0.5).astype(np.float32), ref_img.affine
            )
            p = zab_dir / "Ch1-2_jubrain.nii.gz"
            nib.save(out_img, str(p))
            found["Ch1-2"] = p

    if ch4_ids:
        m4 = _build_mask(ch4_ids)
        if np.sum(m4) > 50:
            img = nib.Nifti1Image(m4, aff)
            r = resample_to_img(img, ref_img, interpolation="nearest")
            out_img = nib.Nifti1Image(
                (np.asarray(r.get_fdata()) > 0.5).astype(np.float32), ref_img.affine
            )
            p = zab_dir / "Ch4_jubrain.nii.gz"
            nib.save(out_img, str(p))
            found["Ch4"] = p

    if len(found) >= 2:
        out = {k: [v] for k, v in found.items()}
        print(f"  Zaborszky (GitHub/JuBrain): {list(out.keys())}", file=sys.stderr)
        return out
    return None


def _fetch_zaborszky_basal_forebrain(
    cache_dir: Path, ref_img: nib.Nifti1Image | None = None
) -> dict[str, list[Path]] | None:
    """
    Fetch Zaborszky basal forebrain masks (Ch1-2, Ch4). Tries siibra, then GitHub/JuBrain, then manual.
    Returns dict: canonical_name -> [paths] (L/R merged into single paths per name).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Path]] = {}

    # Try siibra (Julich-Brain / EBRAINS) - optional
    try:
        import siibra
        parc = siibra.parcellations.get("julich 2.9")
        space = siibra.spaces.get("mni152")
        for region_name in ("Ch1-2", "Ch4"):
            try:
                reg = parc.get_region(region_name)
                if reg is None and region_name == "Ch1-2":
                    for r in parc.leaf_regions:
                        if "ch1" in r.name.lower() and "ch2" in r.name.lower():
                            reg = r
                            break
                if reg is not None:
                    mask = reg.fetch_regional_map(space, "density")
                    if mask is not None:
                        arr = np.asarray(mask)
                        binary = (arr > 0.4).astype(np.float32)
                        if np.any(binary > 0):
                            p = cache_dir / "zaborszky" / f"{region_name}_siibra.nii.gz"
                            p.parent.mkdir(parents=True, exist_ok=True)
                            aff = getattr(mask, "affine", np.eye(4))
                            img = nib.Nifti1Image(binary, aff)
                            nib.save(img, str(p))
                            out.setdefault(region_name, []).append(p)
            except Exception as e:
                print(f"  siibra {region_name}: {e}", file=sys.stderr)
        if len(out) >= 2:
            print(f"  Zaborszky (siibra): {list(out.keys())}", file=sys.stderr)
            return out
    except ImportError:
        pass
    except Exception as e:
        print(f"  siibra: {e}", file=sys.stderr)

    # Try GitHub/JuBrain (needs ref_img for resampling)
    if ref_img is not None:
        gh_out = _fetch_zaborszky_from_github(cache_dir, ref_img)
        if gh_out is not None:
            return gh_out

    # Manual: atlas_cache/zaborszky/ or atlas_cache/
    for base in [cache_dir / "zaborszky", cache_dir]:
        if not base.exists():
            continue
        for p in base.rglob("*.nii*"):
            if p.stat().st_size < 500:
                continue
            n = p.stem.lower().replace(".gz", "").replace(".nii", "")
            if "ch1" in n and "ch2" in n and "ch4" not in n:
                out.setdefault("Ch1-2", []).append(p)
            elif "ch4" in n and "ch1" not in n and "ch2" not in n:
                out.setdefault("Ch4", []).append(p)
        if len(out) >= 2:
            break
    if len(out) >= 2:
        print(f"  Zaborszky (manual): {list(out.keys())}", file=sys.stderr)
        return out
    return None


NEUDORFER_ZENODO_RECORD = "3903588"
NEUDORFER_ZENODO_BASE = f"https://zenodo.org/records/{NEUDORFER_ZENODO_RECORD}/files"


def _fetch_neudorfer_from_zenodo(cache_dir: Path) -> dict[str, list[Path]] | None:
    """
    Fetch Neudorfer hypothalamus atlas from Zenodo (doi 10.5281/zenodo.3903588).
    Downloads atlas_labels_0.5mm.nii.gz and Volumes_names-labels.csv, extracts LH, TM, PA.
    """
    cache_dir = Path(cache_dir)
    neudorfer_dir = cache_dir / "neudorfer"
    neudorfer_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = neudorfer_dir / "atlas_labels_0.5mm.nii.gz"
    csv_path = neudorfer_dir / "Volumes_names-labels.csv"

    if not atlas_path.exists():
        url = f"{NEUDORFER_ZENODO_BASE}/atlas_labels_0.5mm.nii.gz?download=1"
        if not _download_to(atlas_path, url, "Neudorfer atlas"):
            return None
    if not csv_path.exists():
        url = f"{NEUDORFER_ZENODO_BASE}/Volumes_names-labels.csv?download=1"
        if not _download_to(csv_path, url, "Neudorfer labels"):
            return None

    # Parse CSV: Label, Name, Hemisphere, Abbreviation (Neudorfer Zenodo format)
    import csv
    label_map: dict[str, list[int]] = {}  # canonical -> [label_ids]
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    lab_id = int(row[0])
                except ValueError:
                    continue
                name = row[1].lower() if len(row) > 1 else ""
                if "lateral hypothalamus" in name:
                    label_map.setdefault("LH", []).append(lab_id)
                elif "tuberomammillary" in name:
                    label_map.setdefault("TM", []).append(lab_id)
                elif "paraventricular" in name:
                    label_map.setdefault("PA", []).append(lab_id)
    except Exception as e:
        print(f"  Neudorfer CSV parse: {e}", file=sys.stderr)
        return None

    if len(label_map) < 3:
        return None

    img = nib.load(str(atlas_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    out: dict[str, list[Path]] = {}

    for name, label_ids in label_map.items():
        mask = np.zeros_like(data, dtype=np.float32)
        for lid in label_ids:
            mask[data == lid] = 1
        if np.sum(mask > 0) < 10:
            continue
        out_path = neudorfer_dir / f"{name}_zenodo.nii.gz"
        nib.save(nib.Nifti1Image(mask, img.affine), str(out_path))
        out.setdefault(name, []).append(out_path)

    if len(out) >= 3:
        print(f"  Neudorfer (Zenodo): {list(out.keys())}", file=sys.stderr)
        return out
    return None


def _fetch_neudorfer_hypothalamus(cache_dir: Path) -> dict[str, list[Path]] | None:
    """
    Fetch Neudorfer hypothalamic masks (LH, TM, PA). Tries Zenodo first, else manual.
    Returns dict: canonical_name -> [paths] (L/R merged into single path per name).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try Zenodo auto-fetch
    out = _fetch_neudorfer_from_zenodo(cache_dir)
    if out is not None:
        return out

    # 2. Manual: look for pre-extracted files
    want = {
        "LH": ["lateral_hypothalamus", "lh_l", "lh_r", "lh"],
        "TM": ["tuberomammillary", "tm_l", "tm_r", "tm"],
        "PA": ["paraventricular", "pa_l", "pa_r", "pa"],
    }
    out = {}
    for base in [cache_dir / "neudorfer", cache_dir]:
        if not base.exists():
            continue
        for p in base.rglob("*.nii*"):
            if p.stat().st_size < 500:
                continue
            n = p.stem.lower().replace(".gz", "").replace(".nii", "")
            if "pallidum" in n or "palildum" in n:
                continue
            for label, patterns in want.items():
                if any(pat in n for pat in patterns):
                    out.setdefault(label, []).append(p)
                    break
        if len(out) >= 3:
            break
    if len(out) >= 3:
        print(f"  Neudorfer (manual): {list(out.keys())}", file=sys.stderr)
        return out
    return None


def _merge_extra_masks(
    groups: dict[str, list[Path]], ref_img: nib.Nifti1Image, start: int
) -> tuple[np.ndarray, int]:
    """
    Resample mask groups to ref, merge L+R into single parcels.
    groups: canonical_name -> [paths]
    """
    from nilearn.image import resample_to_img

    ref_shape = ref_img.shape
    combined = np.zeros(ref_shape, dtype=np.int32)
    order = ["Ch1-2", "Ch4", "LH", "TM", "PA"]
    names = [k for k in order if k in groups] + sorted(k for k in groups if k not in order)
    n_added = 0
    for name in names:
        paths = groups[name]
        mask_any = np.zeros(ref_shape, dtype=np.float32)
        for p in paths:
            try:
                img = nib.load(str(p))
                data = np.asarray(img.get_fdata(), dtype=np.float32)
                if np.sum(data > 0) < 1:
                    continue
                r = resample_to_img(img, ref_img, interpolation="nearest")
                d = np.asarray(r.get_fdata(), dtype=np.float32)
                mask_any = np.maximum(mask_any, (d > 0).astype(np.float32))
            except Exception as e:
                print(f"  Skip {p.name}: {e}", file=sys.stderr)
        mask = (mask_any > 0) & (combined == 0)
        if np.any(mask):
            combined[mask] = start + n_added
            n_added += 1
    return combined, n_added


def _canonical_nucleus_name(path: Path) -> str:
    """Base name for grouping L/R: LC_l, LC_r -> LC; PAG -> PAG."""
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    for suffix in ("_l", "_r", "_L", "_R"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _merge_brainstem_nuclei(
    nucleus_paths: list[Path], ref_img: nib.Nifti1Image, start: int = 393
) -> tuple[np.ndarray, int]:
    """
    Resample nucleus masks to ref, merge L+R into single parcels (higher SNR).
    Groups bilateral nuclei (e.g. LC_l + LC_r -> one LC parcel).
    """
    from nilearn.image import resample_to_img

    # Group by canonical name (L/R averaged together)
    groups: dict[str, list[Path]] = {}
    for p in nucleus_paths:
        name = _canonical_nucleus_name(p)
        groups.setdefault(name, []).append(p)

    # Merge reticular formation subdivisions into composites (noisy individually at fMRI resolution)
    RETICULAR_MERGE = {
        "MRt": ["mRt", "mRta", "mRtd", "mRtl", "sMRt", "sMRtl", "sMRtm"],  # medullary reticular
        "iMRt": ["iMRt", "iMRtl", "iMRtm", "isRt"],  # intermediate reticular
    }
    for composite, constituents in RETICULAR_MERGE.items():
        paths = []
        for c in constituents:
            if c in groups:
                paths.extend(groups.pop(c))
        if paths:
            groups[composite] = paths

    # Main nuclei first (VTA, LC, SN, PAG, etc.); rest alphabetical
    priority = ("VTA_PBP", "LC", "SN", "SN1", "SN2", "PAG", "DR", "MnR", "PTg", "LDTg_CGPn", "LPB", "MPB", "PnO_PnC", "RN", "RN1", "RN2", "IC", "SC", "ION", "ROb", "RPa", "RMg", "MRt", "iMRt")
    priority_names = [k for k in priority if k in groups]
    rest = sorted(k for k in groups if k not in priority)
    group_names = priority_names + rest

    ref_shape = ref_img.shape
    combined = np.zeros(ref_shape, dtype=np.int32)
    n_added = 0
    for i, name in enumerate(group_names):
        paths = groups[name]
        mask_any = np.zeros(ref_shape, dtype=np.float32)
        for p in paths:
            try:
                img = nib.load(str(p))
                data = np.asarray(img.get_fdata(), dtype=np.float32)
                if np.sum(data > 0) < 1:
                    continue
                r = resample_to_img(img, ref_img, interpolation="nearest")
                d = np.asarray(r.get_fdata(), dtype=np.float32)
                mask_any = np.maximum(mask_any, (d > 0).astype(np.float32))
            except Exception as e:
                print(f"  Skip {p.name}: {e}", file=sys.stderr)
        mask = (mask_any > 0) & (combined == 0)
        if np.any(mask):
            combined[mask] = start + n_added
            n_added += 1
    return combined, n_added


def _remap_tian_to_361_392(tian_data: np.ndarray) -> np.ndarray:
    """Map Tian S2 unique labels to 361..392 (32 parcels)."""
    uniq = np.unique(np.round(tian_data).astype(int))
    uniq = uniq[uniq > 0]
    if len(uniq) == 0:
        return np.zeros_like(tian_data, dtype=np.int32)
    # Use first 32 labels by value; remap to 361..392
    labels_to_use = sorted(uniq)[: N_SUBCORTICAL_TIAN_S2]
    out = np.zeros_like(tian_data, dtype=np.int32)
    for i, lab in enumerate(labels_to_use):
        out[tian_data == lab] = 361 + i
    return out


def build_glasser_tian_atlas(output_path: Path, cache_dir: Path | None = None) -> None:
    """Build combined Glasser 360 (cortex) + Tian S2 (32 subcortical) in MNI 2mm. Labels 1..360, 361..392."""
    from nilearn.image import resample_to_img
    from nilearn.datasets import fetch_atlas_schaefer_2018

    # 2mm reference grid (Schaefer 2mm)
    ref_bunch = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    ref_img = nib.load(ref_bunch["maps"])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    cache = Path(cache_dir) if cache_dir else (data_dir / "atlas_cache")

    glasser_img = _fetch_glasser_mni(cache)
    if glasser_img is None:
        raise RuntimeError("Failed to fetch Glasser 360. Check network or cache.")

    tian_img = _fetch_tian_s2(cache)
    if tian_img is None:
        raise RuntimeError("Failed to fetch Tian S2. Check network or cache.")

    # Resample Glasser to 2mm reference
    glasser_r = resample_to_img(glasser_img, ref_img, interpolation="nearest")
    glasser_data = np.asarray(glasser_r.get_fdata(), dtype=np.float32)
    # Ensure labels 1..360 (Glasser may already be 1..360)
    uniq_g = np.unique(np.round(glasser_data).astype(int))
    uniq_g = uniq_g[uniq_g > 0]
    if len(uniq_g) > N_CORTICAL_GLASSER:
        # Remap to 1..360 by order
        mapping = {v: i + 1 for i, v in enumerate(sorted(uniq_g)[: N_CORTICAL_GLASSER])}
        out_g = np.zeros_like(glasser_data, dtype=np.int32)
        for old, new in mapping.items():
            out_g[glasser_data == old] = new
        glasser_data = out_g
    else:
        glasser_data = np.round(glasser_data).astype(np.int32)

    combined = np.zeros(ref_shape, dtype=np.float32)
    combined[:] = glasser_data

    # Resample Tian to 2mm, remap to 361..392, fill only where cortex is 0
    tian_r = resample_to_img(tian_img, ref_img, interpolation="nearest")
    tian_data = np.asarray(tian_r.get_fdata(), dtype=np.float32)
    subcort = _remap_tian_to_361_392(tian_data)
    mask = (subcort > 0) & (combined == 0)
    combined[mask] = subcort[mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(combined.astype(np.float32), ref_affine)
    nib.save(out_img, str(output_path))
    print(f"Saved {output_path} ({N_PARCELS_392} parcels: Glasser 360 + Tian S2 32)", file=sys.stderr)


def build_glasser_tian_brainstem_atlas(output_path: Path, cache_dir: Path | None = None) -> None:
    """Build Glasser 360 + Tian S2 + Edlow AAN brainstem. Labels 1..360, 361..392, 393..N."""
    from nilearn.image import resample_to_img
    from nilearn.datasets import fetch_atlas_schaefer_2018

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    cache = Path(cache_dir) if cache_dir else (data_dir / "atlas_cache")

    # Build base 392 first into a temp array (reuse logic)
    ref_bunch = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    ref_img = nib.load(ref_bunch["maps"])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    glasser_img = _fetch_glasser_mni(cache)
    if glasser_img is None:
        raise RuntimeError("Failed to fetch Glasser 360.")
    tian_img = _fetch_tian_s2(cache)
    if tian_img is None:
        raise RuntimeError("Failed to fetch Tian S2.")
    nucleus_paths = _fetch_brainstem_navigator(cache)
    if nucleus_paths is None:
        raise RuntimeError("Failed to find Brainstem Navigator. See instructions above.")

    glasser_r = resample_to_img(glasser_img, ref_img, interpolation="nearest")
    glasser_data = np.asarray(glasser_r.get_fdata(), dtype=np.float32)
    uniq_g = np.unique(np.round(glasser_data).astype(int))
    uniq_g = uniq_g[uniq_g > 0]
    if len(uniq_g) > N_CORTICAL_GLASSER:
        mapping = {v: i + 1 for i, v in enumerate(sorted(uniq_g)[: N_CORTICAL_GLASSER])}
        out_g = np.zeros_like(glasser_data, dtype=np.int32)
        for old, new in mapping.items():
            out_g[glasser_data == old] = new
        glasser_data = out_g
    else:
        glasser_data = np.round(glasser_data).astype(np.int32)

    combined = np.zeros(ref_shape, dtype=np.float32)
    combined[:] = glasser_data

    tian_r = resample_to_img(tian_img, ref_img, interpolation="nearest")
    tian_data = np.asarray(tian_r.get_fdata(), dtype=np.float32)
    subcort = _remap_tian_to_361_392(tian_data)
    mask = (subcort > 0) & (combined == 0)
    combined[mask] = subcort[mask]

    # Add Brainstem Navigator nuclei (393+)
    brainstem, n_bs = _merge_brainstem_nuclei(nucleus_paths, ref_img, start=393)
    mask_bs = (brainstem > 0) & (combined == 0)
    combined[mask_bs] = brainstem[mask_bs]

    n_parcels = N_PARCELS_392 + n_bs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(combined.astype(np.float32), ref_affine)
    nib.save(out_img, str(output_path))
    print(
        f"Saved {output_path} ({n_parcels} parcels: Glasser 360 + Tian S2 32 + Brainstem Navigator {n_bs})",
        file=sys.stderr,
    )


def build_glasser_tian_brainstem_bfb_hyp_atlas(output_path: Path, cache_dir: Path | None = None) -> None:
    """Build Glasser 360 + Tian S2 + Brainstem Navigator + Zaborszky (Ch1-2, Ch4) + Neudorfer (LH, TM, PA)."""
    from nilearn.image import resample_to_img
    from nilearn.datasets import fetch_atlas_schaefer_2018

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    cache = Path(cache_dir) if cache_dir else (data_dir / "atlas_cache")

    ref_bunch = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    ref_img = nib.load(ref_bunch["maps"])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    glasser_img = _fetch_glasser_mni(cache)
    if glasser_img is None:
        raise RuntimeError("Failed to fetch Glasser 360.")
    tian_img = _fetch_tian_s2(cache)
    if tian_img is None:
        raise RuntimeError("Failed to fetch Tian S2.")
    nucleus_paths = _fetch_brainstem_navigator(cache)
    if nucleus_paths is None:
        raise RuntimeError("Failed to find Brainstem Navigator. Run download_brainstem_navigator.py")

    bfb = _fetch_zaborszky_basal_forebrain(cache, ref_img=ref_img)
    if bfb is None:
        raise RuntimeError(
            "Zaborszky basal forebrain not found. Run download_bfb_hyp_atlases.py for instructions."
        )
    hyp = _fetch_neudorfer_hypothalamus(cache)
    if hyp is None:
        raise RuntimeError(
            "Neudorfer hypothalamus not found. Run download_bfb_hyp_atlases.py for instructions."
        )

    # Build base (Glasser + Tian + Brainstem)
    glasser_r = resample_to_img(glasser_img, ref_img, interpolation="nearest")
    glasser_data = np.asarray(glasser_r.get_fdata(), dtype=np.float32)
    uniq_g = np.unique(np.round(glasser_data).astype(int))
    uniq_g = uniq_g[uniq_g > 0]
    if len(uniq_g) > N_CORTICAL_GLASSER:
        mapping = {v: i + 1 for i, v in enumerate(sorted(uniq_g)[: N_CORTICAL_GLASSER])}
        out_g = np.zeros_like(glasser_data, dtype=np.int32)
        for old, new in mapping.items():
            out_g[glasser_data == old] = new
        glasser_data = out_g
    else:
        glasser_data = np.round(glasser_data).astype(np.int32)

    combined = np.zeros(ref_shape, dtype=np.float32)
    combined[:] = glasser_data

    tian_r = resample_to_img(tian_img, ref_img, interpolation="nearest")
    tian_data = np.asarray(tian_r.get_fdata(), dtype=np.float32)
    subcort = _remap_tian_to_361_392(tian_data)
    mask = (subcort > 0) & (combined == 0)
    combined[mask] = subcort[mask]

    brainstem, n_bs = _merge_brainstem_nuclei(nucleus_paths, ref_img, start=393)
    mask_bs = (brainstem > 0) & (combined == 0)
    combined[mask_bs] = brainstem[mask_bs]

    start_extra = 393 + n_bs
    extra_groups = {**bfb, **hyp}
    extra, n_extra = _merge_extra_masks(extra_groups, ref_img, start=start_extra)
    # Allow overlap: BFB/Hyp overwrite Tian/brainstem where they overlap (Ch4, Ch1-2 critical for cholinergic)
    mask_extra = extra > 0
    combined[mask_extra] = extra[mask_extra]

    n_parcels = N_PARCELS_392 + n_bs + n_extra
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(combined.astype(np.float32), ref_affine)
    nib.save(out_img, str(output_path))
    print(
        f"Saved {output_path} ({n_parcels} parcels: Glasser 360 + Tian 32 + Brainstem {n_bs} + BFB/Hyp {n_extra})",
        file=sys.stderr,
    )


def build_combined_atlas(
    output_path: Path,
    cortical_only: bool = False,
) -> None:
    schaefer_img = _fetch_schaefer_2mm()
    schaefer_data = np.asarray(schaefer_img.get_fdata(), dtype=np.float32)
    target_affine = schaefer_img.affine
    target_shape = schaefer_data.shape

    if cortical_only:
        combined = schaefer_data
        n_parcels = N_CORTICAL
    else:
        from nilearn.image import resample_to_img
        combined = schaefer_data.copy()
        n_parcels = N_CORTICAL
        aseg_path = _fetch_aseg_mni()
        if aseg_path is not None:
            aseg_img = nib.load(str(aseg_path))
            aseg_resampled = resample_to_img(
                aseg_img,
                schaefer_img,
                interpolation="nearest",
            )
            aseg_data = np.asarray(aseg_resampled.get_fdata(), dtype=np.int32)
            subcort = _remap_aseg_to_401_414(aseg_data)
            # Only add subcortical where cortex is zero (avoid overwriting Schaefer)
            mask = (subcort > 0) & (combined == 0)
            combined[mask] = subcort[mask]
            n_parcels = N_PARCELS
        else:
            # Fallback: Harvard-Oxford subcortical (first 14 regions -> 401..414)
            ho_img = _fetch_harvard_oxford_sub_2mm()
            if ho_img is not None:
                ho_resampled = resample_to_img(
                    ho_img,
                    schaefer_img,
                    interpolation="nearest",
                )
                ho_data = np.asarray(ho_resampled.get_fdata(), dtype=np.float32)
                subcort = _remap_ho_sub_to_401_414(ho_data)
                # Only add subcortical where cortex is zero (avoid overwriting Schaefer)
                mask = (subcort > 0) & (combined == 0)
                combined[mask] = subcort[mask]
                n_parcels = N_PARCELS
                print("Using Harvard-Oxford subcortical (first 14 regions) for 414 parcels.", file=sys.stderr)
            else:
                print("TemplateFlow aseg and Harvard-Oxford sub not available; saving cortical-only (400 parcels).", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(combined.astype(np.float32), target_affine)
    nib.save(out_img, str(output_path))
    print(f"Saved {output_path} ({n_parcels} parcels)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined cortical + subcortical atlas")
    parser.add_argument(
        "--atlas",
        choices=["schaefer+aseg", "glasser+tian", "glasser+tian+brainstem", "glasser+tian+brainstem+bfb+hyp"],
        default="glasser+tian+brainstem",
        help="Atlas: glasser+tian (392), glasser+tian+brainstem (~450), glasser+tian+brainstem+bfb+hyp (~427), or schaefer+aseg (414). Default: glasser+tian+brainstem",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output NIfTI path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache for downloaded atlases")
    parser.add_argument("--cortical-only", action="store_true", help="Cortical only; only for schaefer+aseg")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cache = args.cache_dir or (repo_root / "data" / "atlas_cache")
    if args.atlas == "glasser+tian":
        default_output = repo_root / "data" / "combined_atlas_392.nii.gz"
    elif args.atlas == "glasser+tian+brainstem":
        default_output = repo_root / "data" / "combined_atlas_450.nii.gz"
    elif args.atlas == "glasser+tian+brainstem+bfb+hyp":
        default_output = repo_root / "data" / "combined_atlas_427.nii.gz"
    else:
        default_output = repo_root / "data" / "combined_atlas_414.nii.gz"
    output_path = Path(args.output) if args.output else default_output
    if not output_path.is_absolute():
        output_path = (repo_root.parent / output_path).resolve()

    if args.atlas == "glasser+tian":
        build_glasser_tian_atlas(output_path, cache_dir=cache)
    elif args.atlas == "glasser+tian+brainstem":
        build_glasser_tian_brainstem_atlas(output_path, cache_dir=cache)
    elif args.atlas == "glasser+tian+brainstem+bfb+hyp":
        build_glasser_tian_brainstem_bfb_hyp_atlas(output_path, cache_dir=cache)
    else:
        build_combined_atlas(output_path, cortical_only=args.cortical_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
