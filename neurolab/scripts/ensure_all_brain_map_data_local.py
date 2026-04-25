#!/usr/bin/env python3
"""
Ensure all brain map *source* data is present locally before reparcellation/rebuild.

The maps themselves are often already downloaded (e.g. from a previous run);
they are atlas-independent. What we need here is the *source* data (NeuroQuery model,
NeuroSynth DB, NeuroVault NIfTIs, neuromaps NIfTIs, ontologies, atlas NIfTIs) so the
rebuild can *reparcellate* them to the pipeline atlas (Glasser+Tian, 392 parcels).
Any data in Schaefer or other atlases is reparcellated to Glasser+Tian. This script only fetches what's missing.

**Source data checked (all atlas-independent):**
  - NeuroQuery model (decoder; full-brain -> parcellated at build time)
  - NeuroSynth database (neurosynth_data)
  - NeuroVault images (neurovault_curated_data by default; see NeuroVault acquisition guide)
  - Neuromaps annotations (neuromaps_data)
  - Ontologies (ontologies/)
  - Atlas NIfTIs (Glasser 360, Tian S2) for building combined_atlas_392.nii.gz

**Optional (build scripts fetch on first run):** ENIGMA, abagen/AHBA.

Usage:
  python neurolab/scripts/ensure_all_brain_map_data_local.py --check
  python neurolab/scripts/ensure_all_brain_map_data_local.py --download
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "neurolab" / "data"
scripts_dir = repo_root / "neurolab" / "scripts"


def _run(script: str, args: list[str], desc: str) -> bool:
    path = scripts_dir / script
    if not path.exists():
        print(f"  Skip {desc}: {path.name} not found")
        return False
    cmd = [sys.executable, str(path)] + args
    print(f"\n--- {desc} ---\n  {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=str(repo_root))
    return r.returncode == 0


def check_neuroquery() -> bool:
    """NeuroQuery model is cached by neuroquery package (e.g. ~/.cache/neuroquery)."""
    try:
        from neuroquery import fetch_neuroquery_model
        p = fetch_neuroquery_model()
        return Path(p).exists() if p else False
    except Exception:
        return False


def check_neurosynth(data_dir_ns: Path) -> bool:
    """NeuroSynth: NiMARE fetches to data_dir; look for database files."""
    # fetch_neurosynth returns list of paths; typically coordinates and metadata
    d = data_dir_ns / "neurosynth_data"
    if not d.exists():
        return False
    # NiMARE 0.0.3+ stores versioned subdir or files directly
    for pat in ("*.csv", "*.txt", "*.pkl", "neurosynth/*"):
        if list(d.rglob(pat)):
            return True
    return d.exists() and any(d.iterdir())


def check_neurovault(data_dir_nv: Path) -> bool:
    manifest = data_dir_nv / "manifest.json"
    downloads = data_dir_nv / "downloads" / "neurovault"
    return manifest.exists() or (downloads.exists() and any(downloads.iterdir()))


def check_neuromaps(data_dir_nm: Path) -> bool:
    if not data_dir_nm.exists():
        return False
    return any(data_dir_nm.iterdir())


def check_ontologies(ont_dir: Path) -> bool:
    if not ont_dir.exists():
        return False
    return any(ont_dir.glob("*.owl")) or any(ont_dir.glob("*.obo")) or any(ont_dir.glob("*.ttl"))


def check_atlas_cache(cache_dir: Path) -> bool:
    glasser = cache_dir / "glasser360MNI.nii.gz"
    tian = cache_dir / "Tian_Subcortex_S2_3T_1mm.nii.gz"
    return glasser.exists() and tian.exists()


def download_neuroquery() -> bool:
    try:
        from neuroquery import fetch_neuroquery_model
        fetch_neuroquery_model()
        return True
    except Exception as e:
        print(f"  NeuroQuery fetch failed: {e}", file=sys.stderr)
        return False


def download_neurosynth() -> bool:
    try:
        from nimare.extract import fetch_neurosynth
    except ImportError:
        print("  Install nimare: pip install nimare", file=sys.stderr)
        return False
    out = data_dir / "neurosynth_data"
    out.mkdir(parents=True, exist_ok=True)
    try:
        fetch_neurosynth(
            data_dir=str(out),
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
        return True
    except Exception as e:
        print(f"  NeuroSynth fetch failed: {e}", file=sys.stderr)
        return False


def download_atlas_cache() -> bool:
    """Download Glasser 360 and Tian S2 to atlas_cache (same as build_combined_atlas)."""
    cache = data_dir / "atlas_cache"
    cache.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlopen, Request

    def save(url: str, path: Path, name: str) -> bool:
        if path.exists():
            return True
        try:
            req = Request(url, headers={"User-Agent": "NeuroLab-Atlas/1.0"})
            with urlopen(req, timeout=120) as resp:
                if resp.status != 200:
                    return False
                path.write_bytes(resp.read())
            print(f"  Downloaded {name} -> {path}")
            return True
        except Exception as e:
            print(f"  {name} failed: {e}", file=sys.stderr)
            return False

    g_url = "https://github.com/brainspaces/glasser360/raw/master/glasser360MNI.nii.gz"
    t_url = "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_1mm.nii.gz"
    ok = save(g_url, cache / "glasser360MNI.nii.gz", "Glasser 360")
    ok &= save(t_url, cache / "Tian_Subcortex_S2_3T_1mm.nii.gz", "Tian S2")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure all brain map data is local before rebuild")
    ap.add_argument("--check", action="store_true", help="Only report status; do not download")
    ap.add_argument("--download", action="store_true", help="Download/fetch any missing data")
    ap.add_argument("--skip-neurovault", action="store_true", help="Do not run NeuroVault download")
    ap.add_argument("--skip-neuromaps", action="store_true", help="Do not run neuromaps download")
    ap.add_argument("--skip-ontologies", action="store_true", help="Do not run ontology download")
    ap.add_argument("--skip-atlas-cache", action="store_true", help="Do not download Glasser/Tian to atlas_cache")
    ap.add_argument("--max-neurovault-images", type=int, default=0, help="Cap NeuroVault images (0 = no cap)")
    ap.add_argument("--neurovault-collections", type=int, nargs="+", default=[1952], help="NeuroVault collection IDs (default: 1952 BrainPedia)")
    ap.add_argument("--use-curated-neurovault", action="store_true", default=True, help="Use acquisition guide curated list (download_neurovault_curated.py --all); default: True")
    ap.add_argument("--no-curated-neurovault", action="store_false", dest="use_curated_neurovault", help="Use legacy download_neurovault_data.py instead of curated")
    ap.add_argument("--include-neurovault-pharma", action="store_true", help="Also download NeuroVault pharmacological collections (neurovault_pharma_data) for pharma cache")
    args = ap.parse_args()

    if not args.check and not args.download:
        ap.print_help()
        print("\nUse --check to see what is missing, or --download to fetch everything.")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    status = []

    # 1. NeuroQuery
    nq_ok = check_neuroquery()
    status.append(("NeuroQuery model (decoder)", nq_ok))
    if args.download and not nq_ok:
        print("Fetching NeuroQuery model...")
        status[-1] = ("NeuroQuery model (decoder)", download_neuroquery())

    # 2. NeuroSynth
    ns_ok = check_neurosynth(data_dir)
    status.append(("NeuroSynth data (neurosynth_data)", ns_ok))
    if args.download and not ns_ok:
        print("Fetching NeuroSynth data...")
        status[-1] = ("NeuroSynth data (neurosynth_data)", download_neurosynth())

    # 3. NeuroVault (curated acquisition guide preferred)
    nv_curated_dir = data_dir / "neurovault_curated_data"
    nv_legacy_dir = data_dir / "neurovault_data"
    nv_curated_ok = check_neurovault(nv_curated_dir)
    nv_legacy_ok = check_neurovault(nv_legacy_dir)
    nv_ok = nv_curated_ok or nv_legacy_ok
    status.append(("NeuroVault (curated or legacy)", nv_ok))
    if args.download and not args.skip_neurovault and not nv_ok:
        if getattr(args, "use_curated_neurovault", False):
            nv_args = ["--output-dir", str(nv_curated_dir), "--all"]
            if args.max_neurovault_images > 0:
                nv_args += ["--max-images", str(args.max_neurovault_images)]
            status[-1] = ("NeuroVault curated (acquisition guide)", _run("download_neurovault_curated.py", nv_args, "Download NeuroVault curated"))
        else:
            nv_args = ["--output-dir", str(nv_legacy_dir)]
            if args.max_neurovault_images > 0:
                nv_args += ["--max-images", str(args.max_neurovault_images)]
            if args.neurovault_collections:
                nv_args += ["--collections"] + [str(c) for c in args.neurovault_collections]
            status[-1] = ("NeuroVault task maps (neurovault_data)", _run("download_neurovault_data.py", nv_args, "Download NeuroVault"))

    # 3b. NeuroVault pharma (optional, for neurovault_pharma_cache)
    nv_pharma_dir = data_dir / "neurovault_pharma_data"
    nv_pharma_ok = check_neurovault(nv_pharma_dir)
    status.append(("NeuroVault pharma (neurovault_pharma_data)", nv_pharma_ok))
    if args.download and getattr(args, "include_neurovault_pharma", False) and not nv_pharma_ok:
        status[-1] = ("NeuroVault pharma (neurovault_pharma_data)", _run("download_neurovault_pharma.py", ["--output-dir", str(nv_pharma_dir)], "Download NeuroVault pharma"))

    # 4. Neuromaps
    nm_dir = data_dir / "neuromaps_data"
    nm_ok = check_neuromaps(nm_dir)
    status.append(("Neuromaps annotations (neuromaps_data)", nm_ok))
    if args.download and not args.skip_neuromaps and not nm_ok:
        status[-1] = ("Neuromaps annotations (neuromaps_data)", _run("download_neuromaps_data.py", ["--output-dir", str(nm_dir)], "Download neuromaps"))

    # 5. Ontologies
    ont_dir = data_dir / "ontologies"
    ont_ok = check_ontologies(ont_dir)
    status.append(("Ontologies (ontologies/)", ont_ok))
    if args.download and not args.skip_ontologies and not ont_ok:
        status[-1] = ("Ontologies (ontologies/)", _run("download_ontologies.py", ["--output-dir", str(ont_dir), "--clinical", "--extra"], "Download ontologies"))

    # 6. Atlas cache (Glasser + Tian)
    atlas_cache = data_dir / "atlas_cache"
    atlas_ok = check_atlas_cache(atlas_cache)
    status.append(("Atlas cache (Glasser 360 + Tian S2)", atlas_ok))
    if args.download and not args.skip_atlas_cache and not atlas_ok:
        status[-1] = ("Atlas cache (Glasser 360 + Tian S2)", download_atlas_cache())

    # Report
    print("\n" + "=" * 60)
    print("  Source data status (for reparcellation -> target atlas)")
    print("  (Maps are atlas-independent; rebuild reparcellates to Glasser+Tian 392)")
    print("=" * 60)
    all_ok = True
    for name, ok in status:
        mark = "OK" if ok else "MISSING"
        if not ok:
            all_ok = False
        print(f"  [{mark}] {name}")
    print("=" * 60)
    if all_ok:
        print("  All source data present. Rebuild will reparcellate to your atlas:")
        print("    python neurolab/scripts/rebuild_all_caches.py --n-jobs 30")
    else:
        print("  Run with --download to fetch only what's missing, then run rebuild.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
