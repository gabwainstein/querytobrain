#!/usr/bin/env python3
"""
Run the full Gene Expression PCA pipeline (Phases 1–4).

Phase 1: Fetch AHBA expression via abagen, filter, standardize.
Phase 2: Full-genome PCA + receptor PCA.
Phase 3: Biological labeling (GO, cell-type, receptor loadings, pc_registry).
Phase 4: Drug-to-PC projection (PDSP Ki).

Outputs go to neurolab/data/gene_pca/ by default. Prerequisites:
- abagen, nilearn, scikit-learn, pandas; optional: gseapy (Phase 3).
- Pipeline atlas: run build_combined_atlas.py first.
- Phase 4: run download_pdsp_ki.py and place KiDatabase.csv in neurolab/data/pdsp_ki/.

See neurolab/docs/implementation/gene_expression_pca_plan.md.

Usage (from repo root):
  python neurolab/scripts/run_gene_pca_pipeline.py
  python neurolab/scripts/run_gene_pca_pipeline.py --output-dir neurolab/data/gene_pca --skip-phase3 --skip-phase4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
scripts_dir = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Gene Expression PCA pipeline (Phases 1–4)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: neurolab/data/gene_pca)")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip biological labeling (no gseapy)")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip drug-to-PC (no PDSP)")
    parser.add_argument("--phase1-args", nargs="*", default=[], help="Extra args for Phase 1 (e.g. --variance-percentile 15)")
    args = parser.parse_args()

    out = args.output_dir or (repo_root / "neurolab" / "data" / "gene_pca")
    out_str = str(out) if out.is_absolute() else str(repo_root / out)

    phases = [
        ("Phase 1: expression", ["run_gene_pca_phase1.py", "--output-dir", out_str] + args.phase1_args),
        ("Phase 2: PCA", ["run_gene_pca_phase2.py", "--output-dir", out_str]),
        ("Phase 3: labeling", ["run_gene_pca_phase3.py", "--output-dir", out_str]),
        ("Phase 4: drugs", ["run_gene_pca_phase4.py", "--output-dir", out_str]),
    ]
    if args.skip_phase3:
        phases[2] = None
    if args.skip_phase4:
        phases[3] = None

    for p in phases:
        if p is None:
            continue
        name, cmd = p
        path = scripts_dir / cmd[0]
        if not path.exists():
            print(f"Skip {name}: {path.name} not found", file=sys.stderr)
            continue
        full_cmd = [sys.executable, str(path)] + cmd[1:]
        print(f"\n{'='*60}\n  {name}\n  {' '.join(full_cmd)}\n{'='*60}")
        r = subprocess.run(full_cmd, cwd=str(repo_root))
        if r.returncode != 0:
            print(f"Pipeline failed at {name} (exit {r.returncode})", file=sys.stderr)
            return r.returncode

    print("\nGene PCA pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
