#!/usr/bin/env python3
"""Build and train gene-only for each gradient PC label style; report abagen/PCs recovery."""
import argparse
import subprocess
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
_data = _repo / "neurolab" / "data"
_scripts = _repo / "neurolab" / "scripts"

STYLES = ["hybrid", "standard", "brain_context", "distinctive", "short", "dominant"]
BASE_ARGS = [
    "--cache-dir", str(_data / "unified_cache"),
    "--no-ontology", "--save-term-sources", "--truncate-to-392",
    "--neurovault-cache-dir", str(_data / "neurovault_cache"),
    "--neuromaps-cache-dir", str(_data / "neuromaps_cache"),
    "--enigma-cache-dir", str(_data / "enigma_cache"),
    "--abagen-cache-dir", str(_data / "abagen_cache"),
    "--additional-abagen-cache-dir", str(_data / "abagen_cache_receptor_residual_selected_denoised"),
    "--max-abagen-terms", "2000", "--abagen-add-gradient-pcs", "3",
    "--abagen-gene-info", str(_data / "gene_info.json"),
    "--abagen-pca-variance", "0.95",
]


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*60}\n{desc}\n{' '.join(cmd)}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(_repo))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="*", default=STYLES, help="Label styles to run")
    ap.add_argument("--skip-build", action="store_true", help="Skip build; only train (caches must exist)")
    ap.add_argument("--skip-train", action="store_true", help="Skip train; only build")
    args = ap.parse_args()

    results = []
    for style in args.styles:
        out_dir = _data / f"merged_sources_gradient_pc_{style}"
        model_dir = _data / f"embedding_model_gene_only_{style}"

        if not args.skip_build:
            build_cmd = [sys.executable, str(_scripts / "build_expanded_term_maps.py"),
                        "--output-dir", str(out_dir), "--gradient-pc-label-style", style] + BASE_ARGS
            if not run(build_cmd, f"Build cache: gradient_pc_{style}"):
                print(f"Build failed for {style}", file=sys.stderr)
                continue

        if not args.skip_train:
            train_cmd = [sys.executable, str(_scripts / "train_text_to_brain_embedding.py"),
                        "--cache-dir", str(out_dir), "--output-dir", str(model_dir),
                        "--encoder", "openai", "--encoder-model", "text-embedding-3-large",
                        "--epochs", "100", "--device", "cuda",
                        "--train-on-source", "abagen", "--no-source-weighted-sampling"]
            if not run(train_cmd, f"Train gene-only: {style}"):
                print(f"Train failed for {style}", file=sys.stderr)
                continue

        report_cmd = [sys.executable, str(_scripts / "report_train_correlation_by_collection.py"),
                      "--model-dir", str(model_dir), "--cache-dir", str(out_dir)]
        run(report_cmd, f"Report: {style}")

    print("\nDone. Compare abagen/PCs train recovery across styles.")


if __name__ == "__main__":
    main()
