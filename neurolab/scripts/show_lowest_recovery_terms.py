#!/usr/bin/env python3
"""
Show terms with lowest train recovery (predicted vs real correlation) or test generalization.
Loads split_info.pkl from embedding_model; if train_term_correlations or test_term_correlations
are present, sorts and prints the worst. Run training first to populate these.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Show terms with lowest recovery/generalization correlation")
    parser.add_argument("--model-dir", default="neurolab/data/embedding_model", help="Embedding model dir with split_info.pkl")
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both", help="Show train recovery, test generalization, or both")
    parser.add_argument("-n", "--top", type=int, default=50, help="Number of lowest terms to show (default 50)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    split_path = model_dir / "split_info.pkl"
    if not split_path.exists():
        print(f"split_info.pkl not found at {split_path}", file=sys.stderr)
        print("Run training first: python neurolab/scripts/train_text_to_brain_embedding.py ...", file=sys.stderr)
        return 1

    with open(split_path, "rb") as f:
        split_info = pickle.load(f)

    def _show(corrs: list, title: str) -> None:
        valid = [(t, r) for t, r in corrs if r is not None]
        if not valid:
            print(f"{title}: no correlations available")
            return
        valid.sort(key=lambda x: x[1])
        print(f"\n{title} (lowest {min(args.top, len(valid))} of {len(valid)}):")
        print("-" * 80)
        for t, r in valid[: args.top]:
            print(f"  r={r:7.4f}  {t[:90]}{'...' if len(t) > 90 else ''}")

    if args.mode in ("train", "both"):
        train_corrs = split_info.get("train_term_correlations")
        if train_corrs:
            _show(train_corrs, "Train recovery (lowest predicted vs real correlation)")
        else:
            print("train_term_correlations not in split_info. Re-run training to populate.")

    if args.mode in ("test", "both"):
        test_corrs = split_info.get("test_term_correlations")
        if test_corrs:
            _show(test_corrs, "Test generalization (lowest predicted vs real correlation)")
        else:
            print("test_term_correlations not in split_info. Re-run training to populate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
