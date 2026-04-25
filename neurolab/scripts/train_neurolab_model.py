#!/usr/bin/env python3
"""Neutral training entrypoint for NeuroLab.

This wrapper separates common MLP training runs by data regime while delegating
execution to the existing `train_text_to_brain_embedding.py` script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurolab.training.pipeline_presets import PIPELINES


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NeuroLab training by data regime.")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES.keys()),
        required=True,
        help="Training preset to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args appended after the preset args. Prefix with -- if needed.",
    )
    args = parser.parse_args()

    pipeline = PIPELINES[args.pipeline]
    trainer = REPO_ROOT / "neurolab" / "scripts" / "train_text_to_brain_embedding.py"
    extra = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = [sys.executable, str(trainer), *pipeline.argv(extra)]

    print(f"[train_neurolab_model] pipeline={pipeline.name}")
    print(f"[train_neurolab_model] description={pipeline.description}")
    print("[train_neurolab_model] command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
