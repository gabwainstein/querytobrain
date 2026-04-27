#!/usr/bin/env python3
"""
Review KG-to-brain GNN training results and emit a short markdown report.

Reads:
  - <model-dir>/config.json       (best_val_corr, test_corr, hyperparams)
  - <model-dir>/training_log.json (per-epoch loss + val_corr)
  - Optional: <baseline> (a JSON or .txt with prior text_to_brain test_corr) for comparison.

Writes a markdown report to:
  neurolab/docs/implementation/KG_GNN_TRAINING_REVIEW_<YYYY-MM-DD>.md

…and prints the same to stdout.

If <model-dir> does not exist, prints "training has not run yet" and exits 0.

Usage:
  python neurolab/scripts/review_kg_gnn_training.py
  python neurolab/scripts/review_kg_gnn_training.py --model-dir neurolab/data/kg_brain_gnn_model
  python neurolab/scripts/review_kg_gnn_training.py --baseline neurolab/data/embedding_model/config.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
        return None


def _trajectory_summary(log: list[dict]) -> dict:
    if not log:
        return {"epochs": 0, "final_loss": None, "min_loss": None, "best_val_corr": None,
                "best_val_epoch": None, "plateau": False, "diverged": False}
    losses = [e.get("loss_mean", 0.0) for e in log]
    vals = [e.get("val_corr", 0.0) for e in log]
    n = len(log)
    best_v_idx = max(range(n), key=lambda i: vals[i]) if vals else 0
    # plateau heuristic: best val in first half of training
    plateau = best_v_idx < n // 2 and n >= 6
    # divergence heuristic: last 3 losses strictly increasing
    diverged = n >= 3 and losses[-1] > losses[-2] > losses[-3]
    return {
        "epochs": n,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "best_val_corr": vals[best_v_idx] if vals else None,
        "best_val_epoch": best_v_idx + 1 if vals else None,
        "final_val_corr": vals[-1] if vals else None,
        "plateau": plateau,
        "diverged": diverged,
    }


def _baseline_test_corr(baseline_path: str | None) -> float | None:
    if not baseline_path:
        return None
    p = Path(baseline_path)
    if not p.exists():
        return None
    if p.suffix == ".json":
        obj = _load_json(p)
        if isinstance(obj, dict):
            for key in ("test_corr", "test_correlation", "best_test_corr"):
                if key in obj and isinstance(obj[key], (int, float)):
                    return float(obj[key])
    # .txt fallback: first float on a line containing "test"
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "test" in line.lower():
                for tok in line.replace(",", " ").split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def _verdict(test_corr: float | None, baseline: float | None, traj: dict) -> tuple[str, list[str]]:
    """Return (verdict_paragraph, recommendations)."""
    recs: list[str] = []
    if test_corr is None:
        return ("No test correlation recorded; training likely did not complete.",
                ["Re-run train_kg_to_brain_gnn.py and check stderr."])
    if traj["diverged"]:
        recs.append("Loss trajectory diverged in the last 3 epochs — lower --lr (try 1e-4) or raise --weight-decay.")
    if traj["plateau"]:
        recs.append(f"Best val_corr hit at epoch {traj['best_val_epoch']}/{traj['epochs']} — early stopping or fewer --epochs would save compute.")
    if baseline is None:
        recs.append("Baseline (text_to_brain test_corr) not found; pass --baseline neurolab/data/embedding_model/config.json to compare.")
    if test_corr < 0.30:
        verdict = (
            f"**Weak result.** test_corr = {test_corr:.3f}. The GNN is not learning useful structure yet. "
            "Most likely cause is the cold-start hash-based term feature — the contrastive loss can't compensate "
            "without semantic input embeddings."
        )
        recs.extend([
            "Swap term-feature init to PubMedBERT: precompute embeddings with sentence-transformers (NeuML/pubmedbert-base-embeddings), save as .npy aligned with merged_sources term order, pass via --term-embeddings.",
            "Enable --teacher with calibrate_kg_teacher_from_audit.py output (--teacher-weight 0.5).",
            "Increase --num-layers 4 and --epochs 50.",
        ])
    elif baseline is not None and test_corr < baseline:
        delta = baseline - test_corr
        verdict = (
            f"**Underperforms baseline.** test_corr = {test_corr:.3f} vs baseline {baseline:.3f} (Δ = -{delta:.3f}). "
            "The GNN's region readout is smoothing out structure that the direct CBMA-fitting baseline retains."
        )
        recs.extend([
            "Train with --eval-split term in addition to collection — the gap may close on out-of-vocabulary queries (the GNN's actual edge).",
            "Inspect per-collection bucket scores by adding split-aware reporting to the trainer; the GNN may already win on rare-gene/rare-compound queries while losing aggregate.",
            "Enable --teacher and --link-loss-weight 0.3 to lean harder on KG topology.",
        ])
    else:
        verdict = (
            f"**Promising result.** test_corr = {test_corr:.3f}"
            + (f" vs baseline {baseline:.3f} (+{(test_corr - baseline):.3f})" if baseline is not None else "")
            + ". Recommend wiring `kg_weight` into `enrichment/unified_enrichment.py` defaults "
            "(start with `kg_weight=0.3`) and exposing `--kg-gnn` in `scripts/query.py`."
        )
        recs.extend([
            "Run a per-collection-bucket eval to confirm where the GNN wins/loses; document in NEUROLAB_REPO_ANALYSIS.md.",
            "If validated, set --eval-split both in the next training run for a fuller picture.",
        ])
    return verdict, recs


def _markdown(model_dir: Path, config: dict, log: list[dict], traj: dict,
              baseline: float | None, verdict: str, recs: list[str], today: str) -> str:
    lines: list[str] = []
    lines.append(f"# KG-to-Brain GNN Training Review — {today}\n")
    lines.append(f"**Model dir:** `{model_dir}`\n")
    lines.append("## Headline\n")
    lines.append(verdict + "\n")

    lines.append("## Numbers\n")
    lines.append(f"- best_val_corr: **{config.get('best_val_corr', 'n/a'):.4f}**" if isinstance(config.get('best_val_corr'), (int, float)) else "- best_val_corr: n/a")
    lines.append(f"- test_corr: **{config.get('test_corr', 'n/a'):.4f}**" if isinstance(config.get('test_corr'), (int, float)) else "- test_corr: n/a")
    if baseline is not None:
        lines.append(f"- baseline test_corr: {baseline:.4f}")
    lines.append(f"- epochs: {traj['epochs']}")
    lines.append(f"- best epoch: {traj['best_val_epoch']}")
    lines.append(f"- final_loss: {traj['final_loss']:.4f}" if traj['final_loss'] is not None else "- final_loss: n/a")
    lines.append(f"- min_loss: {traj['min_loss']:.4f}" if traj['min_loss'] is not None else "- min_loss: n/a")
    lines.append(f"- plateau: {traj['plateau']}")
    lines.append(f"- diverged: {traj['diverged']}")
    lines.append("")

    lines.append("## Hyperparameters\n")
    for k in ("hidden_dim", "out_dim", "num_layers", "dropout", "lr", "epochs", "eval_split"):
        if k in config:
            lines.append(f"- {k}: {config[k]}")
    lines.append("")

    lines.append("## Recommendations\n")
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=str, default="neurolab/data/kg_brain_gnn_model")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Path to baseline config.json or report .txt with text_to_brain test_corr")
    ap.add_argument("--report-dir", type=str, default="neurolab/docs/implementation")
    ap.add_argument("--report-name", type=str, default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    today = dt.date.today().isoformat()

    if not model_dir.exists():
        print(f"Training has not run yet — {model_dir} does not exist. Nothing to review.")
        return 0

    config = _load_json(model_dir / "config.json") or {}
    log = _load_json(model_dir / "training_log.json") or []
    if not config and not log:
        print(f"Found {model_dir} but no config.json/training_log.json — partial run? Skipping review.")
        return 0

    traj = _trajectory_summary(log)
    baseline = _baseline_test_corr(args.baseline)
    verdict, recs = _verdict(config.get("test_corr"), baseline, traj)
    md = _markdown(model_dir, config, log, traj, baseline, verdict, recs, today)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    name = args.report_name or f"KG_GNN_TRAINING_REVIEW_{today}.md"
    report_path = report_dir / name
    report_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
