#!/usr/bin/env python3
"""
Embed gene names + descriptions for use as Gene-node init features in the
KG-to-brain GNN.

Reads `data/gene_info.json` (dict of `{symbol: {name, locus_group}}`), formats
each entry as `"<symbol>: <name> (<locus_group>)"`, and embeds with the
configured OpenAI model. Saves an `.npy` matrix aligned to the sorted gene
symbol order (matching what `build_heterogeneous_graph.py` uses) plus a vocab
.pkl for verification.

Output:
  <output-dir>/gene_embeddings.npy   (n_genes, dim) float32
  <output-dir>/gene_vocab.pkl        sorted gene symbols
  <output-dir>/gene_embedding_meta.json  model + dim + counts

Usage:
  python neurolab/scripts/embed_genes.py
  python neurolab/scripts/embed_genes.py --output-dir neurolab/data/gene_embeddings --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip("\"' ")


def _format_gene(symbol: str, info: dict) -> str:
    name = (info or {}).get("name") or ""
    locus = (info or {}).get("locus_group") or ""
    parts = [symbol]
    if name:
        parts.append(f": {name}")
    if locus:
        parts.append(f" ({locus})")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-info", type=str, default="neurolab/data/gene_info.json")
    ap.add_argument("--output-dir", type=str, default="neurolab/data/gene_embeddings")
    ap.add_argument("--model", type=str, default="text-embedding-3-large")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dimensions", type=int, default=0,
                    help="Matryoshka dimensions; 0 = full model dim (3072 for text-embedding-3-large)")
    args = ap.parse_args()

    _load_env(repo_root / ".env")

    gene_info_path = Path(args.gene_info)
    if not gene_info_path.exists():
        sys.stderr.write(f"ERROR: {gene_info_path} not found\n")
        sys.exit(1)

    with open(gene_info_path, "r", encoding="utf-8") as fh:
        gene_info = json.load(fh)

    symbols = sorted(gene_info.keys())
    texts = [_format_gene(s, gene_info[s]) for s in symbols]
    n = len(symbols)
    print(f"genes to embed: {n}", flush=True)
    print(f"sample: {texts[0]}", flush=True)

    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("ERROR: pip install openai\n")
        sys.exit(2)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.stderr.write("ERROR: OPENAI_API_KEY not set (.env loaded? key present?)\n")
        sys.exit(2)
    client = OpenAI()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        chunk = texts[start : start + args.batch_size]
        kwargs = {"model": args.model, "input": chunk}
        if args.dimensions and args.dimensions > 0:
            kwargs["dimensions"] = int(args.dimensions)
        try:
            resp = client.embeddings.create(**kwargs)
        except Exception as exc:
            sys.stderr.write(f"ERROR at batch {start}: {exc}\n")
            sys.exit(3)
        embeddings.extend([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
        elapsed = time.time() - t0
        rate = (start + len(chunk)) / max(elapsed, 1e-3)
        eta = (n - start - len(chunk)) / max(rate, 1e-3)
        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= n:
            print(f"  embedded {start + len(chunk)}/{n}  ({rate:.0f} genes/s, ETA {eta:.0f}s)", flush=True)

    matrix = np.stack(embeddings).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    matrix = matrix / norms

    np.save(output_dir / "gene_embeddings.npy", matrix)
    with open(output_dir / "gene_vocab.pkl", "wb") as fh:
        pickle.dump(symbols, fh)
    meta = {
        "model": args.model,
        "dimensions": int(matrix.shape[1]),
        "n_genes": int(matrix.shape[0]),
        "normalized": True,
    }
    with open(output_dir / "gene_embedding_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nSaved {matrix.shape[0]} x {matrix.shape[1]} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
