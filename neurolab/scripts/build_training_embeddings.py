#!/usr/bin/env python3
"""
Pre-compute embeddings for all training terms. Zero API calls during training.

Loads term_vocab from cache (merged_sources or decoder_cache_expanded), batches embed
via OpenAI or sentence-transformers, and saves to embeddings/. Trainer uses
--use-cached-embeddings to load these instead of encoding on-the-fly.

See CRITICAL_PATH_CACHES_SPEC.md.

Usage (from repo root):
  python neurolab/scripts/build_training_embeddings.py --cache-dir neurolab/data/merged_sources --output-dir neurolab/data/embeddings
  python neurolab/scripts/build_training_embeddings.py --cache-dir neurolab/data/merged_sources --encoder openai --model text-embedding-3-large --dimensions 1536 --batch-size 500
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _encode_openai(texts: list[str], model: str, batch_size: int, dimensions: int | None) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        kwargs = {"input": batch, "model": model}
        if dimensions is not None and dimensions > 0:
            kwargs["dimensions"] = dimensions
        r = client.embeddings.create(**kwargs)
        out.extend([d.embedding for d in r.data])
    return np.array(out, dtype=np.float32)


def _encode_sentence_transformers(texts: list[str], model: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    return m.encode(texts, show_progress_bar=True).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute training embeddings (zero API calls during training)")
    parser.add_argument("--cache-dir", default="neurolab/data/merged_sources", help="Cache with term_maps.npz, term_vocab.pkl")
    parser.add_argument("--output-dir", default="neurolab/data/embeddings", help="Output dir for embeddings")
    parser.add_argument("--encoder", choices=["openai", "sentence-transformers"], default="openai")
    parser.add_argument("--model", default="text-embedding-3-large", help="OpenAI model or sentence-transformers model name")
    parser.add_argument("--dimensions", type=int, default=1536, help="OpenAI dimensions (Matryoshka truncation); 0 = full")
    parser.add_argument("--batch-size", type=int, default=500, help="Terms per OpenAI request")
    parser.add_argument("--max-terms", type=int, default=0, help="Cap terms (0 = all)")
    parser.add_argument("--embedding-prefix", default="", help="Prepend to each term (e.g. 'fMRI neuroscience: ')")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if Path(args.cache_dir).is_absolute() else repo_root / args.cache_dir
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else repo_root / args.output_dir

    pkl_path = cache_dir / "term_vocab.pkl"
    if not pkl_path.exists():
        print(f"Cache not found: {pkl_path}", file=sys.stderr)
        return 1

    with open(pkl_path, "rb") as f:
        terms = pickle.load(f)
    terms = list(terms)

    if args.max_terms and len(terms) > args.max_terms:
        terms = terms[: args.max_terms]
        print(f"Capped to {args.max_terms} terms")

    prefix = (args.embedding_prefix or "").strip()
    if prefix:
        terms_to_embed = [prefix + (" " if prefix else "") + (t or "") for t in terms]
    else:
        terms_to_embed = list(terms)

    print(f"Embedding {len(terms)} terms with {args.encoder}/{args.model}...")

    if args.encoder == "openai":
        _env = repo_root / ".env"
        if _env.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(_env)
            except ImportError:
                pass
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set. Set it for OpenAI embeddings (or add to .env).", file=sys.stderr)
            return 1
        dims = args.dimensions if args.dimensions > 0 else None
        embeddings = _encode_openai(terms_to_embed, args.model, args.batch_size, dims)
    else:
        embeddings = _encode_sentence_transformers(terms_to_embed, args.model)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "all_training_embeddings.npy", embeddings)
    with open(out_dir / "embedding_vocab.pkl", "wb") as f:
        pickle.dump(terms, f)

    metadata = {
        "model": args.model,
        "encoder": args.encoder,
        "dim": int(embeddings.shape[1]),
        "n_terms": len(terms),
        "cache_dir": str(cache_dir),
        "embedding_prefix": prefix,
    }
    with open(out_dir / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "model": args.model,
        "dimensions": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
    }
    with open(out_dir / "embedding_config.json", "w") as f:
        json.dump(config, f, indent=2)

    size_mb = embeddings.nbytes / (1024 * 1024)
    print(f"Saved {len(terms)} x {embeddings.shape[1]} -> {out_dir} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
