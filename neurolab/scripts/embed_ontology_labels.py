#!/usr/bin/env python3
"""
Embed all ontology labels (from every loaded ontology) into a common embedding space
and save to cache. Training and inference can then load this cache instead of re-embedding.

Run from repo root:
  python neurolab/scripts/embed_ontology_labels.py --ontology-dir neurolab/data/ontologies --encoder openai --encoder-model text-embedding-3-large
  python neurolab/scripts/embed_ontology_labels.py --ontology-dir neurolab/data/ontologies --encoder openai --encoder-model text-embedding-3-large
  # Optional: restrict to specific files to save API cost:
  #   --embed-sources cogat.v2.owl mf.owl nbo.owl CogPOver1.owl

Output: neurolab/data/ontology_embeddings/ontology_label_embeddings_{slug}.npy and
        ontology_label_list_{slug}.pkl (slug = encoder + ontology dir hash + rich + optional sources).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

scripts_dir = repo_root / "neurolab" / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


def main() -> int:
    # Load .env from repo root so OPENAI_API_KEY is available (e.g. for --encoder openai)
    _env_path = repo_root / ".env"
    if _env_path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
        except ImportError:
            pass  # optional: pip install python-dotenv

    parser = argparse.ArgumentParser(description="Embed all ontology labels into a common space and cache")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies", help="Directory with OWL/OBO files")
    parser.add_argument("--output-dir", default="neurolab/data/ontology_embeddings", help="Where to save .npy and .pkl")
    parser.add_argument("--encoder", choices=("openai", "sentence-transformers"), default="openai")
    parser.add_argument("--encoder-model", default="text-embedding-3-large", help="e.g. text-embedding-3-large or all-MiniLM-L6-v2")
    parser.add_argument("--embed-sources", nargs="*", default=None, help="Only embed labels from these ontology files (default: all)")
    parser.add_argument("--no-rich-text", action="store_true", help="Embed labels only, not label+synonyms+parents")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ont_dir = Path(args.ontology_dir)
    if not ont_dir.is_absolute():
        ont_dir = repo_root / args.ontology_dir
    if not ont_dir.is_dir():
        print(f"Ontology dir not found: {ont_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from ontology_expansion import load_ontology_index, build_ontology_label_embeddings
    from train_text_to_brain_embedding import get_text_encoder

    print(f"Loading ontologies from {ont_dir}...")
    index = load_ontology_index(str(ont_dir))
    label_to_related = index.get("label_to_related") or {}
    label_to_source = index.get("label_to_source") or {}
    all_labels = sorted(label_to_related.keys())
    if args.embed_sources:
        sources_set = set(args.embed_sources)
        label_list = [l for l in all_labels if label_to_source.get(l, "") in sources_set]
        print(f"Restricted to {len(label_list)} labels from {args.embed_sources}")
    else:
        label_list = all_labels
        print(f"Embedding all {len(label_list)} labels from all ontologies")
    if not label_list:
        print("No labels to embed.", file=sys.stderr)
        return 1

    print(f"Encoder: {args.encoder} / {args.encoder_model}")
    encode_fn, dim = get_text_encoder(args.encoder, args.encoder_model)
    if encode_fn == "tfidf":
        print("TF-IDF not supported for ontology embedding (no fixed-dim vector). Use openai or sentence-transformers.", file=sys.stderr)
        return 1

    use_rich = not args.no_rich_text
    embed_sources = set(args.embed_sources) if args.embed_sources else None
    embeddings, names = build_ontology_label_embeddings(
        index, encode_fn, batch_size=args.batch_size, use_rich_text=use_rich, embed_sources=embed_sources
    )
    assert len(names) == embeddings.shape[0], (len(names), embeddings.shape)

    model_slug = (args.encoder_model or "openai").replace("/", "_")
    ont_hash = hashlib.md5(str(ont_dir.resolve()).encode()).hexdigest()[:8]
    rich_suffix = "_rich" if use_rich else ""
    if embed_sources:
        src_key = hashlib.md5("|".join(sorted(embed_sources)).encode()).hexdigest()[:6]
        cache_slug = f"{model_slug}_{ont_hash}{rich_suffix}_src{src_key}"
    else:
        cache_slug = f"{model_slug}_{ont_hash}{rich_suffix}"

    emb_path = out_dir / f"ontology_label_embeddings_{cache_slug}.npy"
    list_path = out_dir / f"ontology_label_list_{cache_slug}.pkl"
    import numpy as np
    np.save(emb_path, embeddings)
    with open(list_path, "wb") as f:
        pickle.dump(names, f)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1]} -> {emb_path}")
    print(f"Labels -> {list_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
