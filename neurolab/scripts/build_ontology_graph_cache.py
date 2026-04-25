#!/usr/bin/env python3
"""
Build ontology meta-graph cache for query expansion and enrichment.
Queryable meta-graph with bridge edges: CogAt ↔ MONDO ↔ HPO ↔ ChEBI ↔ receptor genes.
Used for training data multiplication and enrichment reports.

Usage:
  python neurolab/scripts/build_ontology_graph_cache.py --ontology-dir neurolab/data/ontologies
  python neurolab/scripts/build_ontology_graph_cache.py --output-dir neurolab/data/ontology_graph_cache
  python neurolab/scripts/build_ontology_graph_cache.py --embed-labels  # add embedding-based bridges

Requires: download_ontologies.py (run with --clinical for full graph), ontology_expansion, networkx.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
_repo_root = _scripts.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ontology meta-graph cache."
    )
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=None,
        help="Ontology files dir (default: neurolab/data/ontologies)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output cache dir (default: neurolab/data/ontology_graph_cache)",
    )
    parser.add_argument(
        "--embed-labels",
        action="store_true",
        help="Compute label embeddings for bridge edges (requires encoder)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Min cosine similarity for bridge edges (default 0.85)",
    )
    args = parser.parse_args()

    try:
        from ontology_expansion import load_ontology_index
        from ontology_meta_graph import build_meta_graph
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print("Run from neurolab/scripts/ or ensure ontology_expansion, ontology_meta_graph are importable.", file=sys.stderr)
        return 1

    root = _scripts.parent
    onto_dir = Path(args.ontology_dir) if args.ontology_dir else root / "data" / "ontologies"
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "ontology_graph_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not onto_dir.exists():
        print(f"Ontology dir not found: {onto_dir}", file=sys.stderr)
        print("Run: python neurolab/scripts/download_ontologies.py --clinical", file=sys.stderr)
        return 1

    print("Loading ontology index...")
    index = load_ontology_index(str(onto_dir))
    label_embeddings = None
    label_list = None
    if args.embed_labels:
        try:
            from embed_ontology_labels import load_or_compute_label_embeddings
            labels = list(index.get("label_to_related", {}).keys())
            emb, labels_out = load_or_compute_label_embeddings(onto_dir, labels)
            if emb is not None and labels_out:
                label_embeddings = emb
                label_list = labels_out
        except ImportError:
            print("--embed-labels requires embed_ontology_labels", file=sys.stderr)

    print("Building meta-graph...")
    G = build_meta_graph(
        index,
        label_embeddings=label_embeddings,
        label_list=label_list,
        similarity_threshold=args.similarity_threshold,
    )

    # Save as pickle (NetworkX graph)
    with open(out_dir / "ontology_meta_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # Save node/edge counts for verification
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "ontology_dir": str(onto_dir),
            "similarity_threshold": args.similarity_threshold,
        }, f, indent=2)

    print(f"Ontology graph cache: {n_nodes} nodes, {n_edges} edges -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
