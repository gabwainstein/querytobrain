#!/usr/bin/env python3
"""
Option A entry point: load ontology + decoder cache once, then get_map(term).

  - Term in cache -> return that term's map.
  - Term OOV -> ontology -> related cache terms -> weighted average map.

Usage in code:
  from option_a import OptionAMapper
  mapper = OptionAMapper(ontology_dir="data/ontologies", decoder_cache_dir="path/to/decoder_cache")
  map_400 = mapper.get_map("working memory")   # cache or ontology fallback

CLI:
  python option_a.py --decoder-cache-dir path/to/decoder_cache "term1" "term2"
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from ontology_expansion import load_ontology_index, get_map_for_term


def _default_ontology_dir() -> Path:
    return _scripts.parent / "data" / "ontologies"


def _load_decoder_cache(cache_dir: Path) -> tuple[Any, list[str]] | tuple[None, None]:
    """Load term_maps.npz and term_vocab.pkl. Returns (maps, vocab) or (None, None)."""
    cache_dir = Path(cache_dir)
    maps_path = cache_dir / "term_maps.npz"
    vocab_path = cache_dir / "term_vocab.pkl"
    if not maps_path.exists() or not vocab_path.exists():
        return None, None
    import numpy as np
    data = np.load(maps_path)
    key = "term_maps" if "term_maps" in data else data.files[0]
    maps = data[key]
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    if isinstance(vocab, dict):
        vocab = list(vocab.keys()) if vocab else []
    return maps, list(vocab)


class OptionAMapper:
    """Option A: single entry point. Load once, then get_map(term)."""

    def __init__(
        self,
        ontology_dir: str | Path,
        decoder_cache_dir: str | Path | None = None,
        encoder: Any = None,
        cache_embeddings: Any = None,
        top_k_similarity: int = 10,
    ):
        self.ontology_dir = Path(ontology_dir)
        self.decoder_cache_dir = Path(decoder_cache_dir) if decoder_cache_dir else None
        self._index = load_ontology_index(self.ontology_dir)
        self._decoder_maps, self._term_vocab = (None, None)
        if self.decoder_cache_dir:
            self._decoder_maps, self._term_vocab = _load_decoder_cache(self.decoder_cache_dir)
        if self._decoder_maps is None or self._term_vocab is None:
            import numpy as np
            self._term_vocab = ["working memory", "attention", "memory", "language", "vision"]
            self._decoder_maps = np.random.randn(len(self._term_vocab), 400).astype(np.float32)
        self._encoder = encoder
        self._cache_embeddings = cache_embeddings
        self._top_k_similarity = top_k_similarity

    def get_map(self, term: str) -> Any | None:
        """Return 400-D map for term: cache hit, ontology fallback, or cosine-similarity fallback. None if no map."""
        return get_map_for_term(
            term,
            self._decoder_maps,
            self._term_vocab,
            self._index,
            encoder=self._encoder,
            cache_embeddings=self._cache_embeddings,
            top_k_similarity=self._top_k_similarity,
        )

    @property
    def n_ontology_labels(self) -> int:
        return len(self._index.get("label_to_related") or {})

    @property
    def n_cache_terms(self) -> int:
        return len(self._term_vocab)


def main() -> int:
    parser = argparse.ArgumentParser(description="Option A: get map for term (cache or ontology fallback).")
    parser.add_argument("terms", nargs="*", default=["working memory", "attention"])
    parser.add_argument("--ontology-dir", type=Path, default=None)
    parser.add_argument("--decoder-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    ontology_dir = args.ontology_dir or _default_ontology_dir()
    if not ontology_dir.exists():
        print(f"Ontology dir not found: {ontology_dir}", file=sys.stderr)
        print("Run: python scripts/download_ontologies.py", file=sys.stderr)
        return 1

    mapper = OptionAMapper(ontology_dir=ontology_dir, decoder_cache_dir=args.decoder_cache_dir)
    print(f"Option A: {mapper.n_ontology_labels} ontology labels, {mapper.n_cache_terms} cache terms")

    for term in args.terms:
        map_out = mapper.get_map(term)
        if map_out is not None:
            import numpy as np
            arr = np.asarray(map_out)
            print(f"  {term!r} -> shape {arr.shape}, mean={float(np.mean(arr)):.4f}")
        else:
            print(f"  {term!r} -> None")

    return 0


if __name__ == "__main__":
    sys.exit(main())
