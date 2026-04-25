"""
Scope guardrail: flag queries that are in-scope vs out-of-scope for brain-map prediction.

In-scope = text is semantically close to the training vocabulary (neuroscience, cognition,
compounds, brain regions). Out-of-scope = random, unrelated, or nonsensical text (e.g.
"the happy knife of a butterfly") whose predicted map would be meaningless.

Uses the same encoder as the text-to-brain model and compares the query embedding to
precomputed training-term embeddings. If max cosine similarity is below a threshold,
the query is flagged out-of-scope.

Requires guardrail embeddings to exist in the embedding model dir (saved when training
with train_text_to_brain_embedding.py). Optional: pass cache_dir to build them from
term_vocab.pkl if missing.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np


class ScopeGuard:
    """
    Guardrail agent: classify whether a query is in-scope for brain-map prediction.

    Uses embedding similarity to the training term set. check(query) returns
    in_scope, score (max cosine sim to training terms), and a short message.
    """

    EMBEDDINGS_FILE = "training_embeddings.npy"
    TERMS_FILE = "training_terms.pkl"

    def __init__(
        self,
        model_dir: str,
        cache_dir: Optional[str] = None,
        threshold: float = 0.25,
    ):
        """
        model_dir: path to the trained embedding model (config.pkl, encoder, and
            optionally training_embeddings.npy + training_terms.pkl).
        cache_dir: decoder cache dir (term_vocab.pkl) used to build guardrail
            embeddings if they are not already in model_dir.
        threshold: min max-cosine-similarity to training terms to be considered
            in_scope. Tune lower to allow more edge cases, higher to be stricter.
        """
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.threshold = threshold
        self._encoder_type: Optional[str] = None
        self._vectorizer = None
        self._st_model = None
        self._embeddings: np.ndarray = None  # (n_terms, dim)
        self._terms: list = []
        self._load()

    def _load(self) -> None:
        config_path = os.path.join(self.model_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Embedding model config not found at {config_path}. "
                "Train with: python neurolab/scripts/train_text_to_brain_embedding.py --output-dir ..."
            )
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        self._encoder_type = config.get("encoder", "tfidf")
        self._config = config
        self._embedding_prefix = (config.get("embedding_prefix") or "").strip()

        emb_path = os.path.join(self.model_dir, self.EMBEDDINGS_FILE)
        terms_path = os.path.join(self.model_dir, self.TERMS_FILE)

        if os.path.exists(emb_path) and os.path.exists(terms_path):
            self._embeddings = np.load(emb_path).astype(np.float64)
            with open(terms_path, "rb") as f:
                self._terms = pickle.load(f)
            if len(self._terms) != self._embeddings.shape[0]:
                self._embeddings = None
                self._terms = []

        if self._embeddings is None or not self._terms:
            if self.cache_dir:
                self._build_and_save_embeddings(config)
            else:
                raise FileNotFoundError(
                    f"Guardrail embeddings not found in {self.model_dir}. "
                    "Either re-run training (which now saves them) or pass cache_dir to build them."
                )

        # Normalize for cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = self._embeddings / norms

        self._load_encoder()

    def _load_encoder(self) -> None:
        if self._encoder_type == "tfidf":
            path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"TF-IDF vectorizer not found: {path}")
            with open(path, "rb") as f:
                self._vectorizer = pickle.load(f)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("sentence-transformers required for this encoder. pip install sentence-transformers")
            name = self._config.get("model_name", "all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(name)

    def _apply_prefix(self, texts: list[str]) -> list[str]:
        """Prepend embedding prefix to each text if set (same as at training)."""
        if not self._embedding_prefix:
            return list(texts)
        p = self._embedding_prefix.rstrip()
        return [p + (" " if p else "") + (t or "") for t in texts]

    def _encode(self, texts: list[str]) -> np.ndarray:
        texts = self._apply_prefix(texts)
        if self._encoder_type == "tfidf":
            return self._vectorizer.transform(texts).toarray().astype(np.float64)
        return self._st_model.encode(texts, show_progress_bar=False).astype(np.float64)

    def _build_and_save_embeddings(self, config: dict) -> None:
        pkl_path = os.path.join(self.cache_dir, "term_vocab.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Decoder cache vocab not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self._terms = pickle.load(f)
        terms_enc = self._apply_prefix(self._terms)
        if self._encoder_type == "tfidf":
            path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cannot build guardrail: {path} missing")
            with open(path, "rb") as f:
                self._vectorizer = pickle.load(f)
            self._embeddings = self._vectorizer.transform(terms_enc).toarray().astype(np.float64)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("sentence-transformers required")
            name = config.get("model_name", "all-MiniLM-L6-v2")
            st = SentenceTransformer(name)
            self._embeddings = st.encode(terms_enc, show_progress_bar=True).astype(np.float64)
        np.save(os.path.join(self.model_dir, self.EMBEDDINGS_FILE), self._embeddings)
        with open(os.path.join(self.model_dir, self.TERMS_FILE), "wb") as f:
            pickle.dump(self._terms, f)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = self._embeddings / norms

    def check(self, query: str) -> Dict[str, Any]:
        """
        Classify whether the query is in-scope for brain-map prediction.

        Returns:
            in_scope: True if max cosine similarity to training terms >= threshold.
            score: max cosine similarity (0–1 typical for normalized embeddings).
            message: short human-readable reason.
        """
        query = (query or "").strip()
        if not query:
            return {
                "in_scope": False,
                "score": 0.0,
                "message": "Empty query.",
            }
        q = self._encode([query]).reshape(-1).astype(np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm <= 0:
            return {
                "in_scope": False,
                "score": 0.0,
                "message": "Query encoded to zero vector (no overlap with training vocabulary).",
            }
        q = q / q_norm
        sims = self._embeddings @ q  # (n_terms,)
        max_sim = float(np.max(sims))
        in_scope = max_sim >= self.threshold
        if in_scope:
            message = f"In scope (similarity to training terms: {max_sim:.3f})."
        else:
            message = f"Out of scope (max similarity to training terms: {max_sim:.3f} < {self.threshold})."
        return {
            "in_scope": in_scope,
            "score": max_sim,
            "message": message,
        }
