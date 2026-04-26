#!/usr/bin/env python3
"""
Train a text-to-brain embedding model to expand the term space.

**How training works:**
1. Load (term, parcellated_map) pairs from the decoder cache (NeuroQuery or NeuroSynth).
2. Encode each term to a vector: TF-IDF (bag-of-words) or sentence-transformers (semantic).
3. Train a regression head (MLP) that maps text embedding -> n_parcels-D parcellated map (Glasser+Tian, 392 parcels).
   Loss = MSE between predicted and ground-truth map from the cache.
4. Save encoder + head. At inference, any text -> embed -> head -> predicted map.

**More terms:** Use a full decoder cache (build with --max-terms 0). Then run this script
with --max-terms 0 (default) to use all terms. More (term, map) pairs = better generalization.

**Train/val/test:** By default we split into train/val/test (e.g. 80/10/10) with a fixed seed (--seed 42).
We train only on the train split; val is for early monitoring. At the end we report:
- **Train recovery:** mean correlation (predicted vs ground-truth) on *training* terms — can the model recover the maps it was trained on?
- **Test generalization:** mean correlation on *held-out test* terms — can it generalize to unseen terms?
A large gap (train >> test) suggests overfitting; use more data, regularization (--dropout, --weight-decay), or PCA (--pca-components). Optionally --final-retrain-on-train-and-val retrains on train+val and saves that model; test metrics are then for that final model.

**Text encoder:** Default is sentence-transformers with PubMedBERT (NeuML/pubmedbert-base-embeddings)
for best generalization. Use --encoder tfidf for a fast run, or --encoder-model all-MiniLM-L6-v2 for
a smaller model. For --encoder openai, use --embedding-prefix "fMRI neuroscience: " (or similar) to
give the embedder domain context; the prefix is saved in config and applied at inference.

**Improving correlation on unseen terms:**
- More data: build full cache (build_term_maps_cache.py --max-terms 0), then train with --max-terms 0.
- Broader text-to-map: build expanded cache (build_expanded_term_maps.py) to add ontology terms with derived maps; train on --cache-dir decoder_cache_expanded for more (text, map) pairs.
- Stronger encoder: --encoder-model all-mpnet-base-v2 (or biomedical).
- Regularization: --dropout 0.2 --weight-decay 1e-5 to reduce overfitting.
- Train longer: --epochs 40 --lr 5e-4.
- Bigger head: --head-hidden 1024.
- Use all data for final model: --final-retrain-on-train-and-val.
- PCA on brain: --pca-components 100 (fixed K), or --pca-variance 0.8/0.9 (K chosen so that explained variance >= 0.8 or 0.9; train/test in PC space, inverse to n_parcels-D for eval).
- **Text augmentation (optional):** --augment-terms adds rule-based variants (title case, "X task/processing/activation") per term; --augment-with-llm adds LLM paraphrases (OpenAI, cached in cache-dir so you only pay once).

  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model
  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --encoder sentence-transformers --encoder-model NeuML/pubmedbert-base-embeddings --max-terms 0 --epochs 50 --dropout 0.2 --weight-decay 1e-5 --pca-components 100 --early-stopping
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from neurolab.enrichment.term_expansion import expand_abbreviations  # noqa: E402


def _eval_metrics(pred: np.ndarray, targets: np.ndarray) -> tuple:
    """Return (mse, mean_correlation). pred and targets (n, n_parcels)."""
    mse = float(np.mean((pred - targets) ** 2))
    n = pred.shape[0]
    corrs = [np.corrcoef(pred[i], targets[i])[0, 1] for i in range(n) if np.isfinite(pred[i]).all() and np.isfinite(targets[i]).all()]
    corr = float(np.mean(corrs)) if corrs else 0.0
    return mse, corr


def _loss_torch(pred, y, loss_type: str, w=None):
    """Compute loss: mse or cosine (1 - Pearson r). pred, y (B, D). w optional (B,) sample weights."""
    if loss_type == "cosine":
        # Pearson r = cosine similarity of centered vectors
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        r = torch.nn.functional.cosine_similarity(pred_c, y_c, dim=1, eps=1e-8)  # (B,)
        neg_r = 1.0 - r
        if w is not None:
            return (w * neg_r).sum() / w.sum().clamp(min=1e-8)
        return neg_r.mean()
    # MSE
    sq = (pred - y).pow(2).mean(dim=1)
    if w is not None:
        return (w * sq).sum() / w.sum().clamp(min=1e-8)
    return sq.mean()


def _apply_embedding_prefix(terms: list, prefix: str) -> list:
    """Prepend prefix to each term if prefix is non-empty (for OpenAI/domain context)."""
    if not (prefix and prefix.strip()):
        return list(terms)
    p = prefix.rstrip()
    return [p + (" " if p else "") + (t or "") for t in terms]


def get_text_encoder(encoder_type: str, encoder_model: str = "all-MiniLM-L6-v2"):
    """Return (encode_fn, dim). encode_fn(list of str) -> (n, dim)."""
    if encoder_type == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        return "tfidf", None
    if encoder_type == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(encoder_model)
            dim = m.get_sentence_embedding_dimension()
            def encode(texts):
                return m.encode(texts, show_progress_bar=False)
            return encode, dim
        except ImportError:
            print("Install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
            sys.exit(1)
    if encoder_type == "openai":
        try:
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                print("OPENAI_API_KEY is not set. Set it to run with --encoder openai (e.g. export OPENAI_API_KEY=sk-...).", file=sys.stderr)
                sys.exit(1)
            client = OpenAI()
            model = encoder_model or "text-embedding-3-small"
            # Default dim for text-embedding-3-small is 1536
            dim_by_model = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}
            dim = dim_by_model.get(model, 1536)
            batch_size = 100
            def encode(texts):
                if isinstance(texts, str):
                    texts = [texts]
                out = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    r = client.embeddings.create(input=batch, model=model)
                    out.extend([d.embedding for d in r.data])
                return np.array(out, dtype=np.float32)
            return encode, dim
        except ImportError:
            print("Install openai: pip install openai. Set OPENAI_API_KEY for LLM embeddings.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if "api_key" in str(e).lower() or "OPENAI" in type(e).__name__:
                print("OpenAI API key invalid or missing. Set OPENAI_API_KEY (e.g. export OPENAI_API_KEY=sk-...).", file=sys.stderr)
            raise
    raise ValueError(f"Unknown encoder: {encoder_type}")


def _get_llm_paraphrases(
    terms: list,
    model: str,
    cache_path: str,
    max_per_term: int = 2,
) -> dict:
    """Return dict term -> list of paraphrase strings. Load from cache or call OpenAI and save."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") == model and "terms" in data:
                cache = data["terms"]
        except Exception:
            pass
    missing = [t for t in terms if t not in cache or not cache[t]]
    if not missing:
        return {t: cache[t] for t in terms}

    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai. Required for --augment-with-llm.", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY for --augment-with-llm (e.g. export OPENAI_API_KEY=sk-...).", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    batch_size = 25
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        prompt = (
            "For each neuroscience/cognitive term below, output 1–2 short paraphrases (same meaning, a few words each). "
            "One line per paraphrase; separate terms with a blank line. Output only the paraphrases, no numbering.\n\n"
            + "\n".join(batch)
        )
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"OpenAI API error: {e}", file=sys.stderr)
            break
        # Parse: split by blank lines, assign to terms in order (each term gets consecutive non-empty lines)
        lines = [ln.strip() for ln in text.splitlines()]
        idx = 0
        for t in batch:
            paraphrases = []
            while idx < len(lines) and lines[idx]:
                paraphrases.append(lines[idx][:200])
                idx += 1
                if len(paraphrases) >= max_per_term:
                    break
            while idx < len(lines) and lines[idx] == "":
                idx += 1
            cache[t] = paraphrases[:max_per_term]
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "terms": cache}, f, indent=0)
    return {t: cache.get(t, []) for t in terms}


def main():
    parser = argparse.ArgumentParser(description="Train text -> brain embedding (Glasser+Tian, 392 parcels)")
    parser.add_argument("--cache-dir", default="neurolab/data/decoder_cache", help="Decoder cache (term_maps.npz, term_vocab.pkl)")
    parser.add_argument("--output-dir", default="neurolab/data/embedding_model", help="Save encoder + head here")
    parser.add_argument("--encoder", choices=["tfidf", "sentence-transformers", "openai"], default="sentence-transformers", help="Text encoder: tfidf | sentence-transformers | openai (LLM API)")
    parser.add_argument("--encoder-model", default="NeuML/pubmedbert-base-embeddings", help="Model name: sentence-transformers (default PubMedBERT) or openai (e.g. text-embedding-3-small)")
    parser.add_argument("--embedding-prefix", default="", help="Prepend this to each term before encoding (e.g. 'fMRI neuroscience: ' for OpenAI). Saved in config for inference.")
    parser.add_argument("--expand-abbreviations", action="store_true", help="Expand neuro acronyms (e.g. BART -> BART (Balloon Analog Risk Task)) before encoding; improves embeddings. Saved in config for inference.")
    parser.add_argument("--head-hidden", type=int, default=512, help="MLP head hidden size (e.g. 1024 for bigger)")
    parser.add_argument("--head-hidden2", type=int, default=0, help="If >0, add a second hidden layer (dim -> head_hidden -> head_hidden2 -> out); can help for semantic vs spatial nonlinearity")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in MLP head (e.g. 0.2 for better generalization to unseen terms)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay for Adam (e.g. 1e-5); TF-IDF path uses --alpha instead")
    parser.add_argument("--alpha", type=float, default=0.0, help="L2 regularization for TF-IDF MLP (sklearn alpha); try 1e-4 or 1e-3 to reduce overfitting and possibly raise test")
    parser.add_argument("--pca-components", type=int, default=0, help="If >0, fit PCA on train maps and predict this many components (then inverse to 400-D); 0=off unless --pca-variance set.")
    parser.add_argument("--pca-variance", type=float, default=0.0, help="If >0 (e.g. 0.8 or 0.9), reduce cache maps to this fraction of variance; train/test in PC space (predict PC loadings, inverse for eval). Overrides --pca-components.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split (reproducible splits)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of terms for validation (e.g. 0.1)")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of terms for held-out test (e.g. 0.1); reported at end, never used for training")
    parser.add_argument("--no-holdout", action="store_true", help="Train on 100%% of data: no val or test split (memorizer uses all known maps)")
    parser.add_argument("--no-stratified-split", action="store_true", help="Use random split instead of stratified (stratified: sources with <50 terms get >=20%% in test for meaningful per-source metrics)")
    parser.add_argument("--final-retrain-on-train-and-val", action="store_true", help="After training, retrain on train+val and save that model (use whole data except test); then report test metrics")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (e.g. 30–50 for full cache)")
    parser.add_argument("--early-stopping", action="store_true", help="Stop when val correlation does not improve for --patience epochs; restore best model")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping: stop after this many epochs without val improvement (default 10)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-terms", type=int, default=0, help="Cap training terms (0 = use all in cache; more terms = better generalization)")
    parser.add_argument("--augment-terms", action="store_true", help="Add rule-based text variants per term (same map) for more training diversity")
    parser.add_argument("--augment-with-llm", action="store_true", help="Add LLM paraphrases per term (same map); uses OpenAI; cache in cache-dir so you only pay once")
    parser.add_argument("--augment-llm-model", default="gpt-4o-mini", help="OpenAI model for paraphrases (default gpt-4o-mini)")
    parser.add_argument("--augment-llm-max-per-term", type=int, default=2, help="Max LLM paraphrases per term (default 2)")
    parser.add_argument("--no-source-weighted-sampling", action="store_true", help="Disable source-weighted batch sampling. Default: when term_sources present, sample batches so each source has ~target fraction (avoids abagen dominating gradients).")
    parser.add_argument("--uniform-memorizer", action="store_true", help="For memorizer: uniform sample weights (1.0) and uniform sampling. Overrides source weights; use for maximizing train recovery.")
    parser.add_argument("--triad-pairwise-loss", action="store_true", help="Add pairwise aux loss: target brain r from hop+emb_dist+interaction regression; see BUILD_MAPS_AND_TRAINING_PIPELINE.md 14c")
    parser.add_argument("--kg-regression-json", default=None, help="JSON from regress_brain_r_on_hop_and_embedding.py (coefficients for r_target); required if --triad-pairwise-loss")
    parser.add_argument("--ontology-dir", default="neurolab/data/ontologies", help="Ontology dir for KG context (default: neurolab/data/ontologies). Required if --kg-context-hops > 0.")
    parser.add_argument("--kg-context-hops", type=int, default=0, help="If >0, append KG triples (ontology relations) to each term before encoding. 1=direct relations, 2=include 2-hop. Requires --ontology-dir.")
    parser.add_argument("--kg-context-mode", choices=("substring", "semantic"), default="substring", help="At inference: substring = match ontology labels in query text; semantic = embed query and pick top-k ontology nodes by cosine similarity (default substring)")
    parser.add_argument("--kg-context-style", choices=("triples", "natural"), default="triples", help="When semantic: triples = append ' | parent: X | child: Y'; natural = prepend 'X is a type of Y. ...' with sim_floor and max_triples (default triples)")
    parser.add_argument("--kg-semantic-top-k", type=int, default=5, help="When --kg-context-mode semantic: number of closest ontology nodes to use (default 5)")
    parser.add_argument("--kg-sim-floor", type=float, default=0.4, help="When semantic + natural: minimum cosine similarity to include a node (default 0.4)")
    parser.add_argument("--kg-max-triples", type=int, default=15, help="When semantic + natural: max triples to prepend, ranked by query_sim × relation_weight (default 15)")
    parser.add_argument("--no-kg-embed-rich-text", action="store_true", help="When semantic: embed ontology labels only (no synonyms/parents/related). Default is rich text for better retrieval.")
    parser.add_argument("--embed-ontology-sources", nargs="*", default=None, help="When semantic KG: only embed labels from these ontology files (e.g. cogat.v2.owl mf.owl nbo.owl CogPOver1.owl). Saves API cost; MONDO/HPO/ChEBI stay in graph for substring match. Default: embed all.")
    parser.add_argument("--use-ontology-retrieval-augmentation", action="store_true", help="At inference: blend MLP prediction with maps from ontology graph-expanded terms (meta-graph + retrieval). Requires --kg-context-hops and semantic mode and --ontology-retrieval-cache-dir.")
    parser.add_argument("--ontology-retrieval-cache-dir", default=None, help="Expanded cache dir (term_maps.npz + term_vocab.pkl) for retrieval augmentation (e.g. neurolab/data/decoder_cache_expanded)")
    parser.add_argument("--ontology-retrieval-alpha", type=float, default=0.3, help="Max weight for retrieval blend: final = (1-alpha)*MLP + alpha*retrieval (default 0.3)")
    parser.add_argument("--ontology-retrieval-max-hops", type=int, default=2, help="Graph expansion max hops for retrieval (default 2)")
    parser.add_argument("--triad-loss-lambda", type=float, default=0.1, help="Weight for triad pairwise loss: total = (1-lambda)*MSE + lambda*L_pair (default 0.1)")
    parser.add_argument("--use-cached-embeddings", action="store_true", help="Load pre-computed embeddings from --embeddings-dir (zero API calls). Run build_training_embeddings.py first.")
    parser.add_argument("--embeddings-dir", default="neurolab/data/embeddings", help="Dir with all_training_embeddings.npy, embedding_vocab.pkl (when --use-cached-embeddings)")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto", help="Device for PyTorch: cuda (force GPU), cpu (force CPU), auto (default: use cuda if available)")
    parser.add_argument("--gene-pca-variance", type=float, default=0.95, help="When refitting gene PCA (dimension mismatch): target explained variance (default 0.95)")
    parser.add_argument("--loss", choices=("mse", "cosine"), default="mse", help="Loss: mse (default) or cosine (1 - correlation, optimizes the reported metric directly)")
    parser.add_argument("--train-on-source", nargs="*", default=None, help="Train only on terms from these sources (e.g. abagen). Requires term_sources.pkl. Use to optimize label format on genetic data alone.")
    args = parser.parse_args()

    # When using OpenAI encoder, default to an OpenAI model (not the sentence-transformers default)
    if getattr(args, "encoder", None) == "openai" and (
        not args.encoder_model or args.encoder_model == "NeuML/pubmedbert-base-embeddings"
    ):
        args.encoder_model = "text-embedding-3-small"

    # Cached embeddings: disable augmentation (no pre-cached embeddings for variants)
    if getattr(args, "use_cached_embeddings", False) and (getattr(args, "augment_terms", False) or getattr(args, "augment_with_llm", False)):
        print("Note: --use-cached-embeddings with augmentation: augmentation disabled (cache has base terms only).")
        args.augment_terms = False
        args.augment_with_llm = False

    # Load .env from repo root so OPENAI_API_KEY is available (e.g. for --encoder openai)
    if getattr(args, "encoder", None) == "openai" or getattr(args, "augment_with_llm", False):
        _env_path = os.path.join(repo_root, ".env")
        if os.path.isfile(_env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(_env_path)
            except ImportError:
                pass  # optional: pip install python-dotenv
        # Strip CRLF/whitespace from API key (Windows .env often has trailing \r)
        _key = os.environ.get("OPENAI_API_KEY")
        if _key:
            os.environ["OPENAI_API_KEY"] = _key.strip()

    cache_dir = args.cache_dir if os.path.isabs(args.cache_dir) else os.path.join(repo_root, args.cache_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load supervision
    npz_path = os.path.join(cache_dir, "term_maps.npz")
    pkl_path = os.path.join(cache_dir, "term_vocab.pkl")
    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print("Cache not found. Build it first: python neurolab/scripts/build_term_maps_cache.py --cache-dir ...", file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path)
    term_maps = data["term_maps"]  # (n, 400)
    with open(pkl_path, "rb") as f:
        terms = pickle.load(f)
    assert len(terms) == term_maps.shape[0]
    # Optional sample weights by source. Neuromaps/receptor 1.0 (audit: scarce PET maps); neurovault_pharma/pharma_neurosynth 1.2; residual terms 0.6.
    SAMPLE_WEIGHT_BY_SOURCE = {"direct": 1.0, "neurovault": 0.8, "ontology": 0.6, "neuromaps": 1.0, "receptor": 1.0, "neuromaps_residual": 0.6, "receptor_residual": 0.6, "enigma": 0.5, "abagen": 0.4, "reference": 0.6, "neurovault_pharma": 1.2, "pharma_neurosynth": 1.2}
    # Target fraction per source when sampling batches (avoids abagen/gene-dominated gradients). Sum = 1.
    SOURCE_SAMPLING_WEIGHTS = {"direct": 0.30, "neurovault": 0.30, "ontology": 0.15, "abagen": 0.10, "enigma": 0.05, "neuromaps": 0.05, "neuromaps_residual": 0.05, "receptor_residual": 0.03, "pharma_neurosynth": 0.03, "neurovault_pharma": 0.02, "receptor": 0.0, "reference": 0.0}
    # Map types for type-conditioned MLP (fMRI vs structural vs PET). Order fixes one-hot indices.
    MAP_TYPES = ["fmri_activation", "structural", "pet_receptor"]
    SOURCE_TO_MAP_TYPE = {"direct": "fmri_activation", "neurovault": "fmri_activation", "ontology": "fmri_activation", "neurovault_pharma": "fmri_activation", "pharma_neurosynth": "fmri_activation", "neuromaps": "pet_receptor", "receptor": "pet_receptor", "neuromaps_residual": "pet_receptor", "receptor_residual": "pet_receptor", "structural": "structural", "enigma": "structural", "abagen": "pet_receptor", "reference": "pet_receptor"}
    term_sources = None
    term_sample_weights = None
    sources_pkl = os.path.join(cache_dir, "term_sources.pkl")
    weights_pkl = os.path.join(cache_dir, "term_sample_weights.pkl")
    if os.path.exists(sources_pkl):
        with open(sources_pkl, "rb") as f:
            term_sources = pickle.load(f)
        assert len(term_sources) == len(terms)
        print("Sample weights by source:", SAMPLE_WEIGHT_BY_SOURCE)
    if os.path.exists(weights_pkl):
        with open(weights_pkl, "rb") as f:
            term_sample_weights = pickle.load(f)
        if len(term_sample_weights) == len(terms):
            print("Using per-term sample weights from term_sample_weights.pkl (overrides source defaults)")
        else:
            term_sample_weights = None
    # Filter to specific source(s) when --train-on-source (e.g. abagen-only for label optimization)
    train_on_sources = getattr(args, "train_on_source", None) or []
    if train_on_sources and term_sources is not None:
        src_set = set(s.lower() for s in train_on_sources)
        idx = [i for i in range(len(terms)) if (term_sources[i] or "").lower() in src_set]
        if not idx:
            print(f"No terms from sources {train_on_sources}; need term_sources.pkl with those sources.", file=sys.stderr)
            sys.exit(1)
        terms = [terms[i] for i in idx]
        term_maps = term_maps[idx]
        term_sources = [term_sources[i] for i in idx]
        term_sample_weights = [term_sample_weights[i] for i in idx] if term_sample_weights is not None else None
        print(f"Filtered to {len(terms)} terms from sources: {train_on_sources}")

    # Keep originals for LLM augmentation (paraphrases use same map as original term)
    original_terms = list(terms)
    original_maps = term_maps
    original_sources = term_sources
    original_sample_weights = term_sample_weights
    # Optional text augmentation: add variant phrasings per term (same map) before split
    if getattr(args, "augment_terms", False):
        def _text_variants(t: str):
            variants = []
            if not t or not isinstance(t, str):
                return variants
            t = t.strip()
            if not t:
                return variants
            seen = {t}
            # Title case (if different)
            title = t.title()
            if title not in seen:
                variants.append(title)
                seen.add(title)
            # Single-word neuro-style phrasings
            if " " not in t and len(t) > 2:
                for suffix in (" task", " processing", " activation"):
                    phrase = t + suffix
                    if phrase not in seen:
                        variants.append(phrase)
                        seen.add(phrase)
                        if len(variants) >= 3:
                            break
            # Multi-word: add "task: " prefix if short
            if " " in t and len(t) < 50 and "task:" not in t.lower():
                prefixed = f"task: {t}"
                if prefixed not in seen:
                    variants.append(prefixed)
                    seen.add(prefixed)
            return variants[:3]  # at most 3 variants per term
        new_terms = []
        new_maps = []
        new_sources = [] if term_sources is not None else None
        new_weights = [] if term_sample_weights is not None else None
        for i, t in enumerate(terms):
            new_terms.append(t)
            new_maps.append(term_maps[i])
            if new_sources is not None:
                new_sources.append(term_sources[i])
            if new_weights is not None:
                new_weights.append(term_sample_weights[i])
            for v in _text_variants(t):
                new_terms.append(v)
                new_maps.append(term_maps[i])
                if new_sources is not None:
                    new_sources.append(term_sources[i])
                if new_weights is not None:
                    new_weights.append(term_sample_weights[i])
        terms = new_terms
        term_maps = np.vstack(new_maps).astype(term_maps.dtype)
        term_sources = new_sources
        term_sample_weights = new_weights
        print(f"Augmented terms: {len(terms)} entries (variants map to same brain map)")
    if getattr(args, "augment_with_llm", False):
        cache_path = os.path.join(cache_dir, "augment_llm_cache.json")
        paraphrases = _get_llm_paraphrases(
            original_terms,
            model=args.augment_llm_model,
            cache_path=cache_path,
            max_per_term=getattr(args, "augment_llm_max_per_term", 2),
        )
        added = 0
        for i, t in enumerate(original_terms):
            for p in paraphrases.get(t, []):
                if p and p.strip() and p.strip() != t:
                    terms.append(p.strip())
                    term_maps = np.vstack([term_maps, original_maps[i : i + 1].astype(term_maps.dtype)])
                    if term_sources is not None and original_sources is not None:
                        term_sources.append(original_sources[i])
                    if term_sample_weights is not None and original_sample_weights is not None:
                        term_sample_weights.append(original_sample_weights[i])
                    added += 1
        if added:
            print(f"Augmented with LLM paraphrases: +{added} entries (cached at {cache_path})")
    if args.max_terms and len(terms) > args.max_terms:
        idx = np.random.default_rng(42).choice(len(terms), args.max_terms, replace=False)
        term_maps = term_maps[idx]
        terms = [terms[i] for i in idx]
        if term_sources is not None:
            term_sources = [term_sources[i] for i in idx]
        if term_sample_weights is not None:
            term_sample_weights = [term_sample_weights[i] for i in idx]
    n = len(terms)
    n_parcels = term_maps.shape[1]
    # Map type per term (for type-conditioned MLP: fmri / structural / pet)
    term_map_types = None
    map_types_pkl = os.path.join(cache_dir, "term_map_types.pkl")
    if os.path.exists(map_types_pkl):
        with open(map_types_pkl, "rb") as f:
            term_map_types = pickle.load(f)
        if len(term_map_types) != n:
            term_map_types = None
    if term_map_types is None and term_sources is not None:
        term_map_types = [SOURCE_TO_MAP_TYPE.get(s, "fmri_activation") for s in term_sources]
    if term_map_types is None:
        term_map_types = ["fmri_activation"] * n
    type_indices = [MAP_TYPES.index(t) if t in MAP_TYPES else 0 for t in term_map_types]
    type_one_hot = np.eye(len(MAP_TYPES), dtype=np.float32)[type_indices]  # (n, num_map_types)
    print(f"Data: {n} terms x {n_parcels} parcels; map types: {len(MAP_TYPES)} ({', '.join(MAP_TYPES)})")

    # Train / val / test split (reproducible; test held out for generalization eval)
    # With --no-holdout: all data for training, no val/test
    rng = np.random.default_rng(args.seed)
    if getattr(args, "no_holdout", False):
        train_idx = np.arange(n, dtype=int)
        val_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)
        print("No holdout: training on 100% of data")
    elif term_sources is not None and not getattr(args, "no_stratified_split", False):
        from collections import defaultdict
        import math
        # Force gene expression gradient terms (various label styles) to always be in train
        def _is_gradient_pc(t):
            s = str(t).replace("Gene:", "").strip()
            return bool(re.search(r"Gene expression gradient|Principal cortical gradient|Cortical gene expression gradient|Dominant gene expression axis", s, re.I) or re.search(r"^(Sensorimotor-association|Metabolic and oxidative|Developmental and synaptic|Cognitive metabolism|Developmental plasticity) (gradient|axis)", s, re.I) or re.search(r"Primary cortex gradient|Metabolic gradient:|Developmental gradient:", s, re.I))
        gradient_pc_indices = [i for i in range(n) if terms[i] and _is_gradient_pc(terms[i])]
        idx_by_source = defaultdict(list)
        for i in range(n):
            if i not in gradient_pc_indices:
                idx_by_source[term_sources[i]].append(i)
        test_idx_list, val_idx_list, train_idx_list = [], [], []
        for src, indices in sorted(idx_by_source.items()):
            n_s = len(indices)
            perm_s = rng.permutation(n_s)
            idx_arr = np.array(indices)[perm_s]
            if n_s < 50:
                n_test_s = max(1, math.ceil(n_s * 0.2))
                n_val_s = max(0, min(n_s - n_test_s, math.ceil(n_s * args.val_frac)))
            else:
                n_test_s = max(0, int(n_s * args.test_frac))
                n_val_s = max(0, min(n_s - n_test_s, int(n_s * args.val_frac)))
            n_train_s = n_s - n_test_s - n_val_s
            test_idx_list.extend(idx_arr[:n_test_s].tolist())
            val_idx_list.extend(idx_arr[n_test_s : n_test_s + n_val_s].tolist())
            train_idx_list.extend(idx_arr[n_test_s + n_val_s :].tolist())
        train_idx_list = gradient_pc_indices + train_idx_list
        test_idx = np.array(test_idx_list, dtype=int)
        val_idx = np.array(val_idx_list, dtype=int)
        train_idx = np.array(train_idx_list, dtype=int)
        if gradient_pc_indices:
            print("Stratified split: gene expression gradient terms forced to train; sources with <50 terms get >=20%% in test")
        else:
            print("Stratified split: sources with <50 terms get >=20%% in test")
    else:
        perm = rng.permutation(n)
        n_test = max(0, int(n * args.test_frac))
        n_val = max(1, int(n * args.val_frac)) if n - n_test > 0 else 0
        test_idx = perm[:n_test] if n_test else np.array([], dtype=int)
        val_idx = perm[n_test : n_test + n_val] if n_val else np.array([], dtype=int)
        train_idx = perm[n_test + n_val :]
    if len(train_idx) < 10:
        print("Too few terms after split; reduce --test-frac and/or --val-frac", file=sys.stderr)
        sys.exit(1)
    train_terms = [terms[i] for i in train_idx]
    train_maps = term_maps[train_idx]
    val_terms = [terms[i] for i in val_idx] if len(val_idx) else []
    val_maps = term_maps[val_idx] if len(val_idx) else np.zeros((0, n_parcels))
    test_terms = [terms[i] for i in test_idx] if len(test_idx) else []
    test_maps = term_maps[test_idx] if len(test_idx) else np.zeros((0, n_parcels))
    train_weights = None
    if getattr(args, "uniform_memorizer", False):
        train_weights = None  # uniform (no per-term weighting)
        print("Uniform memorizer: no sample weights (all terms equal)")
    elif term_sample_weights is not None:
        train_weights = np.array([term_sample_weights[i] for i in train_idx], dtype=np.float32)
        print(f"Sample weights (per-term): train min={train_weights.min():.2f} max={train_weights.max():.2f} mean={train_weights.mean():.2f}")
    elif term_sources is not None:
        train_weights = np.array([SAMPLE_WEIGHT_BY_SOURCE.get(term_sources[i], 0.5) for i in train_idx], dtype=np.float32)
        print(f"Sample weights (by source): train min={train_weights.min():.2f} max={train_weights.max():.2f} mean={train_weights.mean():.2f}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Source-weighted batch sampling: P(draw sample i) ∝ SOURCE_SAMPLING_WEIGHTS[s] / n_s so batches have ~target source mix
    sampling_probs = None
    if getattr(args, "uniform_memorizer", False):
        sampling_probs = None  # uniform random sampling
        print("Uniform memorizer: uniform batch sampling")
    elif term_sources is not None and not getattr(args, "no_source_weighted_sampling", False):
        from collections import Counter
        train_sources = [term_sources[i] for i in train_idx]
        n_per_source = Counter(train_sources)
        probs = np.zeros(len(train_idx), dtype=np.float64)
        for i in range(len(train_idx)):
            s = term_sources[train_idx[i]]
            w = SOURCE_SAMPLING_WEIGHTS.get(s, 0.0)
            n_s = max(n_per_source[s], 1)
            probs[i] = w / n_s
        if probs.sum() > 0:
            probs /= probs.sum()
            sampling_probs = probs
            print("Source-weighted sampling: batches drawn with target source mix (abagen ~10%%, direct+neurovault ~60%%)")

    # Optional gene-only PCA: abagen terms use a separate head that predicts PC loadings (inverse to n_parcels-D at inference)
    gene_pca = None
    gene_loadings = None
    abagen_term_indices = None
    abagen_set = None
    abagen_idx_to_loading_row = None
    n_gene_out = 0
    gene_pca_path = os.path.join(cache_dir, "gene_pca.pkl")
    gene_loadings_path = os.path.join(cache_dir, "gene_loadings.npz")
    abagen_indices_path = os.path.join(cache_dir, "abagen_term_indices.pkl")
    if os.path.exists(abagen_indices_path) and term_sources is not None:
        with open(abagen_indices_path, "rb") as f:
            abagen_term_indices = pickle.load(f)
        abagen_set = set(abagen_term_indices)
        abagen_idx_to_loading_row = {idx: j for j, idx in enumerate(abagen_term_indices)}
        need_refit = False
        if os.path.exists(gene_pca_path) and os.path.exists(gene_loadings_path):
            with open(gene_pca_path, "rb") as f:
                gene_pca = pickle.load(f)
            gene_n_features = getattr(gene_pca, "n_features_in_", getattr(gene_pca, "n_features_", None))
            if gene_n_features is not None and gene_n_features != n_parcels:
                need_refit = True
                print(f"Gene PCA dimension mismatch: PCA has {gene_n_features} features, cache has {n_parcels} parcels. Refitting gene PCA on abagen maps...")
        if need_refit or not os.path.exists(gene_pca_path) or not os.path.exists(gene_loadings_path):
            from sklearn.decomposition import PCA
            from neurolab.parcellation import zscore_cortex_subcortex_separately
            gene_maps = term_maps[abagen_term_indices].astype(np.float64)
            for i in range(gene_maps.shape[0]):
                gene_maps[i] = zscore_cortex_subcortex_separately(gene_maps[i])
            n_abagen = gene_maps.shape[0]
            if n_abagen >= 2:
                max_k = min(n_abagen - 1, n_parcels)
                gene_pca_var = getattr(args, "gene_pca_variance", 0.95)
                pca_full = PCA(n_components=max_k, random_state=42)
                pca_full.fit(gene_maps)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                k = int(np.searchsorted(cumvar, gene_pca_var)) + 1
                k = min(max(1, k), max_k)
                gene_pca = PCA(n_components=k, random_state=42)
                gene_pca.fit(gene_maps)
                gene_loadings = gene_pca.transform(gene_maps).astype(np.float32)
                with open(gene_pca_path, "wb") as f:
                    pickle.dump(gene_pca, f)
                np.savez_compressed(gene_loadings_path, loadings=gene_loadings)
                n_gene_out = gene_pca.n_components_
                var_expl = gene_pca.explained_variance_ratio_.sum()
                print(f"Gene head: {n_gene_out} PC components for {len(abagen_term_indices)} abagen terms (refit, variance {var_expl:.3f}); saved to cache")
            else:
                gene_pca = None
                gene_loadings = None
                abagen_set = set()
                abagen_idx_to_loading_row = {}
                n_gene_out = 0
                print("Too few abagen samples for gene PCA; abagen terms use main head.")
        else:
            gene_loadings = np.load(gene_loadings_path)["loadings"]
            n_gene_out = gene_pca.n_components_
            print(f"Gene head: {n_gene_out} PC components for {len(abagen_term_indices)} abagen terms (predict loadings, inverse at inference)")

    # Optional PCA on brain maps (fit on train only; head predicts PC loadings, we inverse for eval)
    pca = None
    n_out = n_parcels
    if getattr(args, "pca_variance", 0) > 0:
        from sklearn.decomposition import PCA
        max_comp = min(train_maps.shape[0] - 1, n_parcels)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(train_maps)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, args.pca_variance)) + 1
        n_comp = min(max(1, n_comp), max_comp)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(train_maps)
        train_targets = pca.transform(train_maps).astype(np.float32)
        n_out = n_comp
        var_expl = pca.explained_variance_ratio_.sum()
        print(f"PCA (variance {args.pca_variance:.2f}): {n_comp} components (explained variance: {var_expl:.3f})")
    elif getattr(args, "pca_components", 0) > 0:
        from sklearn.decomposition import PCA
        n_comp = min(args.pca_components, train_maps.shape[0] - 1, n_parcels)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(train_maps)
        train_targets = pca.transform(train_maps).astype(np.float32)
        n_out = n_comp
        var_expl = pca.explained_variance_ratio_.sum()
        print(f"PCA: {n_comp} components (explained variance: {var_expl:.3f})")
        if var_expl < 0.9:
            print("  Note: <90% variance explained; consider --pca-variance 0.9 or higher --pca-components for full brain rank.")
    else:
        train_targets = train_maps.astype(np.float32)

    # Text encoder
    if args.encoder == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPRegressor
        embedding_prefix_tfidf = (getattr(args, "embedding_prefix", None) or "").strip()
        if embedding_prefix_tfidf:
            train_terms_enc = _apply_embedding_prefix(train_terms, embedding_prefix_tfidf)
            val_terms_enc = _apply_embedding_prefix(val_terms, embedding_prefix_tfidf)
            terms_enc = _apply_embedding_prefix(terms, embedding_prefix_tfidf)
            print(f"TF-IDF with embedding prefix: {repr(embedding_prefix_tfidf)}")
        else:
            train_terms_enc, val_terms_enc, terms_enc = train_terms, val_terms, terms
        vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
        X_train = vectorizer.fit_transform(train_terms_enc).toarray()
        X_val = vectorizer.transform(val_terms_enc).toarray()
        dim = X_train.shape[1]
        print(f"TF-IDF dim: {dim}")
        reg = MLPRegressor(hidden_layer_sizes=(args.head_hidden,), max_iter=args.epochs, learning_rate_init=args.lr, alpha=args.alpha, random_state=42)
        reg.fit(X_train, train_targets, sample_weight=train_weights if train_weights is not None else None)
        # Train recovery (can model recover maps it was trained on?) — eval in n_parcels-D
        train_pred_pc = reg.predict(X_train)
        train_pred = pca.inverse_transform(train_pred_pc) if pca is not None else train_pred_pc
        _, train_corr = _eval_metrics(train_pred, train_maps)
        print(f"Train recovery (mean corr on training terms): {train_corr:.4f}")
        if len(val_terms) > 0:
            val_pred_pc = reg.predict(X_val)
            val_pred = pca.inverse_transform(val_pred_pc) if pca is not None else val_pred_pc
            mse, corr = _eval_metrics(val_pred, val_maps)
            print(f"Val MSE: {mse:.4f}, mean correlation: {corr:.4f}")
        # Optional: retrain on train+val (whole data except test), then save
        if args.final_retrain_on_train_and_val and (len(train_terms) + len(val_terms)) > len(train_terms):
            train_val_terms = train_terms + val_terms
            train_val_maps = np.vstack([train_maps, val_maps])
            if pca is not None:
                from sklearn.decomposition import PCA as PCAClass
                pca = PCAClass(n_components=n_out, random_state=42)
                pca.fit(train_val_maps)
                train_val_targets = pca.transform(train_val_maps).astype(np.float32)
            else:
                train_val_targets = train_val_maps.astype(np.float32)
            train_val_terms_enc = _apply_embedding_prefix(train_val_terms, embedding_prefix_tfidf) if embedding_prefix_tfidf else train_val_terms
            vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1)
            X_train_val = vectorizer.fit_transform(train_val_terms_enc).toarray()
            train_val_idx = list(train_idx) + list(val_idx)
            train_val_weights = np.array([term_sample_weights[i] for i in train_val_idx], dtype=np.float32) if term_sample_weights is not None else (np.array([SAMPLE_WEIGHT_BY_SOURCE.get(term_sources[i], 0.5) for i in train_val_idx], dtype=np.float32) if term_sources is not None else None)
            reg = MLPRegressor(hidden_layer_sizes=(args.head_hidden,), max_iter=args.epochs, learning_rate_init=args.lr, alpha=args.alpha, random_state=42)
            reg.fit(X_train_val, train_val_targets, sample_weight=train_val_weights)
            print("Retrained on train+val (whole data except test).")
        # Save
        with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(output_dir, "head_mlp.pkl"), "wb") as f:
            pickle.dump(reg, f)
        if pca is not None:
            with open(os.path.join(output_dir, "pca.pkl"), "wb") as f:
                pickle.dump(pca, f)
        with open(os.path.join(output_dir, "config.pkl"), "wb") as f:
            pickle.dump({"encoder": "tfidf", "n_parcels": n_parcels, "pca_components": n_out if pca is not None else 0, "embedding_prefix": embedding_prefix_tfidf}, f)
        # Guardrail: embeddings of all terms for scope check
        X_all = vectorizer.transform(terms_enc).toarray().astype(np.float64)
        np.save(os.path.join(output_dir, "training_embeddings.npy"), X_all)
        with open(os.path.join(output_dir, "training_terms.pkl"), "wb") as f:
            pickle.dump(terms, f)
        # Test set evaluation (generalization to unseen terms) — in n_parcels-D
        test_corr = None
        if len(test_terms) > 0:
            test_terms_enc = _apply_embedding_prefix(test_terms, embedding_prefix_tfidf) if embedding_prefix_tfidf else test_terms
            X_test = vectorizer.transform(test_terms_enc).toarray()
            test_pred_pc = reg.predict(X_test)
            test_pred = pca.inverse_transform(test_pred_pc) if pca is not None else test_pred_pc
            test_mse, test_corr = _eval_metrics(test_pred, test_maps)
            print(f"Test generalization (mean corr on held-out terms): {test_corr:.4f} (MSE: {test_mse:.4f})")
        # Summary: recovery vs generalization
        print(f"Summary: Train recovery = {train_corr:.4f}" + (f", Test generalization = {test_corr:.4f}" if test_corr is not None else "") + (f", gap = {train_corr - test_corr:.4f}" if test_corr is not None else ""))
        # Save split info for reproducibility
        with open(os.path.join(output_dir, "split_info.pkl"), "wb") as f:
            pickle.dump({"train_terms": train_terms, "val_terms": val_terms, "test_terms": test_terms, "seed": args.seed}, f)
        print(f"Saved to {output_dir} (split_info.pkl, guardrail embeddings)")
        return

    if args.encoder in ("sentence-transformers", "openai"):
        embedding_prefix = (getattr(args, "embedding_prefix", None) or "").strip()
        use_cached = getattr(args, "use_cached_embeddings", False)
        embeddings_dir = getattr(args, "embeddings_dir", None) or "neurolab/data/embeddings"
        if use_cached:
            emb_dir = Path(embeddings_dir) if Path(embeddings_dir).is_absolute() else Path(repo_root) / embeddings_dir
            emb_npy = emb_dir / "all_training_embeddings.npy"
            emb_pkl = emb_dir / "embedding_vocab.pkl"
            if not emb_npy.exists() or not emb_pkl.exists():
                print(f"Cached embeddings not found: {emb_npy} or {emb_pkl}. Run build_training_embeddings.py first.", file=sys.stderr)
                sys.exit(1)
            emb_loaded = np.load(emb_npy).astype(np.float32)
            with open(emb_pkl, "rb") as f:
                emb_vocab = pickle.load(f)
            emb_vocab = list(emb_vocab)
            vocab_to_idx = {t: i for i, t in enumerate(emb_vocab)}
            missing = [t for t in terms if t not in vocab_to_idx]
            if missing:
                print(f"Terms not in embedding cache: {len(missing)} missing. Ensure build_training_embeddings.py was run with same --cache-dir.", file=sys.stderr)
                sys.exit(1)
            emb_rows = np.array([vocab_to_idx[t] for t in terms])
            all_emb = emb_loaded[emb_rows]
            dim = all_emb.shape[1]
            X_train = all_emb[train_idx]
            X_val = all_emb[val_idx]
            X_test_pre = all_emb[test_idx] if len(test_idx) else np.zeros((0, dim), dtype=np.float32)
            encoder_dim = dim
            X_train = np.hstack([X_train, type_one_hot[train_idx]]).astype(np.float32)
            X_val = np.hstack([X_val, type_one_hot[val_idx]]).astype(np.float32)
            X_test_pre = np.hstack([X_test_pre, type_one_hot[test_idx]]).astype(np.float32)
            dim = encoder_dim + len(MAP_TYPES)
            kg_context_hops = 0
            expand_abbrev = False
            ontology_dir_resolved = None
            print(f"Using cached embeddings: {all_emb.shape[0]} terms x {encoder_dim} dim from {emb_dir}")
        else:
            encode_fn_raw, dim = get_text_encoder(args.encoder, args.encoder_model)

        if not use_cached:
            expand_abbrev = getattr(args, "expand_abbreviations", False)
            kg_context_hops = max(0, int(getattr(args, "kg_context_hops", 0)))
            kg_index = None
            ontology_dir_resolved = None
            kg_semantic = (getattr(args, "kg_context_mode", "substring") or "substring").strip().lower() == "semantic"
            kg_label_embeddings = None
            kg_label_list = None
            if kg_context_hops > 0:
                ont_dir = getattr(args, "ontology_dir", None) or ""
                if not (ont_dir and os.path.isabs(ont_dir)):
                    ont_dir = os.path.join(repo_root, ont_dir) if ont_dir else os.path.join(repo_root, "neurolab", "data", "ontologies")
                if os.path.isdir(ont_dir):
                    scripts_dir = os.path.join(repo_root, "neurolab", "scripts")
                    if scripts_dir not in sys.path:
                        sys.path.insert(0, scripts_dir)
                    from ontology_expansion import load_ontology_index, get_kg_context  # noqa: E402
                    kg_index = load_ontology_index(ont_dir)
                    ontology_dir_resolved = ont_dir
                    print(f"KG context: {kg_context_hops} hop(s) (ontology dir {ont_dir})")
                    if kg_semantic and args.encoder in ("openai", "sentence-transformers"):
                        from ontology_expansion import build_ontology_label_embeddings, get_kg_augmentation  # noqa: E402
                        import hashlib
                        embed_sources = None
                        if getattr(args, "embed_ontology_sources", None):
                            embed_sources = set(args.embed_ontology_sources)
                        use_rich = not getattr(args, "no_kg_embed_rich_text", False)
                        model_slug = (args.encoder_model or "openai").replace("/", "_")
                        ont_hash = hashlib.md5(str(Path(ont_dir).resolve()).encode()).hexdigest()[:8]
                        rich_suffix = "_rich" if use_rich else ""
                        src_key = hashlib.md5("|".join(sorted(embed_sources)).encode()).hexdigest()[:6] if embed_sources else ""
                        cache_slug = f"{model_slug}_{ont_hash}{rich_suffix}_src{src_key}" if embed_sources else f"{model_slug}_{ont_hash}{rich_suffix}"
                        ontology_embeddings_dir = Path(repo_root) / "neurolab" / "data" / "ontology_embeddings"
                        emb_path = ontology_embeddings_dir / f"ontology_label_embeddings_{cache_slug}.npy"
                        list_path = ontology_embeddings_dir / f"ontology_label_list_{cache_slug}.pkl"
                        if emb_path.exists() and list_path.exists():
                            kg_label_embeddings = np.load(emb_path)
                            with open(list_path, "rb") as f:
                                kg_label_list = pickle.load(f)
                            print(f"Loaded ontology embeddings from cache: {len(kg_label_list)} labels (semantic KG mode)")
                        else:
                            kg_label_embeddings, kg_label_list = build_ontology_label_embeddings(
                                kg_index, lambda batch: encode_fn_raw(batch), use_rich_text=use_rich, embed_sources=embed_sources
                            )
                            print(f"Ontology embeddings: {len(kg_label_list)} labels (semantic KG mode)")
                else:
                    print("ontology-dir not found; disabling KG context", file=sys.stderr)
                    kg_context_hops = 0

            def encode_fn(terms_list):
                if expand_abbrev:
                    terms_list = expand_abbreviations(terms_list)
                if kg_context_hops and kg_index:
                    if kg_semantic and kg_label_embeddings is not None and kg_label_list is not None:
                        top_k = max(1, int(getattr(args, "kg_semantic_top_k", 5)))
                        sim_floor = float(getattr(args, "kg_sim_floor", 0.4))
                        max_triples = max(1, int(getattr(args, "kg_max_triples", 15)))
                        query_embs = encode_fn_raw(terms_list)
                        if query_embs.ndim == 1:
                            query_embs = query_embs.reshape(-1, 1)
                        augmentations = []
                        for i, t in enumerate(terms_list):
                            q_emb = query_embs[i] if query_embs.shape[0] > i else query_embs.ravel()
                            aug = get_kg_augmentation(t, q_emb, kg_label_embeddings, kg_label_list, kg_index, top_k=top_k, sim_floor=sim_floor, max_triples=max_triples)
                            augmentations.append(aug)
                        terms_list = [(aug.strip() + " " + t) if aug and aug.strip() else t for t, aug in zip(terms_list, augmentations)]
                    else:
                        terms_list = [t + get_kg_context(t, kg_index, max_hops=kg_context_hops) for t in terms_list]
                if embedding_prefix:
                    terms_list = _apply_embedding_prefix(terms_list, embedding_prefix)
                return encode_fn_raw(terms_list)

            print(f"Encoder: {args.encoder} / {args.encoder_model} (dim={dim}); embedding prefix: {repr(embedding_prefix)}" + ("; expand abbreviations" if expand_abbrev else "") + (f"; KG context {kg_context_hops} hop(s)" if kg_context_hops else ""))
            openai_embeddings_cached = False
            if args.encoder == "openai":
                model_slug = (args.encoder_model or "text-embedding-3-small").replace("/", "_")
                cache_emb_path = os.path.join(cache_dir, f"openai_embeddings_{model_slug}.npz")
                cache_meta_path = os.path.join(cache_dir, "openai_embedding_meta.pkl")
                if os.path.exists(cache_emb_path) and os.path.exists(cache_meta_path):
                    try:
                        with open(cache_meta_path, "rb") as f:
                            meta = pickle.load(f)
                        if (meta.get("terms") == terms and meta.get("prefix", "") == embedding_prefix
                                and meta.get("expand_abbreviations") == expand_abbrev
                                and meta.get("kg_context_hops") == kg_context_hops
                                and meta.get("ontology_dir") == ontology_dir_resolved
                                and meta.get("kg_context_mode") == getattr(args, "kg_context_mode", "substring")
                                and meta.get("embed_ontology_sources") == (getattr(args, "embed_ontology_sources") or None)):
                            data = np.load(cache_emb_path)
                            all_emb = data["embeddings"].astype(np.float32)
                            X_train = all_emb[train_idx]
                            X_val = all_emb[val_idx]
                            X_test_pre = all_emb[test_idx] if len(test_idx) else np.zeros((0, all_emb.shape[1]), dtype=np.float32)
                            openai_embeddings_cached = True
                            print(f"Loaded cached OpenAI embeddings from {cache_emb_path}")
                    except Exception as e:
                        print(f"OpenAI cache load failed: {e}; re-encoding.")
                if not openai_embeddings_cached:
                    all_emb = encode_fn(terms).astype(np.float32)
                    np.savez_compressed(cache_emb_path, embeddings=all_emb)
                    with open(cache_meta_path, "wb") as f:
                        pickle.dump({
                            "terms": terms, "prefix": embedding_prefix, "expand_abbreviations": expand_abbrev,
                            "kg_context_hops": kg_context_hops, "ontology_dir": ontology_dir_resolved,
                            "kg_context_mode": getattr(args, "kg_context_mode", "substring"),
                            "embed_ontology_sources": getattr(args, "embed_ontology_sources", None) or None,
                        }, f)
                    print(f"Cached OpenAI embeddings -> {cache_emb_path}")
                    X_train = all_emb[train_idx]
                    X_val = all_emb[val_idx]
                    X_test_pre = all_emb[test_idx] if len(test_idx) else np.zeros((0, all_emb.shape[1]), dtype=np.float32)
                    openai_embeddings_cached = True
            if not openai_embeddings_cached:
                X_train = encode_fn(train_terms).astype(np.float32)
                X_val = encode_fn(val_terms).astype(np.float32)
                X_test_pre = None
            encoder_dim = dim
            X_train = np.hstack([X_train, type_one_hot[train_idx]]).astype(np.float32)
            X_val = np.hstack([X_val, type_one_hot[val_idx]]).astype(np.float32)
            if X_test_pre is not None:
                X_test_pre = np.hstack([X_test_pre, type_one_hot[test_idx]]).astype(np.float32)
            dim = encoder_dim + len(MAP_TYPES)
        # Triad pairwise loss: target brain r from hop + emb_dist + interaction (see BUILD_MAPS_AND_TRAINING_PIPELINE.md 14c)
        triad_D = None
        triad_graph_idx = None
        triad_coef = None
        if getattr(args, "triad_pairwise_loss", False):
            if not getattr(args, "kg_regression_json", None) or not getattr(args, "ontology_dir", None):
                print("--triad-pairwise-loss requires --kg-regression-json and --ontology-dir", file=sys.stderr)
                sys.exit(1)
            reg_path = args.kg_regression_json if os.path.isabs(args.kg_regression_json) else os.path.join(repo_root, args.kg_regression_json)
            ont_path = args.ontology_dir if os.path.isabs(args.ontology_dir) else os.path.join(repo_root, args.ontology_dir)
            if not os.path.exists(reg_path):
                print(f"KG regression JSON not found: {reg_path}", file=sys.stderr)
                sys.exit(1)
            with open(reg_path) as f:
                reg = json.load(f)
            triad_coef = reg["coefficients"]
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            try:
                from graph_distance_correlation import _normalize, build_unified_graph  # noqa: E402
                from ontology_term_dendrogram import build_hierarchy_distance_matrix  # noqa: E402
            except ImportError as exc:
                print(
                    f"--triad-pairwise-loss requires graph_distance_correlation and "
                    f"ontology_term_dendrogram, which were removed in the production "
                    f"cleanup. Re-add those scripts to use this experimental flag. "
                    f"(underlying error: {exc})",
                    file=__import__("sys").stderr,
                )
                __import__("sys").exit(1)
            G = build_unified_graph(ont_path)
            graph_nodes = set(G.nodes())
            terms_in_graph_norm = list(dict.fromkeys(_normalize(t) for t in train_terms if _normalize(t) in graph_nodes))
            if len(terms_in_graph_norm) < 2:
                print("Triad: fewer than 2 training terms in ontology graph; disabling triad loss", file=sys.stderr)
            else:
                triad_D = build_hierarchy_distance_matrix(G, terms_in_graph_norm, cutoff=20)
                graph_idx = []
                for t in train_terms:
                    n = _normalize(t)
                    graph_idx.append(terms_in_graph_norm.index(n) if n in terms_in_graph_norm else -1)
                triad_graph_idx = np.array(graph_idx, dtype=np.int64)
                print(f"Triad pairwise loss: {int((triad_graph_idx >= 0).sum())}/{len(train_terms)} terms in graph, lambda={args.triad_loss_lambda:.2f}")
        training_history = []  # (epoch, train_loss, val_corr) for PyTorch path; saved with model
        test_term_corrs = []  # (term, r) per test term for split_info
        model_gene = None  # set in try when gene_pca is used
        try:
            import torch
            import torch.nn as nn
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    print("WARNING: --device cuda requested but CUDA not available; falling back to CPU.", file=sys.stderr)
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda")
            elif args.device == "cpu":
                device = torch.device("cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"PyTorch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
            print(f"Loss: {args.loss}" + (" (1 - correlation)" if args.loss == "cosine" else " (MSE)"))
            layers = [nn.Linear(dim, args.head_hidden), nn.ReLU()]
            if args.dropout > 0:
                layers.append(nn.Dropout(args.dropout))
            if getattr(args, "head_hidden2", 0) > 0:
                layers.append(nn.Linear(args.head_hidden, args.head_hidden2))
                layers.append(nn.ReLU())
                if args.dropout > 0:
                    layers.append(nn.Dropout(args.dropout))
                layers.append(nn.Linear(args.head_hidden2, n_out))
            else:
                layers.append(nn.Linear(args.head_hidden, n_out))
            model = nn.Sequential(*layers).to(device)
            model_gene = None
            if gene_pca is not None and n_gene_out > 0:
                layers_gene = [nn.Linear(dim, args.head_hidden), nn.ReLU()]
                if args.dropout > 0:
                    layers_gene.append(nn.Dropout(args.dropout))
                if getattr(args, "head_hidden2", 0) > 0:
                    layers_gene.append(nn.Linear(args.head_hidden, args.head_hidden2))
                    layers_gene.append(nn.ReLU())
                    if args.dropout > 0:
                        layers_gene.append(nn.Dropout(args.dropout))
                    layers_gene.append(nn.Linear(args.head_hidden2, n_gene_out))
                else:
                    layers_gene.append(nn.Linear(args.head_hidden, n_gene_out))
                model_gene = nn.Sequential(*layers_gene).to(device)
            params = list(model.parameters())
            if model_gene is not None:
                params += list(model_gene.parameters())
            opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            best_val_corr = -1.0
            best_state = None
            best_state_gene = None
            epochs_no_improve = 0
            train_rng = np.random.default_rng(args.seed)
            for ep in range(args.epochs):
                if sampling_probs is not None:
                    perm = train_rng.choice(len(X_train), size=len(X_train), replace=True, p=sampling_probs)
                else:
                    perm = train_rng.permutation(len(X_train))
                X_ep = X_train[perm]
                y_ep = train_targets[perm]
                w_ep = train_weights[perm] if train_weights is not None else None
                model.train()
                epoch_loss_sum = 0.0
                n_batches = 0
                for start in range(0, len(X_ep), args.batch_size):
                    end = min(start + args.batch_size, len(X_ep))
                    x = torch.from_numpy(X_ep[start:end]).to(device)
                    y = torch.from_numpy(y_ep[start:end]).to(device)
                    opt.zero_grad()
                    pred = model(x)
                    # Gene head: for abagen terms use gene head (PC loadings), else main head (n_parcels-D or PC)
                    if model_gene is not None:
                        batch_global = train_idx[perm[start:end]]
                        abagen_mask_np = np.array([g in abagen_set for g in batch_global])
                        non_abagen = ~abagen_mask_np
                        gene_pred = model_gene(x)
                        non_abagen_t = torch.from_numpy(non_abagen).to(x.device)
                        abagen_t = torch.from_numpy(abagen_mask_np).to(x.device)
                        w_batch = torch.from_numpy(w_ep[start:end]).to(device) if w_ep is not None else None
                        loss_main = torch.tensor(0.0, device=x.device)
                        if non_abagen.sum() > 0:
                            w_main = w_batch[non_abagen_t] if w_batch is not None else None
                            loss_main = _loss_torch(pred[non_abagen_t], y[non_abagen_t], args.loss, w_main)
                        loss_gene = torch.tensor(0.0, device=x.device)
                        if abagen_mask_np.sum() > 0:
                            loading_rows = [abagen_idx_to_loading_row[int(g)] for g in batch_global[abagen_mask_np]]
                            gene_targets = torch.from_numpy(gene_loadings[loading_rows].astype(np.float32)).to(device)
                            if w_batch is not None:
                                sq_gene = (gene_pred[abagen_t] - gene_targets).pow(2).mean(dim=1)
                                loss_gene = (w_batch[abagen_t] * sq_gene).sum() / w_batch[abagen_t].sum().clamp(min=1e-8)
                            else:
                                loss_gene = nn.functional.mse_loss(gene_pred[abagen_t], gene_targets)
                        loss = loss_main + loss_gene
                    elif w_ep is not None:
                        w = torch.from_numpy(w_ep[start:end]).to(device)
                        loss = _loss_torch(pred, y, args.loss, w)
                    else:
                        loss = _loss_torch(pred, y, args.loss, None)
                    # Triad pairwise loss: target = composite similarity (predicted brain r from hop + emb_dist + interaction)
                    if triad_D is not None and triad_graph_idx is not None and triad_coef is not None:
                        batch_perm = perm[start:end]
                        gb = triad_graph_idx[batch_perm]
                        B = pred.shape[0]
                        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
                        emb_dist_batch = 1.0 - (x_norm @ x_norm.T)
                        pred_c = pred - pred.mean(dim=1, keepdim=True)
                        pred_std = (pred_c.norm(dim=1, keepdim=True) + 1e-8) * (pred_c.norm(dim=1, keepdim=True).T + 1e-8)
                        pred_r_batch = (pred_c @ pred_c.T) / (pred_std + 1e-8)
                        pred_r_list, r_target_list = [], []
                        for i in range(B):
                            for j in range(i + 1, B):
                                if gb[i] >= 0 and gb[j] >= 0:
                                    hop_ij = float(triad_D[gb[i], gb[j]])
                                    emb_ij = emb_dist_batch[i, j]
                                    # r_t = composite similarity (same as composite_distance_utils.predicted_r)
                                    r_t = triad_coef["intercept"] + triad_coef["hop"] * hop_ij + triad_coef["emb_dist"] * emb_ij + triad_coef["hop_emb_dist"] * (hop_ij * emb_ij)
                                    pred_r_list.append(pred_r_batch[i, j])
                                    r_target_list.append(r_t)
                        if len(pred_r_list) > 0:
                            pred_r_t = torch.stack(pred_r_list)
                            r_target_t = torch.stack(r_target_list).to(device=device, dtype=pred.dtype)
                            loss_pair = ((pred_r_t - r_target_t) ** 2).mean()
                            loss = (1.0 - args.triad_loss_lambda) * loss + args.triad_loss_lambda * loss_pair
                    loss.backward()
                    opt.step()
                    epoch_loss_sum += loss.item()
                    n_batches += 1
                train_loss = epoch_loss_sum / n_batches if n_batches else 0.0
                if len(val_terms) > 0:
                    model.eval()
                    if model_gene is not None:
                        model_gene.eval()
                    with torch.no_grad():
                        vp = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
                        if model_gene is not None:
                            vp_gene = model_gene(torch.from_numpy(X_val).to(device)).cpu().numpy()
                    vp_400 = np.zeros((len(val_terms), n_parcels), dtype=np.float32)
                    for i in range(len(val_terms)):
                        g = val_idx[i]
                        if model_gene is not None and g in abagen_set:
                            vp_400[i] = gene_pca.inverse_transform(vp_gene[i : i + 1]).ravel()
                        else:
                            vp_400[i] = (pca.inverse_transform(vp[i : i + 1]) if pca is not None else vp[i]).ravel()
                    _, val_corr = _eval_metrics(vp_400, val_maps)
                    training_history.append((ep + 1, train_loss, val_corr))
                    if (ep + 1) % 10 == 0:
                        print(f"Epoch {ep+1} train_loss: {train_loss:.4f} val mean corr: {val_corr:.4f}")
                    if args.early_stopping:
                        if val_corr > best_val_corr:
                            best_val_corr = val_corr
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            if model_gene is not None:
                                best_state_gene = {k: v.cpu().clone() for k, v in model_gene.state_dict().items()}
                            else:
                                best_state_gene = None
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if epochs_no_improve >= args.patience:
                            print(f"Early stopping at epoch {ep+1} (no val improvement for {args.patience} epochs)")
                            if best_state is not None:
                                model.load_state_dict(best_state)
                                model.to(device)
                                if model_gene is not None and best_state_gene is not None:
                                    model_gene.load_state_dict(best_state_gene)
                                    model_gene.to(device)
                            break
                else:
                    training_history.append((ep + 1, train_loss, None))
            model.eval()
            if model_gene is not None:
                model_gene.eval()
            with torch.no_grad():
                if len(val_terms) > 0:
                    vp = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
                    if model_gene is not None:
                        vp_gene = model_gene(torch.from_numpy(X_val).to(device)).cpu().numpy()
                        val_pred_400 = np.zeros((len(val_terms), n_parcels), dtype=np.float32)
                        for i in range(len(val_terms)):
                            if val_idx[i] in abagen_set:
                                val_pred_400[i] = gene_pca.inverse_transform(vp_gene[i : i + 1]).ravel()
                            else:
                                val_pred_400[i] = (pca.inverse_transform(vp[i : i + 1]) if pca is not None else vp[i]).ravel()
                    else:
                        val_pred_400 = pca.inverse_transform(vp) if pca is not None else vp
                else:
                    val_pred_400 = np.zeros((0, n_parcels), dtype=np.float32)
        except ImportError:
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(args.head_hidden,), max_iter=args.epochs, learning_rate_init=args.lr, random_state=42)
            model.fit(X_train, train_targets, sample_weight=train_weights if train_weights is not None else None)
            if len(val_terms) > 0:
                val_pred = model.predict(X_val)
                val_pred_400 = pca.inverse_transform(val_pred) if pca is not None else val_pred
            else:
                val_pred_400 = np.zeros((0, n_parcels), dtype=np.float32)
        if len(val_terms) > 0:
            mse, corr = _eval_metrics(val_pred_400, val_maps)
            print(f"Val MSE: {mse:.4f}, mean correlation: {corr:.4f}")
        # Train set correlation (for train vs test comparison / overfitting check)
        if "torch" in sys.modules and hasattr(model, "state_dict"):
            with torch.no_grad():
                train_pred = model(torch.from_numpy(X_train).to(device)).cpu().numpy()
                if model_gene is not None:
                    train_pred_gene = model_gene(torch.from_numpy(X_train).to(device)).cpu().numpy()
                    train_pred_400 = np.zeros((len(train_idx), n_parcels), dtype=np.float32)
                    for i in range(len(train_idx)):
                        if train_idx[i] in abagen_set:
                            train_pred_400[i] = gene_pca.inverse_transform(train_pred_gene[i : i + 1]).ravel()
                        else:
                            train_pred_400[i] = (pca.inverse_transform(train_pred[i : i + 1]) if pca is not None else train_pred[i]).ravel()
                else:
                    train_pred_400 = pca.inverse_transform(train_pred) if pca is not None else train_pred
        else:
            train_pred = model.predict(X_train)
            train_pred_400 = pca.inverse_transform(train_pred) if pca is not None else train_pred
        train_maps_400 = term_maps[train_idx]
        _, train_corr = _eval_metrics(train_pred_400.astype(np.float64), train_maps_400)
        print(f"Train recovery (mean corr on training terms): {train_corr:.4f}")
        train_term_corrs = []
        if term_sources is not None:
            from collections import defaultdict
            train_corrs_by_source = defaultdict(list)
            for i in range(len(train_idx)):
                r = np.corrcoef(train_pred_400[i].ravel(), train_maps_400[i].ravel())[0, 1]
                train_term_corrs.append((train_terms[i], float(r)) if np.isfinite(r) else (train_terms[i], None))
                if np.isfinite(r):
                    train_corrs_by_source[term_sources[train_idx[i]]].append(r)
            print("  Train recovery by source:")
            for src in sorted(train_corrs_by_source.keys()):
                corrs = train_corrs_by_source[src]
                print(f"    {src}: {np.mean(corrs):.4f} (n={len(corrs)})")
        else:
            for i in range(len(train_idx)):
                r = np.corrcoef(train_pred_400[i].ravel(), train_maps_400[i].ravel())[0, 1]
                train_term_corrs.append((train_terms[i], float(r)) if np.isfinite(r) else (train_terms[i], None))
        # Optional: retrain on train+val (whole data except test)
        if args.final_retrain_on_train_and_val and (len(train_terms) + len(val_terms)) > len(train_terms):
            train_val_idx_all = np.concatenate([train_idx, val_idx])
            train_val_maps_400 = np.vstack([term_maps[train_idx], term_maps[val_idx]]).astype(np.float32)
            if use_cached:
                X_train_val = np.hstack([all_emb[train_val_idx_all].astype(np.float32), type_one_hot[train_val_idx_all]]).astype(np.float32)
            else:
                train_val_terms = train_terms + val_terms
                X_train_val = np.hstack([encode_fn(train_val_terms).astype(np.float32), type_one_hot[train_val_idx_all]]).astype(np.float32)
            if pca is not None:
                from sklearn.decomposition import PCA as PCAClass
                pca = PCAClass(n_components=n_out, random_state=42)
                pca.fit(train_val_maps_400)
                train_val_targets = pca.transform(train_val_maps_400).astype(np.float32)
            else:
                train_val_targets = train_val_maps_400
            if "torch" in sys.modules:
                import torch
                import torch.nn as nn
                if args.device == "cuda" and torch.cuda.is_available():
                    device = torch.device("cuda")
                elif args.device == "cpu":
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_val_idx_all = list(train_val_idx_all)  # already set above
                train_val_weights_pt = None if getattr(args, "uniform_memorizer", False) else (np.array([term_sample_weights[i] for i in train_val_idx_all], dtype=np.float32) if term_sample_weights is not None else (np.array([SAMPLE_WEIGHT_BY_SOURCE.get(term_sources[i], 0.5) for i in train_val_idx_all], dtype=np.float32) if term_sources is not None else None))
                layers = [nn.Linear(dim, args.head_hidden), nn.ReLU()]
                if args.dropout > 0:
                    layers.append(nn.Dropout(args.dropout))
                if getattr(args, "head_hidden2", 0) > 0:
                    layers.append(nn.Linear(args.head_hidden, args.head_hidden2))
                    layers.append(nn.ReLU())
                    if args.dropout > 0:
                        layers.append(nn.Dropout(args.dropout))
                    layers.append(nn.Linear(args.head_hidden2, n_out))
                else:
                    layers.append(nn.Linear(args.head_hidden, n_out))
                model = nn.Sequential(*layers).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                for ep in range(args.epochs):
                    model.train()
                    for start in range(0, len(X_train_val), args.batch_size):
                        end = min(start + args.batch_size, len(X_train_val))
                        x = torch.from_numpy(X_train_val[start:end]).to(device)
                        y = torch.from_numpy(train_val_targets[start:end]).to(device)
                        opt.zero_grad()
                        pred = model(x)
                        w = torch.from_numpy(train_val_weights_pt[start:end]).to(device) if train_val_weights_pt is not None else None
                        loss = _loss_torch(pred, y, args.loss, w)
                        loss.backward()
                        opt.step()
                model.eval()
            else:
                train_val_idx_all = list(train_idx) + list(val_idx)
                train_val_weights_enc = np.array([term_sample_weights[i] for i in train_val_idx_all], dtype=np.float32) if term_sample_weights is not None else (np.array([SAMPLE_WEIGHT_BY_SOURCE.get(term_sources[i], 0.5) for i in train_val_idx_all], dtype=np.float32) if term_sources is not None else None)
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(hidden_layer_sizes=(args.head_hidden,), max_iter=args.epochs, learning_rate_init=args.lr, random_state=42)
                model.fit(X_train_val, train_val_targets, sample_weight=train_val_weights_enc)
            print("Retrained on train+val (whole data except test).")
        # Save
        with open(os.path.join(output_dir, "config.pkl"), "wb") as f:
            pickle.dump({
                "encoder": args.encoder,
                "model_name": args.encoder_model,
                "n_parcels": n_parcels,
                "dim": encoder_dim,
                "map_types": MAP_TYPES,
                "head_hidden": args.head_hidden,
                "head_hidden2": getattr(args, "head_hidden2", 0),
                "dropout": getattr(args, "dropout", 0.0),
                "pca_components": n_out if pca is not None else 0,
                "embedding_prefix": embedding_prefix,
                "expand_abbreviations": getattr(args, "expand_abbreviations", False),
                "kg_context_hops": kg_context_hops,
                "ontology_dir": ontology_dir_resolved,
                "kg_context_mode": getattr(args, "kg_context_mode", "substring"),
                "kg_context_style": getattr(args, "kg_context_style", "triples"),
                "kg_semantic_top_k": getattr(args, "kg_semantic_top_k", 5),
                "kg_sim_floor": getattr(args, "kg_sim_floor", 0.4),
                "kg_max_triples": getattr(args, "kg_max_triples", 15),
                "kg_embed_rich_text": not getattr(args, "no_kg_embed_rich_text", False),
                "embed_ontology_sources": getattr(args, "embed_ontology_sources", None) or None,
                "use_ontology_retrieval_augmentation": getattr(args, "use_ontology_retrieval_augmentation", False),
                "ontology_retrieval_cache_dir": getattr(args, "ontology_retrieval_cache_dir", None) or None,
                "ontology_retrieval_alpha": getattr(args, "ontology_retrieval_alpha", 0.3),
                "ontology_retrieval_max_hops": getattr(args, "ontology_retrieval_max_hops", 2),
                "use_gene_head": model_gene is not None,
                "n_gene_out": n_gene_out if gene_pca is not None else 0,
                "loss": getattr(args, "loss", "mse"),
                "uniform_memorizer": getattr(args, "uniform_memorizer", False),
                "cache_dir": cache_dir,
            }, f)
        if pca is not None:
            with open(os.path.join(output_dir, "pca.pkl"), "wb") as f:
                pickle.dump(pca, f)
        if "torch" in sys.modules and hasattr(model, "state_dict"):
            torch.save(model.state_dict(), os.path.join(output_dir, "head_weights.pt"))
            if model_gene is not None:
                torch.save(model_gene.state_dict(), os.path.join(output_dir, "gene_head_weights.pt"))
            if gene_pca is not None:
                with open(os.path.join(output_dir, "gene_pca.pkl"), "wb") as f:
                    pickle.dump(gene_pca, f)
        else:
            with open(os.path.join(output_dir, "head_mlp.pkl"), "wb") as f:
                pickle.dump(model, f)
        # Guardrail: embeddings of all terms for scope check
        if use_cached:
            X_all = all_emb.astype(np.float64)  # already loaded from cache (n, encoder_dim)
        else:
            X_all = encode_fn(terms).astype(np.float64)
        np.save(os.path.join(output_dir, "training_embeddings.npy"), X_all)
        with open(os.path.join(output_dir, "training_terms.pkl"), "wb") as f:
            pickle.dump(terms, f)
        # Test set evaluation (generalization to unseen terms)
        test_corr = None
        if len(test_terms) > 0:
            if X_test_pre is not None:
                X_test = X_test_pre
            else:
                X_test = np.hstack([encode_fn(test_terms).astype(np.float32), type_one_hot[test_idx]]).astype(np.float32)
            if "torch" in sys.modules and hasattr(model, "state_dict"):
                import torch as _torch
                dev = next(model.parameters()).device
                with _torch.no_grad():
                    test_pred = model(_torch.from_numpy(X_test).to(dev)).cpu().numpy()
                    if model_gene is not None:
                        test_pred_gene = model_gene(_torch.from_numpy(X_test).to(dev)).cpu().numpy()
                        test_pred_400 = np.zeros((len(test_terms), n_parcels), dtype=np.float32)
                        for i in range(len(test_terms)):
                            if test_idx[i] in abagen_set:
                                test_pred_400[i] = gene_pca.inverse_transform(test_pred_gene[i : i + 1]).ravel()
                            else:
                                test_pred_400[i] = (pca.inverse_transform(test_pred[i : i + 1]) if pca is not None else test_pred[i]).ravel()
                    else:
                        test_pred_400 = pca.inverse_transform(test_pred) if pca is not None else test_pred
            else:
                test_pred = model.predict(X_test)
                test_pred_400 = pca.inverse_transform(test_pred) if pca is not None else test_pred
            test_mse, test_corr = _eval_metrics(test_pred_400.astype(np.float64), test_maps)
            print(f"Test generalization (mean corr on held-out terms): {test_corr:.4f} (MSE: {test_mse:.4f})")
            # Per-term breakdown for diagnosing poor generalizers
            test_term_corrs.clear()
            for i, term in enumerate(test_terms):
                r = np.corrcoef(test_pred_400[i].ravel(), test_maps[i].ravel())[0, 1]
                test_term_corrs.append((term, float(r)) if np.isfinite(r) else (term, None))
            # Per-source breakdown
            if term_sources is not None:
                from collections import defaultdict
                corrs_by_source = defaultdict(list)
                for i in range(len(test_terms)):
                    src = term_sources[test_idx[i]]
                    r = test_term_corrs[i][1]
                    if r is not None:
                        corrs_by_source[src].append(r)
                print("  Test generalization by source:")
                for src in sorted(corrs_by_source.keys()):
                    corrs = corrs_by_source[src]
                    mean_r = float(np.mean(corrs))
                    print(f"    {src}: {mean_r:.4f} (n={len(corrs)})")
            poor = [(t, r) for t, r in test_term_corrs if r is not None and r < 0.2]
            if poor:
                safe = [(t.encode("ascii", errors="replace").decode() if isinstance(t, str) else str(t), r) for t, r in poor[:10]]
                print(f"  Test terms with r < 0.2 ({len(poor)}): {safe}{'...' if len(poor) > 10 else ''}")
        # Final model: train (or train+val) correlation for same model as saved
        if args.final_retrain_on_train_and_val and (len(train_terms) + len(val_terms)) > len(train_terms):
            if "torch" in sys.modules and hasattr(model, "state_dict"):
                with torch.no_grad():
                    train_val_pred = model(torch.from_numpy(X_train_val).to(device)).cpu().numpy()
            else:
                train_val_pred = model.predict(X_train_val)
            train_val_pred_400 = pca.inverse_transform(train_val_pred) if pca is not None else train_val_pred
            _, train_final_corr = _eval_metrics(train_val_pred_400.astype(np.float64), train_val_maps_400)
            print(f"Train+val recovery (final model): {train_final_corr:.4f}")
        # Summary: recovery vs generalization
        print(f"Summary: Train recovery = {train_corr:.4f}" + (f", Test generalization = {test_corr:.4f}" if test_corr is not None else "") + (f", gap = {train_corr - test_corr:.4f}" if test_corr is not None else ""))
        # Save split info and optional training/eval artifacts
        split_info = {"train_terms": train_terms, "val_terms": val_terms, "test_terms": test_terms, "test_idx": test_idx.tolist(), "seed": args.seed}
        if term_sources is not None:
            split_info["term_sources"] = term_sources
        if test_term_corrs:
            split_info["test_term_correlations"] = list(test_term_corrs)
        if train_term_corrs:
            split_info["train_term_correlations"] = list(train_term_corrs)
        with open(os.path.join(output_dir, "split_info.pkl"), "wb") as f:
            pickle.dump(split_info, f)
        if training_history:
            with open(os.path.join(output_dir, "training_history.pkl"), "wb") as f:
                pickle.dump(training_history, f)
            print(f"Saved training_history.pkl ({len(training_history)} epochs)")
        print(f"Saved to {output_dir} (split_info.pkl, guardrail embeddings)")
        return

    raise ValueError(f"Unsupported encoder: {args.encoder}. Use tfidf, sentence-transformers, or openai.")


if __name__ == "__main__":
    main()
