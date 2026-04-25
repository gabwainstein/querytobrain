"""
Text-to-brain embedding: map arbitrary text to parcellated space (Glasser+Tian, 392-D).

Loads a model trained by scripts/train_text_to_brain_embedding.py. This lets you:
- **Predict maps for text not in NeuroQuery/NeuroSynth:** For any phrase (e.g. novel
  compound, long description), the model predicts the parcellated brain map.
- **Expand the term space:** Decode to nearest known terms or add synthetic terms.
- **Retrieval-first output:** For any query, retrieve top-k nearest training terms
  with provenance (source, evidence map) before returning the predicted map.

The output of embed() / predict_map() is the predicted map (Glasser+Tian, 392 parcels).

**Similar words:** The model is not a literal lookup. You pass any string; the encoder
turns it into a vector; the head maps that to n_parcels-D. So:
- **Sentence-transformers:** Similar meaning → similar embedding → similar map (e.g.
  "noradrenergic" and "norepinephrine" give related maps).
- **TF-IDF:** Only tokens seen at training get weight; synonyms with different words
  can differ. For best generalization to similar wording, use sentence-transformers.

**Multiple words / phrases:** The whole input is encoded as one string → one embedding
→ one map. So "dopamine and serotonin" yields a single predicted map for that phrase,
not two maps averaged. Training terms are often multi-word too (e.g. "working memory").
"""
import os
import pickle
from typing import Optional, Union

import numpy as np

from ..parcellation import get_n_parcels
from .term_expansion import expand_abbreviations


def _n_parcels_from_config(config: dict) -> int:
    """Parcel count from config or pipeline default (Glasser+Tian, 392)."""
    return config.get("n_parcels") or get_n_parcels()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity. a: (n_a, d), b: (n_b, d) -> (n_a, n_b) or a: (d,), b: (n, d) -> (n,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = np.dot(an, bn.T)
    return sim.ravel() if sim.size == sim.shape[1] else sim


class TextToBrainEmbedding:
    """
    Map text -> (n_parcels,) parcellated vector using a trained encoder + head.
    Use predict_map() for text not in any database (predicted map).
    Use retrieve() or predict_map_with_retrieval() for retrieval-first output (top-k evidence + provenance).
    n_parcels is 392 (Glasser+Tian), from config or pipeline.
    """

    def __init__(self, model_dir: str, cache_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._retrieval_cache_dir = (cache_dir or "").strip()
        self.config_path = os.path.join(model_dir, "config.pkl")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Embedding model not found. Train with: "
                f"python neurolab/scripts/train_text_to_brain_embedding.py --output-dir {model_dir}"
            )
        with open(self.config_path, "rb") as f:
            self.config = pickle.load(f)
        self._encoder_type = self.config.get("encoder", "tfidf")
        self._embedding_prefix = (self.config.get("embedding_prefix") or "").strip()
        self._expand_abbreviations = self.config.get("expand_abbreviations", False)
        self._kg_context_hops = max(0, int(self.config.get("kg_context_hops", 0)))
        self._kg_context_mode = (self.config.get("kg_context_mode") or "substring").strip().lower()
        if self._kg_context_mode not in ("substring", "semantic"):
            self._kg_context_mode = "substring"
        self._kg_index = None  # loaded when _kg_context_hops > 0
        self._get_kg_context_fn = None  # set in _load_kg_index (substring or semantic triples)
        self._get_kg_augmentation_fn = None  # set when semantic + natural (needs query_emb in embed())
        self._kg_label_embeddings = None  # for semantic mode
        self._kg_label_list: list = []  # for semantic mode
        self._kg_semantic_top_k = max(1, int(self.config.get("kg_semantic_top_k", 5)))
        self._kg_sim_floor = float(self.config.get("kg_sim_floor", 0.4))
        self._kg_max_triples = max(1, int(self.config.get("kg_max_triples", 15)))
        self._kg_context_style = (self.config.get("kg_context_style") or "triples").strip().lower()
        self._kg_embed_rich_text = self.config.get("kg_embed_rich_text", True)
        if self._kg_context_style not in ("triples", "natural"):
            self._kg_context_style = "triples"
        self._map_types = self.config.get("map_types") or []
        # Ontology retrieval augmentation: blend MLP prediction with maps from graph-expanded terms
        self._use_ontology_retrieval_augmentation = bool(self.config.get("use_ontology_retrieval_augmentation", False))
        self._ontology_retrieval_cache_dir = (self.config.get("ontology_retrieval_cache_dir") or "").strip()
        self._ontology_retrieval_alpha = min(0.5, max(0.0, float(self.config.get("ontology_retrieval_alpha", 0.3))))
        self._ontology_retrieval_max_hops = max(0, int(self.config.get("ontology_retrieval_max_hops", 2)))
        self._meta_graph = None
        self._training_maps_db = None
        self._drug_spatial_maps = None
        self._last_enrichment = None  # set when retrieval augmentation runs in embed()
        self._vectorizer = None
        self._head = None
        self._st_model = None
        self._openai_model = None  # model name for openai encoder (no local weights)
        self._head_torch = None
        self._pca = None  # if set, head outputs components; we inverse_transform to n_parcels-D
        self._gene_head_torch = None  # when use_gene_head: predict PC loadings for pet_receptor type
        self._gene_pca = None  # inverse_transform loadings -> n_parcels
        self._retrieval_embeddings = None  # (n, dim) training embeddings for top-k retrieval
        self._retrieval_terms: list = []  # training term strings
        self._term_to_source: dict = {}  # term -> source (from cache)
        self._term_to_map_idx: dict = {}  # term -> index into cache term_maps
        self._retrieval_term_maps = None  # (n, parcels) from cache for evidence maps
        self._load()
        self._load_retrieval_index()
        if self._kg_context_hops > 0:
            self._load_kg_index()
        if self._use_ontology_retrieval_augmentation:
            self._load_retrieval_augmentation()

    def _load(self) -> None:
        if self._encoder_type == "tfidf":
            with open(os.path.join(self.model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
                self._vectorizer = pickle.load(f)
            with open(os.path.join(self.model_dir, "head_mlp.pkl"), "rb") as f:
                self._head = pickle.load(f)
            return
        if self._encoder_type == "sentence-transformers":
            model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError("sentence-transformers required for this model. pip install sentence-transformers")
            self._load_head_and_pca()
            return
        if self._encoder_type == "openai":
            self._openai_model = self.config.get("model_name", "text-embedding-3-small")
            self._load_head_and_pca()
            return
        raise ValueError(f"Unknown encoder type: {self._encoder_type}")

    def _load_retrieval_index(self) -> None:
        """Load training embeddings and terms for top-k retrieval. Optionally load provenance from cache_dir."""
        emb_path = os.path.join(self.model_dir, "training_embeddings.npy")
        terms_path = os.path.join(self.model_dir, "training_terms.pkl")
        if not os.path.exists(emb_path) or not os.path.exists(terms_path):
            return
        self._retrieval_embeddings = np.load(emb_path).astype(np.float64)
        with open(terms_path, "rb") as f:
            self._retrieval_terms = pickle.load(f)
        if self._retrieval_embeddings.shape[0] != len(self._retrieval_terms):
            self._retrieval_embeddings = None
            self._retrieval_terms = []
            return
        # Provenance from cache_dir (config or init param)
        cache_dir = self._retrieval_cache_dir or self.config.get("cache_dir") or ""
        if not cache_dir:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base, "neurolab", "data", "merged_sources")
        if not os.path.isabs(cache_dir):
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base, cache_dir)
        if os.path.isdir(cache_dir):
            sources_pkl = os.path.join(cache_dir, "term_sources.pkl")
            npz_path = os.path.join(cache_dir, "term_maps.npz")
            vocab_pkl = os.path.join(cache_dir, "term_vocab.pkl")
            if os.path.exists(sources_pkl) and os.path.exists(vocab_pkl):
                with open(sources_pkl, "rb") as f:
                    sources = pickle.load(f)
                with open(vocab_pkl, "rb") as f:
                    vocab = pickle.load(f)
                if isinstance(vocab, dict):
                    vocab = list(vocab.keys()) if vocab else []
                vocab = list(vocab)
                if len(vocab) == len(sources):
                    self._term_to_source = {v: s for v, s in zip(vocab, sources)}
            if os.path.exists(npz_path) and os.path.exists(vocab_pkl):
                data = np.load(npz_path)
                key = "term_maps" if "term_maps" in data else data.files[0]
                self._retrieval_term_maps = np.asarray(data[key], dtype=np.float64)
                with open(vocab_pkl, "rb") as f:
                    vocab = pickle.load(f)
                if isinstance(vocab, dict):
                    vocab = list(vocab.keys()) if vocab else []
                vocab = list(vocab)
                if self._retrieval_term_maps.shape[0] == len(vocab):
                    self._term_to_map_idx = {v: i for i, v in enumerate(vocab)}

    def _get_encode_fn(self):
        """Return encode_fn: list[str] -> np.ndarray (same encoder as model, no prefix)."""
        if self._encoder_type == "openai":
            return self._embed_openai
        if self._encoder_type == "sentence-transformers" and self._st_model is not None:
            return lambda texts: self._st_model.encode(texts, show_progress_bar=False).astype(np.float32)
        return None

    def _load_kg_index(self) -> None:
        """Load ontology index when config has kg_context_hops > 0 (for inference)."""
        if self._kg_context_hops <= 0:
            return
        ont_dir = self.config.get("ontology_dir") or ""
        if not ont_dir or not os.path.isdir(ont_dir):
            return
        import sys
        import hashlib
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from ontology_expansion import (  # noqa: E402
            load_ontology_index,
            get_kg_context_for_query,
            build_ontology_label_embeddings,
            get_kg_context_for_query_semantic,
            get_kg_augmentation,
        )
        self._kg_index = load_ontology_index(ont_dir)
        if self._kg_context_mode == "semantic" and self._encoder_type in ("openai", "sentence-transformers"):
            encode_fn = self._get_encode_fn()
            if encode_fn is not None:
                model_slug = (self.config.get("model_name") or "openai").replace("/", "_")
                ont_hash = hashlib.md5(ont_dir.encode()).hexdigest()[:8]
                rich_suffix = "_rich" if self._kg_embed_rich_text else ""
                embed_sources = self.config.get("embed_ontology_sources")
                if embed_sources:
                    src_key = hashlib.md5("|".join(sorted(embed_sources)).encode()).hexdigest()[:6]
                    cache_slug = f"{model_slug}_{ont_hash}{rich_suffix}_src{src_key}"
                else:
                    cache_slug = f"{model_slug}_{ont_hash}{rich_suffix}"
                emb_path = os.path.join(self.model_dir, f"ontology_label_embeddings_{cache_slug}.npy")
                list_path = os.path.join(self.model_dir, f"ontology_label_list_{cache_slug}.pkl")
                if os.path.exists(emb_path) and os.path.exists(list_path):
                    self._kg_label_embeddings = np.load(emb_path)
                    with open(list_path, "rb") as f:
                        self._kg_label_list = pickle.load(f)
                else:
                    self._kg_label_embeddings, self._kg_label_list = build_ontology_label_embeddings(
                        self._kg_index,
                        encode_fn,
                        batch_size=64,
                        use_rich_text=self._kg_embed_rich_text,
                        embed_sources=embed_sources,
                    )
                    np.save(emb_path, self._kg_label_embeddings)
                    with open(list_path, "wb") as f:
                        pickle.dump(self._kg_label_list, f)
                if self._kg_context_style == "natural":
                    self._get_kg_context_fn = None
                    self._get_kg_augmentation_fn = get_kg_augmentation
                else:
                    self._get_kg_augmentation_fn = None
                    self._get_kg_context_fn = lambda q: get_kg_context_for_query_semantic(
                        q, self._kg_index, encode_fn, self._kg_label_embeddings, self._kg_label_list,
                        top_k=self._kg_semantic_top_k, max_hops=self._kg_context_hops,
                    )
            else:
                self._get_kg_context_fn = lambda q: get_kg_context_for_query(
                    q, self._kg_index, max_hops=self._kg_context_hops
                )
        else:
            self._get_kg_augmentation_fn = None
            self._get_kg_context_fn = lambda q: get_kg_context_for_query(
                q, self._kg_index, max_hops=self._kg_context_hops
            )

    def _load_retrieval_augmentation(self) -> None:
        """Build meta-graph and load training maps for ontology retrieval augmentation."""
        if not self._use_ontology_retrieval_augmentation or not self._kg_index:
            return
        if self._kg_label_embeddings is None or self._kg_label_list is None:
            return
        if not self._ontology_retrieval_cache_dir:
            return
        import sys
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        try:
            from ontology_meta_graph import (  # noqa: E402
                build_meta_graph,
                get_training_maps_db,
            )
        except ImportError:
            return
        cache_dir = self._ontology_retrieval_cache_dir
        if not os.path.isabs(cache_dir):
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(base, cache_dir)
        self._training_maps_db, _ = get_training_maps_db(cache_dir)
        if not self._training_maps_db:
            return
        self._meta_graph = build_meta_graph(
            self._kg_index,
            label_embeddings=self._kg_label_embeddings,
            label_list=self._kg_label_list,
            similarity_threshold=0.85,
            max_bridges_per_node=5,
        )
        # Optional: drug spatial maps from gene PCA Phase 4
        gene_pca_dir = os.path.join(os.path.dirname(cache_dir), "gene_pca")
        if os.path.isdir(gene_pca_dir):
            try:
                import json
                drug_maps_path = os.path.join(gene_pca_dir, "drug_spatial_maps.npy")
                drug_names_path = os.path.join(gene_pca_dir, "drug_names.json")
                if os.path.exists(drug_maps_path) and os.path.exists(drug_names_path):
                    drug_maps = np.load(drug_maps_path)
                    with open(drug_names_path) as f:
                        drug_names = json.load(f)
                    self._drug_spatial_maps = {name: drug_maps[i] for i, name in enumerate(drug_names)}
            except Exception:
                self._drug_spatial_maps = None

    def _load_head_and_pca(self) -> None:
        """Load head (torch or sklearn) and PCA for sentence-transformers / openai encoders."""
        head_path = os.path.join(self.model_dir, "head_weights.pt")
        pca_path = os.path.join(self.model_dir, "pca.pkl")
        if self.config.get("pca_components", 0) > 0 and os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                self._pca = pickle.load(f)
        n_out = self.config.get("pca_components", 0) or _n_parcels_from_config(self.config)
        if os.path.exists(head_path):
            import torch
            dim = self.config["dim"] + len(self.config.get("map_types", []))
            hidden = self.config.get("head_hidden", 512)
            hidden2 = self.config.get("head_hidden2", 0)
            dropout = self.config.get("dropout", 0.0)
            layers = [torch.nn.Linear(dim, hidden), torch.nn.ReLU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            if hidden2 > 0:
                layers.append(torch.nn.Linear(hidden, hidden2))
                layers.append(torch.nn.ReLU())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.Linear(hidden2, n_out))
            else:
                layers.append(torch.nn.Linear(hidden, n_out))
            self._head_torch = torch.nn.Sequential(*layers)
            self._head_torch.load_state_dict(torch.load(head_path, map_location="cpu"))
            self._head_torch.eval()
        if self.config.get("use_gene_head") and os.path.exists(os.path.join(self.model_dir, "gene_head_weights.pt")) and os.path.exists(os.path.join(self.model_dir, "gene_pca.pkl")):
            with open(os.path.join(self.model_dir, "gene_pca.pkl"), "rb") as f:
                self._gene_pca = pickle.load(f)
            n_gene_out = self._gene_pca.n_components_
            gene_head_path = os.path.join(self.model_dir, "gene_head_weights.pt")
            layers_gene = [torch.nn.Linear(dim, hidden), torch.nn.ReLU()]
            if dropout > 0:
                layers_gene.append(torch.nn.Dropout(dropout))
            if hidden2 > 0:
                layers_gene.append(torch.nn.Linear(hidden, hidden2))
                layers_gene.append(torch.nn.ReLU())
                if dropout > 0:
                    layers_gene.append(torch.nn.Dropout(dropout))
                layers_gene.append(torch.nn.Linear(hidden2, n_gene_out))
            else:
                layers_gene.append(torch.nn.Linear(hidden, n_gene_out))
            self._gene_head_torch = torch.nn.Sequential(*layers_gene)
            self._gene_head_torch.load_state_dict(torch.load(gene_head_path, map_location="cpu"))
            self._gene_head_torch.eval()
        else:
            with open(os.path.join(self.model_dir, "head_mlp.pkl"), "rb") as f:
                self._head = pickle.load(f)

    def _embed_openai(self, texts: list) -> np.ndarray:
        """Call OpenAI embeddings API; return (n, dim) float32."""
        _env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        if os.path.isfile(_env):
            try:
                from dotenv import load_dotenv
                load_dotenv(_env)
            except ImportError:
                pass
        from openai import OpenAI
        client = OpenAI()
        model = self._openai_model or "text-embedding-3-small"
        batch_size = 100
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = client.embeddings.create(input=batch, model=model)
            out.extend([d.embedding for d in r.data])
        return np.array(out, dtype=np.float32)

    def _apply_prefix(self, texts: list) -> list:
        """Prepend embedding prefix to each text if set (same as at training)."""
        if not self._embedding_prefix:
            return list(texts)
        p = self._embedding_prefix.rstrip()
        return [p + (" " if p else "") + (t or "") for t in texts]

    def _infer_map_type(self, query: str) -> int:
        """Infer map type from query for type-conditioned MLP. Returns index into self._map_types."""
        if not self._map_types:
            return 0
        q = (query or "").strip().lower()
        # fmri_activation
        if any(w in q for w in ("activity", "activation", "task", "bold", "fmri", "contrast", "response", "working memory")):
            if "fmri_activation" in self._map_types:
                return self._map_types.index("fmri_activation")
        # structural
        if any(w in q for w in ("thickness", "volume", "atrophy", "structural", "structure", "cortical thickness", "subcortical", "enigma")):
            if "structural" in self._map_types:
                return self._map_types.index("structural")
        # pet_receptor (gene expression, receptor, PET)
        if any(w in q for w in ("receptor", "binding", "pet", "transmitter", "dopamine receptor", "serotonin receptor", "gaba receptor", "gene expression", "gene")):
            if "pet_receptor" in self._map_types:
                return self._map_types.index("pet_receptor")
        return 0

    def embed(self, text: Union[str, list]) -> np.ndarray:
        """
        Map text (or list of strings) to parcellated vector(s).
        Returns (n_parcels,) for str input, (n, n_parcels) for list of n strings.
        """
        single = isinstance(text, str)
        if single:
            text = [text]
        raw_texts = list(text)
        if self._expand_abbreviations:
            text = expand_abbreviations(text)
        if self._kg_context_hops and self._kg_index:
            if self._get_kg_augmentation_fn and self._kg_label_embeddings is not None:
                encode_fn = self._get_encode_fn()
                if encode_fn is not None:
                    for i, t in enumerate(text):
                        query_emb = encode_fn([t])[0]
                        context = self._get_kg_augmentation_fn(
                            t, query_emb, self._kg_label_embeddings, self._kg_label_list, self._kg_index,
                            top_k=self._kg_semantic_top_k, sim_floor=self._kg_sim_floor,
                            max_triples=self._kg_max_triples,
                        )
                        text[i] = (context + " " + t) if context else t
            elif self._get_kg_context_fn:
                text = [t + self._get_kg_context_fn(t) for t in text]
        text = self._apply_prefix(text)
        X_semantic = None  # for retrieval augmentation (same space as ontology embeddings)
        if self._encoder_type == "tfidf":
            X = self._vectorizer.transform(text).toarray()
            out = self._head.predict(X)
        else:
            if self._encoder_type == "openai":
                X = self._embed_openai(text)
            else:
                X = self._st_model.encode(text, show_progress_bar=False).astype(np.float32)
            if self._map_types:
                type_indices = [self._infer_map_type(t) for t in raw_texts]
                one_hot = np.eye(len(self._map_types), dtype=np.float32)[type_indices]
                X = np.hstack([X.astype(np.float32), one_hot])
                X_semantic = X[:, : X.shape[1] - len(self._map_types)]
            else:
                X_semantic = X
            if self._head_torch is not None:
                import torch
                with torch.no_grad():
                    main_out = self._head_torch(torch.from_numpy(X)).numpy()
                n_parcels = _n_parcels_from_config(self.config)
                if self._gene_head_torch is not None and "pet_receptor" in self._map_types:
                    pet_idx = self._map_types.index("pet_receptor")
                    pet_mask = np.array([t == pet_idx for t in type_indices])
                    out = np.zeros((len(raw_texts), n_parcels), dtype=np.float64)
                    if pet_mask.any():
                        X_t = torch.from_numpy(X.astype(np.float32))
                        gene_out = self._gene_head_torch(X_t[pet_mask]).numpy()
                        for i, j in enumerate(np.where(pet_mask)[0]):
                            out[j] = self._gene_pca.inverse_transform(gene_out[i : i + 1]).ravel()
                    non_pet = ~pet_mask
                    if non_pet.any():
                        non_pet_out = self._pca.inverse_transform(main_out[non_pet]) if self._pca is not None else main_out[non_pet]
                        out[non_pet] = non_pet_out
                else:
                    out = main_out
            else:
                out = self._head.predict(X)
        out = np.asarray(out, dtype=np.float64)
        n_parcels = _n_parcels_from_config(self.config)
        if self._pca is not None and (out.ndim == 1 or out.shape[1] != n_parcels):
            out = self._pca.inverse_transform(out)
        # Ontology retrieval augmentation: blend with maps from graph-expanded terms
        if (
            self._meta_graph is not None
            and self._training_maps_db is not None
            and X_semantic is not None
        ):
            import sys
            scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            try:
                from ontology_meta_graph import augmented_prediction  # noqa: E402
                n_samples = out.shape[0]
                self._last_enrichment = None
                for i in range(n_samples):
                    q_emb = X_semantic[i] if X_semantic.ndim > 1 else X_semantic
                    pred_i = out[i] if out.ndim > 1 else out
                    final_map, enrichment = augmented_prediction(
                        raw_texts[i],
                        q_emb,
                        pred_i,
                        self._meta_graph,
                        self._training_maps_db,
                        drug_spatial_maps=self._drug_spatial_maps,
                        max_hops=self._ontology_retrieval_max_hops,
                        alpha_retrieval_cap=self._ontology_retrieval_alpha,
                        min_relevance=0.3,
                    )
                    if out.ndim > 1:
                        out[i] = final_map
                    else:
                        out = final_map
                    if n_samples == 1:
                        self._last_enrichment = enrichment
            except ImportError:
                self._last_enrichment = None
        if single:
            return out.ravel()
        return out

    def predict_map(self, text: Union[str, list]) -> np.ndarray:
        """
        Predict parcellated brain map(s) for text that may not be in NeuroQuery/NeuroSynth.
        Same as embed(): returns (n_parcels,) or (n, n_parcels). This is the predicted map for unseen text.
        """
        return self.embed(text)

    def get_last_enrichment(self) -> dict:
        """
        After embed() or predict_map() with retrieval augmentation, returns the last
        enrichment dict (related_diseases, related_phenotypes, related_concepts,
        related_drugs, related_receptors, expansion). Empty dict if not available.
        """
        return self._last_enrichment or {}

    def _embed_text_for_retrieval(self, text: str) -> np.ndarray:
        """Return raw text embedding (same space as training_embeddings) for retrieval. No KG augmentation."""
        if self._expand_abbreviations:
            text = expand_abbreviations(text)
        text = self._apply_prefix([text])[0]
        if self._encoder_type == "tfidf":
            X = self._vectorizer.transform([text]).toarray().astype(np.float64)
        elif self._encoder_type == "openai":
            X = self._embed_openai([text])
        elif self._st_model is not None:
            X = self._st_model.encode([text], show_progress_bar=False).astype(np.float64)
        else:
            return np.array([])
        return X.ravel()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve top-k nearest training terms by embedding similarity (no retraining).
        Returns list of dicts: {term, similarity, source, evidence_map}.
        evidence_map is (n_parcels,) or None if not available from cache.
        """
        if self._retrieval_embeddings is None or len(self._retrieval_terms) == 0:
            return []
        q_emb = self._embed_text_for_retrieval(query)
        if q_emb.size == 0:
            return []
        sim = _cosine_similarity(q_emb, self._retrieval_embeddings)
        order = np.argsort(sim)[::-1]
        out = []
        for i in order[:top_k]:
            s = float(sim[i])
            if s < min_similarity:
                break
            term = self._retrieval_terms[i]
            rec = {"term": term, "similarity": s, "source": self._term_to_source.get(term)}
            idx = self._term_to_map_idx.get(term)
            if idx is not None and self._retrieval_term_maps is not None:
                rec["evidence_map"] = self._retrieval_term_maps[idx].astype(np.float64)
            else:
                rec["evidence_map"] = None
            out.append(rec)
        return out

    def predict_map_with_retrieval(
        self,
        text: str,
        top_k: int = 10,
    ) -> tuple[np.ndarray, dict]:
        """
        Predict brain map and return (map, result) where result has:
          - retrieval: list of {term, similarity, source, evidence_map} (top-k evidence)
          - confidence: best similarity (float)
          - map: the predicted map (same as first return value)
        """
        pred_map = self.predict_map(text)
        retrieval = self.retrieve(text, top_k=top_k)
        best_sim = retrieval[0]["similarity"] if retrieval else 0.0
        return pred_map, {
            "map": pred_map,
            "retrieval": retrieval,
            "confidence": best_sim,
        }
        """
        Predict brain map for a single query and return (map, enrichment).
        Enrichment is populated when retrieval augmentation is enabled; otherwise
        enrichment lists are empty. Use get_last_enrichment() after predict_map(text)
        for the same result when you only need the map first.
        """
        self._last_enrichment = None
        out = self.predict_map(text)
        return out, self.get_last_enrichment()

    def predict_map_to_nifti(
        self,
        text: str,
        output_path: str,
        n_parcels: int | None = None,
    ) -> str:
        """
        Predict map for text and save as 3D NIfTI (e.g. for visualization).
        Uses pipeline parcellation (combined cortical+subcortical when available) to unparcellate -> volume.
        Returns the path written.
        """
        parcellated = self.predict_map(text).ravel()
        n_p = n_parcels if n_parcels is not None else self.config.get("n_parcels")
        if n_p is None:
            from ..parcellation import get_n_parcels
            n_p = get_n_parcels()
        if parcellated.shape[0] != n_p:
            raise ValueError(f"Expected {n_p} parcels, got {parcellated.shape[0]}")
        from ..parcellation import get_masker
        import nibabel as nib
        masker = get_masker(n_parcels=n_p)
        masker.fit()
        volume = masker.inverse_transform(parcellated.reshape(1, -1))
        nib.save(volume, output_path)
        return output_path
