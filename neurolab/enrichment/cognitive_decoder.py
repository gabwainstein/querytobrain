"""
Cognitive term decoder using pre-computed NeuroQuery term maps.

Given a brain activation map (parcellated, e.g. Schaefer 400), finds the top
cognitive/functional terms from the neuroscience literature with similar spatial
patterns. Uses cache built by scripts/build_term_maps_cache.py (Phase 2).

NeuroQuery 1.1.x: vocabulary from model.vectorizer.get_feature_names();
transform([term]) returns dict with "brain_map" (list of images).
"""
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# Terms to exclude — same as build script
EXCLUDE_TERMS = {
    "participants", "studies", "results", "analysis", "data",
    "significant", "compared", "group", "effects", "task",
    "conditions", "trials", "research", "findings", "methods",
    "study", "reported", "brain", "regions", "cortex",
    "activation", "activity", "increased", "decreased",
    "greater", "response", "left", "right", "bilateral",
    "anterior", "posterior", "dorsal", "ventral",
    "patients", "controls", "healthy", "clinical",
}

# Curated categories for grouping (optional)
TERM_CATEGORIES = {
    "attention": ["attention", "attentional", "vigilance", "alerting", "orienting", "selective attention", "sustained attention"],
    "memory": ["memory", "working memory", "episodic", "encoding", "retrieval", "recognition", "recall", "hippocampal"],
    "executive": ["executive", "inhibition", "control", "conflict", "switching", "planning", "decision", "cognitive control"],
    "language": ["language", "semantic", "speech", "reading", "word", "sentence", "comprehension", "verbal", "lexical"],
    "motor": ["motor", "movement", "hand", "finger", "action", "saccade", "eye movement", "grasping"],
    "emotion": ["emotion", "fear", "reward", "punishment", "valence", "arousal", "affective", "threat", "anxiety", "happiness"],
    "social": ["social", "face", "theory of mind", "empathy", "mentalizing", "person", "expression"],
    "visual": ["visual", "perception", "object", "face", "scene", "spatial", "motion", "color"],
    "auditory": ["auditory", "sound", "music", "tone", "pitch", "listening", "acoustic"],
    "default_mode": ["default", "resting", "self", "autobiographical", "mind wandering", "introspection"],
    "pain": ["pain", "nociceptive", "somatosensory", "thermal"],
}


class CognitiveDecoder:
    """Decode brain maps into cognitive/functional terms (reverse inference)."""

    def __init__(self, cache_dir: str = "neurolab/data/decoder_cache"):
        self.cache_dir = cache_dir
        self.n_parcels = None  # set from cache (400 or 414)
        self._load_cache()

    def _load_cache(self) -> None:
        cache_path = os.path.join(self.cache_dir, "term_maps.npz")
        vocab_path = os.path.join(self.cache_dir, "term_vocab.pkl")
        if not os.path.exists(cache_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Cache not found. Run: python neurolab/scripts/build_term_maps_cache.py "
                f"--cache-dir {self.cache_dir}"
            )
        data = np.load(cache_path)
        self.term_maps = data["term_maps"]
        self.n_parcels = self.term_maps.shape[1]
        if "term_maps_z" in data:
            self.term_maps_z = data["term_maps_z"]
        else:
            self.term_maps_z = np.nan_to_num(stats.zscore(self.term_maps, axis=1))
        with open(vocab_path, "rb") as f:
            self.vocabulary = pickle.load(f)
        assert len(self.vocabulary) == self.term_maps.shape[0]

    def decode(
        self,
        parcellated_activation: np.ndarray,
        method: str = "pearson",
        top_n: int = 30,
    ) -> Dict:
        """
        Decode a parcellated brain map into cognitive terms.

        Args:
            parcellated_activation: (n_parcels,) array, e.g. (400,)
            method: "pearson", "spearman", or "cosine"
            top_n: number of top terms to return

        Returns:
            dict with top_terms, all_correlations, category_scores, word_cloud_data, n_terms_evaluated
        """
        activation = np.asarray(parcellated_activation, dtype=np.float64).ravel()
        if activation.shape[0] != self.n_parcels:
            raise ValueError(f"Expected {self.n_parcels} parcels, got {activation.shape[0]}")

        if method == "cosine":
            act_norm = activation / (np.linalg.norm(activation) + 1e-10)
            map_norms = np.linalg.norm(self.term_maps, axis=1, keepdims=True) + 1e-10
            maps_norm = self.term_maps / map_norms
            similarities = maps_norm @ act_norm
            correlations = dict(zip(self.vocabulary, similarities.tolist()))
        elif method == "pearson":
            act_z = np.nan_to_num(stats.zscore(activation))
            rs = (self.term_maps_z @ act_z) / self.n_parcels
            correlations = dict(zip(self.vocabulary, rs.tolist()))
        elif method == "spearman":
            act_rank = stats.rankdata(activation)
            act_z = np.nan_to_num(stats.zscore(act_rank))
            map_ranks = np.apply_along_axis(stats.rankdata, 1, self.term_maps)
            maps_z = np.nan_to_num(stats.zscore(map_ranks, axis=1))
            rs = (maps_z @ act_z) / self.n_parcels
            correlations = dict(zip(self.vocabulary, rs.tolist()))
        else:
            raise ValueError(f"method must be pearson, spearman, or cosine, got {method}")

        sorted_terms = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        category_scores: Dict[str, float] = {}
        for category, keywords in TERM_CATEGORIES.items():
            cat_rs = [
                r for term, r in correlations.items()
                if any(kw in term.lower() for kw in keywords)
            ]
            if cat_rs:
                positive_rs = [r for r in cat_rs if r > 0]
                category_scores[category] = float(np.mean(positive_rs)) if positive_rs else 0.0

        word_cloud_data: List[Tuple[str, float]] = [
            (term, max(0, r))
            for term, r in sorted_terms[:100]
            if r > 0.05
        ]

        return {
            "top_terms": sorted_terms[:top_n],
            "all_correlations": correlations,
            "category_scores": category_scores,
            "word_cloud_data": word_cloud_data,
            "n_terms_evaluated": len(self.vocabulary),
        }
