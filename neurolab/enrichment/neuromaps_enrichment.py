"""
Neuromaps-based enrichment: correlate a parcellated brain map with
neuromaps annotations (receptors, metabolism, etc.) in the same parcellation.

Expects cache built by scripts/build_neuromaps_cache.py:
  annotation_maps.npz (matrix: n_annotations x n_parcels)
  annotation_labels.pkl (list of label strings)
"""
import os
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


class NeuromapsEnrichment:
    """
    Correlate a parcellated brain map with neuromaps annotation maps.
    Matrix shape: (n_annotations, n_parcels). Same n_parcels as decoder (e.g. 400).
    """

    def __init__(
        self,
        cache_dir: str = "neurolab/data/neuromaps_cache",
        n_parcels: int = 400,
    ):
        self.n_parcels = n_parcels
        self.labels: List[str] = []
        self.matrix: np.ndarray  # (n_annotations, n_parcels)
        self._load(cache_dir)

    def _load(self, cache_dir: str) -> None:
        import pickle

        npz_path = os.path.join(cache_dir, "annotation_maps.npz")
        pkl_path = os.path.join(cache_dir, "annotation_labels.pkl")
        if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Neuromaps cache not found. Run: python neurolab/scripts/build_neuromaps_cache.py --cache-dir {cache_dir}"
            )
        data = np.load(npz_path)
        self.matrix = data["matrix"]
        with open(pkl_path, "rb") as f:
            self.labels = pickle.load(f)
        if self.matrix.shape[1] != self.n_parcels:
            raise ValueError(
                f"Neuromaps matrix has {self.matrix.shape[1]} parcels, expected {self.n_parcels}"
            )
        if len(self.labels) != self.matrix.shape[0]:
            self.labels = [f"ann_{i}" for i in range(self.matrix.shape[0])]

    def enrich(
        self,
        parcellated_activation: np.ndarray,
        method: str = "pearson",
    ) -> Dict:
        """
        Correlate input map with each annotation map.

        Returns:
            dict with by_layer["neuromaps"] = [{name, r, p}, ...],
            top_hits, layer_summary.
        """
        activation = np.asarray(parcellated_activation, dtype=np.float64).ravel()
        if activation.shape[0] != self.n_parcels:
            raise ValueError(f"Expected {self.n_parcels} parcels, got {activation.shape[0]}")

        results = []
        for i in range(self.matrix.shape[0]):
            row = self.matrix[i]
            valid = np.isfinite(activation) & np.isfinite(row)
            if valid.sum() < 20:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(activation[valid], row[valid])
            else:
                r, p = stats.spearmanr(activation[valid], row[valid])
            results.append({"name": self.labels[i], "r": float(r), "p": float(p)})
        results.sort(key=lambda x: abs(x["r"]), reverse=True)
        by_layer = {"neuromaps": results}
        top_hits = results[:10]
        layer_summary = {"neuromaps": float(np.mean([abs(x["r"]) for x in results])) if results else 0.0}
        return {"by_layer": by_layer, "top_hits": top_hits, "layer_summary": layer_summary}
