"""
Hansen-style receptor enrichment: correlate a parcellated brain map with
receptor/transporter density maps (same parcellation, e.g. Schaefer 400).

Data source: PET maps (Hansen et al. 2022), not Allen gene expression.
- Hansen = in vivo PET receptor/transporter density (19 receptors, Schaefer-parcellated).
- Allen = gene expression; different modality; not used here.

Expects a CSV or NPZ with receptor x parcel matrix (n_receptors x n_parcels).
Also supports Hansen's native format: CSV with no header, shape (n_parcels x n_receptors)
e.g. receptor_data_scale400.csv from https://github.com/netneurolab/hansen_receptors (results/).
Hansen data is also available via neuromaps (build_neuromaps_cache.py).
If no path is given, placeholder (random) data is used — biological r will be low.
"""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


# Hansen atlas receptor names and systems (from implementation guide)
HANSEN_RECEPTORS = [
    ("D1", "Dopamine"), ("D2", "Dopamine"), ("DAT", "Dopamine"),
    ("5HT1A", "Serotonin"), ("5HT1B", "Serotonin"), ("5HT2A", "Serotonin"),
    ("5HT4", "Serotonin"), ("5HT6", "Serotonin"), ("5HTT", "Serotonin"),
    ("NET", "Norepinephrine"),
    ("alpha4beta2", "Acetylcholine"), ("M1", "Acetylcholine"), ("VAChT", "Acetylcholine"),
    ("mGluR5", "Glutamate"), ("NMDA", "Glutamate"),
    ("GABAA", "GABA"), ("H3", "Histamine"), ("CB1", "Cannabinoid"), ("MOR", "Opioid"),
]


class ReceptorEnrichment:
    """
    Correlate a parcellated brain map with receptor density maps.
    Matrix shape: (n_receptors, n_parcels). Same n_parcels as decoder (e.g. 400).
    """

    def __init__(
        self,
        receptor_matrix_path: Optional[str] = None,
        n_parcels: int = 400,
    ):
        self.n_parcels = n_parcels
        self.receptor_names: List[str] = []
        self.receptor_systems: List[str] = []
        self.matrix: np.ndarray  # (n_receptors, n_parcels)
        if receptor_matrix_path and os.path.exists(receptor_matrix_path):
            self._load(receptor_matrix_path)
        else:
            self._load_placeholder()

    def _load(self, path: str) -> None:
        if path.endswith(".npz"):
            data = np.load(path)
            self.matrix = data["matrix"]
            self.receptor_names = list(data["receptor_names"])
            self.receptor_systems = list(data["receptor_systems"]) if "receptor_systems" in data else [""] * len(self.receptor_names)
        else:
            # Hansen native format has no header: (n_parcels, n_receptors) e.g. (400, 19)
            df_no_header = pd.read_csv(path, header=None)
            if df_no_header.shape[0] == self.n_parcels and df_no_header.shape[1] == len(HANSEN_RECEPTORS):
                self.matrix = df_no_header.values.astype(np.float64).T  # -> (n_receptors, n_parcels)
                self.receptor_names = [r[0] for r in HANSEN_RECEPTORS]
                self.receptor_systems = [r[1] for r in HANSEN_RECEPTORS]
            else:
                df = pd.read_csv(path)
                if "receptor" in df.columns:
                    self.receptor_names = df["receptor"].astype(str).tolist()
                    self.receptor_systems = df["system"].astype(str).tolist() if "system" in df.columns else [""] * len(self.receptor_names)
                    num_cols = [c for c in df.columns if c not in ("receptor", "system") and (c.startswith("parcel_") or str(c).isdigit())]
                    if not num_cols:
                        num_cols = [c for c in df.columns if c not in ("receptor", "system")]
                    self.matrix = df[num_cols].values.astype(np.float64)
                else:
                    # Rows = receptors, columns = parcels
                    self.receptor_names = df.index.astype(str).tolist() if hasattr(df.index, "tolist") else list(range(len(df)))
                    self.receptor_systems = [""] * len(self.receptor_names)
                    self.matrix = df.values.astype(np.float64)
        if self.matrix.shape[1] != self.n_parcels:
            raise ValueError(f"Receptor matrix has {self.matrix.shape[1]} parcels, expected {self.n_parcels}")
        if len(self.receptor_names) != self.matrix.shape[0]:
            self.receptor_names = [f"R{i}" for i in range(self.matrix.shape[0])]
            self.receptor_systems = [""] * self.matrix.shape[0]

    def _load_placeholder(self) -> None:
        """Minimal placeholder so pipeline runs without real Hansen data."""
        self.receptor_names = [r[0] for r in HANSEN_RECEPTORS]
        self.receptor_systems = [r[1] for r in HANSEN_RECEPTORS]
        rng = np.random.default_rng(42)
        self.matrix = rng.standard_normal((len(self.receptor_names), self.n_parcels)).astype(np.float64)

    def enrich(
        self,
        parcellated_activation: np.ndarray,
        method: str = "pearson",
    ) -> Dict:
        """
        Correlate input map with each receptor map.

        Returns:
            dict with by_layer["receptors"] = [{name, system, r, p}, ...],
            top_hits = same sorted by |r|, layer_summary = {"receptors": mean_abs_r}.
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
            results.append({
                "name": self.receptor_names[i],
                "system": self.receptor_systems[i],
                "r": float(r),
                "p": float(p),
            })
        results.sort(key=lambda x: abs(x["r"]), reverse=True)
        by_layer = {"receptors": results}
        top_hits = results[:10]
        layer_summary = {"receptors": float(np.mean([abs(x["r"]) for x in results])) if results else 0.0}
        return {
            "by_layer": by_layer,
            "top_hits": top_hits,
            "layer_summary": layer_summary,
        }
