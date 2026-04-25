"""
Unified enrichment: cognitive decode + biological (receptor + neuromaps) in one call.
Runs on the same parcellated brain map; returns cognitive terms, biological hits, and a text summary.
"""
from typing import Dict, List, Optional, Any

import numpy as np

from .cognitive_decoder import CognitiveDecoder
from .receptor_enrichment import ReceptorEnrichment
from .neuromaps_enrichment import NeuromapsEnrichment


class UnifiedEnrichment:
    """
    One entry point: enrich(parcellated_activation) returns
    cognitive + biological (receptor + optional neuromaps) + summary.
    """

    def __init__(
        self,
        cache_dir: str = "neurolab/data/decoder_cache",
        receptor_path: Optional[str] = None,
        neuromaps_cache_dir: Optional[str] = None,
        enable_cognitive: bool = True,
        enable_biological: bool = True,
        n_parcels: int = 392,
    ):
        self.n_parcels = n_parcels
        self.cognitive: Optional[CognitiveDecoder] = None
        self.receptor: Optional[ReceptorEnrichment] = None
        self.neuromaps: Optional[NeuromapsEnrichment] = None

        if enable_cognitive:
            try:
                self.cognitive = CognitiveDecoder(cache_dir=cache_dir)
            except FileNotFoundError:
                self.cognitive = None
        if enable_biological:
            self.receptor = ReceptorEnrichment(
                receptor_matrix_path=receptor_path,
                n_parcels=n_parcels,
            )
            if neuromaps_cache_dir:
                try:
                    self.neuromaps = NeuromapsEnrichment(
                        cache_dir=neuromaps_cache_dir,
                        n_parcels=n_parcels,
                    )
                except FileNotFoundError:
                    self.neuromaps = None

    @property
    def biological(self) -> Optional[Any]:
        """Backward compat: first biological enricher (receptor)."""
        return self.receptor

    def enrich(
        self,
        parcellated_activation: np.ndarray,
        cognitive_top_n: int = 20,
        biological_method: str = "pearson",
    ) -> Dict:
        """
        Run cognitive decode and biological enrichment (receptor + optional neuromaps); build summary.

        Returns:
            dict with keys: cognitive (optional), biological (merged receptor + neuromaps), summary.
        """
        result: Dict = {}
        activation = np.asarray(parcellated_activation, dtype=np.float64).ravel()
        if activation.shape[0] != self.n_parcels:
            raise ValueError(f"Expected {self.n_parcels} parcels, got {activation.shape[0]}")

        if self.cognitive:
            result["cognitive"] = self.cognitive.decode(
                parcellated_activation,
                top_n=cognitive_top_n,
            )

        biological_parts: List[Dict] = []
        if self.receptor:
            biological_parts.append(("receptor", self.receptor.enrich(parcellated_activation, method=biological_method)))
        if self.neuromaps:
            biological_parts.append(("neuromaps", self.neuromaps.enrich(parcellated_activation, method=biological_method)))

        if biological_parts:
            by_layer: Dict[str, list] = {}
            top_hits: List[Dict] = []
            layer_summary: Dict[str, float] = {}
            for name, res in biological_parts:
                by_layer.update(res.get("by_layer", {}))
                top_hits.extend(res.get("top_hits", []))
                layer_summary.update(res.get("layer_summary", {}))
            top_hits.sort(key=lambda x: abs(x["r"]), reverse=True)
            result["biological"] = {
                "by_layer": by_layer,
                "top_hits": top_hits[:15],
                "layer_summary": layer_summary,
            }

        result["summary"] = self._generate_summary(result)
        return result

    def _generate_summary(self, result: Dict) -> str:
        """Short text summary for LLM or UI."""
        lines = []
        if "cognitive" in result:
            top = result["cognitive"].get("top_terms", [])[:5]
            term_str = ", ".join([f"{t} (r={r:.2f})" for t, r in top])
            lines.append(f"Cognitive profile: {term_str}")
            cats = result["cognitive"].get("category_scores", {})
            if cats:
                top_cat = max(cats.items(), key=lambda x: x[1])
                lines.append(f"Dominant domain: {top_cat[0]} ({top_cat[1]:.2f})")
        if "biological" in result:
            bio = result["biological"]
            if "top_hits" in bio and bio["top_hits"]:
                hits = bio["top_hits"][:3]
                parts = []
                for r in hits:
                    label = r.get("name", "")
                    sys_label = r.get("system", "")
                    rr = r.get("r", 0)
                    if sys_label:
                        parts.append(f"{label} ({sys_label}, r={rr:.2f})")
                    else:
                        parts.append(f"{label} (r={rr:.2f})")
                lines.append(f"Biological enrichment: {', '.join(parts)}")
        return "\n".join(lines) if lines else "No enrichment data."
