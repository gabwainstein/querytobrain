"""
Composite distance/similarity from the triad regression (hop + embedding distance + interaction).

The regression (regress_brain_r_on_hop_and_embedding.py) fits:
  brain_r ~ intercept + coef_hop*hop + coef_emb*emb_dist + coef_int*hop*emb_dist

This module provides:
  - load_regression_coefficients(path) -> dict
  - predicted_r(hop, emb_dist, coef) -> predicted brain-map correlation (composite similarity)
  - composite_distance(hop, emb_dist, coef) -> 1 - predicted_r (higher = more distant)

Use composite similarity as the target in pairwise/contrastive loss, or composite_distance
for ranking/sampling. The interaction term means KG proximity and embedding proximity
mutually validate: close in both -> higher predicted r; disagreement -> lower.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_regression_coefficients(path: str | Path) -> dict[str, float]:
    """Load coefficients from brain_r_hop_embedding_regression.json."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data["coefficients"]


def predicted_r(
    hop: float,
    emb_dist: float,
    coef: dict[str, float],
) -> float:
    """
    Predicted brain-map Pearson r for a pair given hop and embedding distance.

    This is the composite similarity: higher = more similar (higher expected r).
    """
    return (
        coef["intercept"]
        + coef["hop"] * hop
        + coef["emb_dist"] * emb_dist
        + coef["hop_emb_dist"] * (hop * emb_dist)
    )


def composite_distance(
    hop: float,
    emb_dist: float,
    coef: dict[str, float],
) -> float:
    """
    Composite distance: 1 - predicted_r. Higher = more distant.

    Use for ranking or as a single scalar distance combining KG and embedding.
    """
    return 1.0 - predicted_r(hop, emb_dist, coef)
