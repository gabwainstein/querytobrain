"""Preset training pipelines for data-regime-specific MLP runs.

These presets keep the legacy trainer intact while offering a cleaner front door
for common workflows:

- ``fmri_cbma``: cognitive / CBMA-heavy training
- ``gene_receptor``: curated gene + receptor / neuromaps emphasis
- ``multimodal``: CBMA + gene/receptor + KG / ontology alignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TrainingPipeline:
    name: str
    description: str
    args: tuple[str, ...] = field(default_factory=tuple)

    def argv(self, extra_args: Iterable[str] | None = None) -> list[str]:
        out = list(self.args)
        if extra_args:
            out.extend(list(extra_args))
        return out


FMRI_CBMA = TrainingPipeline(
    name="fmri_cbma",
    description="CBMA / fMRI-focused MLP training with the CBMA auxiliary head.",
    args=(
        "--cache-dir",
        "neurolab/data/merged_sources_cbma",
        "--output-dir",
        "neurolab/data/sweeps/personal/fmri_cbma_default",
        "--encoder",
        "openai",
        "--encoder-model",
        "text-embedding-3-large",
        "--expand-abbreviations",
        "--epochs",
        "30",
        "--batch-size",
        "64",
        "--lr",
        "0.001",
        "--dropout",
        "0.2",
        "--weight-decay",
        "0.00001",
        "--use-cbma-head",
        "--cbma-head-sources",
        "cbma_pmc",
        "cbma_pmc_native",
        "--cbma-contrastive-aux",
        "--cbma-contrastive-lambda",
        "0.2",
    ),
)


GENE_RECEPTOR = TrainingPipeline(
    name="gene_receptor",
    description="Curated gene/receptor training with genes on the main head.",
    args=(
        "--cache-dir",
        "neurolab/data/merged_sources_cbma_abagen_tiered_nootropicpriority",
        "--output-dir",
        "neurolab/data/sweeps/personal/gene_receptor_default",
        "--encoder",
        "openai",
        "--encoder-model",
        "text-embedding-3-large",
        "--expand-abbreviations",
        "--epochs",
        "30",
        "--batch-size",
        "64",
        "--lr",
        "0.001",
        "--dropout",
        "0.2",
        "--weight-decay",
        "0.00001",
        "--receptor-ranking-loss",
        "--receptor-ranking-loss-lambda",
        "0.05",
        "--receptor-ranking-loss-type",
        "cosine",
        "--receptor-ranking-bank-sources",
        "neuromaps",
        "receptor",
        "neuromaps_residual",
        "receptor_residual",
    ),
)


MULTIMODAL = TrainingPipeline(
    name="multimodal",
    description="Multimodal training with semantic KG context, ontology alignment, and CBMA auxiliary supervision.",
    args=(
        "--cache-dir",
        "neurolab/data/merged_sources_cbma_abagen_tiered_nootropicpriority",
        "--output-dir",
        "neurolab/data/sweeps/personal/multimodal_default",
        "--encoder",
        "openai",
        "--encoder-model",
        "text-embedding-3-large",
        "--expand-abbreviations",
        "--epochs",
        "50",
        "--batch-size",
        "64",
        "--lr",
        "0.001",
        "--dropout",
        "0.2",
        "--weight-decay",
        "0.00001",
        "--kg-context-hops",
        "2",
        "--kg-context-mode",
        "semantic",
        "--kg-context-style",
        "triples",
        "--kg-semantic-top-k",
        "5",
        "--kg-sim-floor",
        "0.4",
        "--kg-max-triples",
        "15",
        "--use-cbma-head",
        "--cbma-head-sources",
        "cbma_pmc",
        "cbma_pmc_native",
        "--cbma-contrastive-aux",
        "--cbma-contrastive-lambda",
        "0.2",
        "--ontology-alignment-loss",
        "--ontology-alignment-lambda",
        "0.15",
        "--ontology-alignment-proj-dim",
        "256",
        "--kg-teacher-distillation",
        "--kg-teacher-lambda",
        "0.015",
        "--kg-teacher-gamma",
        "0.7",
        "--kg-teacher-max-hops",
        "2",
        "--kg-teacher-learnable-rel-weights",
        "--receptor-ranking-loss",
        "--receptor-ranking-loss-lambda",
        "0.05",
        "--receptor-ranking-loss-type",
        "cosine",
        "--use-ontology-retrieval-augmentation",
        "--ontology-retrieval-cache-dir",
        "neurolab/data/merged_sources_cbma",
        "--ontology-retrieval-alpha",
        "0.3",
    ),
)


PIPELINES: dict[str, TrainingPipeline] = {
    FMRI_CBMA.name: FMRI_CBMA,
    GENE_RECEPTOR.name: GENE_RECEPTOR,
    MULTIMODAL.name: MULTIMODAL,
}

