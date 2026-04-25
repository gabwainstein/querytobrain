# NeuroLab Implementation

This folder holds **implementation plans**: vertical slices, runbooks, and specs that tie product use cases and architecture contracts to buildable work.

## Entry points

- **Pipeline slices (MVP):** [pipeline-slices.md](pipeline-slices.md) — thin vertical slices with acceptance criteria and contract touchpoints.
- **Step-by-step build plan:** [enrichment-pipeline-build-plan.md](enrichment-pipeline-build-plan.md) — Phases 0–6: environment & packages, fetch data/model, build term maps cache, CognitiveDecoder, biological layer, UnifiedEnrichment, API. Ensures correct data fetch, packages, and dataset/model prep before coding.
- **Accuracy and testing:** [accuracy-and-testing.md](accuracy-and-testing.md) — How we define accuracy (text-to-brain: mean correlation; decoder; biological; guardrail), train/val/test protocol, and the exact process to run tests and reproduce numbers. Every pipeline’s evaluation is documented here.
- **Repo analysis (branch):** [NEUROLAB_REPO_ANALYSIS.md](NEUROLAB_REPO_ANALYSIS.md) — What the neurolab repo contains on `feature/neurolab-enrichment-pipeline`: enrichment pipeline, build vs query-to-map scripts, decoder_cache vs unified_cache, and reconciliation notes.
- **Repo coherence review:** [REPO_REVIEW.md](REPO_REVIEW.md) — Full review of docs, code, and scripts for consistency and coherence; findings and applied fixes.
- **NeuroSynth validation & semantic expansion:** [NEUROLAB_NEUROSYNTH_VALIDATION_AND_SEMANTIC_EXPANSION.md](NEUROLAB_NEUROSYNTH_VALIDATION_AND_SEMANTIC_EXPANSION.md) — Ontologies, Option A, query flow, two combine methods (neuroquery_style vs decompose_avg), optional ontology path, full local cache (setup_production, neuromaps fetch-all).
- **Related text→brain tools:** [related-text-to-brain-tools.md](related-text-to-brain-tools.md) — Text2Brain, Chat2Brain (open-source); comparison with NeuroLab; how they can complement (alternative backend, LLM query refinement).
- **Testing runbook:** [TESTING_RUNBOOK.md](TESTING_RUNBOOK.md) — How the pipeline works (one page); test order and dependencies; copy-paste commands for verify_* and E2E.
- **Build maps and training pipeline:** [BUILD_MAPS_AND_TRAINING_PIPELINE.md](BUILD_MAPS_AND_TRAINING_PIPELINE.md) — Step-by-step guide to building ontologies, neuromaps data, decoder/neuroquery/neurosynth caches, merge, neuromaps cache, expanded cache (ontology + neuromaps + receptor), and training the text-to-brain model. Use this to hand off to another agent or reproduce the full build. Includes **§14b KG-regularized contrastive loss** (optional architecture: soft KG constraint vs hard ontology-expansion targets; validate with `ontology_brain_correlation.py` first).
- **Training pipelines:** [TRAINING_PIPELINES.md](TRAINING_PIPELINES.md) — Neutral wrapper around the legacy trainer with preset pipelines for `fmri_cbma`, `gene_receptor`, and `multimodal` runs.
- **Preprocessing, normalization, averaging, weighting, and training:** [PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md](PREPROCESSING_NORMALIZATION_AND_TRAINING_PIPELINE.md) — Single reference for the full pipeline: parcellation/resampling, per-source normalization (global vs cortex/subcortex), NeuroVault averaging and grouping, QC, sample weights, merge (no re-normalization), source-weighted batch sampling, and per-sample loss weights during training.
- **Build the training dataset:** [BUILD_TRAINING_DATASET.md](BUILD_TRAINING_DATASET.md) — One-command and step-by-step: fully preprocessed, curated, normalized, averaged → merged_sources for training.
- **Training dataset build methods:** [TRAINING_DATASET_BUILD_METHODS.md](TRAINING_DATASET_BUILD_METHODS.md) — Thorough step-by-step documentation of all methods used to build the training dataset (parcellation, normalization, averaging, curation, merge).
- **Data preparation status:** [DATA_PREPARATION_STATUS.md](DATA_PREPARATION_STATUS.md) — What is ready (implemented / in place) vs what is still to do (step-by-step commands) before training.
- **Training readiness audit:** [TRAINING_READINESS_AUDIT.md](TRAINING_READINESS_AUDIT.md) — Verification that the training dataset and pipeline are ready to start training.

## Slice template (normative)

Each slice must specify:

- **Goal**
- **Primary UCs / SRs**
- **Contracts touched** (from [docs/architecture/interfaces.md](../architecture/interfaces.md))
- **Acceptance criteria** (verifiable)
- **Non-goals** (to prevent scope creep)
- **Primary risks** (link to [docs/product/risks.md](../product/risks.md))
- **Definition of Done** (link to [docs/process/definition-of-done.md](../process/definition-of-done.md))

## Adding implementation docs

- New runbooks or specs: add under `implementation/` and add an **upstream link** from this README or [docs/README.md](../README.md).
- Follow [documentation-rules](../process/documentation-rules.md): cross-link invariant, no orphans.
