# NeuroLab Scope

## In scope (MVP)

### A. Meta-analytic / atlas-first map retrieval

- Coordinate-based meta-analysis (CBMA) using curated databases (e.g. Neurosynth).
- Image-based meta-analysis (IBMA) where unthresholded maps exist (e.g. NeuroVault).
- Predictive text→map as a **hypothesis generator** only (e.g. NeuroQuery), never as “ground truth.”

### B. Map comparison + contextualization

- Cross-map similarity and enrichment using spatially-constrained nulls (spin tests / surrogate maps).
- Overlay biological ontologies (receptor density, gradients) with explicit caveats.

### C. Visualization

- Web-based 3D brain explorer: cortical surfaces (fsaverage/Conte69), subcortical meshes where licensing allows, volumetric slice views, surface projections.
- Provenance + Evidence Tier UI.

### D. Protocol generation

- “Hypothesis → experiment → analysis plan” via a reproducible analysis layer: design choices, confounds, statistical model, prereg snippets, sample-size heuristics (educational, non-medical).

### E. Integration surface

- Tool registry + dataset registry + license-aware access.
- Viewer manifest builder; minimal external tool API (`neuro.query_cbma`, `neuro.compare_maps`, `neuro.export_viewer_bundle`).

---

## Out of scope (MVP)

- Running fMRIPrep (or full preprocessing) “in chat time” for arbitrary OpenNeuro datasets.
- Clinical decision support, diagnosis, treatment recommendation, or individualized medical advice.
- Causal claims from purely spatial overlap unless supported by causal evidence.
- Providing or recommending controlled substances.

---

## Phase 2 (post-MVP, not in current scope)

- Raw pipeline mode: BIDS ingest, fMRIPrep/qsiprep/MRtrix/FSL, HPC/Kubernetes, caching.
- Multi-modal: DTI tractography, connectomes, task-fMRI contrasts, PET overlays.

---

Historical source material is archived under `docs/external-specs/` and is not part of the default publish surface for this personal repo.
