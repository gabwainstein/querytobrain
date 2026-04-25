# NeuroLab Assumptions

## Product assumptions

- **Users** are primarily researchers, students, and technically curious users interested in brain maps, receptors, and condition comparisons but not necessarily neuroimaging engineers.
- **An external orchestration or analysis layer** may be available, but NeuroLab should not depend on a single agent stack to remain usable.
- **Meta-analytic and pre-computed datasets** (Neurosynth, NeuroVault, NeuroQuery, Hansen receptor atlas, etc.) are sufficient for MVP value; raw BIDS preprocessing is not required for first release.
- **Evidence Tier UX** (A/B/C + â€œHow to validateâ€) is sufficient to curb overinterpretation when we cannot eliminate it entirely.
- **3D viewer** can be built from a manifest (layers, transforms, provenance); the viewer does not need to â€œknow neuroscience,â€ only render the manifest.

## Technical assumptions

- **NiMARE, neuromaps, nilearn, nibabel** (or equivalents) are acceptable dependencies for standalone execution, analysis sandboxes, and optional tool integrations.
- **Standard neuro formats** (NIfTI, GIfTI, MNI152, fsaverage) are the norm for MVP; we do not assume exotic or one-off formats.
- **Content-addressed caching** (hash of inputs/outputs) is feasible for derived maps and manifests.
- **License-separated data stores** (ODbL vs CC0 vs BYOL) can be implemented without commingling that would violate license terms.

## Operational assumptions

- **BYOE (bring-your-own-execution)** or **reference container** are acceptable deployment models; we do not assume a single hosted analysis cluster for all users.
- **External APIs** (Neurosynth, NeuroVault, NeuroQuery, etc.) remain available and reasonably stable for MVP.
- **Documentation-first** workflow is adopted: product and architecture docs are updated before code when scope or contracts change.

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md) and [neurolab_implementation_guide](../../../docs/external-specs/neurolab_implementation_guide.md).

