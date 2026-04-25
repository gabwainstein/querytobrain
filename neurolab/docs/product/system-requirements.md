# NeuroLab System Requirements

## Functional (MVP)

| ID | Requirement | Traceability |
|----|-------------|--------------|
| SR-1 | System shall return meta-analytic (CBMA/IBMA) or hypothesis (NeuroQuery) maps for user queries with &lt;5â€“15 s typical latency for retrieval + preview. | UC-1, UC-2 |
| SR-2 | Every map presented to the user shall have an Evidence Tier (A/B/C) and, for Tier B/C, â€œHow to validateâ€ guidance. | Vision, success-criteria |
| SR-3 | System shall support map comparison (spatial correlation, difference map, null model) and receptor overlay in standard space (MNI152, fsaverage/Conte69). | UC-2, UC-3, UC-4 |
| SR-4 | System shall produce a viewer manifest (layers, transforms, provenance) consumable by the 3D brain explorer. | UC-6 |
| SR-5 | System shall support BYOL (user-uploaded maps) with license-separated storage and correct attribution. | UC-4, constraints |
| SR-6 | NeuroLab shall support standalone execution and clean integration with external tool registries, dataset registries, and manifest builders. | Architecture, constraints |
| SR-7 | Protocol generation (hypothesis â†’ experiment â†’ analysis plan) shall be produced by a reproducible analysis layer with an interpretation checklist and no medical advice. | UC-5, non-goals |
| SR-8 | Export artifacts shall include provenance: dataset + version, algorithm + parameters, coordinate space, content hash where applicable. | Constraints, ADRs |
| SR-9 | License checks shall block or attribute correctly for ODbL/CC0/BYOL; no commingling that violates upstream terms. | Constraints, risks |
| SR-10 | System shall support audit/reconstruction: run config, versions, and artifact lineage traceable. | ADRs (audit) |

## Non-functional (MVP)

| ID | Requirement |
|----|-------------|
| NFR-1 | External integrations shall use a stable, documented tool contract (e.g. `neuro.query_cbma`, `neuro.compare_maps`, `neuro.export_viewer_bundle`). |
| NFR-2 | Viewer shall render from manifest only (no neuroscience logic in viewer); Evidence Tier and provenance UI required. |
| NFR-3 | Documentation shall be kept in sync with product and architecture (docs-first; cross-link invariant). |

---

Historical source material is archived under `docs/external-specs/` and is not part of the default publish surface for this personal repo.

