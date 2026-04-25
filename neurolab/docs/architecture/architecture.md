# NeuroLab Architecture (Baseline)

## Design principle

**Integration-friendly, NeuroLab as a reusable module.** NeuroLab can operate standalone or integrate with a larger orchestration or analysis stack when needed.

Core requirements:

- neuroscience-specific tooling requires dedicated code (map transforms, null models, neuro file formats)
- licensing requires separate data stores (ODbL vs CC0 vs BYOL)
- the viewer should consume manifests rather than embedding analysis logic

## High-level components

1. **User-facing interface**
   Plain-language queries, dataset lookups, map comparison requests, and viewer interactions.

2. **External orchestrator or analysis layer (optional)**
   Conversation handling, workflow coordination, or sandboxed execution when NeuroLab is embedded in a larger system.

3. **NeuroLab core**
   - tool registry + schemas for neuro map retrieval and analysis
   - dataset registry + license-aware data access
   - visualization manifest generation for 3D viewing

4. **NeuroLab front-end**
   3D brain explorer + slice viewer; provenance UI + Evidence Tier UI. Consumes manifest only.

5. **NeuroLab data services**
   License-separated stores (ODbL vs CC0 vs BYOL); caching + content-addressed storage.

## Logical flow

```text
User query
  -> NeuroLab core
     -> map retrieval / enrichment / comparison
     -> optional external analysis layer
     -> manifest + artifacts
     -> viewer or downstream consumer
```

## Minimal tool surface

- `neuro.search_literature(query, constraints)`
- `neuro.map_registry.search(query)`
- `neuro.cbma.run(spec)`
- `neuro.map.compare(map_a, map_b, null_model=...)`
- `neuro.viewer.manifest(maps, surfaces, overlays)`
- `neuro.license.check(assets)`

Example contract:

- `neuro.query_cbma(topic_or_condition, filters, output_spec) -> {artifacts, manifest}`
- `neuro.compare_maps(map_a, map_b, null_model, output_spec) -> {stats, figures, manifest}`
- `neuro.export_viewer_bundle(artifacts, atlas, space) -> {glb/nii, metadata}`

Every response includes a **manifest**: datasets + versions, licenses + attribution, query + filters, software versions + hashes, and transformation steps.

## Evidence Tier

- **Tier A:** Direct image evidence (IBMA / shared statistical maps)
- **Tier B:** Coordinate-based evidence (CBMA)
- **Tier C:** Predictive / model-based hypothesis (e.g. NeuroQuery)

All maps exposed to users must carry a tier; Tier B/C must include “How to validate” guidance.

## Where to go next

- [components.md](components.md)
- [data-flows.md](data-flows.md)
- [interfaces.md](interfaces.md)
- [adrs/README.md](adrs/README.md)

Historical architecture material is archived under `docs/external-specs/` and is not part of the default publish surface for this personal repo.

