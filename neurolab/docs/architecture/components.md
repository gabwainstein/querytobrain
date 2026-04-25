# NeuroLab Components

## Component ownership (who owns what)

| Component | Owner | Responsibility |
|-----------|--------|-----------------|
| **External orchestrator** | Optional external system | Conversation, planning, tool routing, and safety when NeuroLab is embedded in a larger stack. |
| **External analysis runner** | Optional external system | Sandboxed execution of CBMA, map comparison, or protocol-generation workflows. Returns figures, JSON, and manifests. |
| **NeuroLab core** | NeuroLab | Tool registry, dataset registry, license-aware access, viewer manifest builder. Dataset adapters (Neurosynth, NeuroVault, NeuroQuery, Hansen, etc.). |
| **NeuroLab Data Services** | NeuroLab | License-separated stores (ODbL / CC0 / BYOL); cache; content-addressed artifacts. |
| **NeuroLab Front-End** | NeuroLab | 3D brain explorer; consumes manifest only; Evidence Tier + provenance UI. |
| **Map registry** | NeuroLab | Curated registry: dataset id, modality, cognitive term mapping, license, transform space, provenance hash. |

## Boundaries

- **NeuroLab core â†” external orchestrator:** NeuroLab exposes tools and schemas (as defined in [Schema Index](../implementation/SCHEMA_INDEX.md)); an external orchestrator may invoke them. No orchestration logic is required inside NeuroLab.
- **NeuroLab core â†” analysis runner:** NeuroLab provides tool specs and data pointers; an external analysis runner may execute NiMARE/analysis in the userâ€™s or reference containerâ€™s environment.
- **Data Services â†” NeuroLab core:** NeuroLab queries/caches via Data Services; Data Services enforce license routing and do not commingle ODbL/CC0/BYOL in violation of terms.
- **Front-End â†” Back-end:** Front-End receives only manifest (JSON); no direct database or pipeline access. Rendering is client-side from manifest.

## External systems (out of our control)

- Neurosynth (API + data, ODbL).
- NeuroVault (API + data, CC0).
- NeuroQuery (hypothesis maps, Tier C).
- Hansen receptor atlas (or equivalent); BrainSpace, neuromaps.
- Optional external orchestration / analysis stack.

---

**See also:** [architecture.md](architecture.md), [data-flows.md](data-flows.md), [interfaces.md](interfaces.md).

