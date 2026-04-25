# NeuroLab Risks

## Product / UX risks

| Risk | Mitigation |
|------|------------|
| Users treat Tier B/C maps as definitive | Mandatory Evidence Tier labels; “How to validate” for B/C; system prompts that enforce cautious language. |
| Overclaiming mechanism from spatial overlap | Interpretation checklist in every report; explicit “spatial overlap ≠ mechanism” disclosure. |
| Confusion between “access tier” (data) and “Evidence Tier” (quality) | Terminology doc: Evidence Tier A/B/C = evidence quality; Tier 1/2 = data access; Tier S/A/B = dataset priority. |

## Technical risks

| Risk | Mitigation |
|------|------------|
| Neurosynth/NeuroVault/NeuroQuery API or schema changes ([Schema Index](../implementation/SCHEMA_INDEX.md)) | Adapters with version checks; fallbacks and error handling; document supported versions tied to the canonical schema definitions. |
| TEI/data format variations break section mapping or anchoring | Validation at ingest; reason codes for drops; ADR for canonical provenance anchor. |
| License commingling (ODbL vs CC0 vs BYOL) | Separate data stores or clear routing; license check before serving; attribution in every export. |
| AGPL scope creep | Clear boundary between AGPL and non-AGPL components; communicate via stable APIs. |

## Operational risks

| Risk | Mitigation |
|------|------------|
| Cost at scale (API, compute) | Capability first; test in small environments; optional paid API for heavy compute later. |
| Deployment fragmentation (BYOE vs hosted) | Reference container + docs; minimal tool contract so both paths use same plugin API. |

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 §11](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).
