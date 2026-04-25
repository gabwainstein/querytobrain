# NeuroLab Schema Index

This catalog lists every **structured definition or schema** used by NeuroLab so contributors know where to find canonical field definitions before editing code or docs. Link to this file from new specs when referencing a schema.

| Schema | Location | Purpose |
|--------|----------|---------|
| **Dataset registry entry** | `docs/external-specs/neurolab_implementation_guide.md` (��5) and `neurolab/docs/architecture/interfaces.md` | Fields for dataset id, modality, license, access tier, provenance hash; authoritative for registry tables. |
| **Viewer manifest & tool response** | `neurolab/docs/architecture/interfaces.md` | JSON structure for viewer layers, provenance, evidence_tier, software versions, and reason codes. |
| **Term object schema** | `neurolab/docs/implementation/NOOTROPICS_BIOHACKER_DESIGN.md` (��2) | Defines fields on the “term” entity (text, modality, evidence tier, source weights, embeddings). |
| **Router/Fusion return schema** | `neurolab/docs/implementation/ROUTER_FUSION_DESIGN.md` (��5) | Output contract for router + fusion layer (per-head contributions, confidence, manifest refs). |
| **Pharma collection schema** | `neurolab/data/neurovault_pharma_schema.json` + `neurolab/docs/implementation/NEUROVAULT_PHARMA_AUDIT.md` | Curated pharma collections, label prefixes, semantics, exclusions; consumed by `relabel_pharma_terms.py` and merge scripts. |
| **Term label quality / audits** | `neurolab/docs/implementation/TERM_LABEL_QUALITY_GUIDE.md`, `TERM_INFORMATIVENESS_AUDIT.md` | Heuristics for acceptable labels, subject-ID handling, placeholders. |
| **Ontology meta-graph schema** | `neurolab/docs/implementation/ONTOLOGY_ENRICHMENT_IMPLEMENTATION_CHECKLIST.md` & `ROUTER_FUSION_DESIGN.md` | Explains how ontology nodes/edges are stored (no merged schema; meta-graph linking ontologies and datasets). |
| **Cache provenance schema** | `neurolab/docs/implementation/CRITICAL_PATH_CACHES_SPEC.md` | Required metadata files alongside caches (term_sources, map types, sample weights, gradient components). |
| **Supabase deployment plan** | `neurolab/docs/implementation/SUPABASE_DEPLOYMENT_PLAN.md` | Maps these schemas to hosted DB tables, storage buckets, and API services. |

**How to use this index**

1. When a doc or script mentions “the schema,” link to the exact row above.
2. If you add a new structured artifact (JSON, table, manifest), update this file and cross-link it from `neurolab/docs/README.md`.
3. Keep JSON schemas versioned inside the repo (e.g., `neurolab/data/*.json`) and note required fields in the associated doc.

This avoids divergent definitions and gives operators and contributors a single place to confirm structure before touching pipelines or storage.
