# NeuroLab Pipeline Slices (Implementation Plan)

## Purpose

Define thin vertical slices that can be implemented, tested, and shipped end-to-end while preserving:

- Evidence Tier disclosure and provenance
- License separation and attribution
- ResearchAgent plugin boundary (stable tool contract)

This document ties together:

- [Use cases](../product/use-cases.md)
- [System requirements](../product/system-requirements.md)
- [Interfaces](../architecture/interfaces.md)
- [ADRs](../architecture/adrs/README.md)

**Canonical sprint plan:** [NeuroLab_Plugin_Spec_v0.3 §10](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md). This section is an implementation timeline aligned with that spec.

---

## Slice template (normative)

Each slice must specify:

- **Goal**
- **Primary UCs / SRs**
- **Contracts touched**
- **Acceptance criteria** (verifiable)
- **Non-goals**
- **Primary risks**
- **Definition of Done** (link to [definition-of-done.md](../process/definition-of-done.md))

---

## Slice 0: Foundations (tool registry + plugin shell)

**Goal:** Plugin package that registers with ResearchAgent; tool stubs for `neuro.map_registry.search`, `neuro.viewer.manifest`; no real data yet.

**Primary UCs / SRs:** UC-1, UC-6; SR-6, NFR-1.

**Contracts touched:** Tool response (manifest stub), Evidence Tier placeholder.

**Acceptance criteria:**

- Plugin loads in ResearchAgent; orchestrator can route to `neuro.*` tools.
- Stub tools return valid manifest shape (layers can be empty).
- Evidence Tier field present in manifest.

**Non-goals:** Real map retrieval; 3D viewer; Data Analyst integration.

**Primary risks:** ResearchAgent plugin API drift. Mitigation: pin to documented contract; ADR if we diverge.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 1: Map registry + one dataset adapter

**Goal:** Curated map registry (in-memory or config); one adapter (e.g. Neurosynth or NeuroVault) returning real maps; license + attribution in manifest.

**Primary UCs / SRs:** UC-1, UC-3; SR-1, SR-2, SR-8, SR-9.

**Contracts touched:** Map registry entry, viewer manifest (real layers), provenance.

**Acceptance criteria:**

- Registry has at least one dataset with license and attribution.
- One end-to-end path: query → map(s) + manifest with Evidence Tier and attribution.
- License check blocks or allows per [constraints](../product/constraints.md).

**Non-goals:** Multiple adapters; CBMA run; 3D viewer UI.

**Primary risks:** API rate limits or schema changes. Mitigation: version in config; reason codes for failures.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 2: CBMA + Data Analyst integration

**Goal:** Data Analyst runs CBMA (e.g. NiMARE) from plugin-provided spec; results (figures + manifest) returned; Evidence Tier B assigned where appropriate.

**Primary UCs / SRs:** UC-1, UC-2; SR-1, SR-2, SR-6, SR-7, SR-8.

**Contracts touched:** Tool response (artifacts + manifest), run audit.

**Acceptance criteria:**

- `neuro.query_cbma(...)` (or equivalent) triggers Data Analyst run; map + manifest returned.
- Manifest includes algorithm, parameters, Tier, “How to validate” for Tier B.
- Run config/lineage traceable (SR-10).

**Non-goals:** Full protocol generation (UC-5); BYOL.

**Primary risks:** Sandbox environment mismatch. Mitigation: reference container or BYOE docs.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 3: 3D viewer (manifest-driven)

**Goal:** Web-based viewer that loads from manifest; displays layers; shows Evidence Tier and provenance panel.

**Primary UCs / SRs:** UC-6; SR-4, NFR-2.

**Contracts touched:** Viewer manifest (consumed by viewer).

**Acceptance criteria:**

- Viewer loads manifest (JSON); renders volume/surface layers.
- Evidence Tier badge and “What data generated this?” visible.
- No pipeline or DB access from viewer.

**Non-goals:** Offline support; tractography; DTI.

**Primary risks:** Performance with large volumes. Mitigation: level-of-detail or lazy load per manifest URLs.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 4: Map comparison + receptor overlay

**Goal:** `neuro.map.compare` and receptor overlay (e.g. Hansen); null model (e.g. spin) where applicable; interpretation checklist in report.

**Primary UCs / SRs:** UC-2, UC-3; SR-3, SR-2.

**Contracts touched:** Tool response, manifest (multi-layer overlay).

**Acceptance criteria:**

- Compare two maps; get stats + figures + manifest.
- Overlay receptor atlas with meta-analytic map; caveats in UI/report.
- Interpretation checklist present for Tier B/C.

**Non-goals:** User upload (BYOL) in same slice.

**Primary risks:** Null model implementation details. Mitigation: document method in manifest; link to methods.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 5: BYOL (upload + contextualize)

**Goal:** User can provide own map; stored in BYOL store; comparison/overlay with public maps; attribution correct and separate from ODbL/CC0.

**Primary UCs / SRs:** UC-4; SR-5, SR-9.

**Contracts touched:** Data Services boundary, manifest attribution.

**Acceptance criteria:**

- Upload or link → BYOL store only.
- Comparisons produce manifest with separate attribution for user map vs public.
- No commingling that would violate ODbL/CC0.

**Non-goals:** Full BIDS ingest; fMRIPrep.

**Primary risks:** License confusion. Mitigation: clear UX (“your data stays separate”); audit trail.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Slice 6: Protocol generation (hypothesis → experiment + analysis plan)

**Goal:** Data Analyst generates protocol (design, confounds, model, prereg snippet, sample-size heuristics); interpretation checklist; no medical advice.

**Primary UCs / SRs:** UC-5; SR-7.

**Contracts touched:** Report artifact; no new map contract.

**Acceptance criteria:**

- User request for “how to validate” or “design an experiment” → structured protocol.
- Checklist includes population/task/modality mismatch, confounds, falsifiability.
- No clinical or treatment recommendation.

**Non-goals:** Automated prereg submission; IRB guidance.

**Primary risks:** Overclaiming. Mitigation: system prompt + DoD review.

**DoD:** [definition-of-done](../process/definition-of-done.md).

---

## Relationship to canonical sprint plan

The [v0.3 spec §10](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md) defines Sprints 0–6 at a product level. These slices align as follows:

- **Sprint 0** → Slice 0 (foundations)
- **Sprint 1** → Slice 1 (map registry + one adapter)
- **Sprint 2** → Slice 2 (CBMA + Data Analyst)
- **Sprint 3** → Slice 3 (3D viewer)
- **Sprint 4** → Slice 4 (map comparison + overlay)
- **Sprint 5** → Slice 5 (BYOL)
- **Sprint 6** → Slice 6 (protocol generation) + onboarding/experiment design

Detailed week-by-week planning can live in the canonical spec or in runbooks under this folder.
