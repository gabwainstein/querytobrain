# NeuroLab Architecture Decision Records (ADRs)

## Purpose

This directory contains **Architecture Decision Records** for decisions that turn the implementation-agnostic architecture into an implementable system while preserving:

- Evidence Tier disclosure and â€œHow to validateâ€ for Tier B/C
- License separation and attribution
- Integration boundary (orchestration / analysis outside NeuroLab when used as a module)
- Auditability and provenance

Write an ADR when a decision:

- Introduces or changes a **contract boundary** (see [interfaces.md](../interfaces.md))
- Affects **provenance, validation, Evidence Tier, or license handling**
- Changes **source of truth** for registry, manifest, or export
- Introduces or removes **external dependencies** (APIs, ontologies, runtimes)
- Has meaningful **cost, performance, or operational** impact

If a decision can be reversed trivially without downstream impact, use a normal PR description instead.

---

## Status vocabulary

- **Proposed** â€” draft under discussion
- **Accepted** â€” authoritative for implementation
- **Superseded** â€” replaced by a newer ADR
- **Rejected** â€” considered and rejected (kept for context)
- **Deprecated** â€” do not use for new work

---

## Naming and numbering

- Format: `ADR-0001-short-title.md`
- Numbers are monotonically increasing; do not reuse.
- If an ADR is replaced, mark the old one **Superseded** and link to the new one.

---

## ADR index

*(Add new ADRs to this list as they are created.)*

- No ADRs yet. Use [adr-template.md](adr-template.md) to create the first.

---

## Cross-link rules

- ADRs must link to motivating product docs: [scope](../../product/scope.md), [constraints](../../product/constraints.md), [system-requirements](../../product/system-requirements.md).
- ADRs that touch contracts must link to [interfaces.md](../interfaces.md) and [data-flows.md](../data-flows.md).

