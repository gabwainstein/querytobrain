# NeuroLab Definition of Done (DoD)

## Purpose

Defines when a unit of work is **complete and accepted**.

---

## Core completion criteria (normative)

A unit of work is **Done** when all of the following are true:

### A) Acceptance criteria are satisfied

- The acceptance criteria stated in the issue are met.
- If the work introduced measurable criteria, the evaluation method exists (even if minimal).

### B) Artifacts are complete and cross-linked

- All required artifacts (docs, code, tests, config templates) are present.
- Docs maps and indices were updated so nothing is orphaned ([documentation-rules](documentation-rules.md)).
- If contracts were introduced or changed, [interfaces.md](../architecture/interfaces.md) is updated.

### C) Review and decision capture

- The PR is reviewed and review is recorded.
- Durable decisions are captured as ADRs when appropriate ([docs/architecture/adrs/](../architecture/adrs/)).

### D) Traceability is preserved

- Issue → PR → artifacts linkage exists.
- The PR summary makes it obvious what changed and why.

### E) Safety and quality gates are respected

- **Evidence Tier:** Every user-facing map has Tier and, for B/C, “How to validate” (no bypass).
- **License/attribution:** No commingling that violates ODbL/CC0/BYOL; attribution present where required.
- If behavior changes could impact export validity or provenance, a regression check exists or is explicitly backlogged with an issue.

---

## Relationship to other docs

- [Definition of Ready](definition-of-ready.md)
- [Issue types](issue-types.md)
- [Documentation rules](documentation-rules.md)
