# NeuroLab Definition of Ready (DoR)

## Purpose

Defines when a unit of work (Epic, Deliverable, or Task) is **ready to begin** execution.

---

## Core readiness criteria (normative)

A unit of work is **Ready** when:

### A) Product alignment is explicit

- It references at least one product truth doc it serves:
  - [Scope](../product/scope.md)
  - [Success criteria](../product/success-criteria.md)
  - [System requirements](../product/system-requirements.md)
- It is consistent with [non-goals](../product/non-goals.md) and [constraints](../product/constraints.md).

### B) Acceptance criteria exist

- There are **measurable acceptance criteria**, written in a verifiable way.
- If acceptance depends on contracts or Evidence Tier, they are explicitly referenced:
  - [Interfaces](../architecture/interfaces.md)
  - ADRs in [docs/architecture/adrs/](../architecture/adrs/) if applicable.

### C) Scope is bounded

- The work states what is **in scope** and **out of scope** for the unit.
- It identifies expected artifacts (docs, code, tests, config templates).

### D) Dependencies and risks are acknowledged

- External dependencies (APIs, ResearchAgent version, runtimes) are identified.
- High-risk areas are noted and tied to [Risks](../product/risks.md).

### E) Ownership and execution plan

- An accountable owner is assigned (or acknowledged).
- A “first slice” or first step exists so execution can start without ambiguity.

---

## Relationship to other docs

- [Definition of Done](definition-of-done.md)
- [Issue types](issue-types.md)
- [Documentation rules](documentation-rules.md)
