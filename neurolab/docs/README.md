# NeuroLab Docs Map (Human + Agent Navigation)

## Purpose

This is the **navigation map** for NeuroLab documentation. It lets humans and coding agents quickly answer:

- **What is the product truth?**
- **What is the architecture baseline?**
- **What do we implement first?**
- **Where are the process rules?**

---

## Recommended reading order (for a coding agent)

### 1) Product truth (normative inputs)

- [docs/product/README.md](product/README.md)
- [docs/product/vision.md](product/vision.md)
- [docs/product/problem-statement.md](product/problem-statement.md)
- [docs/product/success-criteria.md](product/success-criteria.md)
- [docs/product/scope.md](product/scope.md)
- [docs/product/constraints.md](product/constraints.md)
- [docs/product/non-goals.md](product/non-goals.md)
- [docs/product/assumptions.md](product/assumptions.md)
- [docs/product/risks.md](product/risks.md)
- [docs/product/personas.md](product/personas.md)
- [docs/product/use-cases.md](product/use-cases.md)
- [docs/product/expert-personas.md](product/expert-personas.md) (Science / Application / AI / Coding leads for planning)
- [docs/product/system-requirements.md](product/system-requirements.md)

### 2) Architecture baseline

- [docs/architecture/architecture.md](architecture/architecture.md)
- [docs/architecture/components.md](architecture/components.md)
- [docs/architecture/data-flows.md](architecture/data-flows.md)
- [docs/architecture/interfaces.md](architecture/interfaces.md)
- [docs/architecture/repo-layout.md](architecture/repo-layout.md)
- ADRs: [docs/architecture/adrs/README.md](architecture/adrs/README.md)

### 3) Implementation planning

- [docs/implementation/README.md](implementation/README.md)
- [docs/implementation/pipeline-slices.md](implementation/pipeline-slices.md)
- [docs/implementation/enrichment-pipeline-build-plan.md](implementation/enrichment-pipeline-build-plan.md) (step-by-step: packages, data fetch, cache, decode, biological, API)
- **Accuracy and testing:** [docs/implementation/accuracy-and-testing.md](implementation/accuracy-and-testing.md) â€“ How we define accuracy (mean correlation, train/val/test), evaluation protocol, and how to run tests for every pipeline.
- **Schema index:** [implementation/SCHEMA_INDEX.md](implementation/SCHEMA_INDEX.md) â€“ Pointers to dataset registry, manifest/tool response, pharma schema, ontology meta-graph, and deployment schemas.

### 4) Process rules (how we work)

- [docs/process/README.md](process/README.md)
- [docs/process/documentation-rules.md](process/documentation-rules.md)
- [docs/process/definition-of-ready.md](process/definition-of-ready.md)
- [docs/process/definition-of-done.md](process/definition-of-done.md)
- [docs/process/issue-types.md](process/issue-types.md)

---

## â€œWhere do I put X?â€

| Content | Location |
|--------|-----------|
| Product definition (vision, scope, non-goals) | [docs/product/](product/) |
| Architecture + decomposition | [docs/architecture/architecture.md](architecture/architecture.md) |
| Architecture decisions (ADRs) | [docs/architecture/adrs/](architecture/adrs/) |
| Implementation slices, runbooks, specs | [docs/implementation/](implementation/) |
| **Accuracy definitions and testing process** | [docs/implementation/accuracy-and-testing.md](implementation/accuracy-and-testing.md) |
| Process + governance | [docs/process/](process/) |
| Diagrams | [docs/architecture/diagrams/](architecture/diagrams/) (create as needed) |

---

## Legacy archive

- Older integration and planning documents are archived under `docs/external-specs/` and should be treated as legacy reference material, not current implementation guidance.

