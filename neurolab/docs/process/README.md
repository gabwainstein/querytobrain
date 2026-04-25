# NeuroLab Process Docs

This folder defines **how we work**: documentation rules, definition of ready/done, and issue types. These apply to all work intended to land on `main` for the NeuroLab subproject.

## Key docs

- [documentation-rules.md](documentation-rules.md) — Cross-link invariant, product precedence, no orphans. **Normative** for all docs.
- [definition-of-ready.md](definition-of-ready.md) — When a unit of work is ready to start.
- [definition-of-done.md](definition-of-done.md) — When a unit of work is complete and accepted.
- [issue-types.md](issue-types.md) — Epic → Deliverable → Task; GitHub labels and metadata.

## Principles

- **Docs first:** Product and architecture docs are updated before code when scope or contracts change.
- **Traceability:** Work links to product/architecture (scope, SRs, UCs, interfaces).
- **No orphan docs:** New or renamed docs get upstream links and index updates (see documentation-rules).
- **Product precedence:** If docs conflict, product wins; lower-level docs are updated intentionally with rationale.

## Relationship to parent repo

Process here applies to the **NeuroLab subproject** under `querytobrain`. If `querytobrain` has its own process docs (e.g. in `.github/` or `docs/`), align with them where appropriate; NeuroLab-specific rules (e.g. Evidence Tier in DoD) stay here.
