# NeuroLab Issue Types and Metadata

## Purpose

Defines the **work type taxonomy** (Epic → Deliverable → Task) and **minimum GitHub metadata** so the backlog stays coherent for continuous flow. Use the issues system for all planned work.

---

## Always use the issues system

- **Before starting work:** Create or identify a GitHub issue (Epic, Deliverable, or Task). Use labels and relationships below.
- **In commits and PRs:** Reference the issue (e.g. `Fixes #123` or `Closes #123`) so work is traceable and the issue auto-closes on merge.
- **No ad-hoc PRs without an issue:** If a PR is opened for untracked work, create an issue first (or a quick Task), then add “Closes #N” to the PR.

---

## Work type taxonomy

### Epic

- **Meaning:** Long-lived capability/theme containing multiple Deliverables and/or Tasks.
- **Contents:** Scope, rationale, constraints, list of child issues.
- **Relationships:** Has sub-issues (Deliverables/Tasks).

### Deliverable

- **Meaning:** Concrete, reviewable set of artifacts (docs, architecture, protocol, slice implementation).
- **Contents:** Explicit artifacts, review/exit criteria, traceability to product requirements where applicable.
- **Relationships:** Sub-issue of an Epic when it contributes to that Epic.

### Task

- **Meaning:** Small unit of work that can be completed quickly and reviewed easily.
- **Contents:** Clear acceptance criteria or completion definition; links to artifacts if applicable.
- **Relationships:** May be sub-issue of a Deliverable or Epic.

---

## Labels (recommended)

- `work:epic`
- `work:deliverable`
- `work:task`
- `neurolab` (or equivalent to filter NeuroLab work in querytobrain)

Create these in the repo if they do not exist.

---

## Required metadata (must-fill)

- **Title:** Clear, concise.
- **Description:** Scope, acceptance criteria, link to product/architecture doc where relevant.
- **Labels:** At least one of `work:epic`, `work:deliverable`, `work:task`.
- **Assignee:** When known.
- **Links:** Parent/child issue links (e.g. “Part of Epic #X”, “Deliverables: #Y, #Z”).

---

## Relationship to other docs

- [Definition of Ready](definition-of-ready.md) — work should meet DoR before implementation starts.
- [Definition of Done](definition-of-done.md) — work is done when DoD is satisfied.
- [Documentation rules](documentation-rules.md) — doc-adding PRs must satisfy cross-link invariant.
