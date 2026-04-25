# NeuroLab Documentation Rules (Normative)

## Purpose

Repository-wide rules for creating and changing documentation so the NeuroLab docs remain coherent, navigable, and agent-friendly. These rules apply to **all docs** under `neurolab/docs/` intended for `main`.

---

## 1) Cross-link invariant (required)

Whenever you **add** or **rename** a doc under `neurolab/docs/`, you must:

- Add at least one **upstream** link from a relevant parent/index doc (e.g. [docs/README.md](../README.md), [product/README.md](product/README.md), [architecture/architecture.md](architecture/architecture.md), [process/README.md](process/README.md)).
- Add at least one **downstream** link to relevant child docs where appropriate.
- Update any docs maps or indices that would otherwise point to a dead link.

This prevents orphan docs and keeps navigation deterministic for humans and agents.

---

## 2) Architecture docs stay implementation-agnostic

Architecture docs define boundaries, flows, and contracts — not code structure (file paths, class names). Implementation-specific choices belong in:

- **ADRs** ([docs/architecture/adrs/](../architecture/adrs/)) for durable decisions
- **Implementation** ([docs/implementation/](../implementation/)) for how to build

Exception: Architecture may reference component names and artifact contracts (e.g. manifest shape).

---

## 3) Material doc changes must be auditable

When a documentation change implies a behavior or contract change (e.g. new Evidence Tier rule, new manifest field, new success threshold):

- Link the change to an issue (Epic/Deliverable/Task) where applicable.
- Add rationale in the PR description.
- Consider an ADR if the change is a durable decision.

---

## 4) Product truth precedence (normative)

If docs conflict:

- **docs/product/** defines scope, non-goals, constraints, success criteria, and requirements.
- **docs/architecture/** must conform to product truth.
- **docs/implementation/** must conform to both.

If a lower-level doc needs to “win,” the product (or architecture) doc must be updated intentionally with rationale and traceability.

---

## 5) Contract docs are special (normative)

[docs/architecture/interfaces.md](../architecture/interfaces.md) is the canonical artifact contract source of truth. Changes to contracts should be reviewed with extra care: prefer additive changes; use stable reason codes for drops/backlogs; add or update tests when code follows.

---

## 6) Style (agent readability)

- Prefer **explicit headings**, short paragraphs, and bullet lists.
- Use **relative links** within this repo (no bare URLs for internal docs).
- Keep **normative** rules clearly labeled; separate from optional guidance.

---

## 7) Canonical reference docs (parent repo)

NeuroLab’s long-form canonical specs live in `docs/external-specs/`:

- [NeuroLab_Plugin_Spec_v0.3.md](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md)
- [neurolab_implementation_guide.md](../../../docs/external-specs/neurolab_implementation_guide.md)

When adding content that duplicates or extends these, link to the canonical source and note “Canonical source: …” at the end of the section. Prefer updating the canonical doc or adding a short pointer in `neurolab/docs/` rather than forking the content.
