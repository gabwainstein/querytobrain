# NeuroLab Product Docs

This folder holds the **product truth**: the normative documents that define what NeuroLab is, what it must achieve, and what it must not do.

If there is a conflict between a product doc and an architecture/implementation doc, the product doc wins unless it is explicitly revised through an intentional change process.

**Start with:** [docs/README.md](../README.md) (global docs map).

---

## Recommended reading order (product truth)

1. **Vision (why this exists)**  
   - [vision.md](vision.md)  
   - [problem-statement.md](problem-statement.md)

2. **Success (what â€œgoodâ€ means)**  
   - [success-criteria.md](success-criteria.md)

3. **Scope boundaries (what we own, what we donâ€™t)**  
   - [scope.md](scope.md)  
   - [non-goals.md](non-goals.md)  
   - [constraints.md](constraints.md)  
   - [assumptions.md](assumptions.md)  
   - [risks.md](risks.md)

4. **Users and end-to-end behaviors**  
   - [personas.md](personas.md) (end-users)  
   - [use-cases.md](use-cases.md)  
   - [user-stories.md](user-stories.md) ("As a... I want..." narratives mapped to use cases)  
   - [expert-personas.md](expert-personas.md) (domain experts for planning: Science, Application, AI, Coding)

5. **System requirements (SR/NFR)**  
   - [system-requirements.md](system-requirements.md)

---

## Normative invariants (must hold)

- **Traceability:** scope â†’ success criteria â†’ SRs â†’ use cases â†’ architecture â†’ implementation.
- **Evidence Tier disclosure:** every map shown to users has an Evidence Tier (A/B/C) and, for B/C, â€œHow to validateâ€ guidance.
- **Integration-friendly:** NeuroLab can run standalone or integrate into a larger orchestration / analysis stack without changing its core evidence and provenance guarantees.
- **Terminology (do not conflate):** **Evidence Tier** A/B/C = *evidence quality* (A: IBMA/image, B: CBMA/coords, C: predictive/hypothesis). **Access tier** 1/2 = *data access* (ready-to-go vs guided/BYOD). **Dataset priority** (e.g. Tier S/A/B in the implementation guide) = *ranking of datasets* for nootropics. Use Evidence Tier only for map quality in product/architecture; use access tier only in registry/implementation context.

---

## Where to go next

- Architecture: [docs/architecture/architecture.md](../architecture/architecture.md)  
- Data flows: [docs/architecture/data-flows.md](../architecture/data-flows.md)  
- Implementation: [docs/implementation/pipeline-slices.md](../implementation/pipeline-slices.md)


