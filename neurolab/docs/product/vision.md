# NeuroLab Vision

## One-line pitch

**â€œGoogle Earth for the Brainâ€** â€” natural-language questions â†’ evidence-backed brain maps, mechanistic context, and a 3D explorer, with clear separation of *what is known* vs *what is hypothesized*.

## What we deliver (MVP)

1. **Evidence-backed brain maps** (and explicitly labeled *hypothesis maps* when evidence is weak).  
2. **Mechanistic context** â€” why the map might look that way; what the evidence actually supports.  
3. **A 3D brain explorer** â€” cortex + subcortex/brainstem where possible.  
4. **End-to-end analytical protocols** â€” experiment + analysis design using a reproducible analysis engine or companion workflow.

## Strategic choices

- **Meta-analytic first:** Prioritize CBMA/IBMA and ready-to-map datasets over on-the-fly â€œraw BIDS â†’ fMRIPrepâ€ in MVP. Raw pipelines are Phase 2.  
- **Anti-blobology UX:** Every output is tagged with an **Evidence Tier** and includes a â€œHow to validate thisâ€ playbook so probabilistic maps are not misread as definitive results.  
- **Integration-friendly:** NeuroLab should work standalone or integrate cleanly into a larger agent / orchestration stack.

## Primary audience

Researchers, students, and technically curious users who understand neuroscience concepts (neurotransmitter systems, receptor pharmacology, functional connectivity) but are not necessarily neuroimaging engineers. They want to ask questions in plain language and get analyses and reports without writing Python/MATLAB.

## Operating model (current intent)

- **Core explorer service:** free / non-commercial, educational + personal research.  
- **AI layer:** users may provide their own API keys; the agent mediates tool use and produces reports.  
- **Commercialization boundary (possible future):** paid API for compute-heavy analysis can be offered **separately** from the open visualization layer. Governance tokens must not be framed as medical/clinical endorsement.  
- **Attribution:** build attribution into every export; downstream commercial reuse may affect attribution/compliance expectations.

---

**Canonical source:** [NeuroLab_Plugin_Spec_v0.3 Â§0â€“1](../../../docs/external-specs/NeuroLab_Plugin_Spec_v0.3.md).

