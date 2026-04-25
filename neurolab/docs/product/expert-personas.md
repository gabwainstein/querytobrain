# Expert Personas: Domain Owners for the Enrichment Pipeline

These personas represent **world-expert archetypes** for each domain involved in the brain map enrichment pipeline. Use them to assign ownership, review criteria, and "who would sign off" when making the plan and executing each step. They are not end-users (see [personas.md](personas.md)); they are the **authority** for science, application, AI, and coding decisions.

---

## 1. Science Lead â€” Neuroscience & Methods

**Archetype:** Senior cognitive neuroscientist / neuroimaging methodologist. Has published CBMA/IBMA, reverse inference, and spatial statistics. Knows NeuroSynth, NiMARE, neuromaps, and Evidence Tier literature inside out.

**Owns / decides:**
- Validity of **forward mapping** (text â†’ brain map): Is the semantic/NeuroQuery model appropriate for the claims we make? When do we label Tier A vs B vs C?
- **Reverse inference** (map â†’ cognitive terms): CorrelationDecoder vs pre-computed NeuroQuery term maps â€” tradeoffs, when to use which, how to phrase limitations.
- **Biological enrichment**: Which receptor/oscillation/metabolism/hierarchy maps to include; how to interpret spatial correlations; when to use spin tests / Moran nulls vs parametric p-values.
- **Interpretation guardrails**: What "how to validate" must say; what we never claim (causation, clinical); cognitive term categories and exclusion lists.
- **Provenance and citation**: What goes in the manifest; dataset versions; algorithm parameters; attribution text.

**Review gate:** "Would this pass peer review in a methods-heavy journal?" â€” not that we publish it, but that the science is defensible.

**Pipeline steps they lead:** Textâ†’map validity; cognitive decode method choice; biological layer catalog and interpretation; Evidence Tier assignment; interpretation checklist content.

---

## 2. Application Lead â€” Product & Use Cases

**Archetype:** Product owner for a scientific or neuroscience application. Understands who uses the tool (students, researchers, scientists), what jobs-to-be-done are, and how this fits the broader product ecosystem.

**Owns / decides:**
- **User stories and priorities**: Which enrichment outputs matter most for MVP (cognitive word cloud? receptor bar chart? hierarchy position?); order of build (e.g. cognitive decode before full neuromaps, or vice versa).
- **Use case fit**: Does the pipeline answer "noradrenergic modulation of attention", "psilocybin effects on consciousness", "modafinil vs caffeine" in a way that matches [use-cases.md](use-cases.md)?
- **Integration surface**: How the enrichment pipeline plugs into external tools, manifests, or an agent layer; what the interface says and doesn't say; UX for Evidence Tier and "how to validate".
- **Scope and non-goals**: Keeping MVP bounded (no clinical advice, no raw fMRIPrep in chat); when to add "rigorous mode" (NiMARE CorrelationDecoder) vs stay fast (NeuroQuery only).
- **Audience fit**: Wording, examples, and defaults that resonate with neuroscience users without overclaiming.

**Review gate:** "Would a researcher or student get value without being misled?"

**Pipeline steps they lead:** Feature order; API shape from a product perspective; copy and disclosures; integration with orchestrator and Data Analyst workflows.

---

## 3. AI Lead â€” Models, Semantic Maps & Agents

**Archetype:** ML/NLP engineer or research scientist focused on semantic models, brainâ€“language mapping, and agentic tool use. Knows NeuroQuery, embedding-based brain maps, and how LLMs should call tools and format outputs.

**Owns / decides:**
- **Semantic brain map model**: How text â†’ brain map is produced (NeuroQuery, custom embeddings, or hybrid); resolution, parcellation, and format that downstream decode and enrichment expect.
- **Cognitive decoder inputs/outputs**: Interface between the brain map and the decoder (parcellated vector, space, normalization); what the LLM or agent receives (top terms, categories, summary string).
- **Tool schemas for external integration layers**: How `neuro.*` tools are named, parameterized, and what they return (artifacts, manifest, summary); how the agent decides when to call enrichment vs map-only. See [Schema Index](../implementation/SCHEMA_INDEX.md) for canonical tool/manifest fields.
- **Prompting and guardrails**: System prompts that enforce Evidence Tier language, "how to validate", and no overclaiming; how the agent summarizes enrichment results for the user.
- **Caching and performance**: What to precompute (term maps, annotation matrix); what runs at query time; latency budget for real-time vs "advanced analysis" paths.

**Review gate:** "Is the model and tool design correct, and does the agent use it safely?"

**Pipeline steps they lead:** Textâ†’map pipeline; decoder input contract; tool API design; agent prompts and response formatting; precompute vs on-demand strategy.

---

## 4. Coding Lead â€” Implementation & Platform

**Archetype:** Senior software engineer who ships scientific pipelines: Python (NiMARE, neuromaps, nilearn, NeuroQuery), APIs, and optionally frontend. Knows the stack, testing, and how to keep the codebase maintainable and deployable.

**Owns / decides:**
- **Implementation structure**: Where `CognitiveDecoder`, `MultimodalEnrichment`, `UnifiedEnrichment` live; how they're tested; dependency boundaries (plugin vs Data Analyst vs viewer).
- **Data and cache layout**: Parcellation (e.g. Schaefer 400); cache dirs for term maps and annotation matrix; versioning and invalidation.
- **API and contracts**: Endpoints, request/response schemas, and alignment with [interfaces.md](../architecture/interfaces.md) (manifest, Evidence Tier, provenance); cross-check against [Schema Index](../implementation/SCHEMA_INDEX.md) when fields change.
- **Performance and robustness**: Sub-second decode; handling missing maps or API failures; reason codes and logging for debugging.
- **Frontend (if in scope)**: Word cloud, category bars, enrichment summary; wiring to backend; Evidence Tier and provenance in the UI.
- **CI, packaging, and runbooks**: How to run the pipeline locally and in a container; how Data Analyst or the plugin invokes it.

**Review gate:** "Can we ship this, maintain it, and onboard another developer?"

**Pipeline steps they lead:** All implementation tasks; code review; tests; docs and runbooks; deployment and integration with external runtimes.

---

## How to use these in the plan

- **Each workstream or slice** can be assigned a **primary** expert persona (and optionally a reviewer from another domain).
- **Sign-off:** For science-sensitive changes (new method, new claim, Evidence Tier rule), Science Lead must agree. For product/UX, Application Lead. For model/tools, AI Lead. For implementation, Coding Lead.
- **Conflict:** Product truth ([scope](scope.md), [non-goals](non-goals.md)) wins; then Science (interpretation and methods); then Application (priority and UX); then AI and Coding (how to build it).

---

## Summary table

| Persona           | Domain        | Leads / owns                                                                 | Review gate                          |
|-------------------|---------------|-------------------------------------------------------------------------------|--------------------------------------|
| **Science Lead**  | Neuroscience  | Forward/reverse inference validity, enrichment catalog, Evidence Tier, interpretation | Defensible methods; no overclaim    |
| **Application Lead** | Product    | User stories, use cases, integration, scope, audience UX          | Value without misleading             |
| **AI Lead**       | Models & agents | Semantic brain map, decoder contract, tool schemas ([Schema Index](../implementation/SCHEMA_INDEX.md)), prompts, caching        | Correct and safe agent use           |
| **Coding Lead**   | Implementation | Code structure, API, cache, tests, frontend, CI/deploy                       | Shipable and maintainable            |

---

**See also:** [personas.md](personas.md) (end-users), [use-cases.md](use-cases.md), [cognitive_decoding_addendum](../../../docs/external-specs/cognitive_decoding_addendum.md).

