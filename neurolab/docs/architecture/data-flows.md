# NeuroLab Data Flows

## UC-1: Term/drug → brain map(s)

1. User asks in natural language.  
2. Orchestrator parses intent → routes to NeuroLab `map_registry.search` / literature or CBMA tools.  
3. Plugin queries dataset registry + Data Services (license check) → retrieves or generates map(s).  
4. Plugin assigns Evidence Tier, builds manifest.  
5. Response: map(s) + manifest + “How to validate” (if B/C) → Orchestrator → User; optional 3D viewer from manifest.

**Data:** Map registry metadata → NIfTI/GIFTI or references → manifest (JSON). No raw user PII in map path.

---

## UC-2: Drug vs drug comparison

1. User asks (e.g. modafinil vs caffeine).  
2. Orchestrator creates plan: retrieve map A, map B, optional receptor atlas.  
3. Plugin serves map A, B (and atlas if needed) to Data Analyst (or returns refs for BYOE).  
4. Data Analyst runs comparison (spatial correlation, difference map, receptor correlation) in sandbox → figures, stats, manifest.  
5. Plugin (or Data Analyst) adds interpretation checklist, Tier labels.  
6. Response: figures + report + manifest → User / viewer.

**Data:** Map refs → Data Analyst sandbox → derived figures + manifest. Provenance in manifest.

---

## UC-3: Receptor overlay

1. User asks (e.g. “Where are 5-HT2A receptors densest?” or “Overlay with working memory”).  
2. Orchestrator routes to NeuroLab.  
3. Plugin loads receptor atlas (e.g. Hansen) from Data Services; optionally combines with meta-analytic map.  
4. Manifest built (layers, transforms).  
5. Viewer renders overlay; Tier and caveats shown.

**Data:** Atlas + optional map → manifest → viewer. No analysis in viewer.

---

## UC-4: Upload + contextualize (BYOL)

1. User provides own map (upload or link).  
2. Plugin accepts via BYOL path; stored in license-separated store (no mix with ODbL/CC0).  
3. Comparison/overlay tools run against user map + public maps; attribution tracked.  
4. Response: comparison result + manifest with clear attribution for each source.

**Data:** User map → BYOL store. Public maps from Data Services. Combined only in derived artifacts with correct attribution.

---

## UC-5: Hypothesis → experiment + analysis plan

1. User asks for protocol to test a hypothesis (e.g. from Tier C map).  
2. Orchestrator routes to Data Analyst.  
3. Data Analyst generates protocol (design, confounds, model, prereg snippet, sample-size heuristics).  
4. Response: structured protocol; no medical advice. Interpretation checklist aligned with Evidence Tier.

**Data:** Hypothesis + context → Data Analyst → protocol document. No map data in protocol.

---

## UC-6: 3D explorer

1. User opens viewer (e.g. from a prior response or direct link to manifest).  
2. Front-End loads manifest (layers, transforms, provenance).  
3. Renders surfaces/volumes; shows Evidence Tier badge and “What data generated this?”  
4. No direct call to pipeline or database; all data from manifest.

**Data:** Manifest (JSON) → viewer. No server round-trip for map bytes after manifest load (unless viewer implements lazy load from URLs in manifest).

---

**See also:** [architecture.md](architecture.md), [components.md](components.md), [product use-cases](../product/use-cases.md).
