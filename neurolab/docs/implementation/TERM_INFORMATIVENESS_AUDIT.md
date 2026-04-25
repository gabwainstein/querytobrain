# Term Informativeness Audit

**Criterion:** For each map, we must know (or recover) what it means and what the contrast context is: drug in n-back? task itself? group average of what task? meta-analysis of what?

If we cannot recover this, the term is uninformative and should be discarded.

---

## 0. NeuroVault label improvements (implemented)

**Problem:** NeuroVault terms had `[col6618]` prefixes (meaningless to encoders), atlas collections mislabeled as fMRI, and WM atlas labels like `acoustic tstatA` (statistical jargon).

**Implemented fixes** (`improve_neurovault_labels.py`, `build_neurovault_cache.py`):

1. **Exclude atlas collections** (262 Harvard-Oxford, 264 JHU DTI) — 23 terms removed. These are probabilistic anatomical atlases, not task contrasts.

2. **Strip `[colN]` prefix** — collection_id kept in `term_collection_ids.pkl` for provenance; terms no longer include it. Critical for generalizer: encoder sees cognitive content, not collection ID.

3. **WM atlas (7756–7761) cleanup** — `acoustic tstatA` → `acoustic`, `decision making tstatB` → `decision making_1`. Cognitive terms preserved; tstatA/tstatB removed.

4. **Future builds** — `build_neurovault_cache.py` applies these by default (`--exclude-atlas-collections`, `--strip-col-prefix`). Pipeline runs `improve_neurovault_labels.py` after NeuroVault build.

**For existing caches:** Run `python neurolab/scripts/improve_neurovault_labels.py` to apply improvements in place (or `--output-dir` for a new cache). Then rebuild merged_sources.

---

## 1. JUNK (discard) — 2 terms

| Source | Term | Issue |
|--------|------|-------|
| direct | `fMRI: test` | Placeholder (no semantic content) |
| neurovault_pharma | `fMRI: test2` | Placeholder (no semantic content) |

**Action:** Filter `test\d*` (exact match after stripping prefix).

### 1b. trm_ terms (3 terms) — RECOVERABLE, not junk

| Source | Term |
|--------|------|
| neurovault_pharma | `fMRI: NeuroVault fMRI task trm_4f2453ce33f16` |
| neurovault_pharma | `fMRI: NeuroVault fMRI task trm_5667441c338a7` |
| neurovault_pharma | `fMRI: NeuroVault fMRI task trm_4f241b50caaf7` |

**Context:** `trm_` = Cognitive Atlas task ID. NeuroVault integrates with Cognitive Atlas; when `contrast_definition` is missing, the fallback uses `cognitive_paradigm_cogatlas_id` (task ID). The map itself has meaning (a task contrast); the label is just opaque.

**Action:** Enrich at build time: fetch task name from Cognitive Atlas API (`https://www.cognitiveatlas.org/api/task/{id}`) and replace with human-readable label. If API fails, optionally keep with note or discard. Do **not** treat as junk — the maps are valid.

---

## 2. MOVIE WATCHING (20 terms) — KEEP, relabel

```
fMRI: [col20820] 0058_Movie_1
fMRI: [col20820] 0059_Movie_2
... (20 total)
```

**Context (from NeuroVault + HCP docs):** Collection 20820 = **Task-Evoked Network Atlas: HCP-YA**. Uses tensor ICA (tICA) on HCP fMRI tasks: Emotion, Gambling, Working Memory, Social, and **Movie-Watching**. The `0058_Movie_1`, `0059_Movie_2`, etc. are **tICA components** from the HCP 7T movie-watching task — naturalistic viewing of film excerpts. Each number is a component index, not a random frame; the underlying data are structured task-evoked networks from naturalistic stimulation.

**Action:** **KEEP.** Relabel at merge to: `fMRI: HCP movie-watching tICA component N (naturalistic viewing)` so the contrast context is clear. Movie watching is a valid, interesting cognitive condition.

### 2b. neurovault_pharma regparam (617 terms)

```
fMRI: reason
fMRI: reasonNeg
fMRI: abstract knowledge regparam
fMRI: acoustic processing regparam
fMRI: action perception regparam
fMRI: language processing regparam
...
```

**Context:** Parametric regressor betas from GLMs. "regparam" = regression parameter. We know the regressor name (e.g. "abstract knowledge") but **not**:
- What task (n-back? stroop? resting?)
- What drug/condition (placebo? ketamine? baseline?)
- What population

**Action:** Unclear. These are parametric maps; the regressor name gives semantic domain but not experimental design. Consider discarding unless we can enrich with task/drug metadata from the collection.

### 2c. pharma_neurosynth single-word (5–10 terms) — KEEP

```
fMRI: metabolism
fMRI: positron emission
fMRI: ampa
fMRI: challenge
fMRI: dependence
fMRI: dose
fMRI: dopaminergic
```

**Context (from pharmacological fMRI literature):** These come from NeuroSynth meta-analyses of **pharmacological fMRI** studies. In this domain:
- **metabolism** — brain metabolism in drug/neurochemical studies (glucose, oxygen, regional metabolism)
- **dose** — dose-dependent drug effects
- **challenge** — drug challenge paradigms (e.g. pharmacological provocation)
- **dopaminergic** — dopamine system engagement
- **ampa** — AMPA receptor (glutamate); relevant to drug mechanisms

**Action:** **KEEP.** They have clear domain meaning in pharmacological neuroimaging. Consider prefixing with "Pharmacological fMRI:" if we want to disambiguate from general cognitive meta-analyses.

### 2d. direct (NeuroQuery/NeuroSynth) — short terms (471 terms) — MOSTLY KEEP

```
fMRI: acc
fMRI: acg
fMRI: add
fMRI: adhd
fMRI: age
fMRI: act
fMRI: acth
fMRI: acid
...
```

**Context (from NeuroSynth + neuroscience literature):**
- **acc** — NeuroSynth: 558 studies. In neuroscience, "ACC" consistently = **anterior cingulate cortex** (not "accuracy"). Well-defined brain region.
- **acg** — **Anterior cingulate gyrus** = same as ACC. Standard anatomy term.
- **add** — NeuroSynth: 85 studies. In neuroimaging context often = **ADHD** (attention deficit disorder).
- **adhd** — ADHD meta-analysis. Clear.
- **age** — NeuroSynth Topic 24: age-related, development, aging (1,647+ studies). Meta-analysis of age as covariate/domain.
- **acth** — **Adrenocorticotropic hormone**; HPA axis, cortisol, stress. Neuroscience term.
- **acid** — Could be amino acid, GABA, neurotransmitter context. Slightly ambiguous but domain-specific.

**Action:** **Do NOT discard** all &lt;5 char terms. Many are standard neuroscience abbreviations. Only discard terms that are clearly non-semantic (e.g. pure numbers, "test"). Use a short blacklist (test, test2, etc.) rather than a length cutoff. If desired, expand abbreviations in labels (e.g. "acc" → "anterior cingulate cortex") at display/enrichment time.

---

## 3. BORDERLINE — single words, partial context

### 3a. direct single-word (3,654 terms)

```
fMRI: ability
fMRI: abstract
fMRI: abuse
fMRI: accuracy
fMRI: accumbens
...
```

**Context:** Meta-analysis of "activation associated with studies mentioning X". We know the semantic domain but **not** the specific task/contrast. "Ability" = mixed tasks (WM, reasoning, etc.) averaged.

**Verdict:** Informative at the meta level (domain → brain) but not at the experiment level (task/drug unknown). **Keep** if we accept that NQ/NS maps are inherently "term → brain association" without task detail. **Discard** if we require experiment-level context.

### 3b. direct multi-word (informative)

```
fMRI: working memory n-back
fMRI: stroop interference
fMRI: face recognition
```

**Context:** Same meta-analytic; multi-word narrows the domain. More informative.

**Action:** Keep.

---

## 4. INFORMATIVE — contrast/task/group/meta context clear

### 4a. neuromaps (31 terms)

```
PET: FEOBV binding to VAChT (vesicular acetylcholine transporter)
PET: Raclopride binding to D2 (dopamine receptor)
```

**Context:** PET receptor/transporter binding. Clear: tracer, target, system. No task (not applicable). **Action:** Keep.

### 4b. abagen, reference (712 terms)

```
Gene: ZSCAN22 gene expression from Allen Human Brain Atlas
Gene: 5-hydroxytryptamine receptor 2A gene expression across cortex Serotonin system
```

**Context:** Gene expression. Clear: gene, atlas, system. **Action:** Keep.

### 4c. enigma (52 terms)

```
Structural: ADHD cortical thickness adult
Structural: major depression subcortical volume
```

**Context:** Structural MRI, disorder, measure. Clear. **Action:** Keep.

### 4d. neurovault — with contrast (5,300+ terms)

```
fMRI: [col1541] Angry vs baseline
fMRI: [col833] Contrast analysis: implicit SRTT variants vs explicit SRTT variants
fMRI: [col63] incorrect response (task)
fMRI: [col6618] Read simple sentence vs consonant strings
```

**Context:** Task fMRI, explicit contrast (A vs B). We recover: task/condition, comparison. **Action:** Keep.

---

## Summary: discard vs keep (revised after research)

| Category | Count | Action |
|----------|-------|--------|
| Junk (test, test2 only) | 2 | **Discard** |
| trm_ terms | 3 | **Enrich** — fetch task name from Cognitive Atlas API |
| neurovault Movie IDs (col20820) | 20 | **KEEP** — relabel as "HCP movie-watching tICA component N (naturalistic viewing)" |
| neurovault_pharma regparam | 617 | **Evaluate** — enrich or discard |
| pharma_neurosynth single-word | ~10 | **KEEP** — have pharmacological domain meaning |
| direct short (acc, acg, age, …) | 471 | **KEEP** — standard neuroscience abbreviations |
| direct single-word | 3,654 | **KEEP** — meta-analytic term → brain association |
| All others | ~10,840 | **Keep** |

**Recommended filters to add:**
1. `test` and `test2` only (exact match) — discard placeholders

**Enrichment (not discard):**
- **trm_** terms: fetch task name from Cognitive Atlas API, replace opaque ID with human-readable label
- **Movie_N** (col20820): relabel to "fMRI: HCP movie-watching tICA component N (naturalistic viewing)"

**Do NOT add:** length cutoff for short terms; many (acc, acg, add, adhd, age, acth) are meaningful neuroscience abbreviations.

**To evaluate:** neurovault_pharma regparam — can we recover task/drug from collection metadata? If not, discard.

---

## References (web research, Feb 2025)

- **NeuroSynth:** Yarkoni et al., automated meta-analysis of fMRI studies by term occurrence in abstracts; 558 studies for "acc", 85 for "add"; Topic 24 = age/development/aging
- **ACC/ACG:** In neuroscience literature, ACC = anterior cingulate cortex, ACG = anterior cingulate gyrus (same region)
- **ACTH:** Adrenocorticotropic hormone; HPA axis, cortisol; relevant to stress, Cushing's, fMRI of hormonal effects
- **NeuroVault col 20820:** Task-Evoked Network Atlas HCP-YA; tICA on Emotion, Gambling, WM, Social, Movie-Watching tasks
- **HCP 7T movie-watching:** Naturalistic viewing of film excerpts; Movie_N = tICA component indices
- **trm_:** Cognitive Atlas task ID; NeuroVault fallback when contrast_definition missing; resolvable via Cognitive Atlas API
- **Pharmacological fMRI:** metabolism, dose, challenge = drug/phMRI meta-analysis domain terms
