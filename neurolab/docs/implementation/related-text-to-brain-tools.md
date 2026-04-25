# Related Text-to-Brain Tools: Text2Brain, Chat2Brain, and NeuroLab

This document summarizes **purpose-built “text → brain activation map”** tools, their open-source status, and how they can **complement** the NeuroLab pipeline (NeuroQuery + expandable term space + enrichment).

---

## Same idea: expand beyond the NeuroQuery vocabulary

**In short:** You, Text2Brain, and Chat2Brain are all doing the same thing conceptually: **use NeuroQuery (its data or its term maps) to go beyond fixed-term lookup** so that **arbitrary text** — not just the ~7.5k NeuroQuery terms — can produce a brain map.

| Who | How they expand |
|-----|------------------|
| **NeuroLab (you)** | NeuroQuery term maps → build decoder cache → train **text→400-D** (TF-IDF or sentence-transformers + MLP). Any phrase → predicted parcellated map; then decode + biological enrichment. |
| **Text2Brain** | NeuroQuery **dataset** (~13k studies: text + activation maps) → train **SciBERT + 3D CNN**. Any free-form text → predicted 3D voxel map. |
| **Chat2Brain** | Same as Text2Brain + ChatGPT to refine messy queries (Text2Semantics) and augment training text (ChatAUG). |

So: same goal (expand vocabulary → more maps from open-ended text), different **output space** (you: parcellated 400-D for direct decode/enrich; them: full 3D) and **encoders** (you: lightweight TF-IDF/sentence-transformers; them: SciBERT + 3D generator). Our parcellated design keeps decode and biological enrichment in one space; theirs is geared to standalone 3D visualization.

---

## Text encoding vs map generation: no CNN for text

They do **not** use a CNN to encode text. The pipeline is:

1. **Text → vector:** **SciBERT** (transformer, pretrained on scientific text) encodes the query → 768‑d vector. So **text encoding = transformer**, same family as our sentence-transformers option.
2. **Vector → 3D map:** A **3D CNN** (transposed convolutions) takes that 768‑d vector and **generates** the voxel grid (40×48×40). So the CNN is the **decoder/generator** (embedding → 3D image), not the text encoder.

So: **transformer for text, CNN for producing the 3D volume.** “Better” depends on the goal:

- **If you want full 3D voxel output** (e.g. for visualization at native resolution), their design is appropriate: you need something that upsamples a vector into a 3D image, and a 3D CNN generator is a standard choice.
- **If you want parcellated 400‑D** (for decode + biological enrichment in one space), you don’t need a 3D generator. A simple **MLP head** (text embedding → 400‑D) is enough and stays aligned with our decoder cache. We can still use a **strong text encoder** (sentence-transformers, or even SciBERT if we wanted) and only change the head to 400‑D instead of 3D.

So the CNN in Text2Brain/Chat2Brain is for **map generation**, not for encoding. For text, they use a transformer (SciBERT); we use TF‑IDF or sentence-transformers. We could swap in SciBERT (or another science-oriented encoder) on our side and keep the 400‑D head—that would be the fair comparison for “better text encoding,” not adding a CNN.

---

## 1. Tools overview

| Tool | Paper / source | Open source | Output | Text handling |
|------|----------------|-------------|--------|----------------|
| **NeuroQuery** | Dockès et al. eLife 2020; [neuroquery](https://github.com/neuroquery/neuroquery) | Yes (BSD) | Predicted map (3D or parcellated); vocab ~7,547 terms | TF-IDF + semantic smoothing; supervised on full-text literature |
| **Text2Brain** | Ngo et al. MICCAI 2021; [arxiv](https://arxiv.org/abs/2109.13814) | **Yes (MIT)** [ngohgia/text2brain](https://github.com/ngohgia/text2brain) | Whole-brain 3D NIfTI (40×48×40, MNI) | SciBERT (finetuned) + 3D CNN; free-form text |
| **Chat2Brain** | Wei et al. arXiv:2309.05021; [paper](https://arxiv.org/pdf/2309.05021) | **Yes (Apache-2.0)** [Weiyaonai/Chat2Brain](https://github.com/Weiyaonai/Chat2Brain) | Same as Text2Brain (3D voxel map) | **Text2Brain model + ChatGPT**: (1) text→semantic query (Text2Semantics), (2) ChatAUG data augmentation |
| **NeuroLab (ours)** | — | Yes (this repo) | **Parcellated (400-D Schaefer)** or NIfTI via inverse transform | NeuroQuery and/or **expandable term space** (TF-IDF or sentence-transformers + MLP → 400-D); open vocabulary |

---

## 2. Text2Brain (we don’t have it yet)

- **What it does:** Free-form text → whole-brain activation map in MNI space. SciBERT encodes the query; a 3D CNN generates a voxel map (40×48×40). Trained on ~13k studies from the NeuroQuery dataset.
- **Open source:** Yes. [GitHub: ngohgia/text2brain](https://github.com/ngohgia/text2brain) (MIT). Also [sabunculab/text2brain](https://github.com/sabunculab/text2brain).
- **Usage:** Requires Conda env, checkpoint (`best_loss.pth` from Google Drive), and SciBERT weights. CLI: `python predict.py "self-generated thought" prediction.nii.gz` → NIfTI.
- **Strengths:** Free-form queries (“default network”, “task-unrelated thought”); paper shows good alignment with domain meta-analyses; handles synonymous phrasing better than keyword-only systems.
- **Reference:** Ngo et al., “Text2Brain: Synthesis of Brain Activation Maps from Free-form Text Query”, MICCAI 2021; [arxiv](https://arxiv.org/abs/2109.13814). Demo: [braininterpreter.com](https://braininterpreter.com).

---

## 3. Chat2Brain (LLM + Text2Brain)

- **What it does:** Two-stage pipeline: (1) **Text2Semantics** — user text → ChatGPT → refined “semantic query”; (2) **Semantics → brain map** using the same Text2Brain-style model (SciBERT + 3D CNN). Also uses **ChatAUG** at training: ChatGPT augments titles into synonyms, abstract-like text, keywords, etc.
- **Open source:** Yes. [GitHub: Weiyaonai/Chat2Brain](https://github.com/Weiyaonai/Chat2Brain) (Apache-2.0). Implementation in a Jupyter notebook; builds on Text2Brain.
- **Strengths:** Better on **complex / non-standard** queries (redundant, ambiguous, or incomplete text); paper reports gains in Dice/mIoU in non-standard settings and vs NeuroQuery/Text2Brain when retaining top 10–20% voxels.
- **Reference:** Wei et al., “Chat2Brain: A Method for Mapping Open-Ended Semantic Queries to Brain Activation Maps”, [arXiv:2309.05021](https://arxiv.org/pdf/2309.05021).

---

## 4. How NeuroLab compares

| Aspect | NeuroLab (current) | Text2Brain | Chat2Brain |
|--------|--------------------|------------|------------|
| **Output space** | Parcellated 400-D (Schaefer); NIfTI via masker inverse | Full 3D voxel (40×48×40) | Full 3D voxel |
| **Primary use** | Decode + biological enrichment (receptor, neuromaps) on same 400-D space | Standalone “query → map” | Same + LLM query refinement |
| **Open vocabulary** | Yes (embedding model: any text → 400-D) | Free-form text | Free-form + LLM refinement |
| **Training data** | NeuroQuery term maps (our cache) and/or NeuroSynth | NeuroQuery dataset (~13k) | Same + ChatAUG |
| **Integration** | Native: `query.py`, `predict_map.py`, `UnifiedEnrichment` | External: run script → NIfTI | External: notebook / their pipeline |

We **have NeuroQuery**; we **do not** currently ship or run Text2Brain or Chat2Brain. Our “expandable term space” is a **parcellated** text→brain model (text → 400-D) so everything stays in one space for decoding and biological enrichment.

---

## 5. How they can complement NeuroLab

### Option A: Use Text2Brain or Chat2Brain as an alternative “text → map” backend

- Run their pipeline (e.g. `python predict.py "query" out.nii.gz` or Chat2Brain notebook) to get a **3D NIfTI**.
- **Parcellate** that NIfTI to Schaefer 400 (same masker we use elsewhere).
- Feed the resulting **(400,) vector** into `CognitiveDecoder.decode()` and `UnifiedEnrichment.enrich()`.

**Pros:** Reuse their models for full 3D maps; our enrichment (cognitive + biological) stays unchanged.  
**Cons:** Extra dependency and setup (Conda, checkpoints, SciBERT/ChatGPT); two different “map” representations (theirs 3D, ours 400-D) to maintain.  
**Practical step:** Optional script or adapter: `text2brain_to_enrichment.py "query"` that (1) calls Text2Brain/Chat2Brain if available, (2) parcellates, (3) runs enrichment and prints summary.

### Option B: Adopt only the “LLM query refinement” idea (Chat2Brain-style)

- Keep our current text→map (NeuroQuery or expandable embedding).
- Add an **optional** step: user text → LLM (e.g. ChatGPT or local) → refined short “semantic query” → existing NeuroQuery or `TextToBrainEmbedding.predict_map()`.

**Pros:** No need to run Text2Brain/Chat2Brain; improves robustness for messy or long queries; fits our parcellated pipeline.  
**Cons:** Requires LLM API or local model; need prompts and validation.  
**Practical step:** Optional module or script `refine_query_with_llm(text) -> str` used before `query.py` or `predict_map.py` when enabled.

### Option C: Document as related tools (no code integration)

- In implementation guide and README: list Text2Brain, Chat2Brain, NeuroQuery, Neurosynth; when to use which; that our enrichment can consume **any** map that is parcellated to Schaefer 400 (including from external tools).

Already done in this doc; can add a short “Related tools” subsection to the main README or implementation guide.

---

## 6. Recommendation

1. **Short term:** Treat Text2Brain and Chat2Brain as **documented alternatives** (this doc + link from [implementation/README.md](README.md)). Use **Option C** so users know they can produce maps elsewhere and parcellate → enrich in NeuroLab.
2. **Optional integration:** If we want a second “text → map” backend:
   - **Option A:** Add an optional adapter that runs Text2Brain (or Chat2Brain) → NIfTI → parcellate → `UnifiedEnrichment.enrich()`, with clear docs and Evidence Tier (e.g. Tier C for model-derived maps).
   - **Option B:** Add optional LLM query refinement before our existing NeuroQuery/embedding path; no new map model.
3. **Evidence Tier:** Any map produced by Text2Brain, Chat2Brain, or our embedding model is **predictive/model-based** → label **Evidence Tier C** and point to “how to validate” (e.g. compare to meta-analyses or hold-out studies).

---

## 7. References

- **Text2Brain:** Ngo et al., MICCAI 2021, [arxiv/2109.13814](https://arxiv.org/abs/2109.13814); code [ngohgia/text2brain](https://github.com/ngohgia/text2brain).
- **Chat2Brain:** Wei et al., [arXiv:2309.05021](https://arxiv.org/pdf/2309.05021); code [Weiyaonai/Chat2Brain](https://github.com/Weiyaonai/Chat2Brain).
- **NeuroQuery:** Dockès et al., eLife 2020; [neuroquery](https://github.com/neuroquery/neuroquery).
- **NeuroLab expandable term space:** [expandable-term-space-embeddings.md](expandable-term-space-embeddings.md); [custom-prompt-to-map.md](custom-prompt-to-map.md).
