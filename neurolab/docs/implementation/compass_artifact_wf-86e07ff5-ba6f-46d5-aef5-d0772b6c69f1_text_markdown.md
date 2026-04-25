# Predicting brain map similarity from ontological distance: a methods guide

**Lin similarity with intrinsic information content, computed via nxontology, paired with Pearson correlation under spatial null models from neuromaps/BrainSMASH, is the strongest practical combination for this pipeline.** This replaces raw hop count — which collapses to noise at distance ≥2 because edges near the ontology root span enormous semantic gulfs — with a measure that weights each edge by the information it actually carries. For the cross-ontology case (Cognitive Atlas ↔ UBERON ↔ GO), graph embeddings via OWL2Vec* offer a decisive advantage because IC-based measures fundamentally require a shared hierarchy. No published paper has explicitly correlated standard semantic similarity measures (Resnik, Lin, Jiang-Conrath) against brain activation map similarity — this pipeline would be genuinely novel, positioned between Poldrack et al.'s cognitive encoding models and the mature GO semantic similarity benchmarking literature.

---

## Why hop count dies at distance 2 and information content fixes it

The core problem with raw hop count is that **a single is_a edge near the root of an ontology spans far more semantic territory than one near the leaves**. Moving from "working memory" to "memory" (one hop) crosses a modest conceptual gap; moving from "memory" to "cognitive process" (one hop) crosses an enormous one. Hop count treats these identically. By distance 2, most term pairs have already traversed into such general territory that their brain maps are no more similar than random pairs — explaining the observed collapse from r≈0.39 at d=1 to r≈0.12 at d≥2.

Information Content (IC) solves this by quantifying how specific a concept is. IC is defined as **IC(c) = −log(p(c))**, where p(c) reflects how "general" concept c is. A root concept has IC≈0; a leaf has maximum IC. The three classical IC-based similarity measures then use IC to weight the shared ancestry between two terms:

- **Resnik similarity** = IC(MICA), where MICA is the Most Informative Common Ancestor. Simple, unbounded, strong baseline. Returns the same score for all pairs sharing the same MICA.
- **Lin similarity** = 2 × IC(MICA) / (IC(a) + IC(b)). Normalized to [0,1], penalizes pairs where one term is very specific and the other very general. Analogous to Wu-Palmer but using IC instead of depth.
- **Jiang-Conrath distance** = IC(a) + IC(b) − 2 × IC(MICA). A true metric (distance, not similarity), interpretable as the information not shared between two concepts.

The critical design choice is **how to compute IC**. Two approaches exist:

**Corpus-based IC** estimates p(c) from term frequency in a literature corpus (e.g., PubMed article counts per MeSH term, or NeuroQuery's 13,459 neuroimaging articles). This captures real-world usage patterns but suffers from data sparseness — terms absent from the corpus get undefined IC — and is biased by research trends.

**Intrinsic IC** estimates p(c) from the ontology's own DAG structure. The Sánchez et al. (2011) formula is preferred: IC(c) = −log((leaves(c)/subsumers(c) + 1) / (max_leaves + 1)), where leaves(c) counts leaf descendants and subsumers(c) counts ancestors. This improves on the simpler Seco et al. (2004) formula — IC(c) = 1 − log(hypo(c)+1)/log(max_nodes) — by differentiating concepts with the same descendant count but at different depths, and by handling multiple inheritance natively. **Intrinsic IC consistently outperforms corpus-based IC**: Sánchez et al. reported correlations of **0.77–0.87 with human similarity judgments** versus 0.70–0.73 for corpus-based approaches. For neuroscience ontologies like the Cognitive Atlas, which lack a large annotation corpus comparable to GO's GOA database, intrinsic IC is both more practical and empirically superior.

Path-based measures like **Wu-Palmer** (2×depth(LCA)/(depth(a)+depth(b))) and **Leacock-Chodorow** (−log(shortest_path/(2×max_depth))) improve on raw hop count by normalizing for depth, but they still assume uniform semantic distance per edge at a given depth. On standard benchmarks, they consistently underperform IC-based measures: Wu-Palmer achieves r≈0.74 on the Miller-Charles human judgment dataset versus r≈0.85–0.87 for Lin/JC with Sánchez IC.

**Edge-weighted path length** — assigning different weights to is_a (1.0), part_of (1.5), and regulates (2.0) edges — is a middle ground. Wang et al. (2007) formalized this as multiplicative decay along paths (is_a weight 0.8, part_of weight 0.6), producing competitive results without requiring IC computation. This is worth testing as a corpus-free, IC-free baseline.

---

## Graph embeddings outperform IC measures but shine brightest for cross-ontology terms

For a knowledge graph spanning Cognitive Atlas, UBERON, GO, and the Mental Functioning Ontology, IC-based measures face a fundamental limitation: **they require a shared hierarchical ancestor to compute similarity**. Two terms from different ontologies may have no common ancestor unless the ontologies are formally merged with bridge axioms. Graph embeddings bypass this entirely.

The benchmark evidence is substantial. In Gene Ontology applications, **graph embedding methods outperform IC-based measures by 5–15% AUC** for protein-protein interaction prediction. OPA2Vec significantly outperformed Resnik similarity for both human and yeast PPI prediction. The deepSimDEF system exceeded all IC-based functional similarity measures by over 5–10% F1 on human PPI. These are not marginal improvements.

Three ontology-specific embedding methods are most relevant:

**OWL2Vec*** transforms an OWL ontology into an RDF graph, performs random walks capturing graph structure and lexical information (entity labels, definitions), then trains Word2Vec embeddings. It uniquely captures logical constructors (existential restrictions, class intersections) that generic methods like TransE miss entirely. It significantly outperformed RDF2Vec, TransE, and Onto2Vec on GO subsumption prediction.

**OPA2Vec** extends this by pre-training on PubMed/PMC abstracts for transfer learning, then fine-tuning on combined formal axiom and annotation corpora. The pre-training step assigns biomedical semantics to natural language words in annotation properties, bridging formal and informal ontology content.

**anc2vec** was specifically designed to preserve ancestor relationships in GO, outperforming Onto2Vec and GO2Vec for both PPI prediction and hierarchical structure discrimination. This ancestor-awareness is particularly relevant for ontological distance computation.

All three are implemented in the **mOWL library** (`pip install mowl-borg`), a Python package from KAUST's Bio-Ontology Research Group that also includes description logic embedding methods (ELEmbeddings, Box2EL) and evaluation modules. For ~7,500 terms, OWL2Vec* training completes in minutes, and the resulting 100–200-dimensional embeddings yield a pairwise cosine similarity matrix efficiently.

An important caveat: **embedding methods are typically semi-supervised** (trained on the graph structure as a task), while IC-based methods are fully unsupervised. This makes direct comparison somewhat asymmetric. IC-based measures remain strong for single-ontology, interpretable applications. The practical recommendation is to **test both**: Lin similarity with intrinsic IC within each ontology, and OWL2Vec* cosine similarity for the full cross-ontology knowledge graph, then compare which better predicts brain map similarity.

A hybrid approach — using IC as edge weights for OWL2Vec*'s random walks, or concatenating IC-based features with embedding vectors — has theoretical support (Lastra-Díaz et al. 2019 showed linear combinations of embeddings with ontology-based measures improve state of the art) but no widely-adopted implementation yet.

---

## Pearson correlation is correct for meta-analytic maps but needs spatial null models

For 400-dimensional Schaefer-parcellated vectors from NeuroQuery/NeuroSynth, **Pearson correlation is the appropriate primary metric**, and mutual information is unlikely to reveal substantial additional structure. The reasoning is specific to this data type:

NeuroQuery maps are generated via linear regression, and NeuroSynth maps are z-scores from mass univariate tests. The relationships between two such maps are **predominantly linear** by construction. Song et al. (2012) demonstrated comprehensively that when underlying relationships are approximately linear, MI and Pearson r are approximately monotonically related — MI provides minimal additional information but requires more parameter choices and loses directionality (MI has no sign). With only 400 dimensions, MI estimation via the Kraskov-Stögbauer-Grassberger (KSG) estimator (available as `sklearn.feature_selection.mutual_info_regression`) is feasible but noisy. The scenario where MI would add value — nonlinear relationships where two terms activate the same regions but at different magnitudes — is already captured by Pearson r, which is scale-invariant.

The real methodological concern is **spatial autocorrelation**, not metric choice. Adjacent brain parcels are correlated by construction (shared vasculature, smoothing, genuine biological similarity), inflating naive Pearson r values and producing grossly liberal p-values. Markello and Misic (2021) showed that **naive null models consistently yield elevated false positive rates** even for parcellated data, and found minimal impact of parcellation resolution on null model performance — meaning **Schaefer 400 parcellation does not eliminate the need for spatial null models**.

Three spatial null frameworks are well-validated for parcellated data:

- **Spin permutation tests** (Alexander-Bloch et al., 2018): Rotate brain maps on the cortical sphere, preserving spatial autocorrelation. Fast, simple, the most widely used. Limitation: works only for cortical surface data, and spherical projection introduces geometric distortions.
- **BrainSMASH** (Burt et al., 2020): Generates surrogate maps by variogram matching — computing the variance-as-function-of-distance profile and generating random maps that match it. Works with volumetric, surface, and parcellated data. Requires a **400×400 geodesic distance matrix** for Schaefer parcels.
- **Moran spectral randomization**: Decomposes spatial structure via eigenvectors of a distance-based weight matrix, then randomizes coefficients while preserving the power spectrum. Available in neuromaps.

The **neuromaps** package (Markello et al., 2022, Nature Methods) provides a unified interface: `neuromaps.stats.compare_images(map_a, map_b, nulls=rotated)` accepts any similarity metric and any of 8 null model implementations for parcellated data. For the full 7,500×7,500 pairwise comparison, running individual permutation tests is computationally prohibitive (~28 million pairs). Instead, compute the full Pearson correlation matrix directly, then use a **Mantel test** — correlating the ontological distance matrix against the brain similarity matrix with spatially-aware permutations — to assess the overall relationship.

**Spearman rank correlation** is worth computing as a robustness check, especially given that meta-analytic maps often have many near-zero values with sparse peaks, creating skewed distributions. Dice/Jaccard on thresholded maps and Earth Mover's Distance are less appropriate: the former discards graded information, and the latter addresses spatial displacement that Schaefer parcellation already handles.

---

## No published precedent exists for this exact analysis

Extensive searching confirmed that **no paper has explicitly correlated standard semantic similarity measures (Resnik, Lin, Jiang-Conrath) against brain activation map similarity**. This makes the proposed pipeline genuinely novel. The closest published work falls into four categories:

**Poldrack et al. (2023)** built Cognitive Encoding Models mapping from Cognitive Atlas annotations to brain activation, showing that shared ontological annotations predict brain map similarity — but they did not compute pairwise ontological distance using IC-based measures. **Beam et al. (2021)** used multivariate distance matrix regression to quantify variance among task-fMRI maps explained by ontological dimensions, finding that "task paradigm" categories explain more variance than latent cognitive categories. **Paquola et al. (2024/2025)** used IC-based semantic similarity metrics to evaluate meta-analytic decoding of cortical gradients — the most direct precedent, as they computed IC from NeuroSynth/NeuroQuery corpora and used it as a performance metric. **Dockès et al. (2020)** built NeuroQuery using distributional semantics (TFIDF co-occurrence) rather than ontological structure, explicitly stating their approach relies on corpus statistics rather than ontologies.

The Gene Ontology semantic similarity literature provides mature methodological guidance. Across multiple benchmarks, **IC-based measures consistently outperform pure path-based measures** for predicting functional similarity. Resnik shows the highest raw correlation with sequence similarity; Jiang-Conrath performs best with Best-Match Average (BMA) strategy; no single measure dominates all tasks. The GOSemSim R package (Yu et al., 2010) and the broader Pesquita et al. (2009) review establish the methodological framework that transfers directly to neuroscience ontologies.

---

## Practical implementation blueprint

The recommended implementation uses a three-tier strategy, starting with the simplest validated approach and adding complexity only where it demonstrably improves prediction:

**Tier 1 — IC-based similarity (start here):**
Use **nxontology** (`pip install nxontology`) with **Lin similarity and Sánchez intrinsic IC**. Load each ontology from OBO format, merge into a single DAG with a virtual root node connecting ontology roots, freeze the graph, and compute pairwise similarity. For ~7,500 terms, the full 7,500×7,500 matrix is tractable. Compare against the Pearson correlation matrix of brain maps using a Mantel test with BrainSMASH-generated spatial surrogates.

```python
from nxontology import NXOntology
nxo = NXOntology()
# Add all edges from merged KG
for parent, child in kg_edges:
    nxo.graph.add_edge(parent, child)
nxo.freeze()
sim = nxo.similarity("term_a", "term_b", ic_metric="intrinsic_ic_sanchez")
lin_score = sim.lin  # normalized [0,1]
```

**Tier 2 — Graph embeddings (for cross-ontology terms):**
Use **OWL2Vec*** via the mOWL library (`pip install mowl-borg`) to generate 200-dimensional embeddings of all terms in the merged knowledge graph. Compute pairwise cosine similarity. This captures cross-ontology relationships that IC-based measures cannot handle (terms from Cognitive Atlas versus UBERON have no shared ancestor in their respective hierarchies). Test whether embedding-based distance predicts brain similarity better than, or complementary to, Lin similarity.

**Tier 3 — Hybrid (if warranted):**
If Tier 1 and Tier 2 capture partially independent variance in brain map similarity, combine them. Options include: linear combination of Lin similarity and cosine embedding similarity (with weights optimized via cross-validation), or using IC values as edge weights in OWL2Vec*'s random walks.

For brain map comparison: compute the **full Pearson correlation matrix** across all 7,500 term pairs. For statistical testing of the ontological-distance-to-brain-similarity relationship, use a **Mantel test** with **BrainSMASH** surrogates preserving spatial autocorrelation. Run **Spearman correlation** as a robustness check. Report both the raw Mantel r and the spatial-null-corrected p-value.

Key Python tools to install: **nxontology** (IC-based similarity), **mowl-borg** (graph embeddings), **neuromaps** (brain map comparison with spatial nulls), **brainsmash** (surrogate map generation), and **ssmpy/DiShIn** (alternative IC implementation handling disjunctive common ancestors).

---

## Conclusion

The drop from r≈0.39 at hop distance 1 to r≈0.12 at distance 2+ is not a fundamental ceiling — it is an artifact of using a metric that treats all edges as equal. **Switching to Lin similarity with Sánchez intrinsic IC should substantially extend the range over which ontological distance predicts brain map similarity**, because it compresses the uninformative hops near the root while preserving the discriminative hops near the leaves. The GO literature establishes this empirically: IC-based measures consistently outperform path-based measures for predicting functional similarity.

For the cross-ontology case, graph embeddings via OWL2Vec* are not optional — they are the only principled way to compute similarity between terms that lack a shared hierarchical ancestor. The 5–15% AUC improvement over IC-based measures in GO benchmarks suggests embeddings may further improve prediction.

On the brain map side, Pearson correlation is appropriate for linearly-generated meta-analytic maps, but **spatial null models are essential for valid inference** — even with Schaefer 400 parcellation. The combination of neuromaps and BrainSMASH provides a validated, parcellation-ready framework. MI is a theoretically motivated but practically low-yield addition for this specific data type.

The most impactful next step is not choosing between these measures — it is **testing all three tiers** (Lin IC, OWL2Vec* embeddings, hybrid) against the brain similarity matrix and comparing their predictive power. This comparison would itself be a publishable contribution, bridging the mature GO semantic similarity literature with computational cognitive neuroscience.