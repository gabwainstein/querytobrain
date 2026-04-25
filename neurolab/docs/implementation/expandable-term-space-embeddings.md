# Expandable Term Space: Train Text-to-Brain Embeddings

## Goal

Today the decoder can only return terms from a **fixed vocabulary** (NeuroQuery ~6.3k, NeuroSynth ~1k). To **expand the term space** and **predict maps for text not in any database** we need a way to map **arbitrary text** (e.g. "noradrenergic modulation of attention", "alpha-GPC", novel compounds) into the same representation we use for decoding—the 400-D parcellated brain space. Then we can:

1. **Predict maps that are not present:** For any phrase (including novel compounds or descriptions never in NeuroQuery/NeuroSynth), the model **predicts** the parcellated brain map. That predicted (400,) is the map; it can be saved as 3D NIfTI for visualization or fed into the decoder.
2. **Decode with an open vocabulary:** Embed text → 400-D, then correlate with all cached term maps and return nearest-neighbor terms *or* treat the phrase as a synthetic term.
3. **Add synthetic terms:** Embed new phrases and append them to the decoder cache (term_maps + term_vocab) so they appear in future decodes.

So we **train embeddings**: a model that maps **text → 400-D parcellated vector** (aligned with our existing term maps). The output **is** the predicted map for unseen text.

---

## How training works

1. **Data:** Load (term, parcellated_map) from the decoder cache: `term_vocab.pkl` (list of strings) + `term_maps.npz` (n_terms × 400). Each row is the ground-truth brain map for that term (from NeuroQuery or NeuroSynth).
2. **Encode text:** Turn each term into a fixed-size vector:
   - **TF-IDF:** Bag-of-words with ngrams (e.g. max 4096 features). No extra deps; weaker semantics.
   - **Sentence-transformers:** Pretrained encoder (e.g. `all-MiniLM-L6-v2` 384-d, or `all-mpnet-base-v2` 768-d). Better semantics; needs `pip install sentence-transformers`.
3. **Regression head:** A small MLP (e.g. linear → 512 → ReLU → 400) maps text embedding → 400-D. Trained with **MSE loss** between predicted 400-D and the ground-truth map from the cache.
4. **Save:** Encoder (or TF-IDF vectorizer) + head weights + config. At inference, any text → encode → head → predicted 400-D map.

So we do **not** train the text encoder from scratch; we only train the **head** that maps encoder output → brain space. More (term, map) pairs and a better encoder = better predictions for unseen text.

---

## Approach

**Supervision:** We already have (term, parcellated_map) pairs from our caches:

- NeuroQuery cache: `term_vocab.pkl` (list of strings) + `term_maps.npz` (n_terms × 400).
- Optionally NeuroSynth cache: same format.

**Model:** Two parts:

1. **Text encoder:** Maps a string to a fixed-size vector (e.g. 384-D or 768-D). Options:
   - Lightweight: TF-IDF or a small sentence model (e.g. `all-MiniLM-L6-v2`).
   - Richer: a neuroscience-aware encoder if we fine-tune later.
2. **Regression head:** Maps text embedding → 400-D parcellated vector. Train by minimizing MSE (or cosine loss) between predicted 400-D and the **ground-truth** parcellated map for that term from our cache.

**Training loop:**

- Load `term_maps.npz` and `term_vocab.pkl`.
- For each term: encode(term) → text_vec; head(text_vec) → pred_400d; loss = MSE(pred_400d, term_maps[i]).
- Validate on a held-out set of terms (e.g. 10–20%).
- Save: text encoder (or its config) + regression head weights.

**Inference:**

- `embed(text)` / `predict_map(text)` → (400,) parcellated vector. This **is** the predicted map for that text (including text not in any database). Use it to:
  - **Predict maps not present:** The (400,) is the predicted map; optionally export to 3D NIfTI via `predict_map_to_nifti(text, output_path)`.
  - Correlate with all cached term maps (decode to nearest known terms), or
  - Append (embed(phrase), phrase) to the decoder as a synthetic term.

---

## Data

| Source       | Terms   | Use as supervision |
|-------------|---------|---------------------|
| NeuroQuery  | ~6,308  | Primary (term_maps + term_vocab from decoder_cache). |
| NeuroSynth  | ~1,000  | Optional merge for more (term, map) pairs. |

Same parcellation (Schaefer 400) everywhere so targets are comparable.

---

## Model Options (concise)

| Option              | Pros                         | Cons                    |
|---------------------|------------------------------|-------------------------|
| TF-IDF + MLP head   | No extra deps, fast           | Weak semantic similarity |
| Sentence-transformers + MLP | Good semantics, off-the-shelf | Extra dependency        |
| Fine-tune LM on (term, map) | Best alignment to brain   | More data/compute       |

**Why all-MiniLM-L6-v2 as default?** It was chosen as a **practical default** (small, fast, decent general semantics), **not** because it’s proven best for text→brain. There is no published benchmark that compares encoders on “predict parcellated brain map from text.” For this specific goal, the “best” encoder would be decided by validation (e.g. held-out term–map correlation or decode quality). Options to try:

- **General:** `all-MiniLM-L6-v2` (384-d, fast), `all-mpnet-base-v2` (768-d, often better on semantic tasks).
- **Biomedical / scientific:** Models trained on PubMed or scientific text may better match neuroscience and compound vocabulary, e.g. `NeuML/pubmedbert-base-embeddings` or other [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) / Hugging Face models. Use `--encoder-model <name>` and compare val correlation.

**Recommendation:** Start with sentence-transformers + MLP head; try `all-mpnet-base-v2` or a biomedical model and compare validation correlation. The encoder is configurable via `--encoder-model`.

### PCA on the map space (vector → PCs → reconstruct 400-D)

You can run **PCA on the brain maps** (our term maps are 400-D parcellated; the same idea applies to full 3D MNI): the first X components explain most of the variance and **encode smoothness**—neighboring parcels (or voxels) covary, so the top PCs are smooth spatial patterns. Then:

- **Train:** Head predicts the **X-D PC scores** (vector → vector in PC space), not 400-D directly.
- **Inference:** Predict X-D → `inverse_transform` → 400-D. The reconstruction is a linear combination of smooth basis maps, so you get smoothness without a CNN.

We **already support this**: use `--pca-components X` (e.g. 80). PCA is fit on the **train** term maps (n_terms × 400); the head outputs X-D; at eval we inverse_transform to 400-D. So it’s exactly “vector → vector (PC space) → 400-D.” Choosing **X by variance** (e.g. first X PCs that explain 95% of variance) is a good rule: fit PCA on your train maps, check `np.cumsum(pca.explained_variance_ratio_)`, pick X where it crosses 0.95, then use `--pca-components X`. The script currently uses a fixed X; you can run once with a large X (e.g. 200), inspect explained variance, then re-run with the desired X. See also `compare_pca_components.py` to compare train vs test correlation across different X (often reduces overfitting).

### Using a bigger model and more terms

- **More terms:** Build a full decoder cache (e.g. `build_term_maps_cache.py --max-terms 0` for all NeuroQuery terms, or merge NeuroSynth cache). Then train with `--max-terms 0` so the model sees all (term, map) pairs. More supervision = better generalization to unseen phrases.
- **Bigger encoder:** Use `--encoder sentence-transformers --encoder-model all-mpnet-base-v2` (768-d, better than MiniLM). Or `paraphrase-multilingual-mpnet-base-v2` for multilingual. First run downloads the model.
- **Bigger head / more epochs:** `--head-hidden 1024 --epochs 30` (or 50). More capacity and training time can improve val correlation.
- **Example (full cache + bigger model):**
  ```bash
  python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model --encoder sentence-transformers --encoder-model all-mpnet-base-v2 --max-terms 0 --head-hidden 1024 --epochs 30
  ```

---

## Integration

1. **Train:** `python neurolab/scripts/train_text_to_brain_embedding.py --cache-dir neurolab/data/decoder_cache --output-dir neurolab/data/embedding_model`
2. **Use in pipeline:**
   - **Expandable decoder:** Load embedding model; for a user phrase, call `embed_text(phrase)` and either (a) decode by correlation with existing term maps, or (b) add phrase as synthetic term and then decode (so the phrase itself can appear in top_terms).
   - **Query expansion:** Before calling NeuroQuery for a rare term, try `embed_text(term)` and decode to get related known terms.

---

## Exit Criteria

- [x] Training script runs and saves encoder + head.
- [x] `embed(text)` / `predict_map(text)` returns (400,) array—the **predicted map** for that text (including unseen text). Correlation with cached term maps gives nearest-neighbor terms.
- [x] Optional: `predict_map_to_nifti(text, path)` exports predicted map as 3D NIfTI for visualization.
- [x] Query path supports `--use-embedding-model` for open-vocabulary decoding (local only). Script `predict_map.py` for predict-only (and optional NIfTI save).

---

## See Also

- [enrichment-pipeline-build-plan.md](enrichment-pipeline-build-plan.md) (Phases 0–5)
- [cognitive_decoding_addendum](../../../docs/external-specs/cognitive_decoding_addendum.md)
