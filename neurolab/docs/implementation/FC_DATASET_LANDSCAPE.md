# Functional Connectivity Dataset Landscape

Open resting-state FC datasets for NeuroLab enrichment and pharmacological map training. The pipeline uses **parcel-level vectors (392-D)**, not full FC matrices. FC data enters as **FC degree** (row-sum per parcel), **FC gradient position**, or **network membership** features.

---

## Already parcellated & downloadable

### 1. ENIGMA Toolbox — HCP group-average FC (best fit)

**Source:** `enigmatoolbox.datasets.load_fc()`

Pre-computed resting-state FC matrices from HCP in multiple parcellations. Group-average from unrelated healthy HCP adults. **Glasser 360** maps directly onto pipeline (Glasser 360 + Tian S2 = 392).

```python
from enigmatoolbox.datasets import load_fc
funcMatrix_ctx, funcLabels_ctx, funcMatrix_sctx, funcLabels_sctx = load_fc(parcellation='glasser_360')
# funcMatrix_ctx: cortico-cortical (360×360)
# funcMatrix_sctx: subcortico-cortical (14×360)
```

**Parcellations:** `aparc`, `schaefer_100`, `schaefer_200`, `schaefer_300`, `schaefer_400`, `glasser_360`

**Use:** Normative FC reference for enrichment layer; FC degree = row-sum per parcel → 392-D vector.

**Subcortical:** ENIGMA subcortical is FreeSurfer aseg (14 regions). Pipeline uses Tian S2 (32). `build_fc_cache.py` reparcellates aseg→Tian via volumetric overlap (each Tian parcel gets the FC degree of its parent aseg region). Requires TemplateFlow aseg or precomputed `tian_to_aseg_mapping.npy`. Use `--no-aseg-mapping` to pad with zeros; `--save-mapping` to save the mapping when computed.

---

### 2. Luppi et al. 2023 — Pharmacological FC (Science Advances)

**Paper:** DOI 10.1126/sciadv.eadf8332 — "In vivo mapping of pharmacologically induced functional reorganization onto the human brain's neurotransmitter landscape"

**Content:** 10 drugs × 15 FC contrasts. Regional FC weighted degree change (ΔFC) maps in **Schaefer 100** parcellation. Each map = 100-region vector of baseline-minus-drug FC degree change.

**Drugs:** propofol (3 doses), sevoflurane (2 doses), ketamine, LSD, psilocybin, DMT, ayahuasca, MDMA, modafinil, methylphenidate.

**Data:** Supplementary materials; netneurolab GitHub repos (luppi-cognitive-matching, luppi-neurosynth-control). Luppi 2023 pharmacological FC data may be in Science Advances supplement or Cambridge repository.

**Reparcellation:** Schaefer 100 → Glasser+Tian 392 via surface overlap or nearest-parcel.

---

### 3. netneurolab repos — ready-to-use matrices

| Repo | Content | Parcellation |
|------|---------|--------------|
| [liu_fc-pyspi](https://github.com/netneurolab/liu_fc-pyspi) | Group-average rsfMRI FC; MEG band-resolved connectivity | Schaefer 400; data on OSF |
| [luppi-neurosynth-control](https://github.com/netneurolab/luppi-neurosynth-control) | Functional + structural connectomes | DK68 |
| [luppi-cognitive-matching](https://github.com/netneurolab/luppi-cognitive-matching) | NeuroSynth maps; FC from anesthesia studies | Schaefer 200, DK68 |

---

### 4. HCP 1200 — DataLad functional connectivity

**Source:** [datalad-datasets/hcp-functional-connectivity](https://github.com/datalad-datasets/hcp-functional-connectivity)

Preprocessed FC from WU-Minn HCP1200. Subject-level matrices. Requires HCP data use terms and AWS credentials.

```bash
datalad clone https://github.com/datalad-datasets/hcp-functional-connectivity.git
cd hcp-functional-connectivity && datalad get .  # or specific subjects
```

---

### 5. HCP PTN (parcellated timeseries and netmats)

**Source:** ConnectomeDB — requires HCP account (free)

Group ICA parcellation + timeseries + correlation matrices at 25, 50, 100, 200, 300 components. Task + rest FC.

---

## Requires processing (derivatives exist)

### 6. XCP-D processed datasets

**Source:** [PennLINC/xcp_d](https://github.com/PennLINC/xcp_d)

XCP-D outputs parcellated FC matrices from fMRIPrep outputs. Atlases: Schaefer, Gordon, Glasser. Check OpenNeuro for datasets with xcp_d derivatives (e.g. PsiConnect ds006110).

---

### 7. 1000 Functional Connectomes (FCP/INDI)

**Source:** [fcon_1000.projects.nitrc.org](https://www.nitrc.org/projects/fcon_1000/)

Over 1,200 resting-state subjects. CONN toolbox can import. No pre-computed group maps — run pipeline.

---

## Pipeline integration summary

| Source | Format | Parcellation | Effort | Value |
|-------|--------|--------------|--------|-------|
| **ENIGMA load_fc()** | Group matrix | Glasser 360 | Minutes | Normative FC reference |
| **Luppi 2023 drug FC** | 15 ΔFC vectors × 100 | Schaefer 100 → 392 | Hours | Pharmacological FC, 10 drugs |
| **HCP PTN netmats** | Subject + group | ICA components | Hours | Task + rest FC |
| **netneurolab .npy** | Group consensus | Schaefer 400 | Minutes | FC + SC |

**Key distinction:** Training uses parcel-level activation vectors (392-D). FC enters as FC degree (row-sum), gradient position, or network features — not full 392×392 matrices.

---

## Scripts

- `build_fc_cache.py` — Build FC cache from all sources. Use `--all-sources --all-enigma-parcellations` for full list.
- **See [FC_ACQUISITION_GUIDE.md](FC_ACQUISITION_GUIDE.md)** for healthy vs drug map acquisition steps.
- `download_luppi_fc_maps.py` — Acquisition guide for Luppi 2023 pharmacological FC data
- `download_netneurolab_fc.py` — Acquisition guide + optional OSF download for liu_fc-pyspi (Schaefer 400)

## FC cache full list (`--all-sources --all-enigma-parcellations`)

| Source | Maps | Parcellation | Notes |
|--------|------|--------------|-------|
| ENIGMA normative | 1 | Glasser 360 | Primary normative reference |
| ENIGMA aparc | 1 | DK68 → 392 | Cortical 68, subcortical 14 |
| ENIGMA schaefer_100/200/300/400 | 4 | Schaefer → 392 | Remapped via interpolation |
| Luppi 2023 | 0–15+ | Schaefer 100 → 392 | When `luppi_fc_maps/*.npy` exist |
| netneurolab | 0–N | Various | Any `netneurolab_fc/*.npy` (liu_fc-pyspi, luppi-cognitive-matching, etc.) |

**Total:** 6 ENIGMA maps minimum; +Luppi +netneurolab when data present.
