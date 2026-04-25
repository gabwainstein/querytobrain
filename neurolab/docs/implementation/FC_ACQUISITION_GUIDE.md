# FC Cache Acquisition: Healthy vs Drug Maps

The FC cache provides two map types for training and enrichment:

| Type | Description | Use |
|------|-------------|-----|
| **Healthy** | Normative resting-state FC from control populations | Baseline reference, enrichment layer |
| **Drug** | Pharmacologically induced FC changes (ΔFC) | Drug-effect prediction, nootropic mapping |

---

## Healthy maps (normative FC)

| Source | Maps | Parcellation | Download | Status |
|--------|------|--------------|----------|--------|
| **ENIGMA load_fc()** | 6 | Glasser 360, aparc, Schaefer 100/200/300/400 | Automatic (`pip install enigmatoolbox`) | ✅ Built-in |
| **netneurolab liu_fc-pyspi** | 1 | Schaefer 400 | OSF [osf.io/75je2](https://osf.io/75je2) → `fc_cons_400.npy` | Manual or `--use-osfclient` |
| **netneurolab luppi-neurosynth-control** | N | DK68 | [GitHub](https://github.com/netneurolab/luppi-neurosynth-control) | Check repo for FC matrices |

**Quick healthy setup:**
```powershell
# ENIGMA: already included when you run build_fc_cache
# netneurolab: download healthy group FC (automatic)
python neurolab/scripts/download_netneurolab_fc.py --use-osfclient
# Creates fc_cons_400.npy + pyspi term mean in neurolab/data/netneurolab_fc/
```

---

## Drug maps (pharmacological FC)

| Source | Maps | Drugs | Parcellation | Download | Status |
|--------|------|-------|--------------|----------|--------|
| **Luppi et al. 2023** | ~15–150 | propofol, sevoflurane, ketamine, LSD, psilocybin, DMT, ayahuasca, MDMA, modafinil, methylphenidate | Schaefer 100 | Cambridge / Science Advances supplement | Manual |
| **luppi-cognitive-matching** | Method only | propofol, sevoflurane, ketamine | Schaefer 200, DK68 | Repo has NeuroSynth maps, not pre-computed drug FC | Requires running analysis |

**Luppi 2023 drug FC acquisition:**
1. Run `python neurolab/scripts/download_luppi_fc_maps.py` — downloads PDF to `luppi_fc_maps/`
2. FC matrices may be in Science Advances supplement (separate download)
3. If you have ΔFC vectors (Schaefer 100): save as `*.npy` (shape 100,) in `neurolab/data/luppi_fc_maps/`
4. Run `build_fc_cache.py --all-sources --all-enigma-parcellations`

---

## Build and filter

After acquisition, run:
```powershell
python neurolab/scripts/build_fc_cache.py --output-dir neurolab/data/fc_cache --all-sources --all-enigma-parcellations
```

The cache saves `fc_map_types.pkl` so you can filter:
```python
import pickle
labels = pickle.load(open("neurolab/data/fc_cache/fc_labels.pkl", "rb"))
types = pickle.load(open("neurolab/data/fc_cache/fc_map_types.pkl", "rb"))
healthy_idx = [i for i, t in enumerate(types) if t == "healthy"]
drug_idx = [i for i, t in enumerate(types) if t == "drug"]
```

---

## Summary

| Map type | Built-in | With downloads |
|----------|----------|----------------|
| **Healthy** | 6 (ENIGMA) | +2 (netneurolab liu_fc + pyspi term mean) = 8 total |
| **Drug** | 0 | 15–150 (Luppi 2023) — requires manual extraction from supplement |

Run `download_netneurolab_fc.py --use-osfclient` for healthy FC; run `download_luppi_fc_maps.py` for PDF (drug FC extraction manual).
