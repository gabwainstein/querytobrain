# Brainstem Parcels (Brainstem Navigator) in Pipeline Atlas

The pipeline atlas includes **Brainstem Navigator** nuclei (Hansen et al. 2024, Nature Neuroscience) — the same atlas used in "Integrating brainstem and cortical functional architectures."

## Atlas Options

| Atlas | Parcels | Contents |
|-------|---------|----------|
| `glasser+tian` | 392 | Glasser 360 (cortex) + Tian S2 32 (subcortical) |
| `glasser+tian+brainstem` | ~450 | Above + Brainstem Navigator (58 brainstem + 8 diencephalic nuclei) |

## Build

```bash
python neurolab/scripts/build_combined_atlas.py --atlas glasser+tian+brainstem
# Output: neurolab/data/combined_atlas_450.nii.gz
```

## Brainstem Navigator Download

NITRC requires manual download:

1. Run: `python neurolab/scripts/download_brainstem_navigator.py` (opens NITRC page)
2. Download **BrainstemNavigatorv1.0.zip** (~87 MB)
3. Extract to: `neurolab/data/atlas_cache/`
4. Re-run the build

Source: [NITRC Brainstem Navigator](https://www.nitrc.org/projects/brainstemnavig)

## Parcellation Layout

- **1–360:** Glasser cortical
- **361–392:** Tian S2 subcortical (striatum, thalamus, hippocampus, etc.)
- **393–N:** Brainstem Navigator (VTA, SN, LC, PAG, raphe, PnO, parabrachial, vestibular, etc.)

## Rationale

VTA/SN and other brainstem structures often parcellate to zero with Glasser+Tian alone. Brainstem Navigator provides 58 brainstem + 8 diencephalic nuclei in MNI152, matching the atlas used by Hansen et al. 2024.

## Planned Extensions

- **Basal forebrain (Zaborszky)** — Ch1–4 cholinergic nuclei including nucleus basalis of Meynert
- **Hypothalamus (Neudorfer)** — LH, TM, PA, SCh, PH, VM for orexin, histamine, stress, circadian

See [BASAL_FOREBRAIN_HYPOTHALAMUS_ATLAS.md](BASAL_FOREBRAIN_HYPOTHALAMUS_ATLAS.md) for integration plan.
