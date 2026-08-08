# STEF Stage 1 & Stage 2/3-lite — AKK (Nigeria) and DMIC (India) Corridors

Supplementary code for the GeoAI book chapter case study (Section 7.5): *"Applied
Case Studies: AKK Corridor (Nigeria) and DMIC Corridor (India)"*, part of a Spatial
Transferability Evaluation Framework (STEF) for GeoAI models applied to industrial
infrastructure monitoring.

## What this repository contains

A single Kaggle-runnable notebook (`STEF_Stage1_Kaggle.ipynb`) that computes, from
real public data, for **two** target regions against the same European source
domain:

1. **Stage 1 diagnostics** — Spectral Shift Index (SSI), Label Scarcity Score (LSS)
   proxy, and Infrastructure Informality Index (III) proxy.
2. **Stage 2/3-lite empirical validation** — a Random Forest classifier trained on
   raw Sentinel-2 reflectance against ESA WorldCover land-cover labels, reporting
   in-domain, zero-shot cross-domain, and adapted (few-shot) transfer accuracy.

**Target regions**, both traced from published corridor maps (not straight lines
between endpoint cities):
- **AKK gas corridor, Nigeria**: Ajaokuta → Baro → Suleja → Kaduna → Zaria → Kano
- **DMIC corridor, India**: Dadri → Alwar → Jaipur → Ahmedabad/Gandhinagar →
  Dholera SIR → Surat → JNPT/Mumbai

**Source domain** (both cases): Rhine–Ruhr industrial region, Germany.

## Methodology note: LSS proxy

The LSS proxy uses **VIIRS nighttime-lights radiance** (target-vs-source ratio),
not OpenStreetMap infrastructure-tagging density as originally planned. The
OSM-based approach (via the Overpass API) is blocked by a network restriction in
Kaggle's cloud environment; VIIRS is a GEE-native alternative requiring no external
HTTP calls. Both proxies aim to measure infrastructure-mapping completeness, but
are not numerically equivalent — this is a genuine methodology substitution, not
a re-run of the same method.

## What this is *not*

This is **not** a fine-tuned Prithvi-EO-2.0 / GeoFM domain-adaptation pipeline.
The full pipeline described conceptually in the chapter (Section 7.2) remains
future work. What's here is a classical machine-learning proxy that produces a
real, reproducible measurement of the practical effect Stage 1 predicts.

## Results

See `results/` for the last-run outputs:
- `stef_stage1_nigeria_results.json`, `stef_stage23_nigeria_results.json`
- `stef_stage1_india_results.json`, `stef_stage23_india_results.json`

## Running it

1. Register a (free) Earth Engine project: https://code.earthengine.google.com/register
2. Open `STEF_Stage1_Kaggle.ipynb` in Kaggle (File → Import Notebook), with
   **Internet enabled** in Notebook Settings.
3. Set `PROJECT_ID` in the authentication cell to your own Earth Engine project ID.
4. Run all cells top to bottom. Live Earth Engine/WorldPop/VIIRS calls mean small
   numerical differences from the paper's reported values are expected and normal.

## Requirements

See `requirements.txt`. No GPU required — all cells run on Kaggle's free CPU tier.

## Citation

If you use this code, please cite the book chapter (full citation to be added on
publication). Repository: https://github.com/sauwal948-ops/stef-akk-corridor

## License

MIT — see `LICENSE`.
