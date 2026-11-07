# STEF: spatial-transferability experiment (Random Forest land-cover transfer, Europe to Nigeria and India)

Supplementary code and data for the GeoAI chapter on the Spatial Transferability Evaluation Framework (STEF).
Repository: https://github.com/sauwal948-ops/stef-akk-corridor

## What this is, and what it is not
A classical-ML experiment: a Random Forest is trained on Sentinel-2 reflectance from a European source region
and tested on target regions, using ESA WorldCover v200 (2021) labels and 2021 imagery. It is **not** a
geospatial-foundation-model experiment, uses no EuroSAT or BigEarthNet data, and does **not** predict
infrastructure risk. Adaptation pathways B (LoRA) and C (domain-adversarial) were **not** run.

## Layout
| Path | Content |
|---|---|
| `stef_validation.py` | The single script (stages: `download`, `analyse`, `selftest`) |
| `requirements.txt` | Package versions used for the reported results |
| `corridor_geometries.geojson` | Approximate traced corridor lines, segments and named nodes (not as-built centrelines) |
| `CITATION.cff` | Citation metadata for GitHub's "Cite this repository" |
| `data/` | Cached samples and region statistics (output of the download stage, about 2.7 MB); makes the analysis reproducible without Earth Engine |
| `results_main/` | Sources Rhine-Ruhr and Po Valley (main run) |
| `results_balanced/` | Same, with `class_weight="balanced"` (robustness) |
| `results_bavaria/` | Bavaria as source (robustness) |

Each results folder holds `pair_table.csv`, `correlations.csv`, `per_class_f1.csv`, `confusion_matrices__<source>.json`,
`pathway_rule_evaluation.csv`, `region_characteristics.csv`, `corridor_builtup_profile.csv`,
`corridor_builtup_summary.csv`, `results_summary.json`, `seed_results/`, figures, and `run_manifest.json`
(package versions, configuration, SHA-256 of every data file).

## Reproduce the reported numbers (no Earth Engine account needed)
```
pip install -r requirements.txt
python stef_validation.py analyse --data-dir data --out-dir results_main
python stef_validation.py analyse --data-dir data --out-dir results_balanced --balanced
python stef_validation.py analyse --data-dir data --out-dir results_bavaria --sources "Bavaria DE"
```
The main run took 9,720 s on a CPU-only Kaggle session (two sources, 80 pairs). Finished pairs are cached in
`<out-dir>/pair_cache`, so an interrupted run resumes. `python stef_validation.py selftest` checks the code path
on synthetic data (its numbers are meaningless).

## Design
* Sources: Rhine-Ruhr (DE), Po Valley (IT), Bavaria (DE), each analysed separately.
* Targets (40 per source): the AKK (Nigeria) and DMIC (India) corridors and their 5 and 6 segments (5 km buffers of
  lines traced from published maps; approximate, not as-built centrelines), 24 city regions in Africa, Asia and
  Latin America and European/US controls (15 km-radius discs).
* Classes: tree cover 10, grassland 30, cropland 40, built-up 50; at least 15 sampled pixels per class in both
  domains. The primary analysis uses the 27 targets per source that have all four classes; pixels are sampled at
  natural class proportions.
* Validation: spatially blocked hold-out (0.1 degree blocks, 30 percent held out), 10 seeds, label budgets 50/100/200,
  target-only baseline, and an up-weighted adaptation variant. Chance reference: about 0.22 to 0.24 macro-F1.
* Shift measures: SSI = mean over B5 and B11 of 1 - BC (BC is the Bhattacharyya coefficient; 1 - BC = 1 - exp(-D_B),
  the squared Hellinger distance), SSI over seven bands, RBF MMD, domain-classifier AUC and class-matched AUC.
* Context measures: E (ERA5-Land temperature and precipitation similarity), a night-light activity proxy,
  III_POP (population-density heterogeneity) and III_2C (adds land-surface-temperature variation).

## Limitations
* Exploratory: 27 four-class targets per source, one classifier, one year, three European sources, two corridors.
  WorldCover regional accuracy was not assessed. Only AKK, DMIC and their segments are corridors.
* The night-light proxy measures economic activity, **not** label availability; III_POP does not separate formal
  from informal settings. The OpenStreetMap-based label-scarcity score and tag term of III were not computed.
* The pathway thresholds and equal TS weights are illustrative and uncalibrated; the pathway rule is evaluated, not validated.
* Decisions made after interim results (disclosed in the chapter): E scales fixed after an early version proved
  degenerate; restriction to four classes and the weighted-adaptation variant added after a mixed-class run;
  balanced and Bavaria runs and the pooled analysis added later.

## Data, licensing and responsible use
* Stage A (`download`) uses Google Earth Engine, free for non-commercial research and requiring a paid licence for
  commercial use. Sentinel-2, ESA WorldCover, VIIRS, WorldPop, ERA5-Land and Landsat 8 are accessed through it under
  their own terms. OpenStreetMap is **not** used.
* No operator or agency data were used or consulted. The repository reports land-cover transfer accuracy and
  built-up fractions only, not vulnerability or risk; outputs are aggregated to segments and 10 km sampling points.

## Citation
Cite the chapter (full citation to be added) and the release tag and commit hash used. 

## License
MIT, see `LICENSE`.
