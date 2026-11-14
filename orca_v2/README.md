# ORCA v2: Arabian Sea Marine Ecosystem Indicators & Forecast (2010–2025)

Contains modified E.U. Copernicus Marine Service Information (CMEMS).

Author: Yash Verma (GitHub: Paapiverma)

Based on the original Smart India Hackathon prototype (https://github.com/SakshamBaurai/ORCA_marine_ecosystem_resoning_SIH); the v2 rebuild in this folder (long-record data pipeline, rules, forecast experiments) is by Yash Verma.

## Summary
- **Rule Alignment with Season & Place**: The rule-based indicators trigger primarily along the Southwest coast of India (Kerala/Karnataka) during the summer monsoon (June–September) and agree moderately with a satellite chlorophyll check (Pearson $r = 0.777$, Spearman $r = 0.819$ over the 13 yearly values 2010–2022: satellite chlorophyll means of 1.53, 1.05, 0.69, 1.25, 1.09, 0.74, 1.77, 1.20, 1.31, 0.61, 0.77, 1.20, 1.54 mg/m³ versus upwelling cell-days of 63,649, 52,971, 49,678, 60,899, 41,837, 35,992, 64,306, 62,854, 70,206, 23,374, 16,828, 39,211, 65,260).
- **Flag Forecast Did Not Beat Persistence**: Predicting daily upwelling flags across horizons of 1, 3, and 5 days failed to beat simple persistence on overall F1 score across 2020–2025 (model F1: 0.948, 0.881, 0.838 vs persistence F1: 0.951, 0.891, 0.852).
- **Onset Forecast Outperformed Baseline on Average but Lost in 2021**: Transition forecasting (conditioned on an unflagged cell at day $D$ turning on at day $D+h$) beat the 2010–2017 seasonal day-of-year baseline on average at 3 days (AP 0.328 vs 0.165, 1.99x) and 5 days (AP 0.368 vs 0.229, 1.61x), but lost to the seasonal climatology in 2021 at 5 days (0.382 vs 0.430, 0.889x).
- **Labels Reflect Our Own Heuristic Rules**: The model predicts flags produced by our own rule thresholds, not independent physical oceanographic phenomena or verified coastal sensor data.
- **No Independent Ground Truth**: The pipeline relies entirely on numerical models and satellite altimetry/chlorophyll; no in-situ ground truth dataset or certified event catalog was available for independent validation.

## 1. What It Does
ORCA v2 is an oceanographic data pipeline and machine learning benchmarking suite designed for coastal upwelling and low-oxygen risk (model-estimated) detection across the Eastern Arabian Sea (8.0°N to 22.0°N, 65.0°E to 78.0°E). It covers a 16-year continuous daily record (2010–2025):
- **Rule-Based Environmental Indicators**: Automated daily classification of coastal upwelling events and sub-surface low-oxygen risk using ocean physics and biogeochemistry.
- **Machine Learning Forecast Benchmark**: Spatial and gradient-boosted forecasting models predicting future upwelling events (flag forecast across horizons of 1, 3, and 5 days) and sudden onset events (across horizons of 1 to 15 days).

---

## 2. Data Sources & Specifications
All ocean environmental data are sourced from the **Copernicus Marine Service (CMEMS)**:
- **Physics**: Mercator GLORYS12V1 (`cmems_mod_glo_phy_my_0.083deg_P1D-m`), 1/12° (~9 km) daily, numerical model output. Variables: `thetao` (temperature), `uo` (eastward velocity), `vo` (northward velocity), depth-averaged 0–50 m.
- **Biogeochemistry**: Copernicus biogeochemistry model (`cmems_mod_glo_bgc_my_0.25deg_P1D-m`), 1/4° (~28 km) daily, numerical model output. Variables: `chl` (chlorophyll-a, 0–50 m), `no3` (nitrate, 0–50 m), `o2` (dissolved oxygen, 0–50 m and 50–150 m). Regridded to the 1/12° physics grid with 25 km nearest-neighbor fill for coastal cells. Chlorophyll, nitrate, and oxygen originate from the Copernicus biogeochemistry model (the same model, so they are not independent signals).
- **Sea Level**: Satellite altimetry (DUACS) (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`), 1/8° (~14 km) daily, satellite observation. Variable: `sla` (sea level anomaly). Regridded to the 1/12° physics grid.
- **Wind**: Not used in v2. Surface wind forcing was not used in the long-record indicator rules or forecast models.

| Variable Type | Product Identifier | Resolution | Depth Range | Source Type | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Physics** | Mercator GLORYS12V1 (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) | 1/12° (~9 km) daily | 0–50 m mean | Model Output | Variables: `thetao`, `uo`, `vo`. Numerical model output. |
| **Biogeochemistry** | Copernicus Biogeochemistry Model (`cmems_mod_glo_bgc_my_0.25deg_P1D-m`) | 1/4° (~28 km) daily | 0–50 m & 50–150 m | Model Output | Variables: `chl`, `no3`, `o2`. Chlorophyll, nitrate, oxygen come from the same model (not independent signals). |
| **Sea Level** | Satellite Altimetry (DUACS) (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`) | 1/8° (~14 km) daily | Surface | Satellite Observation | Variable: `sla`. Satellite altimetry. |
| **Wind** | ERA5 10m Wind | 0.25° daily | Surface (10 m) | Reanalysis | **Not used in v2.** |

*Note on 2023–2025 Data Provenance*: Years 2010–2022 were built from the long-record products listed above (`build_long_year.py`). Years 2023–2025 come from earlier raw downloads whose NetCDF headers state the same products as 2022 (physics: `MERCATOR GLORYS12V1`, BGC: `GLOBAL_REANALYSIS_BIO_001_029` / `MERCATOR FREEBIORYS2V4`, sea level: `DUACS DT merged`). Year 2022 was verified identical to the long-record products (mean difference 0.0000 for all variables), but 2023–2025 were NOT compared with the long-record products. The long-record products extend beyond 2025, so the 2023–2025 inputs could be rebuilt from them.

---

## 3. Indicator Rules & Thresholds
Ocean cells are evaluated daily on an offshore swath across the domain:

1. **Spatial Domain & Offshore References**:
   - **Distance from Land**: Computed using a `cKDTree` on spherical coordinates ($R = 6371.0\text{ km}$). Cells with distance $\ge 200.0\text{ km}$ from the nearest land grid cell are designated as offshore reference cells (`offshore = dist_to_land >= 200.0`).
   - **Latitude Slicing**: For every unique latitude coordinate, offshore reference cells within a $\pm 0.5^\circ$ band (`np.abs(ocean_lats - lat_val) <= 0.5 + 1e-4`) are selected.
   - **Reference Calculations**:
     - `ref_thetao`: Daily spatial median of 0–50 m depth-averaged temperature (`thetao`) among offshore reference cells in that $\pm 0.5^\circ$ latitude band (`np.nanmedian(thetao_2d[:, ref_cells], axis=1)`).
     - `ref_o2deep`: Daily spatial median of 50–150 m depth-averaged dissolved oxygen (`o2_50_150m`) among offshore reference cells in that $\pm 0.5^\circ$ latitude band (`np.nanmedian(o2deep_2d[:, ref_cells], axis=1)`).

2. **Northern Latitude Exclusion (`excluded_north`)**:
   - `LAT_CUTOFF = 19.5°N`. Cells with `latitude >= 19.5°N` are flagged `excluded_north = 1` and their indicator flags (`upwelling`, `low_oxygen_risk`) are strictly forced to `0`: the Gulf of Khambhat produced winter false alarms; shallow, tide-stirred water is a probable cause, which was not tested.

3. **Exclusion of Interpolated / Filled Values**:
   - Biogeochemical fields regridded to the 1/12° physics grid use a 25 km nearest-neighbor fill for coastal grid alignment. Cells where values were filled (`no3_filled == 1` or `chl_filled == 1`) are strictly excluded from upwelling classification (`no3_filled == 0` and `chl_filled == 0`).

4. **Upwelling Flag (`upwelling`)**:
   A cell is flagged as upwelling (`1`) if and only if all of the following conditions hold simultaneously:
   - `latitude < 19.5°N` (`excluded_north == 0`)
   - `(thetao - ref_thetao) <= -1.0` (SST drop $\ge 1.0^\circ\text{C}$ relative to the offshore median for that latitude band)
   - `no3 >= 0.5` (Nitrate concentration $\ge 0.5\text{ mmol/m}^3$)
   - `chl >= 0.5` (Chlorophyll-a concentration $\ge 0.5\text{ mg/m}^3$)
   - `no3_filled == 0` (Nitrate is directly modelled, not coastal-filled)
   - `chl_filled == 0` (Chlorophyll is directly modelled, not coastal-filled)

5. **Low-Oxygen Risk Flag (`low_oxygen_risk`)**:
   A cell is flagged for sub-surface low-oxygen risk (model-estimated) (`1`) if and only if:
   - The cell is actively upwelling: `upwelling == 1` (inherits northern exclusion, un-filled validation, temperature drop, nitrate, and chlorophyll criteria)
   - Deep oxygen data are valid: `~np.isnan(o2_50_150m)` and `~np.isnan(ref_o2deep)` (cells in bathymetry shallower than 50 m do not have 50–150 m depth data and evaluate to `0`)
   - Deep oxygen deficit: `(o2_50_150m - ref_o2deep) <= -50.0` (sub-surface dissolved oxygen drops $\ge 50.0\text{ mmol/m}^3$ below the offshore reference median for that latitude band)

---

## 4. How to Run

### Credentials & Authentication
Never store usernames or passwords in code or environment configuration files. Authenticate using the official Copernicus Marine CLI:
```bash
# Log in via interactive prompt (saves token locally to user credentials store)
copernicusmarine login

# Or set credentials via official environment variables in your terminal session:
# export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
# export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
```

### Model Artifacts & Safety
The model files are pickled scikit-learn objects, to be loaded only from a source you trust, trained with scikit-learn version 1.9.1.

### Pipeline Architecture & Rule Generation Chains
Indicator rules are produced through two distinct pipelines depending on the time period:
- **2010–2022 (Long-Record Pipeline)**: Computed directly inside `build_long_year.py` (or batched across years via `run_long_years.py`). The script downloads physics, BGC, and sea level reanalysis NetCDFs, regrids them to the 1/12° grid, evaluates spatial offshore reference medians, applies the 19.5°N northern exclusion, and saves both `ocean_long_YYYY.parquet` and `rules_v3c_YYYY.parquet` inline.
- **2023–2025 (Operational Store Pipeline)**: Produced sequentially through a three-script chain:
  1. `build_merged_v4.py`: Computes day-of-year rolling climatologies and baseline anomalies from `MERGED_V3`, producing `ocean_v4_YYYY.parquet`.
  2. `build_rules_v2.py`: Evaluates spatial offshore reference medians and rule thresholds, producing `rules_v2_YYYY.parquet`.
  3. `build_rules_v3c.py`: Masks out cells north of 19.5°N (`excluded_north = 1`) to generate final `rules_v3c_YYYY.parquet`.

*Note on Execution*: The full pipeline has not been re-run end to end from a clean clone; only the individual steps were run.

### Path Configuration & Environment Variables
All scripts import centralized paths from `orca_v2/config.py`. Every directory can be overridden via environment variables:
- `ORCA_V2_DIR`: Root directory of the `orca_v2` module (default: directory containing `config.py`).
- `ORCA_PROJECT_ROOT`: Project root working directory (default: `C:\ORCA REMASTERED`).
- `ORCA_LONG_DIR`: Directory storing long-record parquets for 2010–2022 (`ocean_long_YYYY.parquet` and `rules_v3c_YYYY.parquet`; default: `G:\My Drive\ORCA_LONG`).
- `MERGED_V3_DIR`: Directory storing 2022–2025 raw merged parquets (default: `ORCA_PROJECT_ROOT / "MERGED_V3"`).
- `MERGED_V4_DIR`: Directory storing rolling climatology and anomaly parquets (default: `ORCA_PROJECT_ROOT / "MERGED_V4"`).
- `RULES_V2_DIR`: Directory storing intermediate rules v2 parquets (default: `ORCA_PROJECT_ROOT / "RULES_V2"`).
- `RULES_V3C_DIR`: Directory storing final indicator rules v3c parquets (default: `ORCA_PROJECT_ROOT / "RULES_V3C"`).
- `RAW_TEMP_DIR`: Staging directory for temporary raw NetCDF downloads (default: `C:\ORCA_TEMP\raw`).
- `LOGS_DIR`: Directory for pipeline execution logs (default: `ORCA_PROJECT_ROOT / "logs"`).
- `MODELS_DIR`: Directory storing trained scikit-learn model artifacts (default: `ORCA_V2_DIR / "models"`).
- `SAMPLE_DATA_DIR`: Directory storing bundled sample parquets (default: `ORCA_V2_DIR / "sample_data"`).
- `FIGURES_DIR`: Directory for diagnostic plots and map figures (default: `ORCA_V2_DIR / "figures"`).

### Execution Order
```bash
# 1. Download and process long-record years (2010–2022)
# Emits ocean_long_YYYY.parquet and rules_v3c_YYYY.parquet
python orca_v2/run_long_years.py 2010

# 2. Build 2023–2025 climatologies and indicator rules
python orca_v2/build_merged_v4.py
python orca_v2/build_rules_v2.py
python orca_v2/build_rules_v3c.py

# 3. Validate against satellite chlorophyll observations (2010–2022)
python orca_v2/check_satellite_chl.py

# 4. Run Forecast Suite
# Benchmark A: Standard flag forecast vs persistence baseline (FORECAST2)
python orca_v2/forecast2_persistence_test.py

# Benchmark B: Upwelling onset forecast (FORECAST3)
python orca_v2/forecast3_onset_test.py

# Benchmark C: Leakage audit, ablations, and horizon sweep (FORECAST4)
python orca_v2/forecast4_robustness_test.py
```

---

## 5. Quantitative Results (Exact CSV Extraction)

### Benchmark A: Upwelling Flag Forecast vs. Persistence (2020–2025, Jun–Sep)
*(Source data: `orca_v2/results/forecast_metrics.csv`)*  
Target: Predicting `upwelling` flag at $D+h$ across all cells with $\text{latitude} < 19.5^\circ\text{N}$.

| Horizon ($h$) | Model / Baseline | Precision | Recall | F1 Score | Onset Events | Onset Precision | Onset Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$h=1$ day** | **Persistence** | **0.9555** | **0.9471** | **0.9513** | 12,952 | 0.0000 | 0.0000 |
| | HistGradientBoosting | 0.9530 | 0.9440 | 0.9485 | 12,952 | 0.3894 | 0.1526 |
| **$h=3$ days** | **Persistence** | **0.9038** | **0.8792** | **0.8913** | 29,580 | 0.0000 | 0.0000 |
| | HistGradientBoosting | 0.8998 | 0.8638 | 0.8814 | 29,580 | 0.4013 | 0.1818 |
| **$h=5$ days** | **Persistence** | **0.8724** | **0.8325** | **0.8520** | 41,002 | 0.0000 | 0.0000 |
| | HistGradientBoosting | 0.8251 | 0.8519 | 0.8383 | 41,002 | 0.3955 | 0.3488 |

*Result*: The gradient-boosted classification model **did NOT beat the persistence baseline** on overall F1 score across horizons $h=1, 3, 5$ on the 2020–2025 test period.

---

### Benchmark B: Upwelling Onset Forecast (FORECAST3)
*(Source data: `orca_v2/results/onset_metrics.csv`)*  
Task: Conditioned on cell being unflagged at day $D$ (`upwelling == 0`), predict transition to `upwelling == 1` at day $D+h$. Evaluated on 2020–2025 (Jun–Sep).

| Horizon ($h$) | Base Rate | Model AP (`ap_v3`) | Season Baseline AP | Spreading Baseline AP | Flicker Baseline AP | Ratio vs Best Baseline | Model Val Cutoff | Model Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$h=1$ day** | 0.001119 | 0.1139 | 0.0741 | 0.0686 | 0.0380 | **1.54x** | 0.99999 | 0.2343 |
| **$h=3$ days** | 0.002554 | **0.3283** | 0.1654 | 0.0936 | 0.0634 | **1.99x** | 0.78132 | 0.4701 |
| **$h=5$ days** | 0.003539 | **0.3683** | 0.2290 | 0.1061 | 0.0721 | **1.61x** | 0.74824 | 0.4909 |

---

### Benchmark C: Ablation Study (FORECAST4, Jun–Sep 2020–2025)
*(Source data: `orca_v2/results/ablation_metrics.csv`)*  
Comparison of model feature subsets against the historical day-of-year seasonal baseline:
- **V1 (Calendar & Location only)**: latitude, longitude, is_coastal, sin_doy, cos_doy (5 features).
- **V2 (Full minus Rule Ingredients)**: full features removing thetao drop, no3, chl, and window thermal signals (15 features).
- **V3 (Full Model)**: all 22 oceanographic, spatial, lag, and calendar features.

| Horizon ($h$) | Total Rows | Base Rate | Season AP | V1 AP (Calendar/Loc) | V2 AP (No Rule Inputs) | V3 AP (Full Model) | Ratio V3 / Season | Ratio V3 / V1 | Ratio V3 / V2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$h=3$ days** | 11,580,711 | 0.002554 | 0.165360 | 0.128452 | 0.243547 | **0.328301** | **1.985x** | **2.556x** | **1.348x** |
| **$h=5$ days** | 11,585,239 | 0.003539 | 0.229029 | 0.141874 | 0.268910 | **0.368250** | **1.608x** | **2.596x** | **1.369x** |

V1 (calendar and location only) scored below the seasonal baseline, so beating V1 is an easy comparison; the meaningful comparison is the seasonal baseline. removing them (V2) lowers average precision by about a quarter (-26% at h=3, -27% at h=5; equivalently V3 is about 35% above V2), so part of the skill comes from conditions near our own rule thresholds. The model predicts our own rule's flag, not independently measured upwelling.

---

### Benchmark D: Year-by-Year Onset Performance (Jun–Sep, 2020–2025)
*(Source data: `orca_v2/results/per_year_metrics.csv`)*  

| Year | Horizon ($h$) | Onset Events | Base Rate | Model AP (`ap_v3`) | Season Baseline AP | Ratio (V3 / Season) | V3 Beats Season? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2020** | $h=3$ | 3,174 | 0.001625 | 0.2986 | 0.1683 | 1.774x | **PASS (True)** |
| **2021** | $h=3$ | 5,616 | 0.002909 | 0.3257 | 0.2932 | 1.111x | **PASS (True)** |
| **2022** | $h=3$ | 6,796 | 0.003566 | 0.3981 | 0.2386 | 1.669x | **PASS (True)** |
| **2023** | $h=3$ | 4,647 | 0.002410 | 0.2795 | 0.1701 | 1.643x | **PASS (True)** |
| **2024** | $h=3$ | 4,986 | 0.002578 | 0.3230 | 0.1727 | 1.870x | **PASS (True)** |
| **2025** | $h=3$ | 4,361 | 0.002261 | 0.3740 | 0.2828 | 1.323x | **PASS (True)** |
| **2020** | $h=5$ | 4,053 | 0.002074 | 0.2965 | 0.2025 | 1.464x | **PASS (True)** |
| **2021** | $h=5$ | 7,722 | 0.003998 | 0.3821 | **0.4300** | **0.889x** | **FAIL (False)** |
| **2022** | $h=5$ | 9,591 | 0.005030 | 0.4134 | 0.3584 | 1.153x | **PASS (True)** |
| **2023** | $h=5$ | 6,285 | 0.003259 | 0.3124 | 0.2417 | 1.292x | **PASS (True)** |
| **2024** | $h=5$ | 7,096 | 0.003668 | 0.3914 | 0.2535 | 1.544x | **PASS (True)** |
| **2025** | $h=5$ | 6,255 | 0.003242 | 0.4554 | 0.3861 | 1.179x | **PASS (True)** |

*Critical Observation*: At $h=5$, the model beat the season baseline in 5 out of 6 years, but **failed to beat the seasonal climatology in 2021** (ratio 0.889x).

---

### Benchmark E: Horizon Sweep (V3 Full Model, Jun–Sep 2020–2025)
*(Source data: `orca_v2/results/horizon_sweep_metrics.csv`)*  

| Horizon ($h$) | Total Rows | Onset Events | Base Rate | Model AP (`ap_v3`) | Season Baseline AP | Ratio V3 / Season |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 day** | 11,576,186 | 12,952 | 0.001119 | 0.113889 | 0.074057 | 1.538x |
| **3 days** | 11,580,711 | 29,580 | 0.002554 | 0.328301 | 0.165360 | **1.985x** |
| **5 days** | 11,585,239 | 41,002 | 0.003539 | 0.368250 | 0.229029 | 1.608x |
| **7 days** | 11,589,884 | 50,076 | 0.004321 | 0.350288 | 0.276670 | 1.266x |
| **10 days** | 11,596,658 | 61,980 | 0.005345 | 0.448134 | 0.340619 | 1.316x |
| **15 days** | 11,608,192 | 77,167 | 0.006648 | 0.500071 | 0.414725 | 1.206x |

---

## 6. Pass / Fail Evaluation Summary

The criteria below were fixed BEFORE each experiment was run, compared here against the empirical outcomes:

> **Note on Problem Selection**: The onset forecast problem (FORECAST3) was chosen and formulated **after seeing the flag forecast results (FORECAST2)**, which showed that predicting all cells is dominated by persistence.

| Experiment / Check | Pre-Fixed Criterion | Empirical Result (from `orca_v2/results/`) | Outcome |
| :--- | :--- | :--- | :--- |
| **Flag Forecast (FORECAST2)** | Model F1 must beat Persistence F1 by at least **0.02** at $h=3$ and $h=5$ (Jun–Sep, 2020–2025). | At $h=3$: Model F1 = 0.8814 vs Persistence F1 = 0.8913 (diff: **-0.0099**)<br>At $h=5$: Model F1 = 0.8383 vs Persistence F1 = 0.8520 (diff: **-0.0137**)<br>*(from `forecast_metrics.csv`)* | **FAIL** |
| **Onset Forecast (FORECAST3)** | Model Average Precision (AP) $\ge 1.25\times$ best of three baselines (Season, Spreading, Flicker) at $h=3$ and $h=5$ on 2020–2025 Jun–Sep, and also separately on 2020–2021 and on 2022–2025.<br>*(Season was the best baseline in all cases)* | **All six ratios exceed 1.25x**:<br>• $h=3$, 2020–2021: Model 0.3124 vs Season 0.1791 (**1.745x**)<br>• $h=3$, 2022–2025: Model 0.3363 vs Season 0.1751 (**1.920x**)<br>• $h=3$, 2020–2025: Model 0.3283 vs Season 0.1654 (**1.985x**)<br>• $h=5$, 2020–2021: Model 0.3382 vs Season 0.2331 (**1.451x**)<br>• $h=5$, 2022–2025: Model 0.3816 vs Season 0.2553 (**1.495x**)<br>• $h=5$, 2020–2025: Model 0.3683 vs Season 0.2290 (**1.608x**)<br>*(from `onset_metrics.csv`)* | **PASS** |
| **Robustness (a): V3 vs V1 (FORECAST4)** | V3 AP $\ge 1.25\times$ V1 AP (calendar & location only) at $h=3$ and $h=5$ (Jun–Sep, 2020–2025). | At $h=3$: V3 0.328301 vs V1 0.128452 (**2.556x**)<br>At $h=5$: V3 0.368250 vs V1 0.141874 (**2.596x**)<br>*(Note: V1 is a weak control because it scored below the seasonal baseline)*<br>*(from `ablation_metrics.csv`)* | **PASS** |
| **Robustness (b): Yearly Consistency (FORECAST4)** | V3 AP beats Season Baseline AP in each of the six test years (2020–2025) at $h=3$ and $h=5$. | At $h=3$: Beats season in 6 of 6 years (1.11x to 1.87x).<br>At $h=5$: Beats season in 5 of 6 years, but **loses in 2021**: Model AP 0.3821 vs Season AP 0.4300 (**0.889x**)<br>*(from `per_year_metrics.csv`)* | **FAIL** |
| **Robustness (c): V2 vs V3 (FORECAST4)** | *Informational only*: measure performance drop when removing rule ingredients (thetao drop, no3, chl). | Removing rule inputs (V2) lowers AP by about a quarter (-26% at $h=3$, 0.2435 vs 0.3283; -27% at $h=5$, 0.2689 vs 0.3683); V3 lifts over V2 by +34.8% at $h=3$ and +36.9% at $h=5$.<br>*(from `ablation_metrics.csv`)* | **Informational** |
| **Robustness (c): Horizon Sweep (FORECAST4)** | *Informational only*: sweep lead time from $h=1$ to $h=15$ days. | Model AP rises from 0.1139 ($h=1$) to 0.5001 ($h=15$), while Season Baseline AP rises from 0.0741 to 0.4147. Ratio vs baseline peaks at $h=3$ (1.985x) and drops to 1.206x at $h=15$.<br>*(from `horizon_sweep_metrics.csv`)* | **Informational** |

---

## 7. Plain-Language Scientific Limitations (Mandatory)

1. **Rule Labels Are Not Ground Truth**: The upwelling indicators (`upwelling == 1`) are heuristic thresholds defined by oceanographic domain reasoning, not in-situ field measurements, CTD casts, or certified ground truth labels. Any ML model trained on them is learning to replicate the rule behavior from ocean features.
2. **Post-Hoc Threshold Selection & Test Overlap**:
   - The exact rule criteria ($\ge 1.0^\circ\text{C}$ offshore drop, $\ge 0.5\text{ mmol/m}^3$ nitrate, $\ge 0.5\text{ mg/m}^3$ chlorophyll, and $\le -50.0\text{ mmol/m}^3$ deep oxygen deficit) were chosen while inspecting 2022–2025 data, which overlaps the 2020–2025 test years. The **2020–2021 test period is the cleanest evaluation period** because those years were not inspected during threshold design.
   - The $19.5^\circ\text{N}$ cutoff was explicitly added because the Gulf of Khambhat produced winter false alarms; shallow, tide-stirred water is a probable cause, which was not tested.
3. **Spatial and Temporal Correlation**: Neighboring ocean cells (1/12° grid) and consecutive days are strongly correlated. Consequently, cell-day sample counts (millions of rows) substantially overstate the amount of independent statistical evidence.
4. **No Confidence Intervals**: All reported metrics (F1 scores, Average Precision, correlations) are point estimates; no bootstrap confidence intervals or standard errors are reported.
5. **Single Numerical Biogeochemical Model**: Chlorophyll, nitrate, and dissolved oxygen do not come from in-situ water samples or independent sensors, but from a single numerical model reanalysis (Copernicus `GLOBAL_MULTIYEAR_BGC_001_029`). Systematic model biases or parameterization artifacts will directly propagate into the indicator flags and ML training.
6. **Geographic and Seasonal Specificity**: All findings and models are specific to the southwest coast of India (Eastern Arabian Sea) and predominantly to the June–September southwest monsoon season. They cannot be generalized to other upwelling regions or winter conditions without retraining and validation.
7. **Limited Observational Verification**: The only observational cross-check performed was 13 yearly values (2010–2022), one region (8–15°N, 72–77.5°E), June–September mean, correlation $r = 0.78$ (Pearson $r = 0.7774, p = 0.0018$; Spearman $r = 0.8187, p = 0.0006$) between yearly upwelling cell-days and satellite chlorophyll (`cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`). This is a small sample (13 points) from one region.
8. **Model-Estimated Low-Oxygen Risk**: The `low_oxygen_risk` flag is derived from a numerical reanalysis model, not direct dissolved oxygen sensors. It is **invalid in bathymetry shallower than 50 m** where vertical mixing and bottom boundary processes differ substantially from model assumptions.
9. **Marine Heatwaves Omitted**: Dedicated marine heatwave (MHW) detection was not implemented due to the lack of an operational 30-year daily climatological baseline required by standard definitions (e.g. Hobday et al.).
10. **Wind Forcing Omitted**: Surface wind forcing (ERA5 10m wind components $u_{10}, v_{10}$) was not incorporated into the final long-record models because the automated CDS download pipeline could not be completed within execution constraints.
11. **Forecast Generalization & Exploration Caveats**:
    - The standard upwelling flag prediction **failed to outperform the simple persistence baseline** across 2020–2025.
    - The onset-specific formulation was formulated **after inspecting initial forecast failures**, making it an exploratory analysis rather than a pre-registered hypothesis test.
    - While the onset model outperforms the seasonal baseline on average, it **underperformed the baseline in 2021 at $h=5$** (0.3821 vs 0.4300, ratio 0.889x).
