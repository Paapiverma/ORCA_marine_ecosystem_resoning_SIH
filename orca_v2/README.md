# ORCA v2: Arabian Sea Marine Ecosystem Indicators & Forecast (2010–2025)

## 1. What It Does
ORCA v2 is an oceanographic data pipeline and machine learning benchmarking suite designed for coastal upwelling and hypoxia-risk detection across the Eastern Arabian Sea (8.0°N to 22.0°N, 65.0°E to 78.0°E). It covers a 16-year continuous daily record (2010–2025):
- **Rule-Based Environmental Indicators**: Automated daily classification of coastal upwelling events and sub-surface low-oxygen risk using ocean physics and biogeochemistry.
- **Machine Learning Forecast Benchmark**: Spatial and gradient-boosted forecasting models predicting future upwelling events and sudden onset events across horizons of 1 to 15 days.

---

## 2. Data Sources & Specifications
All ocean environmental data are sourced from the **Copernicus Marine Service (CMEMS)**:

| Variable Type | Product Identifier | Resolution | Depth Range | Source Type | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Physics** | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | 1/12° (~9 km) daily | 0–50 m mean | Numerical Model | Variables: `thetao` (temperature), `uo` (eastward velocity), `vo` (northward velocity). 0–50 m depth-averaged. |
| **Biogeochemistry** | `cmems_mod_glo_bgc_my_0.25deg_P1D-m` | 1/4° (~28 km) daily | 0–50 m & 50–150 m | Numerical Model | Variables: `chl` (chlorophyll-a), `no3` (nitrate), `o2` (dissolved oxygen). Regridded to physics 1/12° grid with 25 km NN fill. **Note: `chl` and `no3` originate from the exact same biogeochemical model simulation.** |
| **Sea Level** | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | 1/8° (~14 km) daily | Surface | Satellite Observation | Variable: `sla` (sea level anomaly). Regridded to physics 1/12° grid. |

*Note on Near-Real-Time Extension (2023–2025)*: The long-record reanalysis spans 2010–2022. The 2023–2025 operational period uses corresponding CMEMS Near-Real-Time (NRT) products (`cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m`, `cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m`).

---

## 3. Indicator Rules & Thresholds
Ocean cells are evaluated daily on an offshore swath within 200 km of the coastline:

1. **Latitude Boundary**:
   - `excluded_north`: Cells with `latitude >= 19.5°N` are strictly excluded from upwelling classification to prevent shallow tidal false alarms in the Gulf of Khambhat.
2. **Offshore Reference SST (`ref_thetao`)**:
   - For each coastal/ocean cell, `ref_thetao` is the 200 km offshore median SST computed within a $\pm 0.5^\circ$ latitude slice.
3. **Upwelling Flag (`upwelling`)**:
   A cell is flagged as upwelling (`1`) if and only if all of the following conditions hold:
   - `latitude < 19.5°N`
   - `thetao <= ref_thetao - 1.0` (SST drop $\ge 1.0^\circ\text{C}$ relative to offshore waters)
   - `no3 >= 0.5` (Nitrate enrichment $\ge 0.5\text{ mmol/m}^3$)
4. **Low-Oxygen Risk Flag (`low_oxygen_risk`)**:
   - `o2_50_150m < 60.0` (Dissolved oxygen averaged between 50 m and 150 m depth drops below $60.0\text{ mmol/m}^3$).

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

### Path Configuration
All scripts import paths from `orca_v2/config.py`. You can override any default path by setting environment variables:
- `ORCA_LONG_DIR`: Directory storing long-record parquet files (`ocean_long_YYYY.parquet` and `rules_v3c_YYYY.parquet`).
- `MERGED_V3_DIR`: Directory storing operational 2023–2025 data.
- `RAW_TEMP_DIR`: Staging directory for temporary raw NetCDF downloads.

### Execution Order
```bash
# 1. Download and process long-record years (e.g. 2021 down to 2010)
python orca_v2/run_long_years.py --start-year 2021 --end-year 2010

# 2. Build indicator rules
python orca_v2/build_rules_v3c.py

# 3. Run Forecast Suite
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
Task: Conditioned on cell being unflagged at day $D$ (`upwelling == 0`), predict transition to `upwelling == 1` at day $D+h$. Evaluated on 2020–2025 (Jun–Sep).

| Horizon ($h$) | Base Rate | Model AP (`ap_v3`) | Season Baseline AP | Spreading Baseline AP | Flicker Baseline AP | Ratio vs Best Baseline | Model Val Cutoff | Model Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$h=1$ day** | 0.001119 | 0.1139 | 0.0741 | 0.0686 | 0.0380 | **1.54x** | 0.99999 | 0.2343 |
| **$h=3$ days** | 0.002554 | **0.3283** | 0.1654 | 0.0936 | 0.0634 | **1.99x** | 0.78132 | 0.4701 |
| **$h=5$ days** | 0.003539 | **0.3683** | 0.2290 | 0.1061 | 0.0721 | **1.61x** | 0.74824 | 0.4909 |

---

### Benchmark C: Ablation Study (FORECAST4, Jun–Sep 2020–2025)
Comparison of model feature subsets against the historical day-of-year seasonal baseline:
- **V1 (Calendar & Location only)**: latitude, longitude, is_coastal, sin_doy, cos_doy (5 features).
- **V2 (Full minus Rule Ingredients)**: full features removing thetao drop, no3, chl, and window thermal signals (15 features).
- **V3 (Full Model)**: all 22 oceanographic, spatial, lag, and calendar features.

| Horizon ($h$) | Total Rows | Base Rate | Season AP | V1 AP (Calendar/Loc) | V2 AP (No Rule Inputs) | V3 AP (Full Model) | Ratio V3 / Season | Ratio V3 / V1 | Ratio V3 / V2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$h=3$ days** | 11,580,711 | 0.002554 | 0.165360 | 0.128452 | 0.243547 | **0.328301** | **1.985x** | **2.556x** | **1.348x** |
| **$h=5$ days** | 11,585,239 | 0.003539 | 0.229029 | 0.141874 | 0.268910 | **0.368250** | **1.608x** | **2.596x** | **1.369x** |

---

### Benchmark D: Year-by-Year Onset Performance (Jun–Sep, 2020–2025)

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

| Objective / Hypothesis | Evaluation Criterion | Empirical Result | Outcome |
| :--- | :--- | :--- | :--- |
| **Flag Forecast vs Persistence (FORECAST2)** | Model F1 > Persistence F1 over 2020–2025 | At $h=1$: 0.9485 vs 0.9513<br>At $h=3$: 0.8814 vs 0.8913<br>At $h=5$: 0.8383 vs 0.8520 | **FAIL** (Model does not beat persistence) |
| **Onset Forecast vs Climatology (FORECAST3)** | Model AP > Season AP on 2020–2025 Jun–Sep | At $h=3$: 0.3283 vs 0.1654 (+98.5%)<br>At $h=5$: 0.3683 vs 0.2290 (+60.8%) | **PASS** (Clear gain over season baseline) |
| **Yearly Consistency at $h=3$ (FORECAST4)** | Model AP > Season AP in each individual test year | Beats season in 6 of 6 years (2020–2025) | **PASS** |
| **Yearly Consistency at $h=5$ (FORECAST4)** | Model AP > Season AP in each individual test year | Beats season in 5 of 6 years; **loses in 2021 (0.3821 vs 0.4300)** | **FAIL** (Inconsistent across strong anomalous years) |
| **Feature Added Value (FORECAST4)** | Full Model (V3) AP > No-Rule-Features (V2) AP | V3 beats V2 by +34.8% at $h=3$, and +36.9% at $h=5$ | **PASS** |

---

## 7. Plain-Language Scientific Limitations (Mandatory)

1. **Rule Labels Are Not Ground Truth**: The upwelling indicators (`upwelling == 1`) are heuristic thresholds defined by oceanographic domain reasoning, not in-situ field measurements or certified ground truth labels. Any ML model trained on them is learning to replicate the rule behavior from ocean features.
2. **Post-Hoc Threshold Selection**: The exact criteria ($\ge 1.0^\circ\text{C}$ offshore drop, $\ge 0.5\text{ mmol/m}^3$ nitrate) were tuned after examining the 2022 Southwest India (Kerala) upwelling event. The $19.5^\circ\text{N}$ cutoff was explicitly added after encountering false-positive triggers in the shallow, high-turbidity waters of the Gulf of Khambhat.
3. **Limited Observational Verification**: The only observational cross-check performed was a 13-year satellite chlorophyll correlation ($r = 0.75$), which represents a limited coastal sample size.
4. **Model-Estimated Hypoxia**: The `low_oxygen_risk` flag is derived from a numerical reanalysis model, not direct dissolved oxygen sensors. It is **invalid in bathymetry shallower than 50 m** where vertical mixing and bottom boundary processes differ substantially from model assumptions.
5. **Marine Heatwaves Omitted**: Dedicated marine heatwave (MHW) detection was not implemented due to the lack of an operational 30-year daily climatological baseline required by standard definitions (e.g. Hobday et al.).
6. **Wind Forcing Omitted**: Surface wind forcing (ERA5 10m wind components $u_{10}, v_{10}$) was not incorporated into the final long-record models because the automated CDS download pipeline could not be completed within execution constraints.
7. **Forecast Generalization & Exploration Caveats**:
   - The standard upwelling flag prediction **failed to outperform the simple persistence baseline** across 2020–2025.
   - The onset-specific formulation was formulated **after inspecting initial forecast failures**, making it an exploratory analysis rather than a pre-registered hypothesis test.
   - While the onset model outperforms the seasonal baseline on average, it **underperformed the baseline in 2021 at $h=5$**.
   - Test years were inspected multiple times during pipeline development.
