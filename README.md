# ORCA: Arabian Sea Ocean Intelligence & Marine Ecosystem Reasoning

> **Notice**: The current work is in [`orca_v2/`](orca_v2/), including the 16-year continuous daily ocean record (2010–2025), physical and biogeochemical indicator rules, and predictive gradient-boosted forecasting benchmarks. The files and documentation below describe the original hackathon version of the project. Heatwave and hypoxia claims and the chlorophyll 'measurements' wording below were not validated; see orca_v2/ for what was tested and its limits.

ORCA is an oceanographic intelligence engine designed for the Arabian Sea (8°N–22°N, 65°E–77.5°E). It detects multi-parameter marine anomalies to pinpoint Productive Upwelling (Potential Fishing Zones / PFZ), Thermal Stress (Marine Heatwaves / MHW), and Coastal Eutrophication / Hypoxia hazards.

The platform couples physical ocean data (sea surface temperature anomalies, currents, sea level anomalies, and wind stress) with bio-optical chlorophyll measurements, assigns a continuous **Marine Health Index (MHI, 0–100)**, and clusters multi-pixel anomalies into discrete trackable events via DBSCAN.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r core_scripts/requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run core_scripts/app.py
```
*Note: If the full historical dataset (`orca_processed_ocean_data.parquet`) is not present, the dashboard automatically falls back to the bundled sample dataset (`sample_data/`). Note that `sample_data/` contains processed model predictions and clustered events for a single day (August 4, 2022), not raw satellite NetCDF data.*

### 3. Run Live 24-Hour Predictive Forecast
```bash
python core_scripts/predict_forecast.py
```
This fetches tomorrow's Arabian Sea Copernicus Marine forecast layers, generates predictions and clustered events (`latest_forecast_predictions.parquet`, `latest_forecast_events.csv`), and makes them viewable in the dashboard's forecast tab.

---

## Copernicus Marine Credentials

The forecast pipeline queries the Copernicus Marine Service API. The `copernicusmarine` client automatically reads credentials from standard environment variables or a local configuration:

### Option A: Standard Environment Variables
```bash
# Linux/macOS
export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"

# Windows (PowerShell)
$env:COPERNICUSMARINE_SERVICE_USERNAME="your_username"
$env:COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
```

### Option B: CLI Login
```bash
copernicusmarine login
```

---

## Model Architecture & Evaluation

The engine uses an unsupervised **Isolation Forest** paired with domain-guided ecological state classification:

1. **Isolation Forest Backbone**:
   - `n_estimators=150`, `max_samples=512`, `contamination=0.03`, trained on ~2.88 million observations across 2022–2025 across 15 physical and biogeochemical features (`thetao`, `SSTA`, `so`, `salinity_anomaly`, `uo`, `vo`, `current_speed`, `sla`, `sla_anomaly`, `u10`, `v10`, `wind_speed`, `wind_stress_proxy`, `chl`, `chl_sst_ratio`).
2. **Seasonal Dynamic Thresholding (JJAS Southwest Monsoon)**:
   - Evaluated during the Southwest Monsoon (June–September) using dynamic percentile thresholds (matching top 2% anomaly scores) to verify seasonal upwelling signal capture against baseline open-ocean physics.
3. **Surrogate Decision Tree Interpretability**:
   - A surrogate Decision Tree Regressor was fitted on Isolation Forest anomaly decision scores to validate physical reasoning: confirming primary splits on Sea Surface Temperature Anomaly (`SSTA < -0.4°C`), Chlorophyll bloom levels (`chl > 0.3 mg/m³`), and Ekman wind-stress forcing.
4. **Evaluation Limits & Case Study**:
   - Unsupervised model: there is no labelled ground truth dataset and no formal accuracy/precision metric. Validation is qualitative, based on a single case study examining whether detected spatial anomalies correspond to physical oceanographic signatures during the Kerala coastal upwelling event of August 4, 2022 (119,140 km² footprint, minimum SSTA of -4.24°C, peak chlorophyll 5.49 mg/m³, and MHI of 71.4/100).
