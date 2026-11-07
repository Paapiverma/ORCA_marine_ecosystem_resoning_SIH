# run_forecast_pipeline.py
import datetime
import os
from pathlib import Path
import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN

# Base directories relative to script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / 'model_artifacts'

# -------------------------------------------------------------
# 1. Setup Dates & Parameters
# -------------------------------------------------------------
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
date_str = tomorrow.strftime('%Y-%m-%d')

print(f"📡 [1/3] Fetching 24-hour Arabian Sea Forecast for: {date_str}...")

MIN_LON, MAX_LON = 65.0, 77.5
MIN_LAT, MAX_LAT = 8.0, 22.0

datasets_to_fetch = [
    {"id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "vars": ["thetao"], "file": str(BASE_DIR / f"temp_thetao_{date_str}.nc")},
    {"id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", "vars": ["uo", "vo"], "file": str(BASE_DIR / f"temp_cur_{date_str}.nc")}
]

downloaded_files = []
for item in datasets_to_fetch:
    try:
        copernicusmarine.subset(
            dataset_id=item["id"],
            variables=item["vars"],
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,
            start_datetime=f"{date_str}T00:00:00",
            end_datetime=f"{date_str}T23:59:59",
            output_filename=item["file"],
            overwrite=True
        )
        downloaded_files.append(item["file"])
    except Exception as e:
        print(f"⚠️ Warning: Issue fetching {item['id']}: {e}")

if not downloaded_files:
    raise RuntimeError("❌ No forecast layers could be downloaded. Check internet or credentials.")

# -------------------------------------------------------------
# 2. Merge & Process Features (Windows Safe Lock Release)
# -------------------------------------------------------------
print("🔄 [2/3] Merging physical layers into memory...")

loaded_datasets = []
for f in downloaded_files:
    with xr.open_dataset(f) as ds:
        # Load directly into memory so file handle can be safely released
        loaded_datasets.append(ds.load())

# Now that file handles are closed, delete temp .nc files cleanly
for f in downloaded_files:
    if os.path.exists(f):
        try:
            os.remove(f)
        except PermissionError:
            pass

ds_merged = xr.merge(loaded_datasets)

if 'depth' in ds_merged.dims:
    ds_merged = ds_merged.isel(depth=0)

df_forecast = ds_merged.to_dataframe().dropna().reset_index()

# Derive ocean features
df_forecast['time'] = date_str
df_forecast['current_speed'] = np.sqrt(df_forecast['uo']**2 + df_forecast['vo']**2) if 'uo' in df_forecast else 0.2
df_forecast['SSTA'] = df_forecast['thetao'] - df_forecast['thetao'].mean()
df_forecast['sla_anomaly'] = 0.0

csv_filename = BASE_DIR / f"forecast_processed_{date_str}.csv"
df_forecast.to_csv(csv_filename, index=False)
print(f" Saved: {csv_filename} ({len(df_forecast)} grid coordinates)")

# -------------------------------------------------------------
# 3. Model Inference & DBSCAN Event Clustering
# -------------------------------------------------------------
print("🧠 [3/3] Running ORCA Model Inference & Event Clustering...")

iso_forest_path = BASE_DIR / 'orca_isolation_forest.joblib'
if not iso_forest_path.exists():
    iso_forest_path = MODEL_DIR / 'orca_isolation_forest.joblib'

features_path = BASE_DIR / 'orca_features.joblib'
if not features_path.exists():
    features_path = MODEL_DIR / 'orca_features.joblib'

iso_forest = joblib.load(iso_forest_path)
model_features = joblib.load(features_path)

for col in model_features:
    if col not in df_forecast.columns:
        if 'chl' in col:
            df_forecast[col] = 0.2
        elif 'wind' in col:
            df_forecast[col] = 5.0
        else:
            df_forecast[col] = 0.0

df_forecast['anomaly'] = iso_forest.predict(df_forecast[model_features])
df_forecast['anomaly_score'] = iso_forest.decision_function(df_forecast[model_features])

conditions = [
    (df_forecast['anomaly'] == -1) & (df_forecast['SSTA'] < -0.4),
    (df_forecast['anomaly'] == -1) & (df_forecast['SSTA'] > 0.8),
    (df_forecast['anomaly'] == -1)
]
choices = ['Productive Upwelling', 'Thermal Stress (MHW)', 'Dynamic Perturbation']
df_forecast['marine_health_state'] = np.select(conditions, choices, default='Stable Baseline')

# Derive MHI
thermal_stability = np.clip(1.0 - (np.maximum(df_forecast['SSTA'], 0) / 2.5), 0, 1)
circulation = np.clip(df_forecast['current_speed'] / 0.5, 0, 1)
df_forecast['MHI'] = (0.55 * thermal_stability + 0.45 * circulation) * 100

# Event extraction
event_records = []
event_id_series = pd.Series('None', index=df_forecast.index, dtype='object')
anom_slice = df_forecast[df_forecast['marine_health_state'] != 'Stable Baseline']

for state in anom_slice['marine_health_state'].unique():
    pts = anom_slice[anom_slice['marine_health_state'] == state]
    if len(pts) < 3:
        continue
    coords = pts[['latitude', 'longitude']].to_numpy()
    db = DBSCAN(eps=0.6, min_samples=3).fit(coords)
    for c_idx in set(db.labels_):
        if c_idx == -1:
            continue
        c_pts = pts[db.labels_ == c_idx]
        uid = f"FC_{date_str.replace('-', '')}_{state.replace(' ', '_')}_{c_idx}"
        event_id_series.loc[c_pts.index] = uid
        event_records.append({
            'event_id': uid,
            'time': date_str,
            'event_type': state,
            'pixel_count': len(c_pts),
            'estimated_area_km2': len(c_pts) * 740,
            'centroid_lat': round(c_pts['latitude'].mean(), 2),
            'centroid_lon': round(c_pts['longitude'].mean(), 2),
            'min_SSTA': round(c_pts['SSTA'].min(), 2),
            'mean_MHI': round(c_pts['MHI'].mean(), 1)
        })

df_forecast['event_id'] = event_id_series
forecast_events_df = pd.DataFrame(event_records)

# Export UI Artifacts
df_forecast.to_parquet(BASE_DIR / 'latest_forecast_predictions.parquet', compression='snappy')
forecast_events_df.to_csv(BASE_DIR / 'latest_forecast_events.csv', index=False)

print("\n Forecast Pipeline Completed Successfully!")
print(df_forecast['marine_health_state'].value_counts())
