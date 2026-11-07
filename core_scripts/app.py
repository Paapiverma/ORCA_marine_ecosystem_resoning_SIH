import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import datetime
import os

# -------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="ORCA - Arabian Sea Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
    </style>
""", unsafe_allow_html=True)

from pathlib import Path

# Base directories relative to script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / 'sample_data'

# -------------------------------------------------------------
# 2. Cached Data Loaders
# -------------------------------------------------------------
@st.cache_data
def load_historical_data():
    df_path = None
    for candidate in [BASE_DIR / 'orca_processed_ocean_data.parquet', PROJECT_ROOT / 'orca_processed_ocean_data.parquet']:
        if candidate.exists():
            df_path = candidate
            break
    if df_path is None:
        df_path = SAMPLE_DATA_DIR / 'orca_sample_ocean_data.parquet'

    events_path = None
    for candidate in [BASE_DIR / 'orca_detected_events_2022_2025.csv', PROJECT_ROOT / 'orca_detected_events_2022_2025.csv']:
        if candidate.exists():
            events_path = candidate
            break
    if events_path is None:
        events_path = SAMPLE_DATA_DIR / 'orca_sample_detected_events.csv'

    df = pd.read_parquet(df_path)
    df['time'] = pd.to_datetime(df['time']).dt.date
    events = pd.read_csv(events_path)
    events['time'] = pd.to_datetime(events['time']).dt.date
    return df, events

def load_forecast_data():
    for base in [BASE_DIR, PROJECT_ROOT]:
        fc_parquet = base / 'latest_forecast_predictions.parquet'
        fc_csv = base / 'latest_forecast_events.csv'
        if fc_parquet.exists() and fc_csv.exists():
            df_fc = pd.read_parquet(fc_parquet)
            df_fc['time'] = pd.to_datetime(df_fc['time']).dt.date
            events_fc = pd.read_csv(fc_csv)
            events_fc['time'] = pd.to_datetime(events_fc['time']).dt.date
            return df_fc, events_fc
    return None, None

df_hist, events_hist = load_historical_data()
df_fc, events_fc = load_forecast_data()

# -------------------------------------------------------------
# 3. Sidebar Mode Selection & Controls
# -------------------------------------------------------------
st.sidebar.title("🌊 ORCA Controller")
st.sidebar.caption("Real-Time Marine Health & Upwelling Engine")
st.sidebar.divider()

view_mode = st.sidebar.radio(
    "Operational Mode:",
    ["Historical Archives (2022–2025)", "24-Hour AI Predictive Forecast"]
)

if view_mode == "Historical Archives (2022–2025)":
    min_d, max_d = df_hist['time'].min(), df_hist['time'].max()
    default_d = datetime.date(2022, 8, 4) if min_d <= datetime.date(2022, 8, 4) <= max_d else min_d
    selected_date = st.sidebar.date_input("Select Date:", value=default_d, min_value=min_d, max_value=max_d)
    
    day_data = df_hist[df_hist['time'] == selected_date]
    day_events = events_hist[events_hist['time'] == selected_date]
    mode_tag = "Historical Observation"

else:
    if df_fc is not None and not df_fc.empty:
        selected_date = df_fc['time'].iloc[0]
        day_data = df_fc[df_fc['time'] == selected_date]
        day_events = events_fc[events_fc['time'] == selected_date]
        mode_tag = "⚡ 24-Hour Ahead AI Predictive Forecast"
        st.sidebar.success(f"Loaded Forecast: {selected_date}")
    else:
        st.sidebar.error("Forecast data missing. Run `python run_forecast_pipeline.py` first.")
        st.stop()

st.sidebar.divider()
st.sidebar.subheader("Layer Diagnostics")
show_upwelling = st.sidebar.checkbox("Productive Upwelling (PFZ)", value=True)
show_mhw = st.sidebar.checkbox("Thermal Stress (Marine Heatwaves)", value=True)
show_eutro = st.sidebar.checkbox("Eutrophication / Hypoxia Risk", value=True)
show_jets = st.sidebar.checkbox("Dynamic Perturbations (Fronts/Jets)", value=True)
show_baseline = st.sidebar.checkbox("Baseline Ocean (Sampled)", value=False)

# -------------------------------------------------------------
# 4. Top KPI Metric Cards
# -------------------------------------------------------------
st.title(f"{mode_tag} — {selected_date.strftime('%B %d, %Y')}")

upwelling_slice = day_data[day_data['marine_health_state'] == 'Productive Upwelling']
total_area = day_events[day_events['event_type'] == 'Productive Upwelling']['estimated_area_km2'].sum() if not day_events.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Upwelling Footprint", f"{total_area:,} km²")
c2.metric("Cold Core (Min SSTA)", f"{day_data['SSTA'].min():.2f} °C" if len(day_data) else "0.00 °C")
c3.metric("Peak Marine Health Index", f"{day_data['MHI'].max():.1f} / 100" if len(day_data) else "N/A")
c4.metric("Active Anomaly Grid Cells", f"{len(day_data[day_data['marine_health_state'] != 'Stable Baseline']):,}")

st.divider()

# -------------------------------------------------------------
# 5. Geospatial Map & Tactical Advisory
# -------------------------------------------------------------
col_map, col_info = st.columns([2.2, 1])

with col_map:
    st.subheader("Geospatial Diagnostic Map")
    
    # Satellite Imagery Base Layer
    m = folium.Map(
        location=[14.5, 71.5],
        zoom_start=6,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri Satellite'
    )
    
    color_map = {
        'Productive Upwelling': ('#00ff7f', show_upwelling),
        'Thermal Stress (MHW)': ('#ff3333', show_mhw),
        'Eutrophication Risk': ('#bf00ff', show_eutro),
        'Dynamic Perturbation': ('#3399ff', show_jets)
    }

    # Baseline ocean background
    if show_baseline and 'Stable Baseline' in day_data['marine_health_state'].values:
        for _, row in day_data[day_data['marine_health_state'] == 'Stable Baseline'].iloc[::4].iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=1.5,
                color='#888888',
                fill=True,
                fill_opacity=0.3
            ).add_to(m)

    # Diagnostic anomaly layers
    for state, (color, is_visible) in color_map.items():
        if is_visible:
            subset = day_data[day_data['marine_health_state'] == state]
            for _, row in subset.iloc[::2].iterrows():
                popup_html = f"""
                <div style="font-family: sans-serif; font-size: 11px;">
                    <b>State:</b> <span style="color:{color};">{state}</span><br>
                    <b>SSTA:</b> {row['SSTA']:.2f} °C<br>
                    <b>Current Speed:</b> {row.get('current_speed', 0.0):.2f} m/s<br>
                    <b>MHI Score:</b> {row['MHI']:.1f}/100
                </div>
                """
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3.5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=220)
                ).add_to(m)

    # Event centroids
    if not day_events.empty:
        for _, evt in day_events.iterrows():
            folium.Marker(
                location=[evt['centroid_lat'], evt['centroid_lon']],
                popup=f"Event: {evt['event_id']}<br>Area: {evt['estimated_area_km2']:,} km²",
                icon=folium.Icon(color='green' if 'Upwelling' in evt['event_type'] else 'red', icon='info-sign')
            ).add_to(m)

    # Render without returning viewport state to eliminate rerun loops
    st_folium(m, width="100%", height=560, returned_objects=[])

with col_info:
    st.subheader("Clustered Events")
    if not day_events.empty:
        st.dataframe(
            day_events[['event_type', 'estimated_area_km2', 'centroid_lat', 'centroid_lon']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No multi-pixel clustered events detected.")

    st.subheader("🎣 Commercial Fishing Advisory")
    if not upwelling_slice.empty:
        st.success(f"**{len(upwelling_slice)} Target Coordinates Flagged**")
        pfz_export = upwelling_slice[['latitude', 'longitude', 'SSTA', 'current_speed', 'MHI']].copy()
        pfz_export.columns = ['Lat', 'Lon', 'SSTA (°C)', 'Current (m/s)', 'MHI Score']
        
        st.download_button(
            label="📥 Download Target Coordinates (CSV)",
            data=pfz_export.to_csv(index=False).encode('utf-8'),
            file_name=f"ORCA_Target_Advisory_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No high-vitality upwelling zones flagged for this timeframe.")