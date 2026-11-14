"""
Build ONE year long-record Copernicus ocean dataset and comparison pipeline.

Products:
  - Physics: cmems_mod_glo_phy_my_0.083deg_P1D-m (thetao, so, uo, vo, depth 0-50m)
  - Biogeochemistry: cmems_mod_glo_bgc_my_0.25deg_P1D-m (chl, no3, o2, po4, depth 0-200m)
  - Sea level: cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D (sla)

Region: 8.0 to 22.0 N, 65.0 to 78.0 E
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

# Constants
EARTH_RADIUS_KM = 6371.0
COAST_KM = 20.0
MAX_FILL_KM = 25.0
OFFSHORE_KM = 200.0
LAT_HALF_BAND = 0.5 + 1e-4
LAT_CUTOFF = 19.5

FILL_VARS = ("chl", "o2", "no3", "po4")
UNFILLED_VARS = ("sla", "o2_50_150m")

try:
    from config import (
        RAW_TEMP_DIR,
        ORCA_LONG_DIR as DEFAULT_OUT_DIR,
        MERGED_V3_DIR as EXISTING_V3_DIR,
        RULES_V3C_DIR as EXISTING_RULES_DIR,
    )
except ImportError:
    from orca_v2.config import (
        RAW_TEMP_DIR,
        ORCA_LONG_DIR as DEFAULT_OUT_DIR,
        MERGED_V3_DIR as EXISTING_V3_DIR,
        RULES_V3C_DIR as EXISTING_RULES_DIR,
    )


def check_copernicus_credentials():
    """Verify copernicusmarine credentials without logging or exposing secrets."""
    import subprocess
    cmd = ["copernicusmarine", "login", "--check-credentials-valid"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return True, "Valid credentials found."
        return False, "Not logged in: no valid Copernicus Marine credentials found."
    except Exception as e:
        return False, f"Failed to check credentials: {e}"


def latlon_to_xyz_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon degrees to 3D Cartesian coordinates (km) on sphere."""
    lat_r = np.radians(lat.astype(np.float64))
    lon_r = np.radians(lon.astype(np.float64))
    cos_lat = np.cos(lat_r)
    return np.column_stack([
        EARTH_RADIUS_KM * cos_lat * np.cos(lon_r),
        EARTH_RADIUS_KM * cos_lat * np.sin(lon_r),
        EARTH_RADIUS_KM * np.sin(lat_r),
    ])


def download_year_raw(year: int, raw_dir: Path) -> dict:
    """Download physics, bgc, and sla datasets for the specified year using copernicusmarine."""
    import copernicusmarine

    raw_dir.mkdir(parents=True, exist_ok=True)
    phys_file = raw_dir / f"physics_long_{year}.nc"
    bgc_file = raw_dir / f"bgc_long_{year}.nc"
    sla_file = raw_dir / f"sla_long_{year}.nc"

    start_date = f"{year}-01-01T00:00:00"
    end_date = f"{year}-12-31T23:59:59"

    download_stats = {}

    # 1. Physics (0.083 deg, thetao, so, uo, vo, depth 0-50m)
    print(f"\n[1/3] Downloading Physics (cmems_mod_glo_phy_my_0.083deg_P1D-m) for {year}...")
    t0 = time.time()
    res_p = copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so", "uo", "vo"],
        minimum_longitude=65.0,
        maximum_longitude=78.0,
        minimum_latitude=8.0,
        maximum_latitude=22.0,
        minimum_depth=0.0,
        maximum_depth=50.0,
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=phys_file.name,
        output_directory=str(raw_dir),
        overwrite=True,
    )
    phys_file = Path(res_p.file_path)
    t_phys = time.time() - t0
    phys_sz = phys_file.stat().st_size
    print(f"Physics downloaded: {phys_sz / 1e6:.2f} MB in {t_phys:.1f}s")
    download_stats["physics"] = {"file": phys_file, "size_bytes": phys_sz, "time_s": t_phys}

    # 2. Biogeochemistry (0.25 deg, chl, no3, o2, po4, depth 0-200m)
    print(f"\n[2/3] Downloading BGC (cmems_mod_glo_bgc_my_0.25deg_P1D-m) for {year}...")
    t0 = time.time()
    res_b = copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        variables=["chl", "no3", "o2", "po4"],
        minimum_longitude=65.0,
        maximum_longitude=78.0,
        minimum_latitude=8.0,
        maximum_latitude=22.0,
        minimum_depth=0.0,
        maximum_depth=200.0,
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=bgc_file.name,
        output_directory=str(raw_dir),
        overwrite=True,
    )
    bgc_file = Path(res_b.file_path)
    t_bgc = time.time() - t0
    bgc_sz = bgc_file.stat().st_size
    print(f"BGC downloaded: {bgc_sz / 1e6:.2f} MB in {t_bgc:.1f}s")
    download_stats["bgc"] = {"file": bgc_file, "size_bytes": bgc_sz, "time_s": t_bgc}

    # 3. Sea level anomaly (0.125 deg, sla)
    print(f"\n[3/3] Downloading Sea Level (cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D) for {year}...")
    t0 = time.time()
    res_s = copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla"],
        minimum_longitude=65.0,
        maximum_longitude=78.0,
        minimum_latitude=8.0,
        maximum_latitude=22.0,
        start_datetime=start_date,
        end_datetime=end_date,
        output_filename=sla_file.name,
        output_directory=str(raw_dir),
        overwrite=True,
    )
    sla_file = Path(res_s.file_path)
    t_sla = time.time() - t0
    sla_sz = sla_file.stat().st_size
    print(f"Sea level downloaded: {sla_sz / 1e6:.2f} MB in {t_sla:.1f}s")
    download_stats["sla"] = {"file": sla_file, "size_bytes": sla_sz, "time_s": t_sla}

    return download_stats


def process_and_merge(phys_file: Path, bgc_file: Path, sla_file: Path) -> pd.DataFrame:
    """Exact logic of build_merged_v2.py (0-50m mean, regridding to 1/12 deg, o2_50_150m mean)."""
    t0 = time.time()
    print("Loading & 0-50m averaging physics...")
    with xr.open_dataset(phys_file) as ds_p:
        target_lat = ds_p.latitude.values
        target_lon = ds_p.longitude.values
        p_surf = ds_p[['thetao', 'so', 'uo', 'vo']].sel(depth=slice(0, 50.0)).mean(dim='depth').load()
    land_mask = np.isnan(p_surf['thetao'].values)
    print(f"Physics ready in {time.time()-t0:.1f}s. Grid: {len(target_lat)} lats x {len(target_lon)} lons.")

    # Surface BGC (0-50m mean)
    t_step = time.time()
    print("Loading & regridding surface BGC (chl, no3, o2, po4 0-50m)...")
    with xr.open_dataset(bgc_file) as ds_b:
        b_surf = ds_b[['chl', 'no3', 'o2', 'po4']].sel(depth=slice(0, 50.0)).mean(dim='depth').load()
        b_regrid = b_surf.interp(latitude=target_lat, longitude=target_lon, method='linear')
    for v in ['chl', 'no3', 'o2', 'po4']:
        arr = b_regrid[v].values
        arr[land_mask] = np.nan
        b_regrid[v].values = arr
    print(f"Surface BGC ready in {time.time()-t_step:.1f}s.")

    # Deep BGC Oxygen (50-150m mean)
    t_step = time.time()
    print("Loading & regridding deep BGC oxygen (50-150m)...")
    with xr.open_dataset(bgc_file) as ds_b:
        d_sub = ds_b.sel(depth=slice(50.0, 150.0))
        d_mean = d_sub[['o2']].mean(dim='depth').load().rename({'o2': 'o2_50_150m'})
        d_regrid = d_mean.interp(latitude=target_lat, longitude=target_lon, method='linear')
    d_vals = d_regrid['o2_50_150m'].values
    d_vals[land_mask] = np.nan
    d_regrid['o2_50_150m'].values = d_vals
    print(f"Deep BGC oxygen ready in {time.time()-t_step:.1f}s.")

    # Sea level anomaly
    t_step = time.time()
    print("Loading & regridding SSHA (sla)...")
    with xr.open_dataset(sla_file) as ds_s:
        s_regrid = ds_s[['sla']].interp(latitude=target_lat, longitude=target_lon, method='linear').load()
    s_vals = s_regrid['sla'].values
    s_vals[land_mask] = np.nan
    s_regrid['sla'].values = s_vals
    print(f"SSHA ready in {time.time()-t_step:.1f}s.")

    # Align time coordinates to physics time
    time_coords = p_surf['time'].values
    b_regrid['time'] = time_coords
    d_regrid['time'] = time_coords
    s_regrid['time'] = time_coords

    # Merge into DataFrame
    t_step = time.time()
    print("Merging arrays into single DataFrame...")
    combined_ds = xr.merge([
        p_surf,
        b_regrid,
        d_regrid,
        s_regrid
    ])
    df = combined_ds.to_dataframe().reset_index()

    float_cols = ['latitude', 'longitude', 'thetao', 'so', 'uo', 'vo', 'chl', 'sla', 'o2', 'no3', 'po4', 'o2_50_150m']
    df[float_cols] = df[float_cols].astype(np.float32)

    cols_order = ['time', 'latitude', 'longitude'] + [c for c in float_cols if c not in ['latitude', 'longitude']]
    df = df[cols_order]
    print(f"Merged DataFrame formed in {time.time()-t_step:.1f}s ({len(df):,} total rows).")
    return df


def clean_land_and_coastal(df: pd.DataFrame) -> pd.DataFrame:
    """Exact land-drop and is_coastal 20km KD-tree step from clean_merged_v2.py."""
    t_step = time.time()
    print("Building coastal mask and dropping land cells (clean_merged_v2.py logic)...")
    df_day1 = df[df["time"] == df["time"].iloc[0]]
    is_land = df_day1["thetao"].isna().values
    coords = df_day1[["latitude", "longitude"]].values

    lat_r = np.radians(coords[:, 0])
    lon_r = np.radians(coords[:, 1])
    xyz = np.column_stack([
        EARTH_RADIUS_KM * np.cos(lat_r) * np.cos(lon_r),
        EARTH_RADIUS_KM * np.cos(lat_r) * np.sin(lon_r),
        EARTH_RADIUS_KM * np.sin(lat_r),
    ])

    tree = cKDTree(xyz[is_land])
    dists, _ = tree.query(xyz[~is_land])

    coastal_1d = np.zeros(len(coords), dtype=np.int8)
    ocean_idx = np.where(~is_land)[0]
    coastal_1d[ocean_idx[dists <= COAST_KM]] = 1

    lats = np.sort(df_day1["latitude"].unique())
    lons = np.sort(df_day1["longitude"].unique())
    grid = np.zeros((len(lats), len(lons)), dtype=np.int8)
    lat_i = np.searchsorted(lats, coords[:, 0])
    lon_i = np.searchsorted(lons, coords[:, 1])
    grid[lat_i, lon_i] = coastal_1d

    # Drop land rows
    ocean_mask = df["thetao"].notna().values
    df_clean = df.loc[ocean_mask].reset_index(drop=True)

    lat_idx = np.searchsorted(lats, df_clean["latitude"].values)
    lon_idx = np.searchsorted(lons, df_clean["longitude"].values)
    assert np.array_equal(lats[lat_idx], df_clean["latitude"].values)
    assert np.array_equal(lons[lon_idx], df_clean["longitude"].values)

    df_clean["is_coastal"] = grid[lat_idx, lon_idx]
    print(f"Cleaned: {len(df_clean):,} ocean rows in {time.time()-t_step:.1f}s.")
    return df_clean


def resolve_day_fill_indices(lat_day: np.ndarray, lon_day: np.ndarray, nan_mask: np.ndarray, cache: dict):
    """Return (target_idx, source_idx, dists_km) for NaN cells with neighbor <= 25 km."""
    if not nan_mask.any() or nan_mask.all():
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float64)

    coord_key = (len(lat_day), lat_day.tobytes(), lon_day.tobytes())
    xyz = cache.get(("xyz", coord_key))
    if xyz is None:
        xyz = latlon_to_xyz_km(lat_day, lon_day)
        cache[("xyz", coord_key)] = xyz

    mask_key = (coord_key, nan_mask.tobytes())
    cached = cache.get(("map", mask_key))
    if cached is not None:
        return cached

    valid_pos = np.flatnonzero(~nan_mask)
    nan_pos = np.flatnonzero(nan_mask)
    tree = cKDTree(xyz[valid_pos])
    dists, nn_idx = tree.query(xyz[nan_pos], distance_upper_bound=MAX_FILL_KM + 1e-6)
    within = np.isfinite(dists) & (dists <= MAX_FILL_KM)

    result = (nan_pos[within], valid_pos[nn_idx[within]], dists[within])
    cache[("map", mask_key)] = result
    return result


def fill_nearest_neighbor(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Exact nearest-neighbour fill within 25 km from build_merged_v3.py."""
    t_step = time.time()
    print("Performing nearest-neighbour 25km fill (build_merged_v3.py logic)...")
    df_out = df_clean.copy()
    lat_all = df_out["latitude"].values
    lon_all = df_out["longitude"].values
    day_groups = list(df_out.groupby("time", sort=False).indices.values())

    cache = {}
    for var in FILL_VARS:
        orig_vals = df_out[var].values
        filled_vals = orig_vals.copy()
        filled_flag = np.zeros(len(df_out), dtype=np.int8)

        for row_idx in day_groups:
            day_vals = orig_vals[row_idx]
            day_nan = np.isnan(day_vals)
            tgt_rel, src_rel, _ = resolve_day_fill_indices(
                lat_all[row_idx], lon_all[row_idx], day_nan, cache
            )
            if len(tgt_rel) == 0:
                continue
            tgt_abs = row_idx[tgt_rel]
            src_abs = row_idx[src_rel]
            filled_vals[tgt_abs] = orig_vals[src_abs]
            filled_flag[tgt_abs] = 1

        df_out[var] = filled_vals.astype(np.float32)
        df_out[f"{var}_filled"] = filled_flag

    # Reorder columns to exact MERGED_V3 schema (minus u10, v10)
    cols = [
        'time', 'latitude', 'longitude',
        'thetao', 'so', 'uo', 'vo',
        'chl', 'sla', 'o2', 'no3', 'po4', 'o2_50_150m',
        'is_coastal', 'chl_filled', 'o2_filled', 'no3_filled', 'po4_filled'
    ]
    df_out = df_out[cols]
    print(f"Fill complete in {time.time()-t_step:.1f}s.")
    return df_out


def build_spatial_lookup(df_sample: pd.DataFrame, n_cells: int = 18650):
    """Build land distances and per-latitude >=200km reference index lists for rules."""
    d0 = df_sample[df_sample["time"] == df_sample["time"].iloc[0]]
    ocean_coords = d0[["latitude", "longitude"]].to_numpy()
    assert len(ocean_coords) == n_cells, f"Expected {n_cells} cells, got {len(ocean_coords)}"

    lats_u = np.sort(d0["latitude"].unique())
    lons_u = np.sort(d0["longitude"].unique())
    full_grid = np.array([(la, lo) for la in lats_u for lo in lons_u], dtype=np.float32)
    ocean_set = set(map(tuple, ocean_coords))
    land_coords = np.array([pt for pt in map(tuple, full_grid) if pt not in ocean_set], dtype=np.float32)

    tree = cKDTree(latlon_to_xyz_km(land_coords[:, 0], land_coords[:, 1]))
    dist_to_land, _ = tree.query(latlon_to_xyz_km(ocean_coords[:, 0], ocean_coords[:, 1]))

    offshore = dist_to_land >= OFFSHORE_KM
    ocean_lats = ocean_coords[:, 0]
    lat_to_idx = np.searchsorted(lats_u, ocean_lats)
    assert np.array_equal(lats_u[lat_to_idx], ocean_lats)

    band_ref_indices = []
    for lat_val in lats_u:
        idx = np.flatnonzero(offshore & (np.abs(ocean_lats - lat_val) <= LAT_HALF_BAND))
        assert len(idx) > 0, f"No >=200 km reference cells near latitude {lat_val}"
        band_ref_indices.append(idx)

    return ocean_coords, lat_to_idx, band_ref_indices


def compute_rules_v3c(df: pd.DataFrame, n_cells: int = 18650) -> pd.DataFrame:
    """Exact logic of build_rules_v2.py + build_rules_v3c.py (lat < 19.5N cut)."""
    t_step = time.time()
    print("Computing upwelling & low_oxygen_risk rules (v2 + v3c 19.5N cutoff)...")
    ocean_coords, lat_to_idx, band_ref_indices = build_spatial_lookup(df, n_cells)
    ocean_lats = ocean_coords[:, 0]

    n_days = len(df) // n_cells
    assert len(df) == n_days * n_cells

    thetao_2d = df["thetao"].to_numpy().reshape(n_days, n_cells)
    o2deep_2d = df["o2_50_150m"].to_numpy().reshape(n_days, n_cells)
    n_lats = len(band_ref_indices)

    ref_thetao_lat = np.empty((n_days, n_lats), dtype=np.float32)
    ref_o2deep_lat = np.empty((n_days, n_lats), dtype=np.float32)

    for l_idx, ref_cells in enumerate(band_ref_indices):
        ref_thetao_lat[:, l_idx] = np.nanmedian(thetao_2d[:, ref_cells], axis=1).astype(np.float32)
        ref_o2deep_lat[:, l_idx] = np.nanmedian(o2deep_2d[:, ref_cells], axis=1).astype(np.float32)

    ref_thetao = ref_thetao_lat[:, lat_to_idx].reshape(-1)
    ref_o2deep = ref_o2deep_lat[:, lat_to_idx].reshape(-1)

    upwelling = (
        ((df["thetao"].to_numpy() - ref_thetao) <= -1.0)
        & (df["no3"].to_numpy() >= 0.5)
        & (df["chl"].to_numpy() >= 0.5)
        & (df["no3_filled"].to_numpy() == 0)
        & (df["chl_filled"].to_numpy() == 0)
    )
    o2deep = df["o2_50_150m"].to_numpy()
    low_oxygen_risk = upwelling & (~np.isnan(o2deep)) & (~np.isnan(ref_o2deep)) & ((o2deep - ref_o2deep) <= -50.0)

    # 19.5N northern cutoff
    excl_1d = ocean_lats >= LAT_CUTOFF
    excl_col = np.tile(excl_1d.astype(np.int8), n_days)
    keep_mask = excl_col == 0

    upwelling_v3c = np.where(keep_mask, upwelling, False).astype(np.int8)
    low_oxygen_v3c = np.where(keep_mask, low_oxygen_risk, False).astype(np.int8)

    df_rules = pd.DataFrame({
        "time": df["time"],
        "latitude": df["latitude"].astype(np.float32),
        "longitude": df["longitude"].astype(np.float32),
        "is_coastal": df["is_coastal"].astype(np.int8),
        "excluded_north": excl_col,
        "ref_thetao": ref_thetao,
        "upwelling": upwelling_v3c,
        "low_oxygen_risk": low_oxygen_v3c,
    })
    print(f"Rules computed in {time.time()-t_step:.1f}s.")
    return df_rules


def compare_with_existing_v3(df_new: pd.DataFrame, df_old: pd.DataFrame) -> dict:
    """Compare new vs existing MERGED_V3 over valid cells."""
    variables = ["thetao", "so", "uo", "vo", "chl", "sla", "o2", "no3", "po4", "o2_50_150m"]
    comp_results = {}

    for var in variables:
        v_new = df_new[var].to_numpy()
        v_old = df_old[var].to_numpy()

        valid = (~np.isnan(v_new)) & (~np.isnan(v_old))
        n_valid = int(valid.sum())
        if n_valid == 0:
            comp_results[var] = {"valid_cells": 0}
            continue

        diff = np.abs(v_new[valid] - v_old[valid])
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = float(np.max(diff))

        # Correlation
        corr = float(np.corrcoef(v_new[valid], v_old[valid])[0, 1])

        # Relative mean difference (%)
        mean_old = float(np.mean(v_old[valid]))
        mean_new = float(np.mean(v_new[valid]))
        rel_diff_pct = float(abs(mean_new - mean_old) / abs(mean_old) * 100.0) if mean_old != 0 else 0.0

        comp_results[var] = {
            "valid_cells": n_valid,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "correlation": corr,
            "mean_new": mean_new,
            "mean_old": mean_old,
            "rel_diff_pct": rel_diff_pct,
        }

    return comp_results


def evaluate_pass_fail(comp_results: dict, sw_new_celldays: int, sw_old_celldays: int) -> dict:
    """Evaluate PASS/FAIL criteria fixed in advance."""
    # (a) thetao mean absolute diff < 0.05 degC
    thetao_diff = comp_results["thetao"]["mean_abs_diff"]
    pass_a = thetao_diff < 0.05

    # (b) chl, no3, o2 each within 5% relative mean diff
    chl_rel = comp_results["chl"]["rel_diff_pct"]
    no3_rel = comp_results["no3"]["rel_diff_pct"]
    o2_rel = comp_results["o2"]["rel_diff_pct"]
    pass_b = (chl_rel <= 5.0) and (no3_rel <= 5.0) and (o2_rel <= 5.0)

    # (c) sla mean absolute diff < 0.02 m
    sla_diff = comp_results["sla"]["mean_abs_diff"]
    pass_c = sla_diff < 0.02

    # (d) SW-box Jun-Sep upwelling cell-days within 5% of existing
    sw_rel_pct = abs(sw_new_celldays - sw_old_celldays) / sw_old_celldays * 100.0
    pass_d = sw_rel_pct <= 5.0

    return {
        "(a) thetao mean abs diff < 0.05 degC": {
            "value": f"{thetao_diff:.4f} °C",
            "threshold": "< 0.05 °C",
            "pass": pass_a,
        },
        "(b) chl, no3, o2 relative mean diff <= 5%": {
            "value": f"chl: {chl_rel:.2f}%, no3: {no3_rel:.2f}%, o2: {o2_rel:.2f}%",
            "threshold": "all <= 5.0%",
            "pass": pass_b,
        },
        "(c) sla mean abs diff < 0.02 m": {
            "value": f"{sla_diff:.4f} m",
            "threshold": "< 0.02 m",
            "pass": pass_c,
        },
        "(d) SW-box Jun-Sep upwelling cell-days within 5%": {
            "value": f"New: {sw_new_celldays:,} vs Old: {sw_old_celldays:,} (diff: {sw_rel_pct:.2f}%)",
            "threshold": "<= 5.0%",
            "pass": pass_d,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build and validate long-record Copernicus pilot year.")
    parser.add_argument("year", type=int, help="Year to process (e.g. 2022)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Destination folder on Google Drive")
    parser.add_argument("--raw-dir", type=Path, default=RAW_TEMP_DIR, help="Temp folder for raw NetCDFs")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if raw files already exist")
    args = parser.parse_args()

    year = args.year
    out_dir = args.out_dir
    raw_dir = args.raw_dir

    print("=================================================================")
    print(f" ORCA LONG-RECORD PILOT PIPELINE - YEAR: {year} ")
    print("=================================================================")

    # Confirm Google Drive path
    if not out_dir.parent.exists():
        sys.exit(f"ERROR: Google Drive parent directory does not exist: {out_dir.parent}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination folder confirmed: {out_dir}")

    # Check if final file already exists
    final_parquet = out_dir / f"ocean_long_{year}.parquet"
    rules_parquet = out_dir / f"rules_v3c_{year}.parquet"
    if final_parquet.exists():
        print(f"Output file {final_parquet} already exists. Skipping processing for {year}.")
        return

    # Check credentials
    logged_in, msg = check_copernicus_credentials()
    if not logged_in and not args.skip_download:
        print(f"\n[CREDENTIAL CHECK FAILED]: {msg}")
        print("Please run `copernicusmarine login` in your terminal to log in, then rerun this script.")
        sys.exit(1)

    t_total_start = time.time()
    timings = {}

    # Step 1: Download
    phys_file = raw_dir / f"physics_long_{year}.nc"
    bgc_file = raw_dir / f"bgc_long_{year}.nc"
    sla_file = raw_dir / f"sla_long_{year}.nc"

    if args.skip_download and phys_file.exists() and bgc_file.exists() and sla_file.exists():
        print(f"Using existing raw NetCDFs in {raw_dir}")
        dl_stats = {
            "physics": {"file": phys_file, "size_bytes": phys_file.stat().st_size, "time_s": 0},
            "bgc": {"file": bgc_file, "size_bytes": bgc_file.stat().st_size, "time_s": 0},
            "sla": {"file": sla_file, "size_bytes": sla_file.stat().st_size, "time_s": 0},
        }
    else:
        t_dl_start = time.time()
        dl_stats = download_year_raw(year, raw_dir)
        timings["download"] = time.time() - t_dl_start

    total_raw_bytes = sum(s["size_bytes"] for s in dl_stats.values())
    print(f"\nTotal raw NetCDF download size: {total_raw_bytes / 1e6:.2f} MB ({total_raw_bytes / 1e9:.3f} GB)")

    # Step 2: Merge & Regrid
    t_step = time.time()
    df_merged = process_and_merge(phys_file, bgc_file, sla_file)
    timings["merge_regrid"] = time.time() - t_step

    # Step 3: Land-drop & Coastal flag
    t_step = time.time()
    df_clean = clean_land_and_coastal(df_merged)
    timings["clean_coastal"] = time.time() - t_step

    # Step 4: Nearest-neighbour fill
    t_step = time.time()
    df_final = fill_nearest_neighbor(df_clean)
    timings["nn_fill"] = time.time() - t_step

    # Step 5: Atomic write of final parquet
    t_step = time.time()
    tmp_final = out_dir / f".ocean_long_{year}.parquet.tmp"
    print(f"Writing to temporary file {tmp_final}...")
    df_final.to_parquet(tmp_final, compression="snappy", index=False)
    if final_parquet.exists():
        final_parquet.unlink()
    tmp_final.rename(final_parquet)
    final_sz = final_parquet.stat().st_size
    print(f"Final parquet saved: {final_parquet} ({final_sz / 1e6:.2f} MB)")
    timings["save_parquet"] = time.time() - t_step

    # Step 6: Delete raw files
    print("Deleting raw NetCDF files...")
    for s in dl_stats.values():
        f = s["file"]
        if f.exists():
            f.unlink()
            print(f"Deleted {f.name}")

    # Step 7: Apply RULES_V2 and RULES_V3C logic
    t_step = time.time()
    df_rules = compute_rules_v3c(df_final)
    tmp_rules = out_dir / f".rules_v3c_{year}.parquet.tmp"
    df_rules.to_parquet(tmp_rules, compression="snappy", index=False)
    if rules_parquet.exists():
        rules_parquet.unlink()
    tmp_rules.rename(rules_parquet)
    timings["rules_v3c"] = time.time() - t_step
    print(f"Rules parquet saved: {rules_parquet} ({rules_parquet.stat().st_size / 1e6:.2f} MB)")

    # Step 8: Upwelling cell-days in SW box (8-15N, 72-77.5E) Jun-Sep
    df_rules["month"] = pd.DatetimeIndex(df_rules["time"]).month
    sw_mask = (
        (df_rules["latitude"] >= 8.0) & (df_rules["latitude"] <= 15.0)
        & (df_rules["longitude"] >= 72.0) & (df_rules["longitude"] <= 77.5)
        & (df_rules["month"].isin([6, 7, 8, 9]))
    )
    sw_new_upw = int(df_rules.loc[sw_mask, "upwelling"].sum())
    print(f"New 2022 SW Jun-Sep upwelling cell-days: {sw_new_upw:,}")

    # Step 9: Compare with existing MERGED_V3 and RULES_V3C if available
    existing_v3_file = EXISTING_V3_DIR / f"ocean_v3_{year}.parquet"
    existing_rules_file = EXISTING_RULES_DIR / f"rules_v3c_{year}.parquet"

    if existing_v3_file.exists() and existing_rules_file.exists():
        print("\n=================================================================")
        print(" COMPARISON WITH EXISTING MERGED_V3 & RULES_V3C ")
        print("=================================================================")
        df_old_v3 = pd.read_parquet(existing_v3_file)
        comp_results = compare_with_existing_v3(df_final, df_old_v3)

        df_old_rules = pd.read_parquet(existing_rules_file)
        df_old_rules["month"] = pd.DatetimeIndex(df_old_rules["time"]).month
        sw_mask_old = (
            (df_old_rules["latitude"] >= 8.0) & (df_old_rules["latitude"] <= 15.0)
            & (df_old_rules["longitude"] >= 72.0) & (df_old_rules["longitude"] <= 77.5)
            & (df_old_rules["month"].isin([6, 7, 8, 9]))
        )
        sw_old_upw = int(df_old_rules.loc[sw_mask_old, "upwelling"].sum())

        # Comparison Table
        print("\n--- Variable Comparison Over Valid Cells ---")
        rows = []
        for var, d in comp_results.items():
            rows.append({
                "Variable": var,
                "Valid Cells": f"{d['valid_cells']:,}",
                "Mean Abs Diff": f"{d['mean_abs_diff']:.4f}",
                "Max Abs Diff": f"{d['max_abs_diff']:.4f}",
                "Correlation": f"{d['correlation']:.4f}",
                "Rel Diff (%)": f"{d['rel_diff_pct']:.2f}%",
            })
        print(pd.DataFrame(rows).to_string(index=False))

        # PASS / FAIL
        pass_fail = evaluate_pass_fail(comp_results, sw_new_upw, sw_old_upw)
        print("\n--- PASS / FAIL Verification ---")
        pf_rows = []
        all_passed = True
        for criterion, res in pass_fail.items():
            status = "PASS" if res["pass"] else "FAIL"
            if not res["pass"]:
                all_passed = False
            pf_rows.append({
                "Criterion": criterion,
                "Measured Value": res["value"],
                "Threshold": res["threshold"],
                "Status": status,
            })
        print(pd.DataFrame(pf_rows).to_string(index=False))
        print(f"\nOVERALL RESULT: {'ALL PASS' if all_passed else 'FAIL'}")

    # Step 10: Size Estimates for 1993-2021 (29 years)
    print("\n=================================================================")
    print(" SIZE & TIME ESTIMATES FOR 1993-2021 (29 YEARS) ")
    print("=================================================================")
    n_years_hist = 29
    est_raw_gb = (total_raw_bytes / 1e9) * n_years_hist
    est_final_mb = (final_sz / 1e6) * n_years_hist
    est_final_gb = est_final_mb / 1024.0

    print(f"Measured 1-year raw download size : {total_raw_bytes / 1e6:.2f} MB ({total_raw_bytes / 1e9:.3f} GB)")
    print(f"Measured 1-year final parquet size: {final_sz / 1e6:.2f} MB")
    print(f"Estimated 1993-2021 (29 yrs) raw  : {est_raw_gb:.2f} GB")
    print(f"Estimated 1993-2021 (29 yrs) final: {est_final_mb:.1f} MB ({est_final_gb:.2f} GB)")

    print("\nTimings per step (2022):")
    for k, v in timings.items():
        print(f"  - {k:<15}: {v:.1f} s")
    print(f"  - total time     : {time.time()-t_total_start:.1f} s")


if __name__ == "__main__":
    main()
