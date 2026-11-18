"""Download raw Copernicus data, process ocean parquets, and compute rules_v3c for long records. Run: python run_long_years.py"""

import argparse
import datetime
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

EARTH_RADIUS_KM = 6371.0
COAST_KM = 20.0
MAX_FILL_KM = 25.0
OFFSHORE_KM = 200.0
LAT_HALF_BAND = 0.5 + 1e-4
LAT_CUTOFF = 19.5
N_CELLS = 18650

# Omits so and po4 which were dropped in v2.
FILL_VARS = ("chl", "o2", "no3")
OUT_FLOAT_COLS = ['latitude', 'longitude', 'thetao', 'uo', 'vo', 'chl', 'sla', 'o2', 'no3', 'o2_50_150m']
OUT_COLS = [
    'time', 'latitude', 'longitude',
    'thetao', 'uo', 'vo',
    'chl', 'sla', 'o2', 'no3', 'o2_50_150m',
    'is_coastal', 'chl_filled', 'o2_filled', 'no3_filled'
]

try:
    from config import (
        RAW_TEMP_DIR,
        ORCA_LONG_DIR as DEFAULT_OUT_DIR,
        LOGS_DIR as LOG_DIR,
    )
except ImportError:
    from orca_v2.config import (
        RAW_TEMP_DIR,
        ORCA_LONG_DIR as DEFAULT_OUT_DIR,
        LOGS_DIR as LOG_DIR,
    )


def setup_logger(log_file: Path, drive_log_file: Path | None = None):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_long_years")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if drive_log_file is not None:
        try:
            drive_log_file.parent.mkdir(parents=True, exist_ok=True)
            dfh = logging.FileHandler(str(drive_log_file), mode="a", encoding="utf-8")
            dfh.setLevel(logging.INFO)
            dfh.setFormatter(formatter)
            logger.addHandler(dfh)
        except Exception as e:
            ch.handle(logging.LogRecord("run_long_years", logging.WARNING, "", 0, f"Could not bind drive log {drive_log_file}: {e}", (), None))

    return logger


def check_copernicus_credentials():
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
    lat_r = np.radians(lat.astype(np.float64))
    lon_r = np.radians(lon.astype(np.float64))
    cos_lat = np.cos(lat_r)
    return np.column_stack([
        EARTH_RADIUS_KM * cos_lat * np.cos(lon_r),
        EARTH_RADIUS_KM * cos_lat * np.sin(lon_r),
        EARTH_RADIUS_KM * np.sin(lat_r),
    ])


def download_with_retry(name: str, download_func, max_retries: int, wait_seconds: float, logger: logging.Logger):
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{name}] Starting download (attempt {attempt}/{max_retries})...")
            t0 = time.time()
            res = download_func()
            elapsed = time.time() - t0
            file_path = Path(res.file_path)
            file_size_mb = file_path.stat().st_size / 1e6
            logger.info(f"[{name}] Download complete: {file_path.name} ({file_size_mb:.2f} MB in {elapsed:.1f}s)")
            return file_path, file_size_mb, elapsed
        except Exception as e:
            logger.warning(f"[{name}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info(f"[{name}] Waiting {wait_seconds}s before retrying...")
                time.sleep(wait_seconds)
            else:
                logger.error(f"[{name}] All {max_retries} attempts exhausted.")
                raise RuntimeError(f"[{name}] Download failed after {max_retries} attempts: {e}") from e


def download_year_raw(year: int, raw_dir: Path, max_retries: int, wait_seconds: float, logger: logging.Logger) -> dict:
    import copernicusmarine

    raw_dir.mkdir(parents=True, exist_ok=True)
    phys_target = raw_dir / f"physics_long_{year}.nc"
    bgc_target = raw_dir / f"bgc_long_{year}.nc"
    sla_target = raw_dir / f"sla_long_{year}.nc"

    start_date = f"{year}-01-01T00:00:00"
    end_date = f"{year}-12-31T23:59:59"

    download_results = {}

    def _dl_phys():
        return copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao", "uo", "vo"],
            minimum_longitude=65.0,
            maximum_longitude=78.0,
            minimum_latitude=8.0,
            maximum_latitude=22.0,
            minimum_depth=0.0,
            maximum_depth=50.0,
            start_datetime=start_date,
            end_datetime=end_date,
            output_filename=phys_target.name,
            output_directory=str(raw_dir),
            overwrite=True,
        )

    f_p, sz_p, t_p = download_with_retry("Physics", _dl_phys, max_retries, wait_seconds, logger)
    download_results["physics"] = {"file": f_p, "size_mb": sz_p, "time_s": t_p}

    def _dl_bgc():
        return copernicusmarine.subset(
            dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1D-m",
            variables=["chl", "no3", "o2"],
            minimum_longitude=65.0,
            maximum_longitude=78.0,
            minimum_latitude=8.0,
            maximum_latitude=22.0,
            minimum_depth=0.0,
            maximum_depth=200.0,
            start_datetime=start_date,
            end_datetime=end_date,
            output_filename=bgc_target.name,
            output_directory=str(raw_dir),
            overwrite=True,
        )

    f_b, sz_b, t_b = download_with_retry("BGC", _dl_bgc, max_retries, wait_seconds, logger)
    download_results["bgc"] = {"file": f_b, "size_mb": sz_b, "time_s": t_b}

    def _dl_sla():
        return copernicusmarine.subset(
            dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            variables=["sla"],
            minimum_longitude=65.0,
            maximum_longitude=78.0,
            minimum_latitude=8.0,
            maximum_latitude=22.0,
            start_datetime=start_date,
            end_datetime=end_date,
            output_filename=sla_target.name,
            output_directory=str(raw_dir),
            overwrite=True,
        )

    f_s, sz_s, t_s = download_with_retry("SLA", _dl_sla, max_retries, wait_seconds, logger)
    download_results["sla"] = {"file": f_s, "size_mb": sz_s, "time_s": t_s}

    return download_results


def process_and_merge(phys_file: Path, bgc_file: Path, sla_file: Path, logger: logging.Logger) -> pd.DataFrame:
    t0 = time.time()
    logger.info("Loading physics & computing 0-50m mean...")
    with xr.open_dataset(phys_file) as ds_p:
        target_lat = ds_p.latitude.values
        target_lon = ds_p.longitude.values
        p_surf = ds_p[['thetao', 'uo', 'vo']].sel(depth=slice(0, 50.0)).mean(dim='depth').load()
    land_mask = np.isnan(p_surf['thetao'].values)
    logger.info(f"Physics processed in {time.time()-t0:.1f}s. Grid: {len(target_lat)} lats x {len(target_lon)} lons.")

    t_step = time.time()
    logger.info("Loading & regridding surface BGC (chl, no3, o2 0-50m)...")
    with xr.open_dataset(bgc_file) as ds_b:
        b_surf = ds_b[['chl', 'no3', 'o2']].sel(depth=slice(0, 50.0)).mean(dim='depth').load()
        b_regrid = b_surf.interp(latitude=target_lat, longitude=target_lon, method='linear')
    for v in ['chl', 'no3', 'o2']:
        arr = b_regrid[v].values
        arr[land_mask] = np.nan
        b_regrid[v].values = arr
    logger.info(f"Surface BGC ready in {time.time()-t_step:.1f}s.")

    t_step = time.time()
    logger.info("Loading & regridding deep BGC oxygen (50-150m)...")
    with xr.open_dataset(bgc_file) as ds_b:
        d_sub = ds_b.sel(depth=slice(50.0, 150.0))
        d_mean = d_sub[['o2']].mean(dim='depth').load().rename({'o2': 'o2_50_150m'})
        d_regrid = d_mean.interp(latitude=target_lat, longitude=target_lon, method='linear')
    d_vals = d_regrid['o2_50_150m'].values
    d_vals[land_mask] = np.nan
    d_regrid['o2_50_150m'].values = d_vals
    logger.info(f"Deep BGC oxygen ready in {time.time()-t_step:.1f}s.")

    t_step = time.time()
    logger.info("Loading & regridding SSHA (sla)...")
    with xr.open_dataset(sla_file) as ds_s:
        s_regrid = ds_s[['sla']].interp(latitude=target_lat, longitude=target_lon, method='linear').load()
    s_vals = s_regrid['sla'].values
    s_vals[land_mask] = np.nan
    s_regrid['sla'].values = s_vals
    logger.info(f"SSHA ready in {time.time()-t_step:.1f}s.")

    # Align time coordinates to avoid float precision mismatch.
    time_coords = p_surf['time'].values
    b_regrid['time'] = time_coords
    d_regrid['time'] = time_coords
    s_regrid['time'] = time_coords

    t_step = time.time()
    logger.info("Merging arrays into single DataFrame...")
    combined_ds = xr.merge([p_surf, b_regrid, d_regrid, s_regrid])
    df = combined_ds.to_dataframe().reset_index()

    df[OUT_FLOAT_COLS] = df[OUT_FLOAT_COLS].astype(np.float32)
    cols_order = ['time', 'latitude', 'longitude'] + [c for c in OUT_FLOAT_COLS if c not in ['latitude', 'longitude']]
    df = df[cols_order]
    logger.info(f"Merged DataFrame formed in {time.time()-t_step:.1f}s ({len(df):,} total rows).")
    return df


def clean_land_and_coastal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    t_step = time.time()
    logger.info("Building coastal mask and dropping land cells (clean_merged_v2.py logic)...")
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

    ocean_mask = df["thetao"].notna().values
    df_clean = df.loc[ocean_mask].reset_index(drop=True)

    lat_idx = np.searchsorted(lats, df_clean["latitude"].values)
    lon_idx = np.searchsorted(lons, df_clean["longitude"].values)
    assert np.array_equal(lats[lat_idx], df_clean["latitude"].values)
    assert np.array_equal(lons[lon_idx], df_clean["longitude"].values)

    df_clean["is_coastal"] = grid[lat_idx, lon_idx]
    logger.info(f"Cleaned: {len(df_clean):,} ocean rows in {time.time()-t_step:.1f}s.")
    return df_clean


def resolve_day_fill_indices(lat_day: np.ndarray, lon_day: np.ndarray, nan_mask: np.ndarray, cache: dict):
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


def fill_nearest_neighbor(df_clean: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    t_step = time.time()
    logger.info("Performing nearest-neighbour 25km fill (chl, o2, no3)...")
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

    df_out = df_out[OUT_COLS]
    logger.info(f"Fill complete in {time.time()-t_step:.1f}s.")
    return df_out


def build_spatial_lookup(df_sample: pd.DataFrame, n_cells: int = N_CELLS):
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


def compute_rules_v3c(df: pd.DataFrame, logger: logging.Logger, n_cells: int = N_CELLS) -> pd.DataFrame:
    t_step = time.time()
    logger.info("Computing upwelling & low_oxygen_risk rules (lat < 19.5N cutoff)...")
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

    # Latitude cutoff: exclude Gulf of Khambhat false alarms.
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
    logger.info(f"Rules computed in {time.time()-t_step:.1f}s.")
    return df_rules


def append_year_summary(year: int, df_ocean: pd.DataFrame, df_rules: pd.DataFrame,
                        ocean_path: Path, rules_path: Path, seconds_taken: float,
                        out_dir: Path, logger: logging.Logger) -> None:
    csv_path = out_dir / "year_summary.csv"
    data_vars = [c for c in df_ocean.columns if c not in ['time', 'latitude', 'longitude', 'is_coastal'] and not c.endswith('_filled')]
    nan_pcts = {f'{c}_nan_pct': round(float(df_ocean[c].isna().mean() * 100.0), 3) for c in data_vars}

    months = pd.DatetimeIndex(df_ocean['time']).month
    sw_mask = (
        (df_ocean['latitude'] >= 8.0) & (df_ocean['latitude'] <= 15.0)
        & (df_ocean['longitude'] >= 72.0) & (df_ocean['longitude'] <= 77.5)
        & (months.isin([6, 7, 8, 9]))
    )
    sw_df = df_ocean.loc[sw_mask]
    mean_th = round(float(sw_df['thetao'].mean()), 4) if len(sw_df) > 0 else 0.0
    mean_chl = round(float(sw_df['chl'].mean()), 4) if len(sw_df) > 0 else 0.0
    mean_no3 = round(float(sw_df['no3'].mean()), 4) if len(sw_df) > 0 else 0.0

    rules_months = pd.DatetimeIndex(df_rules['time']).month
    sw_rules_mask = (
        (df_rules['latitude'] >= 8.0) & (df_rules['latitude'] <= 15.0)
        & (df_rules['longitude'] >= 72.0) & (df_rules['longitude'] <= 77.5)
        & (rules_months.isin([6, 7, 8, 9]))
    )
    sw_upw_days = int(df_rules.loc[sw_rules_mask, 'upwelling'].sum())

    row = {
        'year': year,
        'rows': len(df_ocean),
        **nan_pcts,
        'sw_mean_thetao': mean_th,
        'sw_mean_chl': mean_chl,
        'sw_mean_no3': mean_no3,
        'sw_upw_cell_days': sw_upw_days,
        'ocean_size_mb': round(ocean_path.stat().st_size / 1e6, 2),
        'rules_size_mb': round(rules_path.stat().st_size / 1e6, 2),
        'seconds_taken': round(seconds_taken, 1),
    }

    if csv_path.exists():
        try:
            df_curr = pd.read_csv(csv_path)
            # Remove any existing row for this year before appending to prevent duplicates
            df_curr = df_curr[df_curr['year'] != year]
            df_new = pd.concat([df_curr, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df_new = pd.DataFrame([row])
    else:
        df_new = pd.DataFrame([row])

    df_new.to_csv(csv_path, index=False)
    logger.info(f"[Year {year}] Appended summary row to {csv_path.name}")


def process_year(year: int, out_dir: Path, raw_dir: Path, max_retries: int, wait_seconds: float, logger: logging.Logger) -> str:
    ocean_final = out_dir / f"ocean_long_{year}.parquet"
    rules_final = out_dir / f"rules_v3c_{year}.parquet"

    if ocean_final.exists() and rules_final.exists():
        logger.info(f">>> [Year {year}] Already exists ({ocean_final.name} and {rules_final.name}). Skipping.")
        return "SKIPPED"

    logger.info(f"=================================================================")
    logger.info(f" PROCESSING YEAR: {year} ")
    logger.info(f"=================================================================")
    t_year_start = time.time()

    dl_stats = download_year_raw(year, raw_dir, max_retries, wait_seconds, logger)
    total_raw_mb = sum(s["size_mb"] for s in dl_stats.values())
    logger.info(f"[Year {year}] Total raw download size: {total_raw_mb:.2f} MB")

    phys_file = dl_stats["physics"]["file"]
    bgc_file = dl_stats["bgc"]["file"]
    sla_file = dl_stats["sla"]["file"]

    df_merged = process_and_merge(phys_file, bgc_file, sla_file, logger)

    df_clean = clean_land_and_coastal(df_merged, logger)

    df_final = fill_nearest_neighbor(df_clean, logger)

    ocean_tmp = out_dir / f".ocean_long_{year}.parquet.tmp"
    logger.info(f"Writing to temporary file {ocean_tmp.name}...")
    df_final.to_parquet(ocean_tmp, compression="snappy", index=False)
    if ocean_final.exists():
        ocean_final.unlink()
    ocean_tmp.rename(ocean_final)
    final_sz_mb = ocean_final.stat().st_size / 1e6
    logger.info(f"[Year {year}] Saved {ocean_final.name} ({final_sz_mb:.2f} MB, {len(df_final):,} rows)")

    df_rules = compute_rules_v3c(df_final, logger)
    rules_tmp = out_dir / f".rules_v3c_{year}.parquet.tmp"
    df_rules.to_parquet(rules_tmp, compression="snappy", index=False)
    if rules_final.exists():
        rules_final.unlink()
    rules_tmp.rename(rules_final)
    rules_sz_mb = rules_final.stat().st_size / 1e6
    logger.info(f"[Year {year}] Saved {rules_final.name} ({rules_sz_mb:.2f} MB)")

    df_rules["month"] = pd.DatetimeIndex(df_rules["time"]).month
    sw_mask = (
        (df_rules["latitude"] >= 8.0) & (df_rules["latitude"] <= 15.0)
        & (df_rules["longitude"] >= 72.0) & (df_rules["longitude"] <= 77.5)
        & (df_rules["month"].isin([6, 7, 8, 9]))
    )
    sw_upw_days = int(df_rules.loc[sw_mask, "upwelling"].sum())
    logger.info(f"[Year {year}] SW Jun-Sep Upwelling cell-days: {sw_upw_days:,}")

    logger.info(f"[Year {year}] Deleting raw NetCDF files...")
    for s in dl_stats.values():
        f = s["file"]
        if f.exists():
            f.unlink()
            logger.info(f"  Deleted: {f.name}")

    total_time = time.time() - t_year_start
    append_year_summary(year, df_final, df_rules, ocean_final, rules_final, total_time, out_dir, logger)

    logger.info(f">>> [Year {year}] Successfully completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    return "SUCCESS"


def main():
    parser = argparse.ArgumentParser(description="Process long-record Copernicus ocean dataset from 2021 down to earliest_year.")
    parser.add_argument("earliest_year", type=int, nargs="?", default=2005, help="Earliest year to process (default: 2005)")
    parser.add_argument("--earliest-year", type=int, dest="earliest_flag", default=None, help="Explicit earliest year flag")
    parser.add_argument("--start-year", type=int, default=2021, help="Year to start from descending (default: 2021)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory on Google Drive")
    parser.add_argument("--raw-dir", type=Path, default=RAW_TEMP_DIR, help="Temporary raw NetCDF download folder")
    parser.add_argument("--max-retries", type=int, default=3, help="Max download retries per product (default: 3)")
    parser.add_argument("--retry-wait", type=float, default=20.0, help="Seconds to wait between retries (default: 20.0)")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to log file")
    args = parser.parse_args()

    earliest_year = args.earliest_flag if args.earliest_flag is not None else args.earliest_year
    start_year = args.start_year
    out_dir = args.out_dir
    raw_dir = args.raw_dir
    max_retries = args.max_retries
    retry_wait = args.retry_wait

    log_file = args.log_file or (LOG_DIR / f"run_long_years_{start_year}_{earliest_year}.log")
    drive_log = out_dir / "run_log.txt"
    logger = setup_logger(log_file, drive_log_file=drive_log)

    logger.info("=================================================================")
    logger.info(f" BATCH PROCESSING: YEARS {start_year} DOWN TO {earliest_year} ")
    logger.info(f" Destination : {out_dir}")
    logger.info(f" Temp Raw Dir: {raw_dir}")
    logger.info(f" Max Retries : {max_retries} (wait {retry_wait}s)")
    logger.info(f" Log File    : {log_file}")
    logger.info("=================================================================")

    if not out_dir.parent.exists():
        logger.error(f"Google Drive parent directory not found: {out_dir.parent}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    logged_in, msg = check_copernicus_credentials()
    if not logged_in:
        logger.error(f"Credentials check failed: {msg}")
        logger.error("Please run `copernicusmarine login` in your terminal first.")
        sys.exit(1)
    logger.info("Copernicus Marine credentials verified.")

    results_summary = {}
    years_to_process = list(range(start_year, earliest_year - 1, -1))
    logger.info(f"Plan: {len(years_to_process)} years to evaluate: {years_to_process}")

    t_batch_start = time.time()
    for year in years_to_process:
        try:
            status = process_year(year, out_dir, raw_dir, max_retries, retry_wait, logger)
            results_summary[year] = status
        except Exception as e:
            logger.error(f">>> [Year {year}] FAILED with error: {e}")
            logger.error(traceback.format_exc())
            results_summary[year] = f"FAILED: {e}"
            # Clean up any leftover temporary files for this year
            tmp_ocean = out_dir / f".ocean_long_{year}.parquet.tmp"
            tmp_rules = out_dir / f".rules_v3c_{year}.parquet.tmp"
            if tmp_ocean.exists():
                tmp_ocean.unlink()
            if tmp_rules.exists():
                tmp_rules.unlink()
            logger.info(f">>> [Year {year}] Cleaned up temp files. Continuing to next year...")

    total_batch_time = time.time() - t_batch_start
    logger.info("\n=================================================================")
    logger.info(" BATCH EXECUTION SUMMARY ")
    logger.info("=================================================================")
    logger.info(f"Total Elapsed Time: {total_batch_time/3600:.2f} hours ({total_batch_time/60:.1f} minutes)")

    summary_df = pd.DataFrame([
        {"Year": y, "Status": s} for y, s in sorted(results_summary.items(), reverse=True)
    ])
    logger.info("\n" + summary_df.to_string(index=False))

    n_success = sum(1 for s in results_summary.values() if s == "SUCCESS")
    n_skipped = sum(1 for s in results_summary.values() if s == "SKIPPED")
    n_failed = sum(1 for s in results_summary.values() if str(s).startswith("FAILED"))
    logger.info(f"Totals: {n_success} Succeeded, {n_skipped} Skipped, {n_failed} Failed.")


if __name__ == "__main__":
    main()
