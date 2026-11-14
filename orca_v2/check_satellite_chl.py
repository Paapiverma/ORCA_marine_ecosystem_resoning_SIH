"""
Satellite Chlorophyll Validation Check (2010–2022).

Downloads monthly satellite chlorophyll observations from CMEMS:
  Dataset: cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M
  Variable: CHL
  Domain: SW Coast box (8.0°N–15.0°N, 72.0°E–77.5°E)
  Period: June–September (JJAS), 2010–2022

Correlates yearly satellite chlorophyll with SW-box upwelling cell-days from year_summary.csv.
Outputs results to orca_v2/results/satellite_check.csv.
"""

from pathlib import Path
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from config import RAW_TEMP_DIR
except ImportError:
    from orca_v2.config import RAW_TEMP_DIR

DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"
VARIABLE = "CHL"
MIN_LAT, MAX_LAT = 8.0, 15.0
MIN_LON, MAX_LON = 72.0, 77.5
YEARS = list(range(2010, 2023))
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_CSV = RESULTS_DIR / "satellite_check.csv"
RAW_DIR = RAW_TEMP_DIR / "satellite_chl"


def download_satellite_chl(nc_path: Path) -> Path:
    """Download monthly satellite chlorophyll NetCDF using copernicusmarine CLI."""
    if nc_path.exists() and nc_path.stat().st_size > 1000:
        print(f"Using existing raw satellite file: {nc_path} ({nc_path.stat().st_size / 1e6:.2f} MB)")
        return nc_path

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "copernicusmarine", "subset",
        "-i", DATASET_ID,
        "-v", VARIABLE,
        "--start-datetime", "2010-06-01T00:00:00",
        "--end-datetime", "2022-09-30T23:59:59",
        "--minimum-longitude", str(MIN_LON),
        "--maximum-longitude", str(MAX_LON),
        "--minimum-latitude", str(MIN_LAT),
        "--maximum-latitude", str(MAX_LAT),
        "--output-directory", str(nc_path.parent),
        "--output-filename", nc_path.name,
        "--overwrite",
    ]
    print(f"Downloading {DATASET_ID} via copernicusmarine CLI...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"copernicusmarine download failed:\n{res.stderr}\n{res.stdout}")
    print(f"Download complete: {nc_path} ({nc_path.stat().st_size / 1e6:.2f} MB)")
    return nc_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nc_path = RAW_DIR / "satellite_chl_2010_2022_jjas.nc"
    download_satellite_chl(nc_path)

    # 1. Read NetCDF and extract Jun-Sep means per year
    with nc.Dataset(nc_path, "r") as ds:
        time_var = ds.variables["time"]
        units = time_var.units
        cal = getattr(time_var, "calendar", "standard")
        dates = nc.num2date(time_var[:], units, cal)
        chl_data = ds.variables["CHL"][:]

    sat_means = {}
    for y in YEARS:
        mask = [bool(d.year == y and d.month in [6, 7, 8, 9]) for d in dates]
        assert sum(mask) == 4, f"Expected 4 Jun-Sep months for {y}, found {sum(mask)}"
        chl_y = chl_data[mask, :, :]
        sat_means[y] = float(np.nanmean(chl_y))

    df_sat = pd.DataFrame(list(sat_means.items()), columns=["year", "satellite_chl_mean"])

    # 2. Read year_summary.csv
    summary_path = RESULTS_DIR / "year_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    df_summary = pd.read_csv(summary_path)
    merged = pd.merge(
        df_sat,
        df_summary[["year", "sw_upw_cell_days"]],
        on="year",
        how="inner",
    ).sort_values("year")

    # 3. Correlations
    p_corr, p_val = pearsonr(merged["sw_upw_cell_days"], merged["satellite_chl_mean"])
    s_corr, s_val = spearmanr(merged["sw_upw_cell_days"], merged["satellite_chl_mean"])

    # 4. Save results CSV
    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved 13-row validation table to: {OUT_CSV}")

    # 5. Print table and summary
    print("\n=== SATELLITE CHLOROPHYLL vs UPWELLING CELL-DAYS (SW Box: 8-15°N, 72-77.5°E, Jun-Sep) ===")
    print(merged.to_string(index=False))
    print("\nCorrelations (13 yearly pairs, 2010–2022):")
    print(f"  Pearson  r = {p_corr:.4f} (p = {p_val:.4e})")
    print(f"  Spearman r = {s_corr:.4f} (p = {s_val:.4e})")


if __name__ == "__main__":
    main()
