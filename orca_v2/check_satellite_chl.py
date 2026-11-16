"""Validate model upwelling against satellite chlorophyll observations (2010-2022). Run: python check_satellite_chl.py"""

from pathlib import Path
import shutil
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
    if nc_path.exists() and nc_path.stat().st_size > 1000:
        print(f"Using existing raw satellite file: {nc_path} ({nc_path.stat().st_size / 1e6:.2f} MB)")
        return nc_path

    # Fallback cached file location.
    fallback_path = Path("C:/ORCA_TEMP/sat_chl_check.nc")
    if fallback_path.exists() and fallback_path.stat().st_size > 1000:
        print(f"Reusing existing check file from {fallback_path} -> {nc_path}")
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback_path, nc_path)
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

    # Extract Jun-Sep means under equal-month and pooled-pixel methods.
    with nc.Dataset(nc_path, "r") as ds:
        time_var = ds.variables["time"]
        units = time_var.units
        cal = getattr(time_var, "calendar", "standard")
        dates = nc.num2date(time_var[:], units, cal)
        chl_data = ds.variables["CHL"][:]

    rows = []
    for y in YEARS:
        m_box_means = []
        y_valid_pixels = []
        for m in [6, 7, 8, 9]:
            matching = [i for i, d in enumerate(dates) if d.year == y and d.month == m]
            assert len(matching) == 1, f"Expected 1 slice for {y}-{m}, found {len(matching)}"
            idx_m = matching[0]
            data_m = chl_data[idx_m, :, :]
            if hasattr(data_m, "filled"):
                data_m = data_m.filled(np.nan)
            valid = data_m[(~np.isnan(data_m)) & (data_m != -999.0)]
            m_box_means.append(float(np.mean(valid)))
            y_valid_pixels.extend(valid.tolist())

        mean_equal_month = float(np.mean(m_box_means))
        mean_pooled_pixels = float(np.mean(y_valid_pixels))

        rows.append({
            "year": y,
            "satellite_chl_equal_month": mean_equal_month,
            "satellite_chl_pooled_pixels": mean_pooled_pixels,
        })

    df_sat = pd.DataFrame(rows)

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

    # Equal weight per month (headline).
    p_corr_a, p_val_a = pearsonr(merged["sw_upw_cell_days"], merged["satellite_chl_equal_month"])
    s_corr_a, s_val_a = spearmanr(merged["sw_upw_cell_days"], merged["satellite_chl_equal_month"])

    # Pooled valid pixels (sensitivity).
    p_corr_b, p_val_b = pearsonr(merged["sw_upw_cell_days"], merged["satellite_chl_pooled_pixels"])
    s_corr_b, s_val_b = spearmanr(merged["sw_upw_cell_days"], merged["satellite_chl_pooled_pixels"])

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved 13-row validation table to: {OUT_CSV}")

    print("\n=== SATELLITE CHLOROPHYLL vs UPWELLING CELL-DAYS (SW Box: 8-15°N, 72-77.5°E, Jun-Sep) ===")
    print(merged.to_string(index=False))
    print("\nCorrelations (13 yearly pairs, 2010–2022):")
    print(f"  Headline (Equal weight per month):  Pearson r = {p_corr_a:.4f} (p = {p_val_a:.4e}), Spearman r = {s_corr_a:.4f} (p = {s_val_a:.4e})")
    print(f"  Sensitivity (Pooled valid pixels):  Pearson r = {p_corr_b:.4f} (p = {p_val_b:.4e}), Spearman r = {s_corr_b:.4f} (p = {s_val_b:.4e})")


if __name__ == "__main__":
    main()
