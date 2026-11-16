"""Compute 15-day rolling DOY climatology and anomalies for 2022-2025. Run: python build_merged_v4.py"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from config import MERGED_V3_DIR as IN_DIR, MERGED_V4_DIR as OUT_DIR
except ImportError:
    from orca_v2.config import MERGED_V3_DIR as IN_DIR, MERGED_V4_DIR as OUT_DIR

YEARS = (2022, 2023, 2024, 2025)
N_CELLS = 18650
N_DOY = 365
ROLL_HALF = 7  # 15-day centered circular window: [-7 .. +7]

ANOM_SPECS = (
    ("thetao", "thetao_anom", "degrees_C", "0-50m temperature"),
    ("chl", "chl_anom", "mg m-3", "0-50m chlorophyll-a"),
    ("no3", "no3_anom", "mmol m-3", "0-50m nitrate"),
    ("o2_50_150m", "o2deep_anom", "mmol m-3", "50-150m dissolved oxygen"),
)


def map_timestamps_to_doy365(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    # Non-Feb-29 dates in leap years shift back by 1 after Feb 29 so Mar 1 = idx 59 across all years.
    is_feb29 = np.asarray((dates.month == 2) & (dates.day == 29), dtype=bool)
    raw_doy = dates.dayofyear.to_numpy(dtype=np.int64)
    shift = (dates.is_leap_year & (raw_doy > 60)).astype(np.int64)
    doy_idx = raw_doy - shift - 1  # 0..364 for non-Feb-29
    return doy_idx, is_feb29


def compute_climatologies() -> tuple[dict[str, np.ndarray], np.ndarray]:
    src_cols = [s[0] for s in ANOM_SPECS]
    sums = {col: np.zeros((N_DOY, N_CELLS), dtype=np.float64) for col in src_cols}
    counts = {col: np.zeros((N_DOY, N_CELLS), dtype=np.int64) for col in src_cols}
    base_coords = None

    for year in YEARS:
        path = IN_DIR / f"ocean_v3_{year}.parquet"
        df = pd.read_parquet(path, columns=["time", "latitude", "longitude", *src_cols])
        n_days = len(df) // N_CELLS
        assert len(df) == n_days * N_CELLS, f"{year} row count not divisible by {N_CELLS}"

        coords = df[["latitude", "longitude"]].to_numpy().reshape(n_days, N_CELLS, 2)
        if base_coords is None:
            base_coords = coords[0].copy()
        assert np.all(coords == base_coords[None, :, :]), f"Grid mismatch in {year}"

        dates = pd.DatetimeIndex(df["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0])
        doy_idx, is_feb29 = map_timestamps_to_doy365(dates)
        keep = ~is_feb29

        for col in src_cols:
            arr = df[col].to_numpy(dtype=np.float64).reshape(n_days, N_CELLS)[keep]
            valid = ~np.isnan(arr)
            np.add.at(sums[col], doy_idx[keep], np.where(valid, arr, 0.0))
            np.add.at(counts[col], doy_idx[keep], valid.astype(np.int64))

    clims = {}
    for col in src_cols:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_clim = np.where(counts[col] > 0, sums[col] / counts[col], np.nan)

        # 15-day circular rolling average across day 365 <-> day 1.
        valid = ~np.isnan(raw_clim)
        val_zero = np.where(valid, raw_clim, 0.0)
        num = np.zeros_like(raw_clim)
        den = np.zeros_like(raw_clim)
        for offset in range(-ROLL_HALF, ROLL_HALF + 1):
            num += np.roll(val_zero, shift=-offset, axis=0)
            den += np.roll(valid, shift=-offset, axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            clims[col] = np.where(den > 0, num / den, np.nan)

    return clims, base_coords


def climatology_for_dates(clim: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    doy_idx, is_feb29 = map_timestamps_to_doy365(dates)
    out = clim[doy_idx].copy()
    if is_feb29.any():
        feb29_clim = 0.5 * (clim[58] + clim[59])  # day 59 (Feb 28) & day 60 (Mar 1)
        out[is_feb29] = feb29_clim
    return out


def write_units_txt(in_units: Path, out_units: Path) -> None:
    lines = in_units.read_text(encoding="utf-8").splitlines()
    if lines and "MERGED_V3" in lines[0]:
        lines[0] = lines[0].replace("MERGED_V3", "MERGED_V4")
    for src_col, anom_col, unit, desc in ANOM_SPECS:
        lines.append(
            f"{anom_col:<13}: {unit} ({desc} minus 2022-2025 15-day circular DOY climatology)"
        )
    out_units.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_units_txt(IN_DIR / "units.txt", OUT_DIR / "units.txt")

    print("Computing 2022-2025 15-day circular DOY climatologies...")
    clims, _ = compute_climatologies()

    anom_cols = [s[1] for s in ANOM_SPECS]
    global_sum = {c: 0.0 for c in anom_cols}
    global_sq = {c: 0.0 for c in anom_cols}
    global_n = {c: 0 for c in anom_cols}
    file_rows = []

    for year in YEARS:
        in_path = IN_DIR / f"ocean_v3_{year}.parquet"
        out_path = OUT_DIR / f"ocean_v4_{year}.parquet"
        df = pd.read_parquet(in_path)
        n_days = len(df) // N_CELLS
        dates = pd.DatetimeIndex(df["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0])

        for src_col, anom_col, _, _ in ANOM_SPECS:
            clim_year = climatology_for_dates(clims[src_col], dates).reshape(-1)
            vals = df[src_col].to_numpy(dtype=np.float64)
            anom = (vals - clim_year).astype(np.float32)

            expected_nan = np.isnan(vals) | np.isnan(clim_year)
            assert np.array_equal(np.isnan(anom), expected_nan), f"NaN mismatch in {anom_col} ({year})"
            assert anom.dtype == np.float32

            df[anom_col] = anom

            valid_vals = anom[~np.isnan(anom)].astype(np.float64)
            global_sum[anom_col] += float(valid_vals.sum())
            global_sq[anom_col] += float(np.square(valid_vals).sum())
            global_n[anom_col] += int(len(valid_vals))

        df.to_parquet(out_path, compression="snappy", index=False)
        file_rows.append({
            "Year": year,
            "File": out_path.name,
            "Rows": f"{len(df):,}",
            "Size (MB)": f"{out_path.stat().st_size / (1024 * 1024):.2f}",
        })
        print(f"Saved {out_path.name}: {len(df):,} rows, {out_path.stat().st_size / (1024 * 1024):.2f} MB")

    print("\n=== SCHEMA (ocean_v4_2022.parquet) ===")
    df_2022 = pd.read_parquet(OUT_DIR / "ocean_v4_2022.parquet")
    print(df_2022.dtypes.to_string())

    print("\n=== FILE SUMMARY ===")
    print(pd.DataFrame(file_rows).to_string(index=False))

    print("\n=== WHOLE-SET ANOMALY STATS (2022-2025) ===")
    whole_rows = []
    for col in anom_cols:
        n = global_n[col]
        mean = global_sum[col] / n
        var = (global_sq[col] - (global_sum[col] ** 2) / n) / (n - 1)
        std = float(np.sqrt(var))
        whole_rows.append({
            "Column": col,
            "Valid N": f"{n:,}",
            "Mean": f"{mean:+.6f}",
            "Std": f"{std:.6f}",
        })
    print(pd.DataFrame(whole_rows).to_string(index=False))

    aug4 = df_2022[df_2022["time"] == pd.Timestamp("2022-08-04")]
    kerala_mask = (
        (aug4["latitude"] >= 8.5) & (aug4["latitude"] <= 13.0)
        & (aug4["longitude"] >= 74.0) & (aug4["longitude"] <= 77.0)
    )
    open_mask = (
        (aug4["latitude"] >= 8.5) & (aug4["latitude"] <= 13.0)
        & (aug4["longitude"] >= 66.0) & (aug4["longitude"] <= 70.0)
    )

    for label, sub in [
        ("KERALA BOX 2022-08-04 (8.5-13N, 74-77E)", aug4[kerala_mask]),
        ("OPEN-OCEAN BOX 2022-08-04 (8.5-13N, 66-70E)", aug4[open_mask]),
    ]:
        print(f"\n=== {label} ===")
        stats_df = pd.DataFrame({
            "Valid N": sub[anom_cols].notna().sum(),
            "Mean": sub[anom_cols].mean(),
            "Std": sub[anom_cols].std(),
        })
        print(stats_df.to_string(float_format=lambda x: f"{x:+.6f}"))

    print("\n=== KERALA BOX DAILY thetao_anom (2022-07-25 to 2022-08-15) ===")
    win = df_2022[
        (df_2022["time"] >= pd.Timestamp("2022-07-25"))
        & (df_2022["time"] <= pd.Timestamp("2022-08-15"))
        & (df_2022["latitude"] >= 8.5) & (df_2022["latitude"] <= 13.0)
        & (df_2022["longitude"] >= 74.0) & (df_2022["longitude"] <= 77.0)
    ]
    daily = win.groupby("time")[["thetao", "thetao_anom"]].mean()
    daily.index = daily.index.strftime("%Y-%m-%d")
    print(daily.to_string(float_format=lambda x: f"{x:+.4f}"))

    print(f"\nCompleted in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
