"""Apply upwelling and low-oxygen risk rules (v2) across 2022-2025. Run: python build_rules_v2.py"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from config import MERGED_V4_DIR as IN_DIR, RULES_V2_DIR as OUT_DIR
except ImportError:
    from orca_v2.config import MERGED_V4_DIR as IN_DIR, RULES_V2_DIR as OUT_DIR

YEARS = (2022, 2023, 2024, 2025)
N_CELLS = 18650
EARTH_RADIUS_KM = 6371.0
OFFSHORE_KM = 200.0
LAT_HALF_BAND = 0.5 + 1e-4  # include exact +/-6 steps (0.5 deg) despite float32 rounding
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
RULES = ("upwelling", "low_oxygen_risk")

REGIONS = (
    ("SW coast (8-15N, 72-77.5E)", 8.0, 15.0, 72.0, 77.5),
    ("NW (18-22N, 65-73E)", 18.0, 22.0, 65.0, 73.0),
    ("Central open sea (12-18N, 65-70E)", 12.0, 18.0, 65.0, 70.0),
)


def to_xyz_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat.astype(np.float64))
    lon_r = np.radians(lon.astype(np.float64))
    cos_lat = np.cos(lat_r)
    return np.column_stack([
        EARTH_RADIUS_KM * cos_lat * np.cos(lon_r),
        EARTH_RADIUS_KM * cos_lat * np.sin(lon_r),
        EARTH_RADIUS_KM * np.sin(lat_r),
    ])


def box_mask(lats: np.ndarray, lons: np.ndarray, lat0: float, lat1: float, lon0: float, lon1: float) -> np.ndarray:
    return (lats >= lat0) & (lats <= lat1) & (lons >= lon0) & (lons <= lon1)


def build_spatial_lookup(sample_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    df0 = pd.read_parquet(sample_path, columns=["time", "latitude", "longitude"])
    d0 = df0[df0["time"] == df0["time"].iloc[0]]
    ocean_coords = d0[["latitude", "longitude"]].to_numpy()
    assert len(ocean_coords) == N_CELLS

    lats_u = np.sort(d0["latitude"].unique())
    lons_u = np.sort(d0["longitude"].unique())
    full_grid = np.array([(la, lo) for la in lats_u for lo in lons_u], dtype=np.float32)
    ocean_set = set(map(tuple, ocean_coords))
    land_coords = np.array([pt for pt in map(tuple, full_grid) if pt not in ocean_set], dtype=np.float32)

    tree = cKDTree(to_xyz_km(land_coords[:, 0], land_coords[:, 1]))
    dist_to_land, _ = tree.query(to_xyz_km(ocean_coords[:, 0], ocean_coords[:, 1]))

    offshore = dist_to_land >= OFFSHORE_KM
    ocean_lats = ocean_coords[:, 0]
    lat_to_idx = np.searchsorted(lats_u, ocean_lats)
    assert np.array_equal(lats_u[lat_to_idx], ocean_lats)

    band_ref_indices = []
    for lat_val in lats_u:
        idx = np.flatnonzero(offshore & (np.abs(ocean_lats - lat_val) <= LAT_HALF_BAND))
        assert len(idx) > 0, f"No >=200 km reference cells near latitude {lat_val}"
        band_ref_indices.append(idx)

    return ocean_coords, dist_to_land, lat_to_idx, band_ref_indices


def compute_year_flags(
    df: pd.DataFrame,
    ocean_coords: np.ndarray,
    lat_to_idx: np.ndarray,
    band_ref_indices: list[np.ndarray],
) -> pd.DataFrame:
    n_days = len(df) // N_CELLS
    assert len(df) == n_days * N_CELLS
    coords = df[["latitude", "longitude"]].to_numpy().reshape(n_days, N_CELLS, 2)
    assert np.all(coords == ocean_coords[None, :, :]), "Grid mismatch inside year"

    thetao_2d = df["thetao"].to_numpy().reshape(n_days, N_CELLS)
    o2deep_2d = df["o2_50_150m"].to_numpy().reshape(n_days, N_CELLS)
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

    out = pd.DataFrame({
        "time": df["time"],
        "latitude": df["latitude"].astype(np.float32),
        "longitude": df["longitude"].astype(np.float32),
        "is_coastal": df["is_coastal"].astype(np.int8),
        "ref_thetao": ref_thetao,
        "upwelling": upwelling.astype(np.int8),
        "low_oxygen_risk": low_oxygen_risk.astype(np.int8),
    })

    for col in RULES:
        assert out[col].dtype == np.int8
        assert np.isin(out[col].to_numpy(), [0, 1]).all()
    assert np.all(out["low_oxygen_risk"].to_numpy() <= out["upwelling"].to_numpy())
    return out


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ocean_coords, dist_to_land, lat_to_idx, band_ref_indices = build_spatial_lookup(
        IN_DIR / f"ocean_v4_{YEARS[0]}.parquet"
    )
    ocean_lats = ocean_coords[:, 0]
    ocean_lons = ocean_coords[:, 1]
    region_masks = {
        name: box_mask(ocean_lats, ocean_lons, lat0, lat1, lon0, lon1)
        for name, lat0, lat1, lon0, lon1 in REGIONS
    }
    sw_name = REGIONS[0][0]
    cen_name = REGIONS[2][0]

    cols_needed = [
        "time", "latitude", "longitude", "is_coastal",
        "thetao", "no3", "chl", "o2_50_150m", "no3_filled", "chl_filled",
    ]

    year_stats = []
    month_counts = {m: {r: 0 for r in RULES} for m in range(1, 13)}
    month_totals = {m: 0 for m in range(1, 13)}
    region_counts = {name: {r: 0 for r in RULES} for name, *_ in REGIONS}
    region_totals = {name: 0 for name, *_ in REGIONS}
    sw_junsep_stats = []
    daily_frames = []
    aug4_df = None

    for year in YEARS:
        df_in = pd.read_parquet(IN_DIR / f"ocean_v4_{year}.parquet", columns=cols_needed)
        df_out = compute_year_flags(df_in, ocean_coords, lat_to_idx, band_ref_indices)
        out_path = OUT_DIR / f"rules_v2_{year}.parquet"
        df_out.to_parquet(out_path, compression="snappy", index=False)

        n_rows = len(df_out)
        n_days = n_rows // N_CELLS
        year_stats.append({
            "Period": str(year),
            "Rows": n_rows,
            "Size (MB)": out_path.stat().st_size / (1024 * 1024),
            **{r: int(df_out[r].sum()) for r in RULES},
        })

        months = pd.DatetimeIndex(df_out["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0]).month
        for m in range(1, 13):
            day_sel = months == m
            if not day_sel.any():
                continue
            row_sel = np.repeat(day_sel, N_CELLS)
            month_totals[m] += int(row_sel.sum())
            for r in RULES:
                month_counts[m][r] += int(df_out[r].to_numpy()[row_sel].sum())

        for name, m1d in region_masks.items():
            region_totals[name] += n_days * int(m1d.sum())
            for r in RULES:
                region_counts[name][r] += int(df_out[r].to_numpy().reshape(n_days, N_CELLS)[:, m1d].sum())

        # SW box Jun-Sep per year vs full-year SW box
        junsep_days = np.isin(months, [6, 7, 8, 9])
        sw_mask = region_masks[sw_name]
        up_2d = df_out["upwelling"].to_numpy().reshape(n_days, N_CELLS)[:, sw_mask]
        lo_2d = df_out["low_oxygen_risk"].to_numpy().reshape(n_days, N_CELLS)[:, sw_mask]
        sw_junsep_stats.append({
            "year": year,
            "upwelling_junsep": int(up_2d[junsep_days].sum()),
            "upwelling_fullyear": int(up_2d.sum()),
            "low_oxy_junsep": int(lo_2d[junsep_days].sum()),
            "low_oxy_fullyear": int(lo_2d.sum()),
        })

        # Daily upwelling geometry for Top 10
        day_dates = df_out["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0]
        up_full = df_out["upwelling"].to_numpy().reshape(n_days, N_CELLS)
        lo_full = df_out["low_oxygen_risk"].to_numpy().reshape(n_days, N_CELLS)
        coastal_1d = df_out["is_coastal"].to_numpy()[:N_CELLS]
        day_counts = up_full.sum(axis=1)
        for d_idx in np.flatnonzero(day_counts > 0):
            sel = up_full[d_idx] == 1
            cnt = int(day_counts[d_idx])
            daily_frames.append({
                "date": day_dates[d_idx],
                "flagged_cells": cnt,
                "low_oxygen_cells": int(lo_full[d_idx].sum()),
                "centroid_lat": float(ocean_lats[sel].mean()),
                "centroid_lon": float(ocean_lons[sel].mean()),
                "share_is_coastal": float(coastal_1d[sel].mean()),
                "mean_dist_km": float(dist_to_land[sel].mean()),
            })

        if year == 2022:
            aug4_df = df_out[df_out["time"] == pd.Timestamp("2022-08-04")].copy()

        print(f"Saved {out_path.name}: {n_rows:,} rows, {out_path.stat().st_size / (1024 * 1024):.2f} MB")

    tot_rows = sum(s["Rows"] for s in year_stats)
    overall_row = {
        "Period": "Overall (2022-2025)",
        "Rows": tot_rows,
        "Size (MB)": sum(s["Size (MB)"] for s in year_stats),
        **{r: sum(s[r] for s in year_stats) for r in RULES},
    }
    print("\n=== 1A. PERCENT FLAGGED PER YEAR & OVERALL ===")
    t1a = []
    for s in [*year_stats, overall_row]:
        t1a.append({
            "Period": s["Period"],
            "Cell-Days": f"{s['Rows']:,}",
            "upwelling": f"{s['upwelling'] / s['Rows'] * 100:.3f}% ({s['upwelling']:,})",
            "low_oxygen_risk": f"{s['low_oxygen_risk'] / s['Rows'] * 100:.3f}% ({s['low_oxygen_risk']:,})",
        })
    print(pd.DataFrame(t1a).to_string(index=False))

    print("\n=== 1B. PERCENT FLAGGED PER CALENDAR MONTH (2022-2025) ===")
    t1b = []
    for m in range(1, 13):
        tot_m = month_totals[m]
        t1b.append({
            "Month": MONTH_NAMES[m - 1],
            "Cell-Days": f"{tot_m:,}",
            "upwelling": f"{month_counts[m]['upwelling'] / tot_m * 100:.3f}% ({month_counts[m]['upwelling']:,})",
            "low_oxygen_risk": f"{month_counts[m]['low_oxygen_risk'] / tot_m * 100:.3f}% ({month_counts[m]['low_oxygen_risk']:,})",
        })
    print(pd.DataFrame(t1b).to_string(index=False))

    print("\n=== 1C. PERCENT FLAGGED PER BOX (2022-2025) ===")
    t1c = []
    for name, *_ in REGIONS:
        tot_r = region_totals[name]
        t1c.append({
            "Box": name,
            "Grid Cells": int(region_masks[name].sum()),
            "Cell-Days": f"{tot_r:,}",
            "upwelling": f"{region_counts[name]['upwelling'] / tot_r * 100:.4f}% ({region_counts[name]['upwelling']:,})",
            "low_oxygen_risk": f"{region_counts[name]['low_oxygen_risk'] / tot_r * 100:.4f}% ({region_counts[name]['low_oxygen_risk']:,})",
        })
    print(pd.DataFrame(t1c).to_string(index=False))

    print("\n=== 2. SW COAST BOX JUN-SEP PER YEAR ===")
    df_sw = pd.DataFrame(sw_junsep_stats)
    df_sw["junsep_share_of_year_upw_%"] = np.where(
        df_sw["upwelling_fullyear"] > 0,
        df_sw["upwelling_junsep"] / df_sw["upwelling_fullyear"] * 100.0,
        np.nan,
    )
    print(df_sw.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    sw_js_all = int(df_sw["upwelling_junsep"].sum())
    sw_tot_all = int(df_sw["upwelling_fullyear"].sum())
    sw_js_share = sw_js_all / sw_tot_all * 100.0 if sw_tot_all > 0 else 0.0
    print(f"SW total Jun-Sep upwelling / ALL SW upwelling (4 yrs): {sw_js_all:,} / {sw_tot_all:,} = {sw_js_share:.2f}%")

    print("\n=== 3. TOP 10 DAYS BY FLAGGED CELLS ===")
    top10 = pd.DataFrame(daily_frames).sort_values("flagged_cells", ascending=False).head(10).copy()
    top10["date"] = pd.to_datetime(top10["date"]).dt.strftime("%Y-%m-%d")
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 4. 2022-08-04 KERALA BOX VS OPEN-OCEAN BOX ===")
    k_mask = box_mask(aug4_df["latitude"].to_numpy(), aug4_df["longitude"].to_numpy(), 8.5, 13.0, 74.0, 77.0)
    o_mask = box_mask(aug4_df["latitude"].to_numpy(), aug4_df["longitude"].to_numpy(), 8.5, 13.0, 66.0, 70.0)
    for lbl, sub in [("Kerala Box (8.5-13N, 74-77E)", aug4_df[k_mask]), ("Open-Ocean Box (8.5-13N, 66-70E)", aug4_df[o_mask])]:
        print(f"{lbl}: cells={len(sub):,} | upwelling={int(sub['upwelling'].sum()):,} | low_oxygen_risk={int(sub['low_oxygen_risk'].sum()):,}")

    print("\n=== 5. PASS / FAIL EVALUATION ===")
    sw_2022 = int(df_sw.loc[df_sw["year"] == 2022, "upwelling_junsep"].iloc[0])
    sw_2024 = int(df_sw.loc[df_sw["year"] == 2024, "upwelling_junsep"].iloc[0])
    ratio_22_24 = (sw_2022 / sw_2024) if sw_2024 > 0 else float("inf")
    cen_pct = region_counts[cen_name]["upwelling"] / region_totals[cen_name] * 100.0
    top10_mean_dist = float(top10["mean_dist_km"].mean())
    top10_max_day_dist = float(top10["mean_dist_km"].max())

    checks = [
        ("a", "Jun-Sep holds >= 60% of SW-box flags", sw_js_share >= 60.0, f"{sw_js_share:.2f}%"),
        ("b", "SW Jun-Sep 2022 <= 10x SW Jun-Sep 2024", (sw_2024 > 0) and (sw_2022 <= 10 * sw_2024), f"2022={sw_2022:,}, 2024={sw_2024:,} (ratio={ratio_22_24:.2f}x)"),
        ("c", "Central open sea < 0.1%", cen_pct < 0.1, f"{cen_pct:.4f}% ({region_counts[cen_name]['upwelling']:,}/{region_totals[cen_name]:,})"),
        ("d", "Top-10 days mean distance to land < 150 km", top10_mean_dist < 150.0, f"mean={top10_mean_dist:.2f} km (max single top-10 day={top10_max_day_dist:.2f} km)"),
    ]
    for code, desc, ok, val in checks:
        print(f"({code}) {'PASS' if ok else 'FAIL'} | {desc} | {val}")

    print(f"\nCompleted in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
