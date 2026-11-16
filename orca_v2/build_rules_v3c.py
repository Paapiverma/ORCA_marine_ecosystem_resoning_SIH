"""Apply latitude < 19.5N cutoff to rules_v2 parquets. Run: python build_rules_v3c.py"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAND_COLOR = "#bdbdbd"
ASPECT = 1.0 / np.cos(np.radians(15.0))

def setup_grid(sample_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    d0 = sample_df[sample_df["time"] == sample_df["time"].iloc[0]]
    lats_u = np.sort(d0["latitude"].unique())
    lons_u = np.sort(d0["longitude"].unique())
    lat_idx = np.searchsorted(lats_u, d0["latitude"].to_numpy())
    lon_idx = np.searchsorted(lons_u, d0["longitude"].to_numpy())
    dlat = float(lats_u[1] - lats_u[0]) / 2.0
    dlon = float(lons_u[1] - lons_u[0]) / 2.0
    extent = [
        float(lons_u[0]) - dlon,
        float(lons_u[-1]) + dlon,
        float(lats_u[0]) - dlat,
        float(lats_u[-1]) + dlat,
    ]
    return lats_u, lons_u, lat_idx, lon_idx, extent

def to_grid(values: np.ndarray, n_lat: int, n_lon: int, lat_idx: np.ndarray, lon_idx: np.ndarray, fill=np.nan) -> np.ndarray:
    grid = np.full((n_lat, n_lon), fill, dtype=np.float64)
    grid[lat_idx, lon_idx] = values
    return grid

def style_map_axes(ax: plt.Axes, title: str, xlabel: bool = True, ylabel: bool = True) -> None:
    ax.set_xlim(65.0, 78.0)
    ax.set_ylim(8.0, 22.0)
    ax.set_aspect(ASPECT)
    ax.set_xticks([66, 68, 70, 72, 74, 76, 78])
    ax.set_yticks([8, 10, 12, 14, 16, 18, 20, 22])
    ax.tick_params(labelsize=9)
    if xlabel:
        ax.set_xlabel("Longitude (°E)", fontsize=10)
    if ylabel:
        ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

def make_freq_cmap():
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color=LAND_COLOR)
    return cmap

def plot_single_freq_map(grid: np.ndarray, extent: list[float], title: str, out_path: Path, cbar_label: str) -> Path:
    cmap = make_freq_cmap()
    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    style_map_axes(ax, title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


try:
    from config import RULES_V2_DIR as IN_DIR, RULES_V3C_DIR as OUT_DIR, FIGURES_DIR as FIG_DIR
except ImportError:
    from orca_v2.config import RULES_V2_DIR as IN_DIR, RULES_V3C_DIR as OUT_DIR, FIGURES_DIR as FIG_DIR

LAT_CUTOFF = 19.5
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
OUT_COLS = [
    "time", "latitude", "longitude", "is_coastal",
    "excluded_north", "ref_thetao", "upwelling", "low_oxygen_risk",
]


def plot_junsep_2x2_v3c(
    grids_by_year: dict[int, np.ndarray],
    extent: list[float],
    out_path: Path,
) -> Path:
    cmap = make_freq_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.2), constrained_layout=True, sharex=True, sharey=True)
    last_im = None
    for idx, (ax, year) in enumerate(zip(axes.flat, YEARS)):
        last_im = ax.imshow(
            grids_by_year[year],
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        r, c = divmod(idx, 2)
        style_map_axes(ax, f"Jun–Sep {year}", xlabel=(r == 1), ylabel=(c == 0))

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of Jun–Sep Days Flagged Upwelling", fontsize=11)
    fig.suptitle("RULES_V3C (lat < 19.5°N) Jun–Sep Upwelling Frequency (2022–2025)", fontsize=13)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df_2022 = pd.read_parquet(IN_DIR / "rules_v2_2022.parquet")
    d0 = df_2022[df_2022["time"] == df_2022["time"].iloc[0]]
    ocean_lats = d0["latitude"].to_numpy()
    ocean_lons = d0["longitude"].to_numpy()
    coastal_1d = d0["is_coastal"].to_numpy()

    excl_1d = ocean_lats >= LAT_CUTOFF
    sw_mask = (ocean_lats >= 8.0) & (ocean_lats <= 15.0) & (ocean_lons >= 72.0) & (ocean_lons <= 77.5)
    assert not np.any(excl_1d & sw_mask), "SW box must lie strictly south of 19.5N"

    lats_u, lons_u, lat_idx, lon_idx, extent = setup_grid(df_2022)
    n_lat, n_lon = len(lats_u), len(lons_u)

    djf_orig = 0
    djf_new = 0
    djf_days_total = 0
    djf_per_cell_orig = np.zeros(N_CELLS, dtype=np.int64)
    djf_per_cell_new = np.zeros(N_CELLS, dtype=np.int64)

    oct_sum_new = np.zeros(N_CELLS, dtype=np.int64)
    oct_days_total = 0
    junsep_grids_new: dict[int, np.ndarray] = {}

    sw_js_orig: dict[int, int] = {}
    sw_js_new: dict[int, int] = {}

    # Track monthly totals (1..12) across 2022-2025
    m_upw_orig = np.zeros(13, dtype=np.int64)
    m_upw_new = np.zeros(13, dtype=np.int64)
    m_lo_orig = np.zeros(13, dtype=np.int64)
    m_lo_new = np.zeros(13, dtype=np.int64)
    m_days = np.zeros(13, dtype=np.int64)

    for year in YEARS:
        df_in = df_2022 if year == 2022 else pd.read_parquet(IN_DIR / f"rules_v2_{year}.parquet")
        n_days = len(df_in) // N_CELLS
        assert len(df_in) == n_days * N_CELLS

        coords = df_in[["latitude", "longitude"]].to_numpy().reshape(n_days, N_CELLS, 2)
        assert np.all(coords[:, :, 0] == ocean_lats[None, :]) and np.all(coords[:, :, 1] == ocean_lons[None, :])

        upw_orig = df_in["upwelling"].to_numpy().reshape(n_days, N_CELLS)
        lo_orig = df_in["low_oxygen_risk"].to_numpy().reshape(n_days, N_CELLS)

        excl_col = np.tile(excl_1d.astype(np.int8), n_days)
        keep_2d = (~excl_1d)[None, :]
        upw_new = np.where(keep_2d, upw_orig, 0).astype(np.int8)
        lo_new = np.where(keep_2d, lo_orig, 0).astype(np.int8)

        # Self-check invariants on modified rows and untouched rows
        assert np.all(upw_new[:, excl_1d] == 0) and np.all(lo_new[:, excl_1d] == 0)
        assert np.array_equal(upw_new[:, ~excl_1d], upw_orig[:, ~excl_1d])
        assert np.array_equal(lo_new[:, ~excl_1d], lo_orig[:, ~excl_1d])

        df_out = pd.DataFrame(
            {
                "time": df_in["time"],
                "latitude": df_in["latitude"].astype(np.float32),
                "longitude": df_in["longitude"].astype(np.float32),
                "is_coastal": df_in["is_coastal"].astype(np.int8),
                "excluded_north": excl_col,
                "ref_thetao": df_in["ref_thetao"].astype(np.float32),
                "upwelling": upw_new.reshape(-1),
                "low_oxygen_risk": lo_new.reshape(-1),
            },
            columns=OUT_COLS,
        )
        df_out.to_parquet(OUT_DIR / f"rules_v3c_{year}.parquet", compression="snappy", index=False)

        dates = pd.DatetimeIndex(df_in["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0])
        months = dates.month

        djf_mask = np.isin(months, [12, 1, 2])
        djf_days_total += int(djf_mask.sum())
        djf_orig += int(upw_orig[djf_mask].sum())
        djf_new += int(upw_new[djf_mask].sum())
        djf_per_cell_orig += upw_orig[djf_mask].sum(axis=0)
        djf_per_cell_new += upw_new[djf_mask].sum(axis=0)

        js_mask = np.isin(months, [6, 7, 8, 9])
        sw_js_orig[year] = int(upw_orig[js_mask][:, sw_mask].sum())
        sw_js_new[year] = int(upw_new[js_mask][:, sw_mask].sum())
        assert sw_js_orig[year] == sw_js_new[year], f"SW Jun-Sep mismatch in {year}"

        junsep_grids_new[year] = to_grid(upw_new[js_mask].mean(axis=0), n_lat, n_lon, lat_idx, lon_idx)

        oct_mask = months == 10
        oct_sum_new += upw_new[oct_mask].sum(axis=0)
        oct_days_total += int(oct_mask.sum())

        for m in range(1, 13):
            mm = months == m
            m_days[m] += int(mm.sum())
            m_upw_orig[m] += int(upw_orig[mm].sum())
            m_upw_new[m] += int(upw_new[mm].sum())
            m_lo_orig[m] += int(lo_orig[mm].sum())
            m_lo_new[m] += int(lo_new[mm].sum())

    print(
        f"Excluded grid cells (lat >= {LAT_CUTOFF}): {int(excl_1d.sum()):,} / {N_CELLS:,} "
        f"({excl_1d.mean() * 100:.2f}%)"
    )

    print("\n=== 2A. DEC-FEB UPWELLING CELL-DAYS & TOP 10 STILL-FLAGGED CELLS ===")
    djf_domain = djf_days_total * N_CELLS
    djf_drop_pct = (djf_orig - djf_new) / djf_orig * 100.0
    print(f"Original (RULES_V2): {djf_orig:>6,} / {djf_domain:,} ({djf_orig / djf_domain * 100:.4f}%)")
    print(
        f"New (RULES_V3C)    : {djf_new:>6,} / {djf_domain:,} ({djf_new / djf_domain * 100:.4f}%) | "
        f"change: {-djf_drop_pct:+.2f}%"
    )
    print(f"Total ocean cells still flagged >=1 day in Dec-Feb: {int((djf_per_cell_new > 0).sum())}")

    cells_df = pd.DataFrame(
        {
            "lat": ocean_lats,
            "lon": ocean_lons,
            "is_coastal": coastal_1d,
            "djf_days_orig": djf_per_cell_orig,
            "djf_days_new": djf_per_cell_new,
        }
    )
    print("\n-- Top 10 Still-Flagged Dec-Feb Cells (RULES_V3C, lat < 19.5) --")
    top_new = cells_df[cells_df["djf_days_new"] > 0].sort_values("djf_days_new", ascending=False).head(10)
    print(top_new[["lat", "lon", "is_coastal", "djf_days_new"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 2B. SW BOX (8-15N, 72-77.5E) JUN-SEP UPWELLING CELL-DAYS ===")
    sw_rows = [{"Year": str(y), "Original": sw_js_orig[y], "RULES_V3C": sw_js_new[y]} for y in YEARS]
    sw_rows.append({"Year": "Total", "Original": sum(sw_js_orig.values()), "RULES_V3C": sum(sw_js_new.values())})
    print(pd.DataFrame(sw_rows).to_string(index=False))

    print("\n=== 2C & 2D. MONTHLY FLAGGED CELL-DAYS (ORIGINAL VS REMOVED VS RULES_V3C) ===")
    m_rows = []
    for m in range(1, 13):
        rem_upw = int(m_upw_orig[m] - m_upw_new[m])
        rem_lo = int(m_lo_orig[m] - m_lo_new[m])
        m_rows.append(
            {
                "Month": MONTH_NAMES[m - 1],
                "Orig_Upw": int(m_upw_orig[m]),
                "Removed_Upw": rem_upw,
                "New_Upw": int(m_upw_new[m]),
                "New_Upw_%": m_upw_new[m] / (m_days[m] * N_CELLS) * 100.0,
                "Orig_LowO2": int(m_lo_orig[m]),
                "Removed_LowO2": rem_lo,
                "New_LowO2": int(m_lo_new[m]),
            }
        )
    m_rows.append(
        {
            "Month": "Total",
            "Orig_Upw": int(m_upw_orig[1:].sum()),
            "Removed_Upw": int((m_upw_orig[1:] - m_upw_new[1:]).sum()),
            "New_Upw": int(m_upw_new[1:].sum()),
            "New_Upw_%": m_upw_new[1:].sum() / (m_days[1:].sum() * N_CELLS) * 100.0,
            "Orig_LowO2": int(m_lo_orig[1:].sum()),
            "Removed_LowO2": int((m_lo_orig[1:] - m_lo_new[1:]).sum()),
            "New_LowO2": int(m_lo_new[1:].sum()),
        }
    )
    print(pd.DataFrame(m_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    js_months = [6, 7, 8, 9]
    js_upw_orig_tot = int(m_upw_orig[js_months].sum())
    js_upw_rem_tot = int((m_upw_orig[js_months] - m_upw_new[js_months]).sum())
    js_upw_rem_share = js_upw_rem_tot / js_upw_orig_tot * 100.0
    js_lo_orig_tot = int(m_lo_orig[js_months].sum())
    js_lo_rem_tot = int((m_lo_orig[js_months] - m_lo_new[js_months]).sum())
    print(
        f"\nJun-Sep domain upwelling flags: orig={js_upw_orig_tot:,}, removed={js_upw_rem_tot:,} "
        f"({js_upw_rem_share:.4f}% of domain Jun-Sep upwelling flags)"
    )
    print(f"Jun-Sep domain low_oxygen_risk flags: orig={js_lo_orig_tot:,}, removed={js_lo_rem_tot:,}")

    print("\n=== 3. PASS / FAIL EVALUATION ===")
    pass_a = djf_drop_pct >= 90.0
    pass_b = sw_js_orig == sw_js_new
    pass_c = js_upw_rem_share < 2.0
    print(f"(a) Dec-Feb flags fall by >= 90%           : {'PASS' if pass_a else 'FAIL'} ({djf_drop_pct:.2f}%)")
    print(f"(b) SW box Jun-Sep unchanged               : {'PASS' if pass_b else 'FAIL'}")
    print(f"(c) Removed Jun-Sep flags < 2% of domain   : {'PASS' if pass_c else 'FAIL'} ({js_upw_rem_share:.4f}%)")

    djf_grid = to_grid(djf_per_cell_new / djf_days_total, n_lat, n_lon, lat_idx, lon_idx)
    oct_grid = to_grid(oct_sum_new / oct_days_total, n_lat, n_lon, lat_idx, lon_idx)
    saved = [
        plot_single_freq_map(
            djf_grid,
            extent,
            f"Upwelling Frequency: Dec–Feb 2022–2025 ({djf_days_total} days)\n(RULES_V3C: lat < 19.5°N)",
            FIG_DIR / "freq_Dec-Feb.png",
            "Fraction of Dec–Feb Days Flagged",
        ),
        plot_single_freq_map(
            oct_grid,
            extent,
            f"Upwelling Frequency: Oct 2022–2025 ({oct_days_total} days)\n(RULES_V3C: lat < 19.5°N)",
            FIG_DIR / "freq_Oct.png",
            "Fraction of Oct Days Flagged",
        ),
        plot_junsep_2x2_v3c(junsep_grids_new, extent, FIG_DIR / "freq_JunSep_2x2.png"),
    ]
    print("\nSaved figures:")
    for f in saved:
        print(f"  {f} ({f.stat().st_size / 1024:.1f} KB)")

    print(f"\nCompleted in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
