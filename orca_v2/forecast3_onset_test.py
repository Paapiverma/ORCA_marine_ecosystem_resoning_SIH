"""Evaluate gradient boosted trees and baselines for onset forecasting (h=1,3,5). Run: python forecast3_onset_test.py"""

import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "forecast3_run_log.txt"
try:
    from config import ORCA_LONG_DIR as LONG_DIR, MERGED_V3_DIR as M3_DIR, RULES_V3C_DIR as R3C_DIR
except ImportError:
    from orca_v2.config import ORCA_LONG_DIR as LONG_DIR, MERGED_V3_DIR as M3_DIR, RULES_V3C_DIR as R3C_DIR

N_CELLS = 18650
YEARS = list(range(2010, 2026))
HORIZONS = (1, 3, 5)

OCEAN_COLS = ["thetao", "no3", "chl", "sla", "o2_50_150m", "uo", "vo"]
RULES_COLS = ["time", "latitude", "longitude", "is_coastal", "excluded_north", "ref_thetao", "upwelling"]

FEATURE_NAMES = [
    "thetao_minus_ref",
    "no3",
    "chl",
    "sla",
    "o2_50_150m",
    "uo",
    "vo",
    "d3_thetao",
    "d3_no3",
    "d3_chl",
    "upw_d1",
    "upw_d2",
    "upw_d3",
    "days_since_flagged",
    "w5_upw_frac",
    "d3_w5_upw_frac",
    "w5_thetao_minus_ref",
    "latitude",
    "longitude",
    "is_coastal",
    "sin_doy",
    "cos_doy",
]


def log_progress(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def map_timestamps_to_doy365(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    is_feb29 = np.asarray((dates.month == 2) & (dates.day == 29), dtype=bool)
    raw_doy = dates.dayofyear.to_numpy(dtype=np.int64)
    shift = (dates.is_leap_year & (raw_doy > 60)).astype(np.int64)
    doy_idx = raw_doy - shift - 1
    return doy_idx, is_feb29


def load_data() -> dict:
    log_progress("START: loading data (2010-2025 continuous)")
    t0 = time.time()

    d0 = pd.read_parquet(
        LONG_DIR / "rules_v3c_2022.parquet",
        columns=["time", "latitude", "longitude", "is_coastal", "excluded_north"],
    )
    d0 = d0[d0["time"] == d0["time"].iloc[0]]
    assert len(d0) == N_CELLS, f"Expected {N_CELLS} cells, got {len(d0)}"

    lats_all = d0["latitude"].to_numpy(dtype=np.float32)
    lons_all = d0["longitude"].to_numpy(dtype=np.float32)
    keep_lat = lats_all < 19.5
    assert np.array_equal(keep_lat, d0["excluded_north"].to_numpy() == 0)
    n_sub = int(keep_lat.sum())

    lat_vec = lats_all[keep_lat]
    lon_vec = lons_all[keep_lat]
    coastal_vec = d0["is_coastal"].to_numpy(dtype=np.float32)[keep_lat]

    unique_lats = np.sort(np.unique(lats_all))
    unique_lons = np.sort(np.unique(lons_all))
    lat_to_r = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_c = {lon: j for j, lon in enumerate(unique_lons)}

    r_all = np.array([lat_to_r[lat] for lat in lats_all], dtype=np.int32)
    c_all = np.array([lon_to_c[lon] for lon in lons_all], dtype=np.int32)
    r_sub = r_all[keep_lat]
    c_sub = c_all[keep_lat]

    n_r = len(unique_lats)
    n_c = len(unique_lons)

    ocean_mask = np.zeros((n_r, n_c), dtype=np.float32)
    ocean_mask[r_all, c_all] = 1.0

    # Number of valid ocean cells in each 5x5 window.
    ocean_count_5x5 = scipy.ndimage.uniform_filter(ocean_mask, size=5, mode="constant", cval=0.0)
    ocean_count_sub = ocean_count_5x5[r_sub, c_sub]
    ocean_count_sub = np.maximum(ocean_count_sub, 1e-6)

    dates_list = []
    upw_list = []
    upw_all_list = []
    th_all_list = []
    ref_all_list = []
    arrays_lists = {col: [] for col in ("thetao", "thetao_minus_ref", "no3", "chl", "sla", "o2_50_150m", "uo", "vo")}

    for year in YEARS:
        t_yr0 = time.time()
        if year <= 2022:
            ocean_path = LONG_DIR / f"ocean_long_{year}.parquet"
            rules_path = LONG_DIR / f"rules_v3c_{year}.parquet"
        else:
            ocean_path = M3_DIR / f"ocean_v3_{year}.parquet"
            rules_path = R3C_DIR / f"rules_v3c_{year}.parquet"

        df_o = pd.read_parquet(ocean_path, columns=["time", *OCEAN_COLS])
        df_r = pd.read_parquet(rules_path, columns=RULES_COLS)

        n_days = len(df_r) // N_CELLS
        assert len(df_o) == len(df_r) == n_days * N_CELLS, f"{year} row mismatch"
        assert np.array_equal(df_o["time"].to_numpy(), df_r["time"].to_numpy()), f"{year} time alignment mismatch"

        t_year = pd.DatetimeIndex(df_r["time"].to_numpy().reshape(n_days, N_CELLS)[:, 0])
        dates_list.append(t_year)

        th_full = df_o["thetao"].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)
        ref_full = df_r["ref_thetao"].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)
        upw_full = df_r["upwelling"].to_numpy(dtype=np.int8).reshape(n_days, N_CELLS)

        upw_all_list.append(upw_full)
        th_all_list.append(th_full)
        ref_all_list.append(ref_full)

        th_2d = th_full[:, keep_lat]
        ref_2d = ref_full[:, keep_lat]

        arrays_lists["thetao"].append(th_2d)
        arrays_lists["thetao_minus_ref"].append((th_2d - ref_2d).astype(np.float32))

        for col in ("no3", "chl", "sla", "o2_50_150m", "uo", "vo"):
            arrays_lists[col].append(df_o[col].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)[:, keep_lat])

        u_2d = upw_full[:, keep_lat]
        upw_list.append(u_2d)

        del df_o, df_r
        gc.collect()
        log_progress(f"Loaded year {year} ({n_days} days) in {time.time() - t_yr0:.1f}s")

    dates = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))
    assert np.all(np.diff(dates.asi8) == 86_400_000_000_000), "Expected contiguous daily grid"
    n_days_total = len(dates)

    doy_idx, is_feb29 = map_timestamps_to_doy365(dates)
    doy_cont = np.where(is_feb29, 59.5, doy_idx.astype(np.float64) + 1.0)
    angles = 2.0 * np.pi * doy_cont / 365.0

    arrays = {col: np.vstack(arrays_lists[col]) for col in arrays_lists}
    upw = np.vstack(upw_list)
    del arrays_lists, upw_list
    gc.collect()

    log_progress("START: precomputing 5x5 spatial window features")
    t_win0 = time.time()
    upw_all_full = np.vstack(upw_all_list)
    th_minus_ref_all_full = np.vstack(th_all_list) - np.vstack(ref_all_list)
    del upw_all_list, th_all_list, ref_all_list
    gc.collect()

    w5_upw_frac = np.empty((n_days_total, n_sub), dtype=np.float32)
    w5_thetao_minus_ref = np.empty((n_days_total, n_sub), dtype=np.float32)

    grid_buf = np.zeros((n_r, n_c), dtype=np.float32)
    for t in range(n_days_total):
        grid_buf.fill(0.0)
        grid_buf[r_all, c_all] = upw_all_full[t]
        conv_u = scipy.ndimage.uniform_filter(grid_buf, size=5, mode="constant", cval=0.0)
        w5_upw_frac[t] = conv_u[r_sub, c_sub] / ocean_count_sub

        grid_buf.fill(0.0)
        grid_buf[r_all, c_all] = th_minus_ref_all_full[t]
        conv_th = scipy.ndimage.uniform_filter(grid_buf, size=5, mode="constant", cval=0.0)
        w5_thetao_minus_ref[t] = conv_th[r_sub, c_sub] / ocean_count_sub

    del upw_all_full, th_minus_ref_all_full
    gc.collect()
    log_progress(f"END: 5x5 spatial window features precomputed in {time.time() - t_win0:.1f}s")

    # Days since last flagged, capped at 30.
    log_progress("START: precomputing days since last flagged (cap 30)")
    t_ds0 = time.time()
    days_since_flagged = np.full((n_days_total, n_sub), 30.0, dtype=np.float32)
    for t in range(1, n_days_total):
        days_since_flagged[t] = np.where(upw[t - 1] == 1, 1.0, np.minimum(30.0, days_since_flagged[t - 1] + 1.0))
    log_progress(f"END: days since last flagged precomputed in {time.time() - t_ds0:.1f}s")

    # Training climatology (2010-2017) smoothed with +/-7-day circular window.
    log_progress("START: precomputing season baseline climatology (2010-2017)")
    train_year_mask = np.isin(dates.year, range(2010, 2018))
    train_dates = dates[train_year_mask]
    train_upw = upw[train_year_mask]

    t_doy_idx, t_is_feb29 = map_timestamps_to_doy365(train_dates)
    t_doy_idx[t_is_feb29] = 59  # Map Feb 29 to day index 58 (Feb 28).

    sums_doy = np.zeros((365, n_sub), dtype=np.float64)
    counts_doy = np.zeros(365, dtype=np.int64)
    for d in range(365):
        m = t_doy_idx == d
        sums_doy[d] = train_upw[m].sum(axis=0)
        counts_doy[d] = m.sum()

    win_sums = np.zeros((365, n_sub), dtype=np.float64)
    win_counts = np.zeros(365, dtype=np.int64)
    for k in range(-7, 8):
        shift_idx = (np.arange(365) + k) % 365
        win_sums += sums_doy[shift_idx]
        win_counts += counts_doy[shift_idx]

    season_clim = (win_sums / win_counts[:, None]).astype(np.float32)
    log_progress("END: season baseline climatology precomputed")

    all_doy_idx, all_is_feb29 = map_timestamps_to_doy365(dates)
    all_doy_idx[all_is_feb29] = 59

    log_progress(f"END: loading data complete in {time.time() - t0:.1f}s (cells: {n_sub:,}, total days: {n_days_total:,})")
    return {
        "dates": dates,
        "doy_idx": all_doy_idx,
        "n_sub": n_sub,
        "arrays": arrays,
        "upw": upw,
        "days_since_flagged": days_since_flagged,
        "w5_upw_frac": w5_upw_frac,
        "w5_thetao_minus_ref": w5_thetao_minus_ref,
        "season_clim": season_clim,
        "lat": lat_vec,
        "lon": lon_vec,
        "coastal": coastal_vec,
        "sin_doy": np.sin(angles).astype(np.float32),
        "cos_doy": np.cos(angles).astype(np.float32),
    }


def build_features_indexed(data: dict, d_idx: np.ndarray, c_idx: np.ndarray) -> np.ndarray:
    assert np.all(d_idx >= 3), "D-3 lag requires day index >= 3"
    N = len(d_idx)
    X = np.empty((N, len(FEATURE_NAMES)), dtype=np.float32)

    arrays = data["arrays"]
    upw = data["upw"]
    w5_upw = data["w5_upw_frac"]

    col_i = 0
    for col in ("thetao_minus_ref", "no3", "chl", "sla", "o2_50_150m", "uo", "vo"):
        X[:, col_i] = arrays[col][d_idx, c_idx]
        col_i += 1

    for col in ("thetao", "no3", "chl"):
        X[:, col_i] = arrays[col][d_idx, c_idx] - arrays[col][d_idx - 3, c_idx]
        col_i += 1

    for lag in (1, 2, 3):
        X[:, col_i] = upw[d_idx - lag, c_idx].astype(np.float32)
        col_i += 1

    X[:, col_i] = data["days_since_flagged"][d_idx, c_idx]
    col_i += 1

    X[:, col_i] = w5_upw[d_idx, c_idx]
    col_i += 1
    X[:, col_i] = w5_upw[d_idx, c_idx] - w5_upw[d_idx - 3, c_idx]
    col_i += 1

    X[:, col_i] = data["w5_thetao_minus_ref"][d_idx, c_idx]
    col_i += 1

    X[:, col_i] = data["lat"][c_idx]
    col_i += 1
    X[:, col_i] = data["lon"][c_idx]
    col_i += 1
    X[:, col_i] = data["coastal"][c_idx]
    col_i += 1
    X[:, col_i] = data["sin_doy"][d_idx]
    col_i += 1
    X[:, col_i] = data["cos_doy"][d_idx]
    col_i += 1

    assert col_i == len(FEATURE_NAMES), f"Feature count mismatch: {col_i} vs {len(FEATURE_NAMES)}"
    return X


def select_best_f1_cutoff(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    p, r = prec[:-1], rec[:-1]
    denom = p + r
    f1_vals = np.zeros_like(denom)
    mask = denom > 0
    f1_vals[mask] = 2.0 * p[mask] * r[mask] / denom[mask]
    best_i = int(np.argmax(f1_vals))
    return float(thresh[best_i]), float(f1_vals[best_i])


def main() -> None:
    t_all = time.time()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FORECAST3 Onset Test Run Log Started\n")

    data = load_data()
    dates = data["dates"]
    upw = data["upw"]
    n_sub = data["n_sub"]
    doy_idx = data["doy_idx"]
    season_clim = data["season_clim"]
    w5_upw = data["w5_upw_frac"]

    idx_2017_end = int(np.where(dates.year == 2017)[0][-1])
    target_1819_idx = np.where(np.isin(dates.year, [2018, 2019]))[0]
    target_test_idx = np.where(np.isin(dates.year, range(2020, 2026)))[0]

    PERIOD_DEFS = [
        ("2020-2021", [2020, 2021]),
        ("2022-2025", [2022, 2023, 2024, 2025]),
        ("2020-2025", list(range(2020, 2026))),
    ]

    metric_rows = []
    imp_rows = []

    for h in HORIZONS:
        log_progress(f"START: processing horizon h={h}")
        t_h = time.time()

        train_stride = 4
        try:
            log_progress(f"START: preparing training rows (stride={train_stride}, keep positives + 10% negatives)")
            d_train = np.arange(3, idx_2017_end + 1 - h, train_stride)
            d_mesh, c_mesh = np.meshgrid(d_train, np.arange(n_sub), indexing="ij")
            upw_d = upw[d_mesh, c_mesh]
            mask_zero_d = upw_d == 0

            y_cand = upw[d_mesh + h, c_mesh][mask_zero_d]
            d_cand = d_mesh[mask_zero_d]
            c_cand = c_mesh[mask_zero_d]
            del d_mesh, c_mesh, upw_d, mask_zero_d
            gc.collect()

            pos_idx = np.where(y_cand == 1)[0]
            neg_idx = np.where(y_cand == 0)[0]
            rng = np.random.default_rng(42)
            n_neg_sample = int(round(0.10 * len(neg_idx)))
            sampled_neg_idx = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            selected_idx = np.sort(np.concatenate([pos_idx, sampled_neg_idx]))

            d_train_sub = d_cand[selected_idx]
            c_train_sub = c_cand[selected_idx]
            y_train = y_cand[selected_idx]
            del d_cand, c_cand, y_cand, pos_idx, neg_idx, sampled_neg_idx, selected_idx
            gc.collect()

            log_progress(f"START: building training features (rows={len(y_train):,})")
            X_train = build_features_indexed(data, d_train_sub, c_train_sub)
            del d_train_sub, c_train_sub
            gc.collect()

            log_progress(f"START: training HistGradientBoostingClassifier h={h} (max_iter=200)")
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
            clf.fit(X_train, y_train)
            del X_train, y_train
            gc.collect()
            log_progress(f"END: training HistGradientBoostingClassifier h={h} complete")
        except MemoryError:
            log_progress(f"WARNING: MemoryError encountered at h={h} with stride=4. Falling back to stride 8.")
            train_stride = 8
            gc.collect()
            d_train = np.arange(3, idx_2017_end + 1 - h, train_stride)
            d_mesh, c_mesh = np.meshgrid(d_train, np.arange(n_sub), indexing="ij")
            upw_d = upw[d_mesh, c_mesh]
            mask_zero_d = upw_d == 0

            y_cand = upw[d_mesh + h, c_mesh][mask_zero_d]
            d_cand = d_mesh[mask_zero_d]
            c_cand = c_mesh[mask_zero_d]
            del d_mesh, c_mesh, upw_d, mask_zero_d
            gc.collect()

            pos_idx = np.where(y_cand == 1)[0]
            neg_idx = np.where(y_cand == 0)[0]
            rng = np.random.default_rng(42)
            n_neg_sample = int(round(0.10 * len(neg_idx)))
            sampled_neg_idx = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            selected_idx = np.sort(np.concatenate([pos_idx, sampled_neg_idx]))

            d_train_sub = d_cand[selected_idx]
            c_train_sub = c_cand[selected_idx]
            y_train = y_cand[selected_idx]
            del d_cand, c_cand, y_cand, pos_idx, neg_idx, sampled_neg_idx, selected_idx
            gc.collect()

            X_train = build_features_indexed(data, d_train_sub, c_train_sub)
            del d_train_sub, c_train_sub
            gc.collect()

            log_progress(f"START: training HistGradientBoostingClassifier h={h} with stride=8 (rows={len(y_train):,})")
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
            clf.fit(X_train, y_train)
            del X_train, y_train
            gc.collect()
            log_progress(f"END: training HistGradientBoostingClassifier h={h} (stride=8) complete")

        log_progress(f"START: predicting validation set (2018-2019, all rows with flag at D=0) for h={h}")
        d_val_all = target_1819_idx - h
        p_val_list = []
        y_val_list = []

        chunk_days = 50
        for start_i in range(0, len(d_val_all), chunk_days):
            sub_d = d_val_all[start_i : start_i + chunk_days]
            d_m, c_m = np.meshgrid(sub_d, np.arange(n_sub), indexing="ij")
            u_d = upw[d_m, c_m]
            m_zero = u_d == 0

            d_chunk = d_m[m_zero]
            c_chunk = c_m[m_zero]
            y_chunk = upw[d_m + h, c_m][m_zero]
            del d_m, c_m, u_d, m_zero

            X_chunk = build_features_indexed(data, d_chunk, c_chunk)
            p_chunk = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)
            del X_chunk, d_chunk, c_chunk

            p_val_list.append(p_chunk)
            y_val_list.append(y_chunk)

        p_val = np.concatenate(p_val_list)
        y_val = np.concatenate(y_val_list)
        del p_val_list, y_val_list
        gc.collect()

        cutoff, val_f1 = select_best_f1_cutoff(y_val, p_val)
        log_progress(f"END: validation complete for h={h} (rows={len(y_val):,}, cutoff={cutoff:.4f}, val_F1={val_f1:.4f})")

        log_progress(f"START: permutation importance h={h} (50k rows, 2018-2019)")
        rng = np.random.default_rng(42)
        sample_d_val = rng.choice(d_val_all, size=min(len(d_val_all), 15), replace=False)
        d_sm, c_sm = np.meshgrid(sample_d_val, np.arange(n_sub), indexing="ij")
        sm_zero = upw[d_sm, c_sm] == 0

        d_samp = d_sm[sm_zero]
        c_samp = c_sm[sm_zero]
        y_samp = upw[d_sm + h, c_sm][sm_zero]
        del d_sm, c_sm, sm_zero

        sample_50k = rng.choice(len(y_samp), size=min(50_000, len(y_samp)), replace=False)
        X_imp = build_features_indexed(data, d_samp[sample_50k], c_samp[sample_50k])
        y_imp = y_samp[sample_50k]
        del d_samp, c_samp, y_samp
        gc.collect()

        perm = permutation_importance(clf, X_imp, y_imp, scoring="average_precision", n_repeats=5, random_state=42, n_jobs=1)
        for rank_i, feat_i in enumerate(np.argsort(perm.importances_mean)[::-1], start=1):
            imp_rows.append(
                {
                    "horizon_days": h,
                    "rank": rank_i,
                    "feature": FEATURE_NAMES[feat_i],
                    "importance_mean": float(perm.importances_mean[feat_i]),
                    "importance_std": float(perm.importances_std[feat_i]),
                }
            )
        del X_imp, y_imp
        gc.collect()
        log_progress(f"END: permutation importance complete for h={h}")

        log_progress(f"START: evaluating test set (2020-2025, all rows with flag at D=0) for h={h}")
        d_test_all = target_test_idx - h

        scores_model_list = []
        scores_season_list = []
        scores_spreading_list = []
        scores_flicker_list = []
        y_test_list = []
        year_list = []
        month_list = []

        for start_i in range(0, len(d_test_all), chunk_days):
            sub_d = d_test_all[start_i : start_i + chunk_days]
            d_m, c_m = np.meshgrid(sub_d, np.arange(n_sub), indexing="ij")
            u_d = upw[d_m, c_m]
            m_zero = u_d == 0

            d_chunk = d_m[m_zero]
            c_chunk = c_m[m_zero]
            y_chunk = upw[d_m + h, c_m][m_zero]

            target_dates_chunk = dates[d_chunk + h]
            year_chunk = target_dates_chunk.year.to_numpy(dtype=np.int16)
            month_chunk = target_dates_chunk.month.to_numpy(dtype=np.int8)

            # Baselines:
            # 1. Season: fraction of days cell was flagged in 2010-2017 within +/- 7 days of target day-of-year
            target_doy = doy_idx[d_chunk + h]
            s_season = season_clim[target_doy, c_chunk]

            # 2. Spreading: 5x5 flagged fraction at D
            s_spreading = w5_upw[d_chunk, c_chunk]

            # 3. Flicker: 1 if flagged on any of D-1..D-3 else 0
            s_flicker = (
                (upw[d_chunk - 1, c_chunk] == 1)
                | (upw[d_chunk - 2, c_chunk] == 1)
                | (upw[d_chunk - 3, c_chunk] == 1)
            ).astype(np.float32)

            del d_m, c_m, u_d, m_zero

            X_chunk = build_features_indexed(data, d_chunk, c_chunk)
            s_model = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)
            del X_chunk, d_chunk, c_chunk

            scores_model_list.append(s_model)
            scores_season_list.append(s_season)
            scores_spreading_list.append(s_spreading)
            scores_flicker_list.append(s_flicker)
            y_test_list.append(y_chunk)
            year_list.append(year_chunk)
            month_list.append(month_chunk)

        s_mod_all = np.concatenate(scores_model_list)
        s_sea_all = np.concatenate(scores_season_list)
        s_spr_all = np.concatenate(scores_spreading_list)
        s_fli_all = np.concatenate(scores_flicker_list)
        y_true_all = np.concatenate(y_test_list)
        years_all = np.concatenate(year_list)
        months_all = np.concatenate(month_list)

        del scores_model_list, scores_season_list, scores_spreading_list, scores_flicker_list
        del y_test_list, year_list, month_list
        gc.collect()

        for period_name, p_years in PERIOD_DEFS:
            p_mask = np.isin(years_all, p_years)
            for subset_name in ("Jun-Sep", "All days"):
                if subset_name == "Jun-Sep":
                    m_mask = np.isin(months_all, [6, 7, 8, 9])
                else:
                    m_mask = np.ones(len(months_all), dtype=bool)

                mask = p_mask & m_mask
                yt = y_true_all[mask]
                sm = s_mod_all[mask]
                ss = s_sea_all[mask]
                sp = s_spr_all[mask]
                sf = s_fli_all[mask]

                N_eval = len(yt)
                base_rate = float(yt.mean()) if N_eval > 0 else 0.0

                ap_model = float(average_precision_score(yt, sm)) if N_eval > 0 else 0.0
                ap_season = float(average_precision_score(yt, ss)) if N_eval > 0 else 0.0
                ap_spreading = float(average_precision_score(yt, sp)) if N_eval > 0 else 0.0
                ap_flicker = float(average_precision_score(yt, sf)) if N_eval > 0 else 0.0

                best_baseline_ap = max(ap_season, ap_spreading, ap_flicker)
                ap_ratio = ap_model / best_baseline_ap if best_baseline_ap > 0 else 0.0

                preds = (sm >= cutoff).astype(np.int8)
                onset_events = int(yt.sum())
                alarms_raised = int(preds.sum())
                tp = int(np.sum((yt == 1) & (preds == 1)))
                precision = tp / alarms_raised if alarms_raised > 0 else 0.0
                recall = tp / onset_events if onset_events > 0 else 0.0
                f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                # Lift: precision of top 1% scored cell-days / base_rate
                K = max(1, int(round(0.01 * N_eval)))
                top1_idx = np.argsort(sm)[::-1][:K]
                prec_top1 = float(yt[top1_idx].mean())
                lift = prec_top1 / base_rate if base_rate > 0 else 0.0

                metric_rows.append(
                    {
                        "horizon_days": h,
                        "period": period_name,
                        "subset": subset_name,
                        "total_rows": N_eval,
                        "base_rate": base_rate,
                        "ap_model": ap_model,
                        "ap_season": ap_season,
                        "ap_spreading": ap_spreading,
                        "ap_flicker": ap_flicker,
                        "best_baseline_ap": best_baseline_ap,
                        "ap_ratio_vs_best": ap_ratio,
                        "val_cutoff": cutoff,
                        "val_f1": val_f1,
                        "onset_events": onset_events,
                        "alarms_raised": alarms_raised,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "prec_top1pct": prec_top1,
                        "lift_top1pct": lift,
                    }
                )

        del s_mod_all, s_sea_all, s_spr_all, s_fli_all, y_true_all, years_all, months_all, clf
        gc.collect()
        log_progress(f"END: processing horizon h={h} complete in {time.time() - t_h:.1f}s")

    metrics_df = pd.DataFrame(metric_rows)
    imp_df = pd.DataFrame(imp_rows)

    metrics_df.to_csv(BASE_DIR / "onset_metrics.csv", index=False)
    imp_df.to_csv(BASE_DIR / "permutation_importance_2018_2019.csv", index=False)

    log_progress("\n" + "=" * 80)
    log_progress("=== ONSET TEST METRICS ===")
    log_progress("=" * 80)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== TOP 10 PERMUTATION IMPORTANCE FEATURES (AVERAGE PRECISION, 2018-2019) ===")
    log_progress("=" * 80)
    top10 = imp_df[imp_df["rank"] <= 10]
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== PASS / FAIL SUMMARY (JUN-SEP: MODEL AP >= 1.25 x BEST BASELINE) ===")
    log_progress("=" * 80)

    # Check (a) at h=3 and h=5 on 2020-2025 Jun-Sep
    pass_a = True
    for h in (3, 5):
        row = metrics_df[(metrics_df["horizon_days"] == h) & (metrics_df["period"] == "2020-2025") & (metrics_df["subset"] == "Jun-Sep")].iloc[0]
        status = "PASS" if row["ap_ratio_vs_best"] >= 1.25 else "FAIL"
        if status == "FAIL":
            pass_a = False
        log_progress(
            f"(a) Horizon h={h} [2020-2025 Jun-Sep]: {status} | "
            f"Model AP={row['ap_model']:.4f}, Best Baseline AP={row['best_baseline_ap']:.4f}, "
            f"Ratio={row['ap_ratio_vs_best']:.3f}x (req >= 1.250x)"
        )

    # Check (b) in 2020-2021 and 2022-2025 separately
    pass_b = True
    for period in ("2020-2021", "2022-2025"):
        for h in (3, 5):
            row = metrics_df[(metrics_df["horizon_days"] == h) & (metrics_df["period"] == period) & (metrics_df["subset"] == "Jun-Sep")].iloc[0]
            status = "PASS" if row["ap_ratio_vs_best"] >= 1.25 else "FAIL"
            if status == "FAIL":
                pass_b = False
            log_progress(
                f"(b) Horizon h={h} [{period} Jun-Sep]: {status} | "
                f"Model AP={row['ap_model']:.4f}, Best Baseline AP={row['best_baseline_ap']:.4f}, "
                f"Ratio={row['ap_ratio_vs_best']:.3f}x (req >= 1.250x)"
            )

    log_progress(f"Overall Criterion (a): {'PASS' if pass_a else 'FAIL'}")
    log_progress(f"Overall Criterion (b): {'PASS' if pass_b else 'FAIL'}")
    log_progress(f"END: entire pipeline completed in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
