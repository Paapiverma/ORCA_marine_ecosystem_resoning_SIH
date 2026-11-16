"""Ablation studies, per-year evaluations, and horizon sweeps for onset forecast robustness. Run: python forecast4_robustness_test.py"""

import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy.ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "forecast4_run_log.txt"
try:
    from config import ORCA_LONG_DIR as LONG_DIR, MERGED_V3_DIR as M3_DIR, RULES_V3C_DIR as R3C_DIR
except ImportError:
    from orca_v2.config import ORCA_LONG_DIR as LONG_DIR, MERGED_V3_DIR as M3_DIR, RULES_V3C_DIR as R3C_DIR

N_CELLS = 18650
YEARS = list(range(2010, 2026))

OCEAN_COLS = ["thetao", "no3", "chl", "sla", "o2_50_150m", "uo", "vo"]
RULES_COLS = ["time", "latitude", "longitude", "is_coastal", "excluded_north", "ref_thetao", "upwelling"]

ALL_FEATURE_NAMES = [
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

# Feature variants
# V1: Calendar and location only (5 features)
V1_INDICES = [17, 18, 19, 20, 21]
# V2: Full minus rule ingredients (remove 0, 1, 2, 7, 8, 9, 16) -> 15 features
V2_INDICES = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]
# V3: Full model (22 features)
V3_INDICES = list(range(22))

SWEEP_HORIZONS = (1, 3, 5, 7, 10, 15)


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

    log_progress("START: precomputing days since last flagged (cap 30)")
    t_ds0 = time.time()
    days_since_flagged = np.full((n_days_total, n_sub), 30.0, dtype=np.float32)
    for t in range(1, n_days_total):
        days_since_flagged[t] = np.where(upw[t - 1] == 1, 1.0, np.minimum(30.0, days_since_flagged[t - 1] + 1.0))
    log_progress(f"END: days since last flagged precomputed in {time.time() - t_ds0:.1f}s")

    log_progress("START: precomputing season baseline climatology (2010-2017)")
    train_year_mask = np.isin(dates.year, range(2010, 2018))
    train_dates = dates[train_year_mask]
    train_upw = upw[train_year_mask]

    t_doy_idx, t_is_feb29 = map_timestamps_to_doy365(train_dates)
    t_doy_idx[t_is_feb29] = 59

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


def build_features_indexed(data: dict, d_idx: np.ndarray, c_idx: np.ndarray, feat_indices: list[int] = None) -> np.ndarray:
    assert np.all(d_idx >= 3), "D-3 lag requires day index >= 3"
    N = len(d_idx)

    arrays = data["arrays"]
    upw = data["upw"]
    w5_upw = data["w5_upw_frac"]

    if feat_indices is None:
        feat_indices = V3_INDICES

    X_full = np.empty((N, 22), dtype=np.float32)

    col_i = 0
    for col in ("thetao_minus_ref", "no3", "chl", "sla", "o2_50_150m", "uo", "vo"):
        X_full[:, col_i] = arrays[col][d_idx, c_idx]
        col_i += 1

    for col in ("thetao", "no3", "chl"):
        X_full[:, col_i] = arrays[col][d_idx, c_idx] - arrays[col][d_idx - 3, c_idx]
        col_i += 1

    for lag in (1, 2, 3):
        X_full[:, col_i] = upw[d_idx - lag, c_idx].astype(np.float32)
        col_i += 1

    X_full[:, col_i] = data["days_since_flagged"][d_idx, c_idx]
    col_i += 1

    X_full[:, col_i] = w5_upw[d_idx, c_idx]
    col_i += 1
    X_full[:, col_i] = w5_upw[d_idx, c_idx] - w5_upw[d_idx - 3, c_idx]
    col_i += 1

    X_full[:, col_i] = data["w5_thetao_minus_ref"][d_idx, c_idx]
    col_i += 1

    X_full[:, col_i] = data["lat"][c_idx]
    col_i += 1
    X_full[:, col_i] = data["lon"][c_idx]
    col_i += 1
    X_full[:, col_i] = data["coastal"][c_idx]
    col_i += 1
    X_full[:, col_i] = data["sin_doy"][d_idx]
    col_i += 1
    X_full[:, col_i] = data["cos_doy"][d_idx]
    col_i += 1

    assert col_i == 22
    if len(feat_indices) == 22:
        return X_full
    return X_full[:, feat_indices]


def select_best_f1_cutoff(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    p, r = prec[:-1], rec[:-1]
    denom = p + r
    f1_vals = np.zeros_like(denom)
    mask = denom > 0
    f1_vals[mask] = 2.0 * p[mask] * r[mask] / denom[mask]
    best_i = int(np.argmax(f1_vals))
    return float(thresh[best_i]), float(f1_vals[best_i])


def train_and_eval_variant(
    data: dict,
    h: int,
    feat_name: str,
    feat_indices: list[int],
    target_1819_idx: np.ndarray,
    target_test_idx: np.ndarray,
    idx_2017_end: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    log_progress(f"START: training {feat_name} for h={h} ({len(feat_indices)} features)")
    t0 = time.time()
    dates = data["dates"]
    upw = data["upw"]
    n_sub = data["n_sub"]

    train_stride = 4
    try:
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

        X_train = build_features_indexed(data, d_train_sub, c_train_sub, feat_indices=feat_indices)
        del d_train_sub, c_train_sub
        gc.collect()

        clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        del X_train, y_train
        gc.collect()
        log_progress(f"END: training {feat_name} h={h} complete in {time.time() - t0:.1f}s")
    except MemoryError:
        log_progress(f"WARNING: MemoryError encountered for {feat_name} h={h}. Falling back to stride 8.")
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

        X_train = build_features_indexed(data, d_train_sub, c_train_sub, feat_indices=feat_indices)
        del d_train_sub, c_train_sub
        gc.collect()

        clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        del X_train, y_train
        gc.collect()
        log_progress(f"END: training {feat_name} h={h} (stride=8) complete")

    log_progress(f"START: evaluating {feat_name} validation (2018-2019) for h={h}")
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

        X_chunk = build_features_indexed(data, d_chunk, c_chunk, feat_indices=feat_indices)
        p_chunk = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)
        del X_chunk, d_chunk, c_chunk

        p_val_list.append(p_chunk)
        y_val_list.append(y_chunk)

    p_val = np.concatenate(p_val_list)
    y_val = np.concatenate(y_val_list)
    del p_val_list, y_val_list
    gc.collect()

    cutoff, val_f1 = select_best_f1_cutoff(y_val, p_val)
    del p_val, y_val
    gc.collect()
    log_progress(f"END: validation for {feat_name} h={h} (cutoff={cutoff:.4f}, val_F1={val_f1:.4f})")

    log_progress(f"START: predicting test set (2020-2025) for {feat_name} h={h}")
    d_test_all = target_test_idx - h
    scores_test_list = []

    for start_i in range(0, len(d_test_all), chunk_days):
        sub_d = d_test_all[start_i : start_i + chunk_days]
        d_m, c_m = np.meshgrid(sub_d, np.arange(n_sub), indexing="ij")
        u_d = upw[d_m, c_m]
        m_zero = u_d == 0

        d_chunk = d_m[m_zero]
        c_chunk = c_m[m_zero]
        del d_m, c_m, u_d, m_zero

        X_chunk = build_features_indexed(data, d_chunk, c_chunk, feat_indices=feat_indices)
        s_chunk = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)
        del X_chunk, d_chunk, c_chunk

        scores_test_list.append(s_chunk)

    scores_test = np.concatenate(scores_test_list)
    del scores_test_list, clf
    gc.collect()
    log_progress(f"END: predicting test set for {feat_name} h={h} complete")
    return scores_test, cutoff, val_f1


def extract_test_ground_truth_and_baselines(data: dict, h: int, target_test_idx: np.ndarray) -> dict:
    log_progress(f"START: extracting ground truth and baselines for test set h={h}")
    t0 = time.time()
    dates = data["dates"]
    upw = data["upw"]
    n_sub = data["n_sub"]
    doy_idx = data["doy_idx"]
    season_clim = data["season_clim"]
    w5_upw = data["w5_upw_frac"]

    d_test_all = target_test_idx - h
    y_test_list = []
    season_list = []
    spreading_list = []
    flicker_list = []
    year_list = []
    month_list = []

    chunk_days = 50
    for start_i in range(0, len(d_test_all), chunk_days):
        sub_d = d_test_all[start_i : start_i + chunk_days]
        d_m, c_m = np.meshgrid(sub_d, np.arange(n_sub), indexing="ij")
        u_d = upw[d_m, c_m]
        m_zero = u_d == 0

        d_chunk = d_m[m_zero]
        c_chunk = c_m[m_zero]
        y_chunk = upw[d_m + h, c_m][m_zero]

        target_dates = dates[d_chunk + h]
        y_test_list.append(y_chunk)
        year_list.append(target_dates.year.to_numpy(dtype=np.int16))
        month_list.append(target_dates.month.to_numpy(dtype=np.int8))

        # Season baseline: climatology at target DOY
        target_doy = doy_idx[d_chunk + h]
        season_list.append(season_clim[target_doy, c_chunk])

        # Spreading baseline: 5x5 fraction at D
        spreading_list.append(w5_upw[d_chunk, c_chunk])

        # Flicker baseline: 1 if flagged on D-1..D-3 else 0
        flicker_list.append(
            (
                (upw[d_chunk - 1, c_chunk] == 1)
                | (upw[d_chunk - 2, c_chunk] == 1)
                | (upw[d_chunk - 3, c_chunk] == 1)
            ).astype(np.float32)
        )
        del d_m, c_m, u_d, m_zero, d_chunk, c_chunk

    res = {
        "y_true": np.concatenate(y_test_list),
        "year": np.concatenate(year_list),
        "month": np.concatenate(month_list),
        "season": np.concatenate(season_list),
        "spreading": np.concatenate(spreading_list),
        "flicker": np.concatenate(flicker_list),
    }
    del y_test_list, year_list, month_list, season_list, spreading_list, flicker_list
    gc.collect()
    log_progress(f"END: ground truth & baselines extracted for h={h} in {time.time() - t0:.1f}s (total rows: {len(res['y_true']):,})")
    return res


def main() -> None:
    t_all = time.time()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FORECAST4 Robustness Checks Started\n")

    data = load_data()
    dates = data["dates"]
    idx_2017_end = int(np.where(dates.year == 2017)[0][-1])
    target_1819_idx = np.where(np.isin(dates.year, [2018, 2019]))[0]
    target_test_idx = np.where(np.isin(dates.year, range(2020, 2026)))[0]

    ablation_rows = []
    per_year_rows = []
    horizon_sweep_rows = []

    v3_cache = {}

    for h in (3, 5):
        log_progress(f"\n{'='*70}\nSTART: Horizon h={h} Processing (Part B & C)\n{'='*70}")
        test_info = extract_test_ground_truth_and_baselines(data, h, target_test_idx)

        scores_v1, cutoff_v1, val_f1_v1 = train_and_eval_variant(
            data, h, "V1_Calendar_Location", V1_INDICES, target_1819_idx, target_test_idx, idx_2017_end
        )

        scores_v2, cutoff_v2, val_f1_v2 = train_and_eval_variant(
            data, h, "V2_Minus_Rules", V2_INDICES, target_1819_idx, target_test_idx, idx_2017_end
        )

        scores_v3, cutoff_v3, val_f1_v3 = train_and_eval_variant(
            data, h, "V3_Full_Model", V3_INDICES, target_1819_idx, target_test_idx, idx_2017_end
        )
        v3_cache[h] = {"scores": scores_v3, "test_info": test_info}

        mask_js_2025 = np.isin(test_info["month"], [6, 7, 8, 9])
        yt_js = test_info["y_true"][mask_js_2025]
        s_sea_js = test_info["season"][mask_js_2025]
        s_v1_js = scores_v1[mask_js_2025]
        s_v2_js = scores_v2[mask_js_2025]
        s_v3_js = scores_v3[mask_js_2025]

        ap_season = float(average_precision_score(yt_js, s_sea_js))
        ap_v1 = float(average_precision_score(yt_js, s_v1_js))
        ap_v2 = float(average_precision_score(yt_js, s_v2_js))
        ap_v3 = float(average_precision_score(yt_js, s_v3_js))
        base_rate_js = float(yt_js.mean())

        ablation_rows.append(
            {
                "horizon_days": h,
                "subset": "Jun-Sep (2020-2025)",
                "total_rows": len(yt_js),
                "base_rate": base_rate_js,
                "ap_season": ap_season,
                "ap_v1": ap_v1,
                "ap_v2": ap_v2,
                "ap_v3": ap_v3,
                "ratio_v3_vs_v1": ap_v3 / ap_v1 if ap_v1 > 0 else 0.0,
                "ratio_v3_vs_season": ap_v3 / ap_season if ap_season > 0 else 0.0,
                "ratio_v3_vs_v2": ap_v3 / ap_v2 if ap_v2 > 0 else 0.0,
            }
        )

        del scores_v1, scores_v2
        gc.collect()

        for yr in (2020, 2021, 2022, 2023, 2024, 2025):
            mask_yr = (test_info["year"] == yr) & np.isin(test_info["month"], [6, 7, 8, 9])
            yt_yr = test_info["y_true"][mask_yr]
            s_v3_yr = scores_v3[mask_yr]
            s_sea_yr = test_info["season"][mask_yr]

            n_events = int(yt_yr.sum())
            ap_v3_yr = float(average_precision_score(yt_yr, s_v3_yr))
            ap_sea_yr = float(average_precision_score(yt_yr, s_sea_yr))
            ratio_yr = ap_v3_yr / ap_sea_yr if ap_sea_yr > 0 else 0.0

            per_year_rows.append(
                {
                    "year": yr,
                    "horizon_days": h,
                    "subset": "Jun-Sep",
                    "total_rows": len(yt_yr),
                    "onset_events": n_events,
                    "base_rate": float(yt_yr.mean()),
                    "ap_v3": ap_v3_yr,
                    "ap_season": ap_sea_yr,
                    "ratio_v3_vs_season": ratio_yr,
                    "v3_beats_season": ap_v3_yr > ap_sea_yr,
                }
            )

    log_progress(f"\n{'='*70}\nSTART: Part D Horizon Sweep for V3\n{'='*70}")
    for h in SWEEP_HORIZONS:
        if h in v3_cache:
            scores_v3 = v3_cache[h]["scores"]
            test_info = v3_cache[h]["test_info"]
        else:
            test_info = extract_test_ground_truth_and_baselines(data, h, target_test_idx)
            scores_v3, cutoff_v3, val_f1_v3 = train_and_eval_variant(
                data, h, f"V3_Full_Model_h{h}", V3_INDICES, target_1819_idx, target_test_idx, idx_2017_end
            )

        mask_js = np.isin(test_info["month"], [6, 7, 8, 9])
        yt_js = test_info["y_true"][mask_js]
        s_v3_js = scores_v3[mask_js]
        s_sea_js = test_info["season"][mask_js]

        ap_v3 = float(average_precision_score(yt_js, s_v3_js))
        ap_sea = float(average_precision_score(yt_js, s_sea_js))
        b_rate = float(yt_js.mean())
        ratio = ap_v3 / ap_sea if ap_sea > 0 else 0.0

        horizon_sweep_rows.append(
            {
                "horizon_days": h,
                "subset": "Jun-Sep (2020-2025)",
                "total_rows": len(yt_js),
                "onset_events": int(yt_js.sum()),
                "base_rate": b_rate,
                "ap_v3": ap_v3,
                "ap_season": ap_sea,
                "ratio_v3_vs_season": ratio,
            }
        )

    df_ablation = pd.DataFrame(ablation_rows)
    df_per_year = pd.DataFrame(per_year_rows)
    df_sweep = pd.DataFrame(horizon_sweep_rows)

    df_ablation.to_csv(BASE_DIR / "ablation_metrics.csv", index=False)
    df_per_year.to_csv(BASE_DIR / "per_year_metrics.csv", index=False)
    df_sweep.to_csv(BASE_DIR / "horizon_sweep_metrics.csv", index=False)

    log_progress("\n" + "=" * 80)
    log_progress("=== PART B: ABLATIONS (2020-2025 Jun-Sep) ===")
    log_progress("=" * 80)
    print(df_ablation.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== PART C: PER YEAR BREAKDOWN (Jun-Sep) ===")
    log_progress("=" * 80)
    print(df_per_year.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== PART D: HORIZON SWEEP (2020-2025 Jun-Sep) ===")
    log_progress("=" * 80)
    print(df_sweep.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== PASS / FAIL CHECKS ===")
    log_progress("=" * 80)

    # Check (a): V3 AP >= 1.25 x V1 AP at h=3 and h=5
    pass_a = True
    for h in (3, 5):
        row = df_ablation[df_ablation["horizon_days"] == h].iloc[0]
        status = "PASS" if row["ratio_v3_vs_v1"] >= 1.25 else "FAIL"
        if status == "FAIL":
            pass_a = False
        log_progress(
            f"(a) Horizon h={h}: {status} | V3 AP={row['ap_v3']:.4f}, V1 AP={row['ap_v1']:.4f}, "
            f"Ratio={row['ratio_v3_vs_v1']:.3f}x (req >= 1.250x)"
        )

    # Check (b): V3 AP > Season baseline in each of the six test years at h=3 and h=5
    pass_b = True
    for h in (3, 5):
        sub_yr = df_per_year[df_per_year["horizon_days"] == h]
        all_beat = bool(sub_yr["v3_beats_season"].all())
        if not all_beat:
            pass_b = False
        log_progress(f"(b) Horizon h={h} all 6 years beat season: {'PASS' if all_beat else 'FAIL'}")
        for _, r in sub_yr.iterrows():
            log_progress(
                f"    Year {int(r['year'])}: {'PASS' if r['v3_beats_season'] else 'FAIL'} | "
                f"V3={r['ap_v3']:.4f}, Season={r['ap_season']:.4f}, Diff={r['ap_v3'] - r['ap_season']:+.4f}"
            )

    log_progress(f"Overall Check (a): {'PASS' if pass_a else 'FAIL'}")
    log_progress(f"Overall Check (b): {'PASS' if pass_b else 'FAIL'}")
    log_progress(f"END: entire pipeline completed in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
