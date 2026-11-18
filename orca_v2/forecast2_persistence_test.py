"""Evaluate persistence baseline against gradient boosted trees for upwelling flag forecasting (h=1,3,5). Run: python forecast2_persistence_test.py"""

import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "forecast2_run_log.txt"
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
    "upw_d0",
    "upw_d1",
    "upw_d2",
    "upw_d3",
    "d3_thetao",
    "d3_no3",
    "d3_chl",
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


def compute_prf1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int, float, float, float]:
    pos = int(y_true.sum())
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = pos - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / pos if pos > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return pos, tp, fp, fn, precision, recall, f1


def map_timestamps_to_doy365(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    is_feb29 = np.asarray((dates.month == 2) & (dates.day == 29), dtype=bool)
    raw_doy = dates.dayofyear.to_numpy(dtype=np.int64)
    shift = (dates.is_leap_year & (raw_doy > 60)).astype(np.int64)
    doy_idx = raw_doy - shift - 1
    return doy_idx, is_feb29


def load_data() -> dict:
    log_progress("START: loading data (2010-2025 continuous)")
    t0 = time.time()

    # Anchor grid on 2022 rules.
    d0 = pd.read_parquet(
        LONG_DIR / "rules_v3c_2022.parquet",
        columns=["time", "latitude", "longitude", "is_coastal", "excluded_north"],
    )
    d0 = d0[d0["time"] == d0["time"].iloc[0]]
    assert len(d0) == N_CELLS, f"Expected {N_CELLS} cells, got {len(d0)}"

    lats_all = d0["latitude"].to_numpy(dtype=np.float32)
    keep_lat = lats_all < 19.5
    assert np.array_equal(keep_lat, d0["excluded_north"].to_numpy() == 0)
    n_sub = int(keep_lat.sum())

    lat_vec = lats_all[keep_lat]
    lon_vec = d0["longitude"].to_numpy(dtype=np.float32)[keep_lat]
    coastal_vec = d0["is_coastal"].to_numpy(dtype=np.float32)[keep_lat]

    dates_list = []
    upw_list = []
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

        th_2d = df_o["thetao"].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)[:, keep_lat]
        ref_2d = df_r["ref_thetao"].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)[:, keep_lat]
        arrays_lists["thetao"].append(th_2d)
        arrays_lists["thetao_minus_ref"].append((th_2d - ref_2d).astype(np.float32))

        for col in ("no3", "chl", "sla", "o2_50_150m", "uo", "vo"):
            arrays_lists[col].append(df_o[col].to_numpy(dtype=np.float32).reshape(n_days, N_CELLS)[:, keep_lat])

        u_2d = df_r["upwelling"].to_numpy(dtype=np.int8).reshape(n_days, N_CELLS)[:, keep_lat]
        upw_list.append(u_2d)

        del df_o, df_r
        gc.collect()
        log_progress(f"Loaded year {year} ({n_days} days) in {time.time() - t_yr0:.1f}s")

    dates = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))
    assert np.all(np.diff(dates.asi8) == 86_400_000_000_000), "Expected contiguous daily grid"

    doy_idx, is_feb29 = map_timestamps_to_doy365(dates)
    doy_cont = np.where(is_feb29, 59.5, doy_idx.astype(np.float64) + 1.0)
    angles = 2.0 * np.pi * doy_cont / 365.0

    arrays = {col: np.vstack(arrays_lists[col]) for col in arrays_lists}
    upw = np.vstack(upw_list)

    log_progress(f"END: loading data complete in {time.time() - t0:.1f}s (cells: {n_sub:,}, total days: {len(dates):,})")
    return {
        "dates": dates,
        "n_sub": n_sub,
        "arrays": arrays,
        "upw": upw,
        "lat": lat_vec,
        "lon": lon_vec,
        "coastal": coastal_vec,
        "sin_doy": np.sin(angles).astype(np.float32),
        "cos_doy": np.cos(angles).astype(np.float32),
    }


def build_features(data: dict, d_indices: np.ndarray, log_tag: str = "") -> np.ndarray:
    assert np.all(d_indices >= 3), "D-3 lag requires day index >= 3"
    if log_tag:
        log_progress(f"START: building features ({log_tag}, days={len(d_indices)})")
    t0 = time.time()
    n_d = len(d_indices)
    n_sub = data["n_sub"]
    X = np.empty((n_d * n_sub, len(FEATURE_NAMES)), dtype=np.float32)

    arrays = data["arrays"]
    upw = data["upw"]

    col_i = 0
    for col in ("thetao_minus_ref", "no3", "chl", "sla", "o2_50_150m", "uo", "vo"):
        X[:, col_i] = arrays[col][d_indices].reshape(-1)
        col_i += 1

    for lag in (0, 1, 2, 3):
        X[:, col_i] = upw[d_indices - lag].astype(np.float32).reshape(-1)
        col_i += 1

    for col in ("thetao", "no3", "chl"):
        diff = arrays[col][d_indices] - arrays[col][d_indices - 3]
        X[:, col_i] = diff.reshape(-1)
        col_i += 1

    X[:, col_i] = np.broadcast_to(data["lat"][None, :], (n_d, n_sub)).reshape(-1)
    col_i += 1
    X[:, col_i] = np.broadcast_to(data["lon"][None, :], (n_d, n_sub)).reshape(-1)
    col_i += 1
    X[:, col_i] = np.broadcast_to(data["coastal"][None, :], (n_d, n_sub)).reshape(-1)
    col_i += 1
    X[:, col_i] = np.broadcast_to(data["sin_doy"][d_indices, None], (n_d, n_sub)).reshape(-1)
    col_i += 1
    X[:, col_i] = np.broadcast_to(data["cos_doy"][d_indices, None], (n_d, n_sub)).reshape(-1)
    col_i += 1

    assert col_i == len(FEATURE_NAMES), f"Feature count mismatch: {col_i} vs {len(FEATURE_NAMES)}"
    if log_tag:
        log_progress(f"END: building features ({log_tag}) complete in {time.time() - t0:.1f}s (rows={len(X):,})")
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


def predict_chunked(clf: HistGradientBoostingClassifier, data: dict, d_indices: np.ndarray, chunk_days: int = 100) -> np.ndarray:
    n_sub = data["n_sub"]
    probs = np.empty(len(d_indices) * n_sub, dtype=np.float32)
    offset = 0
    for start in range(0, len(d_indices), chunk_days):
        sub_d = d_indices[start : start + chunk_days]
        X_chunk = build_features(data, sub_d)
        m = len(X_chunk)
        probs[offset : offset + m] = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)
        offset += m
        del X_chunk
    return probs


def eval_transition_metrics(y_true: np.ndarray, y_pred: np.ndarray, flag_d: np.ndarray) -> dict[str, float]:
    pos, tp, fp, fn, prec, rec, f1 = compute_prf1(y_true, y_pred)

    # Onset: conditioned on flag[D] == 0.
    zero_d = flag_d == 0
    true_onset = zero_d & (y_true == 1)
    pred_onset = zero_d & (y_pred == 1)
    onset_tp = int(np.sum(true_onset & pred_onset))
    onset_n = int(np.sum(true_onset))
    onset_pred_n = int(np.sum(pred_onset))
    onset_rec = onset_tp / onset_n if onset_n > 0 else 0.0
    onset_prec = onset_tp / onset_pred_n if onset_pred_n > 0 else 0.0

    # Offset: conditioned on flag[D] == 1.
    one_d = flag_d == 1
    true_offset = one_d & (y_true == 0)
    pred_offset = one_d & (y_pred == 0)
    offset_tp = int(np.sum(true_offset & pred_offset))
    offset_n = int(np.sum(true_offset))
    offset_rec = offset_tp / offset_n if offset_n > 0 else 0.0

    return {
        "pos_cell_days": pos,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "onset_events": onset_n,
        "onset_precision": onset_prec,
        "onset_recall": onset_rec,
        "offset_events": offset_n,
        "offset_recall": offset_rec,
    }


def main() -> None:
    t_all = time.time()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FORECAST2 Run Log Started\n")

    data = load_data()
    dates = data["dates"]
    upw = data["upw"]
    n_sub = data["n_sub"]

    idx_2017_start = int(np.where(dates.year == 2017)[0][0])
    idx_2017_end = int(np.where(dates.year == 2017)[0][-1])
    target_1819_idx = np.where(np.isin(dates.year, [2018, 2019]))[0]
    target_test_idx = np.where(np.isin(dates.year, range(2020, 2026)))[0]

    test_dates = dates[target_test_idx]

    PERIOD_DEFS = [
        ("2020-2021", np.isin(test_dates.year, [2020, 2021])),
        ("2022-2025", np.isin(test_dates.year, [2022, 2023, 2024, 2025])),
        ("2020-2025", np.ones(len(test_dates), dtype=bool)),
    ]

    metric_rows = []
    imp_rows = []

    # Persistence baseline is identical for both models; record once per (h, period, subset)
    pers_recorded = set()

    for h in HORIZONS:
        log_progress(f"START: processing horizon h={h}")
        t_h = time.time()

        # Validation set for 2018-2019
        d_val = target_1819_idx - h
        y_val = upw[target_1819_idx].reshape(-1)

        # Common test targets and persistence flags for 2020-2025
        d_test = target_test_idx - h
        y_test = upw[target_test_idx]
        flag_d_test = upw[d_test]

        train_stride_main = 4
        d_train_main = np.arange(3, idx_2017_end + 1 - h, train_stride_main)
        try:
            X_train_main = build_features(data, d_train_main, log_tag=f"Main train h={h} (stride={train_stride_main})")
            y_train_main = upw[d_train_main + h].reshape(-1)
            log_progress(f"START: training Main h={h} HistGradientBoostingClassifier (max_iter=200, rows={len(X_train_main):,})")
            clf_main = HistGradientBoostingClassifier(max_iter=200, random_state=42)
            clf_main.fit(X_train_main, y_train_main)
            log_progress(f"END: training Main h={h} HistGradientBoostingClassifier complete")
        except MemoryError:
            log_progress(f"WARNING: MemoryError encountered at h={h} with stride={train_stride_main}. Falling back to every 8th day.")
            train_stride_main = 8
            gc.collect()
            d_train_main = np.arange(3, idx_2017_end + 1 - h, train_stride_main)
            X_train_main = build_features(data, d_train_main, log_tag=f"Main train h={h} fallback (stride=8)")
            y_train_main = upw[d_train_main + h].reshape(-1)
            log_progress(f"START: training Main h={h} HistGradientBoostingClassifier with stride=8 (rows={len(X_train_main):,})")
            clf_main = HistGradientBoostingClassifier(max_iter=200, random_state=42)
            clf_main.fit(X_train_main, y_train_main)
            log_progress(f"END: training Main h={h} HistGradientBoostingClassifier (stride=8) complete")

        del X_train_main, y_train_main
        gc.collect()

        log_progress(f"START: evaluating Main h={h} validation (2018-2019) and tuning F1 cutoff")
        p_val_main = predict_chunked(clf_main, data, d_val, chunk_days=100)
        cutoff_main, val_f1_main = select_best_f1_cutoff(y_val, p_val_main)
        del p_val_main
        log_progress(f"END: evaluating Main h={h} validation complete (cutoff={cutoff_main:.4f}, val_F1={val_f1_main:.4f})")

        log_progress(f"START: evaluating Main h={h} permutation importance (50k rows, 2018-2019)")
        rng = np.random.default_rng(42)
        sample_d_idx = rng.choice(d_val, size=min(len(d_val), 15), replace=False)
        X_val_sample = build_features(data, sample_d_idx)
        y_val_sample = upw[sample_d_idx + h].reshape(-1)
        sample_rows = rng.choice(len(y_val_sample), size=50_000, replace=False)
        X_imp = X_val_sample[sample_rows]
        y_imp = y_val_sample[sample_rows]
        del X_val_sample, y_val_sample
        gc.collect()

        def f1_scorer_main(est, X_s, y_s, thr=cutoff_main) -> float:
            preds = (est.predict_proba(X_s)[:, 1] >= thr).astype(np.int8)
            return float(f1_score(y_s, preds, zero_division=0))

        perm_main = permutation_importance(clf_main, X_imp, y_imp, scoring=f1_scorer_main, n_repeats=5, random_state=42, n_jobs=1)
        for rank_i, feat_i in enumerate(np.argsort(perm_main.importances_mean)[::-1], start=1):
            imp_rows.append(
                {
                    "model": "Main (2010-2017)",
                    "horizon_days": h,
                    "rank": rank_i,
                    "feature": FEATURE_NAMES[feat_i],
                    "importance_mean": float(perm_main.importances_mean[feat_i]),
                    "importance_std": float(perm_main.importances_std[feat_i]),
                }
            )
        del X_imp, y_imp
        gc.collect()
        log_progress(f"END: evaluating Main h={h} permutation importance complete")

        train_stride_ctrl = 2
        d_train_ctrl = np.arange(idx_2017_start, idx_2017_end + 1 - h, train_stride_ctrl)
        X_train_ctrl = build_features(data, d_train_ctrl, log_tag=f"Control train h={h} (stride={train_stride_ctrl})")
        y_train_ctrl = upw[d_train_ctrl + h].reshape(-1)
        log_progress(f"START: training Control h={h} HistGradientBoostingClassifier (max_iter=200, rows={len(X_train_ctrl):,})")
        clf_ctrl = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        clf_ctrl.fit(X_train_ctrl, y_train_ctrl)
        log_progress(f"END: training Control h={h} HistGradientBoostingClassifier complete")

        del X_train_ctrl, y_train_ctrl
        gc.collect()

        log_progress(f"START: evaluating Control h={h} validation (2018-2019) and tuning F1 cutoff")
        p_val_ctrl = predict_chunked(clf_ctrl, data, d_val, chunk_days=100)
        cutoff_ctrl, val_f1_ctrl = select_best_f1_cutoff(y_val, p_val_ctrl)
        del p_val_ctrl
        log_progress(f"END: evaluating Control h={h} validation complete (cutoff={cutoff_ctrl:.4f}, val_F1={val_f1_ctrl:.4f})")

        log_progress(f"START: evaluating Control h={h} permutation importance (50k rows, 2018-2019)")
        sample_d_idx_ctrl = rng.choice(d_val, size=min(len(d_val), 15), replace=False)
        X_val_sample_ctrl = build_features(data, sample_d_idx_ctrl)
        y_val_sample_ctrl = upw[sample_d_idx_ctrl + h].reshape(-1)
        sample_rows_ctrl = rng.choice(len(y_val_sample_ctrl), size=50_000, replace=False)
        X_imp_ctrl = X_val_sample_ctrl[sample_rows_ctrl]
        y_imp_ctrl = y_val_sample_ctrl[sample_rows_ctrl]
        del X_val_sample_ctrl, y_val_sample_ctrl
        gc.collect()

        def f1_scorer_ctrl(est, X_s, y_s, thr=cutoff_ctrl) -> float:
            preds = (est.predict_proba(X_s)[:, 1] >= thr).astype(np.int8)
            return float(f1_score(y_s, preds, zero_division=0))

        perm_ctrl = permutation_importance(clf_ctrl, X_imp_ctrl, y_imp_ctrl, scoring=f1_scorer_ctrl, n_repeats=5, random_state=42, n_jobs=1)
        for rank_i, feat_i in enumerate(np.argsort(perm_ctrl.importances_mean)[::-1], start=1):
            imp_rows.append(
                {
                    "model": "Control (2017)",
                    "horizon_days": h,
                    "rank": rank_i,
                    "feature": FEATURE_NAMES[feat_i],
                    "importance_mean": float(perm_ctrl.importances_mean[feat_i]),
                    "importance_std": float(perm_ctrl.importances_std[feat_i]),
                }
            )
        del X_imp_ctrl, y_imp_ctrl
        gc.collect()
        log_progress(f"END: evaluating Control h={h} permutation importance complete")

        log_progress(f"START: predicting test set (2020-2025) for h={h}")
        probs_main = predict_chunked(clf_main, data, d_test, chunk_days=100).reshape(len(target_test_idx), n_sub)
        preds_main = (probs_main >= cutoff_main).astype(np.int8)
        del probs_main

        probs_ctrl = predict_chunked(clf_ctrl, data, d_test, chunk_days=100).reshape(len(target_test_idx), n_sub)
        preds_ctrl = (probs_ctrl >= cutoff_ctrl).astype(np.int8)
        del probs_ctrl
        log_progress(f"END: predicting test set for h={h} complete")

        for period_name, p_mask in PERIOD_DEFS:
            for subset_name in ("Jun-Sep", "All days"):
                if subset_name == "Jun-Sep":
                    m_mask = np.isin(test_dates.month, [6, 7, 8, 9])
                else:
                    m_mask = np.ones(len(test_dates), dtype=bool)

                mask = p_mask & m_mask
                yt = y_test[mask].reshape(-1)
                yp_main = preds_main[mask].reshape(-1)
                yp_ctrl = preds_ctrl[mask].reshape(-1)
                yp_pers = flag_d_test[mask].reshape(-1)

                m_res_main = eval_transition_metrics(yt, yp_main, yp_pers)
                m_res_ctrl = eval_transition_metrics(yt, yp_ctrl, yp_pers)
                m_res_pers = eval_transition_metrics(yt, yp_pers, yp_pers)

                pers_key = (h, period_name, subset_name)
                if pers_key not in pers_recorded:
                    pers_recorded.add(pers_key)
                    metric_rows.append(
                        {
                            "model": "Persistence",
                            "horizon_days": h,
                            "period": period_name,
                            "subset": subset_name,
                            "train_years": "N/A",
                            "train_stride": "N/A",
                            "cutoff": np.nan,
                            "val_1819_f1": np.nan,
                            **m_res_pers,
                        }
                    )

                metric_rows.append(
                    {
                        "model": "Main (2010-2017)",
                        "horizon_days": h,
                        "period": period_name,
                        "subset": subset_name,
                        "train_years": "2010-2017",
                        "train_stride": train_stride_main,
                        "cutoff": cutoff_main,
                        "val_1819_f1": val_f1_main,
                        **m_res_main,
                    }
                )

                metric_rows.append(
                    {
                        "model": "Control (2017)",
                        "horizon_days": h,
                        "period": period_name,
                        "subset": subset_name,
                        "train_years": "2017",
                        "train_stride": train_stride_ctrl,
                        "cutoff": cutoff_ctrl,
                        "val_1819_f1": val_f1_ctrl,
                        **m_res_ctrl,
                    }
                )

        del preds_main, preds_ctrl, clf_main, clf_ctrl
        gc.collect()
        log_progress(f"END: processing horizon h={h} complete in {time.time() - t_h:.1f}s")

    metrics_df = pd.DataFrame(metric_rows)
    imp_df = pd.DataFrame(imp_rows)

    metrics_df.to_csv(BASE_DIR / "forecast_metrics.csv", index=False)
    imp_df.to_csv(BASE_DIR / "permutation_importance_2018_2019.csv", index=False)

    log_progress("\n" + "=" * 80)
    log_progress("=== FORECAST TEST METRICS: MODEL VS PERSISTENCE ===")
    log_progress("=" * 80)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== TOP 10 PERMUTATION IMPORTANCE FEATURES (2018-2019, 50K RANDOM ROWS) ===")
    log_progress("=" * 80)
    top10 = imp_df[imp_df["rank"] <= 10]
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)

    log_progress("\n" + "=" * 80)
    log_progress("=== PASS / FAIL (2020-2025 Jun-Sep: MAIN F1 - PERSISTENCE F1 >= 0.02) ===")
    log_progress("=" * 80)
    for h in (3, 5):
        sub_main = metrics_df[
            (metrics_df["horizon_days"] == h)
            & (metrics_df["period"] == "2020-2025")
            & (metrics_df["subset"] == "Jun-Sep")
            & (metrics_df["model"] == "Main (2010-2017)")
        ]
        sub_pers = metrics_df[
            (metrics_df["horizon_days"] == h)
            & (metrics_df["period"] == "2020-2025")
            & (metrics_df["subset"] == "Jun-Sep")
            & (metrics_df["model"] == "Persistence")
        ]
        sub_ctrl = metrics_df[
            (metrics_df["horizon_days"] == h)
            & (metrics_df["period"] == "2020-2025")
            & (metrics_df["subset"] == "Jun-Sep")
            & (metrics_df["model"] == "Control (2017)")
        ]

        f1_mod = float(sub_main["f1"].iloc[0])
        f1_pers = float(sub_pers["f1"].iloc[0])
        f1_ctrl = float(sub_ctrl["f1"].iloc[0])

        diff_mod = f1_mod - f1_pers
        diff_ctrl = f1_ctrl - f1_pers
        status = "PASS" if diff_mod >= 0.02 else "FAIL"

        log_progress(
            f"Horizon h={h} (2020-2025 Jun-Sep): {status} | "
            f"Main F1={f1_mod:.4f}, Persistence={f1_pers:.4f}, Diff={diff_mod:+.4f} (req >= +0.02) | "
            f"Control F1={f1_ctrl:.4f}, Control Diff={diff_ctrl:+.4f}"
        )

    log_progress(f"END: entire pipeline completed in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
