"""
Phase 7: Selective Retraining
Compares retraining strategies after approved shifts:
  A) No retraining (baseline — model degrades)
  B) Full retrain (all historical data up to shift)
  C) Window retrain (last N bars only)
  D) Weighted retrain (exponential decay, upweight recent)
  E) Adaptive retrain (shift type + attribution driven)

Measures recovery: rolling MAE → how many bars until model returns to pre-shift error.

Usage:
    python src/retraining/selective.py
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'predictions')
DETECTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'detection')
RETRAINING_DIR = os.path.join(PROJECT_ROOT, 'results', 'retraining')
DECISIONS_DIR = os.path.join(PROJECT_ROOT, 'results', 'decisions')
ATTRIBUTION_DIR = os.path.join(PROJECT_ROOT, 'results', 'attribution')
os.makedirs(RETRAINING_DIR, exist_ok=True)

EXCLUDE_COLS = ['datetime_utc', 'date', 'session', 'target_return', 'target_direction', 'volume']

# XGBoost default params (from tuning results — use EURUSD best as default)
DEFAULT_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}
APPROVED_DECISIONS = {
    'confirm',
    'auto_confirm',
    'reclassify_to_scheduled',
    'reclassify_to_unexpected',
}

WINDOW_SIZE = 180  # 180 4H bars = 30 trading days
EVAL_HORIZON = 180  # evaluate recovery over 180 bars after retraining
ROLLING_WINDOW = 30  # rolling MAE window

GROUPS_PATH = os.path.join(PROCESSED_DIR, 'feature_groups.json')
with open(GROUPS_PATH) as f:
    FEATURE_GROUPS = json.load(f)


def load_pair_params(pair_name: str) -> dict[str, Any]:
    """Load tuned parameters saved by the canonical monitored-model run."""
    summary_path = os.path.join(RESULTS_DIR, 'xgboost_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        best_params = summary.get(pair_name, {}).get('best_params')
        if best_params:
            return {**DEFAULT_PARAMS, **best_params}

    tuning_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_tuning.csv')
    if os.path.exists(tuning_path):
        tuning_df = pd.read_csv(tuning_path)
        if not tuning_df.empty:
            top_row = tuning_df.sort_values('cv_mae').iloc[0]
            return {
                **DEFAULT_PARAMS,
                'learning_rate': float(top_row['learning_rate']),
                'max_depth': int(top_row['max_depth']),
                'n_estimators': int(top_row['n_estimators']),
                'reg_alpha': float(top_row['reg_alpha']),
                'reg_lambda': float(top_row['reg_lambda']),
            }

    return DEFAULT_PARAMS.copy()


def load_prediction_window(pair_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Infer the active evaluation window from the monitored-model predictions."""
    pred_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}_predictions.csv')
    if not os.path.exists(pred_path):
        return None
    pred_df = pd.read_csv(pred_path)
    if pred_df.empty or 'datetime_utc' not in pred_df.columns:
        return None
    pred_df['datetime_utc'] = pd.to_datetime(pred_df['datetime_utc'], errors='coerce')
    pred_df = pred_df.dropna(subset=['datetime_utc'])
    if pred_df.empty:
        return None
    return pred_df['datetime_utc'].min(), pred_df['datetime_utc'].max()


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_shift_events(pair_name: str) -> pd.DataFrame:
    """Load reviewed shifts when available, otherwise fall back to detected shifts."""
    shifts_path = os.path.join(DETECTION_DIR, f'{pair_name}_shifts.csv')
    shifts = pd.read_csv(shifts_path)
    shifts['datetime_utc'] = pd.to_datetime(shifts['datetime_utc'], format='mixed', errors='coerce')
    shifts = shifts.dropna(subset=['datetime_utc'])

    decisions_path = os.path.join(DECISIONS_DIR, f'{pair_name}_decisions.csv')
    if os.path.exists(decisions_path):
        decisions = pd.read_csv(decisions_path)
        if not decisions.empty:
            decisions['datetime_utc'] = pd.to_datetime(decisions['datetime_utc'], format='mixed', errors='coerce')
            decisions = decisions.dropna(subset=['datetime_utc'])
            decisions = decisions.sort_values('datetime_utc').drop_duplicates(subset=['datetime_utc'], keep='last')
            approved = decisions[decisions['decision'].isin(APPROVED_DECISIONS)].copy()
            if not approved.empty:
                shifts = shifts.merge(
                    approved[['datetime_utc', 'decision', 'notes']],
                    on='datetime_utc',
                    how='inner'
                )
                shifts.loc[shifts['decision'] == 'reclassify_to_scheduled', 'type'] = 'scheduled'
                shifts.loc[shifts['decision'] == 'reclassify_to_unexpected', 'type'] = 'unexpected'
            else:
                return shifts.iloc[0:0].copy()

    attr_path = os.path.join(ATTRIBUTION_DIR, f'{pair_name}_attribution.csv')
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        if not attr.empty:
            attr['datetime_utc'] = pd.to_datetime(attr['datetime_utc'], format='mixed', errors='coerce')
            attr = attr.dropna(subset=['datetime_utc'])
            shifts = shifts.merge(
                attr[['datetime_utc', 'dominant_group']].drop_duplicates('datetime_utc'),
                on='datetime_utc',
                how='left'
            )

    if 'dominant_group' not in shifts.columns:
        shifts['dominant_group'] = 'unknown'
    shifts['dominant_group'] = shifts['dominant_group'].fillna('unknown')

    prediction_window = load_prediction_window(pair_name)
    if prediction_window is not None:
        window_start, window_end = prediction_window
        shifts = shifts[(shifts['datetime_utc'] >= window_start) & (shifts['datetime_utc'] <= window_end)]

    # Only high severity (3+) to keep it manageable
    if 'severity' in shifts.columns:
        shifts = shifts[shifts['severity'] >= 3]

    # Deduplicate: keep one shift per week (avoid overlapping retraining windows)
    shifts = shifts.sort_values('datetime_utc')
    filtered = []
    last_dt = None
    for _, row in shifts.iterrows():
        if last_dt is None or (row['datetime_utc'] - last_dt).days >= 7:
            filtered.append(row)
            last_dt = row['datetime_utc']

    return pd.DataFrame(filtered)


def get_group_feature_cols(feature_cols: list[str], groups: list[str]) -> list[str]:
    wanted = set()
    for group in groups:
        wanted.update(FEATURE_GROUPS.get(group, []))
    selected = [c for c in feature_cols if c in wanted]
    return selected if selected else feature_cols


def choose_adaptive_policy(shift_row: pd.Series, feature_cols: list[str]) -> dict[str, Any]:
    shift_type = str(shift_row.get('type', 'unknown'))
    dominant = str(shift_row.get('dominant_group', 'unknown'))

    if dominant == 'technical':
        return {
            'policy': 'technical_window',
            'mode': 'window',
            'feature_cols': get_group_feature_cols(feature_cols, ['technical', 'volatility']),
            'window_size': 90,
        }
    if shift_type == 'scheduled' or dominant in {'macro', 'sentiment'}:
        return {
            'policy': 'event_full',
            'mode': 'full',
            'feature_cols': feature_cols,
        }
    if shift_type == 'unexpected' or dominant == 'volatility':
        return {
            'policy': 'shock_weighted',
            'mode': 'weighted',
            'feature_cols': get_group_feature_cols(feature_cols, ['volatility', 'technical', 'sentiment']),
            'decay': 0.99,
        }
    return {
        'policy': 'default_full',
        'mode': 'full',
        'feature_cols': feature_cols,
    }


def retrain_no_update(
    model: xgb.XGBRegressor,
    X_test_chunk: np.ndarray,
    y_test_chunk: np.ndarray,
) -> np.ndarray:
    """Strategy A: No retraining — use existing model as-is."""
    return model.predict(X_test_chunk)


def retrain_full(
    df: pd.DataFrame,
    feature_cols: list[str],
    shift_idx: int,
    params: dict[str, Any],
) -> xgb.XGBRegressor:
    """Strategy B: Full retrain on all data up to shift point."""
    train_data = df.iloc[:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_window(
    df: pd.DataFrame,
    feature_cols: list[str],
    shift_idx: int,
    window_size: int,
    params: dict[str, Any],
) -> xgb.XGBRegressor:
    """Strategy C: Window retrain — only last N bars before shift."""
    start = max(0, shift_idx - window_size)
    train_data = df.iloc[start:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def retrain_weighted(
    df: pd.DataFrame,
    feature_cols: list[str],
    shift_idx: int,
    params: dict[str, Any],
    decay: float = 0.995,
) -> xgb.XGBRegressor:
    """Strategy D: Weighted retrain — exponential decay, recent data upweighted."""
    train_data = df.iloc[:shift_idx]
    X = train_data[feature_cols].values
    y = train_data['target_return'].values

    # Exponential weights: most recent = 1.0, decays backwards
    n = len(X)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


def retrain_adaptive(
    df: pd.DataFrame,
    feature_cols: list[str],
    shift_idx: int,
    params: dict[str, Any],
    shift_row: pd.Series,
) -> tuple[xgb.XGBRegressor, list[str], str]:
    policy = choose_adaptive_policy(shift_row, feature_cols)
    selected_cols = policy['feature_cols']
    mode = policy['mode']

    if mode == 'window':
        model = retrain_window(df, selected_cols, shift_idx, policy.get('window_size', WINDOW_SIZE), params)
    elif mode == 'weighted':
        model = retrain_weighted(df, selected_cols, shift_idx, params, decay=policy.get('decay', 0.995))
    else:
        model = retrain_full(df, selected_cols, shift_idx, params)

    return model, selected_cols, policy['policy']


def compute_recovery_time(
    rolling_mae: pd.Series | np.ndarray,
    pre_shift_mae: float,
    threshold: float = 1.1,
) -> int:
    """
    How many bars until rolling MAE returns to within threshold × pre-shift level.
    Returns number of bars, or -1 if never recovers.
    """
    target = pre_shift_mae * threshold
    for i, val in enumerate(rolling_mae):
        if pd.notna(val) and val <= target:
            return i
    return -1


def _process_one_shift(
    df: pd.DataFrame,
    feature_cols: list[str],
    pair_params: dict[str, Any],
    base_model: xgb.XGBRegressor,
    shift_row: pd.Series,
) -> dict[str, Any] | None:
    """Run all 5 retraining strategies for a single shift.

    Profiling `run_retraining_experiment` with cProfile showed that >98% of
    wall-clock time is spent inside XGBoost's own `fit()` (retraining from
    scratch for the "full", "weighted", and "adaptive" strategies), not in
    the surrounding pandas/numpy bookkeeping. Each shift's work is otherwise
    independent of every other shift, so this function is factored out to be
    run in parallel across shifts (see `run_retraining_experiment`) instead
    of sequentially, which is the actual fix for the profiled hotspot.
    """
    shift_dt = shift_row['datetime_utc']

    # Find shift index in df
    time_diff = (df['datetime_utc'] - shift_dt).abs()
    shift_idx = time_diff.idxmin()

    # Evaluation window: EVAL_HORIZON bars after shift
    eval_end = min(len(df), shift_idx + EVAL_HORIZON)
    if eval_end - shift_idx < 30:
        return None

    eval_data = df.iloc[shift_idx:eval_end]
    X_eval = eval_data[feature_cols].values
    y_eval = eval_data['target_return'].values

    # Pre-shift baseline MAE (30 bars before shift)
    pre_start = max(0, shift_idx - ROLLING_WINDOW)
    pre_data = df.iloc[pre_start:shift_idx]
    X_pre = pre_data[feature_cols].values
    y_pre = pre_data['target_return'].values
    pre_pred = base_model.predict(X_pre)
    pre_shift_mae = mean_absolute_error(y_pre, pre_pred)

    # --- Strategy A: No retraining ---
    pred_none = base_model.predict(X_eval)
    mae_none = mean_absolute_error(y_eval, pred_none)
    rolling_none = pd.Series(np.abs(y_eval - pred_none)).rolling(ROLLING_WINDOW).mean()
    recovery_none = compute_recovery_time(rolling_none.values, pre_shift_mae)

    # --- Strategy B: Full retrain ---
    model_full = retrain_full(df, feature_cols, shift_idx, pair_params)
    pred_full = model_full.predict(X_eval)
    mae_full = mean_absolute_error(y_eval, pred_full)
    rolling_full = pd.Series(np.abs(y_eval - pred_full)).rolling(ROLLING_WINDOW).mean()
    recovery_full = compute_recovery_time(rolling_full.values, pre_shift_mae)

    # --- Strategy C: Window retrain (30 days) ---
    model_window = retrain_window(df, feature_cols, shift_idx, WINDOW_SIZE, pair_params)
    pred_window = model_window.predict(X_eval)
    mae_window = mean_absolute_error(y_eval, pred_window)
    rolling_window = pd.Series(np.abs(y_eval - pred_window)).rolling(ROLLING_WINDOW).mean()
    recovery_window = compute_recovery_time(rolling_window.values, pre_shift_mae)

    # --- Strategy D: Weighted retrain ---
    model_weighted = retrain_weighted(df, feature_cols, shift_idx, pair_params)
    pred_weighted = model_weighted.predict(X_eval)
    mae_weighted = mean_absolute_error(y_eval, pred_weighted)
    rolling_weighted = pd.Series(np.abs(y_eval - pred_weighted)).rolling(ROLLING_WINDOW).mean()
    recovery_weighted = compute_recovery_time(rolling_weighted.values, pre_shift_mae)

    # --- Strategy E: Adaptive retrain (uses shift type + attribution) ---
    model_adaptive, adaptive_cols, adaptive_policy = retrain_adaptive(
        df, feature_cols, shift_idx, pair_params, shift_row
    )
    pred_adaptive = model_adaptive.predict(eval_data[adaptive_cols].values)
    mae_adaptive = mean_absolute_error(y_eval, pred_adaptive)
    rolling_adaptive = pd.Series(np.abs(y_eval - pred_adaptive)).rolling(ROLLING_WINDOW).mean()
    recovery_adaptive = compute_recovery_time(rolling_adaptive.values, pre_shift_mae)

    return {
        'shift_datetime': str(shift_dt),
        'shift_type': shift_row.get('type', 'unknown'),
        'dominant_group': shift_row.get('dominant_group', 'unknown'),
        'adaptive_policy': adaptive_policy,
        'pre_shift_mae': round(pre_shift_mae, 6),
        # No retrain
        'mae_no_retrain': round(mae_none, 6),
        'recovery_no_retrain': recovery_none,
        # Full
        'mae_full_retrain': round(mae_full, 6),
        'recovery_full_retrain': recovery_full,
        # Window
        'mae_window_retrain': round(mae_window, 6),
        'recovery_window_retrain': recovery_window,
        # Weighted
        'mae_weighted_retrain': round(mae_weighted, 6),
        'recovery_weighted_retrain': recovery_weighted,
        # Adaptive
        'mae_adaptive_retrain': round(mae_adaptive, 6),
        'recovery_adaptive_retrain': recovery_adaptive,
    }


def run_retraining_experiment(pair_name: str, *, n_jobs: int = -1) -> pd.DataFrame | None:
    """Run all 5 strategies on detected shifts for one pair.

    `n_jobs` controls how many shifts are processed in parallel (passed to
    `joblib.Parallel`); `-1` uses all available cores, `1` reproduces the
    original sequential behavior (useful for debugging or profiling).
    """
    print(f"\n{'='*60}")
    print(f"Selective Retraining — {pair_name}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{pair_name}_features.csv'))
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    feature_cols = get_feature_cols(df)
    pair_params = load_pair_params(pair_name)

    # Load base model
    model_path = os.path.join(RESULTS_DIR, f'xgboost_{pair_name}.json')
    base_model = xgb.XGBRegressor()
    base_model.load_model(model_path)

    # Get shift events
    shifts = get_shift_events(pair_name)
    print(f"  Shifts to process: {len(shifts)}")

    if len(shifts) == 0:
        print("  No high-severity shifts in test period. Skipping.")
        return None

    # Each shift's 5-strategy retraining pass is independent of every other
    # shift, and XGBoost's own fit() dominates the per-shift cost (see
    # _process_one_shift docstring), so this is where parallelism actually
    # pays off instead of racing to optimize the pandas bookkeeping around it.
    shift_rows = [row for _, row in shifts.iterrows()]
    processed = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_process_one_shift)(df, feature_cols, pair_params, base_model, shift_row)
        for shift_row in shift_rows
    )
    all_results = [result for result in processed if result is not None]
    all_results.sort(key=lambda result: result['shift_datetime'])
    print(f"  Processed {len(all_results)}/{len(shifts)} shifts (n_jobs={n_jobs})")

    results_df = pd.DataFrame(all_results)

    # Summary
    print(f"\n  STRATEGY COMPARISON (mean across {len(results_df)} shifts):")
    print(f"  {'Strategy':<20s} {'MAE':<12s} {'Avg Recovery (bars)':<20s}")
    print(f"  {'-'*52}")

    strategies = [
        ('No Retrain', 'mae_no_retrain', 'recovery_no_retrain'),
        ('Full Retrain', 'mae_full_retrain', 'recovery_full_retrain'),
        ('Window (30d)', 'mae_window_retrain', 'recovery_window_retrain'),
        ('Weighted', 'mae_weighted_retrain', 'recovery_weighted_retrain'),
        ('Adaptive', 'mae_adaptive_retrain', 'recovery_adaptive_retrain'),
    ]

    summary = {}
    for name, mae_col, rec_col in strategies:
        avg_mae = results_df[mae_col].mean()
        valid_rec = results_df[results_df[rec_col] >= 0][rec_col]
        avg_rec = valid_rec.mean() if len(valid_rec) > 0 else -1
        pct_recovered = (results_df[rec_col] >= 0).mean() * 100
        print(f"  {name:<20s} {avg_mae:<12.6f} {avg_rec:<10.1f} ({pct_recovered:.0f}% recovered)")
        summary[name] = {
            'avg_mae': round(avg_mae, 6),
            'avg_recovery_bars': round(avg_rec, 1),
            'pct_recovered': round(pct_recovered, 1),
        }

    # Save
    out_path = os.path.join(RETRAINING_DIR, f'{pair_name}_retraining_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    json_path = os.path.join(RETRAINING_DIR, f'{pair_name}_retraining_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results_df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=['EURUSD', 'GBPJPY', 'XAUUSD'])
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-shift retraining (-1 = all cores, 1 = sequential/original behavior).",
    )
    args = parser.parse_args()

    all_summaries = {}
    for pair in args.pairs:
        result = run_retraining_experiment(pair, n_jobs=args.n_jobs)
        if result:
            _, summary = result
            all_summaries[pair] = summary

    # Overall summary
    print(f"\n{'='*60}")
    print("SELECTIVE RETRAINING — Overall Summary")
    print(f"{'='*60}")

    overall_path = os.path.join(RETRAINING_DIR, 'retraining_overall_summary.json')
    with open(overall_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved: {overall_path}")
