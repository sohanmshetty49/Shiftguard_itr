# ShiftGuard

**Event-Aware Distribution Shift Detection, Attribution, and Human-in-the-Loop Adaptation for Non-Stationary Forex Markets**

CS 6140 - Machine Learning  
Northeastern University, Silicon Valley

Team:
- Anusha Ravi Kumar - ravikumar.anu@northeastern.edu
- Sohan Mahesh - mahesh.so@northeastern.edu
- Disha Manubhai Patel - patel.dishabe@northeastern.edu

## Overview

ShiftGuard is an ML systems project for handling **distribution shift in forex markets**. The central idea is that forex behavior changes around scheduled macro events, unexpected geopolitical shocks, volatility dislocations, and changing market structure. Instead of treating forecasting as the whole problem, ShiftGuard focuses on the pipeline around a monitored model:

1. train a monitored return model,
2. detect scheduled and unexpected shifts,
3. explain shifts with SHAP-based attribution,
4. materialize approved decisions automatically in the reported regeneration path while keeping a human override layer,
5. selectively retrain after approved shifts,
6. adapt trading posture based on the detected shift context.

The project is therefore about **detecting, explaining, and adapting to non-stationarity**, not about building a pure trading bot.

## Canonical Project Goal

ShiftGuard asks:

- Can we detect distribution shifts in forex markets using event-aware and statistical methods?
- Can we explain those shifts using feature-group attribution?
- Can human review improve the quality of adaptation decisions while staying lightweight?
- Does selective retraining recover model performance better than no retraining or blind retraining?
- Can shift type and dominant attribution group drive a better downstream response than a single static policy?

## Canonical Pipeline

```text
Raw data (price + macro + sentiment + events)
    ->
Feature engineering
    technical / volatility / macro / sentiment / regime
    ->
Monitored model
    XGBoost regressor on next-bar return
    ->
Shift detection
    scheduled = calendar + KS/MMD
    unexpected = anomaly detection
    performance = drift on prediction errors
    ->
SHAP attribution
    what feature group drove the shift?
    ->
Dashboard
    reported regeneration path auto-materializes approved decisions
    dashboard still supports human override / reject / relabel
    ->
Adaptive response
    trading posture selection
    + retraining policy selection
    ->
Selective retraining
    full / window / weighted / adaptive
```

## Why XGBoost Regressor Is Canonical

The submission-facing monitored model is `XGBRegressor`, not `XGBClassifier`.

This choice is deliberate:

- the project needs a continuous error stream for drift monitoring,
- rolling MAE and recovery analysis are central evaluation tools,
- SHAP attribution is straightforward and fast,
- retraining experiments are easier to interpret with a return target.

Classification and trading-oriented experiments may still exist locally as supporting work, but they are not the canonical story of the project.

## Current System Behavior

This repository currently supports:

- a connected end-to-end pipeline from model outputs to detection, attribution, dashboard review, and retraining
- a Streamlit dashboard for lightweight human review and override
- a submission-facing regeneration path that auto-materializes approved decisions for reproducible reported results
- selective retraining that consumes approved dashboard decisions
- an **adaptive ShiftGuard trading policy** used by the trading comparison path
  - `technical` shift -> technical followthrough posture
  - `scheduled + macro/sentiment` shift -> event-aligned posture
  - `unexpected / volatility` shift -> defensive shock posture
- adaptive retraining logic in `src/retraining/selective.py`, where retrain policy can depend on shift type and dominant attribution group

This means attribution is no longer only descriptive; it is beginning to control downstream behavior.

## Data

The repo currently uses:

- 3 instruments: `EURUSD`, `GBPJPY`, `XAUUSD`
- multi-year forex OHLCV data
- macroeconomic calendar data
- macro time series
- sentiment / cross-asset context
- manually curated event context

Processed datasets are stored in `data/processed/` and raw inputs live in `data/raw/`.

## Feature Groups

The canonical submission path uses feature groups intended to capture multiple kinds of market change:

- `technical`
- `volatility`
- `macro`
- `sentiment`

Some legacy experiments also introduced regime-oriented or session-style features, but those are not required to understand the main ShiftGuard submission flow.

## Baseline Setup

The current project-facing comparison structure is:

- **Strong technical baseline**
  - rule-based trading benchmark
  - implemented in `src/models/baseline_technical.py`
  - intended to be stronger than a minimal RSI/MACD toy rule and act as the main hand-crafted benchmark

- **ML direction baseline**
  - standard supervised benchmark
  - implemented in `src/models/baseline_ml_direction.py`
  - used to show what a plain predictive learner can do without explicit shift-aware control

- **Advanced supplementary baseline**
  - stacked ensemble in `src/models/baseline_stacked.py`
  - current structure: `RandomForest + BiLSTM-Attention -> LightGBM meta-learner`
  - included to test whether additional model complexity alone solves non-stationarity
  - run separately from the canonical submission pipeline

- **Main system**
  - ShiftGuard
  - shift detection + attribution + dashboard review + adaptive response

## Submission-Facing Files

Canonical local pipeline:

- `src/models/main_xgboost.py`
- `src/detection/engine.py`
- `src/attribution/shap_analysis.py`
- `src/dashboard/app.py`
- `src/retraining/selective.py`
- `src/run_pipeline.py`
- `src/models/baseline_technical.py`
- `src/models/baseline_ml_direction.py`
- `src/models/winrate_experiment.py`
- `src/run_all_phases.py`
- `src/analysis/generate_figures.py`

Core feature/data pipeline:

- `src/features/build_dataset.py`
- `src/features/technical.py`
- `src/features/volatility.py`
- `src/features/macro.py`
- `src/features/sentiment.py`

## How To Run Locally

### Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Full reviewer-facing regeneration

```bash
python src/run_all_phases.py --clean
```

This command:

1. rebuilds the processed feature matrices,
2. reruns the monitored model, detection, and attribution pipeline,
3. materializes the dashboard's default auto-confirm decisions from detection outputs,
4. reruns selective retraining,
5. refreshes the adaptive win-rate experiment, and
6. regenerates the dashboard-facing statistical summary and figures in `results/figures/`.

The reported results in this repository were generated through this end-to-end wrapper. Concretely, `src/run_all_phases.py` calls `src/run_pipeline.py` with `--auto-confirm-decisions`, so detected shifts are materialized into approved decisions and then passed into selective retraining without manual intervention. The dashboard still supports manual review and override, but those interactive overrides were not used for the current benchmark results.

For a faster local validation run that still exercises the full flow end to end:

```bash
python src/run_all_phases.py --clean --fast
```

### Canonical monitored-model pipeline only

```bash
python src/run_pipeline.py
```

This runs:

1. `src/models/main_xgboost.py`
2. `src/detection/engine.py`
3. `src/attribution/shap_analysis.py`
4. `src/retraining/selective.py` if approved decisions already exist

If you want the pipeline to materialize the default dashboard decisions and continue straight into retraining without opening the UI:

```bash
python src/run_pipeline.py --pairs EURUSD GBPJPY XAUUSD --auto-confirm-decisions
```

The supplementary stacked baseline is not part of `src/run_pipeline.py`; run it separately when you want the advanced complexity comparison:

```bash
python src/models/baseline_stacked.py
```

By default, the pipeline pauses for review after attribution. If you already have approved decision files for the selected pairs and want to continue into retraining immediately, rerun with `--use-existing-decisions`.

If no review decisions are present yet, the pipeline pauses and expects:

```bash
streamlit run src/dashboard/app.py
```

After reviewing or overriding shifts in the dashboard, rerun:

```bash
python src/retraining/selective.py --pairs EURUSD GBPJPY XAUUSD
```

### Step-by-step local flow

```bash
python src/features/build_dataset.py
python src/models/main_xgboost.py
python src/detection/engine.py
python src/attribution/shap_analysis.py
streamlit run src/dashboard/app.py
python src/retraining/selective.py
```

For a quicker local validation run on one pair:

```bash
python src/models/main_xgboost.py --pairs EURUSD --fast
```

For the stricter walk-forward robustness experiment used to compare rolling refits against the current single-fit holdout benchmark:

```bash
python src/analysis/walkforward_report.py --pairs EURUSD GBPJPY XAUUSD
```

This writes comparison tables to `results/walkforward/` and figures to `results/figures/`.

## Performance Profiling

Profiling the reported pipeline with `cProfile` (`scripts/profile_retraining.py`) showed that
`src/retraining/selective.py`'s selective-retraining step was the most expensive part of a full
run, and that **>98% of its wall-clock time was spent inside XGBoost's own `fit`/`train`/`update`
internals**, not in the surrounding pandas/numpy bookkeeping. Concretely, for every detected shift
the original code retrained XGBoost from scratch under 5 policies (no retrain / full / window /
weighted / adaptive), one shift at a time, even though each shift's retraining work is completely
independent of every other shift's.

**Fix:** the per-shift work was factored into `_process_one_shift` and is now run in parallel
across shifts with `joblib.Parallel` (`prefer="processes"`, since XGBoost's internal thread pools
oversubscribe with a `threading` backend). On an 8-core machine this cut EURUSD's retraining
wall-clock time from **146.4s to 88.2s (1.7x)**, with byte-for-byte identical MAE and recovery
results (verified by diffing `results/retraining/EURUSD_retraining_results.csv` before/after).
`run_retraining_experiment(pair, n_jobs=1)` reproduces the original sequential behavior for
debugging or re-profiling.

```bash
python scripts/profile_retraining.py --pair EURUSD
python scripts/profile_retraining.py --pair EURUSD --n-jobs 1  # compare against sequential
```

## Model Tuning

The monitored model's hyperparameters (`learning_rate`, `max_depth`, `n_estimators`, `reg_alpha`,
`reg_lambda`) are selected per pair via grid search with a 5-fold `TimeSeriesSplit` (see
`project_report.tex`, Section 3.3, "Monitored Model and Baselines"). This is classical **XGBoost
hyperparameter tuning** -- a grid search over a fixed set of boosting/regularization parameters --
not LLM fine-tuning; ShiftGuard doesn't train or fine-tune any language model. EURUSD, GBPJPY, and
XAUUSD each end up with different tuned configurations (e.g. EURUSD: depth 5 / lr 0.05; GBPJPY:
depth 3 / lr 0.01), which is evidence the search is actually running per pair rather than reusing
one copied configuration.

## Adaptive Logic

ShiftGuard is moving toward an **adaptive policy system**, not just a detector.

Current intended adaptation logic:

- `technical`-dominant shift
  - loosen participation only when technical structure agrees
  - prefer technical-followthrough trading posture
  - favor shorter, more local retraining windows

- `scheduled + macro/sentiment` shift
  - require event-aware alignment and stronger confirmation
  - favor broader retraining on a fuller historical slice

- `unexpected / volatility` shock
  - become more defensive in trading
  - favor weighted retraining with recent data emphasized

The goal is for **shift type + dominant cause** to determine both:

- how ShiftGuard trades
- how the monitored model retrains

## Current Results Snapshot

The snapshot below reflects a clean local regeneration from the current code path using:

```bash
python src/run_all_phases.py --clean
```

### Shift detection / attribution

Tracked cross-pair shift outputs in `results/detection/`:

- `EURUSD`: `26` scheduled, `5` unexpected, `31` total
- `GBPJPY`: `7` scheduled, `19` unexpected, `1` performance drift, `27` total
- `XAUUSD`: `27` scheduled, `23` unexpected, `50` total

Dominant attribution pattern from `results/attribution/`:

- `EURUSD`: `technical` on all `31/31` analyzed shifts
- `GBPJPY`: `sentiment` on all `27/27` analyzed shifts
- `XAUUSD`: `sentiment` on `34/50` shifts, `technical` on `16/50`

### Monitored model summary

Tracked test-set metrics from `results/predictions/xgboost_summary.json`:

- `EURUSD`: `MAE 0.001141`, directional accuracy `59.5%`
- `GBPJPY`: `MAE 0.001621`, directional accuracy `52.6%`
- `XAUUSD`: `MAE 0.003483`, directional accuracy `51.4%`

### Walk-forward robustness summary

Tracked walk-forward metrics from `results/walkforward/holdout_vs_walkforward.csv`:

- `EURUSD`: walk-forward `MAE 0.001122` vs holdout `0.001141`, directional accuracy `60.87%` vs `59.34%`
- `GBPJPY`: walk-forward `MAE 0.001636` vs holdout `0.001621`, directional accuracy `53.10%` vs `52.45%`
- `XAUUSD`: walk-forward `MAE 0.002632` vs holdout `0.003483`, directional accuracy `55.84%` vs `51.34%`

Headline takeaway:
walk-forward validation improves MAE on `2/3` pairs and directional accuracy on all `3/3` pairs, with the clearest gain on `XAUUSD`.

The generated figure `results/figures/walkforward_metric_comparison.png` compares static holdout versus walk-forward MAE and directional accuracy, while `results/figures/walkforward_rolling_mae.png` shows how the rolling error traces differ over time.

### Selective retraining summary

Tracked saved retraining summaries in `results/retraining/`:

- `EURUSD`
  - No retrain: `MAE 0.001180`, recovery `43.3` bars
  - Full retrain: `MAE 0.001173`, recovery `43.0` bars
  - Adaptive: `MAE 0.001320`, recovery `51.2` bars

- `GBPJPY`
  - No retrain: `MAE 0.001652`, recovery `70.7` bars
  - Full retrain: `MAE 0.001643`, recovery `70.7` bars
  - Adaptive: `MAE 0.001643`, recovery `70.7` bars

- `XAUUSD`
  - No retrain: `MAE 0.004181`, recovery `39.1` bars
  - Weighted: `MAE 0.004148`, recovery `54.9` bars
  - Adaptive: `MAE 0.004208`, recovery `39.1` bars

Across the refreshed run, retraining remains pair-dependent: `Full Retrain` is the most stable low-friction default on `EURUSD` and `GBPJPY`, while `Weighted` is the best MAE variant on `XAUUSD`.

### Adaptive trading summary

Tracked saved trading summaries in `results/winrate/`:

- `EURUSD`: ShiftGuard `58.45%` win rate, `28.1%` market participation, `2.14` profit factor
- `GBPJPY`: ShiftGuard `58.96%` win rate, `39.5%` market participation, `1.98` profit factor
- `XAUUSD`: ShiftGuard `58.03%` win rate, `40.6%` market participation, `1.89` profit factor

The saved figure `results/figures/winrate_comparison.png` now visualizes the risk managed comparison through market participation or trade percentage across strategies. Win rate remains part of the tabular summary, while the figure highlights that ShiftGuard trades more selectively than the always on ML baseline.

Paired statistical tests in `results/figures/statistical_tests.json` remain significant for all three pairs:

- `EURUSD`: `p = 0.003182`
- `GBPJPY`: `p = 0.000045`
- `XAUUSD`: `p < 0.000001`

## Evaluation Story

The strongest evaluation framing for this project is:

- **Detection quality**
  scheduled vs unexpected shift coverage, alert counts, event alignment

- **Attribution quality**
  dominant feature groups and interpretable shift explanations

- **Adaptation quality**
  rolling MAE, recovery time, and retraining efficiency

- **Human-in-the-loop value**
  auto-confirm keeps the workflow efficient, while overrides and rejections improve downstream adaptation quality

- **Adaptive policy quality**
  whether shift type and dominant attribution group improve trading posture and retraining choice, including more selective market participation

The main academic claim is not that ShiftGuard maximizes raw trading profit. The claim is that it provides a more structured and interpretable way to handle shift in a non-stationary financial environment.

## Reproducibility

- The canonical monitored model writes outputs to `results/predictions/`.
- The dashboard reads from `results/detection/`, `results/attribution/`, `results/predictions/`, and writes decisions to `results/decisions/`.
- The reported end-to-end results were regenerated with `src/run_all_phases.py`, which calls `src/run_pipeline.py` with `--auto-confirm-decisions`.
- In that reported path, detected shifts are materialized into approved decisions and passed into selective retraining without manual intervention.
- The standard pipeline still supports human review and manual override through the dashboard, but those interactive overrides were not used in the current reported benchmark configuration.
- Selective retraining consumes approved dashboard decisions when available.
- `src/models/winrate_experiment.py` contains the adaptive trading-policy implementation used by the trading comparison path.
- Analysis/reporting scripts live in `src/analysis/` and tracked result artifacts live under `results/`.
- The reviewer-facing regenerated set is:
  `results/predictions/xgboost_*`,
  `results/detection/*`,
  `results/attribution/*`,
  `results/decisions/*`,
  `results/retraining/*_retraining_*`,
  `results/winrate/*_winrate_*`,
  and `results/figures/*`.

To regenerate the tracked outputs from the canonical path:

```bash
python src/run_all_phases.py --clean --fast
```

## Tests and CI

- Unit tests live in `tests/`.
- GitHub Actions runs the test suite on every push and pull request.
- CI sets `MPLBACKEND=Agg` and `MPLCONFIGDIR=/tmp/matplotlib` so Matplotlib-backed imports stay stable in headless environments.
- The tests currently cover:
  - auto-confirm decision materialization from detection outputs,
  - review gating for the pipeline,
  - prediction-window filtering for detected shifts,
  - SHAP group percentages,
  - reviewer-reject handling in selective retraining.

## References

1. Gama, J. et al. (2014). A survey on concept drift adaptation.
2. Lu, J. et al. (2019). Learning under concept drift: A review.
3. Ganin, Y. et al. (2016). Domain-adversarial training of neural networks.
4. Monarch, R. (2021). Human-in-the-Loop Machine Learning.
5. Amershi, S. et al. (2014). Power to the people.
6. Lundberg, S. and Lee, S. (2017). A unified approach to interpreting model predictions.
