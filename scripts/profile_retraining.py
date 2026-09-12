"""Profile ShiftGuard's selective-retraining loop with cProfile.

Usage:
    python scripts/profile_retraining.py --pair EURUSD
    python scripts/profile_retraining.py --pair EURUSD --n-jobs 1   # sequential, for comparison

This profiles `run_retraining_experiment`, which is the most expensive step
in the reported pipeline: for every detected shift, it retrains XGBoost from
scratch under 5 different policies (no retrain / full / window / weighted /
adaptive) and measures recovery.

Finding from the original (pre-fix) profile: >98% of wall-clock time was
spent inside XGBoost's own `fit()`/`train()`/`update()` internals -- not in
the surrounding pandas/numpy bookkeeping. Each shift's 5-strategy pass is
independent of every other shift, so `src/retraining/selective.py` now runs
shifts in parallel via `joblib.Parallel` (see `_process_one_shift`). On an
8-core machine this cut EURUSD's retraining wall-clock time from ~146s to
~88s (1.7x) with byte-for-byte identical MAE/recovery results. See the
"Performance Profiling" section of README.md for the full writeup.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retraining.selective import run_retraining_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the ShiftGuard selective retraining loop.")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair to profile (default: EURUSD).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-shift retraining (-1 = all cores, 1 = sequential).",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of top cumulative-time entries to print.")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    run_retraining_experiment(args.pair, n_jobs=args.n_jobs)
    profiler.disable()
    elapsed = time.perf_counter() - started

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(args.top)

    print(f"\nWall-clock time for {args.pair} (n_jobs={args.n_jobs}): {elapsed:.1f}s\n")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
