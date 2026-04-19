#!/usr/bin/env python3
"""
RH20T benchmark-v0 scoring entrypoint.
"""

from __future__ import annotations

from pathlib import Path

try:
    from . import score_pilot
except ImportError:
    import score_pilot


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_JSONL = BENCHMARK_DIR / "eval_results_v1" / "rh20t_benchmark_v0_qwen25vl32b_full.jsonl"


score_pilot.DEFAULT_RESULTS_JSONL = str(DEFAULT_RESULTS_JSONL)
score_pilot.BASELINES = dict(score_pilot.BASELINES)
score_pilot.BASELINES.update(
    {
        "T1": 0.25,
        "T2": 0.50,
        "T3": 0.25,
        "T7": 0.50,
        "T6": 0.50,
        "T_temporal": 1.0 / 6.0,
        "T_binary": 0.50,
        "T_progress": 1.0 / 3.0,
    }
)


if __name__ == "__main__":
    score_pilot.main()
