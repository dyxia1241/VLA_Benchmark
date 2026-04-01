#!/usr/bin/env python3
"""
Full benchmark-v1 scoring entrypoint.

This wrapper reuses GM-100/benchmark/eval_v1/score_pilot.py logic,
while switching defaults to full benchmark-v1 outputs and T6-binary baseline.
"""

from __future__ import annotations

from pathlib import Path

try:
    from . import score_pilot
except ImportError:
    import score_pilot


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_JSONL = (
    BENCHMARK_DIR
    / "eval_results_v1"
    / "benchmark_v1_qwen3vl8b_instruct_full.jsonl"
)


# Full benchmark-v1 default result path.
score_pilot.DEFAULT_RESULTS_JSONL = str(DEFAULT_RESULTS_JSONL)

# T6 is now binary in benchmark_v1: random baseline should be 0.5.
score_pilot.BASELINES = dict(score_pilot.BASELINES)
score_pilot.BASELINES["T6"] = 0.50


if __name__ == "__main__":
    score_pilot.main()
