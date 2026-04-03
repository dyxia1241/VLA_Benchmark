#!/usr/bin/env python3
"""
Full benchmark-v1 evaluation entrypoint.

This wrapper reuses GM-100/benchmark/eval_v1/run_pilot_eval.py logic,
while switching defaults from pilot to full benchmark-v1 paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    from . import run_pilot_eval
except ImportError:
    import run_pilot_eval


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSONL = (
    BENCHMARK_DIR
    / "benchmark_v1_curated.jsonl"
)
DEFAULT_OUTPUT_JSONL = (
    BENCHMARK_DIR
    / "eval_results_v1"
    / "benchmark_v1_qwen25vl32b_full.jsonl"
)
DEFAULT_FRAME_DIR = BENCHMARK_DIR / "benchmark_v1_frames_tbinary_20260330"


# Full benchmark-v1 defaults.
run_pilot_eval.DEFAULT_INPUT_JSONL = str(DEFAULT_INPUT_JSONL)
run_pilot_eval.DEFAULT_OUTPUT_JSONL = str(DEFAULT_OUTPUT_JSONL)
run_pilot_eval.DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
run_pilot_eval.DEFAULT_CONCURRENCY = 16

# Use the latest root-release frame directory (includes T_binary v2 coverage).
run_pilot_eval.DEFAULT_FRAME_DIR = str(DEFAULT_FRAME_DIR)
run_pilot_eval.DEFAULT_FRAME_DIR_T3 = str(DEFAULT_FRAME_DIR)
run_pilot_eval.DEFAULT_FRAME_DIR_T6 = str(DEFAULT_FRAME_DIR)


if __name__ == "__main__":
    args = run_pilot_eval.build_argparser().parse_args()
    asyncio.run(run_pilot_eval.run_eval(args))
