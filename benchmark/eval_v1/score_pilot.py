#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Centralized score/config entry points.
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_JSONL = str(BENCHMARK_DIR / "eval_results_v1" / "pilot_results_qwen3vl8b.jsonl")
DEFAULT_PASS_MARGIN = 0.10
DEFAULT_WARN_MARGIN = 0.03
DEFAULT_DEFAULT_BASELINE = 0.25
DEFAULT_IGNORE_INVALID = True

BASELINES = {
    "T1": 0.25,
    "T2": 0.50,
    "T3": 0.25,
    "T4": 0.25,
    "T5": 0.333,
    "T6": 0.333,
    "T8": 0.167,
    "T_temporal": 0.167,
    "T_binary": 0.50,
    "T_progress": 0.333,
}

INVALID_PRED = {"MISSING_FRAME", "ERROR", "INVALID"}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("results_jsonl", nargs="?", default=DEFAULT_RESULTS_JSONL)
    p.add_argument("--pass-margin", type=float, default=DEFAULT_PASS_MARGIN)
    p.add_argument("--warn-margin", type=float, default=DEFAULT_WARN_MARGIN)
    p.add_argument("--default-baseline", type=float, default=DEFAULT_DEFAULT_BASELINE)
    p.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Include INVALID/ERROR/MISSING_FRAME in denominator as incorrect.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ignore_invalid = DEFAULT_IGNORE_INVALID and (not args.keep_invalid)

    results = []
    with open(args.results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    by_type: dict[str, list[bool]] = defaultdict(list)
    by_arm: dict[str, list[bool]] = defaultdict(list)
    invalid_count = 0

    for r in results:
        pred = str(r.get("vlm_answer", ""))
        if ignore_invalid and pred in INVALID_PRED:
            invalid_count += 1
            continue
        t = str(r.get("task_type", "unknown"))
        arm = str(r.get("arm_type", "unknown"))
        correct = bool(r.get("correct", False))
        by_type[t].append(correct)
        by_arm[arm].append(correct)

    print(f"\n{'='*55}")
    print(f"{'Task':<14} {'Acc':>6} {'Random':>8} {'Delta':>8} {'N':>6}")
    print(f"{'-'*55}")
    for t in sorted(by_type):
        c = by_type[t]
        acc = sum(c) / len(c) if c else 0.0
        base = BASELINES.get(t, args.default_baseline)
        flag = " ✅" if acc > base + args.pass_margin else (" ⚠️" if acc < base + args.warn_margin else "")
        print(f"{t:<14} {acc:>6.3f} {base:>8.3f} {acc-base:>+8.3f} {len(c):>6}{flag}")

    print(f"\n{'='*55}")
    print(f"{'Arm type':<16} {'Acc':>6} {'N':>6}")
    print(f"{'-'*30}")
    for arm in sorted(by_arm):
        c = by_arm[arm]
        acc = sum(c) / len(c) if c else 0.0
        print(f"{arm:<16} {acc:>6.3f} {len(c):>6}")

    total = [
        bool(r.get("correct", False))
        for r in results
        if (not ignore_invalid) or (str(r.get("vlm_answer", "")) not in INVALID_PRED)
    ]
    if total:
        print(f"\nOverall: {sum(total)/len(total):.3f}  ({len(total)} valid items)")
    else:
        print("\nOverall: N/A (0 valid items)")
    if ignore_invalid:
        print(f"Ignored invalid predictions: {invalid_count}")


if __name__ == "__main__":
    main()
