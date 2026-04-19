#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path


TASK_FILES = {
    "T1": "t1_gt_items.jsonl",
    "T2": "t2_gt_items.jsonl",
    "T6": "t6_gt_items.jsonl",
    "T7": "t7_gt_items.jsonl",
    "T_temporal": "t_temporal_gt_items.jsonl",
    "T_binary": "t_binary_gt_items.jsonl",
    "T_progress": "t_progress_gt_items.jsonl",
    "T10": "t10_gt_items.jsonl",
    "T11": "t11_gt_items.jsonl",
    "T12": "t12_gt_items.jsonl",
}
TASK_ALIASES = {"T5": "T_progress", "T8": "T_temporal", "T9": "T_binary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a small REASSEMBLE pilot subset from per-task GT pools.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--tasks", default="T1,T2,T5,T6,T7,T8,T9,T10,T11,T12")
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--max-per-recording", type=int, default=1)
    parser.add_argument("--max-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_tasks(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(text).split(","):
        item = token.strip()
        if not item:
            continue
        item = TASK_ALIASES.get(item, item)
        if item not in TASK_FILES:
            raise ValueError(f"Unsupported task: {item}")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def item_group_key(item: dict) -> str:
    task_type = str(item.get("task_type", ""))
    recording_id = str(item.get("recording_id") or item.get("task_id") or "")
    segment_index = item.get("segment_index", "")
    low_index = item.get("low_index", "")
    masked_low_index = item.get("masked_low_index", "")
    interval_id = str(item.get("interval_id", "")).strip()

    if task_type == "T_progress" and interval_id:
        return f"{task_type}|{interval_id}"
    if task_type == "T7":
        return f"{task_type}|{recording_id}|seg={segment_index}"
    if task_type in {"T10", "T11"}:
        return f"{task_type}|{recording_id}|seg={segment_index}|low={low_index}"
    if task_type == "T12":
        return f"{task_type}|{recording_id}|seg={segment_index}|masked={masked_low_index}"
    if task_type in {"T_temporal", "T_binary"}:
        return f"{task_type}|{recording_id}|frames={','.join(str(int(x)) for x in item.get('frame_indices', []) or [])}"
    if task_type == "T1":
        return f"{task_type}|{recording_id}|phase={item.get('phase_source', '')}|frame={item.get('frame_index', '')}"
    if task_type == "T2":
        return f"{task_type}|{recording_id}|label={item.get('label', '')}|frame={item.get('frame_index', '')}"
    if task_type == "T6":
        return f"{task_type}|{recording_id}|label={item.get('speed_level_label', '')}|frame={item.get('frame_index', '')}"
    return f"{task_type}|{recording_id}|frame={item.get('frame_index', '')}"


def sample_diverse_rows(
    rows: list[dict],
    *,
    target: int,
    max_per_recording: int,
    max_per_group: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    if target <= 0 or not rows:
        return [], {
            "available": len(rows),
            "sampled": 0,
            "unique_recordings": 0,
            "recording_cap_relaxed": False,
            "group_cap_relaxed": False,
        }

    pool = list(rows)
    rng.shuffle(pool)

    def take(
        candidates: list[dict],
        *,
        rec_cap: int,
        group_cap: int,
    ) -> tuple[list[dict], Counter[str], Counter[str]]:
        selected: list[dict] = []
        per_recording: Counter[str] = Counter()
        per_group: Counter[str] = Counter()
        for item in candidates:
            recording_id = str(item.get("recording_id") or item.get("task_id") or "")
            group_key = item_group_key(item)
            if rec_cap > 0 and per_recording[recording_id] >= rec_cap:
                continue
            if group_cap > 0 and per_group[group_key] >= group_cap:
                continue
            selected.append(item)
            per_recording[recording_id] += 1
            per_group[group_key] += 1
            if len(selected) >= target:
                break
        return selected, per_recording, per_group

    selected, per_recording, _ = take(pool, rec_cap=max_per_recording, group_cap=max_per_group)
    recording_cap_relaxed = False
    group_cap_relaxed = False

    if len(selected) < target and max_per_group > 0:
        group_cap_relaxed = True
        selected, per_recording, _ = take(pool, rec_cap=max_per_recording, group_cap=0)

    if len(selected) < target and max_per_recording > 0:
        recording_cap_relaxed = True
        selected, per_recording, _ = take(pool, rec_cap=0, group_cap=0)

    return selected, {
        "available": len(rows),
        "sampled": len(selected),
        "unique_recordings": len(per_recording),
        "recording_cap_relaxed": recording_cap_relaxed,
        "group_cap_relaxed": group_cap_relaxed,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_summary_json) if args.output_summary_json else output_jsonl.with_suffix(".summary.json")

    tasks = normalize_tasks(args.tasks)
    combined: list[dict] = []
    summary = {
        "input_dir": str(input_dir),
        "output_jsonl": str(output_jsonl),
        "tasks": tasks,
        "per_task": int(args.per_task),
        "max_per_recording": int(args.max_per_recording),
        "max_per_group": int(args.max_per_group),
        "counts": {},
    }

    for task in tasks:
        rows = load_jsonl(input_dir / TASK_FILES[task])
        picked, task_summary = sample_diverse_rows(
            rows,
            target=args.per_task,
            max_per_recording=int(args.max_per_recording),
            max_per_group=int(args.max_per_group),
            rng=rng,
        )
        summary["counts"][task] = task_summary
        combined.extend(picked)

    with output_jsonl.open("w", encoding="utf-8") as fh:
        for row in combined:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
