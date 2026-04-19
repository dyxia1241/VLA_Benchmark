#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


TASK_FILES = {
    "T1": "t1_gt_items.jsonl",
    "T2": "t2_gt_items.jsonl",
    "T6": "t6_gt_items.jsonl",
    "T_temporal": "t_temporal_gt_items.jsonl",
    "T_binary": "t_binary_gt_items.jsonl",
    "T_progress": "t_progress_gt_items.jsonl",
}
TASK_ALIASES = {"T5": "T_progress", "T8": "T_temporal", "T9": "T_binary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a small RH20T pilot subset from per-task GT pools.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--tasks", default="T1,T2,T5,T6,T8,T9")
    parser.add_argument("--per-task", type=int, default=18)
    parser.add_argument("--max-per-scene", type=int, default=2)
    parser.add_argument("--max-per-group", type=int, default=1)
    parser.add_argument("--min-rating", type=int, default=3)
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


def resolve_task_file(input_dir: Path, task: str) -> Path:
    candidates = [
        input_dir / TASK_FILES[task],
        input_dir / f"{task}.jsonl",
        input_dir / f"{task.lower()}.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return input_dir / TASK_FILES[task]


def item_group_key(item: dict) -> str:
    task_type = str(item.get("task_type", ""))
    scene_dir = str(item.get("scene_dir") or item.get("recording_id") or item.get("task_id") or "")
    interval_id = str(item.get("interval_id", "")).strip()

    if task_type == "T_progress" and interval_id:
        return f"{task_type}|{interval_id}|{item.get('progress_bin', '')}"
    if task_type == "T_temporal":
        return f"{task_type}|{scene_dir}|{interval_id}"
    if task_type == "T_binary":
        return f"{task_type}|{scene_dir}|{item.get('difficulty', '')}|{','.join(str(int(x)) for x in item.get('frame_indices', []) or [])}"
    if task_type == "T1":
        return f"{task_type}|{scene_dir}|{item.get('phase_label', '')}|{item.get('frame_index', '')}"
    if task_type == "T2":
        return f"{task_type}|{scene_dir}|{item.get('label', '')}|{item.get('frame_index', '')}"
    if task_type == "T6":
        return f"{task_type}|{scene_dir}|{item.get('speed_level_label', '')}|{item.get('frame_index', '')}"
    return f"{task_type}|{scene_dir}|{item.get('frame_index', '')}"


def take_rows(
    candidates: list[dict],
    *,
    target: int,
    max_per_scene: int,
    max_per_group: int,
) -> tuple[list[dict], Counter[str], Counter[str]]:
    selected: list[dict] = []
    per_scene: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    for item in candidates:
        scene_dir = str(item.get("scene_dir") or item.get("recording_id") or item.get("task_id") or "")
        group_key = item_group_key(item)
        if max_per_scene > 0 and per_scene[scene_dir] >= max_per_scene:
            continue
        if max_per_group > 0 and per_group[group_key] >= max_per_group:
            continue
        selected.append(item)
        per_scene[scene_dir] += 1
        per_group[group_key] += 1
        if len(selected) >= target:
            break
    return selected, per_scene, per_group


def sample_rows(
    rows: list[dict],
    *,
    target: int,
    max_per_scene: int,
    max_per_group: int,
    min_rating: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    if target <= 0 or not rows:
        return [], {
            "available": len(rows),
            "sampled": 0,
            "high_quality_first": True,
            "scene_cap_relaxed": False,
            "group_cap_relaxed": False,
            "quality_cap_relaxed": False,
        }

    high_quality = [row for row in rows if int(row.get("rating", -1)) >= min_rating]
    low_quality = [row for row in rows if int(row.get("rating", -1)) < min_rating]
    rng.shuffle(high_quality)
    rng.shuffle(low_quality)

    combined_pref = high_quality + low_quality
    selected, per_scene, _ = take_rows(
        combined_pref,
        target=target,
        max_per_scene=max_per_scene,
        max_per_group=max_per_group,
    )
    scene_cap_relaxed = False
    group_cap_relaxed = False
    quality_cap_relaxed = len(selected) > len(high_quality)

    if len(selected) < target and max_per_group > 0:
        group_cap_relaxed = True
        selected, per_scene, _ = take_rows(
            combined_pref,
            target=target,
            max_per_scene=max_per_scene,
            max_per_group=0,
        )

    if len(selected) < target and max_per_scene > 0:
        scene_cap_relaxed = True
        selected, per_scene, _ = take_rows(
            combined_pref,
            target=target,
            max_per_scene=0,
            max_per_group=0,
        )

    return selected, {
        "available": len(rows),
        "sampled": len(selected),
        "unique_scenes": len(per_scene),
        "high_quality_pool": len(high_quality),
        "high_quality_first": True,
        "scene_cap_relaxed": scene_cap_relaxed,
        "group_cap_relaxed": group_cap_relaxed,
        "quality_cap_relaxed": quality_cap_relaxed,
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
        "max_per_scene": int(args.max_per_scene),
        "max_per_group": int(args.max_per_group),
        "min_rating": int(args.min_rating),
        "counts": {},
    }

    for task in tasks:
        rows = load_jsonl(resolve_task_file(input_dir, task))
        picked, task_summary = sample_rows(
            rows,
            target=args.per_task,
            max_per_scene=int(args.max_per_scene),
            max_per_group=int(args.max_per_group),
            min_rating=int(args.min_rating),
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
