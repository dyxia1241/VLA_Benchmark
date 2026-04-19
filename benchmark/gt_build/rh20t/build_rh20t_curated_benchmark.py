#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TASK_FILES = {
    "T1": "t1_gt_items.jsonl",
    "T2": "t2_gt_items.jsonl",
    "T3": "t3_gt_items.jsonl",
    "T7": "t7_gt_items.jsonl",
    "T6": "t6_gt_items.jsonl",
    "T_temporal": "t_temporal_gt_items.jsonl",
    "T_binary": "t_binary_gt_items.jsonl",
    "T_progress": "t_progress_gt_items.jsonl",
}
TASK_ALIASES = {"T5": "T_progress", "T8": "T_temporal", "T9": "T_binary"}
DEFAULT_TARGETS = {
    "T1": 2000,
    "T2": 1500,
    "T3": 1800,
    "T7": 4000,
    "T6": 1500,
    "T_temporal": 1200,
    "T_binary": 1500,
    "T_progress": 2300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated RH20T benchmark release from per-task full GT pools.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-per-type-dir", required=True)
    parser.add_argument("--tasks", default="T1,T2,T5,T6,T8,T9")
    parser.add_argument("--target-json", default="")
    parser.add_argument("--min-rating", type=int, default=4)
    parser.add_argument("--max-calib-quality", type=int, default=3)
    parser.add_argument("--max-per-scene", type=int, default=8)
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_group_key(item: dict) -> str:
    task_type = str(item.get("task_type", ""))
    scene_dir = str(item.get("scene_dir") or item.get("recording_id") or item.get("task_id") or "")
    interval_id = str(item.get("interval_id", "")).strip()
    frame_index = item.get("frame_index", "")

    if task_type == "T_progress":
        return f"{task_type}|{scene_dir}|{interval_id}|{item.get('progress_bin', '')}"
    if task_type == "T_temporal":
        return f"{task_type}|{scene_dir}|{interval_id}|{','.join(str(int(x)) for x in item.get('frame_indices', []) or [])}"
    if task_type == "T_binary":
        return f"{task_type}|{scene_dir}|{item.get('difficulty', '')}|{','.join(str(int(x)) for x in item.get('frame_indices', []) or [])}"
    if task_type == "T7":
        return f"{task_type}|{scene_dir}|{item.get('window_id', '')}|{','.join(str(int(x)) for x in item.get('frame_indices', []) or [])}"
    if task_type == "T3":
        return f"{task_type}|{scene_dir}|{item.get('motion_direction_raw', '')}|{frame_index}"
    if task_type == "T1":
        return f"{task_type}|{scene_dir}|{item.get('phase_label', '')}|{frame_index}"
    if task_type == "T2":
        return f"{task_type}|{scene_dir}|{item.get('label', '')}|{frame_index}"
    if task_type == "T6":
        return f"{task_type}|{scene_dir}|{item.get('speed_level_label', '')}|{frame_index}"
    return f"{task_type}|{scene_dir}|{frame_index}"


def task_sort_key(item: dict) -> tuple:
    rating = int(item.get("rating", -99))
    calib = int(item.get("calib_quality", 99))
    scene_dir = str(item.get("scene_dir") or item.get("recording_id") or item.get("task_id") or "")
    frame_index = int(item.get("frame_index", -1))
    return (-rating, calib, scene_dir, frame_index)


def filter_quality(rows: list[dict], *, min_rating: int, max_calib_quality: int) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        task_type = str(row.get("task_type", ""))
        calib_quality = int(row.get("calib_quality", 99))
        if calib_quality > max_calib_quality:
            continue
        if task_type == "T7":
            out.append(row)
            continue
        if int(row.get("rating", -99)) >= min_rating:
            out.append(row)
    return out


def take_rows(
    rows: list[dict],
    *,
    target: int,
    max_per_scene: int,
    max_per_group: int,
) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    per_scene: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    for item in rows:
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
    return selected, {
        "sampled": len(selected),
        "unique_scenes": len(per_scene),
        "max_scene_load": max(per_scene.values()) if per_scene else 0,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tasks = normalize_tasks(args.tasks)
    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    output_summary_json = Path(args.output_summary_json)
    output_per_type_dir = Path(args.output_per_type_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_per_type_dir.mkdir(parents=True, exist_ok=True)

    targets = dict(DEFAULT_TARGETS)
    if args.target_json:
        targets.update(json.loads(args.target_json))

    combined: list[dict] = []
    by_type_selected: dict[str, list[dict]] = {}
    summary = {
        "dataset": "RH20T",
        "source_dir": str(input_dir),
        "tasks": tasks,
        "targets": {task: int(targets[task]) for task in tasks},
        "min_rating": int(args.min_rating),
        "max_calib_quality": int(args.max_calib_quality),
        "max_per_scene": int(args.max_per_scene),
        "max_per_group": int(args.max_per_group),
        "counts": {
            "raw_by_type": {},
            "quality_filtered_by_type": {},
            "sampled_by_type": {},
            "shortfall_by_type": {},
        },
        "coverage": {
            "unique_scenes_by_type": {},
            "unique_task_ids_by_type": {},
        },
        "scene_list": [],
    }

    scene_union: set[str] = set()
    scene_to_task: dict[str, str] = {}

    for task in tasks:
        rows = load_jsonl(input_dir / TASK_FILES[task])
        summary["counts"]["raw_by_type"][task] = len(rows)
        rows = filter_quality(
            rows,
            min_rating=int(args.min_rating),
            max_calib_quality=int(args.max_calib_quality),
        )
        summary["counts"]["quality_filtered_by_type"][task] = len(rows)
        rng.shuffle(rows)
        rows.sort(key=task_sort_key)
        selected, meta = take_rows(
            rows,
            target=int(targets[task]),
            max_per_scene=int(args.max_per_scene),
            max_per_group=int(args.max_per_group),
        )
        by_type_selected[task] = selected
        combined.extend(selected)
        summary["counts"]["sampled_by_type"][task] = len(selected)
        summary["counts"]["shortfall_by_type"][task] = max(0, int(targets[task]) - len(selected))
        summary["coverage"]["unique_scenes_by_type"][task] = meta["unique_scenes"]
        summary["coverage"]["unique_task_ids_by_type"][task] = len({str(x.get("task_id", "")) for x in selected})
        for row in selected:
            scene_dir = str(row.get("scene_dir") or row.get("recording_id") or "")
            if scene_dir:
                scene_union.add(scene_dir)
                scene_to_task[scene_dir] = str(row.get("task_id", ""))

    combined.sort(key=lambda x: (str(x.get("task_type", "")), task_sort_key(x)))
    write_jsonl(output_jsonl, combined)

    for task in tasks:
        out_name = f"{task}.jsonl"
        write_jsonl(output_per_type_dir / out_name, by_type_selected[task])

    summary["counts"]["total_sampled"] = len(combined)
    summary["counts"]["total_target"] = sum(int(targets[t]) for t in tasks)
    summary["coverage"]["unique_scenes_total"] = len(scene_union)
    summary["coverage"]["unique_task_ids_total"] = len(set(scene_to_task.values()))
    summary["scene_list"] = [
        {"scene_dir": scene_dir, "task_id": scene_to_task.get(scene_dir, "")}
        for scene_dir in sorted(scene_union)
    ]

    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
