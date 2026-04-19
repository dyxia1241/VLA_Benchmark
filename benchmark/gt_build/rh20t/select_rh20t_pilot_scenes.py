#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a compact RH20T pilot scene list from archive audit outputs.")
    parser.add_argument("--scene-index-jsonl", required=True)
    parser.add_argument("--task-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-scenes-per-task", type=int, default=3)
    parser.add_argument("--max-total-scenes", type=int, default=24)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    scene_rows = load_jsonl(Path(args.scene_index_jsonl))
    task_rows = json.loads(Path(args.task_summary_json).read_text(encoding="utf-8"))

    ranked_tasks = sorted(
        task_rows,
        key=lambda row: (
            -int(row.get("n_scenes_with_ft", 0)),
            -int(row.get("n_scenes_with_gripper", 0)),
            -int(row.get("n_scenes_with_tcp", 0)),
            -int(row.get("n_scenes_with_color", 0)),
            row.get("task_id", ""),
        ),
    )

    by_task: dict[str, list[dict]] = {}
    for row in scene_rows:
        by_task.setdefault(str(row.get("task_id", "")), []).append(row)

    for task_id in by_task:
        by_task[task_id] = sorted(
            by_task[task_id],
            key=lambda row: (
                -int(bool(row.get("force_torque"))),
                -int(bool(row.get("gripper"))),
                -int(bool(row.get("tcp"))),
                -int(bool(row.get("high_freq_data"))),
                -int(row.get("n_cam_color", 0)),
                -int(row.get("n_cam_timestamps", 0)),
                str(row.get("scene_dir", "")),
            ),
        )

    picked: list[dict] = []
    total_cap = int(args.max_total_scenes)
    per_task_cap = int(args.max_scenes_per_task)
    for task in ranked_tasks:
        task_id = str(task.get("task_id", ""))
        candidates = by_task.get(task_id, [])
        if not candidates:
            continue
        take = candidates[:per_task_cap]
        for row in take:
            picked.append(
                {
                    "task_id": task_id,
                    "scene_dir": row["scene_dir"],
                    "n_cameras": row.get("n_cameras", 0),
                    "n_cam_color": row.get("n_cam_color", 0),
                    "force_torque": row.get("force_torque", False),
                    "gripper": row.get("gripper", False),
                    "tcp": row.get("tcp", False),
                    "high_freq_data": row.get("high_freq_data", False),
                }
            )
            if len(picked) >= total_cap:
                break
        if len(picked) >= total_cap:
            break

    out = {
        "max_total_scenes": total_cap,
        "max_scenes_per_task": per_task_cap,
        "num_selected": len(picked),
        "selected_scenes": picked,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
