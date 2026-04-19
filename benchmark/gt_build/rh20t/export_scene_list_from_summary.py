#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RH20T selected_scenes.json from an existing benchmark summary.")
    parser.add_argument("--summary-json", default="/data/projects/GM-100/benchmark/rh20t_benchmark_v0_summary.json")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_path = Path(args.output_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_list = summary.get("scene_list", [])
    if not isinstance(scene_list, list) or not scene_list:
        raise ValueError(f"No scene_list found in {summary_path}")
    rows = []
    seen = set()
    for row in scene_list:
        scene_dir = str(row.get("scene_dir", "")).strip()
        task_id = str(row.get("task_id", "")).strip()
        if not scene_dir or not task_id or scene_dir in seen:
            continue
        rows.append({"scene_dir": scene_dir, "task_id": task_id})
        seen.add(scene_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source_summary_json": str(summary_path),
                "num_selected_scenes": len(rows),
                "selected_scenes": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "num_selected_scenes": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
