#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import h5py

from build_aist_t3_pilot import (
    T3_CONTEXT_OFFSETS,
    T3_LABEL_TO_TEXT,
    T3_LABELS,
    build_t3_items_for_episode,
)
from build_aist_pilot_suite import iter_episodes


ANNOTATION_HEADERS = [
    "calibration_id",
    "task_id",
    "task_name",
    "recording_id",
    "episode_file",
    "camera",
    "query_arm",
    "frame_index",
    "frame_indices",
    "auto_qpos_direction_raw",
    "auto_qpos_direction_text",
    "delta_x",
    "delta_y",
    "card_path",
    "human_label",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small AIST T3 direction calibration set for human audit.")
    parser.add_argument("--selected-root", default="/data/projects/GM-100/aist-bimanip/selected20")
    parser.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/aist_t3_direction_calibration_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-arm-direction", type=int, default=3)
    parser.add_argument("--camera", default="cam_high")
    parser.add_argument("--min-delta", type=float, default=0.035)
    parser.add_argument("--purity-ratio", type=float, default=1.35)
    return parser.parse_args()


def select_calibration_rows(
    rows: list[dict[str, Any]],
    per_arm_direction: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for arm in ("left", "right"):
        for raw_dir in T3_LABELS:
            subset = [
                r
                for r in rows
                if r.get("query_arm") == arm and r.get("motion_direction_raw") == raw_dir
            ]
            subset.sort(
                key=lambda r: (
                    float(r.get("dominant_magnitude", 0.0)),
                    float(r.get("direction_purity_ratio", 0.0)),
                ),
                reverse=True,
            )
            bucket = subset[: max(per_arm_direction * 4, per_arm_direction)]
            rng.shuffle(bucket)
            selected.extend(bucket[:per_arm_direction])
    rng.shuffle(selected)
    return selected


def to_calibration_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    raw_dir = str(row.get("motion_direction_raw", ""))
    choices = {
        "A": "Left",
        "B": "Right",
        "C": "Up",
        "D": "Down",
        "E": "Unclear / no reliable visible motion",
    }
    return {
        **row,
        "calibration_id": f"aist_t3_dir_{index:03d}",
        "task_type": "T3_direction_calibration",
        "question": f"Human audit: in the image sequence, which direction does the {row.get('query_arm', '')} robot arm primarily move?",
        "choices": choices,
        "answer": "",
        "needs_human_direction_label": True,
        "auto_qpos_direction_raw": raw_dir,
        "auto_qpos_direction_text": T3_LABEL_TO_TEXT.get(raw_dir, raw_dir),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_annotation_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_HEADERS)
        writer.writeheader()
        for row in rows:
            out = {key: "" for key in ANNOTATION_HEADERS}
            out.update(
                {
                    "calibration_id": row.get("calibration_id", ""),
                    "task_id": row.get("task_id", ""),
                    "task_name": row.get("task_name", ""),
                    "recording_id": row.get("recording_id", ""),
                    "episode_file": row.get("episode_file", ""),
                    "camera": row.get("camera", ""),
                    "query_arm": row.get("query_arm", ""),
                    "frame_index": row.get("frame_index", ""),
                    "frame_indices": json.dumps(row.get("frame_indices", []), ensure_ascii=False),
                    "auto_qpos_direction_raw": row.get("auto_qpos_direction_raw", ""),
                    "auto_qpos_direction_text": row.get("auto_qpos_direction_text", ""),
                    "delta_x": row.get("delta_x", ""),
                    "delta_y": row.get("delta_y", ""),
                    "human_label": "",
                    "notes": "",
                }
            )
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    selected_root = Path(args.selected_root)
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    episodes = iter_episodes(selected_root)
    for ep in episodes:
        with h5py.File(ep, "r") as f:
            rows.extend(
                build_t3_items_for_episode(
                    ep,
                    f,
                    rng,
                    args.camera,
                    min_delta=float(args.min_delta),
                    purity_ratio=float(args.purity_ratio),
                )
            )

    selected = select_calibration_rows(rows, int(args.per_arm_direction), rng)
    calibration = [to_calibration_item(row, idx) for idx, row in enumerate(selected, start=1)]

    jsonl_path = output_dir / "aist_t3_direction_calibration_v0.jsonl"
    pool_path = gt_dir / "aist_t3_direction_calibration_pool.jsonl"
    annotation_path = output_dir / "annotation_sheet.csv"
    write_jsonl(pool_path, rows)
    write_jsonl(jsonl_path, calibration)
    write_annotation_sheet(annotation_path, calibration)

    by_arm = Counter(r["query_arm"] for r in calibration)
    by_auto_dir = Counter(r["auto_qpos_direction_raw"] for r in calibration)
    summary = {
        "selected_root": str(selected_root),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "pool_count": len(rows),
        "calibration_count": len(calibration),
        "by_arm": dict(by_arm),
        "by_auto_qpos_direction": dict(by_auto_dir),
        "per_arm_direction": int(args.per_arm_direction),
        "camera": args.camera,
        "context_offsets": list(T3_CONTEXT_OFFSETS),
        "min_delta": float(args.min_delta),
        "purity_ratio": float(args.purity_ratio),
        "jsonl": str(jsonl_path),
        "pool_jsonl": str(pool_path),
        "annotation_sheet": str(annotation_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
