#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from reassemble_utils import (
    DEFAULT_CAMERA,
    REASSEMBLE_ROOT,
    camera_timestamps,
    filter_recordings,
    load_recording_index,
    nearest_timestamp_indices,
    non_no_action_segments,
    object_from_high_level_text,
    recording_h5_path,
    timestamp_interval_to_frame_range,
)


T3_CONTEXT_OFFSETS = (-20, -10, 0, 10)
T3_LABELS = ("left", "right", "top", "bottom")
T3_LABEL_TO_TEXT = {
    "left": "Left",
    "right": "Right",
    "top": "Up",
    "bottom": "Down",
}
CHOICES = {
    "A": "Left",
    "B": "Right",
    "C": "Up",
    "D": "Down",
}
LABEL_TO_ANSWER = {
    "left": "A",
    "right": "B",
    "top": "C",
    "bottom": "D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build REASSEMBLE T3 direction calibration cards for manual audit.")
    p.add_argument("--dataset-root", default=str(REASSEMBLE_ROOT))
    p.add_argument("--split", default="test_split1")
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/reassemble_t3_direction_calibration_v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-recordings", type=int, default=0)
    p.add_argument("--per-direction", type=int, default=18)
    p.add_argument("--min-delta", type=float, default=0.018)
    p.add_argument("--purity-ratio", type=float, default=1.30)
    p.add_argument("--max-centers-per-segment", type=int, default=2)
    return p.parse_args()


def dominant_direction(delta_x: float, delta_y: float, *, min_delta: float, purity_ratio: float) -> tuple[str | None, dict[str, float]]:
    abs_x = abs(delta_x)
    abs_y = abs(delta_y)
    dominant = max(abs_x, abs_y)
    second = min(abs_x, abs_y)
    ratio = dominant / max(second, 1e-8)
    if dominant < min_delta or ratio < purity_ratio:
        return None, {
            "dominant_magnitude": float(dominant),
            "secondary_magnitude": float(second),
            "purity_ratio": float(ratio),
        }
    if abs_x >= abs_y:
        return ("right" if delta_x > 0 else "left"), {
            "dominant_magnitude": float(dominant),
            "secondary_magnitude": float(second),
            "purity_ratio": float(ratio),
        }
    return ("top" if delta_y > 0 else "bottom"), {
        "dominant_magnitude": float(dominant),
        "secondary_magnitude": float(second),
        "purity_ratio": float(ratio),
    }


def sample_centers(lo: int, hi: int, *, count: int) -> list[int]:
    if hi <= lo:
        return []
    if count <= 1:
        return [int((lo + hi) // 2)]
    grid = np.linspace(lo, hi, num=count)
    out = sorted({int(round(x)) for x in grid})
    return [x for x in out if lo <= x <= hi]


def rows_to_annotation_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "calibration_id",
        "recording_id",
        "segment_index",
        "high_level_text_raw",
        "object_raw",
        "camera",
        "frame_indices",
        "auto_velocity_direction_raw",
        "auto_velocity_direction_text",
        "human_direction_raw",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "calibration_id": row["calibration_id"],
                    "recording_id": row["recording_id"],
                    "segment_index": row["segment_index"],
                    "high_level_text_raw": row.get("high_level_text_raw", ""),
                    "object_raw": row.get("object_raw", ""),
                    "camera": row.get("camera", ""),
                    "frame_indices": ",".join(str(x) for x in row.get("frame_indices", [])),
                    "auto_velocity_direction_raw": row.get("auto_velocity_direction_raw", ""),
                    "auto_velocity_direction_text": row.get("auto_velocity_direction_text", ""),
                    "human_direction_raw": "",
                    "human_notes": "",
                }
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    index_rows = load_recording_index()
    selected = filter_recordings(index_rows, split=args.split, dataset_root=dataset_root)
    if args.limit_recordings > 0:
        selected = selected[: int(args.limit_recordings)]

    guard = max(abs(x) for x in T3_CONTEXT_OFFSETS)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_rows: list[dict[str, Any]] = []

    for row in selected:
        recording_id = str(row.get("recording_id", "")).strip()
        if not recording_id:
            continue
        h5_path = recording_h5_path(dataset_root, recording_id)
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            cam_ts = camera_timestamps(f, args.camera)
            vel_ts = np.asarray(f["timestamps/velocity"][:], dtype=np.float64)
            vel_xy = np.asarray(f["robot_state/velocity"][:, :2], dtype=np.float64)
            if cam_ts.size == 0 or vel_ts.size == 0 or vel_xy.size == 0:
                continue

            segments = non_no_action_segments(row)
            for seg in segments:
                seg_index = int(seg.get("segment_index", -1))
                high_text = str(seg.get("text", "")).strip()
                obj_text = object_from_high_level_text(high_text)
                interval = timestamp_interval_to_frame_range(cam_ts, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
                if interval is None:
                    continue
                lo, hi = interval
                if hi - lo + 1 < (2 * guard + 1):
                    continue
                center_lo = lo + guard
                center_hi = hi - guard
                centers = sample_centers(center_lo, center_hi, count=max(1, int(args.max_centers_per_segment)))
                for center in centers:
                    frame_indices = [int(center + off) for off in T3_CONTEXT_OFFSETS]
                    query_ts = np.asarray([cam_ts[idx] for idx in frame_indices], dtype=np.float64)
                    vel_idx = nearest_timestamp_indices(vel_ts, query_ts)
                    dx = float(np.mean(vel_xy[vel_idx, 0]))
                    dy = float(np.mean(vel_xy[vel_idx, 1]))
                    label, meta = dominant_direction(dx, dy, min_delta=float(args.min_delta), purity_ratio=float(args.purity_ratio))
                    if label not in T3_LABELS:
                        continue
                    item = {
                        "dataset": "REASSEMBLE",
                        "task_type": "T3_direction_calibration",
                        "recording_id": recording_id,
                        "task_id": recording_id,
                        "segment_index": seg_index,
                        "camera": args.camera,
                        "frame_index": int(center),
                        "frame_indices": frame_indices,
                        "display_labels": [f"t{off:+d}" if off != 0 else "t0" for off in T3_CONTEXT_OFFSETS],
                        "question": "Human audit: across these ordered frames, what is the primary image-plane motion direction?",
                        "choices": dict(CHOICES),
                        "answer": LABEL_TO_ANSWER[label],
                        "high_level_text_raw": high_text,
                        "object_raw": obj_text,
                        "auto_velocity_direction_raw": label,
                        "auto_velocity_direction_text": T3_LABEL_TO_TEXT[label],
                        "delta_x": round(dx, 6),
                        "delta_y": round(dy, 6),
                        "dominant_magnitude": round(meta["dominant_magnitude"], 6),
                        "secondary_magnitude": round(meta["secondary_magnitude"], 6),
                        "direction_purity_ratio": round(meta["purity_ratio"], 6),
                    }
                    pool_rows.append(item)
                    buckets[label].append(item)

    calibration: list[dict[str, Any]] = []
    for label in T3_LABELS:
        subset = list(buckets.get(label, []))
        rng.shuffle(subset)
        calibration.extend(subset[: int(max(0, args.per_direction))])
    rng.shuffle(calibration)
    for idx, row in enumerate(calibration, start=1):
        row["calibration_id"] = f"reassemble_t3_calib_{idx:04d}"

    jsonl_path = output_dir / "reassemble_t3_direction_calibration_v0.jsonl"
    pool_path = gt_dir / "reassemble_t3_direction_calibration_pool.jsonl"
    ann_path = output_dir / "annotation_sheet.csv"
    with pool_path.open("w", encoding="utf-8") as fh:
        for row in pool_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in calibration:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows_to_annotation_sheet(ann_path, calibration)

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "camera": args.camera,
        "num_recordings": len(selected),
        "pool_count": len(pool_rows),
        "calibration_count": len(calibration),
        "by_auto_direction": {k: len(v) for k, v in buckets.items()},
        "per_direction": int(args.per_direction),
        "context_offsets": list(T3_CONTEXT_OFFSETS),
        "min_delta": float(args.min_delta),
        "purity_ratio": float(args.purity_ratio),
        "jsonl": str(jsonl_path),
        "pool_jsonl": str(pool_path),
        "annotation_sheet": str(ann_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
