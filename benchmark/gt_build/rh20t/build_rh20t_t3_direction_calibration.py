#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
GT_BUILD_DIR = THIS_DIR.parent
import sys
for _path in (THIS_DIR, GT_BUILD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rh20t_utils import DEFAULT_EXTRACTED_ROOT, PRIMARY_CAMERA, load_task_catalog


T3_CONTEXT_OFFSETS = (-20, -10, 0, 10)
T3_LABELS = ("left", "right", "top", "bottom")
T3_LABEL_TO_TEXT = {
    "left": "Left",
    "right": "Right",
    "top": "Up",
    "bottom": "Down",
}
ANNOTATION_HEADERS = [
    "calibration_id",
    "task_id",
    "task_name",
    "recording_id",
    "scene_dir",
    "camera",
    "frame_index",
    "frame_indices",
    "auto_tcp_direction_raw",
    "auto_tcp_direction_text",
    "delta_x",
    "delta_y",
    "dominant_magnitude",
    "direction_purity_ratio",
    "card_path",
    "human_label",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RH20T T3 direction calibration set for human audit.")
    p.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    p.add_argument("--scene-list-json", default="/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/selected_scenes.json")
    p.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/rh20t_t3_direction_calibration_v0")
    p.add_argument("--camera", default=PRIMARY_CAMERA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-direction", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=0.03)
    p.add_argument("--purity-ratio", type=float, default=1.4)
    return p.parse_args()


def load_selected_scenes(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("selected_scenes", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid selected scene list: {path}")
    return [r for r in rows if isinstance(r, dict) and r.get("scene_dir")]


def dominant_direction(delta_x: float, delta_y: float, min_delta: float, purity_ratio: float) -> tuple[str | None, dict[str, float]]:
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
        label = "right" if delta_x > 0 else "left"
    else:
        label = "top" if delta_y > 0 else "bottom"
    return label, {
        "dominant_magnitude": float(dominant),
        "secondary_magnitude": float(second),
        "purity_ratio": float(ratio),
    }


def candidate_centers(length: int, guard: int) -> list[int]:
    lo = guard
    hi = length - guard - 1
    if hi <= lo:
        return []
    return list(range(lo, hi + 1, 10))


def build_scene_candidates(
    *,
    scene_dir: str,
    task_id: str,
    task_description: str,
    extracted_root: Path,
    camera: str,
    min_delta: float,
    purity_ratio: float,
) -> list[dict[str, Any]]:
    scene_root = extracted_root / scene_dir
    tcp_rows = np.load(scene_root / "transformed" / "tcp_base.npy", allow_pickle=True).item()[camera]
    pts = np.asarray([row["tcp"][:3] for row in tcp_rows], dtype=np.float64)
    n = int(pts.shape[0])
    guard = max(abs(x) for x in T3_CONTEXT_OFFSETS)
    out: list[dict[str, Any]] = []
    seen_per_label: Counter[str] = Counter()
    for center in candidate_centers(n, guard=guard):
        idxs = [int(center + off) for off in T3_CONTEXT_OFFSETS]
        delta_x = float(pts[idxs[-1], 0] - pts[idxs[0], 0])
        delta_y = float(pts[idxs[-1], 1] - pts[idxs[0], 1])
        label, meta = dominant_direction(delta_x, delta_y, min_delta=min_delta, purity_ratio=purity_ratio)
        if label is None:
            continue
        if seen_per_label[label] >= 3:
            continue
        seen_per_label[label] += 1
        out.append(
            {
                "dataset": "RH20T",
                "task_type": "T3_direction_calibration",
                "task_id": task_id,
                "recording_id": scene_dir,
                "scene_dir": scene_dir,
                "camera": camera,
                "arm_type": "single_arm",
                "task_meta_description": task_description,
                "frame_index": int(center),
                "frame_indices": idxs,
                "question": "Human audit: across these time-ordered frames, which image-plane direction does the robot arm primarily move?",
                "choices": {
                    "A": "Left",
                    "B": "Right",
                    "C": "Up",
                    "D": "Down",
                    "E": "Unclear / no reliable visible motion",
                },
                "answer": "",
                "needs_human_direction_label": True,
                "auto_tcp_direction_raw": label,
                "auto_tcp_direction_text": T3_LABEL_TO_TEXT[label],
                "delta_x": round(delta_x, 6),
                "delta_y": round(delta_y, 6),
                "dominant_magnitude": round(meta["dominant_magnitude"], 6),
                "direction_purity_ratio": round(meta["purity_ratio"], 6),
            }
        )
    return out


def select_rows(rows: list[dict[str, Any]], per_direction: int, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in T3_LABELS:
        subset = [r for r in rows if r.get("auto_tcp_direction_raw") == label]
        subset.sort(
            key=lambda r: (
                float(r.get("dominant_magnitude", 0.0)),
                float(r.get("direction_purity_ratio", 0.0)),
            ),
            reverse=True,
        )
        bucket = subset[: max(per_direction * 4, per_direction)]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_direction])
    rng.shuffle(selected)
    return selected


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
                    "task_name": row.get("task_meta_description", ""),
                    "recording_id": row.get("recording_id", ""),
                    "scene_dir": row.get("scene_dir", ""),
                    "camera": row.get("camera", ""),
                    "frame_index": row.get("frame_index", ""),
                    "frame_indices": json.dumps(row.get("frame_indices", []), ensure_ascii=False),
                    "auto_tcp_direction_raw": row.get("auto_tcp_direction_raw", ""),
                    "auto_tcp_direction_text": row.get("auto_tcp_direction_text", ""),
                    "delta_x": row.get("delta_x", ""),
                    "delta_y": row.get("delta_y", ""),
                    "dominant_magnitude": row.get("dominant_magnitude", ""),
                    "direction_purity_ratio": row.get("direction_purity_ratio", ""),
                }
            )
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    extracted_root = Path(args.extracted_root)
    scene_rows = load_selected_scenes(Path(args.scene_list_json))
    catalog = load_task_catalog()
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in scene_rows:
        task_id = str(row["task_id"])
        scene_dir = str(row["scene_dir"])
        task_description = catalog.get(task_id, {}).get("task_description_english", task_id)
        rows.extend(
            build_scene_candidates(
                scene_dir=scene_dir,
                task_id=task_id,
                task_description=task_description,
                extracted_root=extracted_root,
                camera=str(args.camera),
                min_delta=float(args.min_delta),
                purity_ratio=float(args.purity_ratio),
            )
        )

    selected = select_rows(rows, int(args.per_direction), rng)
    calibration = []
    for idx, row in enumerate(selected, start=1):
        item = dict(row)
        item["calibration_id"] = f"rh20t_t3_dir_{idx:03d}"
        calibration.append(item)

    jsonl_path = output_dir / "rh20t_t3_direction_calibration_v0.jsonl"
    pool_path = gt_dir / "rh20t_t3_direction_calibration_pool.jsonl"
    annotation_path = output_dir / "annotation_sheet.csv"
    write_jsonl(pool_path, rows)
    write_jsonl(jsonl_path, calibration)
    write_annotation_sheet(annotation_path, calibration)

    summary = {
        "extracted_root": str(extracted_root),
        "output_dir": str(output_dir),
        "camera": str(args.camera),
        "scene_count": len(scene_rows),
        "pool_count": len(rows),
        "calibration_count": len(calibration),
        "by_auto_direction": dict(Counter(r.get("auto_tcp_direction_raw", "") for r in calibration)),
        "per_direction": int(args.per_direction),
        "context_offsets": list(T3_CONTEXT_OFFSETS),
        "min_delta": float(args.min_delta),
        "purity_ratio": float(args.purity_ratio),
        "jsonl": str(jsonl_path),
        "pool_jsonl": str(pool_path),
        "annotation_sheet": str(annotation_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
