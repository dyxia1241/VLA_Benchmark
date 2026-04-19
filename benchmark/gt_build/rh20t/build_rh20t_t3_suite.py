#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
GT_BUILD_DIR = THIS_DIR.parent
for _path in (THIS_DIR, GT_BUILD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rh20t_utils import DEFAULT_EXTRACTED_ROOT, PRIMARY_CAMERA, load_scene_signals, load_selected_scenes, load_task_catalog


T3_CONTEXT_OFFSETS = (-20, -10, 0, 10)
DEFAULT_DIRECTION_MAPPING = Path(__file__).resolve().with_name("rh20t_t3_direction_mapping_036422060215.json")
T3_LABELS = ("left", "right", "top", "bottom")
T3_LABEL_TO_TEXT = {
    "left": "Left",
    "right": "Right",
    "top": "Up",
    "bottom": "Down",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RH20T full T3 suite from extracted cfg2 scenes.")
    parser.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    parser.add_argument("--scene-list-json", default="/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/selected_scenes.json")
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/pools/t3_gt_items.jsonl")
    parser.add_argument("--summary-json", default="/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/pools/t3_gt_items_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", default=PRIMARY_CAMERA)
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--purity-ratio", type=float, default=1.35)
    parser.add_argument("--max-per-label-per-scene", type=int, default=3)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--direction-mapping", default=str(DEFAULT_DIRECTION_MAPPING))
    return parser.parse_args()


def load_direction_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip().lower() in {"", "none", "identity"}:
        return None
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"RH20T T3 direction mapping not found: {mapping_path}")
    obj = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = obj.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"RH20T T3 direction mapping missing object field 'mapping': {mapping_path}")
    return obj


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


def candidate_centers(length: int, guard: int, step: int) -> list[int]:
    lo = guard
    hi = length - guard - 1
    if hi <= lo:
        return []
    return list(range(lo, hi + 1, step))


def build_choices(correct_label: str, rng: random.Random) -> tuple[dict[str, str], str]:
    letters = ["A", "B", "C", "D"]
    labels = list(T3_LABELS)
    rng.shuffle(labels)
    answer = letters[labels.index(correct_label)]
    choices = {letter: T3_LABEL_TO_TEXT[label] for letter, label in zip(letters, labels)}
    return choices, answer


def build_scene_items(
    scene,
    *,
    rng: random.Random,
    min_delta: float,
    purity_ratio: float,
    max_per_label_per_scene: int,
    step: int,
    direction_mapping: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    pts = np.asarray(scene.pos_xyz[:, :2], dtype=np.float64)
    n = int(pts.shape[0])
    guard = max(abs(x) for x in T3_CONTEXT_OFFSETS)
    centers = candidate_centers(n, guard=guard, step=step)
    mapping = direction_mapping.get("mapping", {}) if direction_mapping else {}
    mapping_source = direction_mapping.get("source_annotation_sheet", "") if direction_mapping else "identity"
    selected_per_label: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for center in centers:
        idxs = [int(center + off) for off in T3_CONTEXT_OFFSETS]
        delta_x = float(pts[idxs[-1], 0] - pts[idxs[0], 0])
        delta_y = float(pts[idxs[-1], 1] - pts[idxs[0], 1])
        auto_label, meta = dominant_direction(delta_x, delta_y, min_delta=min_delta, purity_ratio=purity_ratio)
        if auto_label is None:
            continue
        label = str(mapping.get(auto_label, auto_label)) if direction_mapping else auto_label
        if label not in T3_LABELS:
            continue
        if selected_per_label[label] >= max_per_label_per_scene:
            continue
        selected_per_label[label] += 1
        choices, answer = build_choices(label, rng)
        rows.append(
            {
                "dataset": "RH20T",
                "task_id": scene.task_id,
                "recording_id": scene.scene_dir,
                "scene_dir": scene.scene_dir,
                "camera": scene.camera,
                "task_type": "T3",
                "arm_type": "single_arm",
                "task_meta_description": scene.task_description,
                "rating": int(scene.rating),
                "calib_quality": int(scene.calib_quality),
                "frame_index": int(center),
                "frame_indices": [int(x) for x in idxs],
                "question": "In which direction does the robot arm primarily move across these time-ordered frames?",
                "choices": choices,
                "answer": answer,
                "motion_direction": T3_LABEL_TO_TEXT[label],
                "motion_direction_raw": label,
                "auto_tcp_direction": T3_LABEL_TO_TEXT[auto_label],
                "auto_tcp_direction_raw": auto_label,
                "direction_mapping_source": mapping_source,
                "delta_x": round(delta_x, 6),
                "delta_y": round(delta_y, 6),
                "dominant_magnitude": round(meta["dominant_magnitude"], 6),
                "secondary_magnitude": round(meta["secondary_magnitude"], 6),
                "direction_purity_ratio": round(meta["purity_ratio"], 6),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    extracted_root = Path(args.extracted_root)
    scene_rows = load_selected_scenes(Path(args.scene_list_json))
    catalog = load_task_catalog()
    direction_mapping = load_direction_mapping(args.direction_mapping)

    per_scene_counts: dict[str, int] = {}
    task_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    auto_direction_counts: Counter[str] = Counter()
    skipped_scenes: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for row in scene_rows:
        task_id = str(row["task_id"])
        scene_dir = str(row["scene_dir"])
        task_desc = catalog.get(task_id, {}).get("task_description_english", task_id)
        try:
            scene = load_scene_signals(
                extracted_root,
                scene_dir=scene_dir,
                task_description=task_desc,
                camera=str(args.camera),
            )
        except Exception as exc:  # noqa: BLE001
            skipped_scenes.append({"scene_dir": scene_dir, "task_id": task_id, "reason": str(exc)})
            continue
        scene_items = build_scene_items(
            scene,
            rng=rng,
            min_delta=float(args.min_delta),
            purity_ratio=float(args.purity_ratio),
            max_per_label_per_scene=int(args.max_per_label_per_scene),
            step=int(args.step),
            direction_mapping=direction_mapping,
        )
        per_scene_counts[scene_dir] = len(scene_items)
        for item in scene_items:
            task_counts[str(item.get("task_id", ""))] += 1
            direction_counts[str(item.get("motion_direction_raw", ""))] += 1
            auto_direction_counts[str(item.get("auto_tcp_direction_raw", ""))] += 1
        rows.extend(scene_items)

    write_jsonl(Path(args.output_jsonl), rows)
    summary = {
        "dataset": "RH20T",
        "extracted_root": str(extracted_root),
        "scene_list_json": str(args.scene_list_json),
        "output_jsonl": str(args.output_jsonl),
        "num_selected_scenes": len(scene_rows),
        "num_processed_scenes": len(per_scene_counts),
        "num_skipped_scenes": len(skipped_scenes),
        "skipped_scenes": skipped_scenes,
        "pool_count": len(rows),
        "camera": str(args.camera),
        "context_offsets": list(T3_CONTEXT_OFFSETS),
        "min_delta": float(args.min_delta),
        "purity_ratio": float(args.purity_ratio),
        "max_per_label_per_scene": int(args.max_per_label_per_scene),
        "step": int(args.step),
        "direction_mapping": str(args.direction_mapping) if direction_mapping else "identity",
        "counts_by_task_id": dict(task_counts),
        "counts_by_motion_direction": dict(direction_counts),
        "counts_by_auto_tcp_direction": dict(auto_direction_counts),
        "scene_item_count_summary": {
            "nonempty_scenes": sum(1 for v in per_scene_counts.values() if v > 0),
            "max_items_per_scene": max(per_scene_counts.values()) if per_scene_counts else 0,
            "mean_items_per_scene": round(float(sum(per_scene_counts.values()) / max(1, len(per_scene_counts))), 4),
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
