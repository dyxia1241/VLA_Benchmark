#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import h5py
import numpy as np

from build_aist_pilot_suite import item_base, iter_episodes


T3_CONTEXT_OFFSETS = (-30, -15, 0, 15)
DEFAULT_DIRECTION_MAPPING = Path(__file__).resolve().with_name("aist_t3_direction_mapping_cam_high.json")
T3_LABELS = ("left", "right", "top", "bottom")
T3_LABEL_TO_TEXT = {
    "left": "Left",
    "right": "Right",
    "top": "Up",
    "bottom": "Down",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AIST T3 pilot GT from selected20 episodes.")
    parser.add_argument("--selected-root", default="/data/projects/GM-100/aist-bimanip/selected20")
    parser.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/aist_t3_pilot_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-label", type=int, default=18)
    parser.add_argument("--camera", default="cam_high")
    parser.add_argument("--min-delta", type=float, default=0.035)
    parser.add_argument("--purity-ratio", type=float, default=1.35)
    parser.add_argument("--direction-mapping", default=str(DEFAULT_DIRECTION_MAPPING))
    return parser.parse_args()


def load_direction_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip().lower() in {"", "none", "identity"}:
        return None
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"AIST T3 direction mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    mapping = obj.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"AIST T3 direction mapping missing object field 'mapping': {mapping_path}")
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


def build_choices(correct_label: str, rng: random.Random) -> tuple[dict[str, str], str]:
    letters = ["A", "B", "C", "D"]
    labels = list(T3_LABELS)
    rng.shuffle(labels)
    answer = letters[labels.index(correct_label)]
    choices = {letter: T3_LABEL_TO_TEXT[label] for letter, label in zip(letters, labels)}
    return choices, answer


def candidate_centers(length: int, guard: int) -> list[int]:
    lo = guard
    hi = length - guard - 1
    if hi <= lo:
        return []
    return list(range(lo, hi + 1, 6))


def build_t3_items_for_episode(
    path: Path,
    f: h5py.File,
    rng: random.Random,
    camera: str,
    min_delta: float,
    purity_ratio: float,
    direction_mapping: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    qpos = np.asarray(f["observations/qpos"])
    if qpos.ndim != 2 or qpos.shape[1] < 13:
        return []

    rows: list[dict[str, Any]] = []
    base = item_base(path, f, camera)
    n = qpos.shape[0]
    centers = candidate_centers(n, guard=max(abs(x) for x in T3_CONTEXT_OFFSETS))
    seen_per_arm_label: Counter[tuple[str, str]] = Counter()
    mapping_by_arm = direction_mapping.get("mapping", {}) if direction_mapping else {}
    mapping_source = direction_mapping.get("source_annotation_sheet", "") if direction_mapping else ""

    arm_specs = [
        ("left", 0, 1),
        ("right", 7, 8),
    ]
    for center in centers:
        idxs = [center + off for off in T3_CONTEXT_OFFSETS]
        for arm, x_idx, y_idx in arm_specs:
            delta_x = float(qpos[idxs[-1], x_idx] - qpos[idxs[0], x_idx])
            delta_y = float(qpos[idxs[-1], y_idx] - qpos[idxs[0], y_idx])
            auto_label, meta = dominant_direction(delta_x, delta_y, min_delta=min_delta, purity_ratio=purity_ratio)
            if auto_label is None:
                continue
            if direction_mapping:
                calibrated_label = mapping_by_arm.get(arm, {}).get(auto_label)
                if calibrated_label not in T3_LABELS:
                    continue
            else:
                calibrated_label = auto_label
            label = str(calibrated_label)
            if label not in T3_LABELS:
                continue
            key = (arm, label)
            if seen_per_arm_label[key] >= 3:
                continue
            seen_per_arm_label[key] += 1
            choices, answer = build_choices(label, rng)
            rows.append(
                {
                    **base,
                    "task_type": "T3",
                    "query_arm": arm,
                    "frame_index": int(center),
                    "frame_indices": [int(x) for x in idxs],
                    "question": f"In which direction does the {arm} robot arm primarily move across these time-ordered frames?",
                    "choices": choices,
                    "answer": answer,
                    "motion_direction": T3_LABEL_TO_TEXT[label],
                    "motion_direction_raw": label,
                    "auto_qpos_direction": T3_LABEL_TO_TEXT[auto_label],
                    "auto_qpos_direction_raw": auto_label,
                    "direction_mapping_source": mapping_source,
                    "delta_x": round(delta_x, 6),
                    "delta_y": round(delta_y, 6),
                    "dominant_magnitude": round(meta["dominant_magnitude"], 6),
                    "secondary_magnitude": round(meta["secondary_magnitude"], 6),
                    "direction_purity_ratio": round(meta["purity_ratio"], 6),
                }
            )
    return rows


def cap_rows(rows: list[dict[str, Any]], per_label: int, rng: random.Random) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for arm in ["left", "right"]:
        for label in T3_LABELS:
            subset = [r for r in rows if r.get("query_arm") == arm and r.get("motion_direction_raw") == label]
            rng.shuffle(subset)
            capped.extend(subset[:per_label])
    rng.shuffle(capped)
    return capped


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    selected_root = Path(args.selected_root)
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    direction_mapping = load_direction_mapping(args.direction_mapping)

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
                    direction_mapping=direction_mapping,
                )
            )

    pool_path = gt_dir / "aist_t3_pilot_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    pilot = cap_rows(rows, args.per_label, rng)
    pilot_path = output_dir / "aist_t3_pilot_v0.jsonl"
    with pilot_path.open("w", encoding="utf-8") as fh:
        for row in pilot:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_arm = Counter(r["query_arm"] for r in rows)
    by_label = Counter(r["motion_direction_raw"] for r in rows)
    summary = {
        "selected_root": str(selected_root),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "pool_count": len(rows),
        "pilot_count": len(pilot),
        "by_arm": dict(by_arm),
        "by_label": dict(by_label),
        "per_label": int(args.per_label),
        "camera": args.camera,
        "context_offsets": list(T3_CONTEXT_OFFSETS),
        "min_delta": float(args.min_delta),
        "purity_ratio": float(args.purity_ratio),
        "direction_mapping": args.direction_mapping if direction_mapping else "identity",
        "pilot_jsonl": str(pilot_path),
        "pool_jsonl": str(pool_path),
    }
    (output_dir / "aist_t3_pilot_v0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
