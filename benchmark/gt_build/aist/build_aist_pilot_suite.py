#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
from pathlib import Path
import random
from typing import Any

import h5py
import numpy as np


T4_CHOICES = {
    "A": "Both arms are actively moving",
    "B": "Only the left arm is actively moving",
    "C": "Only the right arm is actively moving",
    "D": "Both arms are idle or barely moving",
}
T4_LABEL_TO_ANSWER = {
    "both_active": "A",
    "left_only": "B",
    "right_only": "C",
    "both_idle": "D",
}
T4_CONTEXT_OFFSETS = (-30, -15, 0, 15)
T6_CHOICES = {
    "A": "Stationary (the queried arm is still or barely moving)",
    "B": "Actively moving (the queried arm is clearly in motion)",
}
T9_CHOICES = {
    "A": "Frame X happens before Frame Y",
    "B": "Frame Y happens before Frame X",
}
TEMPORAL_DISPLAY_LABELS = ["X", "Y", "Z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AIST pilot GT from selected20 hdf5 episodes.")
    parser.add_argument("--selected-root", default="/data/projects/GM-100/aist-bimanip/selected20")
    parser.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/aist_pilot_v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-task-type", type=int, default=30)
    parser.add_argument("--camera", default="cam_high")
    return parser.parse_args()


def _decode_prompt(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def iter_episodes(root: Path) -> list[Path]:
    return sorted(root.glob("task_*/*/episode_*.hdf5"))


def recording_id(path: Path) -> str:
    return f"{path.parents[1].name}_{path.stem}"


def split_arm_speed(qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # AIST ALOHA state/action convention is 14D: 7D left + 7D right.
    left = np.linalg.norm(qvel[:, :6], axis=1)
    right = np.linalg.norm(qvel[:, 7:13], axis=1)
    return left, right


def smooth(x: np.ndarray, width: int = 9) -> np.ndarray:
    if len(x) < width:
        return x.astype(float)
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def sample_centers(n: int, count: int, guard: int, rng: random.Random) -> list[int]:
    lo = guard
    hi = n - guard - 1
    if hi <= lo:
        return []
    grid = np.linspace(lo, hi, num=min(count * 3, max(1, hi - lo + 1)))
    vals = sorted({int(round(x)) for x in grid})
    rng.shuffle(vals)
    return vals[:count]


def item_base(path: Path, f: h5py.File, camera: str) -> dict[str, Any]:
    task_id = path.parents[1].name
    task_name = str(f.attrs.get("Task name", path.parent.name))
    taxonomy = str(f.attrs.get("taxonomy", ""))
    prompt = _decode_prompt(f["text/prompt"][()]) if "text/prompt" in f else task_name.replace("_", " ")
    return {
        "dataset": "AIST-Bimanual",
        "task_id": task_id,
        "task_name": task_name,
        "taxonomy": taxonomy,
        "recording_id": recording_id(path),
        "episode_file": str(path),
        "camera": camera,
        "task_meta_description": prompt,
        "frame_rate": int(f.attrs.get("frame_rate", 50)),
    }


def build_t4(path: Path, f: h5py.File, rng: random.Random, camera: str) -> list[dict[str, Any]]:
    qvel = np.asarray(f["observations/qvel"])
    n = qvel.shape[0]
    left, right = map(smooth, split_arm_speed(qvel))
    # Per-episode robust thresholds. These are intentionally conservative for audit pilot.
    l_hi = max(float(np.quantile(left, 0.70)), 0.08)
    r_hi = max(float(np.quantile(right, 0.70)), 0.08)
    centers = sample_centers(n, count=24, guard=max(abs(x) for x in T4_CONTEXT_OFFSETS), rng=rng)
    rows: list[dict[str, Any]] = []
    base = item_base(path, f, camera)
    seen_labels: Counter[str] = Counter()
    for c in centers:
        idxs = [c + off for off in T4_CONTEXT_OFFSETS]
        l_active = float(np.mean(left[idxs])) >= l_hi
        r_active = float(np.mean(right[idxs])) >= r_hi
        if l_active and r_active:
            label = "both_active"
        elif l_active:
            label = "left_only"
        elif r_active:
            label = "right_only"
        else:
            label = "both_idle"
        if seen_labels[label] >= 3:
            continue
        seen_labels[label] += 1
        answer = T4_LABEL_TO_ANSWER[label]
        rows.append({
            **base,
            "task_type": "T4",
            "frame_index": c,
            "frame_indices": idxs,
            "context_offsets": list(T4_CONTEXT_OFFSETS),
            "question": "Across these time-ordered frames, which bimanual activity state best describes the robot?",
            "choices": T4_CHOICES,
            "answer": answer,
            "label": label,
            "left_speed_mean": round(float(np.mean(left[idxs])), 6),
            "right_speed_mean": round(float(np.mean(right[idxs])), 6),
            "left_threshold": round(l_hi, 6),
            "right_threshold": round(r_hi, 6),
        })
    return rows


def build_t6(path: Path, f: h5py.File, rng: random.Random, camera: str) -> list[dict[str, Any]]:
    qvel = np.asarray(f["observations/qvel"])
    n = qvel.shape[0]
    left, right = map(smooth, split_arm_speed(qvel))
    rows: list[dict[str, Any]] = []
    base = item_base(path, f, camera)
    for arm, speed in [("left", left), ("right", right)]:
        lo = max(float(np.quantile(speed, 0.25)), 0.01)
        hi = max(float(np.quantile(speed, 0.75)), 0.08)
        centers = sample_centers(n, count=28, guard=12, rng=rng)
        per_label: Counter[str] = Counter()
        for c in centers:
            idxs = [c-6, c-3, c, c+3, c+6]
            m = float(np.mean(speed[idxs]))
            if m <= lo:
                label = "stationary"
                answer = "A"
            elif m >= hi:
                label = "actively_moving"
                answer = "B"
            else:
                continue
            if per_label[label] >= 2:
                continue
            per_label[label] += 1
            rows.append({
                **base,
                "task_type": "T6",
                "query_arm": arm,
                "frame_index": c,
                "frame_indices": idxs,
                "question": f"Across these time-ordered frames, is the {arm} robot arm actively moving or stationary?",
                "choices": T6_CHOICES,
                "answer": answer,
                "speed_level_label": label,
                "speed_seq_mean": round(m, 6),
                "stationary_threshold": round(lo, 6),
                "active_threshold": round(hi, 6),
            })
    return rows


def build_t9(path: Path, f: h5py.File, rng: random.Random, camera: str) -> list[dict[str, Any]]:
    n = int(f["observations/qpos"].shape[0])
    base = item_base(path, f, camera)
    rows: list[dict[str, Any]] = []
    pairs = [(0.18, 0.72), (0.25, 0.62), (0.35, 0.82)]
    for pair_idx, (a, b) in enumerate(pairs):
        i = int(round(a * (n - 1)))
        j = int(round(b * (n - 1)))
        if abs(j - i) < 30:
            continue
        if rng.random() < 0.5:
            frame_indices = [i, j]
            display_labels = ["X", "Y"]
            answer = "A"
        else:
            frame_indices = [j, i]
            display_labels = ["X", "Y"]
            answer = "B"
        rows.append({
            **base,
            "task_type": "T9",
            "frame_index": frame_indices[0],
            "frame_indices": frame_indices,
            "question": "Which displayed frame happened earlier in the demonstration?",
            "choices": T9_CHOICES,
            "answer": answer,
            "display_labels": display_labels,
            "chronological_frame_indices": [i, j],
            "pair_index": pair_idx,
        })
    return rows


def build_t8(path: Path, f: h5py.File, rng: random.Random, camera: str) -> list[dict[str, Any]]:
    n = int(f["observations/qpos"].shape[0])
    if n < 80:
        return []
    base = item_base(path, f, camera)
    rows: list[dict[str, Any]] = []
    anchors = [int(round(x)) for x in np.linspace(0.05 * (n - 1), 0.95 * (n - 1), num=18)]
    anchors = sorted({max(0, min(n - 1, a)) for a in anchors})
    if len(anchors) < 3:
        return rows

    for i in range(len(anchors) - 2):
        triplet = [int(anchors[i]), int(anchors[i + 1]), int(anchors[i + 2])]
        if len(set(triplet)) < 3:
            continue
        if (triplet[1] - triplet[0]) < 8 or (triplet[2] - triplet[1]) < 8:
            continue
        order = [0, 1, 2]
        rng.shuffle(order)
        frames_shuf = [triplet[idx] for idx in order]
        labels = list(TEMPORAL_DISPLAY_LABELS)
        rng.shuffle(labels)
        time_rank = sorted(range(3), key=lambda idx: frames_shuf[idx])
        answer = "".join(labels[idx] for idx in time_rank)
        rows.append(
            {
                **base,
                "task_type": "T_temporal",
                "question": "Order these three frames from earliest to latest in the manipulation sequence.",
                "frame_indices": frames_shuf,
                "shuffled_labels": labels,
                "answer": answer,
                "chronological_frame_indices": triplet,
            }
        )
    return rows


def cap_rows(rows: list[dict[str, Any]], per_type: int, rng: random.Random) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task_type in ["T4", "T6", "T9", "T_temporal"]:
        subset = [r for r in rows if r.get("task_type") == task_type]
        rng.shuffle(subset)
        out.extend(subset[:per_type])
    return out


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
            rows.extend(build_t4(ep, f, rng, args.camera))
            rows.extend(build_t6(ep, f, rng, args.camera))
            rows.extend(build_t9(ep, f, rng, args.camera))
            rows.extend(build_t8(ep, f, rng, args.camera))

    all_path = gt_dir / "aist_pilot_pool.jsonl"
    with all_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    pilot = cap_rows(rows, args.per_task_type, rng)
    pilot_path = output_dir / "aist_pilot_v0.jsonl"
    with pilot_path.open("w", encoding="utf-8") as fh:
        for row in pilot:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = Counter(r["task_type"] for r in rows)
    pilot_by_type = Counter(r["task_type"] for r in pilot)
    summary = {
        "selected_root": str(selected_root),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "pool_counts": dict(by_type),
        "pilot_counts": dict(pilot_by_type),
        "pilot_jsonl": str(pilot_path),
        "pool_jsonl": str(all_path),
    }
    (output_dir / "aist_pilot_v0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
