#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GT_BUILD_DIR = THIS_DIR.parent
for _path in (THIS_DIR, GT_BUILD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reassemble_utils import shuffled_binary_choices, shuffled_multiple_choice
from rh20t_utils import (
    DEFAULT_EXTRACTED_ROOT,
    LONG_CONTEXT_OFFSETS,
    PHASE_NAMES,
    PRIMARY_CAMERA,
    T6_ACTIVE_THRESHOLD,
    T6_STATIONARY_THRESHOLD,
    center_from_progress,
    context_indices_from_center,
    evenly_spaced_sample,
    load_scene_signals,
    load_selected_scenes,
    load_task_catalog,
    phase_candidate_rows,
    progress_context,
    stable_subrange,
)


TASK_TO_FILENAME = {
    "T1": "t1_gt_items.jsonl",
    "T2": "t2_gt_items.jsonl",
    "T6": "t6_gt_items.jsonl",
    "T_temporal": "t_temporal_gt_items.jsonl",
    "T_binary": "t_binary_gt_items.jsonl",
    "T_progress": "t_progress_gt_items.jsonl",
}
TASK_ALIASES = {"T5": "T_progress", "T8": "T_temporal", "T9": "T_binary"}

PROGRESS_BIN_SPECS = [
    ("early", 0.25, "Early stage (the current local step has just started)"),
    ("middle", 0.50, "Middle stage (the current local step is underway)"),
    ("late", 0.75, "Late stage (the current local step is nearly complete)"),
]
PROGRESS_CHOICES = {k: text for k, (_, _, text) in zip(["A", "B", "C"], PROGRESS_BIN_SPECS)}
PROGRESS_BIN_TO_ANSWER = {name: key for key, (name, _, _) in zip(["A", "B", "C"], PROGRESS_BIN_SPECS)}

T6_LABEL_TO_TEXT = {
    "actively_moving": "Actively moving (the arm is clearly in motion)",
    "stationary": "Stationary (the arm is still or barely moving)",
}
T6_CHOICE_KEYS = ["A", "B"]

TEMPORAL_DISPLAY_LABELS = ["X", "Y", "Z"]
BINARY_CHOICES = {
    "X": "Image X happened earlier",
    "Y": "Image Y happened earlier",
}
BINARY_QUESTION = (
    "A single comparison image shows two labeled panels from the same robot manipulation scene. "
    "The labels X and Y are arbitrary identifiers and do not indicate temporal order. "
    "Which labeled panel happened earlier in the real manipulation sequence?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RH20T cfg2 pilot GT pools from partially extracted scene folders.")
    parser.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    parser.add_argument(
        "--scene-list-json",
        default="",
        help="Optional selected-scenes json. Defaults to the pilot selected_scenes.json when omitted.",
    )
    parser.add_argument(
        "--tasks",
        default="T1,T2,T5,T6,T8,T9",
        help="Comma-separated task list. Aliases: T5->T_progress, T8->T_temporal, T9->T_binary.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/projects/GM-100/benchmark/rh20t_cfg2_pilot_v0/pools",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", default=PRIMARY_CAMERA)
    return parser.parse_args()


def normalize_tasks_csv(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(text).split(","):
        item = token.strip()
        if not item:
            continue
        item = TASK_ALIASES.get(item, item)
        if item not in TASK_TO_FILENAME:
            raise ValueError(f"Unsupported task: {item}")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_item(scene, task_type: str) -> dict:
    return {
        "dataset": "RH20T",
        "task_id": scene.task_id,
        "recording_id": scene.scene_dir,
        "scene_dir": scene.scene_dir,
        "camera": scene.camera,
        "task_type": task_type,
        "arm_type": "single_arm",
        "task_meta_description": scene.task_description,
        "rating": int(scene.rating),
        "calib_quality": int(scene.calib_quality),
    }


def mean_speed(scene, frame_indices: list[int]) -> float:
    if not frame_indices:
        return 0.0
    return float(np.mean(scene.speed_ema[np.asarray(frame_indices, dtype=int)]))


def build_phase_item(
    *,
    scene,
    frame_index: int,
    phase_name: str,
    phase_source: str,
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_multiple_choice(phase_name, pool=list(PHASE_NAMES), rng=rng, num_choices=4)
    item = base_item(scene, "T1")
    item.update(
        {
            "frame_index": int(frame_index),
            "question": "Which phase best describes the current robot manipulation state?",
            "choices": choices,
            "answer": answer,
            "phase_label": phase_name,
            "phase_source": phase_source,
        }
    )
    return item


def build_contact_item(
    *,
    scene,
    frame_index: int,
    is_contact: bool,
    label_source: str,
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_binary_choices(
        is_contact,
        yes_text="Yes — contact established",
        no_text="No — no contact",
        rng=rng,
    )
    item = base_item(scene, "T2")
    item.update(
        {
            "frame_index": int(frame_index),
            "question": "Is the end effector currently in contact with anything?",
            "choices": choices,
            "answer": answer,
            "label": "contact" if is_contact else "no_contact",
            "label_source": label_source,
        }
    )
    return item


def build_progress_item(
    *,
    scene,
    interval,
    frame_index: int,
    frame_indices: list[int],
    progress_bin: str,
    progress_value: float,
) -> dict:
    item = base_item(scene, "T_progress")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "Across these time-ordered frames, how far along is the current manipulation step?",
            "choices": dict(PROGRESS_CHOICES),
            "answer": PROGRESS_BIN_TO_ANSWER[progress_bin],
            "progress_bin": progress_bin,
            "progress_value": round(float(progress_value), 4),
            "interval_id": interval.interval_id,
            "interval_span": int(interval.span),
            "interval_start": int(interval.start),
            "interval_end": int(interval.end),
        }
    )
    return item


def build_t6_item(
    *,
    scene,
    frame_index: int,
    frame_indices: list[int],
    speed_seq_mean: float,
    label: str,
    source: str,
    stationary_threshold: float,
    active_threshold: float,
    rng: random.Random,
) -> dict:
    options = ["actively_moving", "stationary"]
    rng.shuffle(options)
    choices = {k: T6_LABEL_TO_TEXT[v] for k, v in zip(T6_CHOICE_KEYS, options)}
    answer = T6_CHOICE_KEYS[options.index(label)]
    item = base_item(scene, "T6")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "Across these time-ordered frames, is the robot arm actively moving or stationary?",
            "choices": choices,
            "answer": answer,
            "speed_seq_mean": round(float(speed_seq_mean), 6),
            "speed_level_label": label,
            "speed_source": source,
            "stationary_threshold": round(float(stationary_threshold), 6),
            "fast_threshold": round(float(active_threshold), 6),
        }
    )
    return item


def build_temporal_item(
    *,
    scene,
    frame_indices: list[int],
    interval_id: str,
    rng: random.Random,
) -> dict:
    order = [0, 1, 2]
    rng.shuffle(order)
    frames_shuf = [int(frame_indices[i]) for i in order]
    labels = list(TEMPORAL_DISPLAY_LABELS)
    rng.shuffle(labels)
    time_rank = sorted(range(3), key=lambda idx: frames_shuf[idx])
    answer = "".join(labels[idx] for idx in time_rank)
    item = base_item(scene, "T_temporal")
    item.update(
        {
            "question": "Order these three frames from earliest to latest in the manipulation sequence.",
            "frame_indices": [int(x) for x in frames_shuf],
            "shuffled_labels": labels,
            "answer": answer,
            "interval_id": interval_id,
        }
    )
    return item


def build_binary_item(
    *,
    scene,
    frame_indices: list[int],
    difficulty: str,
    interval_ids: list[str],
    rng: random.Random,
) -> dict:
    frames = [int(x) for x in frame_indices]
    if rng.random() < 0.5:
        display_frames = frames
    else:
        display_frames = [frames[1], frames[0]]
    display_labels = ["X", "Y"]
    if rng.random() < 0.5:
        display_labels = ["Y", "X"]
    answer = display_labels[0] if display_frames[0] < display_frames[1] else display_labels[1]
    item = base_item(scene, "T_binary")
    item.update(
        {
            "question": BINARY_QUESTION,
            "frame_indices": [int(x) for x in display_frames],
            "display_labels": display_labels,
            "choices": dict(BINARY_CHOICES),
            "answer": answer,
            "difficulty": difficulty,
            "interval_ids": interval_ids,
            "frame_gap": int(abs(frames[0] - frames[1])),
        }
    )
    return item


def fallback_contact_rows(scene, want_contact: bool, max_count: int) -> list[int]:
    if want_contact:
        rows = np.where(scene.contact_mask)[0].tolist()
    else:
        mask = (~scene.contact_mask) & (scene.speed_ema <= max(scene.speed_event_lo, 0.028))
        rows = np.where(mask)[0].tolist()
    return evenly_spaced_sample(rows, max_count=max_count, min_gap=18)


def build_scene_items(scene, rng: random.Random, enabled_tasks: set[str]) -> dict[str, list[dict]]:
    per_task: dict[str, list[dict]] = defaultdict(list)
    phase_rows = phase_candidate_rows(scene)

    if "T1" in enabled_tasks:
        for phase_name, rows in phase_rows.items():
            for frame_index in rows:
                per_task["T1"].append(
                    build_phase_item(
                        scene=scene,
                        frame_index=int(frame_index),
                        phase_name=phase_name,
                        phase_source="signal_phase_candidates",
                        rng=rng,
                    )
                )

    if "T2" in enabled_tasks:
        yes_rows = evenly_spaced_sample(
            phase_rows.get("contact", []) + phase_rows.get("hold and carry", []),
            max_count=6,
            min_gap=18,
        )
        no_rows = evenly_spaced_sample(
            phase_rows.get("pre-approach", []) + phase_rows.get("approach", []),
            max_count=6,
            min_gap=18,
        )
        if len(yes_rows) < 3:
            yes_rows = evenly_spaced_sample(yes_rows + fallback_contact_rows(scene, True, max_count=6), max_count=6, min_gap=18)
        if len(no_rows) < 3:
            no_rows = evenly_spaced_sample(no_rows + fallback_contact_rows(scene, False, max_count=6), max_count=6, min_gap=18)
        for frame_index in yes_rows:
            per_task["T2"].append(
                build_contact_item(
                    scene=scene,
                    frame_index=int(frame_index),
                    is_contact=True,
                    label_source="contact_phase_or_mask",
                    rng=rng,
                )
            )
        for frame_index in no_rows:
            per_task["T2"].append(
                build_contact_item(
                    scene=scene,
                    frame_index=int(frame_index),
                    is_contact=False,
                    label_source="noncontact_phase_or_mask",
                    rng=rng,
                )
            )

    if "T_progress" in enabled_tasks:
        for interval in scene.intervals:
            if interval.span < 24:
                continue
            for progress_bin, progress_value, _ in PROGRESS_BIN_SPECS:
                ctx = progress_context(interval.start, interval.end, progress=progress_value, offsets=LONG_CONTEXT_OFFSETS)
                if ctx is None:
                    continue
                center, frames = ctx
                per_task["T_progress"].append(
                    build_progress_item(
                        scene=scene,
                        interval=interval,
                        frame_index=center,
                        frame_indices=frames,
                        progress_bin=progress_bin,
                        progress_value=progress_value,
                    )
                )

    if "T6" in enabled_tasks:
        active_threshold = float(max(T6_ACTIVE_THRESHOLD, scene.speed_event_hi * 0.90))
        stationary_threshold = float(max(T6_STATIONARY_THRESHOLD, min(scene.speed_event_lo, 0.026)))
        active_rows = evenly_spaced_sample(phase_rows.get("transfer", []), max_count=4, min_gap=18)
        stationary_rows = evenly_spaced_sample(
            phase_rows.get("pre-approach", []) + phase_rows.get("hold and carry", []),
            max_count=4,
            min_gap=18,
        )
        if len(active_rows) < 2:
            active_rows = evenly_spaced_sample(
                active_rows + np.where(scene.speed_ema >= active_threshold)[0].tolist(),
                max_count=4,
                min_gap=18,
            )
        if len(stationary_rows) < 2:
            stationary_rows = evenly_spaced_sample(
                stationary_rows + np.where((scene.speed_ema <= stationary_threshold) & (~scene.contact_mask))[0].tolist(),
                max_count=4,
                min_gap=18,
            )

        for source_name, label, centers in (
            ("transfer_or_speed_peak", "actively_moving", active_rows),
            ("pre_or_hold_stable", "stationary", stationary_rows),
        ):
            for center in centers:
                frames = context_indices_from_center(int(center), LONG_CONTEXT_OFFSETS, lo=0, hi=scene.n_rows - 1)
                if frames is None:
                    continue
                speed_seq_mean = mean_speed(scene, frames)
                if label == "actively_moving" and speed_seq_mean < active_threshold:
                    continue
                if label == "stationary" and speed_seq_mean > stationary_threshold:
                    continue
                per_task["T6"].append(
                    build_t6_item(
                        scene=scene,
                        frame_index=int(center),
                        frame_indices=frames,
                        speed_seq_mean=speed_seq_mean,
                        label=label,
                        source=source_name,
                        stationary_threshold=stationary_threshold,
                        active_threshold=active_threshold,
                        rng=rng,
                    )
                )

    if "T_temporal" in enabled_tasks:
        for interval in scene.intervals:
            stable = stable_subrange(interval.start, interval.end, min_margin_frames=4, margin_ratio=0.10)
            if stable is None:
                continue
            lo, hi = stable
            frames = [
                center_from_progress(lo, hi, 0.18),
                center_from_progress(lo, hi, 0.50),
                center_from_progress(lo, hi, 0.82),
            ]
            if len(set(frames)) < 3:
                continue
            if any((b - a) < 8 for a, b in zip(frames[:-1], frames[1:])):
                continue
            per_task["T_temporal"].append(
                build_temporal_item(
                    scene=scene,
                    frame_indices=frames,
                    interval_id=interval.interval_id,
                    rng=rng,
                )
            )

    if "T_binary" in enabled_tasks:
        for interval in scene.intervals:
            stable = stable_subrange(interval.start, interval.end, min_margin_frames=4, margin_ratio=0.10)
            if stable is None:
                continue
            lo, hi = stable
            left = center_from_progress(lo, hi, 0.20)
            right = center_from_progress(lo, hi, 0.80)
            if right - left < 10:
                continue
            per_task["T_binary"].append(
                build_binary_item(
                    scene=scene,
                    frame_indices=[left, right],
                    difficulty="same_interval_far",
                    interval_ids=[interval.interval_id],
                    rng=rng,
                )
            )
        for left_iv, right_iv in zip(scene.intervals[:-1], scene.intervals[1:]):
            left_stable = stable_subrange(left_iv.start, left_iv.end, min_margin_frames=4, margin_ratio=0.10)
            right_stable = stable_subrange(right_iv.start, right_iv.end, min_margin_frames=4, margin_ratio=0.10)
            if left_stable is None or right_stable is None:
                continue
            left_frame = center_from_progress(left_stable[0], left_stable[1], 0.80)
            right_frame = center_from_progress(right_stable[0], right_stable[1], 0.20)
            if right_frame <= left_frame:
                continue
            per_task["T_binary"].append(
                build_binary_item(
                    scene=scene,
                    frame_indices=[left_frame, right_frame],
                    difficulty="cross_interval",
                    interval_ids=[left_iv.interval_id, right_iv.interval_id],
                    rng=rng,
                )
            )

    return per_task


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    enabled_tasks = set(normalize_tasks_csv(args.tasks))
    extracted_root = Path(args.extracted_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_list_json:
        selected_scenes = load_selected_scenes(Path(args.scene_list_json))
    else:
        selected_scenes = load_selected_scenes()
    task_catalog = load_task_catalog()

    per_task: dict[str, list[dict]] = defaultdict(list)
    scene_summaries: list[dict] = []
    skipped_scenes: list[dict] = []

    for row in selected_scenes:
        task_id = str(row["task_id"])
        scene_dir = str(row["scene_dir"])
        task_info = task_catalog.get(task_id, {})
        task_desc = str(task_info.get("task_description_english") or task_info.get("task_description_chinese") or "").strip()
        try:
            scene = load_scene_signals(
                extracted_root,
                scene_dir=scene_dir,
                task_description=task_desc,
                camera=args.camera,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_scenes.append(
                {
                    "scene_dir": scene_dir,
                    "task_id": task_id,
                    "reason": str(exc),
                }
            )
            continue
        scene_items = build_scene_items(scene, rng=rng, enabled_tasks=enabled_tasks)
        scene_summary = {
            "scene_dir": scene.scene_dir,
            "task_id": scene.task_id,
            "rating": int(scene.rating),
            "calib_quality": int(scene.calib_quality),
            "n_rows": int(scene.n_rows),
            "n_intervals": int(len(scene.intervals)),
            "counts": {},
        }
        for task_type, items in scene_items.items():
            per_task[task_type].extend(items)
            scene_summary["counts"][task_type] = len(items)
        scene_summaries.append(scene_summary)

    counts: dict[str, int] = {}
    combined: list[dict] = []
    for task_type in TASK_TO_FILENAME:
        if task_type not in enabled_tasks:
            continue
        rows = per_task.get(task_type, [])
        counts[task_type] = len(rows)
        write_jsonl(output_dir / TASK_TO_FILENAME[task_type], rows)
        combined.extend(rows)

    write_jsonl(output_dir / "all_gt_items.jsonl", combined)
    summary = {
        "dataset": "RH20T",
        "camera": args.camera,
        "tasks": sorted(enabled_tasks),
        "num_selected_scenes": len(selected_scenes),
        "num_processed_scenes": len(scene_summaries),
        "num_skipped_scenes": len(skipped_scenes),
        "skipped_scenes": skipped_scenes,
        "counts": counts,
        "all_gt_items_jsonl": str(output_dir / "all_gt_items.jsonl"),
        "scene_summaries": scene_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
