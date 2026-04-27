#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from reassemble_utils import (
    DEFAULT_CAMERA,
    LONG_CONTEXT_OFFSETS,
    PHASE_NAMES,
    REASSEMBLE_ROOT,
    SHORT_CONTEXT_OFFSETS,
    camera_timestamps,
    distinct_low_level_chain,
    filter_recordings,
    high_level_text,
    high_level_verb,
    is_no_action,
    load_low_level_vocab,
    load_recording_index,
    low_level_segments,
    low_level_to_contact,
    low_level_to_phase,
    midpoint_frame,
    next_distinct_low_level,
    non_no_action_segments,
    object_from_high_level_text,
    progress_context,
    relative_progress_context,
    recording_h5_path,
    shuffled_binary_choices,
    shuffled_multiple_choice,
    speed_window_mean,
    timestamp_interval_to_frame_range,
    velocity_stream,
)


TASK_TO_FILENAME = {
    "T1": "t1_gt_items.jsonl",
    "T2": "t2_gt_items.jsonl",
    "T6": "t6_gt_items.jsonl",
    "T7": "t7_gt_items.jsonl",
    "T_temporal": "t_temporal_gt_items.jsonl",
    "T_binary": "t_binary_gt_items.jsonl",
    "T_progress": "t_progress_gt_items.jsonl",
    "T10": "t10_gt_items.jsonl",
    "T11": "t11_gt_items.jsonl",
    "T12": "t12_gt_items.jsonl",
}
TASK_ALIASES = {
    "T5": "T_progress",
    "T8": "T_temporal",
    "T9": "T_binary",
}
TEMPORAL_DISPLAY_LABELS = ["X", "Y", "Z"]
BINARY_CHOICES = {
    "X": "Image X happened earlier",
    "Y": "Image Y happened earlier",
}
BINARY_QUESTION = (
    "A single comparison image shows two labeled panels from the same robot manipulation recording. "
    "The labels X and Y are arbitrary identifiers and do not indicate temporal order. "
    "Which labeled panel happened earlier in the real manipulation sequence?"
)
PROGRESS_BIN_SPECS = [
    ("early", 0.25, "Early stage (this manipulation step has just started)"),
    ("middle", 0.50, "Middle stage (this manipulation step is about halfway complete)"),
    ("late", 0.75, "Late stage (this manipulation step is nearly finished)"),
]
PROGRESS_CHOICES = {k: text for k, (_, _, text) in zip(["A", "B", "C"], PROGRESS_BIN_SPECS)}
PROGRESS_BIN_TO_ANSWER = {name: key for key, (name, _, _) in zip(["A", "B", "C"], PROGRESS_BIN_SPECS)}
T6_LABEL_TO_TEXT = {
    "actively_moving": "Actively moving (the arm is clearly in motion)",
    "stationary": "Stationary (the arm is still or barely moving)",
}
T6_CHOICE_KEYS = ["A", "B"]
T6_RELATIVE_PROGRESS_VALUES = (0.15, 0.325, 0.50, 0.675, 0.85)
T7_RELATIVE_PROGRESS_VALUES = (0.10, 0.18, 0.26, 0.34)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build REASSEMBLE GT suite from official segments_info and extracted test/train recordings.")
    parser.add_argument("--dataset-root", default=str(REASSEMBLE_ROOT))
    parser.add_argument("--split", default="test_split1")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument(
        "--tasks",
        default="T1,T2,T5,T6,T7,T8,T9,T10,T11,T12",
        help="Comma-separated task list. Aliases: T5->T_progress, T8->T_temporal, T9->T_binary.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/projects/GM-100/benchmark/reassemble_test_split1_suite_v0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-recordings", type=int, default=0, help="0 means all recordings in split")
    parser.add_argument("--t6-low-quantile", type=float, default=0.25)
    parser.add_argument("--t6-high-quantile", type=float, default=0.75)
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
        out.append(item)
        seen.add(item)
    return out


def base_item(recording_id: str, camera: str, task_type: str) -> dict:
    return {
        "dataset": "REASSEMBLE",
        "task_id": recording_id,
        "recording_id": recording_id,
        "camera": camera,
        "task_type": task_type,
        "arm_type": "single_arm",
    }


def build_phase_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    phase_name: str,
    phase_source: str,
    high_text_raw: str,
    object_raw: str,
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_multiple_choice(phase_name, pool=list(PHASE_NAMES), rng=rng, num_choices=4)
    item = base_item(recording_id, camera, "T1")
    item.update(
        {
            "frame_index": int(frame_index),
            "question": "Which phase best describes the current robot manipulation state?",
            "choices": choices,
            "answer": answer,
            "phase_label": phase_name,
            "phase_source": phase_source,
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
        }
    )
    return item


def build_contact_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    is_contact: bool,
    high_text_raw: str,
    object_raw: str,
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_binary_choices(
        is_contact,
        yes_text="Yes — contact established",
        no_text="No — no contact",
        rng=rng,
    )
    item = base_item(recording_id, camera, "T2")
    item.update(
        {
            "frame_index": int(frame_index),
            "question": "Is the end effector currently in contact with anything?",
            "choices": choices,
            "answer": answer,
            "label": "contact" if is_contact else "no_contact",
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
        }
    )
    return item


def build_t7_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    segment_index: int,
    high_text_raw: str,
    object_raw: str,
    success: bool,
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_binary_choices(
        success,
        yes_text="Yes — this operation will succeed",
        no_text="No — this operation will fail",
        rng=rng,
    )
    item = base_item(recording_id, camera, "T7")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "Based on these early frames of the current operation, will this operation eventually succeed?",
            "choices": choices,
            "answer": answer,
            "segment_index": int(segment_index),
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "segment_success": bool(success),
            "frame_sampling": "relative_progress_early",
            "relative_progresses": [float(x) for x in T7_RELATIVE_PROGRESS_VALUES],
        }
    )
    return item


def build_progress_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    segment_index: int,
    high_text_raw: str,
    object_raw: str,
    progress_bin: str,
    progress_value: float,
) -> dict:
    item = base_item(recording_id, camera, "T_progress")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "Across these time-ordered frames, how far along is the current manipulation step?",
            "choices": dict(PROGRESS_CHOICES),
            "answer": PROGRESS_BIN_TO_ANSWER[progress_bin],
            "progress_bin": progress_bin,
            "progress_value": round(float(progress_value), 4),
            "segment_index": int(segment_index),
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "interval_id": f"{recording_id}:{segment_index}",
        }
    )
    return item


def build_t10_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    segment_index: int,
    low_index: int,
    high_text_raw: str,
    object_raw: str,
    low_text_raw: str,
    low_vocab: list[str],
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_multiple_choice(low_text_raw, pool=low_vocab, rng=rng, num_choices=4)
    item = base_item(recording_id, camera, "T10")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "What is the current low-level action in these ordered frames?",
            "choices": choices,
            "answer": answer,
            "segment_index": int(segment_index),
            "low_index": int(low_index),
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "low_level_text_raw": low_text_raw,
        }
    )
    return item


def build_t11_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    segment_index: int,
    low_index: int,
    high_text_raw: str,
    object_raw: str,
    current_low_text: str,
    next_low_text: str,
    low_vocab: list[str],
    rng: random.Random,
) -> dict:
    pool = [x for x in low_vocab if x != current_low_text]
    choices, answer = shuffled_multiple_choice(next_low_text, pool=pool, rng=rng, num_choices=4)
    item = base_item(recording_id, camera, "T11")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": f"Within the ongoing high-level action '{high_text_raw}', what low-level action should happen next?",
            "choices": choices,
            "answer": answer,
            "segment_index": int(segment_index),
            "low_index": int(low_index),
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "current_low_level_text_raw": current_low_text,
            "next_low_level_text_raw": next_low_text,
        }
    )
    return item


def build_t12_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    segment_index: int,
    masked_low_index: int,
    high_text_raw: str,
    object_raw: str,
    prev_text: str,
    masked_text: str,
    next_text: str,
    low_vocab: list[str],
    rng: random.Random,
) -> dict:
    choices, answer = shuffled_multiple_choice(masked_text, pool=low_vocab, rng=rng, num_choices=4)
    item = base_item(recording_id, camera, "T12")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": (
                f"Within the high-level action '{high_text_raw}', which action best fills [MASK] in this local chain: "
                f"{prev_text} > [MASK] > {next_text}?"
            ),
            "choices": choices,
            "answer": answer,
            "segment_index": int(segment_index),
            "masked_low_index": int(masked_low_index),
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "masked_low_level_text_raw": masked_text,
            "local_chain_context": [prev_text, "[MASK]", next_text],
        }
    )
    return item


def build_t6_item(
    *,
    recording_id: str,
    camera: str,
    frame_index: int,
    frame_indices: list[int],
    speed_seq_mean: float,
    label: str,
    high_text_raw: str,
    object_raw: str,
    low_quantile_threshold: float,
    high_quantile_threshold: float,
    rng: random.Random,
) -> dict:
    options = ["actively_moving", "stationary"]
    rng.shuffle(options)
    choices = {k: T6_LABEL_TO_TEXT[v] for k, v in zip(T6_CHOICE_KEYS, options)}
    answer = T6_CHOICE_KEYS[options.index(label)]
    item = base_item(recording_id, camera, "T6")
    item.update(
        {
            "frame_index": int(frame_index),
            "frame_indices": [int(x) for x in frame_indices],
            "question": "Across these time-ordered frames, is the robot arm actively moving or stationary?",
            "choices": choices,
            "answer": answer,
            "speed_seq_mean": round(float(speed_seq_mean), 6),
            "speed_level_label": label,
            "high_level_text_raw": high_text_raw,
            "object_raw": object_raw,
            "stationary_threshold": round(float(low_quantile_threshold), 6),
            "fast_threshold": round(float(high_quantile_threshold), 6),
            "frame_sampling": "relative_progress_window",
            "relative_progresses": [float(x) for x in T6_RELATIVE_PROGRESS_VALUES],
        }
    )
    return item


def valid_frame_range(camera_ts: np.ndarray, start_time: float, end_time: float) -> tuple[int, int] | None:
    frame_range = timestamp_interval_to_frame_range(camera_ts, start_time, end_time)
    if frame_range is None:
        return None
    lo, hi = frame_range
    if hi <= lo:
        return None
    return lo, hi


def segment_midpoint_frame(camera_ts: np.ndarray, segment: dict) -> int | None:
    frame_range = valid_frame_range(camera_ts, float(segment.get("start", 0.0)), float(segment.get("end", 0.0)))
    if frame_range is None:
        return None
    lo, hi = frame_range
    return midpoint_frame(lo, hi)


def build_temporal_items_for_recording(recording_row: dict, camera_ts: np.ndarray, camera: str, rng: random.Random) -> list[dict]:
    recording_id = recording_row["recording_id"]
    segments = [seg for seg in non_no_action_segments(recording_row) if bool(seg.get("success", False))]
    eligible: list[tuple[dict, int]] = []
    for seg in segments:
        frame_idx = segment_midpoint_frame(camera_ts, seg)
        if frame_idx is None:
            continue
        eligible.append((seg, frame_idx))

    items: list[dict] = []
    for pos in range(max(0, len(eligible) - 2)):
        trio = eligible[pos : pos + 3]
        if len(trio) < 3:
            continue
        frames = [frame for _, frame in trio]
        if len(set(frames)) < 3:
            continue
        order = [0, 1, 2]
        rng.shuffle(order)
        frames_shuf = [frames[i] for i in order]
        segs_shuf = [trio[i][0] for i in order]
        labels = list(TEMPORAL_DISPLAY_LABELS)
        rng.shuffle(labels)
        time_rank = sorted(range(3), key=lambda idx: frames_shuf[idx])
        answer = "".join(labels[idx] for idx in time_rank)
        item = base_item(recording_id, camera, "T_temporal")
        item.update(
            {
                "question": "Order these three frames from earliest to latest in the manipulation sequence.",
                "frame_indices": [int(x) for x in frames_shuf],
                "shuffled_labels": labels,
                "answer": answer,
                "high_level_texts": [high_level_text(seg) for seg in segs_shuf],
            }
        )
        items.append(item)
    return items


def build_binary_items_for_recording(recording_row: dict, camera_ts: np.ndarray, camera: str, rng: random.Random) -> list[dict]:
    recording_id = recording_row["recording_id"]
    segments = [seg for seg in non_no_action_segments(recording_row) if bool(seg.get("success", False))]
    eligible: list[tuple[dict, int]] = []
    for seg in segments:
        frame_idx = segment_midpoint_frame(camera_ts, seg)
        if frame_idx is None:
            continue
        eligible.append((seg, frame_idx))

    items: list[dict] = []
    pair_specs = [(1, "adjacent_segment"), (2, "skip_one_segment")]
    for gap, difficulty in pair_specs:
        for pos in range(max(0, len(eligible) - gap)):
            left = eligible[pos]
            right = eligible[pos + gap]
            frames = [left[1], right[1]]
            if frames[0] == frames[1]:
                continue
            if rng.random() < 0.5:
                display_frames = frames
                display_segments = [left[0], right[0]]
            else:
                display_frames = [frames[1], frames[0]]
                display_segments = [right[0], left[0]]
            display_labels = ["X", "Y"]
            if rng.random() < 0.5:
                display_labels = ["Y", "X"]
            answer = display_labels[0] if display_frames[0] < display_frames[1] else display_labels[1]
            item = base_item(recording_id, camera, "T_binary")
            item.update(
                {
                    "question": BINARY_QUESTION,
                    "frame_indices": [int(x) for x in display_frames],
                    "display_labels": display_labels,
                    "choices": dict(BINARY_CHOICES),
                    "answer": answer,
                    "difficulty": difficulty,
                    "high_level_texts": [high_level_text(seg) for seg in display_segments],
                    "frame_gap": int(abs(frames[0] - frames[1])),
                }
            )
            items.append(item)
    return items


def build_recording_items(
    recording_row: dict,
    h5_path: Path,
    camera: str,
    low_vocab: list[str],
    enabled_tasks: set[str],
    rng: random.Random,
) -> tuple[dict[str, list[dict]], list[dict]]:
    per_task: dict[str, list[dict]] = defaultdict(list)
    t6_candidates: list[dict] = []
    recording_id = recording_row["recording_id"]

    with h5py.File(h5_path, "r") as h5_file:
        camera_ts = camera_timestamps(h5_file, camera=camera)
        velocity_ts = None
        translational_speed = None
        if "T6" in enabled_tasks:
            velocity_ts, translational_speed = velocity_stream(h5_file)

        if "T_temporal" in enabled_tasks:
            per_task["T_temporal"].extend(build_temporal_items_for_recording(recording_row, camera_ts, camera, rng))
        if "T_binary" in enabled_tasks:
            per_task["T_binary"].extend(build_binary_items_for_recording(recording_row, camera_ts, camera, rng))

        for seg in recording_row.get("segments", []):
            high_text_raw = high_level_text(seg)
            high_verb = high_level_verb(high_text_raw)
            object_raw = object_from_high_level_text(high_text_raw)
            segment_index = int(seg.get("segment_index", 0))
            seg_is_no_action = is_no_action(high_text_raw)
            segment_frame_range = valid_frame_range(camera_ts, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
            if segment_frame_range is None:
                continue
            seg_lo, seg_hi = segment_frame_range

            if seg_is_no_action:
                mid = midpoint_frame(seg_lo, seg_hi)
                if "T1" in enabled_tasks:
                    per_task["T1"].append(
                        build_phase_item(
                            recording_id=recording_id,
                            camera=camera,
                            frame_index=mid,
                            phase_name="pre-approach",
                            phase_source="no_action_segment",
                            high_text_raw=high_text_raw,
                            object_raw=object_raw,
                            rng=rng,
                        )
                    )
                if "T2" in enabled_tasks:
                    per_task["T2"].append(
                        build_contact_item(
                            recording_id=recording_id,
                            camera=camera,
                            frame_index=mid,
                            is_contact=False,
                            high_text_raw=high_text_raw,
                            object_raw=object_raw,
                            rng=rng,
                        )
                    )
                if "T6" in enabled_tasks and velocity_ts is not None and translational_speed is not None:
                    ctx = relative_progress_context(
                        seg_lo,
                        seg_hi,
                        progress_values=T6_RELATIVE_PROGRESS_VALUES,
                        min_margin_frames=2,
                        margin_ratio=0.05,
                    )
                    if ctx is not None:
                        center, frames = ctx
                        t6_candidates.append(
                            {
                                "recording_id": recording_id,
                                "camera": camera,
                                "frame_index": center,
                                "frame_indices": frames,
                                "speed_seq_mean": speed_window_mean(camera_ts, velocity_ts, translational_speed, frames),
                                "high_level_text_raw": high_text_raw,
                                "object_raw": object_raw,
                            }
                        )
                continue

            if "T7" in enabled_tasks:
                ctx = relative_progress_context(
                    seg_lo,
                    seg_hi,
                    progress_values=T7_RELATIVE_PROGRESS_VALUES,
                    min_margin_frames=2,
                    margin_ratio=0.03,
                )
                if ctx is not None:
                    center, frames = ctx
                    per_task["T7"].append(
                        build_t7_item(
                            recording_id=recording_id,
                            camera=camera,
                            frame_index=center,
                            frame_indices=frames,
                            segment_index=segment_index,
                            high_text_raw=high_text_raw,
                            object_raw=object_raw,
                            success=bool(seg.get("success", False)),
                            rng=rng,
                        )
                    )

            if "T_progress" in enabled_tasks and bool(seg.get("success", False)):
                for bin_name, progress, _ in PROGRESS_BIN_SPECS:
                    ctx = progress_context(seg_lo, seg_hi, progress=progress, offsets=LONG_CONTEXT_OFFSETS)
                    if ctx is None:
                        continue
                    center, frames = ctx
                    per_task["T_progress"].append(
                        build_progress_item(
                            recording_id=recording_id,
                            camera=camera,
                            frame_index=center,
                            frame_indices=frames,
                            segment_index=segment_index,
                            high_text_raw=high_text_raw,
                            object_raw=object_raw,
                            progress_bin=bin_name,
                            progress_value=progress,
                        )
                    )

            lows = low_level_segments(seg)
            for low_pos, low in enumerate(lows):
                frame_range = valid_frame_range(camera_ts, low["start"], low["end"])
                if frame_range is None:
                    continue
                low_lo, low_hi = frame_range
                low_text_raw = low["text"]
                mid = midpoint_frame(low_lo, low_hi)

                if "T1" in enabled_tasks:
                    phase_name = low_level_to_phase(low_text_raw, high_verb)
                    if phase_name is not None:
                        per_task["T1"].append(
                            build_phase_item(
                                recording_id=recording_id,
                                camera=camera,
                                frame_index=mid,
                                phase_name=phase_name,
                                phase_source=f"low_level:{low_text_raw}",
                                high_text_raw=high_text_raw,
                                object_raw=object_raw,
                                rng=rng,
                            )
                        )

                if "T2" in enabled_tasks:
                    is_contact_label = low_level_to_contact(low_text_raw)
                    if is_contact_label is not None:
                        per_task["T2"].append(
                            build_contact_item(
                                recording_id=recording_id,
                                camera=camera,
                                frame_index=mid,
                                is_contact=is_contact_label,
                                high_text_raw=high_text_raw,
                                object_raw=object_raw,
                                rng=rng,
                            )
                        )

                if "T6" in enabled_tasks and velocity_ts is not None and translational_speed is not None:
                    ctx = relative_progress_context(
                        low_lo,
                        low_hi,
                        progress_values=T6_RELATIVE_PROGRESS_VALUES,
                        min_margin_frames=1,
                        margin_ratio=0.05,
                    )
                    if ctx is not None:
                        center, frames = ctx
                        t6_candidates.append(
                            {
                                "recording_id": recording_id,
                                "camera": camera,
                                "frame_index": center,
                                "frame_indices": frames,
                                "speed_seq_mean": speed_window_mean(camera_ts, velocity_ts, translational_speed, frames),
                                "high_level_text_raw": high_text_raw,
                                "object_raw": object_raw,
                            }
                        )

                if "T10" in enabled_tasks:
                    ctx = progress_context(low_lo, low_hi, progress=0.5, offsets=SHORT_CONTEXT_OFFSETS)
                    if ctx is not None:
                        center, frames = ctx
                        per_task["T10"].append(
                            build_t10_item(
                                recording_id=recording_id,
                                camera=camera,
                                frame_index=center,
                                frame_indices=frames,
                                segment_index=segment_index,
                                low_index=int(low.get("low_index", low_pos)),
                                high_text_raw=high_text_raw,
                                object_raw=object_raw,
                                low_text_raw=low_text_raw,
                                low_vocab=low_vocab,
                                rng=rng,
                            )
                        )

                if "T11" in enabled_tasks:
                    nxt = next_distinct_low_level(seg, low_pos)
                    if nxt is not None:
                        ctx = progress_context(low_lo, low_hi, progress=0.75, offsets=SHORT_CONTEXT_OFFSETS)
                        if ctx is not None:
                            center, frames = ctx
                            per_task["T11"].append(
                                build_t11_item(
                                    recording_id=recording_id,
                                    camera=camera,
                                    frame_index=center,
                                    frame_indices=frames,
                                    segment_index=segment_index,
                                    low_index=int(low.get("low_index", low_pos)),
                                    high_text_raw=high_text_raw,
                                    object_raw=object_raw,
                                    current_low_text=low_text_raw,
                                    next_low_text=nxt["text"],
                                    low_vocab=low_vocab,
                                    rng=rng,
                                )
                            )

            if "T12" in enabled_tasks:
                distinct_chain = distinct_low_level_chain(seg)
                for masked_pos in range(1, max(0, len(distinct_chain) - 1)):
                    if masked_pos >= len(distinct_chain) - 1:
                        continue
                    masked = distinct_chain[masked_pos]
                    prev_item = distinct_chain[masked_pos - 1]
                    next_item = distinct_chain[masked_pos + 1]
                    frame_range = valid_frame_range(camera_ts, masked["start"], masked["end"])
                    if frame_range is None:
                        continue
                    low_lo, low_hi = frame_range
                    ctx = progress_context(low_lo, low_hi, progress=0.5, offsets=SHORT_CONTEXT_OFFSETS)
                    if ctx is None:
                        continue
                    center, frames = ctx
                    per_task["T12"].append(
                        build_t12_item(
                            recording_id=recording_id,
                            camera=camera,
                            frame_index=center,
                            frame_indices=frames,
                            segment_index=segment_index,
                            masked_low_index=int(masked.get("low_index", masked_pos)),
                            high_text_raw=high_text_raw,
                            object_raw=object_raw,
                            prev_text=prev_item["text"],
                            masked_text=masked["text"],
                            next_text=next_item["text"],
                            low_vocab=low_vocab,
                            rng=rng,
                        )
                    )

    return per_task, t6_candidates


def finalize_t6_items(
    candidates: list[dict],
    *,
    low_quantile: float,
    high_quantile: float,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"num_candidates": 0}
    values = np.asarray([item["speed_seq_mean"] for item in candidates], dtype=np.float64)
    low_th = float(np.quantile(values, low_quantile))
    high_th = float(np.quantile(values, high_quantile))
    if high_th < low_th:
        high_th = low_th

    items: list[dict] = []
    ambiguous = 0
    for cand in candidates:
        speed_mean = float(cand["speed_seq_mean"])
        if speed_mean <= low_th:
            label = "stationary"
        elif speed_mean >= high_th:
            label = "actively_moving"
        else:
            ambiguous += 1
            continue
        items.append(
            build_t6_item(
                recording_id=cand["recording_id"],
                camera=cand["camera"],
                frame_index=int(cand["frame_index"]),
                frame_indices=[int(x) for x in cand["frame_indices"]],
                speed_seq_mean=speed_mean,
                label=label,
                high_text_raw=cand["high_level_text_raw"],
                object_raw=cand["object_raw"],
                low_quantile_threshold=low_th,
                high_quantile_threshold=high_th,
                rng=rng,
            )
        )

    summary = {
        "num_candidates": int(len(candidates)),
        "num_items": int(len(items)),
        "num_ambiguous_discarded": int(ambiguous),
        "stationary_threshold": round(low_th, 6),
        "fast_threshold": round(high_th, 6),
        "speed_quantiles": {
            "p10": round(float(np.quantile(values, 0.10)), 6),
            "p25": round(float(np.quantile(values, 0.25)), 6),
            "p50": round(float(np.quantile(values, 0.50)), 6),
            "p75": round(float(np.quantile(values, 0.75)), 6),
            "p90": round(float(np.quantile(values, 0.90)), 6),
        },
    }
    return items, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    enabled_tasks = set(normalize_tasks_csv(args.tasks))
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_recording_index()
    recordings = filter_recordings(rows, split=args.split, dataset_root=dataset_root)
    if args.limit_recordings > 0:
        recordings = recordings[: args.limit_recordings]
    low_vocab = load_low_level_vocab()

    per_task: dict[str, list[dict]] = defaultdict(list)
    t6_candidates: list[dict] = []
    recording_ids: list[str] = []

    for recording_row in recordings:
        recording_id = recording_row["recording_id"]
        recording_ids.append(recording_id)
        h5_path = recording_h5_path(dataset_root, recording_id)
        rec_items, rec_t6_candidates = build_recording_items(
            recording_row,
            h5_path,
            args.camera,
            low_vocab,
            enabled_tasks,
            rng,
        )
        for task_type, items in rec_items.items():
            per_task[task_type].extend(items)
        t6_candidates.extend(rec_t6_candidates)

    t6_summary = None
    if "T6" in enabled_tasks:
        t6_items, t6_summary = finalize_t6_items(
            t6_candidates,
            low_quantile=args.t6_low_quantile,
            high_quantile=args.t6_high_quantile,
            rng=rng,
        )
        per_task["T6"] = t6_items

    combined: list[dict] = []
    counts: dict[str, int] = {}
    for task_type in TASK_TO_FILENAME:
        if task_type not in enabled_tasks:
            continue
        rows = per_task.get(task_type, [])
        counts[task_type] = len(rows)
        write_jsonl(output_dir / TASK_TO_FILENAME[task_type], rows)
        combined.extend(rows)

    write_jsonl(output_dir / "all_gt_items.jsonl", combined)
    summary = {
        "dataset": "REASSEMBLE",
        "split": args.split,
        "camera": args.camera,
        "num_recordings": len(recordings),
        "recording_ids": recording_ids,
        "tasks": sorted(enabled_tasks),
        "counts": counts,
        "all_gt_items_jsonl": str(output_dir / "all_gt_items.jsonl"),
        "t6_summary": t6_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
