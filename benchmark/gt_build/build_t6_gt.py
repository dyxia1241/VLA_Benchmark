from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row


LABEL_TO_TEXT = {
    "actively_moving": "Actively moving (the arm is clearly in motion)",
    "stationary": "Stationary (the arm is still or barely moving)",
}
CHOICE_KEYS = ["A", "B"]


def task_episode_paths(dataset_root: Path, task_id: str) -> list[Path]:
    d = dataset_root / task_id / "data" / "chunk-000"
    if not d.exists():
        return []
    return sorted(d.glob("episode_*.parquet"))


def load_episode_df(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        parquet_path,
        columns=[
            "timestamp",
            "frame_index",
            "observation.state.arm.velocity",
            "observation.state.effector.effort",
        ],
    )


def arm_velocity_norm(arm_vel: np.ndarray, primary_arm: str) -> np.ndarray:
    v_left = arm_vel[:, :3]
    v_right = arm_vel[:, 6:9]
    n_left = np.linalg.norm(v_left, axis=1)
    n_right = np.linalg.norm(v_right, axis=1)

    if primary_arm == "left":
        return n_left
    if primary_arm == "right":
        return n_right

    # fallback for both/none: per-frame dominant moving arm
    return np.where(n_left >= n_right, n_left, n_right)


def _sample_per_label(
    label_to_rows: dict[str, list[int]],
    max_per_label: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label, rows in label_to_rows.items():
        if len(rows) <= max_per_label:
            out[label] = sorted(rows)
        else:
            out[label] = sorted(rng.sample(rows, max_per_label))
    return out


def _binary_speed_label(
    speed: float,
    stationary_threshold: float,
    fast_threshold: float,
) -> str | None:
    if speed >= fast_threshold:
        return "actively_moving"
    if speed <= stationary_threshold:
        return "stationary"
    return None


def build_t6_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    accel_th: float,
    velocity_min: float,
    frame_stride: int,
    half_span: int,
    max_per_label: int,
    stationary_threshold: float,
    fast_threshold: float,
    rng: random.Random,
    approach_buffer_frames: int = 30,
    no_contact_start_frame: int = 20,
) -> tuple[list[dict], dict]:
    # `accel_th` is kept only for backward compatibility with old callers.
    del accel_th

    if len(df) < 2:
        return [], {}

    contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    start_row = sampling_start_row(
        n_rows=len(df),
        contact_events=contact_events,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )

    arm_vel = np.vstack(df["observation.state.arm.velocity"].to_numpy())
    primary_arm = str(task_meta.get("primary_arm", "none"))
    vel = arm_velocity_norm(arm_vel, primary_arm=primary_arm)

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy().astype(int)
    else:
        frame_indices = np.arange(len(df), dtype=int)

    need = frame_stride * half_span
    valid_center = np.ones(len(df), dtype=bool)
    valid_center[:need] = False
    valid_center[-need:] = False
    valid_center &= vel >= velocity_min
    valid_center &= np.arange(len(df)) >= start_row

    valid_rows = np.where(valid_center)[0]
    if len(valid_rows) == 0:
        return [], {
            "num_rows": int(len(df)),
            "num_valid_centers": 0,
            "num_items": 0,
            "num_ambiguous_discarded": 0,
            "label_counts_before_sampling": {"actively_moving": 0, "stationary": 0},
            "label_counts_after_sampling": {"actively_moving": 0, "stationary": 0},
            "stationary_threshold": float(stationary_threshold),
            "fast_threshold": float(fast_threshold),
        }

    # Use sequence-mean speed and drop ambiguous mid-speed windows.
    seq_mean_by_row: dict[int, float] = {}
    for ridx in valid_rows.tolist():
        offs = [k * frame_stride for k in range(-half_span, half_span + 1)]
        rows = [ridx + off for off in offs]
        seq_mean_by_row[int(ridx)] = float(np.mean(vel[rows]))

    valid_scores = np.array([seq_mean_by_row[int(r)] for r in valid_rows], dtype=float)

    label_to_rows: dict[str, list[int]] = {
        "actively_moving": [],
        "stationary": [],
    }
    num_ambiguous_discarded = 0
    for ridx in valid_rows.tolist():
        label = _binary_speed_label(
            seq_mean_by_row[int(ridx)],
            stationary_threshold=stationary_threshold,
            fast_threshold=fast_threshold,
        )
        if label is None:
            num_ambiguous_discarded += 1
            continue
        label_to_rows[label].append(int(ridx))

    picked = _sample_per_label(label_to_rows, max_per_label=max_per_label, rng=rng)

    items: list[dict] = []
    for label in ["actively_moving", "stationary"]:
        for ridx in picked[label]:
            offs = [k * frame_stride for k in range(-half_span, half_span + 1)]
            rows = [ridx + off for off in offs]
            fseq = [int(frame_indices[r]) for r in rows]
            speed_seq = [float(vel[r]) for r in rows]
            speed_center = float(vel[ridx])
            speed_seq_mean = float(np.mean(speed_seq))

            options = ["actively_moving", "stationary"]
            rng.shuffle(options)
            choices = {k: LABEL_TO_TEXT[v] for k, v in zip(CHOICE_KEYS, options)}
            answer = CHOICE_KEYS[options.index(label)]

            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_id),
                    "frame_index": int(frame_indices[ridx]),
                    "frame_indices": fseq,
                    "camera": camera,
                    "question": "Across these time-ordered frames, is the robot arm actively moving or stationary?",
                    "choices": choices,
                    "answer": answer,
                    "task_type": "T6",
                    "arm_type": task_meta["arm_type"],
                    "primary_arm": primary_arm,
                    "speed_level_label": label,
                    # Keep legacy key for downstream compatibility.
                    "speed_change_label": label,
                    "speed_center": speed_center,
                    "speed_seq_mean": speed_seq_mean,
                    "speed_seq": speed_seq,
                    "velocity_min": float(velocity_min),
                    "stationary_threshold": float(stationary_threshold),
                    "fast_threshold": float(fast_threshold),
                    "frame_offsets": offs,
                    "frame_stride": int(frame_stride),
                }
            )

    calib = {
        "num_rows": int(len(df)),
        "num_valid_centers": int(len(valid_rows)),
        "num_items": int(len(items)),
        "num_ambiguous_discarded": int(num_ambiguous_discarded),
        "label_counts_before_sampling": {k: int(len(v)) for k, v in label_to_rows.items()},
        "label_counts_after_sampling": {k: int(len(v)) for k, v in picked.items()},
        "stationary_threshold": float(stationary_threshold),
        "fast_threshold": float(fast_threshold),
        "valid_speed_quantiles": {
            "p10": float(np.percentile(valid_scores, 10)),
            "p25": float(np.percentile(valid_scores, 25)),
            "p50": float(np.percentile(valid_scores, 50)),
            "p75": float(np.percentile(valid_scores, 75)),
            "p90": float(np.percentile(valid_scores, 90)),
        },
    }
    return items, calib


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T6 binary motion-state GT items from arm velocity.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default="/data/projects/GM-100/GM100_bimanual_fullscan_20260318/task_type_annotation.csv",
    )
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/benchmark/manual_checks_20260319/t6_gt_items.jsonl")
    parser.add_argument("--output-summary-json", default="/data/projects/GM-100/benchmark/manual_checks_20260319/t6_gt_summary.json")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--accel-th", type=float, default=0.05, help="Deprecated; kept for backward compatibility.")
    parser.add_argument("--velocity-min", type=float, default=0.0, help="Minimum center speed for candidate windows.")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--half-span", type=int, default=2, help="2 means 5 frames: t-6,t-3,t,t+3,t+6")
    parser.add_argument("--max-per-label", type=int, default=3)
    parser.add_argument(
        "--stationary-threshold",
        type=float,
        default=0.2,
        help="Sequence-mean speed <= this value is labeled stationary.",
    )
    parser.add_argument(
        "--fast-threshold",
        type=float,
        default=0.8,
        help="Sequence-mean speed >= this value is labeled actively_moving.",
    )
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    if args.stationary_threshold >= args.fast_threshold:
        raise ValueError("stationary_threshold must be smaller than fast_threshold.")

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}

    items: list[dict] = []
    calib_by_task: dict[str, dict] = {}
    first_task_with_data = None
    first_task_calib = None
    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]

        task_items = 0
        task_valid_centers = 0
        task_ambiguous_discarded = 0
        for p in eps:
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items, calib = build_t6_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                accel_th=args.accel_th,
                velocity_min=args.velocity_min,
                frame_stride=args.frame_stride,
                half_span=args.half_span,
                max_per_label=args.max_per_label,
                stationary_threshold=args.stationary_threshold,
                fast_threshold=args.fast_threshold,
                rng=rng,
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            items.extend(epi_items)
            task_items += len(epi_items)
            task_valid_centers += int(calib.get("num_valid_centers", 0))
            task_ambiguous_discarded += int(calib.get("num_ambiguous_discarded", 0))

            vq = calib.get("valid_speed_quantiles")
            if first_task_with_data is None and vq is not None:
                first_task_with_data = task_id
                first_task_calib = calib

        calib_by_task[task_id] = {
            "num_episodes": int(len(eps)),
            "num_items": int(task_items),
            "num_valid_centers": int(task_valid_centers),
            "num_ambiguous_discarded": int(task_ambiguous_discarded),
        }

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    label_counts = {k: 0 for k in ["actively_moving", "stationary"]}
    for x in items:
        label_counts[x["speed_level_label"]] += 1

    summary = {
        "output_jsonl": str(out_path),
        "output_summary_json": args.output_summary_json,
        "num_items": int(len(items)),
        "num_tasks": int(len(tasks)),
        "camera": args.camera,
        "task_variant": "T6_binary_motion_state",
        "deprecated_accel_th": float(args.accel_th),
        "velocity_min": float(args.velocity_min),
        "frame_stride": int(args.frame_stride),
        "half_span": int(args.half_span),
        "max_per_label": int(args.max_per_label),
        "stationary_threshold": float(args.stationary_threshold),
        "fast_threshold": float(args.fast_threshold),
        "label_counts": label_counts,
        "first_task_for_threshold_check": first_task_with_data,
        "first_task_calibration": first_task_calib,
        "calibration_by_task": calib_by_task,
    }
    Path(args.output_summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
