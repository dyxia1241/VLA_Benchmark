from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_APPROACH_BUFFER_FRAMES = 30
DEFAULT_PRE_APPROACH_MIN_OFFSET = 90
DEFAULT_PRE_APPROACH_MAX_OFFSET = 30
DEFAULT_NO_CONTACT_START_FRAME = 20


def estimate_baseline(df: pd.DataFrame, tail_ratio: float = 0.10, head_ratio: float = 0.10) -> dict:
    """
    Estimate per-arm effort baseline and noise scale.

    Baseline:
      median effort over the last `tail_ratio` of the trajectory.
    Sigma:
      standard deviation of effort over the first `head_ratio` of the trajectory.
    """
    effort = np.vstack(df["observation.state.effector.effort"].to_numpy())
    n = len(effort)
    if n == 0:
        raise ValueError("Empty episode dataframe")

    tail_n = max(1, int(n * tail_ratio))
    head_n = max(1, int(n * head_ratio))

    tail = effort[-tail_n:]
    head = effort[:head_n]

    baseline_left = float(np.median(tail[:, 0]))
    baseline_right = float(np.median(tail[:, 1]))
    sigma_left = float(np.std(head[:, 0], ddof=0))
    sigma_right = float(np.std(head[:, 1], ddof=0))

    return {
        "baseline_left": baseline_left,
        "baseline_right": baseline_right,
        "sigma_left": sigma_left,
        "sigma_right": sigma_right,
        "tail_frames": tail_n,
        "head_frames": head_n,
    }


def _load_episode_df(dataset_root: Path, task_id: str, episode_index: int) -> pd.DataFrame:
    parquet_path = dataset_root / task_id / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    return pd.read_parquet(
        parquet_path,
        columns=[
            "observation.state.effector.effort",
            "observation.state.arm.velocity",
            "timestamp",
        ],
    )


def _detect_single_arm_events(
    effort: np.ndarray,
    baseline: float,
    sigma: float,
    min_persist_frames: int = 5,
    contact_sigma_k: float = 3.0,
    release_sigma_k: float = 2.0,
) -> list[dict]:
    sigma_eff = max(float(sigma), 1e-6)
    contact_th = baseline - contact_sigma_k * sigma_eff
    release_lo = baseline - release_sigma_k * sigma_eff
    release_hi = baseline + release_sigma_k * sigma_eff

    events: list[dict] = []
    n = len(effort)
    state = "idle"
    contact_frame: int | None = None
    i = 0
    while i < n:
        v = float(effort[i])
        if state == "idle":
            j = min(n, i + min_persist_frames)
            if j - i == min_persist_frames and np.all(effort[i:j] < contact_th):
                contact_frame = i
                state = "contacting"
                i = j
                continue
        else:
            if release_lo <= v <= release_hi:
                events.append({"contact_frame": int(contact_frame), "release_frame": int(i)})
                contact_frame = None
                state = "idle"
        i += 1
    return events


def detect_contact_events(df: pd.DataFrame, task_meta: dict, min_persist_frames: int = 5) -> dict:
    arm_type = str(task_meta.get("arm_type", "unknown"))
    has_gripper_motion = bool(task_meta.get("has_gripper_motion", False))
    if not has_gripper_motion:
        return {"left": [], "right": []}

    stats = estimate_baseline(df)
    effort = np.vstack(df["observation.state.effector.effort"].to_numpy())
    left_eff = effort[:, 0]
    right_eff = effort[:, 1]

    out = {"left": [], "right": []}
    if arm_type in {"single_left", "bimanual_sync", "bimanual_sequential"}:
        out["left"] = _detect_single_arm_events(
            left_eff,
            baseline=stats["baseline_left"],
            sigma=stats["sigma_left"],
            min_persist_frames=min_persist_frames,
        )
    if arm_type in {"single_right", "bimanual_sync", "bimanual_sequential"}:
        out["right"] = _detect_single_arm_events(
            right_eff,
            baseline=stats["baseline_right"],
            sigma=stats["sigma_right"],
            min_persist_frames=min_persist_frames,
        )
    return out


def _load_task_meta(annotation_csv: Path, task_id: str) -> dict:
    meta = pd.read_csv(annotation_csv)
    rec = meta[meta["task_id"] == task_id]
    if rec.empty:
        raise KeyError(f"Task not found in annotation csv: {task_id}")
    row = rec.iloc[0].to_dict()
    return row


def _select_velocity_signal(df: pd.DataFrame, task_meta: dict) -> np.ndarray:
    vel = np.vstack(df["observation.state.arm.velocity"].to_numpy())
    vel_l = np.linalg.norm(vel[:, :6], axis=1)
    vel_r = np.linalg.norm(vel[:, 6:], axis=1)
    primary_arm = str(task_meta.get("primary_arm", "none"))
    if primary_arm == "left":
        return vel_l
    if primary_arm == "right":
        return vel_r
    if primary_arm == "both":
        return np.maximum(vel_l, vel_r)
    return np.maximum(vel_l, vel_r)


def _flatten_events(contact_events: dict) -> list[dict]:
    events = list(contact_events.get("left", [])) + list(contact_events.get("right", []))
    events = [e for e in events if "contact_frame" in e and "release_frame" in e]
    events.sort(key=lambda x: (int(x["contact_frame"]), int(x["release_frame"])))
    return events


def first_contact_frame(contact_events: dict) -> int | None:
    events = _flatten_events(contact_events)
    if not events:
        return None
    return int(events[0]["contact_frame"])


def sampling_start_row(
    n_rows: int,
    contact_events: dict,
    approach_buffer_frames: int = DEFAULT_APPROACH_BUFFER_FRAMES,
    no_contact_start_frame: int = DEFAULT_NO_CONTACT_START_FRAME,
) -> int:
    fc = first_contact_frame(contact_events)
    if fc is None:
        return max(0, min(int(no_contact_start_frame), n_rows))
    return max(0, min(int(fc - approach_buffer_frames), n_rows))


def t1_pre_approach_window(
    n_rows: int,
    contact_events: dict,
    pre_min_offset: int = DEFAULT_PRE_APPROACH_MIN_OFFSET,
    pre_max_offset: int = DEFAULT_PRE_APPROACH_MAX_OFFSET,
    no_contact_start_frame: int = DEFAULT_NO_CONTACT_START_FRAME,
) -> tuple[int, int]:
    fc = first_contact_frame(contact_events)
    if fc is None:
        lo = max(0, min(int(no_contact_start_frame), n_rows))
        hi = n_rows
        return lo, hi

    lo = max(0, min(int(fc - pre_min_offset), n_rows))
    hi = max(0, min(int(fc - pre_max_offset), n_rows))
    if hi <= lo:
        return 0, 0
    return lo, hi


def segment_trajectory(
    df: pd.DataFrame,
    contact_events: dict,
    task_meta: dict,
    low_velocity_th: float = 0.3,
    contact_window_frames: int = 5,
) -> np.ndarray:
    """
    Produce per-frame stage labels:
      0 pre_approach, 1 approach, 2 contact, 3 hold_carry, 4 transfer, 5 release
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.int32)

    vel = _select_velocity_signal(df, task_meta)
    labels = np.zeros(n, dtype=np.int32)

    events = _flatten_events(contact_events)
    if not events:
        # No contact events: velocity-only segmentation path.
        active = vel >= low_velocity_th
        if np.any(active):
            first_active = int(np.argmax(active))
            last_active = int(n - 1 - np.argmax(active[::-1]))
            labels[first_active : last_active + 1] = 1

            seg = vel[first_active : last_active + 1]
            peak_idx = first_active + int(np.argmax(seg))
            below = np.where(vel[peak_idx : last_active + 1] < low_velocity_th)[0]
            if len(below) > 0:
                transfer_start = peak_idx + int(below[0])
                labels[transfer_start : last_active + 1] = 4
            if last_active + 1 < n:
                labels[last_active + 1 :] = 5
        return labels

    # Contact-aware path: process each contact/release cycle independently.
    norm_events: list[tuple[int, int]] = []
    for e in events:
        c = max(0, min(int(e["contact_frame"]), n - 1))
        r = max(0, min(int(e["release_frame"]), n - 1))
        if r <= c:
            continue
        norm_events.append((c, r))
    if not norm_events:
        return labels

    prev_release = 0
    for idx, (c, r) in enumerate(norm_events):
        pre_start = prev_release if idx > 0 else 0
        pre_end = c
        if pre_end > pre_start:
            if idx == 0:
                labels[pre_start:pre_end] = np.where(vel[pre_start:pre_end] < low_velocity_th, 0, 1)
            else:
                # Inter-grasp phase should be approach-like by default.
                labels[pre_start:pre_end] = 1

        c_end = min(r, c + contact_window_frames)
        labels[c:c_end] = 2

        hold_start = c_end
        if hold_start < r:
            labels[hold_start:r] = 3
            seg = vel[hold_start:r]
            peak_local = int(np.argmax(seg))
            peak_abs = hold_start + peak_local
            below = np.where(vel[peak_abs:r] < low_velocity_th)[0]
            if len(below) > 0:
                transfer_start = peak_abs + int(below[0])
                transfer_start = max(transfer_start, hold_start)
                labels[transfer_start:r] = 4

        prev_release = max(prev_release, r)

    labels[prev_release:] = 5
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline + contact-event detection sanity checks for GM-100.")
    parser.add_argument(
        "--dataset-root",
        default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot",
        help="Path to gm100-cobotmagic-lerobot root.",
    )
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--task-id", default="task_00046")
    parser.add_argument("--episode-index", type=int, default=129)
    parser.add_argument("--run-detect", action="store_true")
    parser.add_argument("--run-segment", action="store_true")
    args = parser.parse_args()

    df = _load_episode_df(Path(args.dataset_root), args.task_id, args.episode_index)
    stats = estimate_baseline(df)

    print(f"task={args.task_id} episode={args.episode_index:06d} n={len(df)}")
    print(
        "baseline_left={:.6f} baseline_right={:.6f} sigma_left={:.6f} sigma_right={:.6f} "
        "tail_frames={} head_frames={}".format(
            stats["baseline_left"],
            stats["baseline_right"],
            stats["sigma_left"],
            stats["sigma_right"],
            stats["tail_frames"],
            stats["head_frames"],
        )
    )

    if args.run_detect:
        task_meta = _load_task_meta(Path(args.annotation_csv), args.task_id)
        events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
        ts = df["timestamp"].to_numpy()

        def with_time(items: list[dict]) -> list[dict]:
            out = []
            for e in items:
                c = int(e["contact_frame"])
                r = int(e["release_frame"])
                out.append(
                    {
                        "contact_frame": c,
                        "contact_time_sec": float(ts[c]),
                        "release_frame": r,
                        "release_time_sec": float(ts[r]),
                        "duration_frames": int(r - c),
                    }
                )
            return out

        payload = {
            "task_id": args.task_id,
            "episode_index": args.episode_index,
            "arm_type": task_meta.get("arm_type"),
            "has_gripper_motion": bool(task_meta.get("has_gripper_motion", False)),
            "events": {
                "left": with_time(events["left"]),
                "right": with_time(events["right"]),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.run_segment:
        task_meta = _load_task_meta(Path(args.annotation_csv), args.task_id)
        events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
        labels = segment_trajectory(df, contact_events=events, task_meta=task_meta, low_velocity_th=0.3)
        uniq, cnt = np.unique(labels, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}

        # coarse non-decreasing trend check with tiny tolerance to keep debug simple
        dif = np.diff(labels)
        big_fallbacks = int(np.sum(dif < -1))
        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "episode_index": args.episode_index,
                    "arm_type": task_meta.get("arm_type"),
                    "has_gripper_motion": bool(task_meta.get("has_gripper_motion", False)),
                    "label_value_counts": counts,
                    "num_big_fallback_steps": big_fallbacks,
                    "first_120_labels": labels[:120].tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
