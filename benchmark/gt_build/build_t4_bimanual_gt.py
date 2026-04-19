from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


T4_LABELS = {
    0: "both_active",
    1: "left_only",
    2: "right_only",
    3: "both_idle",
}

T4_CONTEXT_OFFSETS = (-6, -3, 0, 3)


@dataclass(frozen=True)
class CandidateCenter:
    row_index: int
    run_id: int
    score: float
    run_start: int
    run_end: int


def load_episode_df(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        parquet_path,
        columns=["timestamp", "frame_index", "observation.state.arm.velocity"],
    )


def task_episode_paths(dataset_root: Path, task_id: str) -> list[Path]:
    d = dataset_root / task_id / "data" / "chunk-000"
    if not d.exists():
        return []
    return sorted(d.glob("episode_*.parquet"))


def translational_speed_norms(arm_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_xyz = arm_vel[:, :3]
    right_xyz = arm_vel[:, 6:9]
    left_speed = np.linalg.norm(left_xyz, axis=1)
    right_speed = np.linalg.norm(right_xyz, axis=1)
    return left_speed, right_speed


def causal_ema(x: np.ndarray, span: int) -> np.ndarray:
    if len(x) == 0:
        return x
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(dtype=float)


def hysteresis_active(speed: np.ndarray, high_threshold: float, low_threshold: float) -> np.ndarray:
    if len(speed) == 0:
        return np.zeros(0, dtype=bool)

    active = np.zeros(len(speed), dtype=bool)
    state = bool(float(speed[0]) >= high_threshold)
    active[0] = state

    for idx in range(1, len(speed)):
        value = float(speed[idx])
        if value >= high_threshold:
            state = True
        elif value <= low_threshold:
            state = False
        active[idx] = state
    return active


def combined_label_ids(left_active: np.ndarray, right_active: np.ndarray) -> np.ndarray:
    labels = np.full(len(left_active), 3, dtype=np.int32)  # both_idle
    labels[left_active & right_active] = 0
    labels[left_active & ~right_active] = 1
    labels[~left_active & right_active] = 2
    return labels


def transition_exclusion_mask(labels: np.ndarray, margin: int) -> np.ndarray:
    keep = np.ones(len(labels), dtype=bool)
    if len(labels) == 0 or margin <= 0:
        return keep

    change_rows = np.where(labels[1:] != labels[:-1])[0] + 1
    for ridx in change_rows.tolist():
        lo = max(0, ridx - margin)
        hi = min(len(labels), ridx + margin + 1)
        keep[lo:hi] = False
    return keep


def stable_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []

    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        value = int(labels[idx])
        if value == current:
            continue
        runs.append((start, idx - 1, current))
        start = idx
        current = value
    runs.append((start, len(labels) - 1, current))
    return runs


def label_confidence(
    label_id: int,
    left_speed_smooth: np.ndarray,
    right_speed_smooth: np.ndarray,
    ridx: int,
    high_threshold: float,
    low_threshold: float,
) -> float:
    left_val = float(left_speed_smooth[ridx])
    right_val = float(right_speed_smooth[ridx])

    if label_id == 0:  # both_active
        return min(left_val - high_threshold, right_val - high_threshold)
    if label_id == 1:  # left_only
        return min(left_val - high_threshold, low_threshold - right_val)
    if label_id == 2:  # right_only
        return min(right_val - high_threshold, low_threshold - left_val)
    # both_idle
    return min(low_threshold - left_val, low_threshold - right_val)


def build_candidates_for_label(
    labels: np.ndarray,
    left_speed_smooth: np.ndarray,
    right_speed_smooth: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    valid_center_mask: np.ndarray,
    transition_keep_mask: np.ndarray,
    min_stable_run_frames: int,
) -> dict[int, list[CandidateCenter]]:
    out: dict[int, list[CandidateCenter]] = {label_id: [] for label_id in T4_LABELS}

    for run_id, (run_start, run_end, label_id) in enumerate(stable_runs(labels)):
        run_len = run_end - run_start + 1
        if run_len < min_stable_run_frames:
            continue

        rows = np.where(valid_center_mask & transition_keep_mask)[0]
        rows = rows[(rows >= run_start) & (rows <= run_end)]
        if len(rows) == 0:
            continue

        for ridx in rows.tolist():
            raw_confidence = label_confidence(
                label_id=label_id,
                left_speed_smooth=left_speed_smooth,
                right_speed_smooth=right_speed_smooth,
                ridx=int(ridx),
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
            if raw_confidence <= 0.0:
                continue
            edge_margin = float(min(ridx - run_start, run_end - ridx))
            score = raw_confidence + 0.01 * edge_margin
            out[int(label_id)].append(
                CandidateCenter(
                    row_index=int(ridx),
                    run_id=int(run_id),
                    score=float(score),
                    run_start=int(run_start),
                    run_end=int(run_end),
                )
            )
    return out


def select_centers(candidates: list[CandidateCenter], max_per_label: int) -> list[CandidateCenter]:
    if not candidates or max_per_label <= 0:
        return []

    by_run: dict[int, list[CandidateCenter]] = {}
    for cand in candidates:
        by_run.setdefault(cand.run_id, []).append(cand)
    for arr in by_run.values():
        arr.sort(key=lambda x: (-x.score, x.row_index))

    picked: list[CandidateCenter] = []
    used_rows: set[int] = set()

    # First pass: take the strongest center from each stable run.
    run_best = [arr[0] for arr in by_run.values() if arr]
    run_best.sort(key=lambda x: (-x.score, x.row_index))
    for cand in run_best:
        if len(picked) >= max_per_label:
            break
        picked.append(cand)
        used_rows.add(cand.row_index)

    if len(picked) >= max_per_label:
        return sorted(picked, key=lambda x: x.row_index)

    # Second pass: fill the remaining budget with the next strongest stable centers.
    remaining: list[CandidateCenter] = []
    for arr in by_run.values():
        remaining.extend(arr[1:])
    remaining.sort(key=lambda x: (-x.score, x.row_index))
    for cand in remaining:
        if len(picked) >= max_per_label:
            break
        if cand.row_index in used_rows:
            continue
        picked.append(cand)
        used_rows.add(cand.row_index)

    return sorted(picked, key=lambda x: x.row_index)


def build_t4_for_episode(
    task_id: str,
    episode_index: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    max_per_label: int,
    ema_span: int,
    active_high_threshold: float,
    active_low_threshold: float,
    transition_margin: int,
    min_stable_run_frames: int,
    rng: random.Random,
) -> list[dict]:
    del rng  # deterministic stable-region selection is preferred for this builder

    n_rows = len(df)
    if n_rows == 0:
        return []

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy(dtype=int)
    else:
        frame_indices = np.arange(n_rows, dtype=int)

    arm_vel = np.vstack(df["observation.state.arm.velocity"].to_numpy())
    left_speed_raw, right_speed_raw = translational_speed_norms(arm_vel)
    left_speed_smooth = causal_ema(left_speed_raw, span=ema_span)
    right_speed_smooth = causal_ema(right_speed_raw, span=ema_span)

    left_active = hysteresis_active(left_speed_smooth, high_threshold=active_high_threshold, low_threshold=active_low_threshold)
    right_active = hysteresis_active(right_speed_smooth, high_threshold=active_high_threshold, low_threshold=active_low_threshold)
    labels = combined_label_ids(left_active=left_active, right_active=right_active)

    context_pre = max(0, -min(T4_CONTEXT_OFFSETS))
    context_post = max(0, max(T4_CONTEXT_OFFSETS))
    valid_center_mask = np.zeros(n_rows, dtype=bool)
    valid_center_mask[context_pre : max(context_pre, n_rows - context_post)] = True
    transition_keep_mask = transition_exclusion_mask(labels=labels, margin=transition_margin)

    candidates_by_label = build_candidates_for_label(
        labels=labels,
        left_speed_smooth=left_speed_smooth,
        right_speed_smooth=right_speed_smooth,
        high_threshold=active_high_threshold,
        low_threshold=active_low_threshold,
        valid_center_mask=valid_center_mask,
        transition_keep_mask=transition_keep_mask,
        min_stable_run_frames=min_stable_run_frames,
    )

    items: list[dict] = []
    for label_id, label_name in T4_LABELS.items():
        chosen = select_centers(candidates_by_label[label_id], max_per_label=max_per_label)
        for cand in chosen:
            ridx = int(cand.row_index)
            context_rows = [ridx + off for off in T4_CONTEXT_OFFSETS]
            context_frames = [int(frame_indices[row]) for row in context_rows]
            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_index),
                    "frame_index": int(frame_indices[ridx]),
                    "frame_indices": context_frames,
                    "frame_offsets": list(T4_CONTEXT_OFFSETS),
                    "camera": camera,
                    "task_type": "T4",
                    "label_id": int(label_id),
                    "label_name": label_name,
                    "arm_type": str(task_meta.get("arm_type", "unknown")),
                    "left_active": bool(left_active[ridx]),
                    "right_active": bool(right_active[ridx]),
                    "left_speed_trans_raw": float(left_speed_raw[ridx]),
                    "right_speed_trans_raw": float(right_speed_raw[ridx]),
                    "left_speed_trans_ema": float(left_speed_smooth[ridx]),
                    "right_speed_trans_ema": float(right_speed_smooth[ridx]),
                    "active_high_threshold": float(active_high_threshold),
                    "active_low_threshold": float(active_low_threshold),
                    "ema_span": int(ema_span),
                    "transition_margin": int(transition_margin),
                    "stable_run_start_frame": int(frame_indices[cand.run_start]),
                    "stable_run_end_frame": int(frame_indices[cand.run_end]),
                    "stable_run_length": int(cand.run_end - cand.run_start + 1),
                    "stable_region_score": float(cand.score),
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build T4 bimanual GT items from translational-speed activity with causal smoothing and stable-region sampling."
    )
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/benchmark/t4_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--ema-span", type=int, default=9)
    parser.add_argument("--active-high-threshold", type=float, default=0.35)
    parser.add_argument("--active-low-threshold", type=float, default=0.15)
    parser.add_argument("--transition-margin", type=int, default=6)
    parser.add_argument("--min-stable-run-frames", type=int, default=12)
    parser.add_argument("--max-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all t4_eligible tasks")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    ann = ann[ann["t4_eligible"] == True]  # noqa: E712
    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    meta_map = {row["task_id"]: row for _, row in ann.iterrows()}

    items: list[dict] = []
    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_index = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_t4_for_episode(
                task_id=task_id,
                episode_index=episode_index,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                max_per_label=args.max_per_label,
                ema_span=args.ema_span,
                active_high_threshold=args.active_high_threshold,
                active_low_threshold=args.active_low_threshold,
                transition_margin=args.transition_margin,
                min_stable_run_frames=args.min_stable_run_frames,
                rng=rng,
            )
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    label_counts = pd.Series([x["label_name"] for x in items]).value_counts().to_dict() if items else {}
    print(
        json.dumps(
            {
                "output_jsonl": str(out_path),
                "num_items": len(items),
                "num_tasks": len(tasks),
                "ema_span": args.ema_span,
                "active_high_threshold": args.active_high_threshold,
                "active_low_threshold": args.active_low_threshold,
                "transition_margin": args.transition_margin,
                "min_stable_run_frames": args.min_stable_run_frames,
                "max_per_label": args.max_per_label,
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
