from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row, segment_trajectory

TEMPORAL_DISPLAY_LABELS = ["X", "Y", "Z"]


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
            "observation.state.effector.effort",
            "observation.state.arm.velocity",
        ],
    )


def _spaced_triplet_ok(frames: list[int], min_gap: int) -> bool:
    a, b, c = sorted(frames)
    return (b - a > min_gap) and (c - b > min_gap) and (c - a > min_gap)


def build_temporal_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    num_questions: int,
    min_gap_frames: int,
    rng: random.Random,
    approach_buffer_frames: int = 30,
    no_contact_start_frame: int = 20,
) -> list[dict]:
    events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    labels = segment_trajectory(df, contact_events=events, task_meta=task_meta, low_velocity_th=0.3)
    start_row = sampling_start_row(
        n_rows=len(df),
        contact_events=events,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy()
    else:
        frame_indices = np.arange(len(df), dtype=int)

    stage_to_rows = {s: np.where((labels == s) & (np.arange(len(df)) >= start_row))[0].tolist() for s in range(6)}
    valid_stages = [s for s, rows in stage_to_rows.items() if len(rows) > 0]
    if len(valid_stages) < 3:
        return []

    items: list[dict] = []
    max_attempts = num_questions * 80
    attempts = 0
    while len(items) < num_questions and attempts < max_attempts:
        attempts += 1
        picked_stages = rng.sample(valid_stages, 3)
        picked_rows = [rng.choice(stage_to_rows[s]) for s in picked_stages]
        picked_frames = [int(frame_indices[r]) for r in picked_rows]

        if not _spaced_triplet_ok(picked_frames, min_gap=min_gap_frames):
            continue

        # Shuffle for display
        order = list(range(3))
        rng.shuffle(order)
        frames_shuf = [picked_frames[i] for i in order]
        stages_shuf = [picked_stages[i] for i in order]

        # Assign display labels randomly to shuffled frames
        display_labels = list(TEMPORAL_DISPLAY_LABELS)
        rng.shuffle(display_labels)

        # Answer = display label order sorted by time
        time_rank = sorted(range(3), key=lambda i: frames_shuf[i])
        answer = "".join(display_labels[i] for i in time_rank)

        items.append(
            {
                "task_id": task_id,
                "episode_id": int(episode_id),
                "camera": camera,
                "task_type": "T_temporal",
                "question": "Order these three frames from earliest to latest in the manipulation sequence.",
                "frame_indices": frames_shuf,
                "shuffled_labels": display_labels,
                "answer": answer,
                "stage_labels": stages_shuf,
                "arm_type": task_meta["arm_type"],
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T_temporal GT items from segmented trajectories.")
    parser.add_argument("--dataset-root", default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default="/home/dayu/projects/GM-100/GM100_bimanual_fullscan_20260318/task_type_annotation.csv",
    )
    parser.add_argument("--output-jsonl", default="/home/dayu/t_temporal_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--num-questions-per-episode", type=int, default=5)
    parser.add_argument("--min-gap-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}

    items: list[dict] = []
    skipped_episode_insufficient_stages = 0
    episode_count = 0

    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_count += 1
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_temporal_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                num_questions=args.num_questions_per_episode,
                min_gap_frames=args.min_gap_frames,
                rng=rng,
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            if not epi_items:
                skipped_episode_insufficient_stages += 1
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_jsonl": str(out_path),
                "num_items": len(items),
                "num_tasks": len(tasks),
                "num_episodes_seen": episode_count,
                "num_episodes_skipped_insufficient_stages": skipped_episode_insufficient_stages,
                "num_questions_per_episode": args.num_questions_per_episode,
                "min_gap_frames": args.min_gap_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
