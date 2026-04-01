from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row, segment_trajectory


PROGRESS_BIN_TO_TEXT = {
    "0-20%": "0-20% (just starting)",
    "20-40%": "20-40% (approaching)",
    "40-60%": "40-60% (in progress)",
    "60-80%": "60-80% (nearly done)",
    "80-100%": "80-100% (finishing)",
}

STAGE_TO_PROGRESS = {
    0: "0-20%",
    1: "20-40%",
    2: "40-60%",
    3: "40-60%",
    4: "60-80%",
    5: "80-100%",
}

PROGRESS_CHOICE_ORDER = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
PROGRESS_CHOICE_KEYS = ["A", "B", "C", "D", "E"]


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


def build_progress_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    max_per_bin: int,
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

    bin_to_rows: dict[str, list[int]] = {k: [] for k in PROGRESS_CHOICE_ORDER}
    for ridx, stage in enumerate(labels.tolist()):
        if ridx < start_row:
            continue
        pbin = STAGE_TO_PROGRESS[int(stage)]
        bin_to_rows[pbin].append(ridx)

    items: list[dict] = []
    for pbin in PROGRESS_CHOICE_ORDER:
        rows = bin_to_rows[pbin]
        if len(rows) == 0:
            continue
        if len(rows) > max_per_bin:
            rows = rng.sample(rows, max_per_bin)
        for ridx in sorted(rows):
            choices = {k: PROGRESS_BIN_TO_TEXT[b] for k, b in zip(PROGRESS_CHOICE_KEYS, PROGRESS_CHOICE_ORDER)}
            answer = PROGRESS_CHOICE_KEYS[PROGRESS_CHOICE_ORDER.index(pbin)]
            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_id),
                    "frame_index": int(frame_indices[ridx]),
                    "camera": camera,
                    "task_type": "T_progress",
                    "question": "What percentage of the manipulation task has been completed?",
                    "choices": choices,
                    "answer": answer,
                    "progress_bin": pbin,
                    "stage_label": int(labels[ridx]),
                    "arm_type": task_meta["arm_type"],
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T_progress GT items from segmented trajectories.")
    parser.add_argument("--dataset-root", default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default="/home/dayu/projects/GM-100/GM100_bimanual_fullscan_20260318/task_type_annotation.csv",
    )
    parser.add_argument("--output-jsonl", default="/home/dayu/t_progress_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--max-per-bin", type=int, default=5)
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
    episode_count = 0
    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_count += 1
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_progress_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                max_per_bin=args.max_per_bin,
                rng=rng,
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    bin_counts = {b: 0 for b in PROGRESS_CHOICE_ORDER}
    for x in items:
        bin_counts[x["progress_bin"]] += 1

    print(
        json.dumps(
            {
                "output_jsonl": str(out_path),
                "num_items": len(items),
                "num_tasks": len(tasks),
                "num_episodes_seen": episode_count,
                "max_per_bin": args.max_per_bin,
                "bin_counts": bin_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
