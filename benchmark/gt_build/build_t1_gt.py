from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import (
    detect_contact_events,
    sampling_start_row,
    segment_trajectory,
    t1_pre_approach_window,
)


PHASE_LABELS = {
    0: "pre-approach",
    1: "approach",
    2: "contact",
    3: "hold and carry",
    4: "transfer",
    5: "release",
}

CHOICE_KEYS = ["A", "B", "C", "D"]


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


def task_episode_paths(dataset_root: Path, task_id: str) -> list[Path]:
    d = dataset_root / task_id / "data" / "chunk-000"
    if not d.exists():
        return []
    return sorted(d.glob("episode_*.parquet"))


def build_choices(correct_phase: int, available_other_phases: list[int], rng: random.Random) -> tuple[dict, str]:
    others = [p for p in available_other_phases if p != correct_phase]
    if len(others) >= 3:
        distractors = rng.sample(others, 3)
    else:
        # Fallback: use remaining phase names to fill to 3 options.
        pool = [p for p in PHASE_LABELS if p != correct_phase]
        rng.shuffle(pool)
        distractors = (others + pool)[:3]

    options = [correct_phase] + distractors
    rng.shuffle(options)
    choices = {k: PHASE_LABELS[p] for k, p in zip(CHOICE_KEYS, options)}
    answer_key = CHOICE_KEYS[options.index(correct_phase)]
    return choices, answer_key


def sample_indices(indices: np.ndarray, max_per_phase: int, rng: random.Random) -> list[int]:
    if len(indices) == 0:
        return []
    if len(indices) >= max_per_phase:
        picked = rng.sample(indices.tolist(), max_per_phase)
    else:
        picked = indices.tolist() + [rng.choice(indices.tolist()) for _ in range(max_per_phase - len(indices))]
    return sorted(int(x) for x in picked)


def build_t1_for_episode(
    task_id: str,
    episode_index: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    max_per_phase: int,
    rng: random.Random,
    approach_buffer_frames: int = 30,
    pre_min_offset: int = 90,
    pre_max_offset: int = 30,
    no_contact_start_frame: int = 20,
) -> list[dict]:
    contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    labels = segment_trajectory(df, contact_events=contact_events, task_meta=task_meta, low_velocity_th=0.3)

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy()
    else:
        frame_indices = np.arange(len(df), dtype=int)

    items: list[dict] = []
    phases_present = [p for p in PHASE_LABELS if np.any(labels == p)]
    rows = np.arange(len(df))
    start_row = sampling_start_row(
        n_rows=len(df),
        contact_events=contact_events,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )
    pre_lo, pre_hi = t1_pre_approach_window(
        n_rows=len(df),
        contact_events=contact_events,
        pre_min_offset=pre_min_offset,
        pre_max_offset=pre_max_offset,
        no_contact_start_frame=no_contact_start_frame,
    )
    for phase in PHASE_LABELS:
        if phase == 0:
            idxs = np.where((labels == 0) & (rows >= pre_lo) & (rows < pre_hi))[0]
        else:
            idxs = np.where((labels == phase) & (rows >= start_row))[0]
        chosen_rows = sample_indices(idxs, max_per_phase=max_per_phase, rng=rng)
        for ridx in chosen_rows:
            choices, answer = build_choices(phase, phases_present, rng)
            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_index),
                    "frame_index": int(frame_indices[ridx]),
                    "camera": camera,
                    "question": "Which phase best describes the current robot manipulation state?",
                    "choices": choices,
                    "answer": answer,
                    "task_type": "T1",
                    "arm_type": task_meta["arm_type"],
                    "hake_category": "TBD",
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T1 phase-classification GT from segmentation labels.")
    parser.add_argument("--dataset-root", default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-jsonl", default="/home/dayu/t1_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--max-per-phase", type=int, default=5)
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--pre-min-offset", type=int, default=90)
    parser.add_argument("--pre-max-offset", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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

    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_index = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            task_meta = meta_map[task_id]
            epi_items = build_t1_for_episode(
                task_id=task_id,
                episode_index=episode_index,
                df=df,
                task_meta=task_meta,
                camera=args.camera,
                max_per_phase=args.max_per_phase,
                rng=rng,
                approach_buffer_frames=args.approach_buffer_frames,
                pre_min_offset=args.pre_min_offset,
                pre_max_offset=args.pre_max_offset,
                no_contact_start_frame=args.no_contact_start_frame,
            )
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
                "camera": args.camera,
                "max_per_phase": args.max_per_phase,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
