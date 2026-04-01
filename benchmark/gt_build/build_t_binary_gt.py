from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row, segment_trajectory


QUESTION_TEXT = (
    "A single comparison image shows two labeled panels from the same robot manipulation episode. "
    "The panel placement and the labels X and Y are arbitrary identifiers and do not indicate temporal order. "
    "Which labeled panel happened earlier in the real manipulation sequence?"
)
CHOICES = {
    "X": "Image X happened earlier",
    "Y": "Image Y happened earlier",
}
DISPLAY_LABEL_VOCAB = ("X", "Y")


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


def _stage_to_rows(labels: np.ndarray, start_row: int) -> dict[int, list[int]]:
    rows = np.arange(len(labels))
    return {s: np.where((labels == s) & (rows >= start_row))[0].tolist() for s in range(6)}


def _sample_stage_pair(
    stage_to_rows: dict[int, list[int]],
    frame_indices: np.ndarray,
    candidate_pairs: list[tuple[int, int]],
    min_gap_frames: int,
    max_gap_frames: int,
    rng: random.Random,
    max_attempts: int = 64,
) -> tuple[int, int, int, int] | None:
    usable_pairs = [(a, b) for a, b in candidate_pairs if stage_to_rows.get(a) and stage_to_rows.get(b)]
    if not usable_pairs:
        return None

    for _ in range(max_attempts):
        sa, sb = rng.choice(usable_pairs)
        ra = rng.choice(stage_to_rows[sa])
        rb = rng.choice(stage_to_rows[sb])
        if ra == rb:
            continue
        frame_gap = abs(int(frame_indices[ra]) - int(frame_indices[rb]))
        if frame_gap < min_gap_frames:
            continue
        if max_gap_frames > 0 and frame_gap > max_gap_frames:
            continue
        return ra, rb, sa, sb
    return None


def _deterministic_display_labels(task_id: str, episode_id: int, frame_a: int, frame_b: int) -> list[str]:
    key = f"{task_id}|{episode_id}|{min(frame_a, frame_b)}|{max(frame_a, frame_b)}|t_binary_v2"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return [DISPLAY_LABEL_VOCAB[0], DISPLAY_LABEL_VOCAB[1]]
    return [DISPLAY_LABEL_VOCAB[1], DISPLAY_LABEL_VOCAB[0]]


def _build_item(
    task_id: str,
    episode_id: int,
    camera: str,
    frame_indices: np.ndarray,
    row_a: int,
    row_b: int,
    stage_a: int,
    stage_b: int,
    difficulty: str,
    task_meta: dict,
    rng: random.Random,
) -> dict:
    frame_a = int(frame_indices[row_a])
    frame_b = int(frame_indices[row_b])

    # Keep the original single random draw for panel order so frame-pair sampling remains stable.
    if rng.random() < 0.5:
        display_frames = [frame_a, frame_b]
        display_stages = [int(stage_a), int(stage_b)]
    else:
        display_frames = [frame_b, frame_a]
        display_stages = [int(stage_b), int(stage_a)]

    display_labels = _deterministic_display_labels(task_id, episode_id, frame_a, frame_b)
    earlier_panel_index = 0 if display_frames[0] < display_frames[1] else 1
    answer = display_labels[earlier_panel_index]

    return {
        "task_id": task_id,
        "episode_id": int(episode_id),
        "camera": camera,
        "task_type": "T_binary",
        "question": QUESTION_TEXT,
        "frame_indices": display_frames,
        "display_labels": display_labels,
        "choices": dict(CHOICES),
        "answer": answer,
        "stage_labels": display_stages,
        "frame_gap": int(abs(frame_a - frame_b)),
        "difficulty": difficulty,
        "arm_type": str(task_meta.get("arm_type", "unknown")),
        "meta": {
            "gt_source": "frame_index_pairwise_order",
            "sampling_strategy": difficulty,
            "stage_gap": int(abs(stage_a - stage_b)),
            "presentation": "composite_panel_v2",
            "earlier_panel_index": int(earlier_panel_index),
            "left_label": display_labels[0],
            "right_label": display_labels[1],
        },
    }


def build_binary_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    num_easy_pairs: int,
    num_hard_pairs: int,
    min_gap_frames: int,
    max_gap_frames: int,
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

    stage_to_rows = _stage_to_rows(labels, start_row=start_row)
    used_pairs: set[tuple[int, int]] = set()
    items: list[dict] = []

    hard_stage_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    easy_stage_pairs = [(a, b) for a in range(6) for b in range(a + 1, 6) if abs(a - b) >= 2]

    def add_items(target_n: int, stage_pairs: list[tuple[int, int]], difficulty: str) -> None:
        attempts = 0
        max_attempts = max(32, target_n * 64)
        while len([x for x in items if x["difficulty"] == difficulty]) < target_n and attempts < max_attempts:
            attempts += 1
            picked = _sample_stage_pair(
                stage_to_rows=stage_to_rows,
                frame_indices=frame_indices,
                candidate_pairs=stage_pairs,
                min_gap_frames=min_gap_frames,
                max_gap_frames=max_gap_frames,
                rng=rng,
            )
            if picked is None:
                break
            row_a, row_b, stage_a, stage_b = picked
            pair_key = tuple(sorted((int(frame_indices[row_a]), int(frame_indices[row_b]))))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            items.append(
                _build_item(
                    task_id=task_id,
                    episode_id=episode_id,
                    camera=camera,
                    frame_indices=frame_indices,
                    row_a=row_a,
                    row_b=row_b,
                    stage_a=stage_a,
                    stage_b=stage_b,
                    difficulty=difficulty,
                    task_meta=task_meta,
                    rng=rng,
                )
            )

    add_items(num_hard_pairs, hard_stage_pairs, "hard_adjacent_stage")
    add_items(num_easy_pairs, easy_stage_pairs, "easy_cross_stage")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T_binary GT items from pairwise frame ordering.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default="/data/projects/GM-100/GM100_bimanual_fullscan_20260318/task_type_annotation.csv",
    )
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/benchmark/t_binary_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--num-easy-pairs-per-episode", type=int, default=3)
    parser.add_argument("--num-hard-pairs-per-episode", type=int, default=2)
    parser.add_argument("--min-gap-frames", type=int, default=60)
    parser.add_argument("--max-gap-frames", type=int, default=240, help="0 means no max-gap constraint.")
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
    difficulty_counts = {
        "easy_cross_stage": 0,
        "hard_adjacent_stage": 0,
    }

    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_count += 1
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_binary_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                num_easy_pairs=args.num_easy_pairs_per_episode,
                num_hard_pairs=args.num_hard_pairs_per_episode,
                min_gap_frames=args.min_gap_frames,
                max_gap_frames=args.max_gap_frames,
                rng=rng,
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            for x in epi_items:
                difficulty_counts[x["difficulty"]] += 1
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
                "num_easy_pairs_per_episode": args.num_easy_pairs_per_episode,
                "num_hard_pairs_per_episode": args.num_hard_pairs_per_episode,
                "min_gap_frames": args.min_gap_frames,
                "max_gap_frames": args.max_gap_frames,
                "difficulty_counts": difficulty_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
