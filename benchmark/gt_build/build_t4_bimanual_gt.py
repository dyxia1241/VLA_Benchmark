from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


T4_LABELS = {
    0: "both_active",
    1: "left_only",
    2: "right_only",
    3: "both_idle",
}


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


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def build_t4_for_episode(
    task_id: str,
    episode_index: int,
    df: pd.DataFrame,
    camera: str,
    max_per_label: int,
    vel_window: int,
    vel_threshold: float,
    rng: random.Random,
) -> list[dict]:
    vel = np.vstack(df["observation.state.arm.velocity"].to_numpy())
    vel_l = np.linalg.norm(vel[:, :6], axis=1)
    vel_r = np.linalg.norm(vel[:, 6:], axis=1)

    vel_l_ma = moving_avg(vel_l, window=vel_window)
    vel_r_ma = moving_avg(vel_r, window=vel_window)

    left_active = vel_l_ma > vel_threshold
    right_active = vel_r_ma > vel_threshold

    labels = np.full(len(df), 3, dtype=np.int32)  # both_idle
    labels[left_active & right_active] = 0
    labels[left_active & ~right_active] = 1
    labels[~left_active & right_active] = 2

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy()
    else:
        frame_indices = np.arange(len(df), dtype=int)

    items: list[dict] = []
    for label_id, label_name in T4_LABELS.items():
        idxs = np.where(labels == label_id)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) >= max_per_label:
            chosen = rng.sample(idxs.tolist(), max_per_label)
        else:
            chosen = idxs.tolist() + [rng.choice(idxs.tolist()) for _ in range(max_per_label - len(idxs))]
        chosen = sorted(int(i) for i in chosen)
        for ridx in chosen:
            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_index),
                    "frame_index": int(frame_indices[ridx]),
                    "camera": camera,
                    "task_type": "T4",
                    "label_id": int(label_id),
                    "label_name": label_name,
                    "left_active": bool(left_active[ridx]),
                    "right_active": bool(right_active[ridx]),
                    "vel_l_norm_ma": float(vel_l_ma[ridx]),
                    "vel_r_norm_ma": float(vel_r_ma[ridx]),
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T4 bimanual GT items from velocity activity labels.")
    parser.add_argument("--dataset-root", default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-jsonl", default="/home/dayu/t4_gt_items.jsonl")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--vel-window", type=int, default=30)
    parser.add_argument("--vel-threshold", type=float, default=0.3)
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
                camera=args.camera,
                max_per_label=args.max_per_label,
                vel_window=args.vel_window,
                vel_threshold=args.vel_threshold,
                rng=rng,
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
                "vel_window": args.vel_window,
                "vel_threshold": args.vel_threshold,
                "max_per_label": args.max_per_label,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
