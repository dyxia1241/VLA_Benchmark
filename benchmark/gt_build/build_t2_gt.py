from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events


def task_episode_paths(dataset_root: Path, task_id: str) -> list[Path]:
    d = dataset_root / task_id / "data" / "chunk-000"
    if not d.exists():
        return []
    return sorted(d.glob("episode_*.parquet"))


def pick_three_episodes(paths: list[Path]) -> list[Path]:
    if len(paths) <= 3:
        return paths
    idxs = sorted({0, len(paths) // 2, len(paths) - 1})
    return [paths[i] for i in idxs]


def load_episode_df(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        parquet_path,
        columns=[
            "timestamp",
            "frame_index",
            "observation.state.effector.effort",
        ],
    )


def _sample_without_replacement(indices: np.ndarray, k: int, rng: random.Random) -> list[int]:
    if len(indices) == 0:
        return []
    if k == 0:
        return []
    if k < 0 or len(indices) <= k:
        return [int(x) for x in indices.tolist()]
    return [int(x) for x in rng.sample(indices.tolist(), k)]


def _build_contact_masks(
    n_rows: int,
    side_events: list[dict],
    margin_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      yes_mask: reliable contact frames (inside event, away from boundaries)
      no_mask: reliable no-contact frames (outside contact + away from boundaries)
      boundary_mask: excluded boundary frames
    """
    yes_mask = np.zeros(n_rows, dtype=bool)
    boundary_mask = np.zeros(n_rows, dtype=bool)

    for e in side_events:
        if "contact_frame" not in e or "release_frame" not in e:
            continue
        c = max(0, min(int(e["contact_frame"]), n_rows - 1))
        r = max(0, min(int(e["release_frame"]), n_rows - 1))
        if r <= c:
            continue

        # Exclude near-boundary frames to reduce ambiguous supervision.
        c_lo = max(0, c - margin_frames)
        c_hi = min(n_rows, c + margin_frames + 1)
        r_lo = max(0, r - margin_frames)
        r_hi = min(n_rows, r + margin_frames + 1)
        boundary_mask[c_lo:c_hi] = True
        boundary_mask[r_lo:r_hi] = True

        # Reliable contact interior.
        y_lo = c + margin_frames
        y_hi = r - margin_frames
        if y_hi > y_lo:
            yes_mask[y_lo:y_hi] = True

    no_mask = (~yes_mask) & (~boundary_mask)
    return yes_mask, no_mask, boundary_mask


def _question_for_side(side: str) -> str:
    if side == "left":
        return "Is the left gripper currently in contact with anything?"
    if side == "right":
        return "Is the right gripper currently in contact with anything?"
    return "Is the robot gripper currently in contact with anything?"


def _build_t2_item(
    task_id: str,
    episode_id: int,
    frame_index: int,
    task_meta: dict,
    camera: str,
    query_gripper: str,
    is_contact: bool,
    rng: random.Random,
) -> dict:
    yes_txt = "Yes — contact established"
    no_txt = "No — no contact"

    if rng.random() < 0.5:
        choices = {"A": yes_txt, "B": no_txt}
        answer = "A" if is_contact else "B"
    else:
        choices = {"A": no_txt, "B": yes_txt}
        answer = "B" if is_contact else "A"

    return {
        "task_id": task_id,
        "episode_id": int(episode_id),
        "frame_index": int(frame_index),
        "camera": camera,
        "question": _question_for_side(query_gripper),
        "choices": choices,
        "answer": answer,
        "task_type": "T2",
        "arm_type": str(task_meta.get("arm_type", "unknown")),
        "hake_category": "TBD",
        "meta": {
            "query_gripper": query_gripper,
            "label": "contact" if is_contact else "no_contact",
            "gt_source": "effector.effort_contact_event",
        },
    }


def build_t2_items_for_task(
    task_id: str,
    episode_paths: list[Path],
    task_meta: dict,
    camera: str,
    rng: random.Random,
    min_hold_frames: int = 10,  # kept for backward compatibility with old caller
    min_persist_frames: int = 5,
    margin_frames: int = 3,
    max_per_class_per_episode: int = 5,
    balanced_sampling: bool = True,
) -> tuple[list[dict], list[dict]]:
    del min_hold_frames  # no longer used in contact-detection T2

    per_episode: list[dict] = []
    items: list[dict] = []

    primary_arm = str(task_meta.get("primary_arm", "both"))
    if primary_arm == "left":
        query_sides = ["left"]
    elif primary_arm == "right":
        query_sides = ["right"]
    else:
        query_sides = ["left", "right"]

    for p in episode_paths:
        episode_id = int(p.stem.split("_")[-1])
        df = load_episode_df(p)
        n_rows = len(df)
        if n_rows == 0:
            continue

        if "frame_index" in df.columns:
            frame_indices = df["frame_index"].to_numpy()
        else:
            frame_indices = np.arange(n_rows, dtype=int)

        events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=min_persist_frames)

        for side in query_sides:
            side_events = list(events.get(side, []))
            yes_mask, no_mask, _ = _build_contact_masks(
                n_rows=n_rows,
                side_events=side_events,
                margin_frames=margin_frames,
            )
            yes_rows = np.where(yes_mask)[0]
            no_rows = np.where(no_mask)[0]

            if balanced_sampling:
                n_pick = min(len(yes_rows), len(no_rows))
                if max_per_class_per_episode > 0:
                    n_pick = min(n_pick, max_per_class_per_episode)
                yes_pick = _sample_without_replacement(yes_rows, n_pick, rng)
                no_pick = _sample_without_replacement(no_rows, n_pick, rng)
            else:
                k = max_per_class_per_episode if max_per_class_per_episode > 0 else -1
                yes_pick = _sample_without_replacement(yes_rows, k, rng)
                no_pick = _sample_without_replacement(no_rows, k, rng)

            for ridx in yes_pick:
                items.append(
                    _build_t2_item(
                        task_id=task_id,
                        episode_id=episode_id,
                        frame_index=int(frame_indices[ridx]),
                        task_meta=task_meta,
                        camera=camera,
                        query_gripper=side,
                        is_contact=True,
                        rng=rng,
                    )
                )
            for ridx in no_pick:
                items.append(
                    _build_t2_item(
                        task_id=task_id,
                        episode_id=episode_id,
                        frame_index=int(frame_indices[ridx]),
                        task_meta=task_meta,
                        camera=camera,
                        query_gripper=side,
                        is_contact=False,
                        rng=rng,
                    )
                )

            per_episode.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_id),
                    "query_gripper": side,
                    "num_events": int(len(side_events)),
                    "num_yes_available": int(len(yes_rows)),
                    "num_no_available": int(len(no_rows)),
                    "num_yes_sampled": int(len(yes_pick)),
                    "num_no_sampled": int(len(no_pick)),
                }
            )

    return items, per_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T2 (contact detection) GT from effector.effort contact events.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default="/data/projects/GM-100/GM100_bimanual_fullscan_20260318/task_type_annotation.csv",
    )
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/benchmark/t2_gt_items.jsonl")
    parser.add_argument("--output-summary-json", default="/data/projects/GM-100/benchmark/t2_gt_summary.json")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--min-persist-frames", type=int, default=5)
    parser.add_argument("--margin-frames", type=int, default=3)
    parser.add_argument(
        "--max-per-class-per-episode",
        type=int,
        default=5,
        help="Max sampled frames per class per episode per queried gripper; 0 means all available.",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Disable per-episode 1:1 contact/no-contact sampling.",
    )
    parser.add_argument(
        "--only-t2-eligible",
        action="store_true",
        help="Use only rows with t2_eligible=True in annotation csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks after filtering")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    if args.only_t2_eligible:
        ann = ann[ann["t2_eligible"] == True]  # noqa: E712
    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}

    items: list[dict] = []
    epi_rows: list[dict] = []
    zero_item_tasks: list[str] = []

    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]

        task_items, epi_stats = build_t2_items_for_task(
            task_id=task_id,
            episode_paths=eps,
            task_meta=meta_map[task_id],
            camera=args.camera,
            rng=rng,
            min_persist_frames=args.min_persist_frames,
            margin_frames=args.margin_frames,
            max_per_class_per_episode=args.max_per_class_per_episode,
            balanced_sampling=(not args.unbalanced),
        )
        if not task_items:
            zero_item_tasks.append(task_id)
        items.extend(task_items)
        epi_rows.extend(epi_stats)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    items_df = pd.DataFrame(items) if items else pd.DataFrame()
    epi_df = pd.DataFrame(epi_rows) if epi_rows else pd.DataFrame()

    answer_counts = items_df["answer"].value_counts().to_dict() if len(items_df) else {}
    task_counts = items_df["task_id"].value_counts().to_dict() if len(items_df) else {}
    label_counts = (
        pd.Series([str(x.get("meta", {}).get("label", "unknown")) for x in items]).value_counts().to_dict()
        if len(items) > 0
        else {}
    )
    gripper_counts = (
        pd.Series([str(x.get("meta", {}).get("query_gripper", "unknown")) for x in items]).value_counts().to_dict()
        if len(items) > 0
        else {}
    )

    sampled_yes = int(epi_df["num_yes_sampled"].sum()) if len(epi_df) else 0
    sampled_no = int(epi_df["num_no_sampled"].sum()) if len(epi_df) else 0
    avail_yes = int(epi_df["num_yes_available"].sum()) if len(epi_df) else 0
    avail_no = int(epi_df["num_no_available"].sum()) if len(epi_df) else 0
    epi_with_both = int(((epi_df["num_yes_available"] > 0) & (epi_df["num_no_available"] > 0)).sum()) if len(epi_df) else 0

    summary = {
        "output_jsonl": str(out_jsonl),
        "num_items": int(len(items)),
        "num_tasks_considered": int(len(tasks)),
        "num_tasks_with_items": int(len(set(items_df["task_id"].tolist()))) if len(items_df) else 0,
        "num_tasks_zero_items": int(len(zero_item_tasks)),
        "zero_item_tasks": zero_item_tasks,
        "num_episode_side_pairs": int(len(epi_rows)),
        "num_episode_side_pairs_with_both_classes": int(epi_with_both),
        "answer_distribution": {k: int(v) for k, v in answer_counts.items()},
        "label_distribution": {k: int(v) for k, v in label_counts.items()},
        "query_gripper_distribution": {k: int(v) for k, v in gripper_counts.items()},
        "sampled_contact_frames": sampled_yes,
        "sampled_no_contact_frames": sampled_no,
        "available_contact_frames": avail_yes,
        "available_no_contact_frames": avail_no,
        "per_task_item_quantiles": (
            {str(k): float(v) for k, v in pd.Series(list(task_counts.values())).quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()}
            if len(task_counts) > 0
            else {}
        ),
        "config": {
            "min_persist_frames": int(args.min_persist_frames),
            "margin_frames": int(args.margin_frames),
            "max_per_class_per_episode": int(args.max_per_class_per_episode),
            "balanced_sampling": bool(not args.unbalanced),
            "only_t2_eligible": bool(args.only_t2_eligible),
            "seed": int(args.seed),
        },
    }

    out_summary = Path(args.output_summary_json)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
