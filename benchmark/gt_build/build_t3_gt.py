from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row


RAW_TO_VISUAL_DIR = {
    "+x": "toward the left side of the scene",
    "-x": "toward the right side of the scene",
    "+y": "toward the bottom of the scene",
    "-y": "toward the top of the scene",
    "+z": "upward (away from the table)",
    "-z": "downward (toward the table)",
}
T3_LETTERS = ("A", "B", "C", "D")


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
            "action.arm.position",
            "observation.state.effector.effort",
        ],
    )


def direction_from_delta_xyz(delta_xyz: np.ndarray) -> tuple[str, float]:
    axis = int(np.argmax(np.abs(delta_xyz)))
    val = float(delta_xyz[axis])
    sign = "+" if val >= 0 else "-"
    comp = ("x", "y", "z")[axis]
    return f"{sign}{comp}", abs(val)


def extract_primary_arm_delta_xyz(action_arm: np.ndarray, primary_arm: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      delta_xyz: (N-1, 3)
      used_arm:  (N-1,) values in {"left","right"}
    """
    d = np.diff(action_arm, axis=0)  # (N-1, 12)
    d_left = d[:, :3]
    d_right = d[:, 6:9]

    if primary_arm == "left":
        used = np.array(["left"] * len(d), dtype=object)
        return d_left, used
    if primary_arm == "right":
        used = np.array(["right"] * len(d), dtype=object)
        return d_right, used

    # primary_arm == both/none fallback: choose dominant arm per step.
    n_left = np.linalg.norm(d_left, axis=1)
    n_right = np.linalg.norm(d_right, axis=1)
    use_left = n_left >= n_right
    out = np.where(use_left[:, None], d_left, d_right)
    used = np.where(use_left, "left", "right")
    return out, used


def build_t3_choices(
    correct_direction_raw: str,
    rng: random.Random,
    preferred_answer_letter: str | None = None,
) -> tuple[dict[str, str], str]:
    """
    Build 4-way choices for T3.
    If preferred_answer_letter is provided, place the correct option on that letter
    (used for global answer-letter balancing).
    """
    distractor_pool = [x for x in RAW_TO_VISUAL_DIR.keys() if x != correct_direction_raw]
    wrong = rng.sample(distractor_pool, 3)

    if preferred_answer_letter in T3_LETTERS:
        letter_to_raw: dict[str, str] = {preferred_answer_letter: correct_direction_raw}
        other_letters = [x for x in T3_LETTERS if x != preferred_answer_letter]
        rng.shuffle(wrong)
        for k, raw_dir in zip(other_letters, wrong):
            letter_to_raw[k] = raw_dir
        answer = preferred_answer_letter
    else:
        options = [correct_direction_raw] + wrong
        rng.shuffle(options)
        letter_to_raw = {k: v for k, v in zip(T3_LETTERS, options)}
        answer = T3_LETTERS[options.index(correct_direction_raw)]

    choices = {k: RAW_TO_VISUAL_DIR[v] for k, v in letter_to_raw.items()}
    return choices, answer


def build_t3_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    camera: str,
    max_per_dir: int,
    static_norm_threshold: float,
    dominant_min: float,
    purity_ratio: float,
    rng: random.Random,
    answer_counts: dict[str, int] | None = None,
    balance_answer_letters: bool = True,
    approach_buffer_frames: int = 30,
    no_contact_start_frame: int = 20,
) -> list[dict]:
    action_arm = np.vstack(df["action.arm.position"].to_numpy())
    if len(action_arm) < 2:
        return []

    contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    start_row = sampling_start_row(
        n_rows=len(df),
        contact_events=contact_events,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )

    primary_arm = str(task_meta.get("primary_arm", "none"))
    delta_xyz, used_arm = extract_primary_arm_delta_xyz(action_arm, primary_arm=primary_arm)
    norms = np.linalg.norm(delta_xyz, axis=1)

    # delta[t] is between row t and t+1, use frame t for labeling.
    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy()[:-1]
    else:
        frame_indices = np.arange(len(delta_xyz), dtype=int)

    valid = np.zeros(len(delta_xyz), dtype=bool)
    raw_dir_ids: list[str | None] = []
    for i, dxyz in enumerate(delta_xyz):
        abs_xyz = np.abs(dxyz)
        dominant = float(np.max(abs_xyz))
        second = float(np.partition(abs_xyz, -2)[-2])
        pure_enough = dominant >= (purity_ratio * second if second > 0 else 0.0)
        valid[i] = (i >= start_row) and (norms[i] >= static_norm_threshold) and (dominant >= dominant_min) and pure_enough
        if valid[i]:
            dkey, _ = direction_from_delta_xyz(dxyz)
            raw_dir_ids.append(dkey)
        else:
            raw_dir_ids.append(None)
    if not np.any(valid):
        return []

    items: list[dict] = []
    for dkey in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        idxs = [i for i, k in enumerate(raw_dir_ids) if k == dkey]
        if len(idxs) == 0:
            continue
        if len(idxs) > max_per_dir:
            idxs = rng.sample(idxs, max_per_dir)
        for i in sorted(idxs):
            preferred_answer_letter = None
            if balance_answer_letters and answer_counts is not None:
                min_count = min(answer_counts.values())
                least_used = [k for k in T3_LETTERS if answer_counts[k] == min_count]
                preferred_answer_letter = rng.choice(least_used)
            choices, answer = build_t3_choices(
                correct_direction_raw=dkey,
                rng=rng,
                preferred_answer_letter=preferred_answer_letter,
            )
            if answer_counts is not None:
                answer_counts[answer] += 1

            abs_xyz = np.abs(delta_xyz[i])
            dominant = float(np.max(abs_xyz))
            second = float(np.partition(abs_xyz, -2)[-2])
            items.append(
                {
                    "task_id": task_id,
                    "episode_id": int(episode_id),
                    "frame_index": int(frame_indices[i]),
                    "camera": camera,
                    "question": "In which direction does the robot arm primarily move in the scene?",
                    "choices": choices,
                    "answer": answer,
                    "task_type": "T3",
                    "arm_type": task_meta["arm_type"],
                    "primary_arm": primary_arm,
                    "motion_direction": RAW_TO_VISUAL_DIR[dkey],
                    "motion_direction_raw": dkey,
                    "delta_xyz": [float(x) for x in delta_xyz[i].tolist()],
                    "delta_norm": float(norms[i]),
                    "dominant_component_abs": dominant,
                    "second_component_abs": second,
                    "direction_purity_ratio": float(dominant / (second + 1e-9)),
                    "used_arm_for_delta": str(used_arm[i]),
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T3 motion-direction GT.")
    parser.add_argument("--dataset-root", default="/home/dayu/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-jsonl", default="/home/dayu/t3_gt_items.jsonl")
    parser.add_argument("--output", dest="output_jsonl", help="Alias of --output-jsonl.")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--max-per-dir", type=int, default=5)
    parser.add_argument("--static-norm-threshold", type=float, default=0.005)
    parser.add_argument("--dominant-min", type=float, default=0.01)
    parser.add_argument("--purity-ratio", type=float, default=2.0)
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks-csv", default="", help="Comma-separated task list; empty means all tasks from annotation.")
    parser.add_argument("--tasks", dest="tasks_csv", help="Alias of --tasks-csv.")
    parser.add_argument(
        "--no-balance-answer-letters",
        action="store_true",
        help="Disable answer-letter balancing (A/B/C/D) for T3 options.",
    )
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    if args.tasks_csv.strip():
        tasks = [x.strip() for x in args.tasks_csv.split(",") if x.strip()]
    else:
        tasks = ann["task_id"].tolist()
        if args.limit_tasks > 0:
            tasks = tasks[: args.limit_tasks]
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}

    items: list[dict] = []
    answer_letter_counts: dict[str, int] = {k: 0 for k in T3_LETTERS}
    for task_id in tasks:
        if task_id not in meta_map:
            continue
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_t3_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                max_per_dir=args.max_per_dir,
                static_norm_threshold=args.static_norm_threshold,
                dominant_min=args.dominant_min,
                purity_ratio=args.purity_ratio,
                rng=rng,
                answer_counts=answer_letter_counts,
                balance_answer_letters=bool(not args.no_balance_answer_letters),
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # quick label stats
    label_counts = {}
    for d in RAW_TO_VISUAL_DIR.keys():
        label_counts[d] = int(sum(1 for x in items if x["motion_direction_raw"] == d))

    print(
        json.dumps(
            {
                "output_jsonl": str(out_path),
                "num_items": len(items),
                "num_tasks": len(tasks),
                "camera": args.camera,
                "max_per_dir": args.max_per_dir,
                "static_norm_threshold": args.static_norm_threshold,
                "dominant_min": args.dominant_min,
                "purity_ratio": args.purity_ratio,
                "balance_answer_letters": bool(not args.no_balance_answer_letters),
                "label_counts": label_counts,
                "answer_letter_counts": answer_letter_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
