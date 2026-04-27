from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from local_step_intervals import build_local_step_intervals_for_episode, load_task_name_map


FRAME_OFFSETS = (-6, -3, 0, 3, 6)
PROGRESS_BIN_SPECS = [
    ("early", 0.15, 0.35, "Early stage (this manipulation step has just started)"),
    ("middle", 0.40, 0.60, "Middle stage (this manipulation step is about halfway complete)"),
    ("late", 0.65, 0.85, "Late stage (this manipulation step is nearly finished)"),
]
CHOICE_KEYS = ["A", "B", "C"]
CHOICES = {k: text for k, (_, _, _, text) in zip(CHOICE_KEYS, PROGRESS_BIN_SPECS)}
BIN_TO_ANSWER = {name: key for key, (name, _, _, _) in zip(CHOICE_KEYS, PROGRESS_BIN_SPECS)}


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


def parse_offsets_csv(text: str) -> tuple[int, ...]:
    offsets: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        offsets.append(int(token))
    if not offsets:
        raise ValueError("offset list is empty")
    if 0 not in offsets:
        raise ValueError("offset list must include 0")
    return tuple(offsets)


def context_margin(offsets: tuple[int, ...]) -> int:
    return max(abs(min(offsets)), abs(max(offsets)))


def build_progress_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict,
    task_name_raw: str,
    camera: str,
    frame_offsets: tuple[int, ...],
    min_interval_span: int,
    max_per_bin_per_interval: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    intervals = build_local_step_intervals_for_episode(
        task_id=task_id,
        episode_id=episode_id,
        df=df,
        task_meta=task_meta,
        task_name_raw=task_name_raw,
    )

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy().astype(int)
    else:
        frame_indices = np.arange(len(df), dtype=int)

    margin = context_margin(frame_offsets)
    items: list[dict] = []
    stats = {
        "num_intervals_total": int(len(intervals)),
        "num_intervals_short_filtered": 0,
        "num_intervals_without_valid_context": 0,
        "num_intervals_without_progress_bins": 0,
        "num_items": 0,
        "bin_counts": {name: 0 for name, _, _, _ in PROGRESS_BIN_SPECS},
    }

    for interval in intervals:
        start_row = int(interval.start_row)
        end_row = int(interval.end_row)
        span = int(end_row - start_row)
        if span < min_interval_span:
            stats["num_intervals_short_filtered"] += 1
            continue

        last_active_row = end_row - 1
        if last_active_row <= start_row:
            stats["num_intervals_short_filtered"] += 1
            continue

        lo = start_row + margin
        hi = end_row - margin
        if hi <= lo:
            stats["num_intervals_without_valid_context"] += 1
            continue

        denom = max(1, last_active_row - start_row)
        bin_to_rows: dict[str, list[int]] = {name: [] for name, _, _, _ in PROGRESS_BIN_SPECS}

        for center_row in range(lo, hi):
            window_rows = [center_row + off for off in frame_offsets]
            if min(window_rows) < start_row or max(window_rows) > last_active_row:
                continue
            progress_value = float(center_row - start_row) / float(denom)
            for bin_name, lo_u, hi_u, _ in PROGRESS_BIN_SPECS:
                if lo_u <= progress_value <= hi_u:
                    bin_to_rows[bin_name].append(center_row)
                    break

        if not any(bin_to_rows.values()):
            stats["num_intervals_without_progress_bins"] += 1
            continue

        for bin_name, _, _, _ in PROGRESS_BIN_SPECS:
            candidate_rows = bin_to_rows[bin_name]
            if not candidate_rows:
                continue
            if max_per_bin_per_interval <= 0 or len(candidate_rows) <= max_per_bin_per_interval:
                picked_rows = list(candidate_rows)
            else:
                picked_rows = rng.sample(candidate_rows, max_per_bin_per_interval)
            for center_row in sorted(picked_rows):
                progress_value = float(center_row - start_row) / float(denom)
                window_rows = [center_row + off for off in frame_offsets]
                window_frames = [int(frame_indices[r]) for r in window_rows]
                items.append(
                    {
                        "task_id": task_id,
                        "episode_id": int(episode_id),
                        "frame_index": int(frame_indices[center_row]),
                        "frame_indices": window_frames,
                        "camera": camera,
                        "task_type": "T_progress",
                        "question": "Across these time-ordered frames, how far along is the current manipulation step?",
                        "choices": dict(CHOICES),
                        "answer": BIN_TO_ANSWER[bin_name],
                        "progress_bin": bin_name,
                        "progress_value": round(progress_value, 4),
                        "interval_id": interval.interval_id,
                        "interval_order": int(interval.interval_order),
                        "interval_start_frame": int(frame_indices[start_row]),
                        "interval_end_frame": int(frame_indices[last_active_row]),
                        "interval_span_frames": int(frame_indices[last_active_row] - frame_indices[start_row]),
                        "interval_span_rows": int(span),
                        "active_arm_pattern": interval.active_arm_pattern,
                        "interval_source": "signal_native_local_step_interval_v1",
                        "merge_confidence": interval.merge_confidence,
                        "serial_repetition_risk": interval.serial_repetition_risk,
                        "frame_offsets": list(frame_offsets),
                        "arm_type": str(task_meta.get("arm_type", "unknown")),
                        "primary_arm": str(task_meta.get("primary_arm", "none")),
                        "task_name_raw": task_name_raw,
                    }
                )
                stats["bin_counts"][bin_name] += 1

    stats["num_items"] = int(len(items))
    return items, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T_progress v2 GT from signal-native local-step intervals.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument(
        "--output-jsonl",
        default=(
            "/data/projects/GM-100/benchmark/previous_results/manual_checks_20260320/"
            "root_release_source_20260414_tprogress_v2/t_progress_gt_items.jsonl"
        ),
    )
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--frame-offsets", default="-6,-3,0,3,6")
    parser.add_argument("--min-interval-span", type=int, default=24)
    parser.add_argument("--max-per-bin-per-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    frame_offsets = parse_offsets_csv(args.frame_offsets)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    meta_map = {str(r["task_id"]): r.to_dict() for _, r in ann.iterrows()}
    task_name_map = load_task_name_map(dataset_root, tasks)

    items: list[dict] = []
    episode_count = 0
    episodes_with_intervals = 0
    total_interval_count = 0
    total_short_filtered = 0
    total_no_context = 0
    total_no_bins = 0
    bin_counts = {name: 0 for name, _, _, _ in PROGRESS_BIN_SPECS}

    for task_id in tasks:
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_count += 1
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            task_meta = meta_map[task_id]
            epi_items, stats = build_progress_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=task_meta,
                task_name_raw=task_name_map.get(task_id, ""),
                camera=args.camera,
                frame_offsets=frame_offsets,
                min_interval_span=args.min_interval_span,
                max_per_bin_per_interval=args.max_per_bin_per_interval,
                rng=rng,
            )
            if stats["num_intervals_total"] > 0:
                episodes_with_intervals += 1
            total_interval_count += int(stats["num_intervals_total"])
            total_short_filtered += int(stats["num_intervals_short_filtered"])
            total_no_context += int(stats["num_intervals_without_valid_context"])
            total_no_bins += int(stats["num_intervals_without_progress_bins"])
            for bin_name, count in stats["bin_counts"].items():
                bin_counts[bin_name] += int(count)
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(out_path),
        "num_items": len(items),
        "num_tasks": len(tasks),
        "num_episodes_seen": episode_count,
        "num_episodes_with_intervals": episodes_with_intervals,
        "num_intervals_total": total_interval_count,
        "num_intervals_short_filtered": total_short_filtered,
        "num_intervals_without_valid_context": total_no_context,
        "num_intervals_without_progress_bins": total_no_bins,
        "frame_offsets": list(frame_offsets),
        "min_interval_span": int(args.min_interval_span),
        "max_per_bin_per_interval": int(args.max_per_bin_per_interval),
        "bin_counts": bin_counts,
    }

    if args.output_summary_json:
        summary_path = Path(args.output_summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
