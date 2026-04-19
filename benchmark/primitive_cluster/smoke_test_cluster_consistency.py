from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_cluster_pilot import (
    DEFAULT_ANNOTATION_CSV,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TASK_NAME_CSV,
    build_anchor_events,
    build_cluster_proposals,
    build_raw_events,
    infer_serial_repetition_risk,
    load_episode_df,
    load_task_meta,
    load_task_name_map,
    task_episode_paths,
)
from segmentation import _select_velocity_signal, detect_contact_events


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
DEFAULT_PILOT_SUMMARY_CSV = (
    REPO_ROOT / "benchmark" / "primitive_cluster" / "runs" / "one_episode_per_task_v0" / "task_cluster_summary.csv"
)


def bucket_name(cluster_count: int) -> str:
    if cluster_count == 0:
        return "zero"
    if cluster_count == 1:
        return "one"
    if cluster_count <= 3:
        return "two_to_three"
    return "four_plus"


def parse_bucket_spec(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, value = piece.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


def choose_tasks(
    pilot_rows: list[dict[str, str]],
    bucket_spec: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pilot_rows:
        by_bucket[bucket_name(int(row["cluster_count"]))].append(row)

    selected: list[dict[str, str]] = []
    for bucket in ["zero", "one", "two_to_three", "four_plus"]:
        pool = by_bucket.get(bucket, [])
        if not pool:
            continue
        n = min(bucket_spec.get(bucket, 0), len(pool))
        if n <= 0:
            continue
        picked = rng.sample(pool, n) if len(pool) > n else list(pool)
        picked.sort(key=lambda x: x["task_id"])
        selected.extend(picked)
    selected.sort(key=lambda x: x["task_id"])
    return selected


def summarize_counts(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {
            "mode_cluster_count": "",
            "mode_frequency": 0,
            "min_cluster_count": 0,
            "max_cluster_count": 0,
            "cluster_count_range": 0,
            "exact_consistent_all": False,
            "near_consistent_all": False,
        }
    count_counter = Counter(counts)
    mode_cluster_count, mode_frequency = sorted(count_counter.items(), key=lambda x: (-x[1], x[0]))[0]
    min_count = min(counts)
    max_count = max(counts)
    return {
        "mode_cluster_count": mode_cluster_count,
        "mode_frequency": mode_frequency,
        "min_cluster_count": min_count,
        "max_cluster_count": max_count,
        "cluster_count_range": max_count - min_count,
        "exact_consistent_all": len(set(counts)) == 1,
        "near_consistent_all": (max_count - min_count) <= 1,
    }


def summarize_nonzero_counts(counts: list[int]) -> dict[str, Any]:
    nonzero = [x for x in counts if x > 0]
    if not nonzero:
        return {
            "n_nonzero_cluster_episodes": 0,
            "exact_consistent_nonzero": False,
            "near_consistent_nonzero": False,
            "min_nonzero_cluster_count": "",
            "max_nonzero_cluster_count": "",
            "nonzero_cluster_count_range": "",
        }
    return {
        "n_nonzero_cluster_episodes": len(nonzero),
        "exact_consistent_nonzero": len(set(nonzero)) == 1,
        "near_consistent_nonzero": (max(nonzero) - min(nonzero)) <= 1,
        "min_nonzero_cluster_count": min(nonzero),
        "max_nonzero_cluster_count": max(nonzero),
        "nonzero_cluster_count_range": max(nonzero) - min(nonzero),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test cluster-count consistency across sampled episodes.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--annotation-csv", default=str(DEFAULT_ANNOTATION_CSV))
    parser.add_argument("--task-name-csv", default=str(DEFAULT_TASK_NAME_CSV))
    parser.add_argument("--pilot-summary-csv", default=str(DEFAULT_PILOT_SUMMARY_CSV))
    parser.add_argument(
        "--bucket-spec",
        default="zero:2,one:4,two_to_three:6,four_plus:6",
        help="How many tasks to sample per pilot bucket.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmark" / "primitive_cluster" / "runs" / "smoke_consistency_v0"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    annotation_csv = Path(args.annotation_csv)
    task_name_csv = Path(args.task_name_csv)
    pilot_summary_csv = Path(args.pilot_summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows = list(csv.DictReader(pilot_summary_csv.open()))
    bucket_spec = parse_bucket_spec(args.bucket_spec)
    task_rows = choose_tasks(pilot_rows, bucket_spec=bucket_spec, seed=args.seed)

    task_meta_map = load_task_meta(annotation_csv)
    task_name_map = load_task_name_map(task_name_csv)
    rng = random.Random(args.seed)

    episode_rows: list[dict[str, Any]] = []
    task_summary_rows: list[dict[str, Any]] = []

    for task_row in task_rows:
        task_id = task_row["task_id"]
        task_meta = task_meta_map[task_id]
        task_name_info = task_name_map.get(task_id, {})
        task_name_raw = task_name_info.get("task_name_raw", task_row.get("task_name_raw", ""))
        serial_risk = infer_serial_repetition_risk(task_name_raw)

        paths = task_episode_paths(dataset_root, task_id)
        if not paths:
            continue
        k = min(args.episodes_per_task, len(paths))
        sampled_paths = rng.sample(paths, k) if len(paths) > k else list(paths)
        sampled_paths.sort()

        sampled_cluster_counts: list[int] = []
        sampled_episode_ids: list[int] = []
        sampled_raw_counts: list[int] = []
        sampled_anchor_counts: list[int] = []

        for path in sampled_paths:
            episode_id = int(path.stem.split("_")[-1])
            df = load_episode_df(path)
            contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
            raw_events = build_raw_events(task_id, episode_id, contact_events)
            anchors = build_anchor_events(task_id, episode_id, raw_events)
            velocity_signal = _select_velocity_signal(df, task_meta)
            proposals, _ = build_cluster_proposals(
                task_id=task_id,
                episode_id=episode_id,
                anchors=anchors,
                velocity_signal=velocity_signal,
                task_serial_risk=serial_risk,
            )

            raw_count = len(raw_events)
            anchor_count = len(anchors)
            cluster_count = len(proposals)
            sampled_cluster_counts.append(cluster_count)
            sampled_episode_ids.append(episode_id)
            sampled_raw_counts.append(raw_count)
            sampled_anchor_counts.append(anchor_count)

            episode_rows.append(
                {
                    "task_id": task_id,
                    "task_name_raw": task_name_raw,
                    "arm_type": str(task_meta.get("arm_type", "")),
                    "task_serial_repetition_risk": serial_risk,
                    "pilot_bucket": bucket_name(int(task_row["cluster_count"])),
                    "pilot_cluster_count": int(task_row["cluster_count"]),
                    "episode_id": episode_id,
                    "n_frames": int(len(df)),
                    "raw_contact_events_total": raw_count,
                    "anchor_event_count": anchor_count,
                    "cluster_count": cluster_count,
                    "episode_path_rel": str(path.relative_to(REPO_ROOT)),
                }
            )

        all_summary = summarize_counts(sampled_cluster_counts)
        nonzero_summary = summarize_nonzero_counts(sampled_cluster_counts)
        task_summary_rows.append(
            {
                "task_id": task_id,
                "task_name_raw": task_name_raw,
                "arm_type": str(task_meta.get("arm_type", "")),
                "task_serial_repetition_risk": serial_risk,
                "pilot_bucket": bucket_name(int(task_row["cluster_count"])),
                "pilot_cluster_count": int(task_row["cluster_count"]),
                "n_sampled_episodes": len(sampled_episode_ids),
                "sampled_episode_ids": "|".join(str(x) for x in sampled_episode_ids),
                "sampled_cluster_counts": "|".join(str(x) for x in sampled_cluster_counts),
                "sampled_anchor_counts": "|".join(str(x) for x in sampled_anchor_counts),
                "sampled_raw_contact_counts": "|".join(str(x) for x in sampled_raw_counts),
                "n_zero_cluster_episodes": sum(1 for x in sampled_cluster_counts if x == 0),
                **all_summary,
                **nonzero_summary,
            }
        )

    task_summary_rows.sort(key=lambda x: (x["pilot_bucket"], x["task_id"]))
    episode_rows.sort(key=lambda x: (x["task_id"], x["episode_id"]))

    task_summary_csv = output_dir / "task_consistency_summary.csv"
    with task_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_summary_rows[0].keys()) if task_summary_rows else [])
        writer.writeheader()
        writer.writerows(task_summary_rows)

    episode_csv = output_dir / "episode_cluster_counts.csv"
    with episode_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else [])
        writer.writeheader()
        writer.writerows(episode_rows)

    overall = {
        "n_tasks_sampled": len(task_summary_rows),
        "n_episodes_sampled": len(episode_rows),
        "episodes_per_task_target": args.episodes_per_task,
        "bucket_spec": bucket_spec,
        "exact_consistent_all_rate": (
            sum(bool(r["exact_consistent_all"]) for r in task_summary_rows) / len(task_summary_rows)
            if task_summary_rows
            else 0.0
        ),
        "near_consistent_all_rate": (
            sum(bool(r["near_consistent_all"]) for r in task_summary_rows) / len(task_summary_rows)
            if task_summary_rows
            else 0.0
        ),
        "exact_consistent_nonzero_rate": (
            sum(bool(r["exact_consistent_nonzero"]) for r in task_summary_rows) / len(task_summary_rows)
            if task_summary_rows
            else 0.0
        ),
        "near_consistent_nonzero_rate": (
            sum(bool(r["near_consistent_nonzero"]) for r in task_summary_rows) / len(task_summary_rows)
            if task_summary_rows
            else 0.0
        ),
        "mean_cluster_count_range": (
            float(np.mean([int(r["cluster_count_range"]) for r in task_summary_rows])) if task_summary_rows else 0.0
        ),
        "files": {
            "task_consistency_summary_csv": str(task_summary_csv),
            "episode_cluster_counts_csv": str(episode_csv),
        },
    }

    by_bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_summary_rows:
        by_bucket_rows[row["pilot_bucket"]].append(row)
    overall["by_bucket"] = {
        bucket: {
            "n_tasks": len(rows),
            "exact_consistent_all_rate": sum(bool(r["exact_consistent_all"]) for r in rows) / len(rows),
            "near_consistent_all_rate": sum(bool(r["near_consistent_all"]) for r in rows) / len(rows),
            "exact_consistent_nonzero_rate": sum(bool(r["exact_consistent_nonzero"]) for r in rows) / len(rows),
            "near_consistent_nonzero_rate": sum(bool(r["near_consistent_nonzero"]) for r in rows) / len(rows),
            "mean_cluster_count_range": float(np.mean([int(r["cluster_count_range"]) for r in rows])),
        }
        for bucket, rows in sorted(by_bucket_rows.items())
        if rows
    }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
