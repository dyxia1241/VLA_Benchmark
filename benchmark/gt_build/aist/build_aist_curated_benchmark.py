#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_aist_pilot_suite import build_t4, build_t6, build_t8, build_t9, iter_episodes  # noqa: E402
from build_aist_t3_pilot import DEFAULT_DIRECTION_MAPPING, build_t3_items_for_episode, load_direction_mapping  # noqa: E402

TASK_TYPES = ["T3", "T4", "T6", "T9", "T_temporal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curated AIST benchmark from selected20 episodes.")
    p.add_argument("--selected-root", default="/data/projects/GM-100/aist-bimanip/selected20")
    p.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/aist_benchmark_v0")
    p.add_argument("--camera", default="cam_high")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quota", default="all")
    p.add_argument("--max-per-episode-per-type", type=int, default=4)
    p.add_argument("--t8-target", type=int, default=2200)
    p.add_argument("--direction-mapping", default=str(DEFAULT_DIRECTION_MAPPING))
    return p.parse_args()


def parse_quota(text: str) -> dict[str, int] | None:
    if text.strip().lower() == "all":
        return None
    out: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = int(value)
    return out


def diverse_sample(rows: list[dict], quota: int, max_per_episode: int, rng: random.Random) -> list[dict]:
    pool = list(rows)
    rng.shuffle(pool)
    selected: list[dict] = []
    selected_ids: set[int] = set()
    per_episode: Counter[str] = Counter()
    for idx, item in enumerate(pool):
        ep = str(item.get("recording_id", ""))
        if max_per_episode > 0 and per_episode[ep] >= max_per_episode:
            continue
        selected.append(item)
        selected_ids.add(idx)
        per_episode[ep] += 1
        if len(selected) >= quota:
            return selected
    for idx, item in enumerate(pool):
        if idx in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= quota:
            break
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    by_type_dir = output_dir / "curated_by_type"
    by_type_dir.mkdir(parents=True, exist_ok=True)
    quota = parse_quota(args.quota)
    direction_mapping = load_direction_mapping(args.direction_mapping)

    rows: list[dict] = []
    episodes = iter_episodes(Path(args.selected_root))
    for ep in episodes:
        with h5py.File(ep, "r") as f:
            rows.extend(
                build_t3_items_for_episode(
                    ep,
                    f,
                    rng,
                    args.camera,
                    min_delta=0.035,
                    purity_ratio=1.35,
                    direction_mapping=direction_mapping,
                )
            )
            rows.extend(build_t4(ep, f, rng, args.camera))
            rows.extend(build_t6(ep, f, rng, args.camera))
            rows.extend(build_t9(ep, f, rng, args.camera))
            rows.extend(build_t8(ep, f, rng, args.camera))

    pool_path = output_dir / "aist_benchmark_v0_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    curated: list[dict] = []
    summary = {
        "selected_root": args.selected_root,
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "pool_counts": dict(Counter(r["task_type"] for r in rows)),
        "quota": quota if quota is not None else "all",
        "curated_counts": {},
        "direction_mapping": args.direction_mapping if direction_mapping else "identity",
    }
    for task_type in TASK_TYPES:
        subset = [r for r in rows if r["task_type"] == task_type]
        if task_type == "T_temporal":
            if quota is None:
                target = int(max(0, args.t8_target))
            else:
                target = int(max(0, quota.get(task_type, args.t8_target)))
            if target <= 0:
                picked = []
            else:
                picked = diverse_sample(subset, min(target, len(subset)), args.max_per_episode_per_type, rng)
        elif quota is None:
            picked = subset
        else:
            picked = diverse_sample(subset, quota.get(task_type, 0), args.max_per_episode_per_type, rng)
        curated.extend(picked)
        summary["curated_counts"][task_type] = len(picked)
        with (by_type_dir / f"{task_type}.jsonl").open("w", encoding="utf-8") as fh:
            for row in picked:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    curated_path = output_dir / "aist_benchmark_v0_curated.jsonl"
    with curated_path.open("w", encoding="utf-8") as fh:
        for row in curated:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["curated_total"] = len(curated)
    summary["t8_target"] = int(args.t8_target)
    summary["pool_jsonl"] = str(pool_path)
    summary["curated_jsonl"] = str(curated_path)
    (output_dir / "aist_benchmark_v0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
