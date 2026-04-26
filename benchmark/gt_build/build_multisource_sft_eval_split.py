#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / "splits_v1"
DEFAULT_RESULTS_DIR = BENCHMARK_ROOT / "eval_results_v1" / "splits_v1"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    slug: str
    input_jsonl: Path
    group_mode: str
    frame_dir_default: Path
    frame_dir_t3_multi: Path
    frame_dir_t6_multi: Path
    baselines: dict[str, float]
    task_weights: dict[str, float]


DATASET_CONFIGS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        name="GM100",
        slug="gm100",
        input_jsonl=BENCHMARK_ROOT / "benchmark_v1_curated.jsonl",
        group_mode="gm100_task_episode",
        frame_dir_default=BENCHMARK_ROOT / "benchmark_v1_frames_tbinary_20260330",
        frame_dir_t3_multi=BENCHMARK_ROOT / "benchmark_v1_frames_tbinary_20260330",
        frame_dir_t6_multi=BENCHMARK_ROOT / "benchmark_v1_frames_tbinary_20260330",
        baselines={
            "T1": 0.25,
            "T2": 0.50,
            "T3": 0.25,
            "T4": 0.25,
            "T6": 0.50,
            "T_temporal": 1.0 / 6.0,
            "T_binary": 0.50,
            "T_progress": 1.0 / 3.0,
        },
        task_weights={},
    ),
    DatasetConfig(
        name="RH20T",
        slug="rh20t",
        input_jsonl=BENCHMARK_ROOT / "rh20t_benchmark_v0_curated.jsonl",
        group_mode="recording_id",
        frame_dir_default=BENCHMARK_ROOT / "rh20t_benchmark_v0_frames",
        frame_dir_t3_multi=BENCHMARK_ROOT / "rh20t_benchmark_v0_frames",
        frame_dir_t6_multi=BENCHMARK_ROOT / "rh20t_benchmark_v0_frames",
        baselines={
            "T1": 0.25,
            "T2": 0.50,
            "T3": 0.25,
            "T6": 0.50,
            "T7": 0.50,
            "T_temporal": 1.0 / 6.0,
            "T_binary": 0.50,
            "T_progress": 1.0 / 3.0,
        },
        task_weights={},
    ),
    DatasetConfig(
        name="REASSEMBLE",
        slug="reassemble",
        input_jsonl=BENCHMARK_ROOT / "reassemble_benchmark_v0_curated.jsonl",
        group_mode="recording_id",
        frame_dir_default=BENCHMARK_ROOT / "reassemble_benchmark_v0_frames",
        frame_dir_t3_multi=BENCHMARK_ROOT / "reassemble_benchmark_v0_frames",
        frame_dir_t6_multi=BENCHMARK_ROOT / "reassemble_benchmark_v0_frames",
        baselines={
            "T1": 0.25,
            "T2": 0.50,
            "T6": 0.50,
            "T7": 0.50,
            "T_temporal": 1.0 / 6.0,
            "T_binary": 0.50,
            "T_progress": 1.0 / 3.0,
            "T10": 0.25,
            "T11": 0.25,
            "T12": 0.25,
        },
        task_weights={
            "T10": 2.5,
            "T11": 3.0,
            "T12": 5.0,
        },
    ),
    DatasetConfig(
        name="AIST",
        slug="aist",
        input_jsonl=BENCHMARK_ROOT / "aist_benchmark_v0" / "aist_benchmark_v0_curated.jsonl",
        group_mode="recording_id",
        frame_dir_default=BENCHMARK_ROOT / "aist_benchmark_v0_frames",
        frame_dir_t3_multi=BENCHMARK_ROOT / "aist_benchmark_v0_frames",
        frame_dir_t6_multi=BENCHMARK_ROOT / "aist_benchmark_v0_frames",
        baselines={
            "T3": 0.25,
            "T4": 0.25,
            "T6": 0.50,
            "T9": 0.50,
            "T_temporal": 1.0 / 6.0,
        },
        task_weights={},
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build episode-level 85/15 SFT/eval split for the multisource benchmark.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--eval-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--swap-passes", type=int, default=4)
    p.add_argument("--split-version", default="splits_v1")
    return p.parse_args()


def group_id_for_item(cfg: DatasetConfig, item: dict[str, Any]) -> str:
    if cfg.group_mode == "recording_id":
        recording_id = str(item.get("recording_id", "")).strip()
        if not recording_id:
            raise KeyError(f"{cfg.name} item missing recording_id")
        return recording_id
    if cfg.group_mode == "gm100_task_episode":
        task_id = str(item.get("task_id", "")).strip()
        episode_id = item.get("episode_id", None)
        if not task_id or episode_id is None:
            raise KeyError("GM100 item missing task_id or episode_id")
        return f"{task_id}__episode_{int(episode_id):06d}"
    raise ValueError(f"Unsupported group mode: {cfg.group_mode}")


def load_items(cfg: DatasetConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cfg.input_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                item.setdefault("dataset", cfg.name)
                rows.append(item)
    return rows


def counter_add(a: Counter[str], b: Counter[str]) -> Counter[str]:
    out = Counter(a)
    out.update(b)
    return out


def counter_sub(a: Counter[str], b: Counter[str]) -> Counter[str]:
    out = Counter(a)
    for key, value in b.items():
        out[key] -= value
        if out[key] <= 0:
            out.pop(key, None)
    return out


def objective(
    counts: Counter[str],
    *,
    target_counts: dict[str, float],
    task_weights: dict[str, float],
    total_items: int,
    target_total_items: float,
) -> float:
    score = 0.0
    for task_type, target in target_counts.items():
        weight = float(task_weights.get(task_type, 1.0))
        actual = float(counts.get(task_type, 0))
        score += weight * abs(actual - target) / max(target, 1.0)
        if target >= 1.0 and actual <= 0.0:
            score += 2.5 * weight
    score += 0.35 * abs(float(total_items) - target_total_items) / max(target_total_items, 1.0)
    return score


def greedy_select(
    group_ids: list[str],
    group_task_counts: dict[str, Counter[str]],
    eval_group_count: int,
    *,
    target_counts: dict[str, float],
    task_weights: dict[str, float],
    target_total_items: float,
    rng: random.Random,
) -> tuple[set[str], Counter[str], int]:
    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()
    selected_total = 0
    candidates = list(group_ids)
    rng.shuffle(candidates)
    for _ in range(eval_group_count):
        best_group = None
        best_score = None
        for group_id in candidates:
            if group_id in selected:
                continue
            new_counts = counter_add(selected_counts, group_task_counts[group_id])
            new_total = selected_total + int(sum(group_task_counts[group_id].values()))
            score = objective(
                new_counts,
                target_counts=target_counts,
                task_weights=task_weights,
                total_items=new_total,
                target_total_items=target_total_items,
            )
            if best_score is None or score < best_score - 1e-12:
                best_group = group_id
                best_score = score
        if best_group is None:
            break
        selected.add(best_group)
        selected_counts.update(group_task_counts[best_group])
        selected_total += int(sum(group_task_counts[best_group].values()))
    return selected, selected_counts, selected_total


def ranked_candidate_pool(
    group_ids: list[str],
    group_task_counts: dict[str, Counter[str]],
    current_counts: Counter[str],
    *,
    target_counts: dict[str, float],
    task_weights: dict[str, float],
    limit: int,
    reverse: bool,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for group_id in group_ids:
        counts = group_task_counts[group_id]
        delta = 0.0
        for task_type, value in counts.items():
            target = float(target_counts.get(task_type, 0.0))
            current = float(current_counts.get(task_type, 0.0))
            deficit = target - current
            weight = float(task_weights.get(task_type, 1.0))
            delta += weight * float(value) * deficit / max(target, 1.0)
        scored.append((delta, group_id))
    scored.sort(reverse=reverse)
    return [group_id for _, group_id in scored[: max(1, limit)]]


def improve_selection(
    selected: set[str],
    group_ids: list[str],
    group_task_counts: dict[str, Counter[str]],
    *,
    target_counts: dict[str, float],
    task_weights: dict[str, float],
    target_total_items: float,
    swap_passes: int,
) -> tuple[set[str], Counter[str], int, float]:
    selected_counts: Counter[str] = Counter()
    selected_total = 0
    for group_id in selected:
        selected_counts.update(group_task_counts[group_id])
        selected_total += int(sum(group_task_counts[group_id].values()))
    current_score = objective(
        selected_counts,
        target_counts=target_counts,
        task_weights=task_weights,
        total_items=selected_total,
        target_total_items=target_total_items,
    )

    selected_list = sorted(selected)
    unselected_list = [group_id for group_id in group_ids if group_id not in selected]

    for _ in range(max(0, swap_passes)):
        best_pair: tuple[str, str] | None = None
        best_counts: Counter[str] | None = None
        best_total = selected_total
        best_score = current_score

        out_candidates = ranked_candidate_pool(
            selected_list,
            group_task_counts,
            selected_counts,
            target_counts=target_counts,
            task_weights=task_weights,
            limit=min(len(selected_list), 24),
            reverse=False,
        )
        in_candidates = ranked_candidate_pool(
            unselected_list,
            group_task_counts,
            selected_counts,
            target_counts=target_counts,
            task_weights=task_weights,
            limit=min(len(unselected_list), 24),
            reverse=True,
        )

        for group_out in out_candidates:
            minus_counts = counter_sub(selected_counts, group_task_counts[group_out])
            minus_total = selected_total - int(sum(group_task_counts[group_out].values()))
            for group_in in in_candidates:
                new_counts = counter_add(minus_counts, group_task_counts[group_in])
                new_total = minus_total + int(sum(group_task_counts[group_in].values()))
                score = objective(
                    new_counts,
                    target_counts=target_counts,
                    task_weights=task_weights,
                    total_items=new_total,
                    target_total_items=target_total_items,
                )
                if score < best_score - 1e-12:
                    best_pair = (group_out, group_in)
                    best_counts = new_counts
                    best_total = new_total
                    best_score = score

        if best_pair is None or best_counts is None:
            break

        group_out, group_in = best_pair
        selected.remove(group_out)
        selected.add(group_in)
        selected_counts = best_counts
        selected_total = best_total
        current_score = best_score
        selected_list = sorted(selected)
        unselected_list = [group_id for group_id in group_ids if group_id not in selected]

    return selected, selected_counts, selected_total, current_score


def solve_dataset_split(
    cfg: DatasetConfig,
    items: list[dict[str, Any]],
    *,
    eval_ratio: float,
    restarts: int,
    swap_passes: int,
    seed: int,
) -> dict[str, Any]:
    group_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for item in items:
        group_id = group_id_for_item(cfg, item)
        group_items[group_id].append(item)
        group_task_counts[group_id][str(item.get("task_type", ""))] += 1
        totals[str(item.get("task_type", ""))] += 1

    group_ids = sorted(group_items)
    num_groups = len(group_ids)
    eval_group_count = max(1, round(num_groups * float(eval_ratio)))
    target_counts = {task_type: float(count) * float(eval_ratio) for task_type, count in totals.items()}
    target_total_items = float(sum(totals.values())) * float(eval_ratio)
    task_weights = {task_type: 1.0 for task_type in totals}
    task_weights.update(cfg.task_weights)

    best_selected: set[str] | None = None
    best_counts: Counter[str] | None = None
    best_total = 0
    best_score = None
    root_rng = random.Random(seed)

    for restart_idx in range(max(1, restarts)):
        local_rng = random.Random(root_rng.randint(0, 10**12) + restart_idx)
        selected, selected_counts, selected_total = greedy_select(
            group_ids,
            group_task_counts,
            eval_group_count,
            target_counts=target_counts,
            task_weights=task_weights,
            target_total_items=target_total_items,
            rng=local_rng,
        )
        selected, selected_counts, selected_total, score = improve_selection(
            selected,
            group_ids,
            group_task_counts,
            target_counts=target_counts,
            task_weights=task_weights,
            target_total_items=target_total_items,
            swap_passes=swap_passes,
        )
        if best_score is None or score < best_score - 1e-12:
            best_selected = set(selected)
            best_counts = Counter(selected_counts)
            best_total = selected_total
            best_score = score

    assert best_selected is not None
    assert best_counts is not None

    eval_group_ids = best_selected
    sft_group_ids = set(group_ids) - eval_group_ids

    def enrich_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in rows:
            group_id = group_id_for_item(cfg, item)
            enriched = dict(item)
            enriched.setdefault("dataset", cfg.name)
            enriched["source_dataset"] = cfg.name
            enriched["split"] = split_name
            enriched["split_version"] = "splits_v1"
            enriched["split_group_id"] = group_id
            out.append(enriched)
        return out

    eval_rows = enrich_rows(
        [item for group_id in group_ids if group_id in eval_group_ids for item in group_items[group_id]],
        "eval",
    )
    sft_rows = enrich_rows(
        [item for group_id in group_ids if group_id in sft_group_ids for item in group_items[group_id]],
        "sft",
    )

    sft_counts = Counter(str(item.get("task_type", "")) for item in sft_rows)
    eval_counts = Counter(str(item.get("task_type", "")) for item in eval_rows)

    return {
        "dataset": cfg.name,
        "slug": cfg.slug,
        "group_mode": cfg.group_mode,
        "num_items_total": len(items),
        "num_groups_total": num_groups,
        "eval_ratio_target": float(eval_ratio),
        "num_groups_eval": len(eval_group_ids),
        "num_groups_sft": len(sft_group_ids),
        "num_items_eval": len(eval_rows),
        "num_items_sft": len(sft_rows),
        "task_counts_total": dict(totals),
        "task_counts_eval": dict(eval_counts),
        "task_counts_sft": dict(sft_counts),
        "task_target_eval": {k: round(v, 3) for k, v in target_counts.items()},
        "eval_group_ids": sorted(eval_group_ids),
        "sft_group_ids": sorted(sft_group_ids),
        "eval_rows": eval_rows,
        "sft_rows": sft_rows,
        "baselines": dict(cfg.baselines),
        "frame_dir_default": str(cfg.frame_dir_default),
        "frame_dir_t3_multi": str(cfg.frame_dir_t3_multi),
        "frame_dir_t6_multi": str(cfg.frame_dir_t6_multi),
        "selection_objective": float(best_score or 0.0),
        "selection_task_weights": task_weights,
        "selection_group_count_target": int(eval_group_count),
        "selection_item_count_target": round(target_total_items, 3),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict[str, Any]] = []
    merged_sft: list[dict[str, Any]] = []
    merged_eval: list[dict[str, Any]] = []
    eval_manifest_entries: list[dict[str, Any]] = []

    for cfg in DATASET_CONFIGS:
        items = load_items(cfg)
        summary = solve_dataset_split(
            cfg,
            items,
            eval_ratio=float(args.eval_ratio),
            restarts=int(args.restarts),
            swap_passes=int(args.swap_passes),
            seed=int(args.seed),
        )

        sft_path = output_dir / f"{cfg.slug}_sft.jsonl"
        eval_path = output_dir / f"{cfg.slug}_eval.jsonl"
        write_jsonl(sft_path, summary["sft_rows"])
        write_jsonl(eval_path, summary["eval_rows"])

        merged_sft.extend(summary["sft_rows"])
        merged_eval.extend(summary["eval_rows"])

        result_jsonl = results_dir / f"{cfg.slug}_eval_results.jsonl"
        eval_manifest_entries.append(
            {
                "dataset": cfg.name,
                "slug": cfg.slug,
                "eval_jsonl": str(eval_path),
                "sft_jsonl": str(sft_path),
                "frame_dir_default": summary["frame_dir_default"],
                "frame_dir_t3_multi": summary["frame_dir_t3_multi"],
                "frame_dir_t6_multi": summary["frame_dir_t6_multi"],
                "result_jsonl": str(result_jsonl),
                "baselines": summary["baselines"],
                "num_items_eval": summary["num_items_eval"],
                "num_items_sft": summary["num_items_sft"],
                "num_groups_eval": summary["num_groups_eval"],
                "num_groups_sft": summary["num_groups_sft"],
            }
        )

        dataset_summaries.append(
            {
                key: value
                for key, value in summary.items()
                if key not in {"eval_rows", "sft_rows"}
            }
        )

    merged_sft_path = output_dir / "all_sft_merged.jsonl"
    merged_eval_path = output_dir / "all_eval_merged.jsonl"
    write_jsonl(merged_sft_path, merged_sft)
    write_jsonl(merged_eval_path, merged_eval)

    summary_obj = {
        "split_version": args.split_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "eval_ratio": float(args.eval_ratio),
        "restarts": int(args.restarts),
        "swap_passes": int(args.swap_passes),
        "datasets": dataset_summaries,
        "num_items_total": int(sum(x["num_items_total"] for x in dataset_summaries)),
        "num_items_eval_total": int(sum(x["num_items_eval"] for x in dataset_summaries)),
        "num_items_sft_total": int(sum(x["num_items_sft"] for x in dataset_summaries)),
        "merged_sft_jsonl": str(merged_sft_path),
        "merged_eval_jsonl": str(merged_eval_path),
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_obj = {
        "split_version": args.split_version,
        "created_at": summary_obj["created_at"],
        "summary_json": str(summary_path),
        "merged_sft_jsonl": str(merged_sft_path),
        "merged_eval_jsonl": str(merged_eval_path),
        "datasets": eval_manifest_entries,
    }
    manifest_path = output_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
