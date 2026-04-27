#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = BENCHMARK_DIR / "splits_v1" / "eval_manifest.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "eval_results_v1" / "splits_v1"
INVALID_PRED = {"MISSING_FRAME", "ERROR", "INVALID"}
TASK_TYPE_ALIASES = {
    "T9": "T_binary",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score all eval-set result files from a shared manifest.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument(
        "--model",
        default=None,
        help="Optional model name used to resolve a model-specific subdirectory under --results-dir.",
    )
    p.add_argument("--keep-invalid", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def model_to_slug(model: str) -> str:
    text = str(model).strip()
    if not text:
        return "unknown_model"
    text = text.replace("/", "_").replace(".", "-")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_model"


def canonical_task_type(task_type: str) -> str:
    return TASK_TYPE_ALIASES.get(str(task_type), str(task_type))


def canonicalize_baselines(baselines: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for raw_key, value in baselines.items():
        merged[canonical_task_type(str(raw_key))] = float(value)
    return merged


def resolve_results_dir(base_results_dir: Path, model: str | None) -> tuple[Path, str | None]:
    if not model:
        return base_results_dir, None
    model_slug = model_to_slug(model)
    return base_results_dir / model_slug, model_slug


def load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_rows(rows: list[dict], *, keep_invalid: bool) -> tuple[dict[str, list[bool]], int]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    invalid = 0
    for row in rows:
        pred = str(row.get("vlm_answer", ""))
        if (not keep_invalid) and pred in INVALID_PRED:
            invalid += 1
            continue
        by_task[canonical_task_type(str(row.get("task_type", "unknown")))].append(
            bool(row.get("correct", False))
        )
    return by_task, invalid


def summarize_dataset(
    *,
    dataset_name: str,
    slug: str,
    rows: list[dict],
    baselines: dict[str, float],
    keep_invalid: bool,
) -> dict:
    baselines = canonicalize_baselines(baselines)
    by_task, invalid = score_rows(rows, keep_invalid=keep_invalid)
    task_summary: dict[str, dict[str, float | int]] = {}
    total_valid = 0
    total_correct = 0
    for task_type in sorted(by_task):
        vals = by_task[task_type]
        n = len(vals)
        correct = sum(1 for x in vals if x)
        acc = (correct / n) if n else 0.0
        task_summary[task_type] = {
            "n": int(n),
            "correct": int(correct),
            "acc": acc,
            "baseline": float(baselines.get(task_type, 0.25)),
        }
        total_valid += n
        total_correct += correct

    overall_acc = (total_correct / total_valid) if total_valid else 0.0
    macro_task_acc = (
        sum(info["acc"] for info in task_summary.values()) / len(task_summary)
        if task_summary
        else 0.0
    )
    return {
        "dataset": dataset_name,
        "slug": slug,
        "num_rows_raw": len(rows),
        "num_rows_valid": int(total_valid),
        "num_invalid_ignored": int(invalid),
        "overall_acc": overall_acc,
        "macro_task_acc": macro_task_acc,
        "tasks": task_summary,
    }


def main() -> None:
    args = parse_args()
    manifest = load_json(Path(args.manifest))
    results_dir, model_slug = resolve_results_dir(Path(args.results_dir), args.model)

    dataset_summaries: list[dict] = []
    merged_task_correct: dict[str, list[bool]] = defaultdict(list)
    total_valid = 0
    total_correct = 0

    for entry in manifest.get("datasets", []):
        result_path = results_dir / f"{entry['slug']}_eval_results.jsonl"
        rows = load_results(result_path)
        summary = summarize_dataset(
            dataset_name=str(entry["dataset"]),
            slug=str(entry["slug"]),
            rows=rows,
            baselines=dict(entry.get("baselines", {})),
            keep_invalid=bool(args.keep_invalid),
        )
        dataset_summaries.append(summary)
        for task_type, info in summary["tasks"].items():
            vals = [True] * int(info["correct"]) + [False] * (int(info["n"]) - int(info["correct"]))
            merged_task_correct[task_type].extend(vals)
            total_valid += int(info["n"])
            total_correct += int(info["correct"])

    overall_micro = (total_correct / total_valid) if total_valid else 0.0
    dataset_macro = (
        sum(x["overall_acc"] for x in dataset_summaries) / len(dataset_summaries)
        if dataset_summaries
        else 0.0
    )
    task_macro = (
        sum(sum(1 for v in vals if v) / len(vals) for vals in merged_task_correct.values()) / len(merged_task_correct)
        if merged_task_correct
        else 0.0
    )

    merged_task_summary = {
        task_type: {
            "n": int(len(vals)),
            "correct": int(sum(1 for v in vals if v)),
            "acc": (sum(1 for v in vals if v) / len(vals)) if vals else 0.0,
        }
        for task_type, vals in sorted(merged_task_correct.items())
    }

    report = {
        "manifest": str(Path(args.manifest)),
        "results_dir": str(results_dir),
        "model": args.model,
        "model_slug": model_slug,
        "task_type_aliases": TASK_TYPE_ALIASES,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "keep_invalid": bool(args.keep_invalid),
        "overall_micro_acc": overall_micro,
        "dataset_macro_acc": dataset_macro,
        "task_macro_acc": task_macro,
        "datasets": dataset_summaries,
        "merged_tasks": merged_task_summary,
    }
    out_name = "score_all_eval_sets_summary.json" if not model_slug else f"score_summary_{model_slug}.json"
    out_path = results_dir / out_name
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
