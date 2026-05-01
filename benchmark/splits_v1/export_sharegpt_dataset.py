#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = THIS_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent
DEFAULT_MANIFEST = THIS_DIR / "eval_manifest.json"
DEFAULT_SFT_JSONL = THIS_DIR / "all_sft_merged.jsonl"
DEFAULT_EVAL_JSONL = THIS_DIR / "all_eval_merged.jsonl"
DEFAULT_OUTPUT_DIR = THIS_DIR / "sharegpt_export_v1"
DEFAULT_T3_OFFSETS = "-10,-5,0,5"


def load_run_pilot_eval_module():
    mod_path = BENCHMARK_DIR / "eval_v1" / "run_pilot_eval.py"
    spec = importlib.util.spec_from_file_location("run_pilot_eval_export_helper", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export merged SFT/eval splits into LLaMA-Factory ShareGPT format.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--sft-jsonl", default=str(DEFAULT_SFT_JSONL))
    p.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL_JSONL))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--t3-offsets", default=DEFAULT_T3_OFFSETS)
    p.add_argument("--copy-mode", choices=["copy", "hardlink"], default="hardlink")
    return p.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_frame_dir_map(manifest: dict[str, Any]) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for entry in manifest.get("datasets", []):
        keys = {
            str(entry.get("dataset", "")).strip().upper(),
            str(entry.get("slug", "")).strip().upper(),
        }
        source_key = str(entry.get("dataset", "")).strip().upper()
        if source_key == "AIST":
            keys.add("AIST-BIMANUAL")
        frame_dirs = {
            "default": Path(entry["frame_dir_default"]),
            "T3_multi": Path(entry["frame_dir_t3_multi"]),
            "T6_multi": Path(entry["frame_dir_t6_multi"]),
        }
        for k in keys:
            if k:
                out[k] = frame_dirs
    return out


def answer_text(item: dict[str, Any]) -> str:
    answer = str(item.get("answer", "")).strip()
    choices = item.get("choices")
    if isinstance(choices, dict) and answer in choices:
        return f"{answer}. {choices[answer]}"
    return answer


def prompt_text(item: dict[str, Any], image_count: int) -> str:
    question = str(item.get("question", "")).strip()
    choices = item.get("choices")
    lines: list[str] = []
    image_count = max(1, int(image_count))
    lines.append(f"Number of images: {image_count}")
    if question:
        lines.append(question)
    if isinstance(choices, dict) and choices:
        lines.append("Options:")
        for key, value in choices.items():
            lines.append(f"{key}. {value}")
        lines.append("Answer with the option label and text.")
    else:
        lines.append("Answer briefly.")
    return "\n".join(["<image>"] * image_count + lines)


def export_record(
    *,
    item: dict[str, Any],
    frame_paths: list[Path],
    images_dir: Path,
    copy_mode: str,
) -> dict[str, Any]:
    image_rel_paths: list[str] = []
    for src in frame_paths:
        if not src.exists():
            raise FileNotFoundError(f"missing frame for export: {src}")
        dst = images_dir / src.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "hardlink":
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        image_rel_paths.append(str(Path("images") / dst.name))

    return {
        "conversations": [
            {"from": "human", "value": prompt_text(item, len(image_rel_paths))},
            {"from": "gpt", "value": answer_text(item)},
        ],
        "images": image_rel_paths,
        "dataset": item.get("dataset"),
        "source_dataset": item.get("source_dataset"),
        "task_type": item.get("task_type"),
        "task_id": item.get("task_id"),
        "split": item.get("split"),
        "split_group_id": item.get("split_group_id"),
        "recording_id": item.get("recording_id"),
        "episode_id": item.get("episode_id", item.get("episode_index")),
        "camera": item.get("camera"),
        "answer": item.get("answer"),
    }


def export_split(
    *,
    input_jsonl: Path,
    output_json: Path,
    images_dir: Path,
    helper: Any,
    frame_dir_map: dict[str, dict[str, Path]],
    t3_offsets: list[int],
    copy_mode: str,
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    count = 0
    missing = 0

    with input_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            item = helper.normalize_item_for_eval(item)
            dataset_key = str(item.get("source_dataset") or item.get("dataset") or "").strip().upper()
            frame_dirs = frame_dir_map.get(dataset_key)
            if frame_dirs is None:
                raise KeyError(f"no frame-dir mapping for dataset={dataset_key!r}")
            frame_paths = helper.get_frame_paths(item, frame_dirs=frame_dirs, t3_offsets=t3_offsets)
            frame_paths = helper.resolve_missing_frame_paths(item, frame_paths, t3_offsets=t3_offsets)
            if any(not p.exists() for p in frame_paths):
                missing += 1
                missing_paths = [str(p) for p in frame_paths if not p.exists()]
                raise FileNotFoundError(
                    f"export encountered missing frame(s) for {item.get('split_group_id')}: {missing_paths[:3]}"
                )
            rows.append(
                export_record(
                    item=item,
                    frame_paths=frame_paths,
                    images_dir=images_dir,
                    copy_mode=copy_mode,
                )
            )
            count += 1
            if count % 5000 == 0:
                print(f"[INFO] exported {count} items from {input_jsonl.name}")

    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"items": count, "missing": missing}


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "gmbench_train_sharegpt": {
            "file_name": "train_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
        "gmbench_test_sharegpt": {
            "file_name": "test_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    helper = load_run_pilot_eval_module()
    manifest = load_manifest(Path(args.manifest))
    frame_dir_map = build_frame_dir_map(manifest)
    t3_offsets = helper.parse_offsets_csv(args.t3_offsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] prewarming GM100 fallback frames for SFT split")
    sft_items = []
    with Path(args.sft_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                sft_items.append(json.loads(line))
    helper.prewarm_gm100_missing_frames(
        items=sft_items,
        frame_dirs=frame_dir_map["GM100"],
        t3_offsets=t3_offsets,
    )

    print("[INFO] prewarming GM100 fallback frames for eval split")
    eval_items = []
    with Path(args.eval_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                eval_items.append(json.loads(line))
    helper.prewarm_gm100_missing_frames(
        items=eval_items,
        frame_dirs=frame_dir_map["GM100"],
        t3_offsets=t3_offsets,
    )

    train_stats = export_split(
        input_jsonl=Path(args.sft_jsonl),
        output_json=output_dir / "train_sharegpt.json",
        images_dir=images_dir,
        helper=helper,
        frame_dir_map=frame_dir_map,
        t3_offsets=t3_offsets,
        copy_mode=args.copy_mode,
    )
    test_stats = export_split(
        input_jsonl=Path(args.eval_jsonl),
        output_json=output_dir / "test_sharegpt.json",
        images_dir=images_dir,
        helper=helper,
        frame_dir_map=frame_dir_map,
        t3_offsets=t3_offsets,
        copy_mode=args.copy_mode,
    )
    write_dataset_info(output_dir)

    summary = {
        "train_items": train_stats["items"],
        "test_items": test_stats["items"],
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "copy_mode": args.copy_mode,
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
