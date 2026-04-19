#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALLOWED_LABELS = ("left", "right", "top", "bottom")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit RH20T T3 image-direction mapping from human calibration sheet.")
    p.add_argument(
        "--annotation-sheet",
        default="/data/projects/GM-100/benchmark/rh20t_t3_direction_calibration_v0/annotation_sheet.csv",
    )
    p.add_argument(
        "--output-json",
        default="/data/projects/GM-100/benchmark/gt_build/rh20t/rh20t_t3_direction_mapping_036422060215.json",
    )
    p.add_argument(
        "--camera",
        default="036422060215",
    )
    p.add_argument(
        "--context-offsets",
        default="-20,-10,0,10",
        help="Comma-separated frame offsets used by the RH20T T3 builder.",
    )
    return p.parse_args()


def parse_offsets(text: str) -> list[int]:
    vals: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals or 0 not in vals:
        raise ValueError("context offsets must be non-empty and include 0")
    return vals


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append({str(k): v for k, v in row.items()})
    return rows


def main() -> None:
    args = parse_args()
    annotation_path = Path(args.annotation_sheet)
    output_path = Path(args.output_json)
    offsets = parse_offsets(args.context_offsets)

    rows = load_rows(annotation_path)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_by_auto: Counter[str] = Counter()
    kept = 0
    unclear = 0
    invalid = 0

    for row in rows:
        auto_raw = str(row.get("auto_tcp_direction_raw", "")).strip().lower()
        human = str(row.get("human_label", "")).strip().lower()
        if auto_raw not in ALLOWED_LABELS:
            invalid += 1
            continue
        total_by_auto[auto_raw] += 1
        if human == "unclear":
            unclear += 1
            continue
        if human not in ALLOWED_LABELS:
            invalid += 1
            continue
        counts[auto_raw][human] += 1
        kept += 1

    mapping: dict[str, str] = {}
    filtered_buckets: list[dict[str, Any]] = []
    bucket_meta: dict[str, Any] = {}
    for auto_raw in ALLOWED_LABELS:
        bucket = counts.get(auto_raw, Counter())
        bucket_dict = {label: int(bucket[label]) for label in bucket}
        bucket_meta[auto_raw] = bucket_dict
        if not bucket:
            filtered_buckets.append(
                {
                    "auto_tcp_direction_raw": auto_raw,
                    "reason": "No valid human labels after excluding unclear/invalid rows.",
                }
            )
            continue
        ranked = bucket.most_common()
        top_label, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        support = int(sum(bucket.values()))
        margin = int(top_count - second_count)
        confidence = float(top_count / max(1, support))
        mapping[auto_raw] = top_label
        bucket_meta[auto_raw] = {
            "counts": bucket_dict,
            "selected_label": top_label,
            "support": support,
            "top_count": int(top_count),
            "second_count": int(second_count),
            "margin": margin,
            "confidence": round(confidence, 6),
        }
        if len(ranked) > 1 and top_count == second_count:
            filtered_buckets.append(
                {
                    "auto_tcp_direction_raw": auto_raw,
                    "reason": f"Tie after excluding unclear rows: {bucket_dict}",
                }
            )

    result = {
        "camera": str(args.camera),
        "source_annotation_sheet": str(annotation_path),
        "context_offsets": offsets,
        "allowed_human_labels": list(ALLOWED_LABELS),
        "mapping": mapping,
        "filtered_buckets": filtered_buckets,
        "calibration_counts": bucket_meta,
        "summary": {
            "num_rows_total": len(rows),
            "num_rows_used": kept,
            "num_unclear_skipped": unclear,
            "num_invalid_skipped": invalid,
            "total_by_auto_tcp_direction": {k: int(v) for k, v in total_by_auto.items()},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
