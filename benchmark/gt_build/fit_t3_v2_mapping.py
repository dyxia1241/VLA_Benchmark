from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from build_t3_gt import (
    T3_LABELS,
    apply_mapping_to_xy,
    mapping_candidates,
    planar_label_from_image_delta,
)


VALID_HUMAN_LABELS = set(T3_LABELS) | {"unclear", "skip", ""}
LABEL_TO_UNIT_VECTOR = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "top": (0.0, -1.0),
    "bottom": (0.0, 1.0),
}


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["calibration_id"])] = obj
    return out


def load_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def normalize_label(text: str) -> str:
    value = str(text or "").strip().lower()
    aliases = {
        "up": "top",
        "down": "bottom",
        "left": "left",
        "right": "right",
        "top": "top",
        "bottom": "bottom",
        "unclear": "unclear",
        "skip": "skip",
        "": "",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported human_label={text!r}")
    return aliases[value]


def usable_rows(candidates: dict[str, dict[str, Any]], annotations: list[dict[str, str]]) -> list[tuple[dict[str, Any], str]]:
    rows: list[tuple[dict[str, Any], str]] = []
    for row in annotations:
        cid = str(row.get("calibration_id", "")).strip()
        if not cid or cid not in candidates:
            continue
        label = normalize_label(row.get("human_label", ""))
        if label in {"", "unclear", "skip"}:
            continue
        rows.append((candidates[cid], label))
    return rows


def parse_exclude_buckets(text: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for chunk in str(text or "").split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid exclude bucket: {part!r}")
        query_arm, robot_direction_raw = [x.strip() for x in part.split(":", 1)]
        if query_arm not in {"left", "right"}:
            raise ValueError(f"Invalid query_arm in exclude bucket: {part!r}")
        if robot_direction_raw not in {"+x", "-x", "+y", "-y"}:
            raise ValueError(f"Invalid robot_direction_raw in exclude bucket: {part!r}")
        pairs.add((query_arm, robot_direction_raw))
    return pairs


def filter_rows_by_exclude_buckets(
    rows: list[tuple[dict[str, Any], str]],
    exclude_pairs: set[tuple[str, str]],
) -> list[tuple[dict[str, Any], str]]:
    if not exclude_pairs:
        return list(rows)
    return [
        (cand, label)
        for cand, label in rows
        if (str(cand["query_arm"]), str(cand["robot_direction_raw"])) not in exclude_pairs
    ]


def dedup_temporal_neighbors(
    rows: list[tuple[dict[str, Any], str]],
    max_gap: int,
) -> list[tuple[dict[str, Any], str]]:
    if max_gap <= 0:
        return list(rows)

    ordered = sorted(
        rows,
        key=lambda x: (
            str(x[0]["task_id"]),
            int(x[0]["episode_id"]),
            str(x[0]["query_arm"]),
            str(x[0]["robot_direction_raw"]),
            x[1],
            int(x[0]["frame_index"]),
        ),
    )
    groups: list[list[tuple[dict[str, Any], str]]] = []
    cur: list[tuple[dict[str, Any], str]] = []
    for cand, label in ordered:
        key = (
            str(cand["task_id"]),
            int(cand["episode_id"]),
            str(cand["query_arm"]),
            str(cand["robot_direction_raw"]),
            label,
        )
        if not cur:
            cur = [(cand, label)]
            continue
        prev_cand, prev_label = cur[-1]
        prev_key = (
            str(prev_cand["task_id"]),
            int(prev_cand["episode_id"]),
            str(prev_cand["query_arm"]),
            str(prev_cand["robot_direction_raw"]),
            prev_label,
        )
        if key == prev_key and int(cand["frame_index"]) - int(prev_cand["frame_index"]) <= int(max_gap):
            cur.append((cand, label))
        else:
            groups.append(cur)
            cur = [(cand, label)]
    if cur:
        groups.append(cur)
    return [group[len(group) // 2] for group in groups]


def predict_label(candidate: dict[str, Any], mapping: Any, purity_ratio: float) -> str | None:
    image_xy = apply_mapping_to_xy(tuple(candidate["robot_plane_displacement_xy"]), mapping)
    label, _, _, _ = planar_label_from_image_delta(image_xy, dominant_min=1e-9, purity_ratio=purity_ratio)
    return label


def score_global(rows: list[tuple[dict[str, Any], str]], purity_ratio: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for mapping in mapping_candidates():
        total = 0
        correct = 0
        for cand, human_label in rows:
            pred = predict_label(cand, mapping, purity_ratio=purity_ratio)
            total += 1
            correct += int(pred == human_label)
        acc = float(correct / total) if total else 0.0
        results.append(
            {
                "scope": "global",
                "mapping_name": mapping.name,
                "mapping": mapping.to_dict(),
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
        )
    return sorted(results, key=lambda x: (-x["accuracy"], -x["correct"], x["mapping_name"]))


def score_per_arm(rows: list[tuple[dict[str, Any], str]], purity_ratio: float) -> list[dict[str, Any]]:
    mappings = mapping_candidates()
    results: list[dict[str, Any]] = []
    for left_mapping, right_mapping in product(mappings, mappings):
        total = 0
        correct = 0
        for cand, human_label in rows:
            mapping = left_mapping if str(cand["query_arm"]) == "left" else right_mapping
            pred = predict_label(cand, mapping, purity_ratio=purity_ratio)
            total += 1
            correct += int(pred == human_label)
        acc = float(correct / total) if total else 0.0
        results.append(
            {
                "scope": "per_arm",
                "left_mapping_name": left_mapping.name,
                "right_mapping_name": right_mapping.name,
                "mappings": {
                    "left": left_mapping.to_dict(),
                    "right": right_mapping.to_dict(),
                },
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
        )
    return sorted(
        results,
        key=lambda x: (-x["accuracy"], -x["correct"], x["left_mapping_name"], x["right_mapping_name"]),
    )


def fit_linear_per_arm(rows: list[tuple[dict[str, Any], str]]) -> dict[str, np.ndarray]:
    mats: dict[str, np.ndarray] = {}
    for arm in ("left", "right"):
        xs: list[tuple[float, float]] = []
        ys: list[tuple[float, float]] = []
        for cand, label in rows:
            if str(cand["query_arm"]) != arm:
                continue
            xs.append((float(cand["robot_plane_displacement_xy"][0]), float(cand["robot_plane_displacement_xy"][1])))
            ys.append(LABEL_TO_UNIT_VECTOR[label])
        if not xs:
            raise ValueError(f"No usable rows for arm={arm!r} after filtering.")
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        mat, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
        mats[arm] = mat
    return mats


def predict_linear_label(candidate: dict[str, Any], mat: np.ndarray, purity_ratio: float) -> str | None:
    robot_xy = np.asarray(tuple(candidate["robot_plane_displacement_xy"]), dtype=float)
    image_xy = robot_xy @ mat
    label, _, _, _ = planar_label_from_image_delta(
        (float(image_xy[0]), float(image_xy[1])),
        dominant_min=1e-9,
        purity_ratio=purity_ratio,
    )
    return label


def bucket_accuracy(rows: list[tuple[dict[str, Any], str]], predictor) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for cand, human_label in rows:
        key = f"{cand['query_arm']}|{cand['robot_direction_raw']}"
        pred = predictor(cand)
        bucket = stats.setdefault(key, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(pred == human_label)
    for key, bucket in stats.items():
        total = int(bucket["total"])
        bucket["accuracy"] = float(bucket["correct"] / total) if total else 0.0
    return dict(sorted(stats.items()))


def score_linear_per_arm(rows: list[tuple[dict[str, Any], str]], purity_ratio: float) -> dict[str, Any]:
    mats = fit_linear_per_arm(rows)
    total = 0
    correct = 0
    for cand, human_label in rows:
        mat = mats[str(cand["query_arm"])]
        pred = predict_linear_label(cand, mat, purity_ratio=purity_ratio)
        total += 1
        correct += int(pred == human_label)
    acc = float(correct / total) if total else 0.0
    return {
        "scope": "per_arm",
        "mapping_kind": "linear",
        "mappings": {
            "left": {
                "kind": "linear",
                "name": "left_linear_ls",
                "matrix": mats["left"].tolist(),
            },
            "right": {
                "kind": "linear",
                "name": "right_linear_ls",
                "matrix": mats["right"].tolist(),
            },
        },
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "bucket_accuracy": bucket_accuracy(
            rows,
            predictor=lambda cand: predict_linear_label(cand, mats[str(cand["query_arm"])], purity_ratio=purity_ratio),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit T3 v2 scene-direction mapping from human calibration labels.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--purity-ratio", type=float, default=2.0)
    parser.add_argument("--global-accept-accuracy", type=float, default=0.9)
    parser.add_argument("--per-arm-accept-accuracy", type=float, default=0.9)
    parser.add_argument("--fit-linear", action="store_true", help="Also fit per-arm 2x2 linear mappings when signed-axis search is insufficient.")
    parser.add_argument("--dedup-max-gap", type=int, default=0, help="Collapse near-duplicate temporal neighbors before fitting. 0 disables dedup.")
    parser.add_argument(
        "--exclude-buckets",
        default="",
        help="Comma-separated query_arm:robot_direction_raw pairs to exclude, e.g. right:+x,right:-x",
    )
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.candidates_jsonl))
    annotations = load_annotations(Path(args.annotations_csv))
    rows_raw = usable_rows(candidates, annotations)
    if not rows_raw:
        raise ValueError("No usable human labels found. Fill calibration_annotations.csv first.")
    exclude_pairs = parse_exclude_buckets(args.exclude_buckets)
    rows_fit = dedup_temporal_neighbors(rows_raw, max_gap=int(args.dedup_max_gap))
    rows_fit = filter_rows_by_exclude_buckets(rows_fit, exclude_pairs)
    if not rows_fit:
        raise ValueError("No usable rows remain after dedup / bucket exclusion.")

    global_results = score_global(rows_fit, purity_ratio=args.purity_ratio)
    best_global = global_results[0]

    chosen_config: dict[str, Any] | None = None
    per_arm_results: list[dict[str, Any]] = []
    linear_result: dict[str, Any] | None = None
    if best_global["accuracy"] >= float(args.global_accept_accuracy):
        chosen_config = {
            "mapping_scope": "global",
            "mapping_kind": "signed_axis",
            "mappings": {
                "global": best_global["mapping"],
            },
            "selected_from": "global_search",
            "num_labeled_rows_raw": len(rows_raw),
            "num_labeled_rows_fit": len(rows_fit),
            "accuracy": best_global["accuracy"],
        }
    else:
        per_arm_results = score_per_arm(rows_fit, purity_ratio=args.purity_ratio)
        best_per_arm = per_arm_results[0]
        if best_per_arm["accuracy"] >= float(args.per_arm_accept_accuracy):
            chosen_config = {
                "mapping_scope": "per_arm",
                "mapping_kind": "signed_axis",
                "mappings": best_per_arm["mappings"],
                "selected_from": "per_arm_search",
                "num_labeled_rows_raw": len(rows_raw),
                "num_labeled_rows_fit": len(rows_fit),
                "accuracy": best_per_arm["accuracy"],
            }
        elif args.fit_linear:
            linear_result = score_linear_per_arm(rows_fit, purity_ratio=args.purity_ratio)
            chosen_config = {
                "mapping_scope": "per_arm",
                "mapping_kind": "linear",
                "mappings": linear_result["mappings"],
                "selected_from": "per_arm_linear_search",
                "num_labeled_rows_raw": len(rows_raw),
                "num_labeled_rows_fit": len(rows_fit),
                "accuracy": linear_result["accuracy"],
            }
        else:
            chosen_config = {
                "mapping_scope": "global",
                "mapping_kind": "signed_axis",
                "mappings": {
                    "global": best_global["mapping"],
                },
                "selected_from": "best_global_below_threshold",
                "num_labeled_rows_raw": len(rows_raw),
                "num_labeled_rows_fit": len(rows_fit),
                "accuracy": best_global["accuracy"],
            }

    if exclude_pairs:
        chosen_config["exclude_robot_direction_raw_pairs"] = [
            {"query_arm": query_arm, "robot_direction_raw": robot_direction_raw}
            for query_arm, robot_direction_raw in sorted(exclude_pairs)
        ]

    out_path = Path(args.output_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chosen_config, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "num_labeled_rows_raw": len(rows_raw),
        "num_labeled_rows_fit": len(rows_fit),
        "exclude_buckets": sorted(f"{a}:{d}" for a, d in exclude_pairs),
        "best_global": best_global,
        "top5_global": global_results[:5],
        "top5_per_arm": per_arm_results[:5],
        "linear_result": linear_result,
        "chosen_config": chosen_config,
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
