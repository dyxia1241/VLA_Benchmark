from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
GT_BUILD_DIR = REPO_ROOT / "benchmark" / "gt_build"
if str(GT_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(GT_BUILD_DIR))

from segmentation import _select_velocity_signal, detect_contact_events  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "gm100-cobotmagic-lerobot"
DEFAULT_ANNOTATION_CSV = REPO_ROOT / "benchmark" / "gt_build" / "task_type_annotation.csv"
DEFAULT_TASK_NAME_CSV = (
    REPO_ROOT / "benchmark" / "manual_audit" / "semantic_affordance_audit" / "task_semantic_annotation_v1.csv"
)

IMMEDIATE_GAP_MAX = 15
SHORT_GAP_MAX = 45
MEDIUM_GAP_MAX = 90
ANCHOR_OVERLAP_MERGE_GAP = 3


SERIAL_REPEAT_PATTERNS = [
    r"one[- ]by[- ]one",
    r"\bitems?\b",
    r"\bobjects?\b",
    r"\bmultiple\b",
    r"\bseveral\b",
    r"\bsort\b",
    r"\borganize\b",
    r"\bstack\b",
    r"\barrange\b",
    r"\bbus-table\b",
    r"\btable-setting\b",
    r"\bput-coins\b",
    r"\bremove-objects\b",
    r"\bplace-seven\b",
]


@dataclass
class RawEvent:
    raw_event_id: str
    arm: str
    contact_frame: int
    release_frame: int


@dataclass
class AnchorEvent:
    anchor_event_id: str
    anchor_order: int
    anchor_start_frame: int
    anchor_end_frame: int
    active_arm_pattern: str
    source_raw_event_ids: list[str]
    source_arms: list[str]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_task_meta(annotation_csv: Path) -> dict[str, dict[str, Any]]:
    rows = pd.read_csv(annotation_csv)
    meta_map: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        rec = row.to_dict()
        rec["has_gripper_motion"] = parse_bool(rec.get("has_gripper_motion", False))
        meta_map[str(rec["task_id"])] = rec
    return meta_map


def load_task_name_map(task_name_csv: Path) -> dict[str, dict[str, str]]:
    if not task_name_csv.exists():
        return {}
    rows = pd.read_csv(task_name_csv)
    out: dict[str, dict[str, str]] = {}
    for _, row in rows.iterrows():
        out[str(row["task_id"])] = {
            "task_name_raw": str(row.get("task_name_raw", "")),
            "task_name_readable": str(row.get("task_name_readable", "")),
            "metadata_status": str(row.get("metadata_status", "")),
        }
    return out


def task_episode_paths(dataset_root: Path, task_id: str) -> list[Path]:
    chunk_dir = dataset_root / task_id / "data" / "chunk-000"
    if not chunk_dir.exists():
        return []
    return sorted(chunk_dir.glob("episode_*.parquet"))


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


def infer_serial_repetition_risk(task_name_raw: str) -> str:
    text = task_name_raw.strip().lower()
    if not text:
        return "unknown"
    score = 0
    for pattern in SERIAL_REPEAT_PATTERNS:
        if re.search(pattern, text):
            score += 1
    if re.search(r"\b(seven|eight|nine|ten|twelve)\b", text):
        score += 1
    if score >= 2:
        return "high"
    if score == 1:
        return "medium"
    return "low"


def build_raw_events(task_id: str, episode_id: int, contact_events: dict[str, list[dict[str, Any]]]) -> list[RawEvent]:
    out: list[RawEvent] = []
    event_idx = 1
    for arm in ("left", "right"):
        for event in contact_events.get(arm, []):
            c = int(event["contact_frame"])
            r = int(event["release_frame"])
            if r <= c:
                continue
            out.append(
                RawEvent(
                    raw_event_id=f"{task_id}__{episode_id}__re{event_idx:03d}",
                    arm=arm,
                    contact_frame=c,
                    release_frame=r,
                )
            )
            event_idx += 1
    out.sort(key=lambda x: (x.contact_frame, x.release_frame, x.arm))
    return out


def _active_arm_pattern(arms: set[str]) -> str:
    if arms == {"left"}:
        return "left"
    if arms == {"right"}:
        return "right"
    if arms == {"left", "right"}:
        return "both"
    return "unknown"


def build_anchor_events(task_id: str, episode_id: int, raw_events: list[RawEvent]) -> list[AnchorEvent]:
    if not raw_events:
        return []

    anchors: list[AnchorEvent] = []
    group: list[RawEvent] = [raw_events[0]]
    current_end = raw_events[0].release_frame

    def flush(group_events: list[RawEvent], order: int) -> AnchorEvent:
        arms = {e.arm for e in group_events}
        return AnchorEvent(
            anchor_event_id=f"{task_id}__{episode_id}__a{order:03d}",
            anchor_order=order,
            anchor_start_frame=min(e.contact_frame for e in group_events),
            anchor_end_frame=max(e.release_frame for e in group_events),
            active_arm_pattern=_active_arm_pattern(arms),
            source_raw_event_ids=[e.raw_event_id for e in group_events],
            source_arms=sorted(arms),
        )

    for raw_event in raw_events[1:]:
        if raw_event.contact_frame <= current_end + ANCHOR_OVERLAP_MERGE_GAP:
            group.append(raw_event)
            current_end = max(current_end, raw_event.release_frame)
            continue
        anchors.append(flush(group, len(anchors) + 1))
        group = [raw_event]
        current_end = raw_event.release_frame

    anchors.append(flush(group, len(anchors) + 1))
    return anchors


def gap_bucket(gap_frames: int) -> str:
    if gap_frames <= IMMEDIATE_GAP_MAX:
        return "immediate"
    if gap_frames <= SHORT_GAP_MAX:
        return "short"
    if gap_frames <= MEDIUM_GAP_MAX:
        return "medium"
    return "long"


def transition_strength(velocity_signal: np.ndarray, prev_end: int, next_start: int) -> str:
    if next_start <= prev_end:
        return "low"
    seg = velocity_signal[prev_end:next_start]
    if len(seg) == 0:
        return "low"
    peak = float(np.max(seg))
    mean = float(np.mean(seg))
    if peak < 0.45 and mean < 0.25:
        return "low"
    if peak < 1.00 and mean < 0.55:
        return "medium"
    return "high"


def pairwise_features(
    prev_anchor: AnchorEvent,
    next_anchor: AnchorEvent,
    velocity_signal: np.ndarray,
    task_serial_risk: str,
) -> dict[str, Any]:
    gap_frames = int(next_anchor.anchor_start_frame - prev_anchor.anchor_end_frame)
    same_active_arm_pattern = "yes" if prev_anchor.active_arm_pattern == next_anchor.active_arm_pattern else "no"
    inter_anchor_transition_strength = transition_strength(
        velocity_signal,
        prev_end=prev_anchor.anchor_end_frame,
        next_start=next_anchor.anchor_start_frame,
    )

    if gap_frames > SHORT_GAP_MAX:
        inter_anchor_reset_hint = "yes"
    elif gap_frames <= IMMEDIATE_GAP_MAX:
        inter_anchor_reset_hint = "no"
    else:
        inter_anchor_reset_hint = "unknown"

    explicit_retry_hint = "unknown"
    if gap_frames <= IMMEDIATE_GAP_MAX and same_active_arm_pattern == "yes" and inter_anchor_transition_strength == "low":
        explicit_retry_hint = "yes"
    elif gap_frames > SHORT_GAP_MAX or same_active_arm_pattern == "no":
        explicit_retry_hint = "no"

    serial_repetition_hint = "no"
    if task_serial_risk == "high" and gap_frames > 8:
        serial_repetition_hint = "yes"
    elif task_serial_risk == "medium" and gap_frames > IMMEDIATE_GAP_MAX:
        serial_repetition_hint = "yes"

    return {
        "gap_frames": gap_frames,
        "gap_bucket": gap_bucket(gap_frames),
        "same_active_arm_pattern": same_active_arm_pattern,
        "inter_anchor_reset_hint": inter_anchor_reset_hint,
        "explicit_retry_hint": explicit_retry_hint,
        "serial_repetition_hint": serial_repetition_hint,
        "inter_anchor_transition_strength": inter_anchor_transition_strength,
        "view_stability_hint": "unknown",
    }


def should_merge_pair(
    features: dict[str, Any],
    current_cluster_anchor_count: int,
) -> tuple[bool, str, list[str]]:
    reason_codes: list[str] = []

    if features["gap_bucket"] == "long":
        return False, "low", ["long_gap_block"]
    if features["serial_repetition_hint"] == "yes":
        return False, "low", ["serial_repetition_risk_block"]
    if features["same_active_arm_pattern"] == "no":
        return False, "low", ["arm_pattern_mismatch_block"]
    if features["inter_anchor_reset_hint"] == "yes" and features["explicit_retry_hint"] != "yes":
        return False, "low", ["reset_hint_block"]

    if features["gap_bucket"] == "immediate":
        reason_codes.append("adjacent_immediate_gap")
    elif features["gap_bucket"] == "short":
        reason_codes.append("adjacent_short_gap")

    if features["same_active_arm_pattern"] == "yes":
        reason_codes.append("same_arm_pattern")
    if features["explicit_retry_hint"] == "yes":
        reason_codes.append("explicit_retry_hint")
    if features["inter_anchor_transition_strength"] == "low":
        reason_codes.append("low_transition_strength")
    if features["inter_anchor_reset_hint"] == "no":
        reason_codes.append("no_reset_evidence")

    if (
        features["gap_bucket"] in {"immediate", "short"}
        and features["explicit_retry_hint"] == "yes"
        and features["same_active_arm_pattern"] != "no"
        and features["inter_anchor_reset_hint"] == "no"
        and features["serial_repetition_hint"] != "yes"
        and current_cluster_anchor_count + 1 <= 3
    ):
        return True, "high", reason_codes

    if (
        features["gap_bucket"] == "immediate"
        and features["explicit_retry_hint"] != "no"
        and features["same_active_arm_pattern"] == "yes"
        and features["inter_anchor_reset_hint"] == "no"
        and features["serial_repetition_hint"] == "no"
        and features["inter_anchor_transition_strength"] == "low"
        and current_cluster_anchor_count + 1 <= 2
    ):
        return True, "medium", reason_codes

    return False, "low", reason_codes or ["uncertain_pair_keep_split"]


def build_cluster_proposals(
    task_id: str,
    episode_id: int,
    anchors: list[AnchorEvent],
    velocity_signal: np.ndarray,
    task_serial_risk: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not anchors:
        return [], []

    proposals: list[dict[str, Any]] = []
    pair_decisions: list[dict[str, Any]] = []

    current = {
        "task_id": task_id,
        "episode_id": int(episode_id),
        "proposal_cluster_id": f"{task_id}__{episode_id}__pc001",
        "proposal_order": 1,
        "source_anchor_event_ids": [anchors[0].anchor_event_id],
        "source_raw_event_ids": list(anchors[0].source_raw_event_ids),
        "proposal_start_frame": anchors[0].anchor_start_frame,
        "proposal_end_frame": anchors[0].anchor_end_frame,
        "active_arm_patterns": [anchors[0].active_arm_pattern],
        "source_anchor_orders": [anchors[0].anchor_order],
        "merge_trace": [],
        "proposal_merge_confidence": "low",
        "proposal_reason_codes": [],
        "serial_repetition_risk": task_serial_risk,
    }
    last_anchor = anchors[0]

    for next_anchor in anchors[1:]:
        features = pairwise_features(last_anchor, next_anchor, velocity_signal, task_serial_risk)
        merge, confidence, reason_codes = should_merge_pair(
            features=features,
            current_cluster_anchor_count=len(current["source_anchor_event_ids"]),
        )
        pair_decisions.append(
            {
                "task_id": task_id,
                "episode_id": int(episode_id),
                "prev_anchor_event_id": last_anchor.anchor_event_id,
                "next_anchor_event_id": next_anchor.anchor_event_id,
                "merge": merge,
                "proposal_merge_confidence": confidence,
                "proposal_reason_codes": reason_codes,
                **features,
            }
        )

        if merge:
            current["source_anchor_event_ids"].append(next_anchor.anchor_event_id)
            current["source_raw_event_ids"].extend(next_anchor.source_raw_event_ids)
            current["proposal_end_frame"] = next_anchor.anchor_end_frame
            current["active_arm_patterns"].append(next_anchor.active_arm_pattern)
            current["source_anchor_orders"].append(next_anchor.anchor_order)
            current["merge_trace"].append(
                {
                    "prev_anchor_event_id": last_anchor.anchor_event_id,
                    "next_anchor_event_id": next_anchor.anchor_event_id,
                    "proposal_merge_confidence": confidence,
                    "proposal_reason_codes": reason_codes,
                    "gap_frames": features["gap_frames"],
                }
            )
            current["proposal_reason_codes"] = sorted(
                set(current["proposal_reason_codes"]).union(reason_codes)
            )
            if confidence == "high":
                current["proposal_merge_confidence"] = "high"
            elif current["proposal_merge_confidence"] != "high":
                current["proposal_merge_confidence"] = "medium"
        else:
            proposals.append(current)
            current = {
                "task_id": task_id,
                "episode_id": int(episode_id),
                "proposal_cluster_id": f"{task_id}__{episode_id}__pc{len(proposals) + 1:03d}",
                "proposal_order": len(proposals) + 1,
                "source_anchor_event_ids": [next_anchor.anchor_event_id],
                "source_raw_event_ids": list(next_anchor.source_raw_event_ids),
                "proposal_start_frame": next_anchor.anchor_start_frame,
                "proposal_end_frame": next_anchor.anchor_end_frame,
                "active_arm_patterns": [next_anchor.active_arm_pattern],
                "source_anchor_orders": [next_anchor.anchor_order],
                "merge_trace": [],
                "proposal_merge_confidence": "low",
                "proposal_reason_codes": [],
                "serial_repetition_risk": task_serial_risk,
            }
        last_anchor = next_anchor

    proposals.append(current)
    return proposals, pair_decisions


def pick_representative_episode(
    dataset_root: Path,
    task_id: str,
    task_meta: dict[str, Any],
) -> tuple[int | None, Path | None, pd.DataFrame | None, dict[str, list[dict[str, Any]]] | None, list[RawEvent] | None]:
    paths = task_episode_paths(dataset_root, task_id)
    if not paths:
        return None, None, None, None, None

    first_episode_id = int(paths[0].stem.split("_")[-1])
    first_df = load_episode_df(paths[0])
    first_contact_events = detect_contact_events(first_df, task_meta=task_meta, min_persist_frames=5)
    first_raw_events = build_raw_events(task_id, first_episode_id, first_contact_events)

    if not task_meta.get("has_gripper_motion", False):
        return first_episode_id, paths[0], first_df, first_contact_events, first_raw_events

    if first_raw_events:
        return first_episode_id, paths[0], first_df, first_contact_events, first_raw_events

    for path in paths[1:]:
        episode_id = int(path.stem.split("_")[-1])
        df = load_episode_df(path)
        contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
        raw_events = build_raw_events(task_id, episode_id, contact_events)
        if raw_events:
            return episode_id, path, df, contact_events, raw_events

    return first_episode_id, paths[0], first_df, first_contact_events, first_raw_events


def summarize_cluster_shapes(proposals: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not proposals:
        return "", "", ""
    cluster_sizes = [len(p["source_anchor_event_ids"]) for p in proposals]
    cluster_spans = [p["proposal_end_frame"] - p["proposal_start_frame"] for p in proposals]
    arm_patterns = ["+".join(p["active_arm_patterns"]) for p in proposals]
    return (
        "|".join(str(x) for x in cluster_sizes),
        "|".join(str(x) for x in cluster_spans),
        "|".join(arm_patterns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Primitive-cluster pilot: one episode per task.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--annotation-csv", default=str(DEFAULT_ANNOTATION_CSV))
    parser.add_argument("--task-name-csv", default=str(DEFAULT_TASK_NAME_CSV))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmark" / "primitive_cluster" / "runs" / "one_episode_per_task_v0"),
    )
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    annotation_csv = Path(args.annotation_csv)
    task_name_csv = Path(args.task_name_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_meta_map = load_task_meta(annotation_csv)
    task_name_map = load_task_name_map(task_name_csv)
    task_ids = sorted(task_meta_map)
    if args.limit_tasks > 0:
        task_ids = task_ids[: args.limit_tasks]

    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        task_meta = task_meta_map[task_id]
        name_info = task_name_map.get(task_id, {})
        episode_id, episode_path, df, contact_events, raw_events = pick_representative_episode(
            dataset_root=dataset_root,
            task_id=task_id,
            task_meta=task_meta,
        )

        base_row = {
            "task_id": task_id,
            "task_name_raw": name_info.get("task_name_raw", ""),
            "task_name_readable": name_info.get("task_name_readable", ""),
            "metadata_status": name_info.get("metadata_status", ""),
            "arm_type": str(task_meta.get("arm_type", "")),
            "primary_arm": str(task_meta.get("primary_arm", "")),
            "has_gripper_motion": bool(task_meta.get("has_gripper_motion", False)),
        }

        if episode_path is None or episode_id is None:
            summary_rows.append(
                {
                    **base_row,
                    "episode_id": "",
                    "n_frames": 0,
                    "raw_contact_events_total": 0,
                    "raw_contact_events_left": 0,
                    "raw_contact_events_right": 0,
                    "anchor_event_count": 0,
                    "cluster_count": 0,
                    "task_serial_repetition_risk": infer_serial_repetition_risk(base_row["task_name_raw"]),
                    "cluster_sizes": "",
                    "cluster_spans": "",
                    "cluster_arm_patterns": "",
                    "episode_path_rel": "",
                }
            )
            continue

        assert df is not None
        assert contact_events is not None
        assert raw_events is not None
        anchors = build_anchor_events(task_id, episode_id, raw_events)
        velocity_signal = _select_velocity_signal(df, task_meta)
        serial_risk = infer_serial_repetition_risk(base_row["task_name_raw"])
        proposals, pair_decisions = build_cluster_proposals(
            task_id=task_id,
            episode_id=episode_id,
            anchors=anchors,
            velocity_signal=velocity_signal,
            task_serial_risk=serial_risk,
        )
        cluster_sizes, cluster_spans, cluster_arm_patterns = summarize_cluster_shapes(proposals)

        summary_rows.append(
            {
                **base_row,
                "episode_id": int(episode_id),
                "n_frames": int(len(df)),
                "raw_contact_events_total": len(raw_events),
                "raw_contact_events_left": len(contact_events.get("left", [])),
                "raw_contact_events_right": len(contact_events.get("right", [])),
                "anchor_event_count": len(anchors),
                "cluster_count": len(proposals),
                "task_serial_repetition_risk": serial_risk,
                "cluster_sizes": cluster_sizes,
                "cluster_spans": cluster_spans,
                "cluster_arm_patterns": cluster_arm_patterns,
                "episode_path_rel": str(episode_path.relative_to(REPO_ROOT)),
            }
        )

        for proposal in proposals:
            cluster_rows.append(
                {
                    **proposal,
                    "episode_path_rel": str(episode_path.relative_to(REPO_ROOT)),
                    "n_frames": int(len(df)),
                }
            )
        pair_rows.extend(pair_decisions)

    summary_rows.sort(key=lambda x: x["task_id"])
    cluster_rows.sort(key=lambda x: (x["task_id"], x["episode_id"], x["proposal_order"]))
    pair_rows.sort(key=lambda x: (x["task_id"], x["episode_id"], x["prev_anchor_event_id"]))

    summary_csv = output_dir / "task_cluster_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    cluster_jsonl = output_dir / "cluster_proposals.jsonl"
    with cluster_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pair_jsonl = output_dir / "pair_merge_decisions.jsonl"
    with pair_jsonl.open("w", encoding="utf-8") as f:
        for row in pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cluster_counts = [int(row["cluster_count"]) for row in summary_rows]
    manifest = {
        "dataset_root": str(dataset_root),
        "annotation_csv": str(annotation_csv),
        "task_name_csv": str(task_name_csv),
        "output_dir": str(output_dir),
        "n_tasks": len(summary_rows),
        "n_tasks_with_nonzero_clusters": int(sum(1 for x in cluster_counts if x > 0)),
        "mean_cluster_count_per_task_sample": float(np.mean(cluster_counts)) if cluster_counts else 0.0,
        "median_cluster_count_per_task_sample": float(np.median(cluster_counts)) if cluster_counts else 0.0,
        "max_cluster_count_per_task_sample": int(max(cluster_counts)) if cluster_counts else 0,
        "files": {
            "task_cluster_summary_csv": str(summary_csv),
            "cluster_proposals_jsonl": str(cluster_jsonl),
            "pair_merge_decisions_jsonl": str(pair_jsonl),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
