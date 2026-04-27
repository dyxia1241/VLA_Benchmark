from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segmentation import _select_velocity_signal, detect_contact_events


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


@dataclass(frozen=True)
class RawEvent:
    raw_event_id: str
    arm: str
    contact_row: int
    release_row: int


@dataclass(frozen=True)
class AnchorEvent:
    anchor_event_id: str
    anchor_order: int
    anchor_start_row: int
    anchor_end_row: int
    active_arm_pattern: str
    source_raw_event_ids: list[str]
    source_arms: list[str]


@dataclass(frozen=True)
class LocalStepInterval:
    interval_id: str
    interval_order: int
    start_row: int
    end_row: int
    active_arm_pattern: str
    source_anchor_event_ids: list[str]
    source_raw_event_ids: list[str]
    merge_confidence: str
    reason_codes: list[str]
    serial_repetition_risk: str

    @property
    def span(self) -> int:
        return int(self.end_row - self.start_row)


def load_task_name_map(dataset_root: Path, task_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for task_id in task_ids:
        meta_path = dataset_root / task_id / "meta" / "tasks.jsonl"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                first = next((line for line in fh if line.strip()), "")
            if not first:
                continue
            obj = json.loads(first)
            task_name = str(obj.get("task", "")).strip().lower()
            if task_name:
                out[task_id] = task_name
        except Exception:
            continue
    return out


def infer_serial_repetition_risk(task_name_raw: str) -> str:
    text = str(task_name_raw or "").strip().lower()
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


def build_raw_events(
    task_id: str,
    episode_id: int,
    contact_events: dict[str, list[dict[str, Any]]],
) -> list[RawEvent]:
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
                    contact_row=c,
                    release_row=r,
                )
            )
            event_idx += 1
    out.sort(key=lambda x: (x.contact_row, x.release_row, x.arm))
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
    current_group: list[RawEvent] = [raw_events[0]]
    current_end = raw_events[0].release_row

    def flush(group_events: list[RawEvent], order: int) -> AnchorEvent:
        arms = {e.arm for e in group_events}
        return AnchorEvent(
            anchor_event_id=f"{task_id}__{episode_id}__a{order:03d}",
            anchor_order=order,
            anchor_start_row=min(e.contact_row for e in group_events),
            anchor_end_row=max(e.release_row for e in group_events),
            active_arm_pattern=_active_arm_pattern(arms),
            source_raw_event_ids=[e.raw_event_id for e in group_events],
            source_arms=sorted(arms),
        )

    for raw_event in raw_events[1:]:
        if raw_event.contact_row <= current_end + ANCHOR_OVERLAP_MERGE_GAP:
            current_group.append(raw_event)
            current_end = max(current_end, raw_event.release_row)
            continue
        anchors.append(flush(current_group, len(anchors) + 1))
        current_group = [raw_event]
        current_end = raw_event.release_row

    anchors.append(flush(current_group, len(anchors) + 1))
    return anchors


def gap_bucket(gap_rows: int) -> str:
    if gap_rows <= IMMEDIATE_GAP_MAX:
        return "immediate"
    if gap_rows <= SHORT_GAP_MAX:
        return "short"
    if gap_rows <= MEDIUM_GAP_MAX:
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
    gap_rows = int(next_anchor.anchor_start_row - prev_anchor.anchor_end_row)
    same_active_arm_pattern = "yes" if prev_anchor.active_arm_pattern == next_anchor.active_arm_pattern else "no"
    inter_anchor_transition_strength = transition_strength(
        velocity_signal,
        prev_end=prev_anchor.anchor_end_row,
        next_start=next_anchor.anchor_start_row,
    )

    if gap_rows > SHORT_GAP_MAX:
        inter_anchor_reset_hint = "yes"
    elif gap_rows <= IMMEDIATE_GAP_MAX:
        inter_anchor_reset_hint = "no"
    else:
        inter_anchor_reset_hint = "unknown"

    explicit_retry_hint = "unknown"
    if gap_rows <= IMMEDIATE_GAP_MAX and same_active_arm_pattern == "yes" and inter_anchor_transition_strength == "low":
        explicit_retry_hint = "yes"
    elif gap_rows > SHORT_GAP_MAX or same_active_arm_pattern == "no":
        explicit_retry_hint = "no"

    serial_repetition_hint = "no"
    if task_serial_risk == "high" and gap_rows > 8:
        serial_repetition_hint = "yes"
    elif task_serial_risk == "medium" and gap_rows > IMMEDIATE_GAP_MAX:
        serial_repetition_hint = "yes"

    return {
        "gap_rows": gap_rows,
        "gap_bucket": gap_bucket(gap_rows),
        "same_active_arm_pattern": same_active_arm_pattern,
        "inter_anchor_reset_hint": inter_anchor_reset_hint,
        "explicit_retry_hint": explicit_retry_hint,
        "serial_repetition_hint": serial_repetition_hint,
        "inter_anchor_transition_strength": inter_anchor_transition_strength,
    }


def should_merge_pair(
    features: dict[str, Any],
    current_interval_anchor_count: int,
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
        and current_interval_anchor_count + 1 <= 3
    ):
        return True, "high", reason_codes

    if (
        features["gap_bucket"] == "immediate"
        and features["explicit_retry_hint"] != "no"
        and features["same_active_arm_pattern"] == "yes"
        and features["inter_anchor_reset_hint"] == "no"
        and features["serial_repetition_hint"] == "no"
        and features["inter_anchor_transition_strength"] == "low"
        and current_interval_anchor_count + 1 <= 2
    ):
        return True, "medium", reason_codes

    return False, "low", reason_codes or ["uncertain_pair_keep_split"]


def build_local_step_intervals_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict[str, Any],
    task_name_raw: str = "",
) -> list[LocalStepInterval]:
    if len(df) == 0:
        return []
    if not bool(task_meta.get("has_gripper_motion", False)):
        return []

    contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    raw_events = build_raw_events(task_id, episode_id, contact_events)
    anchors = build_anchor_events(task_id, episode_id, raw_events)
    if not anchors:
        return []

    velocity_signal = _select_velocity_signal(df, task_meta)
    serial_risk = infer_serial_repetition_risk(task_name_raw)

    intervals: list[LocalStepInterval] = []
    current = {
        "interval_id": f"{task_id}__{episode_id}__lsi001",
        "interval_order": 1,
        "start_row": anchors[0].anchor_start_row,
        "end_row": anchors[0].anchor_end_row,
        "active_arm_patterns": [anchors[0].active_arm_pattern],
        "source_anchor_event_ids": [anchors[0].anchor_event_id],
        "source_raw_event_ids": list(anchors[0].source_raw_event_ids),
        "merge_confidence": "low",
        "reason_codes": [],
    }
    last_anchor = anchors[0]

    def flush_interval(record: dict[str, Any]) -> LocalStepInterval:
        patterns = list(dict.fromkeys(record["active_arm_patterns"]))
        if patterns == ["left"]:
            active_arm_pattern = "left"
        elif patterns == ["right"]:
            active_arm_pattern = "right"
        elif set(patterns) == {"left", "right", "both"} or set(patterns) == {"left", "right"} or patterns == ["both"]:
            active_arm_pattern = "both"
        else:
            active_arm_pattern = patterns[0] if patterns else "unknown"

        return LocalStepInterval(
            interval_id=str(record["interval_id"]),
            interval_order=int(record["interval_order"]),
            start_row=int(record["start_row"]),
            end_row=int(record["end_row"]),
            active_arm_pattern=active_arm_pattern,
            source_anchor_event_ids=list(record["source_anchor_event_ids"]),
            source_raw_event_ids=list(record["source_raw_event_ids"]),
            merge_confidence=str(record["merge_confidence"]),
            reason_codes=sorted(set(str(x) for x in record["reason_codes"])),
            serial_repetition_risk=serial_risk,
        )

    for next_anchor in anchors[1:]:
        features = pairwise_features(last_anchor, next_anchor, velocity_signal, serial_risk)
        merge, confidence, reason_codes = should_merge_pair(
            features=features,
            current_interval_anchor_count=len(current["source_anchor_event_ids"]),
        )

        if merge:
            current["source_anchor_event_ids"].append(next_anchor.anchor_event_id)
            current["source_raw_event_ids"].extend(next_anchor.source_raw_event_ids)
            current["end_row"] = next_anchor.anchor_end_row
            current["active_arm_patterns"].append(next_anchor.active_arm_pattern)
            current["reason_codes"].extend(reason_codes)
            if confidence == "high":
                current["merge_confidence"] = "high"
            elif current["merge_confidence"] != "high":
                current["merge_confidence"] = "medium"
        else:
            intervals.append(flush_interval(current))
            current = {
                "interval_id": f"{task_id}__{episode_id}__lsi{len(intervals) + 1:03d}",
                "interval_order": len(intervals) + 1,
                "start_row": next_anchor.anchor_start_row,
                "end_row": next_anchor.anchor_end_row,
                "active_arm_patterns": [next_anchor.active_arm_pattern],
                "source_anchor_event_ids": [next_anchor.anchor_event_id],
                "source_raw_event_ids": list(next_anchor.source_raw_event_ids),
                "merge_confidence": "low",
                "reason_codes": [],
            }
        last_anchor = next_anchor

    intervals.append(flush_interval(current))
    return intervals
