from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segmentation import detect_contact_events, sampling_start_row


T3_CONTEXT_OFFSETS = (-10, -5, 0, 5)
T3_LABELS = ("left", "right", "top", "bottom")
T3_LETTERS = ("A", "B", "C", "D")
T3_LABEL_TO_TEXT = {
    "left": "toward the left side of the scene",
    "right": "toward the right side of the scene",
    "top": "toward the top of the scene",
    "bottom": "toward the bottom of the scene",
}
ROBOT_RAW_DIRECTIONS = ("+x", "-x", "+y", "-y")
DEFAULT_PROVISIONAL_MAPPING_NAME = "noswap_x_left_y_down"


@dataclass(frozen=True)
class PlanarMapping:
    name: str
    horizontal_source: str
    horizontal_sign: int
    vertical_source: str
    vertical_sign: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizontal_source": self.horizontal_source,
            "horizontal_sign": int(self.horizontal_sign),
            "vertical_source": self.vertical_source,
            "vertical_sign": int(self.vertical_sign),
        }


@dataclass(frozen=True)
class LinearPlanarMapping:
    name: str
    matrix: tuple[tuple[float, float], tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "name": self.name,
            "matrix": [
                [float(self.matrix[0][0]), float(self.matrix[0][1])],
                [float(self.matrix[1][0]), float(self.matrix[1][1])],
            ],
        }


@dataclass(frozen=True)
class T3MappingConfig:
    mapping_kind: str
    mapping_scope: str
    mappings: dict[str, Any]
    mapping_source: str
    exclude_robot_direction_raw_pairs: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class ArmWindowCandidate:
    arm: str
    valid: bool
    query_net_xy: tuple[float, float]
    query_segments_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    robot_direction_raw: str | None
    net_norm: float
    dominant_component_abs: float
    second_component_abs: float
    planar_purity_ratio: float
    consistency_support: int
    contradictory_segments: int
    score: float
    major_axis: str | None
    major_sign: int


@dataclass(frozen=True)
class T3WindowCandidate:
    task_id: str
    episode_id: int
    frame_index: int
    frame_indices: tuple[int, int, int, int]
    frame_offsets: tuple[int, int, int, int]
    camera: str
    arm_type: str
    primary_arm: str
    query_arm: str
    query_arm_mode: str
    task_meta: dict[str, Any]
    robot_direction_raw: str
    robot_plane_displacement_xy: tuple[float, float]
    robot_plane_segments_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    query_net_norm: float
    other_arm_net_norm: float
    dominant_component_abs: float
    second_component_abs: float
    planar_purity_ratio: float
    consistency_support: int
    contradictory_segments: int
    candidate_score: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_id": int(self.episode_id),
            "frame_index": int(self.frame_index),
            "frame_indices": [int(x) for x in self.frame_indices],
            "frame_offsets": [int(x) for x in self.frame_offsets],
            "camera": self.camera,
            "task_type": "T3",
            "arm_type": self.arm_type,
            "primary_arm": self.primary_arm,
            "query_arm": self.query_arm,
            "query_arm_mode": self.query_arm_mode,
            "task_meta": self.task_meta,
            "robot_direction_raw": self.robot_direction_raw,
            "robot_plane_displacement_xy": [float(x) for x in self.robot_plane_displacement_xy],
            "robot_plane_segments_xy": [[float(v) for v in seg] for seg in self.robot_plane_segments_xy],
            "query_net_norm": float(self.query_net_norm),
            "other_arm_net_norm": float(self.other_arm_net_norm),
            "dominant_component_abs": float(self.dominant_component_abs),
            "second_component_abs": float(self.second_component_abs),
            "planar_purity_ratio": float(self.planar_purity_ratio),
            "consistency_support": int(self.consistency_support),
            "contradictory_segments": int(self.contradictory_segments),
            "candidate_score": float(self.candidate_score),
        }


def parse_offsets_csv(text: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("offset list is empty")
    if 0 not in vals:
        raise ValueError("offset list must include 0")
    if len(vals) != 4:
        raise ValueError("T3 currently expects exactly 4 offsets")
    if sorted(vals) != list(vals):
        raise ValueError("offset list must be sorted from earliest to latest")
    return tuple(vals)


def mapping_candidates() -> list[PlanarMapping]:
    out: list[PlanarMapping] = []
    for swap_axes in (False, True):
        horizontal_source = "y" if swap_axes else "x"
        vertical_source = "x" if swap_axes else "y"
        for horizontal_sign in (-1, 1):
            for vertical_sign in (-1, 1):
                h_name = "right" if horizontal_sign > 0 else "left"
                v_name = "down" if vertical_sign > 0 else "up"
                name = (
                    f"{'swap' if swap_axes else 'noswap'}_"
                    f"{horizontal_source}_{h_name}_"
                    f"{vertical_source}_{v_name}"
                )
                out.append(
                    PlanarMapping(
                        name=name,
                        horizontal_source=horizontal_source,
                        horizontal_sign=horizontal_sign,
                        vertical_source=vertical_source,
                        vertical_sign=vertical_sign,
                    )
                )
    return out


_MAPPING_BY_NAME = {m.name: m for m in mapping_candidates()}


def mapping_from_dict(obj: dict[str, Any]) -> PlanarMapping | LinearPlanarMapping:
    if str(obj.get("kind", "")).strip().lower() == "linear" or "matrix" in obj:
        matrix = obj.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in matrix):
            raise ValueError(f"Invalid linear mapping matrix: {obj}")
        return LinearPlanarMapping(
            name=str(obj.get("name", "linear_mapping")).strip() or "linear_mapping",
            matrix=(
                (float(matrix[0][0]), float(matrix[0][1])),
                (float(matrix[1][0]), float(matrix[1][1])),
            ),
        )

    name = str(obj.get("name", "")).strip()
    if name and name in _MAPPING_BY_NAME:
        return _MAPPING_BY_NAME[name]
    horizontal_source = str(obj["horizontal_source"]).strip()
    vertical_source = str(obj["vertical_source"]).strip()
    horizontal_sign = int(obj["horizontal_sign"])
    vertical_sign = int(obj["vertical_sign"])
    return PlanarMapping(
        name=name or f"custom_{horizontal_source}_{horizontal_sign}_{vertical_source}_{vertical_sign}",
        horizontal_source=horizontal_source,
        horizontal_sign=horizontal_sign,
        vertical_source=vertical_source,
        vertical_sign=vertical_sign,
    )


def load_mapping_config(path: str | Path | None) -> T3MappingConfig:
    if path is None:
        provisional = _MAPPING_BY_NAME[DEFAULT_PROVISIONAL_MAPPING_NAME]
        return T3MappingConfig(
            mapping_kind="signed_axis",
            mapping_scope="global",
            mappings={"default": provisional},
            mapping_source="provisional_default",
            exclude_robot_direction_raw_pairs=frozenset(),
        )

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scope = str(raw.get("mapping_scope", "global")).strip().lower()
    mapping_kind = str(raw.get("mapping_kind", "")).strip().lower()
    mappings = raw.get("mappings", {})
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError(f"Invalid mapping config: {path}")
    exclude_raw = raw.get("exclude_robot_direction_raw_pairs", [])
    exclude_pairs: set[tuple[str, str]] = set()
    if isinstance(exclude_raw, list):
        for item in exclude_raw:
            if not isinstance(item, dict):
                continue
            query_arm = str(item.get("query_arm", "")).strip()
            robot_direction_raw = str(item.get("robot_direction_raw", "")).strip()
            if query_arm and robot_direction_raw:
                exclude_pairs.add((query_arm, robot_direction_raw))

    if scope == "global":
        if "global" not in mappings:
            raise ValueError(f"Global mapping config missing 'global' entry: {path}")
        parsed = {"default": mapping_from_dict(mappings["global"])}
        resolved_kind = mapping_kind or ("linear" if isinstance(parsed["default"], LinearPlanarMapping) else "signed_axis")
        return T3MappingConfig(
            mapping_kind=resolved_kind,
            mapping_scope="global",
            mappings=parsed,
            mapping_source=str(raw.get("selected_from", "config")).strip() or "config",
            exclude_robot_direction_raw_pairs=frozenset(exclude_pairs),
        )

    if scope == "per_arm":
        missing = [k for k in ("left", "right") if k not in mappings]
        if missing:
            raise ValueError(f"Per-arm mapping config missing keys {missing}: {path}")
        parsed = {
            "left": mapping_from_dict(mappings["left"]),
            "right": mapping_from_dict(mappings["right"]),
        }
        first = next(iter(parsed.values()))
        resolved_kind = mapping_kind or ("linear" if isinstance(first, LinearPlanarMapping) else "signed_axis")
        return T3MappingConfig(
            mapping_kind=resolved_kind,
            mapping_scope="per_arm",
            mappings=parsed,
            mapping_source=str(raw.get("selected_from", "config")).strip() or "config",
            exclude_robot_direction_raw_pairs=frozenset(exclude_pairs),
        )

    raise ValueError(f"Unsupported mapping_scope={scope!r} in {path}")


def mapping_for_arm(mapping_config: T3MappingConfig, query_arm: str) -> PlanarMapping | LinearPlanarMapping:
    if query_arm in mapping_config.mappings:
        return mapping_config.mappings[query_arm]
    if "default" in mapping_config.mappings:
        return mapping_config.mappings["default"]
    raise KeyError(f"No mapping found for query_arm={query_arm!r}")


def apply_mapping_to_xy(
    delta_xy: tuple[float, float],
    mapping: PlanarMapping | LinearPlanarMapping,
) -> tuple[float, float]:
    if isinstance(mapping, LinearPlanarMapping):
        vec = np.asarray([float(delta_xy[0]), float(delta_xy[1])], dtype=float) @ np.asarray(mapping.matrix, dtype=float)
        return float(vec[0]), float(vec[1])

    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    source = {"x": dx, "y": dy}
    image_dx = float(mapping.horizontal_sign) * float(source[mapping.horizontal_source])
    image_dy = float(mapping.vertical_sign) * float(source[mapping.vertical_source])
    return image_dx, image_dy


def planar_label_from_image_delta(
    image_delta_xy: tuple[float, float],
    dominant_min: float,
    purity_ratio: float,
) -> tuple[str | None, float, float, float]:
    image_dx, image_dy = float(image_delta_xy[0]), float(image_delta_xy[1])
    abs_h = abs(image_dx)
    abs_v = abs(image_dy)
    dominant = max(abs_h, abs_v)
    second = min(abs_h, abs_v)
    ratio = float(dominant / (second + 1e-9))
    if dominant < dominant_min:
        return None, dominant, second, ratio
    if abs_h >= purity_ratio * second:
        return ("right" if image_dx >= 0 else "left"), dominant, second, ratio
    if abs_v >= purity_ratio * second:
        return ("bottom" if image_dy >= 0 else "top"), dominant, second, ratio
    return None, dominant, second, ratio


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
            "observation.state.arm.position",
            "observation.state.effector.effort",
        ],
    )


def arm_xy_from_state_position(state_arm: np.ndarray, arm: str) -> np.ndarray:
    if arm == "left":
        return state_arm[:, :2]
    if arm == "right":
        return state_arm[:, 6:8]
    raise ValueError(f"Unsupported arm={arm!r}")


def robot_raw_direction(delta_xy: tuple[float, float]) -> tuple[str, float, float]:
    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    if abs(dx) >= abs(dy):
        return ("+x" if dx >= 0 else "-x"), abs(dx), abs(dy)
    return ("+y" if dy >= 0 else "-y"), abs(dy), abs(dx)


def evaluate_arm_window(
    arm_xy: np.ndarray,
    rows: tuple[int, int, int, int],
    net_min: float,
    dominant_min: float,
    purity_ratio: float,
    segment_support_fraction: float,
    segment_support_min: float,
    min_consistent_segments: int,
    contradictory_segment_tol: float,
) -> ArmWindowCandidate:
    pts = arm_xy[list(rows)]
    segments = np.diff(pts, axis=0)
    net = pts[-1] - pts[0]
    net_dx = float(net[0])
    net_dy = float(net[1])
    net_norm = float(np.linalg.norm(net))
    raw_dir, dominant, second = robot_raw_direction((net_dx, net_dy))
    purity = float(dominant / (second + 1e-9))
    major_axis_index = 0 if raw_dir.endswith("x") else 1
    major_axis = "x" if major_axis_index == 0 else "y"
    major_sign = 1 if raw_dir.startswith("+") else -1

    if net_norm < net_min or dominant < dominant_min or purity < purity_ratio:
        return ArmWindowCandidate(
            arm="",
            valid=False,
            query_net_xy=(net_dx, net_dy),
            query_segments_xy=tuple((float(seg[0]), float(seg[1])) for seg in segments),
            robot_direction_raw=raw_dir,
            net_norm=net_norm,
            dominant_component_abs=dominant,
            second_component_abs=second,
            planar_purity_ratio=purity,
            consistency_support=0,
            contradictory_segments=0,
            score=0.0,
            major_axis=major_axis,
            major_sign=major_sign,
        )

    required_segment_major = max(segment_support_min, segment_support_fraction * dominant)
    segment_major = segments[:, major_axis_index]
    segment_minor = np.abs(segments[:, 1 - major_axis_index])
    segment_major_abs = np.abs(segment_major)
    strong_major = segment_major_abs >= required_segment_major
    sign_consistent = np.sign(segment_major + 1e-12) == np.sign(float(major_sign))
    segment_pure = segment_major_abs >= purity_ratio * (segment_minor + 1e-9)
    support_mask = strong_major & sign_consistent & segment_pure
    contradictory_mask = (segment_major_abs >= contradictory_segment_tol) & (~sign_consistent)
    support = int(np.sum(support_mask))
    contradictory = int(np.sum(contradictory_mask))
    valid = support >= int(min_consistent_segments) and contradictory == 0
    score = float(dominant * purity * max(1, support)) if valid else 0.0

    return ArmWindowCandidate(
        arm="",
        valid=valid,
        query_net_xy=(net_dx, net_dy),
        query_segments_xy=tuple((float(seg[0]), float(seg[1])) for seg in segments),
        robot_direction_raw=raw_dir,
        net_norm=net_norm,
        dominant_component_abs=dominant,
        second_component_abs=second,
        planar_purity_ratio=purity,
        consistency_support=support,
        contradictory_segments=contradictory,
        score=score,
        major_axis=major_axis,
        major_sign=major_sign,
    )


def with_arm(candidate: ArmWindowCandidate, arm: str) -> ArmWindowCandidate:
    return ArmWindowCandidate(
        arm=arm,
        valid=bool(candidate.valid),
        query_net_xy=tuple(float(x) for x in candidate.query_net_xy),
        query_segments_xy=tuple(tuple(float(v) for v in seg) for seg in candidate.query_segments_xy),
        robot_direction_raw=candidate.robot_direction_raw,
        net_norm=float(candidate.net_norm),
        dominant_component_abs=float(candidate.dominant_component_abs),
        second_component_abs=float(candidate.second_component_abs),
        planar_purity_ratio=float(candidate.planar_purity_ratio),
        consistency_support=int(candidate.consistency_support),
        contradictory_segments=int(candidate.contradictory_segments),
        score=float(candidate.score),
        major_axis=candidate.major_axis,
        major_sign=int(candidate.major_sign),
    )


def pick_query_arm(
    primary_arm: str,
    left_candidate: ArmWindowCandidate,
    right_candidate: ArmWindowCandidate,
    secondary_arm_ratio_max: float,
    secondary_arm_abs_max: float,
    bimanual_dominance_ratio: float,
) -> tuple[str | None, str | None, float]:
    candidates = {"left": left_candidate, "right": right_candidate}

    if primary_arm in {"left", "right"}:
        chosen = candidates[primary_arm]
        other = candidates["right" if primary_arm == "left" else "left"]
        if not chosen.valid:
            return None, None, 0.0
        other_limit = max(secondary_arm_abs_max, chosen.net_norm * secondary_arm_ratio_max)
        if other.net_norm > other_limit:
            return None, None, 0.0
        return primary_arm, "fixed_primary", other.net_norm

    valid = [cand for cand in (left_candidate, right_candidate) if cand.valid]
    if not valid:
        return None, None, 0.0
    if len(valid) == 1:
        chosen = valid[0]
        other = right_candidate if chosen.arm == "left" else left_candidate
        other_limit = max(secondary_arm_abs_max, chosen.net_norm * secondary_arm_ratio_max)
        if other.net_norm > other_limit:
            return None, None, 0.0
        return chosen.arm, "dominant_window", other.net_norm

    ordered = sorted(valid, key=lambda x: (-x.score, -x.net_norm, x.arm))
    chosen = ordered[0]
    other = ordered[1]
    if chosen.net_norm < other.net_norm * bimanual_dominance_ratio:
        return None, None, 0.0
    other_limit = max(secondary_arm_abs_max, chosen.net_norm * secondary_arm_ratio_max)
    if other.net_norm > other_limit:
        return None, None, 0.0
    return chosen.arm, "dominant_window", other.net_norm


def build_t3_candidate_windows_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict[str, Any],
    camera: str,
    offsets: tuple[int, int, int, int],
    net_min: float,
    dominant_min: float,
    purity_ratio: float,
    segment_support_fraction: float,
    segment_support_min: float,
    min_consistent_segments: int,
    contradictory_segment_tol: float,
    secondary_arm_ratio_max: float,
    secondary_arm_abs_max: float,
    bimanual_dominance_ratio: float,
    approach_buffer_frames: int,
    no_contact_start_frame: int,
) -> list[T3WindowCandidate]:
    state_arm = np.vstack(df["observation.state.arm.position"].to_numpy())
    if len(state_arm) == 0:
        return []

    contact_events = detect_contact_events(df, task_meta=task_meta, min_persist_frames=5)
    start_row = sampling_start_row(
        n_rows=len(df),
        contact_events=contact_events,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )

    if "frame_index" in df.columns:
        frame_indices = df["frame_index"].to_numpy(dtype=int)
    else:
        frame_indices = np.arange(len(df), dtype=int)

    left_xy = arm_xy_from_state_position(state_arm, arm="left")
    right_xy = arm_xy_from_state_position(state_arm, arm="right")
    primary_arm = str(task_meta.get("primary_arm", "none"))
    context_pre = max(0, -min(offsets))
    context_post = max(0, max(offsets))

    out: list[T3WindowCandidate] = []
    for ridx in range(max(start_row, context_pre), len(df) - context_post):
        rows = tuple(int(ridx + off) for off in offsets)
        left_candidate = with_arm(
            evaluate_arm_window(
                arm_xy=left_xy,
                rows=rows,
                net_min=net_min,
                dominant_min=dominant_min,
                purity_ratio=purity_ratio,
                segment_support_fraction=segment_support_fraction,
                segment_support_min=segment_support_min,
                min_consistent_segments=min_consistent_segments,
                contradictory_segment_tol=contradictory_segment_tol,
            ),
            arm="left",
        )
        right_candidate = with_arm(
            evaluate_arm_window(
                arm_xy=right_xy,
                rows=rows,
                net_min=net_min,
                dominant_min=dominant_min,
                purity_ratio=purity_ratio,
                segment_support_fraction=segment_support_fraction,
                segment_support_min=segment_support_min,
                min_consistent_segments=min_consistent_segments,
                contradictory_segment_tol=contradictory_segment_tol,
            ),
            arm="right",
        )

        query_arm, query_arm_mode, other_net_norm = pick_query_arm(
            primary_arm=primary_arm,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            secondary_arm_ratio_max=secondary_arm_ratio_max,
            secondary_arm_abs_max=secondary_arm_abs_max,
            bimanual_dominance_ratio=bimanual_dominance_ratio,
        )
        if query_arm is None or query_arm_mode is None:
            continue

        chosen = left_candidate if query_arm == "left" else right_candidate
        out.append(
            T3WindowCandidate(
                task_id=task_id,
                episode_id=int(episode_id),
                frame_index=int(frame_indices[ridx]),
                frame_indices=tuple(int(frame_indices[x]) for x in rows),
                frame_offsets=offsets,
                camera=camera,
                arm_type=str(task_meta["arm_type"]),
                primary_arm=primary_arm,
                query_arm=query_arm,
                query_arm_mode=query_arm_mode,
                task_meta={
                    "primary_arm": primary_arm,
                    "arm_type": str(task_meta["arm_type"]),
                },
                robot_direction_raw=str(chosen.robot_direction_raw),
                robot_plane_displacement_xy=tuple(float(x) for x in chosen.query_net_xy),
                robot_plane_segments_xy=tuple(tuple(float(v) for v in seg) for seg in chosen.query_segments_xy),
                query_net_norm=float(chosen.net_norm),
                other_arm_net_norm=float(other_net_norm),
                dominant_component_abs=float(chosen.dominant_component_abs),
                second_component_abs=float(chosen.second_component_abs),
                planar_purity_ratio=float(chosen.planar_purity_ratio),
                consistency_support=int(chosen.consistency_support),
                contradictory_segments=int(chosen.contradictory_segments),
                candidate_score=float(chosen.score),
            )
        )
    return out


def build_t3_choices(
    correct_direction: str,
    rng: random.Random,
    preferred_answer_letter: str | None = None,
) -> tuple[dict[str, str], str]:
    distractor_pool = [x for x in T3_LABELS if x != correct_direction]
    wrong = rng.sample(distractor_pool, 3)

    if preferred_answer_letter in T3_LETTERS:
        letter_to_label: dict[str, str] = {preferred_answer_letter: correct_direction}
        other_letters = [x for x in T3_LETTERS if x != preferred_answer_letter]
        rng.shuffle(wrong)
        for key, label in zip(other_letters, wrong):
            letter_to_label[key] = label
        answer = preferred_answer_letter
    else:
        options = [correct_direction] + wrong
        rng.shuffle(options)
        letter_to_label = {k: v for k, v in zip(T3_LETTERS, options)}
        answer = T3_LETTERS[options.index(correct_direction)]

    choices = {key: T3_LABEL_TO_TEXT[label] for key, label in letter_to_label.items()}
    return choices, answer


def select_candidates_by_bucket(
    candidates: list[T3WindowCandidate],
    max_per_bucket: int,
    rng: random.Random,
) -> list[T3WindowCandidate]:
    buckets: dict[str, list[T3WindowCandidate]] = {key: [] for key in ROBOT_RAW_DIRECTIONS}
    for cand in candidates:
        if cand.robot_direction_raw in buckets:
            buckets[cand.robot_direction_raw].append(cand)

    selected: list[T3WindowCandidate] = []
    for key in ROBOT_RAW_DIRECTIONS:
        bucket = sorted(buckets[key], key=lambda x: (-x.candidate_score, x.frame_index, x.query_arm))
        if len(bucket) > max_per_bucket:
            top = bucket[: max_per_bucket * 3]
            bucket = sorted(rng.sample(top, max_per_bucket), key=lambda x: x.frame_index)
        selected.extend(bucket[:max_per_bucket])
    return sorted(selected, key=lambda x: (x.episode_id, x.frame_index, x.query_arm))


def build_t3_items_from_candidates(
    candidates: list[T3WindowCandidate],
    mapping_config: T3MappingConfig,
    rng: random.Random,
    answer_counts: dict[str, int] | None = None,
    balance_answer_letters: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if (cand.query_arm, cand.robot_direction_raw) in mapping_config.exclude_robot_direction_raw_pairs:
            continue
        mapping = mapping_for_arm(mapping_config, cand.query_arm)
        image_delta_xy = apply_mapping_to_xy(cand.robot_plane_displacement_xy, mapping)
        direction, dominant, second, ratio = planar_label_from_image_delta(
            image_delta_xy=image_delta_xy,
            dominant_min=1e-9,
            purity_ratio=2.0,
        )
        if direction is None:
            continue

        preferred_answer_letter = None
        if balance_answer_letters and answer_counts is not None:
            min_count = min(answer_counts.values())
            least_used = [key for key in T3_LETTERS if answer_counts[key] == min_count]
            preferred_answer_letter = rng.choice(least_used)
        choices, answer = build_t3_choices(
            correct_direction=direction,
            rng=rng,
            preferred_answer_letter=preferred_answer_letter,
        )
        if answer_counts is not None:
            answer_counts[answer] += 1

        out.append(
            {
                "task_id": cand.task_id,
                "episode_id": int(cand.episode_id),
                "frame_index": int(cand.frame_index),
                "frame_indices": [int(x) for x in cand.frame_indices],
                "frame_offsets": [int(x) for x in cand.frame_offsets],
                "camera": cand.camera,
                "question": f"In which direction does the {cand.query_arm} robot arm primarily move in the scene?",
                "choices": choices,
                "answer": answer,
                "task_type": "T3",
                "arm_type": cand.arm_type,
                "primary_arm": cand.primary_arm,
                "query_arm": cand.query_arm,
                "query_arm_mode": cand.query_arm_mode,
                "motion_direction": T3_LABEL_TO_TEXT[direction],
                "motion_direction_raw": direction,
                "robot_direction_raw": cand.robot_direction_raw,
                "robot_plane_displacement_xy": [float(x) for x in cand.robot_plane_displacement_xy],
                "robot_plane_segments_xy": [[float(v) for v in seg] for seg in cand.robot_plane_segments_xy],
                "image_plane_displacement_xy": [float(x) for x in image_delta_xy],
                "delta_norm": float(cand.query_net_norm),
                "dominant_component_abs": float(dominant),
                "second_component_abs": float(second),
                "direction_purity_ratio": float(ratio),
                "planar_purity_ratio_robot": float(cand.planar_purity_ratio),
                "consistency_support": int(cand.consistency_support),
                "other_arm_net_norm": float(cand.other_arm_net_norm),
                "mapping_name": mapping.name,
                "mapping_kind": mapping_config.mapping_kind,
                "mapping_source": mapping_config.mapping_source,
            }
        )
    return out


def build_t3_items_for_episode(
    task_id: str,
    episode_id: int,
    df: pd.DataFrame,
    task_meta: dict[str, Any],
    camera: str,
    max_per_dir: int,
    net_min: float,
    dominant_min: float,
    purity_ratio: float,
    segment_support_fraction: float,
    segment_support_min: float,
    min_consistent_segments: int,
    contradictory_segment_tol: float,
    secondary_arm_ratio_max: float,
    secondary_arm_abs_max: float,
    bimanual_dominance_ratio: float,
    offsets: tuple[int, int, int, int],
    mapping_config: T3MappingConfig,
    rng: random.Random,
    answer_counts: dict[str, int] | None = None,
    balance_answer_letters: bool = True,
    approach_buffer_frames: int = 30,
    no_contact_start_frame: int = 20,
) -> list[dict[str, Any]]:
    candidates = build_t3_candidate_windows_for_episode(
        task_id=task_id,
        episode_id=episode_id,
        df=df,
        task_meta=task_meta,
        camera=camera,
        offsets=offsets,
        net_min=net_min,
        dominant_min=dominant_min,
        purity_ratio=purity_ratio,
        segment_support_fraction=segment_support_fraction,
        segment_support_min=segment_support_min,
        min_consistent_segments=min_consistent_segments,
        contradictory_segment_tol=contradictory_segment_tol,
        secondary_arm_ratio_max=secondary_arm_ratio_max,
        secondary_arm_abs_max=secondary_arm_abs_max,
        bimanual_dominance_ratio=bimanual_dominance_ratio,
        approach_buffer_frames=approach_buffer_frames,
        no_contact_start_frame=no_contact_start_frame,
    )
    selected = select_candidates_by_bucket(candidates, max_per_bucket=max_per_dir, rng=rng)
    return build_t3_items_from_candidates(
        candidates=selected,
        mapping_config=mapping_config,
        rng=rng,
        answer_counts=answer_counts,
        balance_answer_letters=balance_answer_letters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T3 v2 motion-direction GT from clip-consistent planar state positions.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-jsonl", default="/data/projects/GM-100/t3_gt_items.jsonl")
    parser.add_argument("--output", dest="output_jsonl", help="Alias of --output-jsonl.")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--max-per-dir", type=int, default=5)
    parser.add_argument("--net-min", type=float, default=0.02)
    parser.add_argument("--dominant-min", type=float, default=0.015)
    parser.add_argument("--purity-ratio", type=float, default=2.0)
    parser.add_argument("--segment-support-fraction", type=float, default=0.18)
    parser.add_argument("--segment-support-min", type=float, default=0.003)
    parser.add_argument("--min-consistent-segments", type=int, default=2)
    parser.add_argument("--contradictory-segment-tol", type=float, default=0.004)
    parser.add_argument("--secondary-arm-ratio-max", type=float, default=0.45)
    parser.add_argument("--secondary-arm-abs-max", type=float, default=0.012)
    parser.add_argument("--bimanual-dominance-ratio", type=float, default=1.75)
    parser.add_argument("--context-offsets", default=",".join(str(x) for x in T3_CONTEXT_OFFSETS))
    parser.add_argument("--mapping-config", default="", help="Optional JSON config chosen from human calibration. If empty, uses provisional default mapping.")
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks-csv", default="", help="Comma-separated task list; empty means all tasks from annotation.")
    parser.add_argument("--tasks", dest="tasks_csv", help="Alias of --tasks-csv.")
    parser.add_argument(
        "--no-balance-answer-letters",
        action="store_true",
        help="Disable answer-letter balancing (A/B/C/D) for T3 options.",
    )
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks in annotation csv")
    parser.add_argument("--limit-episodes-per-task", type=int, default=0, help="0 means all episodes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root)
    ann = pd.read_csv(args.annotation_csv)
    if args.tasks_csv.strip():
        tasks = [x.strip() for x in args.tasks_csv.split(",") if x.strip()]
    else:
        tasks = ann["task_id"].tolist()
        if args.limit_tasks > 0:
            tasks = tasks[: args.limit_tasks]
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}
    offsets = parse_offsets_csv(args.context_offsets)
    mapping_config = load_mapping_config(args.mapping_config.strip() or None)

    items: list[dict[str, Any]] = []
    answer_letter_counts: dict[str, int] = {key: 0 for key in T3_LETTERS}
    for task_id in tasks:
        if task_id not in meta_map:
            continue
        eps = task_episode_paths(dataset_root, task_id)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for p in eps:
            episode_id = int(p.stem.split("_")[-1])
            df = load_episode_df(p)
            epi_items = build_t3_items_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                max_per_dir=args.max_per_dir,
                net_min=args.net_min,
                dominant_min=args.dominant_min,
                purity_ratio=args.purity_ratio,
                segment_support_fraction=args.segment_support_fraction,
                segment_support_min=args.segment_support_min,
                min_consistent_segments=args.min_consistent_segments,
                contradictory_segment_tol=args.contradictory_segment_tol,
                secondary_arm_ratio_max=args.secondary_arm_ratio_max,
                secondary_arm_abs_max=args.secondary_arm_abs_max,
                bimanual_dominance_ratio=args.bimanual_dominance_ratio,
                offsets=offsets,
                mapping_config=mapping_config,
                rng=rng,
                answer_counts=answer_letter_counts,
                balance_answer_letters=bool(not args.no_balance_answer_letters),
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            items.extend(epi_items)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    label_counts = {label: int(sum(1 for x in items if x["motion_direction_raw"] == label)) for label in T3_LABELS}
    query_arm_counts = {arm: int(sum(1 for x in items if x["query_arm"] == arm)) for arm in ("left", "right")}
    mapping_names = sorted({str(x.get("mapping_name", "")) for x in items})

    print(
        json.dumps(
            {
                "output_jsonl": str(out_path),
                "num_items": len(items),
                "num_tasks": len(tasks),
                "camera": args.camera,
                "context_offsets": list(offsets),
                "max_per_dir": int(args.max_per_dir),
                "net_min": float(args.net_min),
                "dominant_min": float(args.dominant_min),
                "purity_ratio": float(args.purity_ratio),
                "segment_support_fraction": float(args.segment_support_fraction),
                "segment_support_min": float(args.segment_support_min),
                "min_consistent_segments": int(args.min_consistent_segments),
                "secondary_arm_ratio_max": float(args.secondary_arm_ratio_max),
                "secondary_arm_abs_max": float(args.secondary_arm_abs_max),
                "bimanual_dominance_ratio": float(args.bimanual_dominance_ratio),
                "balance_answer_letters": bool(not args.no_balance_answer_letters),
                "label_counts": label_counts,
                "query_arm_counts": query_arm_counts,
                "answer_letter_counts": answer_letter_counts,
                "mapping_names": mapping_names,
                "mapping_config": str(args.mapping_config or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
