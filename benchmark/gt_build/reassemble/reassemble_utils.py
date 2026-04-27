from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REASSEMBLE_ROOT = REPO_ROOT / "reassemble-tuwien-researchdata"
DERIVED_DIR = REPO_ROOT / "benchmark" / "manual_audit" / "semantic_affordance_audit" / "derived"
RECORDING_INDEX_JSONL = DERIVED_DIR / "reassemble_recording_index_v1.jsonl"
VOCAB_JSON = DERIVED_DIR / "reassemble_vocab_v1.json"
DEFAULT_CAMERA = "hand"
SHORT_CONTEXT_OFFSETS = (-6, -3, 0, 3)
LONG_CONTEXT_OFFSETS = (-6, -3, 0, 3, 6)
PHASE_NAMES = (
    "pre-approach",
    "approach",
    "contact",
    "hold and carry",
    "transfer",
    "release",
)
LOW_LEVEL_VOCAB_FALLBACK = (
    "Approach",
    "Align",
    "Grasp",
    "Lift",
    "Pull",
    "Push",
    "Release",
    "Twist",
    "Nudge",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").strip().split())


def load_recording_index(path: Path = RECORDING_INDEX_JSONL) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_low_level_vocab(path: Path = VOCAB_JSON) -> list[str]:
    if not path.exists():
        return list(LOW_LEVEL_VOCAB_FALLBACK)
    obj = json.loads(path.read_text(encoding="utf-8"))
    vocab = obj.get("low_level_vocab", {})
    if not isinstance(vocab, dict) or not vocab:
        return list(LOW_LEVEL_VOCAB_FALLBACK)
    return list(vocab.keys())


def recording_h5_path(dataset_root: Path, recording_id: str) -> Path:
    return dataset_root / "data" / f"{recording_id}.h5"


def filter_recordings(
    rows: list[dict[str, Any]],
    *,
    split: str,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if normalize_text(row.get("split")) != split:
            continue
        rid = normalize_text(row.get("recording_id"))
        if not rid:
            continue
        if not recording_h5_path(dataset_root, rid).exists():
            continue
        out.append(row)
    return sorted(out, key=lambda item: normalize_text(item.get("recording_id")))


def is_no_action(text: str) -> bool:
    return normalize_text(text).rstrip(".").lower() == "no action"


def high_level_text(segment: dict[str, Any]) -> str:
    return normalize_text(segment.get("text", "")).rstrip(".")


def high_level_verb(text: str) -> str:
    text = normalize_text(text).rstrip(".")
    if not text:
        return ""
    return text.split(maxsplit=1)[0].lower()


def object_from_high_level_text(text: str) -> str:
    text = normalize_text(text).rstrip(".")
    if not text or is_no_action(text):
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def non_no_action_segments(recording_row: dict[str, Any]) -> list[dict[str, Any]]:
    return [seg for seg in recording_row.get("segments", []) if not is_no_action(high_level_text(seg))]


def low_level_segments(high_segment: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in high_segment.get("low_level", []):
        text = normalize_text(item.get("text", "")).rstrip(".")
        if not text:
            continue
        out.append(
            {
                "low_index": int(item.get("low_index", 0)),
                "text": text,
                "success": bool(item.get("success", False)),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
            }
        )
    return out


def distinct_low_level_chain(high_segment: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_text = ""
    for item in low_level_segments(high_segment):
        text = item["text"]
        if text == last_text:
            continue
        out.append(item)
        last_text = text
    return out


def next_distinct_low_level(high_segment: dict[str, Any], low_pos: int) -> dict[str, Any] | None:
    lows = low_level_segments(high_segment)
    if low_pos < 0 or low_pos >= len(lows):
        return None
    cur = lows[low_pos]["text"]
    for nxt in lows[low_pos + 1 :]:
        if nxt["text"] != cur:
            return nxt
    return None


def camera_timestamps(h5_file: h5py.File, camera: str = DEFAULT_CAMERA) -> np.ndarray:
    return np.asarray(h5_file[f"timestamps/{camera}"][:], dtype=np.float64)


def velocity_stream(h5_file: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(h5_file["timestamps/velocity"][:], dtype=np.float64)
    vel = np.asarray(h5_file["robot_state/velocity"][:, :3], dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    return ts, speed


def timestamp_interval_to_frame_range(
    timestamps: np.ndarray,
    start_time: float,
    end_time: float,
) -> tuple[int, int] | None:
    if timestamps.size == 0:
        return None
    lo = int(np.searchsorted(timestamps, start_time, side="left"))
    hi = int(np.searchsorted(timestamps, end_time, side="right")) - 1
    lo = max(0, min(lo, int(timestamps.size - 1)))
    hi = max(0, min(hi, int(timestamps.size - 1)))
    if hi < lo:
        return None
    return lo, hi


def stable_subrange(
    lo: int,
    hi: int,
    *,
    min_margin_frames: int = 2,
    margin_ratio: float = 0.15,
) -> tuple[int, int] | None:
    span = hi - lo + 1
    if span <= 0:
        return None
    margin = max(min_margin_frames, int(round(span * margin_ratio)))
    max_margin = max(0, (span - 1) // 3)
    margin = min(margin, max_margin)
    stable_lo = lo + margin
    stable_hi = hi - margin
    if stable_hi < stable_lo:
        stable_lo = lo
        stable_hi = hi
    if stable_hi < stable_lo:
        return None
    return stable_lo, stable_hi


def center_from_progress(lo: int, hi: int, progress: float) -> int:
    if hi <= lo:
        return lo
    progress = float(max(0.0, min(1.0, progress)))
    return int(round(lo + progress * (hi - lo)))


def context_indices_from_center(
    center: int,
    offsets: tuple[int, ...],
    *,
    lo: int,
    hi: int,
) -> list[int] | None:
    frames = [center + off for off in offsets]
    if min(frames) < lo or max(frames) > hi:
        return None
    return frames


def progress_context(
    lo: int,
    hi: int,
    *,
    progress: float,
    offsets: tuple[int, ...],
    min_margin_frames: int = 2,
    margin_ratio: float = 0.15,
) -> tuple[int, list[int]] | None:
    stable = stable_subrange(lo, hi, min_margin_frames=min_margin_frames, margin_ratio=margin_ratio)
    if stable is None:
        return None
    stable_lo, stable_hi = stable
    center = center_from_progress(stable_lo, stable_hi, progress)
    frames = context_indices_from_center(center, offsets, lo=lo, hi=hi)
    if frames is not None:
        return center, frames
    min_off = min(offsets)
    max_off = max(offsets)
    c_lo = lo - min_off
    c_hi = hi - max_off
    if c_hi < c_lo:
        return None
    center = center_from_progress(c_lo, c_hi, progress)
    frames = [center + off for off in offsets]
    return center, frames


def relative_progress_context(
    lo: int,
    hi: int,
    *,
    progress_values: tuple[float, ...],
    min_margin_frames: int = 2,
    margin_ratio: float = 0.05,
) -> tuple[int, list[int]] | None:
    if not progress_values:
        return None
    stable = stable_subrange(lo, hi, min_margin_frames=min_margin_frames, margin_ratio=margin_ratio)
    if stable is None:
        return None
    stable_lo, stable_hi = stable
    span = stable_hi - stable_lo + 1
    if span < len(progress_values):
        return None

    frames = [center_from_progress(stable_lo, stable_hi, p) for p in progress_values]
    if any(b <= a for a, b in zip(frames[:-1], frames[1:])):
        # Fallback to a strictly increasing sequence inside the stable interval.
        lin = np.linspace(stable_lo, stable_hi, num=len(progress_values))
        frames = [int(round(x)) for x in lin]
        fixed: list[int] = []
        last = stable_lo - 1
        for idx, frame in enumerate(frames):
            min_allowed = last + 1
            max_allowed = stable_hi - (len(progress_values) - idx - 1)
            frame = max(min_allowed, min(frame, max_allowed))
            fixed.append(frame)
            last = frame
        frames = fixed

    if any(frame < lo or frame > hi for frame in frames):
        return None
    if any(b <= a for a, b in zip(frames[:-1], frames[1:])):
        return None
    center = frames[len(frames) // 2]
    return center, frames


def midpoint_frame(lo: int, hi: int) -> int:
    return int((lo + hi) // 2)


def nearest_timestamp_indices(source_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    if source_timestamps.size == 0:
        raise ValueError("source_timestamps is empty")
    idx = np.searchsorted(source_timestamps, query_timestamps, side="left")
    idx = np.clip(idx, 0, int(source_timestamps.size - 1))
    left = np.clip(idx - 1, 0, int(source_timestamps.size - 1))
    choose_left = np.abs(source_timestamps[left] - query_timestamps) <= np.abs(source_timestamps[idx] - query_timestamps)
    return np.where(choose_left, left, idx).astype(int)


def speed_window_mean(
    camera_ts: np.ndarray,
    velocity_ts: np.ndarray,
    translational_speed: np.ndarray,
    frame_indices: list[int],
) -> float:
    query_ts = np.asarray([camera_ts[idx] for idx in frame_indices], dtype=np.float64)
    speed_idx = nearest_timestamp_indices(velocity_ts, query_ts)
    return float(np.mean(translational_speed[speed_idx]))


def low_level_to_phase(low_text: str, high_verb: str) -> str | None:
    low_text = normalize_text(low_text).title()
    high_verb = normalize_text(high_verb).lower()
    if not low_text:
        return None
    if low_text == "No Action":
        return "pre-approach"
    if low_text == "Approach":
        return "approach" if high_verb == "pick" else "transfer"
    if low_text in {"Align", "Push", "Twist", "Pull", "Grasp", "Nudge"}:
        return "contact"
    if low_text == "Lift":
        return "hold and carry"
    if low_text == "Release":
        return "release"
    return None


def low_level_to_contact(low_text: str) -> bool | None:
    low_text = normalize_text(low_text).title()
    if not low_text:
        return None
    if low_text in {"No Action", "Approach"}:
        return False
    if low_text in {"Align", "Push", "Twist", "Pull", "Grasp", "Lift", "Release", "Nudge"}:
        return True
    return None


def shuffled_multiple_choice(
    correct_text: str,
    *,
    pool: list[str],
    rng,
    num_choices: int = 4,
) -> tuple[dict[str, str], str]:
    unique_pool = [normalize_text(x) for x in pool if normalize_text(x)]
    seen = {normalize_text(correct_text)}
    distractors: list[str] = []
    for cand in unique_pool:
        if cand in seen:
            continue
        seen.add(cand)
        distractors.append(cand)
    rng.shuffle(distractors)
    options = [normalize_text(correct_text)] + distractors[: max(0, num_choices - 1)]
    if len(options) < num_choices:
        filler = [x for x in PHASE_NAMES if x not in options]
        for item in filler:
            if len(options) >= num_choices:
                break
            options.append(item)
    rng.shuffle(options)
    keys = [chr(ord("A") + i) for i in range(num_choices)]
    choices = {k: v for k, v in zip(keys, options)}
    answer = keys[options.index(normalize_text(correct_text))]
    return choices, answer


def shuffled_binary_choices(
    correct_is_yes: bool,
    *,
    yes_text: str,
    no_text: str,
    rng,
) -> tuple[dict[str, str], str]:
    if rng.random() < 0.5:
        choices = {"A": yes_text, "B": no_text}
        return choices, ("A" if correct_is_yes else "B")
    choices = {"A": no_text, "B": yes_text}
    return choices, ("B" if correct_is_yes else "A")


def write_camera_bytes_to_temp_mp4(h5_file: h5py.File, camera: str) -> tempfile.NamedTemporaryFile:
    obj = h5_file[camera][()]
    payload = bytes(obj.tobytes())
    tmp = tempfile.NamedTemporaryFile(suffix=f"_{camera}.mp4", delete=False)
    tmp.write(payload)
    tmp.flush()
    return tmp
