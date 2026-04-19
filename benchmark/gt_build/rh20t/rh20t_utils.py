from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
RH20T_PILOT_ROOT = BENCHMARK_ROOT / "rh20t_cfg2_pilot_v0"
DEFAULT_EXTRACTED_ROOT = RH20T_PILOT_ROOT / "extracted_primary_cam" / "RH20T_cfg2"
DEFAULT_SELECTED_SCENES_JSON = RH20T_PILOT_ROOT / "selected_scenes.json"
DEFAULT_TASK_CATALOG_JSON = (
    BENCHMARK_ROOT / "manual_audit" / "semantic_affordance_audit" / "catalogs" / "rh20t_task_catalog_v1.json"
)

PRIMARY_CAMERA = "036422060215"
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

EVENT_SPEED_HI_FLOOR = 0.040
EVENT_SPEED_LO_FLOOR = 0.022
CONTACT_FORCE_HI_FLOOR = 2.50
CONTACT_FORCE_LO_FLOOR = 1.50
T6_STATIONARY_THRESHOLD = 0.020
T6_ACTIVE_THRESHOLD = 0.045
INTERVAL_MERGE_GAP = 20
CONTACT_MIN_LEN = 6
MOVING_MIN_LEN = 8


@dataclass(frozen=True)
class Interval:
    interval_id: str
    start: int
    end: int
    first_contact: int | None
    last_contact: int | None
    peak_force_row: int
    peak_speed_row: int
    contact_ratio: float
    moving_ratio: float

    @property
    def span(self) -> int:
        return int(self.end - self.start + 1)


@dataclass(frozen=True)
class SceneSignals:
    scene_dir: str
    task_id: str
    task_description: str
    camera: str
    rating: int
    calib_quality: int
    timestamps: np.ndarray
    pos_xyz: np.ndarray
    force_xyz: np.ndarray
    speed_raw: np.ndarray
    speed_ema: np.ndarray
    force_mag_raw: np.ndarray
    force_mag_ema: np.ndarray
    gripper_width: np.ndarray
    speed_event_hi: float
    speed_event_lo: float
    force_contact_hi: float
    force_contact_lo: float
    moving_mask: np.ndarray
    contact_mask: np.ndarray
    intervals: list[Interval]

    @property
    def n_rows(self) -> int:
        return int(self.timestamps.shape[0])


def load_task_catalog(path: Path = DEFAULT_TASK_CATALOG_JSON) -> dict[str, dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    tasks = obj.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"Invalid RH20T task catalog: {path}")
    out: dict[str, dict[str, str]] = {}
    for task_id, payload in tasks.items():
        if not isinstance(payload, dict):
            continue
        out[str(task_id)] = {
            "task_description_english": str(payload.get("task_description_english", "")).strip(),
            "task_description_chinese": str(payload.get("task_description_chinese", "")).strip(),
        }
    return out


def load_selected_scenes(path: Path = DEFAULT_SELECTED_SCENES_JSON) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("selected_scenes", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid selected-scenes json: {path}")
    return [row for row in rows if isinstance(row, dict)]


def ema1d(values: np.ndarray, span: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"ema1d expects 1D input, got shape {arr.shape}")
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.size == 0:
        return out
    alpha = 2.0 / float(span + 1)
    out[0] = float(arr[0])
    for i in range(1, int(arr.size)):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


def hysteresis_mask(mask_hi: np.ndarray, mask_lo: np.ndarray) -> np.ndarray:
    hi = np.asarray(mask_hi, dtype=bool)
    lo = np.asarray(mask_lo, dtype=bool)
    if hi.shape != lo.shape:
        raise ValueError("mask_hi and mask_lo must have the same shape")
    state = False
    out = np.zeros_like(hi, dtype=bool)
    for i in range(int(hi.size)):
        if not state and bool(hi[i]):
            state = True
        elif state and not bool(lo[i]):
            state = False
        out[i] = state
    return out


def mask_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    out: list[tuple[int, int]] = []
    i = 0
    n = int(arr.size)
    while i < n:
        if not bool(arr[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(arr[j]):
            j += 1
        if j - i >= int(min_len):
            out.append((i, j - 1))
        i = j
    return out


def merge_interaction_runs(
    contact_runs: list[tuple[int, int]],
    moving_runs: list[tuple[int, int]],
    *,
    merge_gap: int,
) -> list[tuple[int, int]]:
    sources = [(a, b) for a, b in contact_runs] + [(a, b) for a, b in moving_runs]
    if not sources:
        return []
    sources.sort(key=lambda x: (x[0], x[1]))

    merged: list[list[int]] = [[int(sources[0][0]), int(sources[0][1])]]
    for start, end in sources[1:]:
        cur = merged[-1]
        gap = int(start) - int(cur[1]) - 1
        if gap <= int(merge_gap):
            cur[1] = max(int(cur[1]), int(end))
        else:
            merged.append([int(start), int(end)])
    return [(int(a), int(b)) for a, b in merged]


def stable_subrange(
    lo: int,
    hi: int,
    *,
    min_margin_frames: int = 4,
    margin_ratio: float = 0.12,
) -> tuple[int, int] | None:
    if hi < lo:
        return None
    span = int(hi - lo + 1)
    if span <= 0:
        return None
    margin = max(int(min_margin_frames), int(round(span * float(margin_ratio))))
    margin = min(margin, max(0, (span - 1) // 3))
    s_lo = int(lo + margin)
    s_hi = int(hi - margin)
    if s_hi < s_lo:
        s_lo, s_hi = int(lo), int(hi)
    if s_hi < s_lo:
        return None
    return s_lo, s_hi


def center_from_progress(lo: int, hi: int, progress: float) -> int:
    if hi <= lo:
        return int(lo)
    p = float(max(0.0, min(1.0, progress)))
    return int(round(float(lo) + p * float(hi - lo)))


def context_indices_from_center(center: int, offsets: tuple[int, ...], *, lo: int, hi: int) -> list[int] | None:
    frames = [int(center + off) for off in offsets]
    if min(frames) < int(lo) or max(frames) > int(hi):
        return None
    return frames


def progress_context(
    lo: int,
    hi: int,
    *,
    progress: float,
    offsets: tuple[int, ...],
    min_margin_frames: int = 4,
    margin_ratio: float = 0.12,
) -> tuple[int, list[int]] | None:
    stable = stable_subrange(lo, hi, min_margin_frames=min_margin_frames, margin_ratio=margin_ratio)
    if stable is None:
        return None
    s_lo, s_hi = stable
    center = center_from_progress(s_lo, s_hi, progress)
    frames = context_indices_from_center(center, offsets, lo=lo, hi=hi)
    if frames is not None:
        return int(center), frames
    min_off = int(min(offsets))
    max_off = int(max(offsets))
    c_lo = int(lo - min_off)
    c_hi = int(hi - max_off)
    if c_hi < c_lo:
        return None
    center = center_from_progress(c_lo, c_hi, progress)
    return int(center), [int(center + off) for off in offsets]


def relative_progress_context(
    lo: int,
    hi: int,
    *,
    progress_values: tuple[float, ...],
    min_gap_frames: int = 6,
) -> tuple[int, list[int]] | None:
    stable = stable_subrange(lo, hi, min_margin_frames=4, margin_ratio=0.10)
    if stable is None:
        return None
    s_lo, s_hi = stable
    frames = [center_from_progress(s_lo, s_hi, p) for p in progress_values]
    if len(set(frames)) != len(frames):
        return None
    frames = sorted(int(x) for x in frames)
    if any((b - a) < int(min_gap_frames) for a, b in zip(frames[:-1], frames[1:])):
        return None
    center = int(frames[len(frames) // 2])
    return center, frames


def evenly_spaced_sample(rows: list[int], *, max_count: int, min_gap: int = 20) -> list[int]:
    if max_count <= 0 or not rows:
        return []
    uniq = sorted({int(x) for x in rows})
    picked: list[int] = []
    for row in uniq:
        if not picked or (row - picked[-1]) >= int(min_gap):
            picked.append(int(row))
    if len(picked) <= max_count:
        return picked
    idxs = np.linspace(0, len(picked) - 1, num=max_count)
    return [int(picked[int(round(i))]) for i in idxs]


def build_intervals(
    scene_dir: str,
    contact_mask: np.ndarray,
    moving_mask: np.ndarray,
    force_mag: np.ndarray,
    speed_ema: np.ndarray,
) -> list[Interval]:
    contact_runs = mask_runs(contact_mask, min_len=CONTACT_MIN_LEN)
    moving_runs = mask_runs(moving_mask, min_len=MOVING_MIN_LEN)
    merged = merge_interaction_runs(contact_runs, moving_runs, merge_gap=INTERVAL_MERGE_GAP)
    intervals: list[Interval] = []
    for idx, (start, end) in enumerate(merged, start=1):
        if int(end - start + 1) < 24:
            continue
        local_contact = np.where(contact_mask[start : end + 1])[0]
        local_moving = np.where(moving_mask[start : end + 1])[0]
        first_contact = int(start + local_contact[0]) if local_contact.size else None
        last_contact = int(start + local_contact[-1]) if local_contact.size else None
        peak_force_row = int(start + np.argmax(force_mag[start : end + 1]))
        peak_speed_row = int(start + np.argmax(speed_ema[start : end + 1]))
        intervals.append(
            Interval(
                interval_id=f"{scene_dir}:i{idx:03d}",
                start=int(start),
                end=int(end),
                first_contact=first_contact,
                last_contact=last_contact,
                peak_force_row=peak_force_row,
                peak_speed_row=peak_speed_row,
                contact_ratio=float(np.mean(contact_mask[start : end + 1])),
                moving_ratio=float(np.mean(moving_mask[start : end + 1])),
            )
        )
    return intervals


def load_scene_signals(
    extracted_root: Path,
    *,
    scene_dir: str,
    task_description: str,
    camera: str = PRIMARY_CAMERA,
) -> SceneSignals:
    scene_root = Path(extracted_root) / str(scene_dir)
    if not scene_root.exists():
        raise FileNotFoundError(f"Missing scene root: {scene_root}")

    meta = json.loads((scene_root / "metadata.json").read_text(encoding="utf-8"))
    ts_obj = np.load(scene_root / f"cam_{camera}" / "timestamps.npy", allow_pickle=True).item()
    camera_ts = np.asarray(ts_obj["color"], dtype=np.int64)

    tcp = np.load(scene_root / "transformed" / "tcp_base.npy", allow_pickle=True).item()[camera]
    ft = np.load(scene_root / "transformed" / "force_torque_base.npy", allow_pickle=True).item()[camera]
    gripper = np.load(scene_root / "transformed" / "gripper.npy", allow_pickle=True).item()[camera]

    timestamps = np.asarray([int(row["timestamp"]) for row in tcp], dtype=np.int64)

    # Most RH20T cameras align exactly to the transformed streams. Some cameras,
    # however, carry a trailing extra video frame while the transformed streams
    # stop one step earlier. Keep the exact shared prefix instead of failing.
    if timestamps.shape != camera_ts.shape or not np.array_equal(timestamps, camera_ts):
        common = min(int(timestamps.shape[0]), int(camera_ts.shape[0]))
        if common <= 0:
            raise ValueError(f"Empty timestamp stream for scene={scene_dir} camera={camera}")
        if not np.array_equal(timestamps[:common], camera_ts[:common]):
            raise ValueError(f"Timestamp mismatch for scene={scene_dir} camera={camera}")
        timestamps = timestamps[:common]
        camera_ts = camera_ts[:common]
        tcp = tcp[:common]
        ft = ft[:common]

    valid_mask = np.asarray(
        [
            (row.get("tcp") is not None)
            and (ft_row.get("zeroed") is not None)
            and (int(ts) in gripper)
            and (gripper[int(ts)].get("gripper_info") is not None)
            for row, ft_row, ts in zip(tcp, ft, timestamps)
        ],
        dtype=bool,
    )
    if not np.any(valid_mask):
        raise ValueError(f"No valid synchronized signal rows for scene={scene_dir} camera={camera}")
    valid_rows = np.where(valid_mask)[0]
    lo = int(valid_rows[0])
    hi = int(valid_rows[-1])
    timestamps = timestamps[lo : hi + 1]
    tcp = tcp[lo : hi + 1]
    ft = ft[lo : hi + 1]
    valid_mask = valid_mask[lo : hi + 1]
    if not np.all(valid_mask):
        keep_idx = np.where(valid_mask)[0]
        timestamps = timestamps[keep_idx]
        tcp = [tcp[int(i)] for i in keep_idx.tolist()]
        ft = [ft[int(i)] for i in keep_idx.tolist()]

    if timestamps.shape[0] < 24:
        raise ValueError(f"Too few valid rows after cleanup for scene={scene_dir} camera={camera}")

    pos_xyz = np.stack([np.asarray(row["tcp"][:3], dtype=np.float64) for row in tcp], axis=0)
    force_xyz = np.stack([np.asarray(row["zeroed"][:3], dtype=np.float64) for row in ft], axis=0)
    gripper_width = np.asarray([float(gripper[int(ts)]["gripper_info"][0]) for ts in timestamps], dtype=np.float64)

    dt = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64) / 1000.0
    dt[0] = float(np.median(dt[1:])) if dt.shape[0] > 1 else 0.04
    dt = np.clip(dt, 1e-3, 1.0)

    pos_delta = np.diff(pos_xyz, axis=0, prepend=pos_xyz[[0]])
    speed_raw = np.linalg.norm(pos_delta, axis=1) / dt
    speed_ema = ema1d(speed_raw, span=9)
    force_mag_raw = np.linalg.norm(force_xyz, axis=1)
    force_mag_ema = ema1d(force_mag_raw, span=9)

    speed_event_hi = float(max(EVENT_SPEED_HI_FLOOR, np.percentile(speed_ema, 85)))
    speed_event_lo = float(max(EVENT_SPEED_LO_FLOOR, speed_event_hi * 0.55))
    force_contact_hi = float(max(CONTACT_FORCE_HI_FLOOR, np.percentile(force_mag_ema, 80)))
    force_contact_lo = float(max(CONTACT_FORCE_LO_FLOOR, force_contact_hi * 0.60))

    moving_mask = hysteresis_mask(speed_ema >= speed_event_hi, speed_ema >= speed_event_lo)
    contact_mask = hysteresis_mask(force_mag_ema >= force_contact_hi, force_mag_ema >= force_contact_lo)
    intervals = build_intervals(scene_dir, contact_mask, moving_mask, force_mag_ema, speed_ema)

    task_id = str(scene_dir).split("_user_")[0]
    return SceneSignals(
        scene_dir=str(scene_dir),
        task_id=task_id,
        task_description=str(task_description).strip(),
        camera=str(camera),
        rating=int(meta.get("rating", -1)),
        calib_quality=int(meta.get("calib_quality", -1)),
        timestamps=timestamps,
        pos_xyz=pos_xyz,
        force_xyz=force_xyz,
        speed_raw=speed_raw,
        speed_ema=speed_ema,
        force_mag_raw=force_mag_raw,
        force_mag_ema=force_mag_ema,
        gripper_width=gripper_width,
        speed_event_hi=speed_event_hi,
        speed_event_lo=speed_event_lo,
        force_contact_hi=force_contact_hi,
        force_contact_lo=force_contact_lo,
        moving_mask=moving_mask,
        contact_mask=contact_mask,
        intervals=intervals,
    )


def phase_candidate_rows(scene: SceneSignals) -> dict[str, list[int]]:
    n = scene.n_rows
    rows = np.arange(n, dtype=int)
    candidates: dict[str, list[int]] = {name: [] for name in PHASE_NAMES}
    if not scene.intervals:
        return candidates

    first_interval = scene.intervals[0]
    pre_lo = max(12, first_interval.start - 60)
    pre_hi = max(pre_lo, first_interval.start - 18)
    if pre_hi > pre_lo:
        mask = (~scene.contact_mask[pre_lo:pre_hi]) & (scene.speed_ema[pre_lo:pre_hi] <= max(scene.speed_event_lo, 0.028))
        candidates["pre-approach"] = [int(pre_lo + idx) for idx in np.where(mask)[0].tolist()]

    for interval in scene.intervals:
        lo = int(interval.start)
        hi = int(interval.end)
        span = int(hi - lo + 1)
        if span < 18:
            continue

        fc = interval.first_contact
        lc = interval.last_contact
        if fc is not None and fc - lo >= 8:
            approach_rows = rows[lo : fc - 2]
            mask = ~scene.contact_mask[approach_rows]  # type: ignore[index]
            candidates["approach"].extend(int(x) for x in approach_rows[mask].tolist())
        else:
            cut = lo + max(6, int(round(span * 0.20)))
            approach_rows = rows[lo:cut]
            if approach_rows.size:
                mask = ~scene.contact_mask[approach_rows]  # type: ignore[index]
                candidates["approach"].extend(int(x) for x in approach_rows[mask].tolist())

        if fc is not None:
            c_lo = fc
            c_hi = min(hi, fc + max(8, int(round(span * 0.10))))
            candidates["contact"].extend(int(x) for x in range(c_lo, c_hi + 1))

        if fc is not None and lc is not None and (lc - fc) >= 14:
            mid_lo = fc + 8
            mid_hi = max(mid_lo, lc - 8)
            hold_rows = rows[mid_lo : mid_hi + 1]
            if hold_rows.size:
                hold_mask = scene.contact_mask[hold_rows] & (scene.speed_ema[hold_rows] <= max(T6_STATIONARY_THRESHOLD, scene.speed_event_lo * 1.15))
                candidates["hold and carry"].extend(int(x) for x in hold_rows[hold_mask].tolist())

        transfer_rows = rows[lo : hi + 1]
        transfer_mask = scene.moving_mask[transfer_rows] & (scene.speed_ema[transfer_rows] >= max(T6_ACTIVE_THRESHOLD, scene.speed_event_hi * 0.90))
        candidates["transfer"].extend(int(x) for x in transfer_rows[transfer_mask].tolist())

        if lc is not None:
            rel_lo = min(hi, lc + 4)
            rel_hi = min(hi, lc + max(14, int(round(span * 0.15))))
            if rel_hi > rel_lo:
                release_rows = rows[rel_lo : rel_hi + 1]
                candidates["release"].extend(int(x) for x in release_rows.tolist())
            else:
                tail_lo = max(lo, hi - 10)
                candidates["release"].extend(int(x) for x in range(tail_lo, hi + 1))
        else:
            tail_lo = max(lo, hi - max(10, int(round(span * 0.12))))
            candidates["release"].extend(int(x) for x in range(tail_lo, hi + 1))

    out: dict[str, list[int]] = {}
    for phase, phase_rows in candidates.items():
        out[phase] = evenly_spaced_sample(phase_rows, max_count=3, min_gap=18)
    return out
