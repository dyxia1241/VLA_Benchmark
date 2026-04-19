#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROBOT_DIR_RE = re.compile(
    r"^(task_\d+_user_\d+_scene_\d+_cfg_\d+)$"
)
HUMAN_DIR_RE = re.compile(
    r"^(task_\d+_user_\d+_scene_\d+_cfg_\d+_human(?:_\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for RH20T cfg tar.gz without full extraction.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-metadata", type=int, default=24, help="How many metadata.json files to parse for rating-field discovery")
    return parser.parse_args()


def normalize_task_id(scene_dir: str) -> str:
    return scene_dir.split("_user_")[0]


def normalize_robot_scene_dir(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 2:
        return None
    name = parts[1]
    if ROBOT_DIR_RE.match(name):
        return name
    return None


def normalize_human_scene_dir(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 2:
        return None
    name = parts[1]
    if HUMAN_DIR_RE.match(name):
        return name
    return None


def safe_json_load(raw: bytes) -> dict[str, Any] | None:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            return json.loads(raw.decode("utf-8", "ignore"))
        except Exception:
            return None


def rating_candidate_keys(meta: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in meta.keys():
        lk = str(key).lower()
        if "rating" in lk or "score" in lk or "completion" in lk or "quality" in lk or "success" in lk:
            keys.append(str(key))
    return sorted(keys)


def maybe_numeric(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return None
    return None


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert archive.exists(), f"Missing archive: {archive}"

    robot_scene_files: defaultdict[str, set[str]] = defaultdict(set)
    human_scene_files: defaultdict[str, set[str]] = defaultdict(set)
    task_to_robot_scenes: defaultdict[str, set[str]] = defaultdict(set)
    task_to_human_scenes: defaultdict[str, set[str]] = defaultdict(set)
    camera_serials: Counter[str] = Counter()
    metadata_samples: list[dict[str, Any]] = []
    metadata_candidate_key_counter: Counter[str] = Counter()
    rating_value_counter: Counter[str] = Counter()
    calibration_value_counter: Counter[str] = Counter()
    n_metadata_parsed = 0

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            name = member.name
            if member.isdir():
                continue

            robot_scene = normalize_robot_scene_dir(name)
            human_scene = normalize_human_scene_dir(name)

            if robot_scene:
                rel = "/".join(name.split("/")[2:])
                robot_scene_files[robot_scene].add(rel)
                task_to_robot_scenes[normalize_task_id(robot_scene)].add(robot_scene)
                parts = name.split("/")
                if len(parts) > 2 and parts[2].startswith("cam_"):
                    camera_serials[parts[2][4:]] += 1

                if rel == "metadata.json" and n_metadata_parsed < int(args.sample_metadata):
                    fh = tf.extractfile(member)
                    if fh is not None:
                        meta = safe_json_load(fh.read())
                        if meta is not None:
                            n_metadata_parsed += 1
                            candidate_keys = rating_candidate_keys(meta)
                            metadata_candidate_key_counter.update(candidate_keys)
                            sample = {
                                "scene_dir": robot_scene,
                                "candidate_keys": candidate_keys,
                            }
                            for key in candidate_keys:
                                sample[key] = meta.get(key)
                                numeric = maybe_numeric(meta.get(key))
                                if numeric is not None:
                                    if "calib" in key.lower():
                                        calibration_value_counter[str(numeric)] += 1
                                    if any(tok in key.lower() for tok in ("rating", "completion", "score", "success")):
                                        rating_value_counter[str(numeric)] += 1
                            metadata_samples.append(sample)

            elif human_scene:
                rel = "/".join(name.split("/")[2:])
                human_scene_files[human_scene].add(rel)
                task_to_human_scenes[normalize_task_id(human_scene)].add(human_scene)

    scene_rows: list[dict[str, Any]] = []
    availability_counter: Counter[str] = Counter()
    per_task_summary: list[dict[str, Any]] = []

    for scene_dir in sorted(robot_scene_files):
        files = robot_scene_files[scene_dir]
        task_id = normalize_task_id(scene_dir)
        availability = {
            "metadata": "metadata.json" in files,
            "tcp": "transformed/tcp.npy" in files,
            "tcp_base": "transformed/tcp_base.npy" in files,
            "gripper": "transformed/gripper.npy" in files,
            "force_torque": "transformed/force_torque.npy" in files,
            "force_torque_base": "transformed/force_torque_base.npy" in files,
            "high_freq_data": "transformed/high_freq_data.npy" in files,
            "joint": "transformed/joint.npy" in files,
            "robot_command": "robot_command/tcpcommand_timestamp.npy" in files,
        }
        cam_dirs = sorted(
            {
                path.split("/")[0]
                for path in files
                if path.startswith("cam_") and "/" in path
            }
        )
        n_cam_color = sum(1 for cam in cam_dirs if f"{cam}/color.mp4" in files)
        n_cam_depth = sum(1 for cam in cam_dirs if f"{cam}/depth.mp4" in files)
        n_cam_timestamps = sum(1 for cam in cam_dirs if f"{cam}/timestamps.npy" in files)

        row = {
            "task_id": task_id,
            "scene_dir": scene_dir,
            "n_cameras": len(cam_dirs),
            "n_cam_color": n_cam_color,
            "n_cam_depth": n_cam_depth,
            "n_cam_timestamps": n_cam_timestamps,
            **availability,
        }
        scene_rows.append(row)
        for key, value in availability.items():
            if value:
                availability_counter[key] += 1
        if len(cam_dirs) > 0:
            availability_counter["scene_has_any_camera"] += 1
        if n_cam_color > 0:
            availability_counter["scene_has_color"] += 1
        if n_cam_timestamps > 0:
            availability_counter["scene_has_timestamps"] += 1

    for task_id in sorted(task_to_robot_scenes):
        scenes = sorted(task_to_robot_scenes[task_id])
        task_rows = [row for row in scene_rows if row["task_id"] == task_id]
        per_task_summary.append(
            {
                "task_id": task_id,
                "n_robot_scenes": len(scenes),
                "n_human_scenes": len(task_to_human_scenes.get(task_id, set())),
                "n_scenes_with_ft": sum(1 for row in task_rows if row["force_torque"]),
                "n_scenes_with_gripper": sum(1 for row in task_rows if row["gripper"]),
                "n_scenes_with_tcp": sum(1 for row in task_rows if row["tcp"]),
                "n_scenes_with_color": sum(1 for row in task_rows if row["n_cam_color"] > 0),
                "n_scenes_with_timestamps": sum(1 for row in task_rows if row["n_cam_timestamps"] > 0),
            }
        )

    summary = {
        "archive": str(archive),
        "robot_task_count": len(task_to_robot_scenes),
        "robot_scene_count": len(robot_scene_files),
        "human_scene_count": len(human_scene_files),
        "camera_serials": sorted(camera_serials.keys()),
        "camera_serial_count": len(camera_serials),
        "signal_availability_counts": dict(availability_counter),
        "metadata_candidate_key_counts": dict(metadata_candidate_key_counter),
        "metadata_rating_value_counter_from_sample": dict(rating_value_counter),
        "metadata_calibration_value_counter_from_sample": dict(calibration_value_counter),
        "metadata_samples_parsed": n_metadata_parsed,
    }

    (output_dir / "rh20t_cfg2_archive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rh20t_cfg2_metadata_samples.json").write_text(
        json.dumps(metadata_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rh20t_cfg2_scene_index.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in scene_rows) + ("\n" if scene_rows else ""),
        encoding="utf-8",
    )
    (output_dir / "rh20t_cfg2_task_summary.json").write_text(
        json.dumps(per_task_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
