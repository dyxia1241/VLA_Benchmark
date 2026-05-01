#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read LIBERO spatial HDF5 demos and export summaries with preview frames."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_data/libero_spatial"),
        help="Directory containing LIBERO spatial .hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw_data/libero_spatial_extracted"),
        help="Directory to write extracted summaries and preview images.",
    )
    parser.add_argument(
        "--preview-demo",
        default="demo_0",
        help="Demo key used for preview frame export.",
    )
    return parser.parse_args()


def demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def dataset_meta(dataset: h5py.Dataset) -> dict[str, object]:
    meta = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
    }
    compression = dataset.compression
    if compression is not None:
        meta["compression"] = compression
    return meta


def save_preview_images(demo_group: h5py.Group, out_dir: Path) -> list[str]:
    saved_files: list[str] = []
    obs_group = demo_group["obs"]
    for camera in ("agentview_rgb", "eye_in_hand_rgb"):
        if camera not in obs_group:
            continue
        frames = obs_group[camera]
        if len(frames) == 0:
            continue
        frame_indices = sorted(set([0, len(frames) // 2, len(frames) - 1]))
        for frame_index in frame_indices:
            image = Image.fromarray(frames[frame_index])
            image_name = f"{camera}_frame_{frame_index:04d}.png"
            image.save(out_dir / image_name)
            saved_files.append(image_name)
    return saved_files


def summarize_file(src_path: Path, dst_root: Path, preview_demo: str) -> dict[str, object]:
    with h5py.File(src_path, "r") as h5_file:
        data_group = h5_file["data"]
        demo_names = sorted(data_group.keys(), key=demo_sort_key)
        file_out_dir = dst_root / src_path.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        demo_summaries: list[dict[str, object]] = []
        total_steps = 0
        preview_images: list[str] = []

        for demo_name in demo_names:
            demo_group = data_group[demo_name]
            actions = demo_group["actions"]
            obs_group = demo_group["obs"]
            step_count = int(actions.shape[0])
            total_steps += step_count

            obs_meta = {
                name: dataset_meta(obs_group[name]) for name in sorted(obs_group.keys())
            }

            demo_summary = {
                "demo": demo_name,
                "steps": step_count,
                "datasets": {
                    "actions": dataset_meta(demo_group["actions"]),
                    "dones": dataset_meta(demo_group["dones"]),
                    "rewards": dataset_meta(demo_group["rewards"]),
                    "robot_states": dataset_meta(demo_group["robot_states"]),
                    "states": dataset_meta(demo_group["states"]),
                    "obs": obs_meta,
                },
                "first_action": np.asarray(demo_group["actions"][0]).round(6).tolist(),
                "last_done": int(demo_group["dones"][-1]),
                "reward_sum": int(np.asarray(demo_group["rewards"]).sum()),
            }
            demo_summaries.append(demo_summary)

            if demo_name == preview_demo:
                preview_images = save_preview_images(demo_group, file_out_dir)

        summary = {
            "source_file": str(src_path),
            "file_size_bytes": src_path.stat().st_size,
            "demo_count": len(demo_names),
            "total_steps": total_steps,
            "preview_demo": preview_demo if preview_demo in data_group else None,
            "preview_images": preview_images,
            "root_attrs": {key: json_value(value) for key, value in h5_file.attrs.items()},
            "demos": demo_summaries,
        }

        summary_path = file_out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "file": src_path.name,
            "summary_path": str(summary_path),
            "output_dir": str(file_out_dir),
            "demo_count": len(demo_names),
            "total_steps": total_steps,
            "preview_images": preview_images,
        }


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"No .hdf5 files found in {input_dir}")

    index = []
    for src_path in files:
        index.append(summarize_file(src_path, output_dir, args.preview_demo))

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(index)} summaries to {index_path}")


if __name__ == "__main__":
    main()
