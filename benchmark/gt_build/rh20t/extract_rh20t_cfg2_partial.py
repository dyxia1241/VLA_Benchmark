#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partially extract RH20T cfg2 archive for pilot building.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--scene-list-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--camera",
        default="036422060215",
        help="Primary camera serial to extract timestamps/color for. Use '*' to keep all cameras.",
    )
    parser.add_argument(
        "--mode",
        choices=["signals_only", "with_color", "color_only"],
        default="signals_only",
        help="signals_only extracts metadata/transformed/timestamps; with_color also extracts color.mp4; color_only keeps only metadata + primary-camera color stream",
    )
    return parser.parse_args()


def keep_member(rel: str, *, mode: str, camera: str) -> bool:
    if mode == "color_only":
        if rel == "metadata.json":
            return True
        if rel.startswith("cam_") and "/" in rel:
            cam_dir, leaf = rel.split("/", 1)
            cam_name = cam_dir.replace("cam_", "", 1)
            if camera != "*" and cam_name != camera:
                return False
            return leaf in {"timestamps.npy", "color.mp4"}
        return False

    if rel == "metadata.json":
        return True
    if rel == "robot_command/tcpcommand_timestamp.npy":
        return True
    if rel.startswith("transformed/") and rel.endswith(".npy"):
        return True
    if rel.startswith("cam_") and "/" in rel:
        cam_dir, leaf = rel.split("/", 1)
        cam_name = cam_dir.replace("cam_", "", 1)
        if camera != "*" and cam_name != camera:
            return False
        if leaf == "timestamps.npy":
            return True
        if mode == "with_color" and leaf == "color.mp4":
            return True
    return False


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    scene_list = json.loads(Path(args.scene_list_json).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = scene_list.get("selected_scenes", [])
    selected_dirs = {str(row["scene_dir"]) for row in selected}

    extracted = 0
    matched = 0
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if member.isdir():
                continue
            name = member.name
            parts = name.split("/")
            if len(parts) < 3 or parts[0] != "RH20T_cfg2":
                continue
            scene_dir = parts[1]
            if scene_dir not in selected_dirs:
                continue
            rel = "/".join(parts[2:])
            if not keep_member(rel, mode=args.mode, camera=str(args.camera).strip()):
                continue

            matched += 1
            target = output_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fh = tf.extractfile(member)
            if fh is None:
                continue
            with target.open("wb") as out:
                out.write(fh.read())
            extracted += 1

    summary = {
        "archive": str(archive),
        "output_root": str(output_root),
        "mode": args.mode,
        "camera": str(args.camera),
        "num_selected_scenes": len(selected_dirs),
        "num_matched_files": matched,
        "num_extracted_files": extracted,
    }
    (output_root / "partial_extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
