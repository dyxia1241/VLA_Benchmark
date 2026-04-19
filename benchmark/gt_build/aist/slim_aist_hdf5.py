#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Slim AIST episode hdf5 files by keeping only selected RGB cameras."
    )
    p.add_argument("--input", help="Single input hdf5 path.")
    p.add_argument("--output", help="Output hdf5 path for --input mode.")
    p.add_argument("--root", help="Root directory containing episode_*.hdf5 files.")
    p.add_argument(
        "--keep-images",
        nargs="+",
        default=["cam_high", "cam_low"],
        help="RGB image streams to keep under observations/images.",
    )
    p.add_argument(
        "--keep-depth",
        nargs="*",
        default=[],
        help="Depth streams to keep under observations/depth. Default keeps none.",
    )
    p.add_argument(
        "--keep-sound",
        action="store_true",
        help="Keep observations/sound. Default drops sound.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files in place. Valid with --input or --root.",
    )
    return p.parse_args()


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key in src.keys():
        dst[key] = src[key]


def should_drop_group(path: str, keep_sound: bool) -> bool:
    if path == "observations/sound" and not keep_sound:
        return True
    return False


def should_copy_dataset(
    path: str,
    keep_images: set[str],
    keep_depth: set[str],
    keep_sound: bool,
) -> bool:
    parts = path.split("/")
    if parts[:2] == ["observations", "images"]:
        return len(parts) == 3 and parts[-1] in keep_images
    if parts[:2] == ["observations", "depth"]:
        return len(parts) == 3 and parts[-1] in keep_depth
    if parts[:2] == ["observations", "sound"]:
        return keep_sound
    return True


def copy_filtered_group(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    prefix: str,
    keep_images: set[str],
    keep_depth: set[str],
    keep_sound: bool,
) -> bool:
    kept_any = False
    for name in src_group.keys():
        src_obj = src_group[name]
        child_path = f"{prefix}/{name}" if prefix else name

        if isinstance(src_obj, h5py.Dataset):
            if should_copy_dataset(child_path, keep_images, keep_depth, keep_sound):
                src_group.copy(name, dst_group, name=name)
                kept_any = True
            continue

        if should_drop_group(child_path, keep_sound):
            continue

        child_dst = dst_group.create_group(name)
        copy_attrs(src_obj.attrs, child_dst.attrs)
        child_kept = copy_filtered_group(
            src_obj,
            child_dst,
            child_path,
            keep_images,
            keep_depth,
            keep_sound,
        )
        if not child_kept and len(child_dst.attrs) == 0:
            del dst_group[name]
            continue
        kept_any = True
    return kept_any


def slim_one(
    src_path: Path,
    dst_path: Path,
    keep_images: Iterable[str],
    keep_depth: Iterable[str],
    keep_sound: bool,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    keep_images_set = set(keep_images)
    keep_depth_set = set(keep_depth)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        copy_filtered_group(
            src,
            dst,
            "",
            keep_images_set,
            keep_depth_set,
            keep_sound,
        )


def slim_in_place(
    path: Path,
    keep_images: Iterable[str],
    keep_depth: Iterable[str],
    keep_sound: bool,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    slim_one(path, tmp_path, keep_images, keep_depth, keep_sound)
    os.replace(tmp_path, path)


def iter_hdf5_files(root: Path) -> list[Path]:
    return sorted(root.rglob("episode_*.hdf5"))


def main() -> None:
    args = parse_args()
    if not args.input and not args.root:
        raise SystemExit("one of --input or --root is required")
    if args.output and not args.input:
        raise SystemExit("--output requires --input")
    if args.root and args.output:
        raise SystemExit("--output cannot be used with --root")

    processed = 0
    if args.input:
        src = Path(args.input)
        if args.in_place:
            slim_in_place(src, args.keep_images, args.keep_depth, args.keep_sound)
        else:
            if not args.output:
                raise SystemExit("--output is required when not using --in-place")
            slim_one(
                src,
                Path(args.output),
                args.keep_images,
                args.keep_depth,
                args.keep_sound,
            )
        processed = 1
    else:
        root = Path(args.root)
        files = iter_hdf5_files(root)
        for idx, path in enumerate(files, start=1):
            if args.in_place:
                slim_in_place(path, args.keep_images, args.keep_depth, args.keep_sound)
            else:
                raise SystemExit("--root mode currently requires --in-place")
            processed += 1
            if idx % 10 == 0 or idx == len(files):
                print(f"[AIST] slimmed {idx}/{len(files)} files")

    print(
        f"[AIST] completed; files={processed}; keep_images={args.keep_images}; "
        f"keep_depth={args.keep_depth}; keep_sound={args.keep_sound}"
    )


if __name__ == "__main__":
    main()
