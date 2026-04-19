#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import h5py
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AIST compressed hdf5 image frames for pilot/eval items.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="observations/images")
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def frame_indices(item: dict[str, Any]) -> list[int]:
    if isinstance(item.get('frame_indices'), list):
        return [int(x) for x in item['frame_indices']]
    if 'frame_index' in item:
        return [int(item['frame_index'])]
    return []


def decode_jpeg(arr) -> Image.Image:
    data = bytes(arr.tolist()).rstrip(b'\x00')
    return Image.open(io.BytesIO(data)).convert('RGB')


def main() -> None:
    args=parse_args()
    items=load_items(Path(args.input_jsonl))
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, h5py.File] = {}
    written=0
    try:
        for item in items:
            ep=str(item['episode_file'])
            camera=str(item.get('camera','cam_high'))
            recording_id=str(item.get('recording_id'))
            if ep not in cache:
                cache[ep]=h5py.File(ep,'r')
            f=cache[ep]
            ds=f[f"{args.image_root}/{camera}"]
            for idx in frame_indices(item):
                out=out_dir / f"{recording_id}_{idx}_{camera}.jpg"
                if out.exists():
                    continue
                img=decode_jpeg(ds[idx])
                img.save(out, format='JPEG', quality=95)
                written += 1
    finally:
        for f in cache.values():
            f.close()
    summary={'input_jsonl': str(args.input_jsonl), 'output_dir': str(out_dir), 'num_items': len(items), 'frames_written': written}
    (out_dir / 'extract_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
