#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import decord
from PIL import Image


decord.bridge.set_bridge("native")


@dataclass(frozen=True)
class FrameRequest:
    scene_dir: str
    frame_index: int
    camera: str

    def out_name(self) -> str:
        return f"{self.scene_dir}_{self.frame_index}_{self.camera}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RH20T pilot JPEG frames referenced by JSONL items.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def load_requests(path: Path) -> list[FrameRequest]:
    reqs: dict[tuple[str, int, str], FrameRequest] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            scene_dir = str(item.get("scene_dir") or item.get("recording_id") or "").strip()
            camera = str(item.get("camera", "")).strip()
            if not scene_dir or not camera:
                continue
            if "frame_index" in item:
                req = FrameRequest(scene_dir=scene_dir, frame_index=int(item["frame_index"]), camera=camera)
                reqs[(req.scene_dir, req.frame_index, req.camera)] = req
            for frame_index in item.get("frame_indices", []) or []:
                req = FrameRequest(scene_dir=scene_dir, frame_index=int(frame_index), camera=camera)
                reqs[(req.scene_dir, req.frame_index, req.camera)] = req
    return list(reqs.values())


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    extracted_root = Path(args.extracted_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reqs = load_requests(input_jsonl)
    by_stream: dict[tuple[str, str], list[FrameRequest]] = defaultdict(list)
    for req in reqs:
        by_stream[(req.scene_dir, req.camera)].append(req)

    ok = 0
    skipped_existing = 0
    failed = 0
    failed_msgs: list[str] = []
    processed = 0
    started = time.time()
    progress_every = max(1, int(args.progress_every))

    for (scene_dir, camera), stream_reqs in by_stream.items():
        mp4_path = extracted_root / scene_dir / f"cam_{camera}" / "color.mp4"
        if not mp4_path.exists():
            for req in stream_reqs:
                failed += 1
                failed_msgs.append(f"{req.out_name()} :: missing mp4 {mp4_path}")
            continue

        try:
            vr = decord.VideoReader(str(mp4_path), ctx=decord.cpu(0))
            n_frames = len(vr)
            for req in stream_reqs:
                out_path = output_dir / req.out_name()
                processed += 1
                try:
                    if args.skip_existing and out_path.exists():
                        skipped_existing += 1
                    else:
                        if req.frame_index < 0 or req.frame_index >= n_frames:
                            raise IndexError(f"frame_index {req.frame_index} out of range len={n_frames}")
                        arr = vr[req.frame_index].asnumpy()
                        Image.fromarray(arr).save(out_path, format="JPEG", quality=95)
                        ok += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    failed_msgs.append(f"{req.out_name()} :: {exc}")
                if processed % progress_every == 0:
                    elapsed = max(1e-6, time.time() - started)
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "total": len(reqs),
                                "ok": ok,
                                "skipped_existing": skipped_existing,
                                "failed": failed,
                                "requests_per_sec": round(processed / elapsed, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001
            for req in stream_reqs:
                failed += 1
                failed_msgs.append(f"{req.out_name()} :: {exc}")

    summary = {
        "input_jsonl": str(input_jsonl),
        "extracted_root": str(extracted_root),
        "output_dir": str(output_dir),
        "num_requests_deduped": len(reqs),
        "num_ok": ok,
        "num_skipped_existing": skipped_existing,
        "num_failed": failed,
        "num_processed_total": processed,
        "num_unique_streams": len(by_stream),
    }
    (output_dir / "extract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_msgs:
        (output_dir / "extract_failures.log").write_text("\n".join(failed_msgs) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
