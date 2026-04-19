#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import decord
import h5py
from PIL import Image

from reassemble_utils import REASSEMBLE_ROOT, recording_h5_path, write_camera_bytes_to_temp_mp4


decord.bridge.set_bridge("native")


@dataclass(frozen=True)
class FrameRequest:
    recording_id: str
    frame_index: int
    camera: str

    def out_name(self) -> str:
        return f"{self.recording_id}_{self.frame_index}_{self.camera}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract JPEG frames referenced by REASSEMBLE GT jsonl using embedded H5 MP4 payloads.")
    parser.add_argument("--dataset-root", default=str(REASSEMBLE_ROOT))
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def load_requests(path: Path) -> list[FrameRequest]:
    reqs: dict[tuple[str, int, str], FrameRequest] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            recording_id = str(item.get("recording_id") or item.get("task_id") or "").strip()
            camera = str(item.get("camera", "hama1")).strip()
            if not recording_id:
                continue
            if "frame_index" in item:
                req = FrameRequest(recording_id=recording_id, frame_index=int(item["frame_index"]), camera=camera)
                reqs[(req.recording_id, req.frame_index, req.camera)] = req
            for frame_index in item.get("frame_indices", []) or []:
                req = FrameRequest(recording_id=recording_id, frame_index=int(frame_index), camera=camera)
                reqs[(req.recording_id, req.frame_index, req.camera)] = req
    return list(reqs.values())


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reqs = load_requests(input_jsonl)
    by_stream: dict[tuple[str, str], list[FrameRequest]] = defaultdict(list)
    for req in reqs:
        by_stream[(req.recording_id, req.camera)].append(req)

    ok = 0
    skipped_existing = 0
    failed = 0
    failed_msgs: list[str] = []
    processed = 0
    started = time.time()
    progress_every = max(1, int(args.progress_every))

    for (recording_id, camera), stream_reqs in by_stream.items():
        h5_path = recording_h5_path(dataset_root, recording_id)
        if not h5_path.exists():
            for req in stream_reqs:
                failed += 1
                failed_msgs.append(f"{req.out_name()} :: missing h5 {h5_path}")
            continue

        tmp = None
        try:
            with h5py.File(h5_path, "r") as h5_file:
                tmp = write_camera_bytes_to_temp_mp4(h5_file, camera)
            vr = decord.VideoReader(tmp.name, ctx=decord.cpu(0))
            for req in stream_reqs:
                out_path = output_dir / req.out_name()
                processed += 1
                try:
                    if args.skip_existing and out_path.exists():
                        skipped_existing += 1
                    else:
                        if req.frame_index < 0 or req.frame_index >= len(vr):
                            raise IndexError(f"frame_index {req.frame_index} out of range len={len(vr)}")
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
                                "fps_requests_per_sec": round(processed / elapsed, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001
            for req in stream_reqs:
                failed += 1
                failed_msgs.append(f"{req.out_name()} :: {exc}")
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    summary = {
        "input_jsonl": str(input_jsonl),
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
