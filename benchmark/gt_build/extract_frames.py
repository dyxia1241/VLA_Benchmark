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
    task_id: str
    episode_id: int
    frame_index: int
    camera: str
    suffix: str = ""

    def out_name(self) -> str:
        base = f"{self.task_id}_{self.episode_id}_{self.frame_index}_{self.camera}"
        if self.suffix:
            return f"{base}_{self.suffix}.jpg"
        return f"{base}.jpg"


def video_path(dataset_root: Path, req: FrameRequest) -> Path:
    return (
        dataset_root
        / req.task_id
        / "videos"
        / "chunk-000"
        / f"observation.images.{req.camera}"
        / f"episode_{req.episode_id:06d}.mp4"
    )


def parse_offsets_csv(text: str) -> list[int]:
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
    return vals


def parse_item_to_requests(
    item: dict,
    t3_context_radius: int,
    t3_offsets: list[int],
    t6_context: bool,
) -> list[FrameRequest]:
    task_id = item["task_id"]
    camera = item.get("camera", "camera_top")
    task_type = item.get("task_type", "")
    reqs: list[FrameRequest] = []

    def add_one(ep: int, fi: int, suffix: str = "") -> None:
        reqs.append(FrameRequest(task_id=task_id, episode_id=int(ep), frame_index=int(fi), camera=camera, suffix=suffix))

    if "frame_index" in item and "episode_id" in item:
        ep = int(item["episode_id"])
        fi = int(item["frame_index"])
        if task_type == "T6" and t6_context:
            for off in [-6, -3, 0, 3, 6]:
                fi2 = fi + off
                if fi2 < 0:
                    continue
                suffix = "t0" if off == 0 else f"t{off:+d}"
                add_one(ep, fi2, suffix=suffix)
        elif task_type == "T3" and t3_context_radius > 0:
            # T3 v2 uses a wider ordered context to make planar motion visually answerable.
            # If --t3-offsets is provided, it overrides radius-based offsets.
            offsets = t3_offsets if t3_offsets else list(range(-t3_context_radius, t3_context_radius))
            for off in offsets:
                fi2 = fi + off
                if fi2 < 0:
                    continue
                suffix = "t0" if off == 0 else f"t{off:+d}"
                add_one(ep, fi2, suffix=suffix)
        elif task_type in {"T4", "T_progress"} and isinstance(item.get("frame_indices"), list):
            # T4 consumes an ordered 4-frame window; T_progress v2 consumes an ordered 5-frame window.
            for fi2 in item["frame_indices"]:
                add_one(ep, int(fi2))
        else:
            add_one(ep, fi)

    # T2 pairs
    if "frame_A" in item:
        fa = item["frame_A"]
        add_one(int(fa["episode_id"]), int(fa["frame_index"]))
    if "frame_B" in item:
        fb = item["frame_B"]
        add_one(int(fb["episode_id"]), int(fb["frame_index"]))

    # Multi-frame tasks that use raw frame filenames without suffixes.
    if task_type in {"T_temporal", "T_binary"} and "frame_indices" in item and "episode_id" in item:
        ep = int(item["episode_id"])
        for fi in item["frame_indices"]:
            add_one(ep, int(fi))

    return reqs


def extract_one(vr: decord.VideoReader, vpath: Path, frame_index: int) -> Image.Image:
    if frame_index < 0 or frame_index >= len(vr):
        raise IndexError(f"frame_index out of range: {frame_index} (len={len(vr)}) for {vpath}")
    arr = vr[frame_index].asnumpy()
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract JPEG frames referenced by pilot_qa_raw.jsonl")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument("--input-jsonl", default="/data/projects/GM-100/benchmark/manual_checks_20260319/pilot_qa_raw.jsonl")
    parser.add_argument("--output-dir", default="/data/projects/GM-100/benchmark/manual_checks_20260319/pilot_frames")
    parser.add_argument("--t3-context-radius", type=int, default=0, help="For T3, additionally extract +/-N neighboring frames.")
    parser.add_argument(
        "--t3-offsets",
        default="-10,-5,0,5",
        help="Comma-separated T3 offsets when --t3-context-radius > 0; T3 v2 default is -10,-5,0,+5.",
    )
    parser.add_argument("--t6-context", action="store_true", help="For T6, extract fixed 5-frame context: t-6,t-3,t0,t+3,t+6.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction if output JPEG already exists. Useful for resume after interruption.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N frame requests.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t3_offsets = parse_offsets_csv(args.t3_offsets) if args.t3_context_radius > 0 else []

    reqs: list[FrameRequest] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            reqs.extend(
                parse_item_to_requests(
                    item,
                    t3_context_radius=args.t3_context_radius,
                    t3_offsets=t3_offsets,
                    t6_context=args.t6_context,
                )
            )

    # de-duplicate by full output filename
    dedup = {}
    for r in reqs:
        dedup[r.out_name()] = r
    reqs = list(dedup.values())

    by_video: dict[str, list[FrameRequest]] = defaultdict(list)
    for r in reqs:
        by_video[str(video_path(dataset_root, r))].append(r)

    ok = 0
    skipped_existing = 0
    failed = 0
    failed_msgs: list[str] = []
    processed = 0
    started = time.time()
    progress_every = max(1, int(args.progress_every))

    for vpath_str, video_reqs in by_video.items():
        vpath = Path(vpath_str)
        try:
            if not vpath.exists():
                raise FileNotFoundError(f"video missing: {vpath}")
            vr = decord.VideoReader(str(vpath), ctx=decord.cpu(0))
            for r in video_reqs:
                out_path = out_dir / r.out_name()
                processed += 1
                try:
                    if args.skip_existing and out_path.exists():
                        skipped_existing += 1
                    else:
                        img = extract_one(vr, vpath, r.frame_index)
                        img.save(out_path, format="JPEG", quality=95)
                        ok += 1
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    failed_msgs.append(f"{r.out_name()} :: {e}")

                if processed % progress_every == 0:
                    elapsed = max(1e-6, time.time() - started)
                    rate = processed / elapsed
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "total": len(reqs),
                                "ok": ok,
                                "skipped_existing": skipped_existing,
                                "failed": failed,
                                "fps_requests_per_sec": round(rate, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            for r in video_reqs:
                failed += 1
                failed_msgs.append(f"{r.out_name()} :: {e}")

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_dir": str(out_dir),
        "t3_offsets": t3_offsets,
        "num_requests_deduped": len(reqs),
        "num_ok": ok,
        "num_skipped_existing": skipped_existing,
        "num_failed": failed,
        "num_processed_total": processed,
        "num_unique_videos": len(by_video),
    }
    (out_dir / "extract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_msgs:
        (out_dir / "extract_failures.log").write_text("\n".join(failed_msgs) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
