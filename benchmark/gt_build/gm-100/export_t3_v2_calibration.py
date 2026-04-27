from __future__ import annotations

import argparse
import csv
import json
import random
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import decord
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_t3_gt import (
    ROBOT_RAW_DIRECTIONS,
    T3_CONTEXT_OFFSETS,
    build_t3_candidate_windows_for_episode,
    load_episode_df,
    parse_offsets_csv,
    task_episode_paths,
)


decord.bridge.set_bridge("native")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]
CARD_BG = (245, 242, 236)
CARD_INK = (28, 28, 28)
CARD_MUTED = (92, 92, 92)
CARD_PANEL_BG = (252, 251, 247)
CARD_META_BG = (232, 238, 228)
CARD_META_BORDER = (113, 132, 106)
CARD_WIDTH = 1760
CARD_MARGIN = 40
FRAME_GUTTER = 24
FRAME_LABEL_H = 52

ANNOTATION_HEADERS = [
    "calibration_id",
    "task_id",
    "task_name",
    "episode_id",
    "frame_index",
    "frame_indices",
    "query_arm",
    "robot_direction_raw",
    "card_path",
    "human_label",
    "notes",
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_task_names(dataset_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for meta_path in sorted(dataset_root.glob("task_*/meta/tasks.jsonl")):
        task_id = meta_path.parts[-3]
        task_name = task_id
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                first = next((line for line in fh if line.strip()), "")
            if first:
                obj = json.loads(first)
                raw = str(obj.get("task", "")).strip()
                if raw:
                    task_name = raw.replace("-", " ")
        except Exception:
            pass
        out[task_id] = task_name
    return out


def video_path(dataset_root: Path, task_id: str, episode_id: int, camera: str) -> Path:
    return (
        dataset_root
        / task_id
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4"
    )


def extract_frame(vr: decord.VideoReader, frame_index: int) -> Image.Image:
    arr = vr[frame_index].asnumpy()
    return Image.fromarray(arr)


def fit_frame(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    canvas = Image.new("RGB", (target_w, target_h), CARD_PANEL_BG)
    src = img.convert("RGB")
    src.thumbnail((target_w, target_h))
    x = (target_w - src.width) // 2
    y = (target_h - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def visual_motion_score_from_frames(frames: list[Image.Image]) -> float:
    if len(frames) < 2:
        return 0.0
    gray = []
    for frame in frames:
        img = frame.convert("L").resize((160, 120))
        gray.append(np.asarray(img, dtype=np.float32))
    step_scores = [float(np.mean(np.abs(gray[idx + 1] - gray[idx]))) for idx in range(len(gray) - 1)]
    overall = float(np.mean(np.abs(gray[-1] - gray[0])))
    return float(sum(step_scores) + overall)


def make_card(
    out_path: Path,
    candidate: dict[str, Any],
    frames: list[Image.Image],
    task_name: str,
) -> None:
    title_font = load_font(26, bold=True)
    meta_font = load_font(18, bold=False)
    section_font = load_font(17, bold=True)
    body_font = load_font(15, bold=False)

    draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    panel_w = (CARD_WIDTH - 2 * CARD_MARGIN - 3 * FRAME_GUTTER) // 4
    panel_h = int(panel_w * 0.75)
    card_h = 820
    canvas = Image.new("RGB", (CARD_WIDTH, card_h), CARD_BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((CARD_MARGIN, 28), f"T3 V2 Calibration {candidate['calibration_id']}", font=title_font, fill=CARD_INK)
    draw.text(
        (CARD_MARGIN, 66),
        f"task={candidate['task_id']} | episode={candidate['episode_id']} | arm={candidate['query_arm']} | camera={candidate['camera']}",
        font=meta_font,
        fill=CARD_MUTED,
    )

    meta_top = 106
    meta_h = 72
    draw.rounded_rectangle(
        (CARD_MARGIN, meta_top, CARD_WIDTH - CARD_MARGIN, meta_top + meta_h),
        radius=18,
        fill=CARD_META_BG,
        outline=CARD_META_BORDER,
        width=2,
    )
    draw.text((CARD_MARGIN + 18, meta_top + 12), "Task Meta", font=section_font, fill=CARD_INK)
    draw.text((CARD_MARGIN + 18, meta_top + 38), task_name, font=meta_font, fill=CARD_INK)

    grid_top = meta_top + meta_h + 24
    labels = [f"t{off:+d}" if off != 0 else "t0" for off in candidate["frame_offsets"]]
    for idx, (img, label) in enumerate(zip(frames, labels)):
        x0 = CARD_MARGIN + idx * (panel_w + FRAME_GUTTER)
        y0 = grid_top
        x1 = x0 + panel_w
        y1 = y0 + panel_h + FRAME_LABEL_H
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=CARD_PANEL_BG, outline=CARD_INK, width=2)
        draw.rounded_rectangle((x0 + 10, y0 + 10, x1 - 10, y0 + 10 + FRAME_LABEL_H - 10), radius=12, fill=CARD_INK)
        tw = draw_probe.textbbox((0, 0), label, font=section_font)
        text_w = tw[2] - tw[0]
        text_h = tw[3] - tw[1]
        draw.text((x0 + (panel_w - text_w) / 2, y0 + 12 + (FRAME_LABEL_H - text_h) / 2 - 5), label, font=section_font, fill=(255, 255, 255))
        panel = fit_frame(img, panel_w - 26, panel_h - 20)
        canvas.paste(panel, (x0 + 13, y0 + FRAME_LABEL_H))

    question_top = grid_top + panel_h + FRAME_LABEL_H + 28
    draw.text((CARD_MARGIN, question_top), "Question", font=section_font, fill=CARD_INK)
    qtext = f"In which direction does the {candidate['query_arm']} robot arm primarily move in the scene?"
    draw.text((CARD_MARGIN, question_top + 28), qtext, font=body_font, fill=CARD_INK)
    draw.text(
        (CARD_MARGIN, question_top + 62),
        "Allowed labels: left / right / top / bottom / unclear",
        font=body_font,
        fill=CARD_MUTED,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=95)


def bucket_key(candidate: dict[str, Any]) -> str:
    return f"{candidate['query_arm']}|{candidate['robot_direction_raw']}"


def expected_bucket_keys() -> list[str]:
    return [f"{arm}|{raw_dir}" for arm in ("left", "right") for raw_dir in ROBOT_RAW_DIRECTIONS]


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(candidate.get("candidate_score", 0.0)),
        float(candidate.get("query_net_norm", 0.0)),
        -int(candidate.get("episode_id", 0)),
        -int(candidate.get("frame_index", 0)),
    )


def trim_bucket_pool(
    candidates: list[dict[str, Any]],
    max_items: int,
    per_task_cap: int,
) -> list[dict[str, Any]]:
    if len(candidates) <= max_items:
        return candidates

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_task[str(cand["task_id"])].append(cand)

    trimmed: list[dict[str, Any]] = []
    for task_id, arr in by_task.items():
        arr_sorted = sorted(arr, key=candidate_rank_key, reverse=True)
        trimmed.extend(arr_sorted[:per_task_cap])

    if len(trimmed) <= max_items:
        return sorted(trimmed, key=candidate_rank_key, reverse=True)
    return sorted(trimmed, key=candidate_rank_key, reverse=True)[:max_items]


def select_diverse_bucket(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in sorted(
        candidates,
        key=lambda x: (
            -float(x.get("visual_motion_score", 0.0)),
            -float(x["candidate_score"]),
            x["episode_id"],
            x["frame_index"],
        ),
    ):
        by_task[str(cand["task_id"])].append(cand)

    task_order = sorted(
        by_task,
        key=lambda tid: (
            -float(by_task[tid][0].get("visual_motion_score", 0.0)),
            -float(by_task[tid][0]["candidate_score"]),
            tid,
        ),
    )
    picked: list[dict[str, Any]] = []
    used = set()
    while len(picked) < target:
        progress = False
        for task_id in task_order:
            arr = by_task[task_id]
            while arr and arr[0]["calibration_id"] in used:
                arr.pop(0)
            if not arr:
                continue
            cand = arr.pop(0)
            picked.append(cand)
            used.add(cand["calibration_id"])
            progress = True
            if len(picked) >= target:
                break
        if not progress:
            break
    return picked[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export T3 v2 human-calibration package.")
    parser.add_argument("--dataset-root", default="/data/projects/GM-100/gm100-cobotmagic-lerobot")
    parser.add_argument(
        "--annotation-csv",
        default=str(Path(__file__).resolve().with_name("task_type_annotation.csv")),
    )
    parser.add_argument("--output-dir", default="/data/projects/GM-100/t3_v2_calibration_20260413")
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--context-offsets", default=",".join(str(x) for x in T3_CONTEXT_OFFSETS))
    parser.add_argument("--per-bucket", type=int, default=6)
    parser.add_argument("--net-min", type=float, default=0.02)
    parser.add_argument("--dominant-min", type=float, default=0.015)
    parser.add_argument("--purity-ratio", type=float, default=2.0)
    parser.add_argument("--segment-support-fraction", type=float, default=0.18)
    parser.add_argument("--segment-support-min", type=float, default=0.003)
    parser.add_argument("--min-consistent-segments", type=int, default=2)
    parser.add_argument("--contradictory-segment-tol", type=float, default=0.004)
    parser.add_argument("--secondary-arm-ratio-max", type=float, default=0.45)
    parser.add_argument("--secondary-arm-abs-max", type=float, default=0.012)
    parser.add_argument("--bimanual-dominance-ratio", type=float, default=1.75)
    parser.add_argument("--approach-buffer-frames", type=int, default=30)
    parser.add_argument("--no-contact-start-frame", type=int, default=20)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--limit-episodes-per-task", type=int, default=0)
    parser.add_argument("--bucket-pool-multiplier", type=int, default=12)
    parser.add_argument("--visual-score-pool-multiplier", type=int, default=6)
    parser.add_argument("--visual-motion-min", type=float, default=5.0)
    parser.add_argument(
        "--max-candidates-per-bucket",
        type=int,
        default=0,
        help="Optional hard cap for in-memory preselection pool per bucket. 0 means auto.",
    )
    parser.add_argument(
        "--per-task-cap-per-bucket",
        type=int,
        default=8,
        help="When trimming in-memory bucket pools, keep at most this many candidates per task.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    offsets = parse_offsets_csv(args.context_offsets)
    ann = pd.read_csv(args.annotation_csv)
    task_names = load_task_names(dataset_root)
    rng = random.Random(args.seed)

    tasks = ann["task_id"].tolist()
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    rng.shuffle(tasks)
    meta_map = {r["task_id"]: r for _, r in ann.iterrows()}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_pool_size = int(args.per_bucket) * max(1, int(args.bucket_pool_multiplier))
    max_candidates_per_bucket = int(args.max_candidates_per_bucket)
    if max_candidates_per_bucket <= 0:
        max_candidates_per_bucket = max(target_pool_size * 4, int(args.per_bucket) * int(args.visual_score_pool_multiplier) * 4)
    bucket_keys = expected_bucket_keys()
    num_candidates_seen = 0
    missing_video_episodes = 0
    calibration_index = 0
    stop_early = False
    for task_id in tasks:
        if task_id not in meta_map:
            continue
        eps = task_episode_paths(dataset_root, task_id)
        rng.shuffle(eps)
        if args.limit_episodes_per_task > 0:
            eps = eps[: args.limit_episodes_per_task]
        for parquet_path in eps:
            episode_id = int(parquet_path.stem.split("_")[-1])
            vpath = video_path(dataset_root, task_id, episode_id, args.camera)
            if not vpath.exists():
                missing_video_episodes += 1
                continue
            df = load_episode_df(parquet_path)
            candidates = build_t3_candidate_windows_for_episode(
                task_id=task_id,
                episode_id=episode_id,
                df=df,
                task_meta=meta_map[task_id],
                camera=args.camera,
                offsets=offsets,
                net_min=args.net_min,
                dominant_min=args.dominant_min,
                purity_ratio=args.purity_ratio,
                segment_support_fraction=args.segment_support_fraction,
                segment_support_min=args.segment_support_min,
                min_consistent_segments=args.min_consistent_segments,
                contradictory_segment_tol=args.contradictory_segment_tol,
                secondary_arm_ratio_max=args.secondary_arm_ratio_max,
                secondary_arm_abs_max=args.secondary_arm_abs_max,
                bimanual_dominance_ratio=args.bimanual_dominance_ratio,
                approach_buffer_frames=args.approach_buffer_frames,
                no_contact_start_frame=args.no_contact_start_frame,
            )
            for cand in candidates:
                obj = cand.to_json_dict()
                obj["calibration_id"] = f"c{calibration_index:04d}"
                obj["task_name"] = task_names.get(task_id, task_id)
                key = bucket_key(obj)
                grouped[key].append(obj)
                if len(grouped[key]) > max_candidates_per_bucket:
                    grouped[key] = trim_bucket_pool(
                        grouped[key],
                        max_items=max_candidates_per_bucket,
                        per_task_cap=max(1, int(args.per_task_cap_per_bucket)),
                    )
                calibration_index += 1
                num_candidates_seen += 1
            if all(len(grouped.get(key, [])) >= target_pool_size for key in bucket_keys):
                stop_early = True
                break
            del candidates
            del df
            gc.collect()
        if stop_early:
            break

    selected: list[dict[str, Any]] = []
    video_cache: dict[str, decord.VideoReader] = {}
    for query_arm in ("left", "right"):
        for raw_dir in ROBOT_RAW_DIRECTIONS:
            key = f"{query_arm}|{raw_dir}"
            bucket = sorted(
                grouped.get(key, []),
                key=lambda x: (-float(x["candidate_score"]), x["episode_id"], x["frame_index"]),
            )
            visual_pool_size = max(int(args.per_bucket), int(args.per_bucket) * int(args.visual_score_pool_multiplier))
            visual_pool = bucket[:visual_pool_size]
            scored_pool: list[dict[str, Any]] = []
            for cand in visual_pool:
                vpath = video_path(dataset_root, cand["task_id"], int(cand["episode_id"]), cand["camera"])
                if str(vpath) not in video_cache:
                    video_cache[str(vpath)] = decord.VideoReader(str(vpath), ctx=decord.cpu(0))
                vr = video_cache[str(vpath)]
                frames = [extract_frame(vr, int(fi)) for fi in cand["frame_indices"]]
                cand_copy = dict(cand)
                cand_copy["visual_motion_score"] = visual_motion_score_from_frames(frames)
                scored_pool.append(cand_copy)
            passing = [cand for cand in scored_pool if float(cand["visual_motion_score"]) >= float(args.visual_motion_min)]
            pool_for_selection = passing if len(passing) >= int(args.per_bucket) else scored_pool
            selected.extend(select_diverse_bucket(pool_for_selection, target=args.per_bucket))

    selected = sorted(selected, key=lambda x: x["calibration_id"])
    for idx, cand in enumerate(selected):
        cand["calibration_id"] = f"t3c{idx:03d}"

    with (out_dir / "calibration_candidates.jsonl").open("w", encoding="utf-8") as fh:
        for cand in selected:
            fh.write(json.dumps(cand, ensure_ascii=False) + "\n")

    annotation_rows: list[dict[str, Any]] = []
    for cand in selected:
        vpath = video_path(dataset_root, cand["task_id"], int(cand["episode_id"]), cand["camera"])
        if str(vpath) not in video_cache:
            video_cache[str(vpath)] = decord.VideoReader(str(vpath), ctx=decord.cpu(0))
        vr = video_cache[str(vpath)]
        frames = [extract_frame(vr, int(fi)) for fi in cand["frame_indices"]]
        card_path = cards_dir / f"{cand['calibration_id']}__{cand['task_id']}__ep{int(cand['episode_id'])}__f{int(cand['frame_index'])}.jpg"
        make_card(card_path, cand, frames, task_name=str(cand["task_name"]))
        annotation_rows.append(
            {
                "calibration_id": cand["calibration_id"],
                "task_id": cand["task_id"],
                "task_name": cand["task_name"],
                "episode_id": int(cand["episode_id"]),
                "frame_index": int(cand["frame_index"]),
                "frame_indices": json.dumps(cand["frame_indices"], ensure_ascii=False),
                "query_arm": cand["query_arm"],
                "robot_direction_raw": cand["robot_direction_raw"],
                "card_path": str(card_path.resolve()),
                "human_label": "",
                "notes": "",
            }
        )

    with (out_dir / "calibration_annotations.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_HEADERS)
        writer.writeheader()
        writer.writerows(annotation_rows)

    print(
        json.dumps(
            {
                "output_dir": str(out_dir.resolve()),
                "num_candidates_total": int(num_candidates_seen),
                "num_selected": len(selected),
                "per_bucket": int(args.per_bucket),
                "bucket_pool_multiplier": int(args.bucket_pool_multiplier),
                "visual_score_pool_multiplier": int(args.visual_score_pool_multiplier),
                "visual_motion_min": float(args.visual_motion_min),
                "buckets": {
                    key: int(sum(1 for row in selected if bucket_key(row) == key))
                    for key in bucket_keys
                },
                "bucket_pool_sizes": {
                    key: int(len(grouped.get(key, [])))
                    for key in bucket_keys
                },
                "context_offsets": list(offsets),
                "early_stop_reached": bool(stop_early),
                "missing_video_episodes": int(missing_video_episodes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
