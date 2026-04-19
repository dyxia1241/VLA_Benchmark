#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
GT_BUILD_DIR = THIS_DIR.parent
import sys
for _path in (THIS_DIR, GT_BUILD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reassemble_utils import shuffled_binary_choices
from rh20t_utils import DEFAULT_EXTRACTED_ROOT, PRIMARY_CAMERA, load_task_catalog


T7_RELATIVE_PROGRESS_VALUES = (0.10, 0.18, 0.26, 0.34)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build RH20T T7 failure-detection suite from success+failure scenes.')
    p.add_argument('--extracted-root', default=str(DEFAULT_EXTRACTED_ROOT))
    p.add_argument('--success-scene-list-json', default='/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/selected_scenes.json')
    p.add_argument('--failure-scene-list-json', default='/data/projects/GM-100/benchmark/rh20t_cfg2_failure_v0/selected_failure_scenes.json')
    p.add_argument('--output-jsonl', default='/data/projects/GM-100/benchmark/rh20t_cfg2_failure_v0/t7_gt_items.jsonl')
    p.add_argument('--camera', default=PRIMARY_CAMERA)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-success-per-scene', type=int, default=6)
    p.add_argument('--max-fail-per-scene', type=int, default=10)
    p.add_argument('--target-total', type=int, default=4000)
    return p.parse_args()


def load_scene_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding='utf-8'))
    rows = obj.get('selected_scenes', [])
    return [r for r in rows if isinstance(r, dict) and r.get('scene_dir')]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def frame_windows(n_rows: int) -> list[list[int]]:
    windows: list[list[int]] = []
    if n_rows < 24:
        return windows

    early_limit = max(18, int(round((n_rows - 1) * 0.42)))
    sub_gap = max(5, min(30, int(round(n_rows * 0.03))))
    start_min = max(0, int(round((n_rows - 1) * 0.02)))
    start_max = early_limit - 3 * sub_gap
    step = max(6, int(round(n_rows * 0.03)))

    if start_max < start_min:
        fallback_sets = (
            (0.05, 0.10, 0.15, 0.20),
            T7_RELATIVE_PROGRESS_VALUES,
            (0.14, 0.22, 0.30, 0.38),
        )
        for values in fallback_sets:
            idxs = [int(round(v * max(0, n_rows - 1))) for v in values]
            if min(idxs) < 0 or max(idxs) >= n_rows:
                continue
            if len(set(idxs)) < 4:
                continue
            windows.append(idxs)
    else:
        for start in range(start_min, start_max + 1, step):
            idxs = [int(start + i * sub_gap) for i in range(4)]
            if min(idxs) < 0 or max(idxs) >= n_rows:
                continue
            if len(set(idxs)) < 4:
                continue
            windows.append(idxs)

    dedup = []
    seen = set()
    for w in windows:
        key = tuple(w)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(w)
    return dedup


def evenly_pick_windows(windows: list[list[int]], cap: int) -> list[list[int]]:
    if cap <= 0 or len(windows) <= cap:
        return list(windows)
    if cap == 1:
        return [windows[len(windows) // 2]]
    out: list[list[int]] = []
    last_idx = len(windows) - 1
    for i in range(cap):
        pos = int(round(i * last_idx / max(1, cap - 1)))
        out.append(windows[pos])
    return out


def make_item(*, scene_dir: str, task_id: str, task_description: str, camera: str, frame_indices: list[int], success: bool, rating: int, calib_quality: int, scene_length: int, window_id: int, rng: random.Random) -> dict:
    choices, answer = shuffled_binary_choices(
        success,
        yes_text='Yes — this episode will succeed',
        no_text='No — this episode will fail',
        rng=rng,
    )
    center = int(frame_indices[len(frame_indices) // 2])
    return {
        'dataset': 'RH20T',
        'task_type': 'T7',
        'task_id': task_id,
        'recording_id': scene_dir,
        'scene_dir': scene_dir,
        'camera': camera,
        'arm_type': 'single_arm',
        'task_meta_description': task_description,
        'frame_index': center,
        'frame_indices': [int(x) for x in frame_indices],
        'question': 'Based on these early frames of the episode, will this manipulation attempt eventually succeed?',
        'choices': choices,
        'answer': answer,
        'scene_success': bool(success),
        'rating': int(rating),
        'calib_quality': int(calib_quality),
        'frame_sampling': 'relative_progress_early',
        'relative_progresses': [round(float(x) / max(1, scene_length - 1), 4) for x in frame_indices],
        'scene_length': int(scene_length),
        'window_id': int(window_id),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    extracted_root = Path(args.extracted_root)
    catalog = load_task_catalog()
    success_rows = load_scene_rows(Path(args.success_scene_list_json))
    fail_rows = load_scene_rows(Path(args.failure_scene_list_json))

    items: list[dict] = []
    counts = Counter()
    skipped = Counter()
    skipped_examples: list[dict[str, Any]] = []

    def build_from_rows(rows: list[dict[str, Any]], success: bool, per_scene_cap: int) -> None:
        for row in rows:
            scene_dir = str(row['scene_dir'])
            task_id = str(row['task_id'])
            scene_root = extracted_root / scene_dir
            if not scene_root.exists():
                skipped['missing_scene_root'] += 1
                if len(skipped_examples) < 16:
                    skipped_examples.append({'scene_dir': scene_dir, 'reason': 'missing_scene_root'})
                continue
            color_mp4 = scene_root / f'cam_{str(args.camera)}' / 'color.mp4'
            if not color_mp4.exists():
                skipped['missing_color_mp4'] += 1
                if len(skipped_examples) < 16:
                    skipped_examples.append({'scene_dir': scene_dir, 'reason': 'missing_color_mp4'})
                continue
            tcp_path = scene_root / 'transformed' / 'tcp_base.npy'
            if not tcp_path.exists():
                skipped['missing_tcp_base'] += 1
                if len(skipped_examples) < 16:
                    skipped_examples.append({'scene_dir': scene_dir, 'reason': 'missing_tcp_base'})
                continue
            meta = json.loads((scene_root / 'metadata.json').read_text(encoding='utf-8'))
            rating = int(meta.get('rating', row.get('rating', -1)))
            calib_quality = int(meta.get('calib_quality', row.get('calib_quality', -1)))
            tcp_obj = np.load(tcp_path, allow_pickle=True).item()
            if str(args.camera) not in tcp_obj:
                skipped['missing_camera_in_tcp_base'] += 1
                if len(skipped_examples) < 16:
                    skipped_examples.append(
                        {
                            'scene_dir': scene_dir,
                            'reason': 'missing_camera_in_tcp_base',
                            'available_cameras': sorted(map(str, tcp_obj.keys())),
                        }
                    )
                continue
            tcp_rows = tcp_obj[str(args.camera)]
            n_rows = len(tcp_rows)
            windows = evenly_pick_windows(frame_windows(n_rows), max(1, int(per_scene_cap)))
            task_description = catalog.get(task_id, {}).get('task_description_english', task_id)
            for window_idx, w in enumerate(windows, start=1):
                items.append(
                    make_item(
                        scene_dir=scene_dir,
                        task_id=task_id,
                        task_description=task_description,
                        camera=str(args.camera),
                        frame_indices=w,
                        success=success,
                        rating=rating,
                        calib_quality=calib_quality,
                        scene_length=n_rows,
                        window_id=window_idx,
                        rng=rng,
                    )
                )
                counts['success' if success else 'failure'] += 1

    build_from_rows(success_rows, True, int(args.max_success_per_scene))
    build_from_rows(fail_rows, False, int(args.max_fail_per_scene))

    rng.shuffle(items)
    if int(args.target_total) > 0 and len(items) > int(args.target_total):
        fail_items = [x for x in items if not x['scene_success']]
        succ_items = [x for x in items if x['scene_success']]
        target_fail = min(len(fail_items), int(args.target_total) // 2)
        target_succ = min(len(succ_items), int(args.target_total) - target_fail)
        items = fail_items[:target_fail] + succ_items[:target_succ]
        rng.shuffle(items)

    write_jsonl(Path(args.output_jsonl), items)
    summary = {
        'output_jsonl': str(args.output_jsonl),
        'total_items': len(items),
        'success_items': sum(1 for x in items if x['scene_success']),
        'failure_items': sum(1 for x in items if not x['scene_success']),
        'unique_scenes': len({x['scene_dir'] for x in items}),
        'unique_tasks': len({x['task_id'] for x in items}),
        'skipped_scenes': dict(skipped),
        'skipped_examples': skipped_examples,
    }
    Path(str(args.output_jsonl).replace('.jsonl', '_summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
