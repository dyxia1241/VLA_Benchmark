from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import decord


decord.bridge.set_bridge('native')


TYPE_TO_FILE = {
    'T1': 't1_gt_items.jsonl',
    'T2': 't2_gt_items.jsonl',
    'T3': 't3_gt_items.jsonl',
    'T4': 't4_gt_items.jsonl',
    'T6': 't6_gt_items.jsonl',
    'T_temporal': 't_temporal_gt_items.jsonl',
    'T_binary': 't_binary_gt_items.jsonl',
    'T_progress': 't_progress_gt_items.jsonl',
}

DEFAULT_TARGETS = {
    'T1': 3000,
    'T2': 1000,
    'T3': 2500,
    'T4': 1500,
    'T6': 1500,
    'T_temporal': 2000,
    'T_binary': 1500,
    'T_progress': 2500,
}

DEFAULT_PER_TASK_CAPS = {
    'T1': 30,
    'T2': 10,
    'T3': 25,
    'T4': 70,
    'T6': 15,
    'T_temporal': 20,
    'T_binary': 15,
    'T_progress': 25,
}

DEFAULT_BOUNDARY_FILTER_TASK_TYPES = ('T3', 'T4', 'T6', 'T_temporal', 'T_binary')
DEFAULT_HEAD_GUARD_FRAMES = 90
DEFAULT_TAIL_GUARD_FRAMES = 90
T3_CONTEXT_OFFSETS = (-10, -5, 0, 5)
T6_CONTEXT_OFFSETS = (-6, -3, 0, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build balanced sampled QA set from 8 GT JSONL files.')
    parser.add_argument(
        '--dataset-root',
        default='/data/projects/GM-100/gm100-cobotmagic-lerobot',
        help='Root of GM-100 dataset with task_*/videos/chunk-000/observation.images.camera_top/*.mp4',
    )
    parser.add_argument(
        '--gt-dir',
        default='/data/projects/GM-100/benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2',
        help='Directory containing 8 task GT JSONL files.',
    )
    parser.add_argument(
        '--output-jsonl',
        default='/data/projects/GM-100/benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/benchmark_v1_curated.jsonl',
    )
    parser.add_argument(
        '--output-summary-json',
        default='/data/projects/GM-100/benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/benchmark_v1_curated_healthcheck.json',
    )
    parser.add_argument(
        '--output-per-type-dir',
        default='/data/projects/GM-100/benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/benchmark_v1_curated_by_type',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--boundary-filter-task-types',
        default=','.join(DEFAULT_BOUNDARY_FILTER_TASK_TYPES),
        help='Comma-separated task types that use boundary-based top-view filtering.',
    )
    parser.add_argument(
        '--head-guard-frames',
        type=int,
        default=DEFAULT_HEAD_GUARD_FRAMES,
        help='Discard items whose relevant frame(s) fall within the first N frames of an episode.',
    )
    parser.add_argument(
        '--tail-guard-frames',
        type=int,
        default=DEFAULT_TAIL_GUARD_FRAMES,
        help='Discard items whose relevant frame(s) fall within the last N frames of an episode.',
    )

    for t, v in DEFAULT_TARGETS.items():
        parser.add_argument(f'--target-{t.lower()}', type=int, default=v)
    for t, v in DEFAULT_PER_TASK_CAPS.items():
        parser.add_argument(f'--cap-{t.lower()}', type=int, default=v)
    return parser.parse_args()


def parse_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def parse_task_types_csv(text: str) -> set[str]:
    return {x.strip() for x in str(text).split(',') if x.strip()}


def extract_episode_ids(item: dict) -> list[int]:
    ids: list[int] = []

    if 'episode_id' in item:
        eid = parse_int(item.get('episode_id'))
        if eid is not None:
            ids.append(eid)

    for k in ['frame_A', 'frame_B', 'frame_X', 'frame_Y', 'frame_Z']:
        v = item.get(k)
        if isinstance(v, dict):
            eid = parse_int(v.get('episode_id'))
            if eid is not None:
                ids.append(eid)

    out: list[int] = []
    seen = set()
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def video_path(dataset_root: Path, task_id: str, episode_id: int) -> Path:
    return (
        dataset_root
        / task_id
        / 'videos'
        / 'chunk-000'
        / 'observation.images.camera_top'
        / f'episode_{episode_id:06d}.mp4'
    )


def video_exists(dataset_root: Path, task_id: str, episode_id: int, cache: dict[tuple[str, int], bool]) -> bool:
    key = (task_id, episode_id)
    if key in cache:
        return cache[key]

    ok = video_path(dataset_root, task_id, episode_id).exists()
    cache[key] = ok
    return ok


def video_num_frames(
    dataset_root: Path,
    task_id: str,
    episode_id: int,
    cache: dict[tuple[str, int], int],
) -> int:
    key = (task_id, episode_id)
    if key in cache:
        return cache[key]

    vpath = video_path(dataset_root, task_id, episode_id)
    if not vpath.exists():
        cache[key] = 0
        return 0

    vr = decord.VideoReader(str(vpath), ctx=decord.cpu(0))
    n_frames = int(len(vr))
    cache[key] = n_frames
    return n_frames


def sample_with_task_cap(
    entries: list[dict],
    target: int,
    per_task_cap: int,
    rng: random.Random,
) -> list[dict]:
    if target <= 0 or not entries:
        return []

    by_task: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        by_task[entry['task_id']].append(i)

    provisional: list[int] = []
    selected_set: set[int] = set()

    for idxs in by_task.values():
        local = idxs[:]
        rng.shuffle(local)
        k = min(len(local), per_task_cap)
        pick = local[:k]
        provisional.extend(pick)
        selected_set.update(pick)

    if len(provisional) > target:
        return [entries[i] for i in rng.sample(provisional, target)]

    if len(provisional) < target:
        remain = [i for i in range(len(entries)) if i not in selected_set]
        need = min(target - len(provisional), len(remain))
        if need > 0:
            provisional.extend(rng.sample(remain, need))

    return [entries[i] for i in provisional]


def item_boundary_frame_indices(task_type: str, obj: dict) -> list[int]:
    if task_type == 'T3':
        center = parse_int(obj.get('frame_index'))
        if center is None:
            return []
        return [center + off for off in T3_CONTEXT_OFFSETS]

    if task_type == 'T4':
        frame_indices = obj.get('frame_indices')
        if isinstance(frame_indices, list) and frame_indices:
            return [int(x) for x in frame_indices]
        center = parse_int(obj.get('frame_index'))
        return [center] if center is not None else []

    if task_type == 'T6':
        frame_indices = obj.get('frame_indices')
        if isinstance(frame_indices, list) and frame_indices:
            return [int(x) for x in frame_indices]
        center = parse_int(obj.get('frame_index'))
        if center is None:
            return []
        return [center + off for off in T6_CONTEXT_OFFSETS]

    if task_type == 'T_progress':
        frame_indices = obj.get('frame_indices')
        if isinstance(frame_indices, list) and frame_indices:
            return [int(x) for x in frame_indices]
        center = parse_int(obj.get('frame_index'))
        return [center] if center is not None else []

    if task_type in {'T_temporal', 'T_binary'}:
        frame_indices = obj.get('frame_indices')
        if isinstance(frame_indices, list):
            return [int(x) for x in frame_indices]
        return []

    return []


def passes_boundary_guard(
    task_type: str,
    obj: dict,
    num_frames: int,
    head_guard_frames: int,
    tail_guard_frames: int,
    boundary_filter_task_types: set[str],
) -> bool:
    if task_type not in boundary_filter_task_types:
        return True
    if head_guard_frames <= 0 and tail_guard_frames <= 0:
        return True
    if num_frames <= 0 or num_frames <= head_guard_frames + tail_guard_frames:
        return False

    frame_indices = item_boundary_frame_indices(task_type, obj)
    if not frame_indices:
        return False

    lo = int(head_guard_frames)
    hi = int(num_frames - tail_guard_frames)
    return all(lo <= int(fi) < hi for fi in frame_indices)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    gt_dir = Path(args.gt_dir)
    boundary_filter_task_types = parse_task_types_csv(args.boundary_filter_task_types)

    targets = {
        'T1': args.target_t1,
        'T2': args.target_t2,
        'T3': args.target_t3,
        'T4': args.target_t4,
        'T6': args.target_t6,
        'T_temporal': args.target_t_temporal,
        'T_binary': args.target_t_binary,
        'T_progress': args.target_t_progress,
    }
    caps = {
        'T1': args.cap_t1,
        'T2': args.cap_t2,
        'T3': args.cap_t3,
        'T4': args.cap_t4,
        'T6': args.cap_t6,
        'T_temporal': args.cap_t_temporal,
        'T_binary': args.cap_t_binary,
        'T_progress': args.cap_t_progress,
    }

    video_exists_cache: dict[tuple[str, int], bool] = {}
    video_length_cache: dict[tuple[str, int], int] = {}
    rng_master = random.Random(args.seed)

    raw_counts: dict[str, int] = {}
    after_video_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}
    sampled_counts: dict[str, int] = {}
    missing_video_counts: dict[str, int] = {}
    missing_episode_counts: dict[str, int] = {}
    boundary_removed_counts: dict[str, int] = {}

    task_counts_after_filter: dict[str, Counter] = {}
    task_counts_after_sample: dict[str, Counter] = {}

    answer_dist_filtered: dict[str, Counter] = {}
    answer_dist_sampled: dict[str, Counter] = {}
    t4_label_filtered = Counter()
    t4_label_sampled = Counter()
    temporal_perm_filtered = Counter()
    temporal_perm_sampled = Counter()
    t6_label_filtered = Counter()
    t6_label_sampled = Counter()

    selected_by_type: dict[str, list[dict]] = {}

    for i, (task_type, fname) in enumerate(TYPE_TO_FILE.items()):
        path = gt_dir / fname
        if not path.exists():
            raise FileNotFoundError(f'Missing GT file for {task_type}: {path}')

        entries: list[dict] = []
        raw = 0
        after_video = 0
        miss_video = 0
        miss_epi = 0
        boundary_removed = 0

        task_counter = Counter()
        ans_counter = Counter()

        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw += 1
                obj = json.loads(line)

                task_id = str(obj.get('task_id', ''))
                if not task_id:
                    miss_epi += 1
                    continue

                episode_ids = extract_episode_ids(obj)
                if not episode_ids:
                    miss_epi += 1
                    continue

                if not all(video_exists(dataset_root, task_id, eid, video_exists_cache) for eid in episode_ids):
                    miss_video += 1
                    continue
                after_video += 1

                episode_id = episode_ids[0]
                n_frames = video_num_frames(dataset_root, task_id, episode_id, video_length_cache)
                if not passes_boundary_guard(
                    task_type=task_type,
                    obj=obj,
                    num_frames=n_frames,
                    head_guard_frames=args.head_guard_frames,
                    tail_guard_frames=args.tail_guard_frames,
                    boundary_filter_task_types=boundary_filter_task_types,
                ):
                    boundary_removed += 1
                    continue

                answer = obj.get('answer')
                label = None
                if task_type == 'T4':
                    label = str(obj.get('label_name', 'unknown'))
                elif task_type == 'T6':
                    label = str(obj.get('speed_level_label', obj.get('speed_change_label', 'unknown')))

                entry = {
                    'task_id': task_id,
                    'answer': str(answer) if answer is not None else None,
                    'label': label,
                    'raw': line,
                }
                entries.append(entry)
                task_counter[task_id] += 1
                if answer is not None:
                    ans_counter[str(answer)] += 1
                if task_type == 'T4':
                    t4_label_filtered[label] += 1
                if task_type == 'T_temporal' and answer is not None:
                    temporal_perm_filtered[str(answer)] += 1
                if task_type == 'T6' and label is not None:
                    t6_label_filtered[label] += 1

        raw_counts[task_type] = raw
        after_video_counts[task_type] = after_video
        filtered_counts[task_type] = len(entries)
        missing_video_counts[task_type] = miss_video
        missing_episode_counts[task_type] = miss_epi
        boundary_removed_counts[task_type] = boundary_removed
        task_counts_after_filter[task_type] = task_counter
        answer_dist_filtered[task_type] = ans_counter

        rng = random.Random(args.seed + 1000 * (i + 1))
        selected = sample_with_task_cap(entries, target=targets[task_type], per_task_cap=caps[task_type], rng=rng)
        selected_by_type[task_type] = selected
        sampled_counts[task_type] = len(selected)

        s_task_counter = Counter(e['task_id'] for e in selected)
        s_ans_counter = Counter(e['answer'] for e in selected if e['answer'] is not None)
        task_counts_after_sample[task_type] = s_task_counter
        answer_dist_sampled[task_type] = s_ans_counter

        if task_type == 'T4':
            for entry in selected:
                t4_label_sampled[entry['label']] += 1
        if task_type == 'T_temporal':
            for entry in selected:
                if entry['answer'] is not None:
                    temporal_perm_sampled[entry['answer']] += 1
        if task_type == 'T6':
            for entry in selected:
                if entry['label'] is not None:
                    t6_label_sampled[entry['label']] += 1

    output_per_type_dir = Path(args.output_per_type_dir)
    output_per_type_dir.mkdir(parents=True, exist_ok=True)

    all_selected_lines: list[str] = []
    for task_type in TYPE_TO_FILE:
        arr = selected_by_type[task_type]
        rng_master.shuffle(arr)
        out_type = output_per_type_dir / f'{task_type}.jsonl'
        with out_type.open('w', encoding='utf-8') as f:
            for entry in arr:
                f.write(entry['raw'] + '\n')
                all_selected_lines.append(entry['raw'])

    rng_master.shuffle(all_selected_lines)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open('w', encoding='utf-8') as f:
        for line in all_selected_lines:
            f.write(line + '\n')
    all_task_ids = sorted(
        {task_id for counter in task_counts_after_filter.values() for task_id in counter.keys()}
        | {task_id for counter in task_counts_after_sample.values() for task_id in counter.keys()}
    )

    zero_after_filter = {
        task_type: sorted([task_id for task_id in all_task_ids if task_counts_after_filter[task_type].get(task_id, 0) == 0])
        for task_type in TYPE_TO_FILE
    }
    zero_after_sample = {
        task_type: sorted([task_id for task_id in all_task_ids if task_counts_after_sample[task_type].get(task_id, 0) == 0])
        for task_type in TYPE_TO_FILE
    }

    summary = {
        'config': {
            'dataset_root': str(dataset_root),
            'gt_dir': str(gt_dir),
            'output_jsonl': str(output_jsonl),
            'output_per_type_dir': str(output_per_type_dir),
            'seed': int(args.seed),
            'targets': targets,
            'per_task_caps': caps,
            'type_to_file': TYPE_TO_FILE,
            'boundary_filter_task_types': sorted(boundary_filter_task_types),
            'head_guard_frames': int(args.head_guard_frames),
            'tail_guard_frames': int(args.tail_guard_frames),
            't3_context_offsets': list(T3_CONTEXT_OFFSETS),
            't6_context_offsets': list(T6_CONTEXT_OFFSETS),
        },
        'counts': {
            'raw_by_type': raw_counts,
            'after_video_filter_by_type': after_video_counts,
            'removed_by_boundary_guard_by_type': boundary_removed_counts,
            'after_boundary_filter_by_type': filtered_counts,
            'sampled_by_type': sampled_counts,
            'target_by_type': targets,
            'shortfall_by_type': {task_type: int(max(0, targets[task_type] - sampled_counts[task_type])) for task_type in TYPE_TO_FILE},
            'missing_video_by_type': missing_video_counts,
            'missing_episode_id_by_type': missing_episode_counts,
            'total_raw': int(sum(raw_counts.values())),
            'total_after_video_filter': int(sum(after_video_counts.values())),
            'total_removed_by_boundary_guard': int(sum(boundary_removed_counts.values())),
            'total_after_boundary_filter': int(sum(filtered_counts.values())),
            'total_sampled': int(sum(sampled_counts.values())),
            'total_target': int(sum(targets.values())),
        },
        'coverage': {
            'tasks_with_items_after_filter_by_type': {task_type: int(len(task_counts_after_filter[task_type])) for task_type in TYPE_TO_FILE},
            'tasks_with_items_after_sample_by_type': {task_type: int(len(task_counts_after_sample[task_type])) for task_type in TYPE_TO_FILE},
            'zero_tasks_after_filter_by_type': {task_type: zero_after_filter[task_type] for task_type in TYPE_TO_FILE},
            'zero_tasks_after_sample_by_type': {task_type: zero_after_sample[task_type] for task_type in TYPE_TO_FILE},
        },
        'distributions': {
            'answer_filtered_by_type': {task_type: dict(answer_dist_filtered[task_type]) for task_type in TYPE_TO_FILE},
            'answer_sampled_by_type': {task_type: dict(answer_dist_sampled[task_type]) for task_type in TYPE_TO_FILE},
            't4_label_filtered': dict(t4_label_filtered),
            't4_label_sampled': dict(t4_label_sampled),
            'temporal_perm_filtered': dict(temporal_perm_filtered),
            'temporal_perm_sampled': dict(temporal_perm_sampled),
            't6_speed_label_filtered': dict(t6_label_filtered),
            't6_speed_label_sampled': dict(t6_label_sampled),
        },
    }

    output_summary = Path(args.output_summary_json)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
