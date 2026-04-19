#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path


META_RE = re.compile(r'^(RH20T_cfg2/)?(task_\d+_user_\d+_scene_\d+_cfg_\d+)/metadata\.json$')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Select RH20T failure scenes directly from cfg2 archive metadata.')
    p.add_argument('--archive', default='/data/projects/GM-100/RH20T_cfg2.tar.gz')
    p.add_argument('--success-scene-list-json', default='/data/projects/GM-100/benchmark/rh20t_cfg2_expanded_v0/selected_scenes.json')
    p.add_argument('--output-json', default='/data/projects/GM-100/benchmark/rh20t_cfg2_failure_v0/selected_failure_scenes.json')
    p.add_argument('--max-fail-per-task', type=int, default=3)
    p.add_argument('--max-total-fail', type=int, default=229)
    p.add_argument('--max-calib-quality', type=int, default=3)
    return p.parse_args()


def load_success_tasks(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding='utf-8'))
    rows = obj.get('selected_scenes', [])
    return {str(r.get('task_id')) for r in rows if isinstance(r, dict) and r.get('task_id')}


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    success_tasks = load_success_tasks(Path(args.success_scene_list_json))
    by_task: dict[str, list[dict]] = defaultdict(list)

    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            mm = META_RE.match(member.name)
            if not mm:
                continue
            scene_dir = mm.group(2)
            task_id = scene_dir.split('_user_')[0]
            if task_id not in success_tasks:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            meta = json.load(fh)
            rating = int(meta.get('rating', -99))
            calib_quality = int(meta.get('calib_quality', 99))
            if rating > 1:
                continue
            if calib_quality > int(args.max_calib_quality):
                continue
            by_task[task_id].append(
                {
                    'scene_dir': scene_dir,
                    'task_id': task_id,
                    'rating': rating,
                    'calib_quality': calib_quality,
                }
            )

    selected: list[dict] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda r: (r['calib_quality'], r['rating'], r['scene_dir']))
        selected.extend(rows[: int(args.max_fail_per_task)])

    if len(selected) > int(args.max_total_fail):
        selected = selected[: int(args.max_total_fail)]

    out = {
        'archive': str(archive),
        'source_success_scene_list_json': str(args.success_scene_list_json),
        'num_selected_scenes': len(selected),
        'max_fail_per_task': int(args.max_fail_per_task),
        'selected_scenes': selected,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output_json': str(out_path), 'num_selected_scenes': len(selected)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
