#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
SELECTED_ROOT="${ROOT}/aist-bimanip/selected20"
PILOT_DIR="${ROOT}/benchmark/aist_pilot_v0"
FULL_DIR="${ROOT}/benchmark/aist_benchmark_v0"
FULL_FRAME_DIR="${ROOT}/benchmark/aist_benchmark_v0_frames"
LOG_DIR="${ROOT}/aist-bimanip/logs"
CAMERA="${AIST_CAMERA:-cam_high}"
PER_TASK_TYPE="${AIST_PILOT_PER_TYPE:-30}"
T8_TARGET="${AIST_T8_TARGET:-2200}"

mkdir -p "${LOG_DIR}"

echo "[AIST] rebuild start camera=${CAMERA} pilot_per_type=${PER_TASK_TYPE} t8_target=${T8_TARGET} $(date -Is)"

rm -rf "${PILOT_DIR}/frames" "${PILOT_DIR}/cards" "${PILOT_DIR}/gt"
rm -f "${PILOT_DIR}/aist_pilot_v0.jsonl" "${PILOT_DIR}/aist_pilot_v0_summary.json"
rm -rf "${FULL_DIR}" "${FULL_FRAME_DIR}"

python "${ROOT}/benchmark/gt_build/aist/build_aist_pilot_suite.py" \
  --selected-root "${SELECTED_ROOT}" \
  --output-dir "${PILOT_DIR}" \
  --per-task-type "${PER_TASK_TYPE}" \
  --camera "${CAMERA}"

python "${ROOT}/benchmark/gt_build/aist/extract_aist_frames.py" \
  --input-jsonl "${PILOT_DIR}/aist_pilot_v0.jsonl" \
  --output-dir "${PILOT_DIR}/frames"

python "${ROOT}/benchmark/gt_build/aist/render_aist_pilot_cards.py" \
  --input-jsonl "${PILOT_DIR}/aist_pilot_v0.jsonl" \
  --frame-dir "${PILOT_DIR}/frames" \
  --output-dir "${PILOT_DIR}"

python "${ROOT}/benchmark/gt_build/aist/build_aist_curated_benchmark.py" \
  --selected-root "${SELECTED_ROOT}" \
  --output-dir "${FULL_DIR}" \
  --camera "${CAMERA}" \
  --quota all \
  --t8-target "${T8_TARGET}"

python "${ROOT}/benchmark/gt_build/aist/extract_aist_frames.py" \
  --input-jsonl "${FULL_DIR}/aist_benchmark_v0_curated.jsonl" \
  --output-dir "${FULL_FRAME_DIR}"

python - <<'PY' "${SELECTED_ROOT}"
import json
from pathlib import Path
import re

selected_root = Path(__import__("sys").argv[1])
rows = []
for task_dir in sorted(selected_root.glob("task_*")):
    children = [p for p in sorted(task_dir.iterdir()) if p.is_dir()]
    if not children:
        continue
    task_name_dir = children[0]
    eps = sorted(task_name_dir.glob("episode_*.hdf5"))
    ep_ids = []
    for ep in eps:
        m = re.match(r"episode_(\d+)\.hdf5$", ep.name)
        if m:
            ep_ids.append(int(m.group(1)))
    rows.append({
        "task_id": task_dir.name,
        "task_name": task_name_dir.name,
        "found": len(eps),
        "kept": len(eps),
        "kept_episode_ids": ep_ids,
    })

(selected_root / "selection_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"selection_tasks": len(rows)}, ensure_ascii=False))
PY

echo "[AIST] rebuild done $(date -Is)"
