#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
LOG_DIR="${ROOT}/aist-bimanip/logs"
RAW_DIR="${ROOT}/aist-bimanip/raw"
SELECTED_ROOT="${ROOT}/aist-bimanip/selected20"

echo "== AIST Queue Status =="
date -Is
echo

echo "-- running processes --"
ps -ef | rg 'run_aist_selected10_queue|download_aist_task_watchdog|timeout 300s wget|wget -c .*aist-bimanip|wget -c https://www.dropbox.com/.*/AIST|insert_rod_board|handover_blue_block|insert_usb_cable|open_toolbox' || true
echo

echo "-- raw zips --"
find "${RAW_DIR}" -maxdepth 1 -type f -printf '%f\t%s\n' | sort || true
echo

echo "-- selected20 episode counts --"
python - <<'PY' "${SELECTED_ROOT}"
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
rows = []
for task_dir in sorted(root.glob("task_*")):
    subdirs = [p for p in sorted(task_dir.iterdir()) if p.is_dir()]
    if not subdirs:
        continue
    task_name_dir = subdirs[0]
    eps = sorted(task_name_dir.glob("episode_*.hdf5"))
    rows.append({
        "task_id": task_dir.name,
        "task_name": task_name_dir.name,
        "kept": len(eps),
    })
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
echo

echo "-- queue log tail --"
tail -n 40 "${LOG_DIR}/selected10.queue.log" 2>/dev/null || true
echo

echo "-- current watchdog tails --"
for f in "${LOG_DIR}"/task_*.watchdog.log; do
  [[ -e "${f}" ]] || continue
  echo "### $(basename "${f}")"
  tail -n 10 "${f}" || true
done
