#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
EXTRACTED_ROOT="${ROOT}/benchmark/rh20t_cfg2_expanded_v0/extracted_primary_cam/RH20T_cfg2"
CALIB_DIR="${ROOT}/benchmark/rh20t_t3_direction_calibration_v0"
CAMERA="${RH20T_CAMERA:-036422060215}"
PER_DIRECTION="${RH20T_T3_CALIB_PER_DIRECTION:-8}"

rm -rf "${CALIB_DIR}/frames" "${CALIB_DIR}/cards" "${CALIB_DIR}/gt"
rm -f "${CALIB_DIR}/rh20t_t3_direction_calibration_v0.jsonl" "${CALIB_DIR}/annotation_sheet.csv" "${CALIB_DIR}/summary.json" "${CALIB_DIR}/cards_manifest.jsonl" "${CALIB_DIR}/render_summary.json"

python "${ROOT}/benchmark/gt_build/rh20t/build_rh20t_t3_direction_calibration.py" \
  --extracted-root "${EXTRACTED_ROOT}" \
  --scene-list-json "${ROOT}/benchmark/rh20t_cfg2_expanded_v0/selected_scenes.json" \
  --output-dir "${CALIB_DIR}" \
  --camera "${CAMERA}" \
  --per-direction "${PER_DIRECTION}"

python "${ROOT}/benchmark/gt_build/rh20t/extract_rh20t_pilot_frames.py" \
  --input-jsonl "${CALIB_DIR}/rh20t_t3_direction_calibration_v0.jsonl" \
  --extracted-root "${EXTRACTED_ROOT}" \
  --output-dir "${CALIB_DIR}/frames"

python "${ROOT}/benchmark/gt_build/rh20t/render_rh20t_pilot_cards.py" \
  --input-jsonl "${CALIB_DIR}/rh20t_t3_direction_calibration_v0.jsonl" \
  --frame-dir "${CALIB_DIR}/frames" \
  --output-dir "${CALIB_DIR}"

python - <<'PY'
import csv
import json
from pathlib import Path

base = Path('/data/projects/GM-100/benchmark/rh20t_t3_direction_calibration_v0')
manifest = {}
with (base / 'cards_manifest.jsonl').open('r', encoding='utf-8') as fh:
    for line in fh:
        row = json.loads(line)
        manifest[int(row['item_index'])] = row['card_path']

rows = []
with (base / 'annotation_sheet.csv').open('r', encoding='utf-8', newline='') as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames or []
    for idx, row in enumerate(reader, start=1):
        row['card_path'] = manifest.get(idx, '')
        rows.append(row)

with (base / 'annotation_sheet.csv').open('w', encoding='utf-8', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
