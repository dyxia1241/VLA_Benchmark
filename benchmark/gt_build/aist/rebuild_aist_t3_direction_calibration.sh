#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
SELECTED_ROOT="${ROOT}/aist-bimanip/selected20"
CALIB_DIR="${ROOT}/benchmark/aist_t3_direction_calibration_v0"
CAMERA="${AIST_CAMERA:-cam_high}"
PER_ARM_DIRECTION="${AIST_T3_CALIB_PER_ARM_DIRECTION:-3}"

echo "[AIST-T3-CALIB] rebuild start camera=${CAMERA} per_arm_direction=${PER_ARM_DIRECTION} $(date -Is)"

rm -rf "${CALIB_DIR}/frames" "${CALIB_DIR}/cards" "${CALIB_DIR}/gt"
rm -f "${CALIB_DIR}/aist_t3_direction_calibration_v0.jsonl" "${CALIB_DIR}/annotation_sheet.csv" "${CALIB_DIR}/summary.json"

python "${ROOT}/benchmark/gt_build/aist/build_aist_t3_direction_calibration.py" \
  --selected-root "${SELECTED_ROOT}" \
  --output-dir "${CALIB_DIR}" \
  --per-arm-direction "${PER_ARM_DIRECTION}" \
  --camera "${CAMERA}"

python "${ROOT}/benchmark/gt_build/aist/extract_aist_frames.py" \
  --input-jsonl "${CALIB_DIR}/aist_t3_direction_calibration_v0.jsonl" \
  --output-dir "${CALIB_DIR}/frames"

python "${ROOT}/benchmark/gt_build/aist/render_aist_t3_pilot_cards.py" \
  --input-jsonl "${CALIB_DIR}/aist_t3_direction_calibration_v0.jsonl" \
  --frame-dir "${CALIB_DIR}/frames" \
  --output-dir "${CALIB_DIR}"

python - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/data/projects/GM-100/benchmark/aist_t3_direction_calibration_v0")
manifest = {}
with (base / "cards_manifest.jsonl").open("r", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        manifest[int(row["item_index"])] = row["card_path"]

rows = []
with (base / "annotation_sheet.csv").open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames or []
    for idx, row in enumerate(reader, start=1):
        row["card_path"] = manifest.get(idx, "")
        rows.append(row)

with (base / "annotation_sheet.csv").open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "[AIST-T3-CALIB] rebuild done $(date -Is)"
