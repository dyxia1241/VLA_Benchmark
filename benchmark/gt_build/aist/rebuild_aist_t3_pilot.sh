#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
SELECTED_ROOT="${ROOT}/aist-bimanip/selected20"
PILOT_DIR="${ROOT}/benchmark/aist_t3_pilot_v0"
LOG_DIR="${ROOT}/aist-bimanip/logs"
CAMERA="${AIST_CAMERA:-cam_high}"
PER_LABEL="${AIST_T3_PILOT_PER_LABEL:-18}"

mkdir -p "${LOG_DIR}"

echo "[AIST-T3] rebuild start camera=${CAMERA} per_label=${PER_LABEL} $(date -Is)"

rm -rf "${PILOT_DIR}/frames" "${PILOT_DIR}/cards" "${PILOT_DIR}/gt"
rm -f "${PILOT_DIR}/aist_t3_pilot_v0.jsonl" "${PILOT_DIR}/aist_t3_pilot_v0_summary.json"

python "${ROOT}/benchmark/gt_build/aist/build_aist_t3_pilot.py" \
  --selected-root "${SELECTED_ROOT}" \
  --output-dir "${PILOT_DIR}" \
  --per-label "${PER_LABEL}" \
  --camera "${CAMERA}"

python "${ROOT}/benchmark/gt_build/aist/extract_aist_frames.py" \
  --input-jsonl "${PILOT_DIR}/aist_t3_pilot_v0.jsonl" \
  --output-dir "${PILOT_DIR}/frames"

python "${ROOT}/benchmark/gt_build/aist/render_aist_t3_pilot_cards.py" \
  --input-jsonl "${PILOT_DIR}/aist_t3_pilot_v0.jsonl" \
  --frame-dir "${PILOT_DIR}/frames" \
  --output-dir "${PILOT_DIR}"

echo "[AIST-T3] rebuild done $(date -Is)"
