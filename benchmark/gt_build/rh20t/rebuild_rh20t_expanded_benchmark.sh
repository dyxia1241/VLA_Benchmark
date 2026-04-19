#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
ARCHIVE="${ROOT}/RH20T_cfg2.tar.gz"
CAMERA="${RH20T_CAMERA:-036422060215}"
WORK_DIR="${ROOT}/benchmark/rh20t_cfg2_expanded_v0"
SCENE_LIST="${WORK_DIR}/selected_scenes.json"
EXTRACT_ROOT="${WORK_DIR}/extracted_primary_cam"
POOLS_DIR="${WORK_DIR}/pools"
OUTPUT_JSONL="${ROOT}/benchmark/rh20t_benchmark_v0_curated.jsonl"
OUTPUT_SUMMARY="${ROOT}/benchmark/rh20t_benchmark_v0_summary.json"
OUTPUT_BY_TYPE="${ROOT}/benchmark/rh20t_benchmark_v0_curated_by_type"
FRAME_DIR="${ROOT}/benchmark/rh20t_benchmark_v0_frames"
TARGET_JSON='{"T1":9000,"T2":6000,"T3":5400,"T_progress":8000,"T6":4500,"T_temporal":3000,"T_binary":4500}'

echo "[RH20T-expanded] start camera=${CAMERA} $(date -Is)"
echo "[RH20T-expanded] target_json=${TARGET_JSON}"

python "${ROOT}/benchmark/gt_build/rh20t/export_scene_list_from_summary.py" \
  --summary-json "${ROOT}/benchmark/rh20t_benchmark_v0_summary.json" \
  --output-json "${SCENE_LIST}"

rm -rf "${EXTRACT_ROOT}" "${POOLS_DIR}" "${OUTPUT_BY_TYPE}" "${FRAME_DIR}"
rm -f "${OUTPUT_JSONL}" "${OUTPUT_SUMMARY}"

python "${ROOT}/benchmark/gt_build/rh20t/extract_rh20t_cfg2_partial.py" \
  --archive "${ARCHIVE}" \
  --scene-list-json "${SCENE_LIST}" \
  --output-root "${EXTRACT_ROOT}" \
  --camera "${CAMERA}" \
  --mode with_color

python "${ROOT}/benchmark/gt_build/rh20t/build_rh20t_pilot_suite.py" \
  --extracted-root "${EXTRACT_ROOT}/RH20T_cfg2" \
  --scene-list-json "${SCENE_LIST}" \
  --tasks "T1,T2,T5,T6,T8,T9" \
  --output-dir "${POOLS_DIR}" \
  --camera "${CAMERA}"

python "${ROOT}/benchmark/gt_build/rh20t/fit_rh20t_t3_direction_mapping.py" \
  --annotation-sheet "${ROOT}/benchmark/rh20t_t3_direction_calibration_v0/annotation_sheet.csv" \
  --output-json "${ROOT}/benchmark/gt_build/rh20t/rh20t_t3_direction_mapping_${CAMERA}.json" \
  --camera "${CAMERA}"

python "${ROOT}/benchmark/gt_build/rh20t/build_rh20t_t3_suite.py" \
  --extracted-root "${EXTRACT_ROOT}/RH20T_cfg2" \
  --scene-list-json "${SCENE_LIST}" \
  --output-jsonl "${POOLS_DIR}/t3_gt_items.jsonl" \
  --summary-json "${POOLS_DIR}/t3_gt_items_summary.json" \
  --camera "${CAMERA}" \
  --direction-mapping "${ROOT}/benchmark/gt_build/rh20t/rh20t_t3_direction_mapping_${CAMERA}.json"

python "${ROOT}/benchmark/gt_build/rh20t/build_rh20t_curated_benchmark.py" \
  --input-dir "${POOLS_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --output-summary-json "${OUTPUT_SUMMARY}" \
  --output-per-type-dir "${OUTPUT_BY_TYPE}" \
  --tasks "T1,T2,T3,T5,T6,T8,T9" \
  --target-json "${TARGET_JSON}" \
  --min-rating 4 \
  --max-calib-quality 3 \
  --max-per-scene 80 \
  --max-per-group 1

python "${ROOT}/benchmark/gt_build/rh20t/extract_rh20t_pilot_frames.py" \
  --input-jsonl "${OUTPUT_JSONL}" \
  --extracted-root "${EXTRACT_ROOT}/RH20T_cfg2" \
  --output-dir "${FRAME_DIR}" \
  --progress-every 2000

echo "[RH20T-expanded] done $(date -Is)"
