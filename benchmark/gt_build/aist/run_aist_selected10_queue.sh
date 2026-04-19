#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/projects/GM-100"
MANIFEST="${ROOT}/benchmark/gt_build/aist_bimanual_manifest.csv"
TASK_FILE="${1:-${ROOT}/benchmark/gt_build/aist/aist_selected10_tasks.txt}"
LOG_DIR="${ROOT}/aist-bimanip/logs"
SELECTED_ROOT="${ROOT}/aist-bimanip/selected20"
WATCHDOG="${ROOT}/benchmark/gt_build/aist/download_aist_task_watchdog.sh"
REBUILD="${ROOT}/benchmark/gt_build/aist/rebuild_aist_outputs.sh"
QUEUE_LOG="${LOG_DIR}/selected10.queue.log"
LOCK_FILE="${LOG_DIR}/selected10.queue.lock"

mkdir -p "${LOG_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[AIST] selected10 queue already running; lock=${LOCK_FILE}" | tee -a "${QUEUE_LOG}"
  exit 0
fi

echo "[AIST] selected10 queue start $(date -Is)" | tee -a "${QUEUE_LOG}"
echo "[AIST] task_file=${TASK_FILE}" | tee -a "${QUEUE_LOG}"

mapfile -t TASKS < <(sed 's/#.*$//' "${TASK_FILE}" | awk 'NF > 0 {print $1}')

for task_id in "${TASKS[@]}"; do
  existing=0
  if [[ -d "${SELECTED_ROOT}/${task_id}" ]]; then
    existing=$(find "${SELECTED_ROOT}/${task_id}" -name 'episode_*.hdf5' 2>/dev/null | wc -l)
  fi

  echo "[AIST] task=${task_id} existing_selected20=${existing} $(date -Is)" | tee -a "${QUEUE_LOG}"

  if [[ "${existing}" -lt 20 ]]; then
    echo "[AIST] start download/extract ${task_id} $(date -Is)" | tee -a "${QUEUE_LOG}"
    bash "${WATCHDOG}" "${MANIFEST}" "${task_id}" < /dev/null >> "${QUEUE_LOG}" 2>&1
    echo "[AIST] finish download/extract ${task_id} $(date -Is)" | tee -a "${QUEUE_LOG}"
  else
    echo "[AIST] skip download for ${task_id}; selected20 already present" | tee -a "${QUEUE_LOG}"
  fi

  echo "[AIST] start rebuild after ${task_id} $(date -Is)" | tee -a "${QUEUE_LOG}"
  bash "${REBUILD}" < /dev/null >> "${QUEUE_LOG}" 2>&1
  echo "[AIST] finish rebuild after ${task_id} $(date -Is)" | tee -a "${QUEUE_LOG}"
done

echo "[AIST] selected10 queue finished $(date -Is)" | tee -a "${QUEUE_LOG}"
