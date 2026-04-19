#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-/data/projects/GM-100/benchmark/gt_build/aist_bimanual_manifest.csv}"
TASK_ID="${2:-}"
ROOT="/data/projects/GM-100/aist-bimanip"
RAW_DIR="${ROOT}/raw"
SELECTED_DIR="${ROOT}/selected20"
TMP_DIR="${ROOT}/tmp_unpack/${TASK_ID}"
LOG_DIR="${ROOT}/logs"

if [[ -z "${TASK_ID}" ]]; then
  echo "usage: $0 <manifest.csv> <task_id>" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}" "${SELECTED_DIR}" "${TMP_DIR}" "${LOG_DIR}"

row=$(awk -F, -v t="${TASK_ID}" 'NR>1 && $1==t {print $0}' "${MANIFEST}")
if [[ -z "${row}" ]]; then
  echo "task not found in manifest: ${TASK_ID}" >&2
  exit 1
fi

task_name=$(printf '%s\n' "${row}" | cut -d, -f2)
url=$(printf '%s\n' "${row}" | cut -d, -f4-)
zip_path="${RAW_DIR}/${TASK_ID}__${task_name}.zip"
log_path="${LOG_DIR}/${TASK_ID}.wget.log"

echo "[AIST] task=${TASK_ID} name=${task_name}"
echo "[AIST] download -> ${zip_path}"
max_attempts="${AIST_MAX_ATTEMPTS:-30}"
attempt=1
while true; do
  echo "[AIST] wget attempt ${attempt}/${max_attempts}"
  set +e
  wget --tries=3 --timeout=60 --read-timeout=60 --continue "${url}" -O "${zip_path}" 2>&1 | tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -eq 0 ]]; then
    break
  fi
  if [[ ${attempt} -ge ${max_attempts} ]]; then
    echo "[AIST] download failed after ${max_attempts} attempts: ${TASK_ID}" >&2
    exit ${rc}
  fi
  attempt=$((attempt + 1))
  sleep 10
done
unzip -tqq "${zip_path}"

python - <<'PY' "${zip_path}" "${TMP_DIR}" "${SELECTED_DIR}" "${TASK_ID}" "${task_name}"
import os, re, sys, shutil, zipfile
zip_path, tmp_dir, selected_dir, task_id, task_name = sys.argv[1:]
pat = re.compile(rf"{re.escape(task_name)}/episode_(\d+)\.hdf5$")
with zipfile.ZipFile(zip_path) as zf:
    names = []
    for info in zf.infolist():
        m = pat.match(info.filename)
        if m:
            names.append((int(m.group(1)), info.filename))
    names.sort()
    if not names:
        raise SystemExit(f"no episode hdf5 found in {zip_path}")
    n = len(names)
    k = min(20, n)
    if n <= k:
        keep_idx = list(range(n))
    else:
        keep_idx = []
        used = set()
        for i in range(k):
            idx = round(i * (n - 1) / (k - 1))
            while idx in used and idx + 1 < n:
                idx += 1
            used.add(idx)
            keep_idx.append(idx)
        keep_idx = sorted(keep_idx)
    keep = [names[i] for i in keep_idx]
    dst_dir = os.path.join(selected_dir, task_id, task_name)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    for ep_id, member in keep:
        base = os.path.basename(member)
        tmp_path = os.path.join(tmp_dir, base)
        with zf.open(member) as src, open(tmp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        shutil.move(tmp_path, os.path.join(dst_dir, base))
    print({
        'task_id': task_id,
        'task_name': task_name,
        'episodes_total_in_zip': n,
        'episodes_kept': len(keep),
        'kept_episode_ids': [ep_id for ep_id, _ in keep],
    })
PY

rm -rf "${TMP_DIR}"
rm -f "${zip_path}"

echo "[AIST] done task=${TASK_ID}; kept 20-or-fewer episodes under ${SELECTED_DIR}/${TASK_ID}/${task_name}"
