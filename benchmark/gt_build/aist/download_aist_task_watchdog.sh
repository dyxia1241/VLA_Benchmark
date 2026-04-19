#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-benchmark/gt_build/aist_bimanual_manifest.csv}"
TASK_ID="${2:-}"
ROOT="/data/projects/GM-100/aist-bimanip"
RAW_DIR="${ROOT}/raw"
SELECTED_DIR="${ROOT}/selected20"
TMP_DIR="${ROOT}/tmp_unpack/${TASK_ID}"
LOG_DIR="${ROOT}/logs"
SLIM_SCRIPT="/data/projects/GM-100/benchmark/gt_build/aist/slim_aist_hdf5.py"
CHUNK_SECONDS="${AIST_CHUNK_SECONDS:-300}"
MAX_ATTEMPTS="${AIST_MAX_ATTEMPTS:-200}"
AIST_KEEP_CAMERAS="${AIST_KEEP_CAMERAS:-cam_high cam_low}"

if [[ -z "${TASK_ID}" ]]; then
  echo "usage: $0 <manifest.csv> <task_id>" >&2
  exit 1
fi
mkdir -p "${RAW_DIR}" "${SELECTED_DIR}" "${TMP_DIR}" "${LOG_DIR}"

row=$(awk -F, -v t="${TASK_ID}" 'NR>1 && $1==t {print $0}' "${MANIFEST}")
if [[ -z "${row}" ]]; then
  echo "task not found: ${TASK_ID}" >&2
  exit 1
fi

task_name=$(printf '%s\n' "${row}" | cut -d, -f2)
url=$(printf '%s\n' "${row}" | cut -d, -f4-)
zip_path="${RAW_DIR}/${TASK_ID}__${task_name}.zip"
log_path="${LOG_DIR}/${TASK_ID}.watchdog.log"
lock_path="${LOG_DIR}/${TASK_ID}.watchdog.lock"

exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "[AIST] watchdog already running for ${TASK_ID}; lock=${lock_path}" | tee -a "${log_path}"
  exit 0
fi

echo "[AIST] watchdog task=${TASK_ID} name=${task_name}" | tee -a "${log_path}"
echo "[AIST] zip=${zip_path}" | tee -a "${log_path}"

attempt=1
while [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; do
  before=0
  [[ -f "${zip_path}" ]] && before=$(stat -c '%s' "${zip_path}")
  echo "[AIST] attempt ${attempt}/${MAX_ATTEMPTS}; before=${before}; $(date -Is)" | tee -a "${log_path}"
  set +e
  timeout "${CHUNK_SECONDS}s" wget -c "${url}" -O "${zip_path}" >>"${log_path}" 2>&1
  rc=$?
  set -e
  after=0
  [[ -f "${zip_path}" ]] && after=$(stat -c '%s' "${zip_path}")
  echo "[AIST] attempt ${attempt} rc=${rc}; after=${after}; delta=$((after-before)); $(date -Is)" | tee -a "${log_path}"

  if [[ -s "${zip_path}" ]] && unzip -tqq "${zip_path}" >/dev/null 2>&1; then
    echo "[AIST] zip complete: ${zip_path}" | tee -a "${log_path}"
    python - <<'PY' "${zip_path}" "${TMP_DIR}" "${SELECTED_DIR}" "${TASK_ID}" "${task_name}" "${SLIM_SCRIPT}" "${AIST_KEEP_CAMERAS}" | tee -a "${log_path}"
import os, re, sys, shutil, zipfile, json, subprocess, shlex
zip_path, tmp_dir, selected_dir, task_id, task_name, slim_script, keep_cameras_raw = sys.argv[1:]
keep_cameras = [x for x in shlex.split(keep_cameras_raw) if x]
pat = re.compile(rf"{re.escape(task_name)}/episode_(\d+)\.hdf5$")
with zipfile.ZipFile(zip_path) as zf:
    names=[]
    for info in zf.infolist():
        m=pat.match(info.filename)
        if m:
            names.append((int(m.group(1)), info.filename))
    names.sort()
    if not names:
        raise SystemExit(f"no episode hdf5 found in {zip_path}")
    k=min(20,len(names))
    if len(names)<=k:
        keep_idx=list(range(len(names)))
    else:
        keep_idx=sorted({round(i*(len(names)-1)/(k-1)) for i in range(k)})
        # In rare rounding collisions, fill from the front.
        x=0
        while len(keep_idx)<k:
            if x not in keep_idx:
                keep_idx.append(x)
            x+=1
        keep_idx=sorted(keep_idx[:k])
    dst_dir=os.path.join(selected_dir, task_id, task_name)
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    kept=[]
    for idx in keep_idx:
        ep_id, member=names[idx]
        base=os.path.basename(member)
        tmp_path=os.path.join(tmp_dir, base)
        dst_path=os.path.join(dst_dir, base)
        with zf.open(member) as src, open(tmp_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        cmd = [
            sys.executable,
            slim_script,
            "--input",
            tmp_path,
            "--output",
            dst_path,
            "--keep-images",
            *keep_cameras,
        ]
        subprocess.run(cmd, check=True)
        os.remove(tmp_path)
        kept.append(ep_id)
    print(json.dumps({
        "task_id":task_id,
        "task_name":task_name,
        "episodes_total_in_zip":len(names),
        "episodes_kept":len(kept),
        "kept_episode_ids":kept,
        "kept_rgb_cameras":keep_cameras,
        "dropped_modalities":["wrist_rgb","depth","sound"]
    }, indent=2))
PY
    rm -rf "${TMP_DIR}"
    rm -f "${zip_path}"
    echo "[AIST] done ${TASK_ID}; raw zip removed" | tee -a "${log_path}"
    exit 0
  fi

  attempt=$((attempt+1))
  sleep 5
done

echo "[AIST] failed/incomplete after ${MAX_ATTEMPTS} attempts: ${TASK_ID}" | tee -a "${log_path}" >&2
exit 2
