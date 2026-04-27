#!/usr/bin/env bash
set -euo pipefail

# AIST Bimanual subset downloader
#
# 设计目标：
# 1. 不依赖浏览器，优先 aria2c 断点续传
# 2. 默认只下载双臂相关 task
# 3. 兼容中国网络环境下常见代理设置（https_proxy/http_proxy/all_proxy）
# 4. 不直接内置官方 task 链接；通过 manifest CSV 解耦，避免网页结构变动时反复改脚本
#
# manifest CSV 列格式：
# task_id,task_name,taxonomy,url
# 例如：
# task_048,hand over blue block,Synchronous Bimanual,https://www.dropbox.com/.../task_048.zip?dl=0
#
# 默认 taxonomy 过滤：
# - Synchronous Bimanual
# - Asynchronous Bimanual
# - collaboration
#
# 用法示例：
#   bash benchmark/gt_build/download_aist_bimanual_subset.sh \
#     --manifest benchmark/gt_build/aist_bimanual_manifest.csv \
#     --dry-run
#
#   bash benchmark/gt_build/download_aist_bimanual_subset.sh \
#     --manifest benchmark/gt_build/aist_bimanual_manifest.csv \
#     --output-dir /data/projects/GM-100/aist-bimanip/raw

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MANIFEST="${SCRIPT_DIR}/aist_bimanual_manifest.csv"
DEFAULT_OUTPUT_DIR="/data/projects/GM-100/aist-bimanip/raw"
DEFAULT_LOG="${SCRIPT_DIR}/aist_bimanual_download_failures.log"

MANIFEST="${DEFAULT_MANIFEST}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
FAIL_LOG="${DEFAULT_LOG}"
DRY_RUN=0
FORCE=0
MAX_CONNECTIONS=16
SPLITS=16
MIN_SPLIT_SIZE="1M"
INCLUDE_TAXONOMY_REGEX='(Synchronous Bimanual|Asynchronous Bimanual|collaboration)'
INCLUDE_TASK_IDS=""
EXCLUDE_TASK_IDS=""

usage() {
  cat <<'EOF'
用法:
  bash benchmark/gt_build/download_aist_bimanual_subset.sh [选项]

选项:
  --manifest <path>              manifest CSV 路径
  --output-dir <path>            下载目录，默认 /data/projects/GM-100/aist-bimanip/raw
  --fail-log <path>              失败日志路径
  --include-taxonomy <regex>     taxonomy 过滤正则
  --include-task-ids <csv>       仅下载指定 task_id，逗号分隔
  --exclude-task-ids <csv>       排除指定 task_id，逗号分隔
  --max-connections <int>        aria2c -x，默认 16
  --splits <int>                 aria2c -s，默认 16
  --min-split-size <size>        aria2c -k，默认 1M
  --dry-run                      只打印计划，不执行下载
  --force                        即使本地文件存在也继续尝试下载
  -h, --help                     显示帮助

manifest CSV 列:
  task_id,task_name,taxonomy,url

说明:
  1. 支持 Dropbox 分享链接，脚本会自动把 dl=0 改成 dl=1。
  2. 若设置了 https_proxy/http_proxy/all_proxy，aria2c/wget 会自动继承。
  3. 优先使用 aria2c；若系统无 aria2c，则降级到 wget。
EOF
}

require_int() {
  local v="${1:-}"
  [[ "$v" =~ ^[0-9]+$ ]] || {
    echo "数值参数非法: $v" >&2
    exit 1
  }
}

trim() {
  local s="${1:-}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

slugify() {
  local s
  s="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "$s" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
  printf '%s' "$s"
}

normalize_dropbox_url() {
  local url="${1:-}"
  if [[ "$url" == *"dropbox.com"* ]]; then
    if [[ "$url" == *"dl=0"* ]]; then
      url="${url/dl=0/dl=1}"
    elif [[ "$url" == *"dl=1"* ]]; then
      :
    elif [[ "$url" == *\?* ]]; then
      url="${url}&dl=1"
    else
      url="${url}?dl=1"
    fi
  fi
  printf '%s' "$url"
}

csv_contains_id() {
  local csv="${1:-}"
  local needle="${2:-}"
  if [[ -z "$csv" ]]; then
    return 1
  fi
  local item
  IFS=',' read -r -a _ids <<<"$csv"
  for item in "${_ids[@]}"; do
    item="$(trim "$item")"
    [[ -n "$item" && "$item" == "$needle" ]] && return 0
  done
  return 1
}

download_with_aria2() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  aria2c \
    -c \
    -x "${MAX_CONNECTIONS}" \
    -s "${SPLITS}" \
    -k "${MIN_SPLIT_SIZE}" \
    --file-allocation=none \
    --dir "${out_dir}" \
    --out "${out_name}" \
    "${url}"
}

download_with_wget() {
  local url="$1"
  local out_dir="$2"
  local out_name="$3"
  mkdir -p "${out_dir}"
  wget -c "${url}" -O "${out_dir}/${out_name}"
}

is_complete_zip() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  [[ -s "$path" ]] || return 1
  unzip -tqq "$path" >/dev/null 2>&1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --fail-log)
      FAIL_LOG="${2:-}"
      shift 2
      ;;
    --include-taxonomy)
      INCLUDE_TAXONOMY_REGEX="${2:-}"
      shift 2
      ;;
    --include-task-ids)
      INCLUDE_TASK_IDS="${2:-}"
      shift 2
      ;;
    --exclude-task-ids)
      EXCLUDE_TASK_IDS="${2:-}"
      shift 2
      ;;
    --max-connections)
      MAX_CONNECTIONS="${2:-}"
      shift 2
      ;;
    --splits)
      SPLITS="${2:-}"
      shift 2
      ;;
    --min-split-size)
      MIN_SPLIT_SIZE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_int "${MAX_CONNECTIONS}"
require_int "${SPLITS}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest 不存在: ${MANIFEST}" >&2
  echo "请先基于模板创建: ${DEFAULT_MANIFEST}" >&2
  exit 1
fi

if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
else
  echo "未找到 aria2c 或 wget，请先安装其一。" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
: > "${FAIL_LOG}"

echo "AIST downloader plan"
echo "  manifest: ${MANIFEST}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  downloader: ${DOWNLOADER}"
echo "  taxonomy_regex: ${INCLUDE_TAXONOMY_REGEX}"
echo "  include_task_ids: ${INCLUDE_TASK_IDS:-<all matched>}"
echo "  exclude_task_ids: ${EXCLUDE_TASK_IDS:-<none>}"
echo "  dry_run: ${DRY_RUN}"
echo "  proxy:https_proxy=${https_proxy:-${HTTPS_PROXY:-<unset>}}"

planned=0
skipped=0
downloaded=0
failed=0

while IFS=',' read -r raw_task_id raw_task_name raw_taxonomy raw_url _rest; do
  task_id="$(trim "${raw_task_id}")"
  task_name="$(trim "${raw_task_name}")"
  taxonomy="$(trim "${raw_taxonomy}")"
  url="$(trim "${raw_url}")"

  [[ -z "${task_id}" ]] && continue
  [[ "${task_id}" == "task_id" ]] && continue
  [[ "${task_id}" == \#* ]] && continue
  [[ -z "${url}" ]] && {
    echo "跳过 ${task_id}: url 为空" >&2
    skipped=$((skipped + 1))
    continue
  }

  if ! printf '%s' "${taxonomy}" | grep -Eiq "${INCLUDE_TAXONOMY_REGEX}"; then
    skipped=$((skipped + 1))
    continue
  fi

  if [[ -n "${INCLUDE_TASK_IDS}" ]] && ! csv_contains_id "${INCLUDE_TASK_IDS}" "${task_id}"; then
    skipped=$((skipped + 1))
    continue
  fi

  if csv_contains_id "${EXCLUDE_TASK_IDS}" "${task_id}"; then
    skipped=$((skipped + 1))
    continue
  fi

  url="$(normalize_dropbox_url "${url}")"
  safe_name="${task_id}__$(slugify "${task_name}")"
  out_name="${safe_name}.zip"
  out_path="${OUTPUT_DIR}/${out_name}"

  planned=$((planned + 1))
  echo "[PLAN] ${task_id} | ${taxonomy} | ${task_name}"
  echo "       -> ${out_path}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  if [[ -f "${out_path}" && "${FORCE}" -ne 1 ]]; then
    if is_complete_zip "${out_path}"; then
      echo "[SKIP] 已存在且完整 ${out_path}"
      skipped=$((skipped + 1))
      continue
    fi
    echo "[RESUME] 检测到残缺文件，继续续传 ${out_path}"
  fi

  if [[ "${DOWNLOADER}" == "aria2c" ]]; then
    if download_with_aria2 "${url}" "${OUTPUT_DIR}" "${out_name}"; then
      downloaded=$((downloaded + 1))
    else
      echo "${task_id},${url}" >> "${FAIL_LOG}"
      failed=$((failed + 1))
    fi
  else
    if download_with_wget "${url}" "${OUTPUT_DIR}" "${out_name}"; then
      downloaded=$((downloaded + 1))
    else
      echo "${task_id},${url}" >> "${FAIL_LOG}"
      failed=$((failed + 1))
    fi
  fi
done < "${MANIFEST}"

echo
echo "AIST download summary"
echo "  planned: ${planned}"
echo "  downloaded: ${downloaded}"
echo "  skipped: ${skipped}"
echo "  failed: ${failed}"
echo "  fail_log: ${FAIL_LOG}"
