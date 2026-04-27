#!/bin/bash
set -u -o pipefail

# 默认配置（保持与旧脚本兼容）
BASE_URL="https://hf-mirror.com/datasets/rhos-ai/gm100-cobotmagic-lerobot/resolve/main"
TARGET_DIR="../gm100-cobotmagic-lerobot"
FAIL_LOG="download_failures.log"
TASK_START=1
TASK_END=110
EPISODES_PER_TASK=3
MODE="all" # all | parquet | video
EPISODE_SELECTION="even" # even | first
FAIL_FAST=0
TRIES=3
TIMEOUT=30

CAMS=("camera_top" "camera_wrist_left" "camera_wrist_right")

ok_count=0
fail_count=0
task_done=0

usage() {
  cat <<'EOF'
用法:
  bash download.sh [选项]

选项:
  --mode <all|parquet|video>     下载模式，默认 all
  --task-start <int>             起始 task 编号，默认 1
  --task-end <int>               结束 task 编号，默认 110
  --episodes-per-task <int>      每任务采样 episode 数（用于 all/video），默认 3
  --episode-selection <even|first>
                                 episode采样策略（all/video）：
                                 even=均匀采样（默认）
                                 first=取最前面的N个episode（从000000开始）
  --target-dir <path>            下载目标目录
  --base-url <url>               数据集基础 URL
  --fail-log <path>              失败日志路径
  --cameras <csv>                相机列表，逗号分隔
  --tries <int>                  wget 重试次数，默认 3
  --timeout <int>                wget 超时（秒），默认 30
  --fail-fast                    任一文件下载失败时立即退出
  -h, --help                     显示帮助

示例:
  # 默认行为：task_00001..task_00110，采样3条，下载 parquet+视频
  bash download.sh

  # 只下 parquet（每个 task 全量 episode）
  bash download.sh --mode parquet --task-start 1 --task-end 110

  # 只下视频（每任务采样 20 条 episode）
  bash download.sh --mode video --episodes-per-task 20

  # 只下视频（每任务取最前面 50 条 episode，仅top相机）
  bash download.sh --mode video --episodes-per-task 50 --episode-selection first --cameras camera_top
EOF
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_cameras() {
  local csv="$1"
  local -a parsed=()
  IFS=',' read -r -a parsed <<<"$csv"
  if [ "${#parsed[@]}" -eq 0 ]; then
    echo "无效 --cameras 参数: $csv" >&2
    exit 1
  fi
  CAMS=("${parsed[@]}")
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --task-start)
      TASK_START="${2:-}"
      shift 2
      ;;
    --task-end)
      TASK_END="${2:-}"
      shift 2
      ;;
    --episodes-per-task)
      EPISODES_PER_TASK="${2:-}"
      shift 2
      ;;
    --episode-selection)
      EPISODE_SELECTION="${2:-}"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --fail-log)
      FAIL_LOG="${2:-}"
      shift 2
      ;;
    --cameras)
      parse_cameras "${2:-}"
      shift 2
      ;;
    --tries)
      TRIES="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --fail-fast)
      FAIL_FAST=1
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

if [[ "$MODE" != "all" && "$MODE" != "parquet" && "$MODE" != "video" ]]; then
  echo "--mode 仅支持 all|parquet|video，收到: $MODE" >&2
  exit 1
fi
if [[ "$EPISODE_SELECTION" != "even" && "$EPISODE_SELECTION" != "first" ]]; then
  echo "--episode-selection 仅支持 even|first，收到: $EPISODE_SELECTION" >&2
  exit 1
fi

for v in "$TASK_START" "$TASK_END" "$EPISODES_PER_TASK" "$TRIES" "$TIMEOUT"; do
  if ! is_positive_int "$v"; then
    echo "数值参数必须是非负整数，收到: $v" >&2
    exit 1
  fi
done

if [ "$TASK_START" -gt "$TASK_END" ]; then
  echo "--task-start 不能大于 --task-end" >&2
  exit 1
fi

if [ "$EPISODES_PER_TASK" -eq 0 ] && [ "$MODE" != "parquet" ]; then
  echo "--episodes-per-task 在 all/video 模式下必须 >= 1" >&2
  exit 1
fi

: > "$FAIL_LOG"

download_file() {
  local url="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  if wget --show-progress -c --tries="$TRIES" --timeout="$TIMEOUT" "$url" -P "$out_dir"; then
    ok_count=$((ok_count + 1))
    return 0
  fi

  echo "$url" >> "$FAIL_LOG"
  fail_count=$((fail_count + 1))
  if [ "$FAIL_FAST" -eq 1 ]; then
    echo "下载失败，--fail-fast 生效，立即退出: $url" >&2
    exit 1
  fi
  return 1
}

load_all_episodes() {
  local episodes_file="$1"
  local -n out_array="$2"
  local -a ids=()

  if [ ! -f "$episodes_file" ]; then
    out_array=(0)
    return 0
  fi

  mapfile -t ids < <(
    grep -oE '"episode_index"[[:space:]]*:[[:space:]]*[0-9]+' "$episodes_file" \
      | grep -oE '[0-9]+'
  )

  if [ "${#ids[@]}" -eq 0 ]; then
    out_array=(0)
    return 0
  fi

  out_array=("${ids[@]}")
}

select_evenly_spaced_episodes() {
  local -n all_ids="$1"
  local -n out_ids="$2"
  local need="$3"
  local total="${#all_ids[@]}"
  local -A seen=()

  if [ "$total" -eq 0 ]; then
    out_ids=(0)
    return 0
  fi

  if [ "$need" -ge "$total" ]; then
    out_ids=("${all_ids[@]}")
    return 0
  fi

  out_ids=()
  if [ "$need" -eq 1 ]; then
    out_ids=("${all_ids[0]}")
    return 0
  fi

  local k=0
  while [ "$k" -lt "$need" ]; do
    local idx=$((k * (total - 1) / (need - 1)))
    local eid="${all_ids[$idx]}"
    if [ -z "${seen[$eid]+x}" ]; then
      out_ids+=("$eid")
      seen["$eid"]=1
    fi
    k=$((k + 1))
  done
}

select_first_episodes() {
  local -n all_ids="$1"
  local -n out_ids="$2"
  local need="$3"
  local total="${#all_ids[@]}"

  if [ "$total" -eq 0 ]; then
    out_ids=(0)
    return 0
  fi

  if [ "$need" -ge "$total" ]; then
    out_ids=("${all_ids[@]}")
    return 0
  fi

  out_ids=("${all_ids[@]:0:$need}")
}

echo "🚀 模式: $MODE"
echo "🚀 范围: task_$(printf "%05d" "$TASK_START") 到 task_$(printf "%05d" "$TASK_END")"
echo "🚀 目标目录: $TARGET_DIR"
if [ "$MODE" != "parquet" ]; then
  echo "🚀 每任务采样 episode 数: $EPISODES_PER_TASK"
  echo "🚀 采样策略: $EPISODE_SELECTION"
fi

for ((i=TASK_START; i<=TASK_END; i++)); do
  TASK=$(printf "task_%05d" "$i")
  echo "========== 处理 $TASK =========="

  DIR_META="$TARGET_DIR/$TASK/meta"
  download_file "$BASE_URL/$TASK/meta/info.json" "$DIR_META"
  download_file "$BASE_URL/$TASK/meta/episodes.jsonl" "$DIR_META"
  download_file "$BASE_URL/$TASK/meta/tasks.jsonl" "$DIR_META"

  EPISODES_FILE="$DIR_META/episodes.jsonl"
  all_episode_ids=()
  load_all_episodes "$EPISODES_FILE" all_episode_ids

  if [ "$MODE" = "parquet" ]; then
    parquet_ids=("${all_episode_ids[@]}")
    echo "[$TASK] parquet 全量 episode 数: ${#parquet_ids[@]}"
    if [ "${#parquet_ids[@]}" -le 1 ]; then
      echo "⚠️ [$TASK] episodes.jsonl 解析到的 episode 数异常少 (${#parquet_ids[@]})，请检查" >&2
    fi
  else
    selected_ids=()
    if [ "$EPISODE_SELECTION" = "first" ]; then
      select_first_episodes all_episode_ids selected_ids "$EPISODES_PER_TASK"
    else
      select_evenly_spaced_episodes all_episode_ids selected_ids "$EPISODES_PER_TASK"
    fi
    echo "[$TASK] 采样 episode: ${selected_ids[*]}"
  fi

  if [ "$MODE" = "all" ] || [ "$MODE" = "parquet" ]; then
    if [ "$MODE" = "all" ]; then
      parquet_ids=("${selected_ids[@]}")
    fi
    DIR_PARQUET="$TARGET_DIR/$TASK/data/chunk-000"
    for eid in "${parquet_ids[@]}"; do
      EP=$(printf "episode_%06d" "$eid")
      download_file "$BASE_URL/$TASK/data/chunk-000/$EP.parquet" "$DIR_PARQUET"
    done
  fi

  if [ "$MODE" = "all" ] || [ "$MODE" = "video" ]; then
    for eid in "${selected_ids[@]}"; do
      EP=$(printf "episode_%06d" "$eid")
      for CAM in "${CAMS[@]}"; do
        DIR_VID="$TARGET_DIR/$TASK/videos/chunk-000/observation.images.$CAM"
        download_file "$BASE_URL/$TASK/videos/chunk-000/observation.images.$CAM/$EP.mp4" "$DIR_VID"
      done
    done
  fi

  task_done=$((task_done + 1))
done

echo "✅ 完成 $task_done 个任务下载流程"
echo "✅ 成功文件数: $ok_count"
echo "⚠️ 失败文件数: $fail_count"
echo "⚠️ 失败清单: $FAIL_LOG"
