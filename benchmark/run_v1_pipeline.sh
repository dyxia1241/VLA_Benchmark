#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"

GT_DIR_DEFAULT="$BENCHMARK_DIR/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2"
MODEL_DEFAULT="qwen3-vl-plus"

GT_DIR="$GT_DIR_DEFAULT"
MODEL="$MODEL_DEFAULT"
RUN_TAG=""
FRAME_DIR=""
RESULT_JSONL=""
API_KEY=""

SKIP_SAMPLING=0
SKIP_EXTRACT=0
SKIP_EVAL=0
SKIP_SCORE=0

usage() {
  cat <<'EOF'
Usage:
  ./run_v1_pipeline.sh [options]

Options:
  --gt-dir PATH            GT directory containing t*_gt_items.jsonl
                           default: benchmark/previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2
  --model NAME             VLM model name
                           default: qwen3-vl-plus
  --run-tag TAG            Run tag used for frame/result naming
                           default: <timestamp>_<model>
  --frame-dir PATH         Output frame directory
                           default: benchmark/benchmark_v1_frames_<run_tag>
  --result-jsonl PATH      Output eval jsonl
                           default: benchmark/eval_results_v1/benchmark_v1_<run_tag>.jsonl
  --api-key KEY            Optional API key (otherwise use env vars)

  --skip-sampling          Skip build_sampling_pipeline.py
  --skip-extract           Skip extract_frames.py
  --skip-eval              Skip run_benchmark_v1_eval.py
  --skip-score             Skip score_benchmark_v1.py

  -h, --help               Show this help

Examples:
  ./run_v1_pipeline.sh
  ./run_v1_pipeline.sh --model qwen3-vl-plus --run-tag exp_001
  ./run_v1_pipeline.sh --skip-sampling --skip-extract --result-jsonl eval_results_v1/existing.jsonl --skip-eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt-dir)
      GT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --frame-dir)
      FRAME_DIR="$2"
      shift 2
      ;;
    --result-jsonl)
      RESULT_JSONL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --skip-sampling)
      SKIP_SAMPLING=1
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --skip-score)
      SKIP_SCORE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_TAG" ]]; then
  MODEL_TAG="$(echo "$MODEL" | tr '/:' '__')"
  RUN_TAG="$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
fi

if [[ -z "$FRAME_DIR" ]]; then
  FRAME_DIR="$BENCHMARK_DIR/benchmark_v1_frames_${RUN_TAG}"
fi
if [[ -z "$RESULT_JSONL" ]]; then
  RESULT_JSONL="$BENCHMARK_DIR/eval_results_v1/benchmark_v1_${RUN_TAG}.jsonl"
fi

CURATED_JSONL="$GT_DIR/benchmark_v1_curated.jsonl"
CURATED_HEALTH="$GT_DIR/benchmark_v1_curated_healthcheck.json"
CURATED_BYTYPE="$GT_DIR/benchmark_v1_curated_by_type"

mkdir -p "$BENCHMARK_DIR/eval_results_v1"

if [[ ! -d "$GT_DIR" ]]; then
  echo "[ERROR] GT dir not found: $GT_DIR" >&2
  exit 1
fi

echo "[INFO] BENCHMARK_DIR=$BENCHMARK_DIR"
echo "[INFO] GT_DIR=$GT_DIR"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] RUN_TAG=$RUN_TAG"
echo "[INFO] CURATED_JSONL=$CURATED_JSONL"
echo "[INFO] FRAME_DIR=$FRAME_DIR"
echo "[INFO] RESULT_JSONL=$RESULT_JSONL"

if [[ "$SKIP_SAMPLING" -eq 0 ]]; then
  echo "[STEP] Sampling benchmark_v1 curated set"
  python3 "$BENCHMARK_DIR/gt_build/build_sampling_pipeline.py" \
    --gt-dir "$GT_DIR" \
    --output-jsonl "$CURATED_JSONL" \
    --output-summary-json "$CURATED_HEALTH" \
    --output-per-type-dir "$CURATED_BYTYPE"
else
  echo "[SKIP] sampling"
fi

if [[ ! -f "$CURATED_JSONL" ]]; then
  echo "[ERROR] Curated jsonl not found: $CURATED_JSONL" >&2
  exit 1
fi

if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
  echo "[STEP] Extracting frames"
  python3 "$BENCHMARK_DIR/gt_build/extract_frames.py" \
    --input-jsonl "$CURATED_JSONL" \
    --output-dir "$FRAME_DIR" \
    --t6-context
else
  echo "[SKIP] frame extraction"
fi

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "[STEP] Running evaluation"
  EVAL_CMD=(
    python3 "$BENCHMARK_DIR/eval_v1/run_benchmark_v1_eval.py"
    --input "$CURATED_JSONL"
    --frame-dir-default "$FRAME_DIR"
    --frame-dir-t3-multi "$FRAME_DIR"
    --frame-dir-t6-multi "$FRAME_DIR"
    --output "$RESULT_JSONL"
    --model "$MODEL"
  )
  if [[ -n "$API_KEY" ]]; then
    EVAL_CMD+=(--api-key "$API_KEY")
  fi
  "${EVAL_CMD[@]}"
else
  echo "[SKIP] evaluation"
fi

if [[ "$SKIP_SCORE" -eq 0 ]]; then
  if [[ ! -f "$RESULT_JSONL" ]]; then
    echo "[ERROR] Result jsonl not found: $RESULT_JSONL" >&2
    exit 1
  fi
  echo "[STEP] Scoring"
  python3 "$BENCHMARK_DIR/eval_v1/score_benchmark_v1.py" "$RESULT_JSONL"
else
  echo "[SKIP] scoring"
fi

echo "[DONE] pipeline finished"
