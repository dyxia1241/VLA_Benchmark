# GM-100 Benchmark (Refactored)

本目录是 GM-100 的评测主工作区，按职责拆分：

- `eval_v1/`：评测执行与评分入口。
- `gt_build/`：GT 构建、采样、抽帧。
- `legacy_collision_vqa/`：历史探索链路（非当前主线）。
- `previous_results/manual_checks_20260320/`：历史 full GT 产物（已归档）。
- `eval_results_v1/`：模型结果输出目录。

## 快速开始

在 `benchmark/` 目录下执行：

```bash
cd /data/projects/GM-100/benchmark
```

准备 API key（二选一示例）：

```bash
export OPENAI_API_KEY=YOUR_KEY
# 或
export DASHSCOPE_API_KEY=YOUR_KEY
```

## A. 从 curated 直接闭环（当前 root latest 口径）

当前 root release 已放在 `benchmark/` 根目录（8 题型，`15,500` items）：

- `benchmark_v1_curated.jsonl`
- `benchmark_v1_curated_healthcheck.json`

然后执行：

```bash
# 1) 抽帧（可复用已有目录）
python gt_build/extract_frames.py \
  --input-jsonl benchmark_v1_curated.jsonl \
  --output-dir benchmark_v1_frames_tbinary_20260330 \
  --t6-context

# 2) 评测
python eval_v1/run_benchmark_v1_eval.py \
  --input benchmark_v1_curated.jsonl \
  --frame-dir-default benchmark_v1_frames_tbinary_20260330 \
  --model qwen3-vl-plus

# 3) 评分
python eval_v1/score_benchmark_v1.py \
  eval_results_v1/benchmark_v1_qwen3vl_plus_full.jsonl
```

## B. 最大链路闭环评测（采样 -> 抽帧 -> 评测 -> 评分）

当前 latest full GT source snapshot 已在：
`previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2`

执行：

```bash
./run_v1_pipeline.sh \
  --gt-dir previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2 \
  --model qwen3-vl-plus
```

说明：当前默认脚本已对齐到 latest source snapshot；只有在复现历史版本时才需要显式改 `--gt-dir`。

## 常用参数

```bash
./run_v1_pipeline.sh --help
./run_v1_pipeline.sh --run-tag exp_001 --model qwen3-vl-plus
./run_v1_pipeline.sh --skip-sampling --skip-extract --result-jsonl eval_results_v1/existing.jsonl --skip-eval
```

## Manual Audit Bootstrap

Use `manual_audit/` to export a pilot human-audit slice from the current root release:

```bash
cd /data/projects/GM-100/benchmark
python manual_audit/export_audit_subset.py
```

Default outputs:

- `manual_audit/pilot_audit_v1/audit_items.csv`
- `manual_audit/pilot_audit_v1/annotator_a.csv`
- `manual_audit/pilot_audit_v1/annotator_b.csv`
- `manual_audit/pilot_audit_v1/audit_summary.json`
- `manual_audit/pilot_audit_v1/review_assets/` (`T_binary` composite review images)

Supporting docs:

- `manual_audit/audit_guideline_v1.md`
- `manual_audit/audit_template_v1.csv`

After two annotators finish, score agreement and export an adjudication table:

```bash
python manual_audit/score_audit_annotations.py \
  --annotator-a manual_audit/pilot_audit_v1/annotator_a.csv \
  --annotator-b manual_audit/pilot_audit_v1/annotator_b.csv
```

## 可移植性约定

- 路径尽量使用仓库相对路径，不写死 `/data/projects/...`。
- API key 不写入脚本与仓库。
- 大体量产物（frames/results/full_gt jsonl）建议不提交到 Git。
