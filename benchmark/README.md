# GM-100 Benchmark (Refactored)

本目录是 GM-100 的评测主工作区，按职责拆分：

- `eval_v1/`：评测执行与评分入口。
- `gt_build/`：GT 构建、采样、抽帧。
- `manual_audit/gt_audit/`：item-level 人工审计导出、标注与汇总。
- `manual_audit/semantic_affordance_audit/`：task-level semantic / affordance 标注工作流。
- `legacy_collision_vqa/`：历史探索链路（非当前主线）。
- `previous_results/manual_checks_20260320/`：当前 latest source snapshot 所在目录（目录名保留为历史归档命名）。
- `eval_results_v1/`：模型结果输出目录。

## 当前进展

- root latest benchmark 已收敛到 `8` 题型：`T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`，总量 `15,500`。
- 当前默认 source snapshot：`previous_results/manual_checks_20260320/full_gt_task00001_00110_live_20260331_tbinary_v2/`。
- 当前默认 frame cache：`benchmark_v1_frames_tbinary_20260330/`。
- formal item-level manual audit 已落到 `manual_audit/gt_audit/full_audit_v1/`，当前仓库快照包含 `470` 条审计样本与 `annotator_a` 工作簿。
- task-level taxonomy 工作流已迁到 `manual_audit/semantic_affordance_audit/`，semantic / affordance 主表、guide 和 workbook 导出脚本已对齐到新路径。
- `README.md`、`ARCHITECTURE_MAP.md`、`NEXT_STEPS.md`、`benchmark_card.md` 已按当前目录结构同步。

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
RESULT_JSONL=eval_results_v1/benchmark_v1_qwen3vl_plus_full.jsonl

# 1) 抽帧（可复用已有目录）
python gt_build/extract_frames.py \
  --input-jsonl benchmark_v1_curated.jsonl \
  --output-dir benchmark_v1_frames_tbinary_20260330 \
  --t6-context

# 2) 评测
python eval_v1/run_benchmark_v1_eval.py \
  --input benchmark_v1_curated.jsonl \
  --frame-dir-default benchmark_v1_frames_tbinary_20260330 \
  --model qwen3-vl-plus \
  --output "$RESULT_JSONL"

# 3) 评分
python eval_v1/score_benchmark_v1.py \
  "$RESULT_JSONL"
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

## Optional Task-Meta Prompt

默认 benchmark 协议 **不** prepend task meta，保持和主结果一致。

如果要做 prompt ablation / prompt augmentation，可直接在评测入口开启：

```bash
python eval_v1/run_benchmark_v1_eval.py \
  --input benchmark_v1_curated.jsonl \
  --frame-dir-default benchmark_v1_frames_tbinary_20260330 \
  --model qwen3-vl-plus \
  --prepend-task-meta
```

当前实现会默认从 `benchmark/GM100 List.xlsx` 读取 task meta；若某个 task 在 xlsx 中缺失，则 fallback 到 dataset `tasks.jsonl`。

当前代码中的 exact prepend template 为：

```text
You are given image(s) from a robot manipulation episode.

Task context: The overall task in this episode is "<task_meta>".

This task context is provided only as background. Do not rely on the task name alone. Answer based on the visual evidence in the provided image(s).

Now answer the following question:
<prompt_body>
```

其中 `<prompt_body>` 指原本该题自己的 question + choices / answer-format 指令块。

注意：

- `--prepend-task-meta` 是可选分析开关，不是默认 benchmark protocol。
- 开启后得到的结果应与“无 task meta prompt”的主 benchmark 结果分开记录和汇报。

## Manual Audit Bootstrap

当前仓库里已经有一套正式导出的全量人工审计包：

- `manual_audit/gt_audit/full_audit_v1/audit_subset.jsonl`
- `manual_audit/gt_audit/full_audit_v1/audit_items.csv`
- `manual_audit/gt_audit/full_audit_v1/annotator_a.csv`
- `manual_audit/gt_audit/full_audit_v1/annotator_a.xlsx`
- `manual_audit/gt_audit/full_audit_v1/audit_summary.json`
- `manual_audit/gt_audit/full_audit_v1/audit_cards/`

其中 `full_audit_v1` 是当前主用的人审版本，覆盖 `470` 题，包含 `T1/T2/T3/T4/T6/T_temporal/T_binary/T_progress`。当前检入快照里保留了 `annotator_a` 工作簿；第二位标注者的回收文件需要在双人标注开始后单独放回同目录。

如果要从当前 root release 重新导出一个新的试标/审计子集，可以运行：

```bash
cd /data/projects/GM-100/benchmark
python manual_audit/gt_audit/export_audit_subset.py \
  --output-dir manual_audit/your_audit_dir
```

建议显式传 `--output-dir`。当前仓库不再保留历史 pilot audit 目录；如需新的试标包，请自行指定输出位置，例如：

```bash
python manual_audit/gt_audit/export_audit_subset.py \
  --output-dir manual_audit/your_audit_dir
```

Supporting docs:

- `manual_audit/gt_audit/audit_guideline_v1.md`
- `manual_audit/gt_audit/audit_template_v1.csv`

完成双人标注后，再运行：

```bash
python manual_audit/gt_audit/score_audit_annotations.py \
  --annotator-a manual_audit/gt_audit/full_audit_v1/annotator_a.csv \
  --annotator-b path/to/annotator_b.csv
```

## 可移植性约定

- 路径尽量使用仓库相对路径，不写死 `/data/projects/...`。
- API key 不写入脚本与仓库。
- 大体量产物（frames/results/full_gt jsonl）建议不提交到 Git。
