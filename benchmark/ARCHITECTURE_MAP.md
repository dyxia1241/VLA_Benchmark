# Benchmark Architecture Map

更新时间：2026-04-26  
适用范围：`/data/projects/GM-100/benchmark`

定位说明：

- 本文件记录当前稳定主线的目录结构、最小运行闭环、默认入口和历史兼容链路。
- `NEURIPS_ED_MASTER_PLAN.md` 负责记录论文主线、任务定义、数据源角色和投稿边界。
- `benchmark_card.md` 负责对外叙述 benchmark 定义、可靠性和 caveats。
- `README.md` 只保留一个简短入口说明，避免和本文件重复维护。

## 1. Current Mainline

当前稳定主线已经收敛到四个数据源：

- `GM-100`
- `RH20T`
- `REASSEMBLE`
- `AIST`

当前冻结 benchmark 总量：

- Full benchmark：`57,892`
- Eval split：`9,051`
- SFT split：`48,841`

当前默认 protocol：

- 四源分别维护各自 eval jsonl 与 frame cache
- `splits_v1` 是当前默认 `SFT / Eval` protocol
- 默认 one-click eval 入口是 `run_all_eval_sets.py`
- 默认汇总入口是 `score_all_eval_sets.py`

## 2. Minimal Runtime Closure

如果你的目标只是“跑当前主线 eval 并汇总结果”，最小闭环只依赖下面这些文件和目录。

### 2.1 Code

- `benchmark/eval_v1/run_all_eval_sets.py`
- `benchmark/eval_v1/run_pilot_eval.py`
- `benchmark/eval_v1/score_all_eval_sets.py`

### 2.2 Manifest And Splits

- `benchmark/splits_v1/eval_manifest.json`
- `benchmark/splits_v1/gm100_eval.jsonl`
- `benchmark/splits_v1/rh20t_eval.jsonl`
- `benchmark/splits_v1/reassemble_eval.jsonl`
- `benchmark/splits_v1/aist_eval.jsonl`

### 2.3 Frame Caches

- `benchmark/benchmark_v1_frames_tbinary_20260330/`
- `benchmark/rh20t_benchmark_v0_frames/`
- `benchmark/reassemble_benchmark_v0_frames/`
- `benchmark/aist_benchmark_v0_frames/`

### 2.4 Runtime Outputs

默认输出目录：

- `benchmark/eval_results_v1/splits_v1/<model_slug>/`

运行后会生成：

- `gm100_eval_results.jsonl`
- `rh20t_eval_results.jsonl`
- `reassemble_eval_results.jsonl`
- `aist_eval_results.jsonl`
- `run_all_eval_sets_summary.json`
- `score_summary_<model_slug>.json` 或 `score_all_eval_sets_summary.json`

### 2.5 Non-core But Relevant

- `benchmark/eval_results_v1/_frame_fallback_cache/gm100/`  
  这是 GM100 缺帧时的 fallback 缓存目录。当前主线 eval 已可直接复用其中已抽出的缓存帧。
- `raw_data/gm100-cobotmagic-lerobot/`  
  只有当 GM100 主 frame cache 和 fallback cache 都缺帧时，`run_pilot_eval.py` 才会回源到这里抽帧。
- `raw_data/aist-bimanip/`、`raw_data/reassemble-tuwien-researchdata/`  
  当前不属于主线 runtime-core。

## 3. Directory Layout

```text
GM-100/
├── raw_data/
│   ├── gm100-cobotmagic-lerobot/                  # GM-100 原始视频源；仅 GM100 fallback 抽帧时需要
│   ├── reassemble-tuwien-researchdata/            # REASSEMBLE 原始 recording 源
│   └── aist-bimanip/                              # AIST 原始数据源
└── benchmark/
    ├── README.md
    ├── ARCHITECTURE_MAP.md
    ├── NEURIPS_ED_MASTER_PLAN.md
    ├── benchmark_card.md
    ├── eval_v1/
    │   ├── run_pilot_eval.py
    │   ├── run_all_eval_sets.py
    │   ├── score_pilot.py
    │   ├── score_all_eval_sets.py
    │   └── run_eval_by_dataset/                   # 单源兼容入口；不是当前 paper 主线
    ├── gt_build/
    │   ├── build_multisource_sft_eval_split.py
    │   ├── extract_frames.py
    │   ├── build_sampling_pipeline.py
    │   ├── rh20t/
    │   └── aist/
    ├── benchmark_v1_frames_tbinary_20260330/
    ├── rh20t_benchmark_v0_frames/
    ├── reassemble_benchmark_v0_frames/
    ├── aist_benchmark_v0_frames/
    ├── splits_v1/
    │   ├── gm100_eval.jsonl
    │   ├── rh20t_eval.jsonl
    │   ├── reassemble_eval.jsonl
    │   ├── aist_eval.jsonl
    │   ├── all_sft_merged.jsonl
    │   ├── all_eval_merged.jsonl
    │   ├── split_summary.json
    │   └── eval_manifest.json
    ├── eval_results_v1/
    │   ├── _frame_fallback_cache/
    │   └── splits_v1/
    ├── previous_results/
    └── run_v1_pipeline.sh                         # GM100 单源历史兼容链路
```

## 4. Data Flow

### 4.1 Four-source Build Flow

```text
raw datasets
    |
    v
dataset-specific GT builders
    |
    v
per-source curated jsonl
    |
    v
frame extraction
    |
    v
per-source frame caches
    |
    v
build_multisource_sft_eval_split.py
    |
    v
benchmark/splits_v1/
    |
    v
run_all_eval_sets.py
    |
    v
score_all_eval_sets.py
```

### 4.2 Split Build

`benchmark/gt_build/build_multisource_sft_eval_split.py` 负责：

- 从四个 per-source curated jsonl 构建 `85/15` SFT/Eval split
- 保证 `episode / recording / scene` 级别隔离
- 生成 `benchmark/splits_v1/eval_manifest.json`

### 4.3 Unified Eval

`benchmark/eval_v1/run_all_eval_sets.py` 负责：

- 读取 `eval_manifest.json`
- 顺序调用 `run_pilot_eval.py`
- 为四个数据集分别写出结果 JSONL

`benchmark/eval_v1/score_all_eval_sets.py` 负责：

- 读取 `eval_manifest.json`
- 根据 `slug` 找到各数据集结果 JSONL
- 用 manifest 内 baseline 汇总整体结果

## 5. Entrypoints

### 5.1 Default Mainline

重建 split：

```bash
cd /data/projects/GM-100
python3 benchmark/gt_build/build_multisource_sft_eval_split.py
```

运行四源 eval：

```bash
cd /data/projects/GM-100

python3 benchmark/eval_v1/run_all_eval_sets.py \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --api-base http://35.220.164.252:3888/v1/ \
  --api-key YOUR_KEY
```

汇总结果：

```bash
cd /data/projects/GM-100
python3 benchmark/eval_v1/score_all_eval_sets.py --model Qwen/Qwen2.5-VL-32B-Instruct
```

### 5.2 GM100 Legacy Single-source Pipeline

`benchmark/run_v1_pipeline.sh` 现在的定位是：

- 只服务于旧的 GM100 `benchmark_v1` 单源链路
- 不属于当前四源 paper mainline
- 适用于从一个 GM100 GT snapshot 重新走一遍 `采样 -> 抽帧 -> eval -> score`

它的默认用途不是跑当前 paper 主线，而是：

- 复现旧的单源 GM100 release
- 检查某个历史 `GT_DIR` 是否还能产出 `benchmark_v1`
- 为兼容旧实验保留一个一键脚本

如果你在整理仓库，它应该被归类为：

- `legacy-compatible`

而不是：

- `runtime-core`

## 6. Current Conventions

1. 论文命名使用 `T5/T8/T9`，工程兼容 `T_progress/T_temporal/T_binary`。
2. AIST 当前包含 `T3/T4/T6/T8/T9`，其中工程里 `T8` 落成 `T_temporal`。
3. `run_pilot_eval.py` 已兼容 AIST `T9` 双帧读取。
4. `REASSEMBLE` 默认视角固定为 `hand`。
5. `RH20T` 默认视角固定为当前选定主相机。
6. 当前四源主线读取帧都来自各自 `*_frames` 目录；只有 GM100 仍保留原始视频 fallback。

## 7. Document Responsibilities

1. `README.md`：简短入口页，只告诉读者主线文档在哪里。
2. `ARCHITECTURE_MAP.md`：记录当前稳定目录、运行闭环、默认入口和兼容链路。
3. `NEURIPS_ED_MASTER_PLAN.md`：记录论文主线、任务定义、边界和风险。
4. `benchmark_card.md`：对外定义 benchmark、协议、可靠性和 caveats。
